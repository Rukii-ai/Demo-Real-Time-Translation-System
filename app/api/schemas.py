"""Pydantic schemas for API request/response models.

This module defines all the data models used by the translation system API.
These models provide:
- Request validation
- Response serialization
- API documentation
- Type safety
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# ENUMS
# ============================================================================

class ServiceType(str, Enum):
    """Available service types for the translation system."""
    TRANSCRIPTION = "transcription"  # Audio -> Text
    TRANSLATION = "translation"      # Text -> Translated Text
    SYNTHESIS = "synthesis"          # Text -> Audio


class PipelineType(str, Enum):
    """Available pipeline types for chaining services."""
    TRANSCRIBE_TRANSLATE = "transcribe_translate"  # Audio -> Text -> Translated Text
    TRANSLATE_SYNTHESIZE = "translate_synthesize"  # Text -> Translated Text -> Audio
    FULL = "full"  # Audio -> Text -> Translated Text -> Audio


class LanguageCode(str, Enum):
    """Common NLLB language codes.
    
    These use the NLLB format: language_Family
    """
    ENGLISH = "eng_Latn"
    FRENCH = "fra_Latn"
    SPANISH = "spa_Latn"
    GERMAN = "deu_Latn"
    ITALIAN = "ita_Latn"
    PORTUGUESE = "por_Latn"
    CHINESE = "zho_Hans"
    JAPANESE = "jpn_Jpan"
    RUSSIAN = "rus_Cyrl"
    ARABIC = "ara_Arab"


# ============================================================================
# BASE MODELS
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(True, description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if operation failed")


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


# ============================================================================
# TRANSCRIPTION MODELS
# ============================================================================

class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    audio_path: Optional[str] = Field(None, description="Path to audio file (server-side)")
    audio_url: Optional[str] = Field(None, description="URL to download audio from")
    language: Optional[str] = Field(None, description="Expected language code (auto-detect if None)")
    model: str = Field("small", description="Whisper model to use (tiny, base, small, medium, large)")
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"}
        if v not in allowed:
            raise ValueError(f"Model must be one of: {allowed}")
        return v


class TranscriptionSegment(BaseModel):
    """A single segment of transcribed text with timing."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Confidence score")


class TranscriptionResponse(BaseResponse, TimestampedModel):
    """Response model for audio transcription."""
    text: str = Field("", description="Full transcribed text")
    segments: List[TranscriptionSegment] = Field([], description="Individual segments")
    language: Optional[str] = Field(None, description="Detected language code")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")


# ============================================================================
# TRANSLATION MODELS
# ============================================================================

class TranslationRequest(BaseModel):
    """Request model for text translation."""
    text: str = Field(..., min_length=1, description="Text to translate")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if None)")
    target_language: str = Field(..., min_length=1, description="Target language code")
    model: str = Field("facebook/nllb-200-distilled-600M", description="NLLB model to use")
    
    @field_validator("target_language")
    @classmethod
    def validate_target_language(cls, v: str) -> str:
        # Basic validation - language codes should be in format xxx_Xxxx
        if len(v) < 3:
            raise ValueError("Language code must be at least 3 characters")
        return v


class TranslationResponse(BaseResponse, TimestampedModel):
    """Response model for text translation."""
    original_text: str = Field("", description="Original input text")
    translated_text: str = Field("", description="Translated text")
    source_language: Optional[str] = Field(None, description="Detected source language")
    target_language: str = Field("", description="Target language")


# ============================================================================
# SYNTHESIS (TTS) MODELS
# ============================================================================

class SynthesisRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field("eng_Latn", description="Language code for voice selection")
    voice: Optional[str] = Field(None, description="Specific voice ID (auto-select if None)")
    speaker: Optional[int] = Field(None, description="Speaker ID for multi-speaker voices")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed factor")
    output_format: Literal["wav", "mp3", "ogg"] = Field("wav", description="Output audio format")


class SynthesisResponse(BaseResponse, TimestampedModel):
    """Response model for text-to-speech synthesis."""
    text: str = Field("", description="Text that was synthesized")
    audio_url: Optional[str] = Field(None, description="URL to download the audio file")
    audio_path: Optional[str] = Field(None, description="Server-side path to the audio file")
    voice: str = Field("", description="Voice ID used for synthesis")
    language: str = Field("", description="Language code")
    duration_seconds: Optional[float] = Field(None, description="Estimated audio duration")


# ============================================================================
# PIPELINE MODELS
# ============================================================================

class PipelineRequest(BaseModel):
    """Request model for pipeline processing."""
    pipeline_type: PipelineType = Field(..., description="Type of pipeline to run")
    
    # Input options (one must be provided)
    audio_path: Optional[str] = Field(None, description="Input audio file path")
    text: Optional[str] = Field(None, description="Input text")
    audio_url: Optional[str] = Field(None, description="URL to download input audio")
    
    # Language settings
    source_language: Optional[str] = Field(None, description="Source language (auto-detect if None)")
    target_language: str = Field(..., description="Target language code")
    
    # Model settings
    whisper_model: str = Field("small", description="Whisper model for transcription")
    nllb_model: str = Field("facebook/nllb-200-distilled-600M", description="NLLB model for translation")
    voice: Optional[str] = Field(None, description="Piper voice for synthesis")
    
    # Output settings
    output_format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (for TTS)")
    return_audio_data: bool = Field(False, description="Whether to return audio data in response")
    
    @field_validator("pipeline_type")
    @classmethod
    def validate_pipeline_inputs(cls, v, values):
        """Validate that appropriate inputs are provided for the pipeline type."""
        data = values.data
        
        if v == PipelineType.TRANSCRIBE_TRANSLATE:
            # Needs audio input
            if not data.get("audio_path") and not data.get("audio_url"):
                raise ValueError("TRANSCRIBE_TRANSLATE pipeline requires audio_path or audio_url")
        
        elif v == PipelineType.TRANSLATE_SYNTHESIZE:
            # Needs text input
            if not data.get("text"):
                raise ValueError("TRANSLATE_SYNTHESIZE pipeline requires text input")
        
        elif v == PipelineType.FULL:
            # Needs audio input
            if not data.get("audio_path") and not data.get("audio_url"):
                raise ValueError("FULL pipeline requires audio_path or audio_url")
        
        return v


class PipelineStageResult(BaseModel):
    """Result from a single pipeline stage."""
    stage: str = Field(..., description="Stage name")
    success: bool = Field(True, description="Whether stage succeeded")
    input: Any = Field(None, description="Stage input")
    output: Any = Field(None, description="Stage output")
    processing_time_ms: Optional[float] = Field(None, description="Stage processing time")
    error: Optional[str] = Field(None, description="Error message if stage failed")


class PipelineResponse(BaseResponse, TimestampedModel):
    """Response model for pipeline processing."""
    pipeline_type: str = Field("", description="Type of pipeline that was run")
    stages: List[PipelineStageResult] = Field([], description="Results from each stage")
    
    # Final outputs (depending on pipeline type)
    transcript: Optional[str] = Field(None, description="Transcription result")
    translated_text: Optional[str] = Field(None, description="Translation result")
    audio_url: Optional[str] = Field(None, description="URL to synthesized audio")
    audio_data: Optional[str] = Field(None, description="Base64-encoded audio data (if requested)")
    
    # Language info
    source_language: Optional[str] = Field(None, description="Detected/used source language")
    target_language: Optional[str] = Field(None, description="Target language used")


# ============================================================================
# HEALTH & STATUS MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Overall health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field("1.0.0", description="API version")


class ReadinessCheck(BaseModel):
    """Individual readiness check result."""
    name: str = Field(..., description="Check name")
    ready: bool = Field(..., description="Whether check passed")
    message: Optional[str] = Field(None, description="Optional message")


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str = Field(..., description="ready or unready")
    checks: List[ReadinessCheck] = Field([], description="Individual check results")


# ============================================================================
# VOICE/MODEL INFO MODELS
# ============================================================================

class VoiceInfo(BaseModel):
    """Information about an available voice."""
    id: str = Field(..., description="Voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    language: str = Field(..., description="Language code")
    gender: Optional[str] = Field(None, description="Voice gender if known")
    quality: Optional[str] = Field(None, description="Voice quality tier")


class AvailableVoicesResponse(BaseModel):
    """Response listing available voices."""
    voices: List[VoiceInfo] = Field([], description="Available voices")
    default_voice: Optional[str] = Field(None, description="Default voice ID")


class ModelInfo(BaseModel):
    """Information about an AI model."""
    name: str = Field(..., description="Model name/ID")
    type: str = Field(..., description="Model type (whisper, nllb, piper)")
    size: Optional[str] = Field(None, description="Model size description")
    loaded: bool = Field(False, description="Whether model is currently loaded")
    device: Optional[str] = Field(None, description="Device model is running on")


class ModelsResponse(BaseModel):
    """Response listing available models."""
    models: List[ModelInfo] = Field([], description="Available models")


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field with error (if applicable)")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Main error message")
    details: List[ErrorDetail] = Field([], description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
