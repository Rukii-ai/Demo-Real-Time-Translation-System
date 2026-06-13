"""API routes for the translation system.

This module defines all the REST API endpoints for:
- Health checks and system status
- Audio transcription (ASR)
- Text translation
- Text-to-speech synthesis
- Full pipeline processing
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.api.schemas import (
    # Request models
    PipelineRequest,
    SynthesisRequest,
    TranscriptionRequest,
    TranslationRequest,
    # Response models
    AvailableVoicesResponse,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    PipelineResponse,
    PipelineStageResult,
    ReadinessResponse,
    SynthesisResponse,
    TranscriptionResponse,
    TranslationResponse,
    VoiceInfo,
    # Enums and types
    PipelineType,
    ServiceType,
)
from app.asr.transcription import TranscriptionService
from app.asr.whisper_engine import WhisperEngine
from app.pipeline.pipeline import Pipeline
from app.translation.translator import Translator
from app.tts.synthesizer import Synthesizer
from app.tts.piper_engine import PiperEngine
from app.utils.audio import validate_audio_file
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Create the main API router
router = APIRouter(prefix="/api/v1")

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check if the API is running and healthy.
    
    Returns basic health status and uptime information.
    """
    return HealthResponse(
        status="ok",
        uptime_seconds=time.time() % 86400,  # Placeholder - would track actual uptime
        version="1.0.0",
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check() -> ReadinessResponse:
    """Check if all required services are ready to handle requests.
    
    Returns detailed readiness status for:
    - Database connection
    - Model loading status
    - Cache availability
    """
    from app.api.schemas import ReadinessCheck
    
    checks = []
    all_ready = True
    
    # Check models
    try:
        # Quick check if Whisper is available
        _ = WhisperEngine._instance is not None
        checks.append(ReadinessCheck(name="whisper_model", ready=True))
    except Exception:
        checks.append(ReadinessCheck(name="whisper_model", ready=False, message="Model not loaded"))
        all_ready = False
    
    # For now, assume other services are ready
    # In production, you'd check database, cache, etc.
    checks.append(ReadinessCheck(name="database", ready=True))
    checks.append(ReadinessCheck(name="cache", ready=True))
    
    return ReadinessResponse(
        status="ready" if all_ready else "unready",
        checks=checks,
    )


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@router.get("/voices", response_model=AvailableVoicesResponse, tags=["Info"])
async def list_voices() -> AvailableVoicesResponse:
    """List all available TTS voices.
    
    Returns information about locally available Piper voices
    that can be used for text-to-speech synthesis.
    """
    try:
        engine = PiperEngine()
        voice_ids = engine.available_voices()
        
        voices = []
        for vid in voice_ids:
            # Parse voice info from ID (e.g., "fr_FR-siwis-medium")
            parts = vid.split("-")
            lang = parts[0] if len(parts) > 0 else "unknown"
            name = parts[1] if len(parts) > 1 else vid
            quality = parts[2] if len(parts) > 2 else "medium"
            
            voices.append(VoiceInfo(
                id=vid,
                name=name,
                language=lang,
                quality=quality,
            ))
        
        return AvailableVoicesResponse(
            voices=voices,
            default_voice=voice_ids[0] if voice_ids else None,
        )
    except Exception as e:
        logger.error("Failed to list voices: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list voices: {str(e)}",
        )


@router.get("/models", response_model=ModelsResponse, tags=["Info"])
async def list_models() -> ModelsResponse:
    """List all available AI models and their loading status.
    
    Returns information about:
    - Whisper ASR models
    - NLLB translation models
    - Piper TTS voices
    """
    from app.api.schemas import ModelInfo
    
    models = [
        ModelInfo(
            name="whisper-small",
            type="whisper",
            size="small",
            loaded=WhisperEngine._instance is not None,
            device="cpu",
        ),
        ModelInfo(
            name="nllb-200-distilled-600M",
            type="nllb",
            size="600M",
            loaded=False,  # Would check singleton status
            device="cpu",
        ),
    ]
    
    return ModelsResponse(models=models)


# ============================================================================
# SERVICE ENDPOINTS
# ============================================================================

@router.post("/transcribe", response_model=TranscriptionResponse, tags=["Services"])
async def transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Expected language code"),
    model: str = Form("small", description="Whisper model to use"),
) -> TranscriptionResponse:
    """Transcribe audio to text using Whisper.
    
    This endpoint accepts an audio file and returns the transcribed text,
    along with timing information for each segment.
    
    Supported audio formats: WAV, MP3, FLAC, OGG, M4A
    
    Returns:
        TranscriptionResponse with text, segments, language, and duration.
    """
    start_time = time.time()
    
    try:
        # Save uploaded file to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = Path(tmp.name)
        
        try:
            # Validate the audio file
            validation = validate_audio_file(temp_path)
            if not validation["valid"]:
                raise ValueError(f"Invalid audio file: {validation['error']}")
            
            # Create transcription service
            engine = WhisperEngine(model_name=model, device="cpu", compute_type="int8")
            service = TranscriptionService(engine=engine)
            
            # Transcribe
            result = service.transcribe(temp_path, language=language)
            
            # Build response
            segments = [
                TranscriptionSegment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                )
                for s in result.get("segments", [])
            ]
            
            return TranscriptionResponse(
                success=True,
                text=result.get("text", ""),
                segments=segments,
                language=result.get("language"),
                duration=result.get("duration"),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


@router.post("/translate", response_model=TranslationResponse, tags=["Services"])
async def translate(request: TranslationRequest) -> TranslationResponse:
    """Translate text from one language to another.
    
    This endpoint accepts text and language codes, then returns
the translated text using the NLLB model.
    
    Returns:
        TranslationResponse with original and translated text.
    """
    start_time = time.time()
    
    try:
        # Create translator
        translator = Translator(
            engine_kwargs={"model_name": request.model, "device": "cpu"}
        )
        
        # Translate
        result = translator.translate(
            request.text,
            src_lang=request.source_language,
            tgt_lang=request.target_language,
        )
        
        return TranslationResponse(
            success=True,
            original_text=request.text,
            translated_text=result.get("text", ""),
            source_language=request.source_language or "auto-detected",
            target_language=request.target_language,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@router.post("/synthesize", response_model=SynthesisResponse, tags=["Services"])
async def synthesize(request: SynthesisRequest) -> SynthesisResponse:
    """Synthesize text to speech.
    
    This endpoint accepts text and voice parameters, then generates
    an audio file using the Piper TTS engine.
    
    Returns:
        SynthesisResponse with audio file URL and metadata.
    """
    start_time = time.time()
    
    try:
        # Create synthesizer
        engine = PiperEngine()
        synth = Synthesizer(engine=engine, out_dir=Path("data/tts"))
        
        # Synthesize
        output_path = synth.synthesize_to_file(
            text=request.text,
            lang_code=request.language,
            voice=request.voice,
            speaker=request.speaker,
        )
        
        return SynthesisResponse(
            success=True,
            text=request.text,
            audio_path=str(output_path),
            audio_url=f"/api/v1/audio/{output_path.name}",
            voice=request.voice or synth.voice_for_language(request.language),
            language=request.language,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}",
        )


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

@router.post("/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Run a multi-stage pipeline.
    
    This endpoint chains multiple services together based on the pipeline type:
    - transcribe_translate: Audio -> Text -> Translated Text
    - translate_synthesize: Text -> Translated Text -> Audio
    - full: Audio -> Text -> Translated Text -> Audio
    
    Returns:
        PipelineResponse with results from each stage and final output.
    """
    start_time = time.time()
    pipeline = Pipeline()
    stages = []
    
    try:
        if request.pipeline_type == PipelineType.TRANSCRIBE_TRANSLATE:
            # Stage 1: Transcription
            stage1_start = time.time()
            
            # Get audio from path or URL
            audio_path = request.audio_path
            if not audio_path and request.audio_url:
                # Would download from URL here
                raise NotImplementedError("Audio URL download not yet implemented")
            
            # Transcribe
            engine = WhisperEngine(model_name=request.whisper_model)
            service = TranscriptionService(engine=engine)
            asr_result = service.transcribe(audio_path, language=request.source_language)
            
            stage1 = PipelineStageResult(
                stage="transcription",
                success=True,
                input=audio_path,
                output=asr_result.get("text"),
                processing_time_ms=(time.time() - stage1_start) * 1000,
            )
            stages.append(stage1)
            
            # Stage 2: Translation
            stage2_start = time.time()
            translator = Translator(engine_kwargs={"model_name": request.nllb_model})
            trans_result = translator.translate(
                asr_result.get("text"),
                src_lang=asr_result.get("language"),
                tgt_lang=request.target_language,
            )
            
            stage2 = PipelineStageResult(
                stage="translation",
                success=True,
                input=asr_result.get("text"),
                output=trans_result.get("text"),
                processing_time_ms=(time.time() - stage2_start) * 1000,
            )
            stages.append(stage2)
            
            return PipelineResponse(
                success=True,
                pipeline_type=request.pipeline_type.value,
                stages=stages,
                transcript=asr_result.get("text"),
                translated_text=trans_result.get("text"),
                source_language=asr_result.get("language"),
                target_language=request.target_language,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        elif request.pipeline_type == PipelineType.TRANSLATE_SYNTHESIZE:
            # Stage 1: Translation
            stage1_start = time.time()
            translator = Translator(engine_kwargs={"model_name": request.nllb_model})
            trans_result = translator.translate(
                request.text,
                src_lang=request.source_language,
                tgt_lang=request.target_language,
            )
            
            stage1 = PipelineStageResult(
                stage="translation",
                success=True,
                input=request.text,
                output=trans_result.get("text"),
                processing_time_ms=(time.time() - stage1_start) * 1000,
            )
            stages.append(stage1)
            
            # Stage 2: Synthesis
            stage2_start = time.time()
            engine = PiperEngine()
            synth = Synthesizer(engine=engine, out_dir=Path("data/tts"))
            output_path = synth.synthesize_to_file(
                text=trans_result.get("text"),
                lang_code=request.target_language,
                voice=request.voice,
            )
            
            stage2 = PipelineStageResult(
                stage="synthesis",
                success=True,
                input=trans_result.get("text"),
                output=str(output_path),
                processing_time_ms=(time.time() - stage2_start) * 1000,
            )
            stages.append(stage2)
            
            return PipelineResponse(
                success=True,
                pipeline_type=request.pipeline_type.value,
                stages=stages,
                translated_text=trans_result.get("text"),
                audio_url=f"/api/v1/audio/{output_path.name}",
                target_language=request.target_language,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        elif request.pipeline_type == PipelineType.FULL:
            # Full pipeline: Audio -> Transcription -> Translation -> Synthesis
            
            # Stage 1: Transcription
            stage1_start = time.time()
            engine = WhisperEngine(model_name=request.whisper_model)
            service = TranscriptionService(engine=engine)
            
            audio_path = request.audio_path
            asr_result = service.transcribe(audio_path, language=request.source_language)
            
            stage1 = PipelineStageResult(
                stage="transcription",
                success=True,
                input=audio_path,
                output=asr_result.get("text"),
                processing_time_ms=(time.time() - stage1_start) * 1000,
            )
            stages.append(stage1)
            
            # Stage 2: Translation
            stage2_start = time.time()
            translator = Translator(engine_kwargs={"model_name": request.nllb_model})
            trans_result = translator.translate(
                asr_result.get("text"),
                src_lang=asr_result.get("language"),
                tgt_lang=request.target_language,
            )
            
            stage2 = PipelineStageResult(
                stage="translation",
                success=True,
                input=asr_result.get("text"),
                output=trans_result.get("text"),
                processing_time_ms=(time.time() - stage2_start) * 1000,
            )
            stages.append(stage2)
            
            # Stage 3: Synthesis
            stage3_start = time.time()
            piper_engine = PiperEngine()
            synth = Synthesizer(engine=piper_engine, out_dir=Path("data/tts"))
            output_path = synth.synthesize_to_file(
                text=trans_result.get("text"),
                lang_code=request.target_language,
                voice=request.voice,
            )
            
            stage3 = PipelineStageResult(
                stage="synthesis",
                success=True,
                input=trans_result.get("text"),
                output=str(output_path),
                processing_time_ms=(time.time() - stage3_start) * 1000,
            )
            stages.append(stage3)
            
            return PipelineResponse(
                success=True,
                pipeline_type=request.pipeline_type.value,
                stages=stages,
                transcript=asr_result.get("text"),
                translated_text=trans_result.get("text"),
                audio_url=f"/api/v1/audio/{output_path.name}",
                source_language=asr_result.get("language"),
                target_language=request.target_language,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )


# ============================================================================
# AUDIO FILE SERVING
# ============================================================================

@router.get("/audio/{filename}", tags=["Audio"])
async def get_audio_file(filename: str):
    """Download a generated audio file by filename.
    
    Args:
        filename: Name of the audio file to download
        
    Returns:
        The audio file as a download response
    """
    file_path = Path("data/tts") / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: {filename}",
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/wav",
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

def setup_error_handlers(app):
    """Set up global error handlers for the FastAPI app.
    
    Args:
        app: The FastAPI application instance
    """
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "message": str(exc) if False else "An unexpected error occurred",  # Hide details in production
            },
        )
