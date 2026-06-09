"""Model wrapper for transcription models."""

from __future__ import annotations

import io
from app.utils.logger import get_logger
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel


logger = get_logger(__name__)


class WhisperEngine:
    _instance: WhisperEngine | None = None
    _lock = Lock()

    def __new__(cls, model_name: str = "base", device: str | None = None, compute_type: str | None = None):
        # Create exactly one engine instance so model loading happens once per process.
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "base", device: str | None = None, compute_type: str | None = None):
        # Skip re-initialization when the singleton already finished setup.
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        # Use GPU when available; otherwise default to CPU for compatibility.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Choose a default compute type that matches the selected device.
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        logger.info(
            "Initializing WhisperEngine with model=%s device=%s compute_type=%s",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(model_name, device=self.device, compute_type=self.compute_type)
        self._initialized = True
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: str | Path | bytes | np.ndarray, 
                   language: str | None = None, task: str = "transcribe",
                   ) -> dict[str, Any]:
        # Normalize all accepted audio inputs into what faster-whisper expects.
        logger.info(
            "Whisper transcription started (input_type=%s language=%s task=%s)",
            type(audio).__name__,
            language,
            task,
        )
        audio_source, temp_file = self._prepare_audio(audio)

        try:
            segments_iter, info = self.model.transcribe(
                audio_source,
                language=language,
                task=task,
            )
        except Exception:
            logger.exception("Whisper transcription failed")
            raise
        finally:
            if temp_file is not None:
                logger.info("Cleaning up temporary audio file used during preprocessing")
                temp_file.unlink(missing_ok=True)

        segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            for segment in segments_iter
        ]

        # Drop blank segments so the final transcript is easier to consume.
        segments = [
            segment
            for segment in segments
            if segment["text"]
        ]

        # Join segment text into one plain transcript string.
        text = " ".join(segment["text"] for segment in segments).strip()

        logger.info(
            "Whisper transcription finished (segments=%d language=%s duration=%s)",
            len(segments),
            getattr(info, "language", language),
            getattr(info, "duration", None),
        )

        return {
            "text": text,
            "segments": segments,
            "language": getattr(info, "language", language),
            "duration": getattr(info, "duration", None),
        }

    def _prepare_audio(
        self, audio: str | Path | bytes | np.ndarray
    ) -> tuple[str | np.ndarray, Path | None]:
        # Accept raw numpy audio directly.
        if isinstance(audio, np.ndarray):
            return self._normalize_numpy_audio(audio), None

        # Accept pathlib file paths.
        if isinstance(audio, Path):
            return self._load_audio_file(audio), None

        # Accept string file paths.
        if isinstance(audio, str):
            return self._load_audio_file(Path(audio)), None

        # Accept in-memory byte payloads.
        if isinstance(audio, bytes):
            return self._load_audio_bytes(audio), None

        raise TypeError("audio must be a file path, bytes, or numpy array")

    def _load_audio_file(self, path: Path) -> np.ndarray:
        if not path.exists():
            logger.error("Audio file not found at path=%s", path)
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Read as float32 for stable model input.
        try:
            audio, _sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            logger.exception("Failed to decode audio file via soundfile path=%s", path)
            logger.info("Fallback decoding may be handled by upstream helper scripts")
            raise
        return self._normalize_numpy_audio(audio)

    def _load_audio_bytes(self, audio: bytes) -> np.ndarray:
        buffer = io.BytesIO(audio)
        # Decode bytes through soundfile exactly like file-based reads.
        try:
            data, _sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
        except Exception:
            logger.exception("Failed to decode audio bytes payload via soundfile")
            logger.info("Fallback decoding may be handled by upstream helper scripts")
            raise
        return self._normalize_numpy_audio(data)

    def _normalize_numpy_audio(self, audio: np.ndarray) -> np.ndarray:
        # Force ndarray so downstream operations are consistent.
        array = np.asarray(audio)

        # If audio has multiple channels, average to mono.
        if array.ndim == 2:
            array = array.mean(axis=1)

        # Ensure dtype matches what the transcription backend expects.
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        return array