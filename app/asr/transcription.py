"""Transcription service for handling audio transcription tasks."""

from __future__ import annotations

from app.utils.logger import get_logger
import time
from pathlib import Path
from typing import Any
import numpy as np

from app.asr.whisper_engine import WhisperEngine


logger = get_logger(__name__)


class TranscriptionService:
    # Keep the service lightweight: if no engine is provided, build one lazily.
    def __init__(self, engine: WhisperEngine | None = None):
        self.engine = engine or WhisperEngine()
        logger.info("TranscriptionService initialized (engine=%s)", type(self.engine).__name__)

    def transcribe(self, audio: str | Path | bytes | np.ndarray,
                   language: str | None = None,) -> dict[str, Any]:
        # Delegate model work to the engine, then normalize the output shape.
        start = time.perf_counter()
        logger.info(
            "Transcription request received (input_type=%s requested_language=%s)",
            type(audio).__name__,
            language,
        )

        try:
            raw_result = self.engine.transcribe(audio, language=language)
        except Exception:
            logger.exception("Transcription failed in service layer")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Transcription request completed (resolved_language=%s duration_ms=%.2f)",
            raw_result.get("language"),
            elapsed_ms,
        )
        return self._format_result(raw_result)

    def _format_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        # Ensure text is always a clean string for API callers.
        text = str(raw_result.get("text", "")).strip()
        # Use an empty list if segment details are missing.
        segments = raw_result.get("segments", [])

        return {
            "text": text,
            "segments": segments,
            "language": raw_result.get("language"),
            "duration": raw_result.get("duration"),
        }