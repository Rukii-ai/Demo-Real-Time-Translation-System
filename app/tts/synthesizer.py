"""High-level text-to-speech service for translated text.

`PiperEngine` does the model work. `Synthesizer` handles app-level concerns:
choosing a voice for a language, creating stable filenames, using the cache, and
logging what happened.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
import threading
import time
from typing import Optional

from app.tts.piper_engine import DEFAULT_VOICE_IDS_BY_NLLB_LANG, PiperEngine
from app.utils.logger import get_logger


logger = get_logger(__name__)

_DEFAULT_OUT_DIR = Path("data/tts")


def _safe_filename(text: str, max_len: int = 64) -> str:
    """Turn arbitrary text into a short filename-safe label."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    return (cleaned[:max_len] or "tts").strip("_")


class Synthesizer:
    """Convenience layer for creating translated audio files."""

    def __init__(
        self,
        engine: Optional[PiperEngine] = None,
        out_dir: Optional[Path] = None,
        cache: bool = True,
        max_files_per_minute: int = 30,
    ):
        self.engine = engine or PiperEngine()
        self.out_dir = Path(out_dir or _DEFAULT_OUT_DIR)
        self.cache = cache
        self.max_files_per_minute = max(1, int(max_files_per_minute))

        # These voice IDs match Piper's official voice naming style. A voice is
        # usable after its `.onnx` and `.onnx.json` files exist in models/piper.
        self.voice_map = dict(DEFAULT_VOICE_IDS_BY_NLLB_LANG)

        # Protect batch file creation from bursts. This is separate from the
        # engine's synthesis semaphore because it controls file-level pacing.
        self._file_rate_lock = threading.Lock()
        self._recent_file_times: list[float] = []

    def available_voices(self) -> list[str]:
        """Return local Piper voices the engine can load right now."""
        return self.engine.available_voices()

    def voice_for_language(self, lang_code: str) -> str:
        """Choose the best Piper voice ID for an NLLB language code."""
        if lang_code in self.voice_map:
            return self.voice_map[lang_code]

        available = self.available_voices()
        if available:
            fallback = available[0]
            logger.warning(
                "No voice mapped for language=%s; falling back to local voice=%s",
                lang_code,
                fallback,
            )
            return fallback

        fallback = self.voice_map["eng_Latn"]
        logger.warning(
            "No voice mapped for language=%s and no local voices found; using configured fallback=%s",
            lang_code,
            fallback,
        )
        return fallback

    def synthesize_to_file(
        self,
        text: str,
        lang_code: str,
        voice: Optional[str] = None,
        filename: Optional[str] = None,
        speaker: Optional[int] = None,
    ) -> Path:
        """Create a WAV file for translated text and return its path."""
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        self._wait_for_file_rate_limit()
        voice = voice or self.voice_for_language(lang_code)

        # Use text + voice + language for the cache key so repeated translations
        # reuse the same audio file.
        cache_key = hashlib.sha1(f"{lang_code}::{voice}::{text}".encode("utf-8")).hexdigest()
        safe_name = filename or f"{_safe_filename(text, max_len=32)}_{cache_key[:8]}.wav"
        out_path = self.out_dir / safe_name

        if self.cache and out_path.exists():
            logger.info("Using cached TTS file (path=%s)", out_path)
            return out_path

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Creating TTS file (lang=%s voice=%s output=%s)",
            lang_code,
            voice,
            out_path,
        )
        return self.engine.synthesize(text=text, voice=voice, out_path=out_path, speaker=speaker)

    def _wait_for_file_rate_limit(self) -> None:
        """Limit how quickly many output files can be created in batch jobs."""
        window_seconds = 60.0
        now = time.perf_counter()

        with self._file_rate_lock:
            self._recent_file_times = [
                started_at
                for started_at in self._recent_file_times
                if now - started_at < window_seconds
            ]

            if len(self._recent_file_times) >= self.max_files_per_minute:
                oldest = self._recent_file_times[0]
                wait_for = window_seconds - (now - oldest)
                if wait_for > 0:
                    logger.info("Rate limiting TTS file creation for %.2f seconds", wait_for)
                    time.sleep(wait_for)

            self._recent_file_times.append(time.perf_counter())
