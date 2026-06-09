"""Tests for the ASR module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.asr.transcription import TranscriptionService
from app.asr.whisper_engine import WhisperEngine


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "asr"


def _is_usable_audio(path: Path) -> bool:
    # Basic sanity check so tests skip cleanly when fixture files are missing.
    return path.exists() and path.is_file() and path.stat().st_size > 0


@pytest.fixture(scope="module")
def service() -> TranscriptionService:
    # Reuse one engine for all tests in this module to avoid repeated model startup cost.
    engine = WhisperEngine(model_name="small", device="cpu", compute_type="int8")
    return TranscriptionService(engine=engine)


@pytest.mark.parametrize(
    "filename",
    [
        "short_audio.mp3",
        "multilingual_audio.mp3",
        "noisy_audio.mp3",
    ],
)
def test_speech_audio_transcription(service, filename):
    # Build the full path to the current fixture under test.
    path = FIXTURE_DIR / filename
    if not _is_usable_audio(path):
        pytest.skip(f"Missing or empty audio fixture: {path}")

    # Measure runtime so extremely slow regressions are caught.
    start = time.perf_counter()
    result = service.transcribe(path)
    elapsed = time.perf_counter() - start

    transcript = str(result["text"]).strip()
    assert isinstance(result["segments"], list)
    assert elapsed < 120

    # Some fixtures are noisy or multilingual, so this is a smoke test rather than
    # a transcript-accuracy benchmark.
    assert transcript or result["segments"]


def test_silence_transcription(service):
    # Silence input should not generate meaningful transcript text.
    path = FIXTURE_DIR / "silence.mp3"
    if not _is_usable_audio(path):
        pytest.skip(f"Missing or empty audio fixture: {path}")

    result = service.transcribe(path)
    transcript = str(result["text"]).strip()

    assert isinstance(result["segments"], list)
    assert transcript == "" or transcript.lower() == "you"