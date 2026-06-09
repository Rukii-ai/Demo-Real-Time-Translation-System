"""End-to-end test runner: convert audio -> ASR -> NLLB translation.

This script is a developer helper to run a single fixture through the
ASR and translation pipeline in-process. It performs an ffmpeg conversion
if needed, then uses `WhisperEngine` and `Translator` programmatically.

Use from the project root with the project venv active.
"""
from pathlib import Path
import subprocess
import sys
import logging

import imageio_ffmpeg as ffmpeg

from app.utils.logger import get_logger
from app.asr.whisper_engine import WhisperEngine
from app.asr.transcription import TranscriptionService
from app.translation.translator import Translator


logger = get_logger("scripts.run_e2e_test")


def convert_to_wav(src: Path, dst: Path) -> None:
    """Convert an input audio file to 16k mono WAV using ffmpeg binary."""
    ff = ffmpeg.get_ffmpeg_exe()
    logger.info("Converting %s -> %s using ffmpeg=%s", src, dst, ff)
    subprocess.run([ff, "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_e2e_test.py <path-to-audio> [tgt_lang]")
        sys.exit(1)

    src = Path(sys.argv[1])
    tgt = sys.argv[2] if len(sys.argv) > 2 else "fra_Latn"

    if not src.exists():
        logger.error("Audio file not found: %s", src)
        sys.exit(2)

    fixed = src.with_name(src.stem + "_fixed.wav")
    try:
        convert_to_wav(src, fixed)
    except Exception as exc:
        logger.exception("ffmpeg conversion failed: %s", exc)
        sys.exit(3)

    # ASR
    engine = WhisperEngine(model_name="small", device="cpu", compute_type="int8")
    service = TranscriptionService(engine=engine)
    asr_result = service.transcribe(fixed)
    text = asr_result.get("text", "")
    lang = asr_result.get("language")
    print("TRANSCRIPT:\n", text)
    print("ASR language:", lang)

    # Translation
    translator = Translator(engine_kwargs={"model_name": "facebook/nllb-200-distilled-600M", "device": "cpu"})
    translation = translator.translate(text, src_lang=lang, tgt_lang=tgt)
    print("\nTRANSLATION:\n", translation.get("text"))


if __name__ == "__main__":
    main()
