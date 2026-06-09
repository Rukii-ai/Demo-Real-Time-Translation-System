"""Small helper to run ASR -> translation pipeline from the command line.

Usage:
  python scripts/translate_runner.py --file tests/fixtures/asr/testfile_2.wav --tgt fra_Latn

This script will:
- load project .env (so HF token is available)
- transcribe the provided audio using `WhisperEngine`/`TranscriptionService`
- translate the transcribed text using `Translator` (which uses `NLLBEngine`)

Note: NLLB models are large; if you don't have a local cached model or token,
the translation step may fail or download large weights. For quick tests you can
pass `--mock` to use a fake translator engine.
"""

from pathlib import Path
from dotenv import load_dotenv
import argparse

from app.utils.logger import get_logger
from app.asr.whisper_engine import WhisperEngine
from app.asr.transcription import TranscriptionService
from app.translation.translator import Translator

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = get_logger("scripts.translate_runner")


class DummyEngine:
    """Simple fake engine used when `--mock` is passed. Returns deterministic output."""

    def translate(self, texts, src_lang, tgt_lang, **kwargs):
        single = isinstance(texts, str)
        if single:
            return {"text": f"[MOCK:{tgt_lang}] {texts}", "raw": None}
        return {"text": [f"[MOCK:{tgt_lang}] {t}" for t in texts], "raw": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", required=True, help="Path to audio file")
    p.add_argument("--model", default="small", help="Whisper model name (faster-whisper)")
    p.add_argument("--nllb_model", default="facebook/nllb-200-distilled-600M", help="NLLB model name for translation")
    p.add_argument("--tgt", required=True, help="Target language code for NLLB (e.g. fra_Latn)")
    p.add_argument("--mock", action="store_true", help="Use fake translator engine (no HF model)")
    args = p.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        raise SystemExit(2)

    # ASR
    logger.info("Starting ASR (model=%s) for %s", args.model, audio_path)
    engine = WhisperEngine(model_name=args.model, device="cpu", compute_type="int8")
    service = TranscriptionService(engine=engine)
    asr_result = service.transcribe(audio_path)
    transcript = asr_result.get("text", "")
    logger.info("ASR complete (language=%s duration=%s)", asr_result.get("language"), asr_result.get("duration"))
    print("TRANSCRIPT:\n", transcript)

    # Translator
    # By default we create a Translator that will construct an NLLBEngine singleton.
    # You can also use --mock to avoid downloading or loading HF models during quick tests.
    if args.mock:
        translator = Translator(engine=DummyEngine())
    else:
        # Pass requested model name into engine_kwargs so the translator creates the
        # NLLB engine with the desired model.
        translator = Translator(engine_kwargs={"model_name": args.nllb_model, "device": "cpu"})

    logger.info("Translating to %s (using NLLB model=%s)", args.tgt, args.nllb_model)
    # Use the language detected by ASR as the source language when available.
    translation = translator.translate(transcript, src_lang=asr_result.get("language"), tgt_lang=args.tgt)
    print("\nTRANSLATION:\n", translation.get("text"))


if __name__ == "__main__":
    main()
