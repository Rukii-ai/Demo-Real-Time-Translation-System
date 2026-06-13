"""Run translation-system text-to-speech from the command line.

Examples:
  python scripts/synthesize.py --translated-text "Bonjour tout le monde" --tgt fra_Latn
  python scripts/synthesize.py --file tests/fixtures/asr/testfile_fixed.wav --tgt fra_Latn --mock-translation

The full pipeline is:
  audio file -> Whisper ASR -> NLLB translation -> Piper TTS -> WAV file
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Running `python scripts/synthesize.py` puts scripts/ on sys.path. Add the
    # project root too, so imports like `from app...` work from any directory.
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.transcription import TranscriptionService
from app.asr.whisper_engine import WhisperEngine
from app.translation.translator import Translator
from app.tts.piper_engine import PiperEngine
from app.tts.synthesizer import Synthesizer
from app.utils.logger import get_logger


load_dotenv(PROJECT_ROOT / ".env")
logger = get_logger("scripts.synthesize")


class DummyTranslationEngine:
    """Tiny test translator that avoids loading NLLB during TTS smoke tests."""

    def translate(self, texts, src_lang, tgt_lang, **kwargs):
        single = isinstance(texts, str)
        if single:
            return {"text": f"[MOCK:{tgt_lang}] {texts}", "raw": None}
        return {"text": [f"[MOCK:{tgt_lang}] {text}" for text in texts], "raw": None}


def _transcribe_with_fallback(service: TranscriptionService, path: Path) -> dict:
    """Transcribe audio, retrying through ffmpeg when soundfile cannot decode it."""
    try:
        return service.transcribe(path)
    except Exception as exc:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(tmp_path)],
                check=True,
                capture_output=True,
            )
            return service.transcribe(tmp_path)
        except Exception:
            raise RuntimeError(f"Failed to transcribe '{path}' directly and after ffmpeg decode.") from exc
        finally:
            tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Collect command-line options for text-only or full audio pipeline runs."""
    parser = argparse.ArgumentParser(description="Synthesize translated text with Piper.")
    parser.add_argument("--file", "-f", help="Optional source audio file for ASR -> translation -> TTS")
    parser.add_argument("--translated-text", help="Already translated text; skips ASR and translation")
    parser.add_argument("--source-text", help="Source text to translate before TTS; skips ASR")
    parser.add_argument("--tgt", default="fra_Latn", help="Target NLLB language code, e.g. fra_Latn")
    parser.add_argument("--src", default=None, help="Optional source NLLB language code")
    parser.add_argument("--voice", default=None, help="Optional Piper voice ID or .onnx path")
    parser.add_argument("--out-dir", default="data/tts", help="Directory for generated WAV files")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Piper device")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model for ASR")
    parser.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M", help="NLLB model name")
    parser.add_argument("--mock-translation", action="store_true", help="Use a fake translator for quick TTS tests")
    parser.add_argument("--no-cache", action="store_true", help="Always regenerate the WAV file")
    return parser.parse_args()


def main() -> None:
    """Run the requested pipeline and print the important outputs."""
    args = parse_args()

    if args.translated_text:
        translated_text = args.translated_text
        transcript = None
        translation = {"text": translated_text, "raw": None}
    else:
        if args.source_text:
            transcript = args.source_text
            asr_language = args.src
        else:
            if not args.file:
                raise SystemExit("Pass --translated-text, --source-text, or --file.")

            audio_path = Path(args.file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            logger.info("Starting ASR for %s", audio_path)
            asr_engine = WhisperEngine(model_name=args.whisper_model, device="cpu", compute_type="int8")
            asr_service = TranscriptionService(engine=asr_engine)
            asr_result = _transcribe_with_fallback(asr_service, audio_path)
            transcript = str(asr_result.get("text", "")).strip()
            asr_language = asr_result.get("language")

        if args.mock_translation:
            translator = Translator(engine=DummyTranslationEngine())
        else:
            translator = Translator(engine_kwargs={"model_name": args.nllb_model, "device": "cpu"})

        logger.info("Translating text to %s", args.tgt)
        translation = translator.translate(transcript, src_lang=args.src or asr_language, tgt_lang=args.tgt)
        translated_text = translation.get("text")

    if isinstance(translated_text, list):
        translated_text = " ".join(str(item) for item in translated_text)
    translated_text = str(translated_text or "").strip()
    if not translated_text:
        raise RuntimeError("Translation produced empty text; nothing to synthesize.")

    piper_engine = PiperEngine(device=args.device)
    synth = Synthesizer(engine=piper_engine, out_dir=Path(args.out_dir), cache=(not args.no_cache))
    out_wav = synth.synthesize_to_file(translated_text, lang_code=args.tgt, voice=args.voice)

    if transcript is not None:
        print("TRANSCRIPT:")
        print(transcript)
        print()

    print("TRANSLATION:")
    print(translated_text)
    print()
    print("Translated audio written to:", out_wav)


if __name__ == "__main__":
    main()
