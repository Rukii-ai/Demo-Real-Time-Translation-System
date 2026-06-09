from pathlib import Path

from app.asr.whisper_engine import WhisperEngine
from app.asr.transcription import TranscriptionService
from app.translation.translator import Translator
import subprocess
import tempfile
import imageio_ffmpeg


class DummyEngine:
    """A tiny fake translation engine for tests.

    It returns the input prefixed with a marker so we can assert the translation
    plumbing without downloading large HF models.
    """

    def translate(self, texts, src_lang, tgt_lang, **kwargs):
        single = isinstance(texts, str)
        if single:
            return {"text": f"[MOCK-{tgt_lang}] {texts}", "raw": None}
        return {"text": [f"[MOCK-{tgt_lang}] {t}" for t in texts], "raw": None}


def test_asr_to_translation_pipeline():
    # Use the existing ASR fixture that is present in the repo. This will exercise
    # the TranscriptionService -> WhisperEngine path (which already handles
    # incorrectly labeled fixtures via ffmpeg fallback).
    fixture = Path("tests/fixtures/asr/testfile_2.wav")
    assert fixture.exists(), "Expected ASR fixture to be present"

    # Initialize ASR (small model is fast for CPU in this repo setup).
    engine = WhisperEngine(model_name="small", device="cpu", compute_type="int8")
    service = TranscriptionService(engine=engine)

    # The fixture is a mislabeled WAV that soundfile cannot decode directly;
    # convert it via ffmpeg (the same fallback our runner uses) and transcribe the result.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run([ffmpeg, "-y", "-i", str(fixture), "-ac", "1", "-ar", "16000", tmp.name], check=True)
    asr_result = service.transcribe(tmp.name)
    text = asr_result.get("text")
    assert text and isinstance(text, str)

    # Inject our dummy translation engine so the test stays lightweight.
    translator = Translator(engine=DummyEngine())
    translated = translator.translate(text, src_lang=None, tgt_lang="fra_Latn")

    assert "[MOCK-fra_Latn]" in translated["text"]
