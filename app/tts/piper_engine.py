"""Piper text-to-speech engine wrapper.

This file is the low-level TTS adapter for the translation system. It knows how
to load Piper voice models and write translated text to WAV files.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import threading
import time
import wave
from typing import Optional

from app.utils.logger import get_logger

try:
    # piper-tts 1.4.2 exposes these public classes from the `piper` package.
    from piper import PiperVoice, SynthesisConfig
except Exception:  # pragma: no cover - exercised only when dependency is absent
    PiperVoice = None
    SynthesisConfig = None


logger = get_logger(__name__)

try:
    PIPER_PACKAGE_VERSION = metadata.version("piper-tts")
except metadata.PackageNotFoundError:  # pragma: no cover
    PIPER_PACKAGE_VERSION = "not-installed"


DEFAULT_VOICES_DIR = Path("models/piper")

# These are official Piper voice IDs. The corresponding files should live at:
# models/piper/<voice-id>.onnx and models/piper/<voice-id>.onnx.json.
DEFAULT_VOICE_IDS_BY_NLLB_LANG = {
    "eng_Latn": "en_US-lessac-medium",
    "fra_Latn": "fr_FR-siwis-medium",
    "spa_Latn": "es_ES-sharvard-medium",
    "deu_Latn": "de_DE-thorsten-medium",
    "ita_Latn": "it_IT-paola-medium",
    "por_Latn": "pt_BR-faber-medium",
}


def _default_device() -> str:
    """Prefer CPU unless the caller explicitly asks for CUDA."""
    return "cpu"


class PiperEngine:
    """Small, reusable wrapper around Piper's Python API.

    The class is a singleton because loading voice models can be expensive. One
    engine can safely serve many requests while limiting concurrent synthesis.
    """

    _instance_lock = threading.Lock()
    _instance: Optional["PiperEngine"] = None

    def __new__(cls, *args, **kwargs):
        # Reuse one engine object so model caches and locks are shared.
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        device: Optional[str] = None,
        voices_dir: Optional[Path] = None,
        max_concurrent: int = 1,
        min_interval_seconds: float = 0.05,
    ):
        if getattr(self, "_initialized", False):
            return

        # Piper's Python package accepts use_cuda=True/False when loading a voice.
        self.device = (device or _default_device()).lower()
        self.voices_dir = Path(voices_dir or DEFAULT_VOICES_DIR)
        self.max_concurrent = max(1, int(max_concurrent))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))

        # Cache loaded voice objects by voice ID/path so repeat calls are fast.
        self._voices: dict[str, PiperVoice] = {}
        self._voice_lock = threading.Lock()

        # This semaphore prevents too many ONNX synthesis jobs from running at
        # once, which protects CPU/RAM when batch-generating files.
        self._synthesis_slots = threading.Semaphore(self.max_concurrent)
        self._rate_lock = threading.Lock()
        self._last_synthesis_started = 0.0

        self._initialized = True
        logger.info(
            "PiperEngine initialized (piper_tts=%s device=%s voices_dir=%s max_concurrent=%d)",
            PIPER_PACKAGE_VERSION,
            self.device,
            self.voices_dir,
            self.max_concurrent,
        )

    @property
    def package_version(self) -> str:
        """Return the installed `piper-tts` package version."""
        return PIPER_PACKAGE_VERSION

    def available_voices(self) -> list[str]:
        """Return voice IDs that already have local ONNX model/config files."""
        if not self.voices_dir.exists():
            return []

        voices: list[str] = []
        for model_path in sorted(self.voices_dir.glob("*.onnx")):
            config_path = self._config_path_for_model(model_path)
            if config_path.exists():
                voices.append(model_path.stem)
        return voices

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Optional[Path] = None,
        speaker: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w_scale: Optional[float] = None,
        volume: float = 1.0,
    ) -> Path:
        """Synthesize `text` with a Piper voice and write it to a WAV file."""
        if PiperVoice is None or SynthesisConfig is None:
            raise RuntimeError("piper-tts is not importable. Install it with `pip install piper-tts`.")

        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        if out_path is None:
            out_path = Path("data/tts/output.wav")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self._wait_for_rate_limit()
        logger.info("Waiting for Piper synthesis slot (voice=%s output=%s)", voice, out_path)

        with self._synthesis_slots:
            start = time.perf_counter()
            try:
                piper_voice = self._load_voice(voice)
                syn_config = SynthesisConfig(
                    speaker_id=speaker,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w_scale,
                    volume=volume,
                )

                with wave.open(str(out_path), "wb") as wav_file:
                    # PiperVoice.synthesize_wav writes the correct WAV header
                    # and audio frames for us.
                    piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            except Exception:
                logger.exception("Piper synthesis failed (voice=%s output=%s)", voice, out_path)
                raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Piper synthesis completed (output=%s duration_ms=%.2f)", out_path, elapsed_ms)
        return out_path

    def _wait_for_rate_limit(self) -> None:
        """Space out synthesis starts so batch jobs do not spike resources."""
        if self.min_interval_seconds <= 0:
            return

        with self._rate_lock:
            now = time.perf_counter()
            wait_for = self.min_interval_seconds - (now - self._last_synthesis_started)
            if wait_for > 0:
                logger.info("Rate limiting Piper synthesis for %.3f seconds", wait_for)
                time.sleep(wait_for)
            self._last_synthesis_started = time.perf_counter()

    def _load_voice(self, voice: str) -> PiperVoice:
        """Load a Piper voice model once and reuse it for future calls."""
        model_path = self._resolve_model_path(voice)
        cache_key = str(model_path.resolve())

        with self._voice_lock:
            if cache_key in self._voices:
                return self._voices[cache_key]

            config_path = self._config_path_for_model(model_path)
            logger.info("Loading Piper voice model=%s config=%s", model_path, config_path)
            loaded_voice = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=(self.device == "cuda"),
                download_dir=self.voices_dir,
            )
            self._voices[cache_key] = loaded_voice
            return loaded_voice

    def _resolve_model_path(self, voice: str) -> Path:
        """Resolve a voice ID or explicit path to a local `.onnx` file."""
        candidate = Path(voice)
        if candidate.suffix == ".onnx" and candidate.exists():
            return candidate

        model_path = self.voices_dir / f"{voice}.onnx"
        if model_path.exists() and self._config_path_for_model(model_path).exists():
            return model_path

        available = ", ".join(self.available_voices()) or "none"
        raise FileNotFoundError(
            f"Piper voice '{voice}' is not downloaded in {self.voices_dir}. "
            f"Available local voices: {available}. "
            f"Download one with: .venv\\Scripts\\python -m piper.download_voices {voice} "
            f"--download-dir {self.voices_dir}"
        )

    @staticmethod
    def _config_path_for_model(model_path: Path) -> Path:
        """Piper voice configs are named like `<voice-id>.onnx.json`."""
        return Path(f"{model_path}.json")
