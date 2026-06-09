"""Unified ASR helper script.

This script combines two previous helpers:
1) running ASR across all fixture files
2) comparing outputs across multiple model sizes
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import imageio_ffmpeg

from app.asr.transcription import TranscriptionService
from app.asr.whisper_engine import WhisperEngine

# Load the project-level .env so HF_TOKEN is available when a model download is needed.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _transcribe_with_fallback(service: TranscriptionService, path: Path) -> dict:
    """Try direct transcription first, then decode with ffmpeg if the file is mislabeled."""
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


def run_fixtures(model_name: str) -> None:
    """Transcribe all fixture files with one model and print outputs."""
    fixture_dir = Path("tests/fixtures/asr")
    # Discover fixtures dynamically so the script still works when fixture names change.
    files = sorted(
        [p.name for p in fixture_dir.iterdir() if p.is_file()]
    ) if fixture_dir.exists() else []

    if not files:
        print(f"No fixture files found in: {fixture_dir}")
        return

    engine = WhisperEngine(model_name=model_name, device="cpu", compute_type="int8")
    service = TranscriptionService(engine=engine)

    for name in files:
        path = fixture_dir / name
        print(f"=== {name} ({model_name}) ===")
        if not path.exists():
            print("MISSING")
            continue

        result = _transcribe_with_fallback(service, path)
        print(f"TEXT: {result['text']!r}")
        print(f"LANGUAGE: {result['language']}")
        print(f"SEGMENTS: {result['segments']}")
        print()


def compare_models(models: list[str], target_file: Path) -> None:
    """Transcribe one target file with each model so you can compare quality/speed."""
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    for model_name in models:
        print(f"=== {model_name} ===")
        engine = WhisperEngine(model_name=model_name, device="cpu", compute_type="int8")
        service = TranscriptionService(engine=engine)
        result = _transcribe_with_fallback(service, target_file)
        print(f"TEXT: {result['text']!r}")
        print(f"LANGUAGE: {result['language']}")
        print(f"SEGMENTS: {result['segments']}")
        print()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for choosing fixture run or model comparison mode."""
    parser = argparse.ArgumentParser(description="Run ASR fixtures or compare ASR models.")
    parser.add_argument(
        "--mode",
        choices=["fixtures", "compare"],
        default="fixtures",
        help="fixtures: run many fixture files with one model; compare: run one file with many models",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model used in fixtures mode (default: small)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "small"],
        help="Models used in compare mode (default: base small)",
    )
    parser.add_argument(
        "--file",
        default="tests/fixtures/asr/testfile_2.wav",
        help="File used in compare mode",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point that dispatches to the selected mode."""
    args = parse_args()

    if args.mode == "fixtures":
        run_fixtures(model_name=args.model)
        return

    compare_models(models=args.models, target_file=Path(args.file))


if __name__ == "__main__":
    main()
