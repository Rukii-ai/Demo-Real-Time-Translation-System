"""Audio utilities for the translation system.

This module provides reusable audio processing functions including:
- Audio file loading and validation
- Audio format conversion and normalization
- FFmpeg fallback for problematic audio files
- Audio quality fixing utilities
"""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import soundfile as sf

from app.utils.logger import get_logger

try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_AVAILABLE = True
except ImportError:
    IMAGEIO_FFMPEG_AVAILABLE = False

logger = get_logger(__name__)

# Standard audio settings for the translation system
SAMPLE_RATE = 16000  # 16kHz is standard for speech models
CHANNELS = 1  # Mono audio


def get_ffmpeg_path() -> str | None:
    """Get the path to ffmpeg executable.
    
    Tries imageio-ffmpeg first, then falls back to system ffmpeg.
    
    Returns:
        Path to ffmpeg executable or None if not found.
    """
    # Try imageio-ffmpeg first (bundled ffmpeg)
    if IMAGEIO_FFMPEG_AVAILABLE:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    
    # Fall back to system ffmpeg
    try:
        result = subprocess.run(
            ["where", "ffmpeg"] if Path("/").drive == "C:" else ["which", "ffmpeg"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    
    return None


def convert_to_standard_wav(
    input_path: Path,
    output_path: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> Path:
    """Convert any audio file to standard WAV format using FFmpeg.
    
    Args:
        input_path: Path to input audio file.
        output_path: Path for output WAV file. If None, creates temp file.
        sample_rate: Target sample rate (default: 16000).
        channels: Number of channels (default: 1 for mono).
    
    Returns:
        Path to the converted WAV file.
    
    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it or imageio-ffmpeg: pip install imageio-ffmpeg"
        )
    
    # Create temp file if no output path specified
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
    
    try:
        # Run ffmpeg conversion
        subprocess.run(
            [
                ffmpeg,
                "-y",  # Overwrite output
                "-i", str(input_path),  # Input file
                "-ar", str(sample_rate),  # Sample rate
                "-ac", str(channels),  # Channels
                "-c:a", "pcm_s16le",  # PCM 16-bit little-endian
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Converted %s to %s", input_path, output_path)
        return output_path
    except subprocess.CalledProcessError as e:
        # Clean up temp file on failure
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e


def load_audio(
    audio: str | Path | bytes | np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    dtype: str = "float32",
) -> np.ndarray:
    """Load audio from various sources into a numpy array.
    
    Supports:
    - File paths (str or Path)
    - Raw bytes (in-memory audio)
    - numpy arrays (passed through after validation)
    
    Args:
        audio: Audio source (path, bytes, or array)
        sample_rate: Expected sample rate (default: 16000)
        dtype: Data type for output array (default: float32)
    
    Returns:
        Numpy array of audio samples.
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        TypeError: If audio type is not supported
        RuntimeError: If audio cannot be decoded
    """
    # Handle numpy arrays directly
    if isinstance(audio, np.ndarray):
        return _normalize_audio(audio, dtype)
    
    # Handle pathlib paths
    if isinstance(audio, Path):
        return _load_audio_file(audio, sample_rate, dtype)
    
    # Handle string paths
    if isinstance(audio, str):
        return _load_audio_file(Path(audio), sample_rate, dtype)
    
    # Handle bytes
    if isinstance(audio, bytes):
        return _load_audio_bytes(audio, sample_rate, dtype)
    
    raise TypeError(f"Unsupported audio type: {type(audio)}")


def _load_audio_file(
    path: Path,
    sample_rate: int,
    dtype: str,
    fallback_to_ffmpeg: bool = True,
) -> np.ndarray:
    """Load audio from a file path.
    
    Args:
        path: Path to audio file
        sample_rate: Expected sample rate
        dtype: Output data type
        fallback_to_ffmpeg: Whether to try ffmpeg on decode failure
    
    Returns:
        Audio as numpy array
    """
    if not path.exists():
        logger.error("Audio file not found: %s", path)
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        # Try to load with soundfile first
        audio, sr = sf.read(str(path), dtype=dtype, always_2d=False)
        if sr != sample_rate:
            # Resample if needed (would need librosa, skip for now)
            logger.warning(
                "Audio sample rate mismatch: expected %d, got %d",
                sample_rate,
                sr,
            )
        return _normalize_audio(audio, dtype)
    except Exception as e:
        logger.warning("Failed to decode audio with soundfile: %s", e)
        
        # Try ffmpeg fallback if enabled
        if fallback_to_ffmpeg:
            logger.info("Attempting ffmpeg fallback for %s", path)
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                
                converted = convert_to_standard_wav(path, tmp_path, sample_rate, CHANNELS)
                audio, sr = sf.read(str(converted), dtype=dtype, always_2d=False)
                
                # Clean up temp file
                converted.unlink(missing_ok=True)
                
                return _normalize_audio(audio, dtype)
            except Exception as ffmpeg_error:
                logger.error("FFmpeg fallback failed: %s", ffmpeg_error)
                raise RuntimeError(
                    f"Could not decode audio file {path}. "
                    f"Original error: {e}. "
                    f"FFmpeg fallback error: {ffmpeg_error}"
                ) from ffmpeg_error
        
        raise RuntimeError(f"Failed to decode audio file {path}: {e}") from e


def _load_audio_bytes(
    audio: bytes,
    sample_rate: int,
    dtype: str,
) -> np.ndarray:
    """Load audio from raw bytes.
    
    Args:
        audio: Raw audio bytes
        sample_rate: Expected sample rate
        dtype: Output data type
    
    Returns:
        Audio as numpy array
    """
    buffer = io.BytesIO(audio)
    try:
        data, sr = sf.read(buffer, dtype=dtype, always_2d=False)
        if sr != sample_rate:
            logger.warning(
                "Audio sample rate mismatch: expected %d, got %d",
                sample_rate,
                sr,
            )
        return _normalize_audio(data, dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio bytes: {e}") from e


def _normalize_audio(
    audio: np.ndarray,
    dtype: str = "float32",
) -> np.ndarray:
    """Normalize audio to the target data type and shape.
    
    Args:
        audio: Input audio array (can be 1D or 2D)
        dtype: Target data type
    
    Returns:
        Normalized audio array
    """
    # Ensure numpy array
    audio = np.asarray(audio)
    
    # Convert dtype if needed
    if dtype == "float32":
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
    elif dtype == "int16":
        if audio.dtype == np.float32:
            audio = (audio * 32767.0).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
    
    # Ensure 1D (mono) if 2D but single channel
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio.flatten()
    
    return audio


def fix_wav_file(input_path: Path, output_path: Path | None = None) -> Path:
    """Fix malformed or non-standard WAV files.
    
    Uses ffmpeg to re-encode WAV files that may have issues like:
    - Incorrect headers
    - Wrong byte order
    - Corrupted chunks
    - Non-standard formats
    
    Args:
        input_path: Path to the problematic WAV file
        output_path: Where to save the fixed file. If None, creates temp file.
    
    Returns:
        Path to the fixed WAV file
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
    
    # Use ffmpeg to re-encode the file with standard WAV settings
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not available for WAV fixing")
    
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", str(input_path),
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", str(SAMPLE_RATE),  # 16kHz
                "-ac", str(CHANNELS),  # Mono
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Fixed WAV file: %s -> %s", input_path, output_path)
        return output_path
    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to fix WAV file: {e.stderr.decode() if e.stderr else 'Unknown error'}") from e


def validate_audio_file(path: Path) -> dict[str, Any]:
    """Validate an audio file and return its properties.
    
    Args:
        path: Path to audio file
    
    Returns:
        Dictionary with audio properties:
        - valid: Whether the file is valid
        - sample_rate: Sample rate in Hz
        - channels: Number of channels
        - duration: Duration in seconds
        - format: Audio format
        - error: Error message if invalid
    """
    result = {
        "valid": False,
        "sample_rate": None,
        "channels": None,
        "duration": None,
        "format": None,
        "error": None,
    }
    
    if not path.exists():
        result["error"] = f"File not found: {path}"
        return result
    
    try:
        # Try to read with soundfile
        info = sf.info(str(path))
        result["valid"] = True
        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["duration"] = info.duration
        result["format"] = info.format
    except Exception as e:
        # Try ffmpeg as fallback
        ffmpeg = get_ffmpeg_path()
        if ffmpeg:
            try:
                result_ffprobe = subprocess.run(
                    [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
                     "format=duration", "-show_entries", "stream=sample_rate,channels",
                     "-of", "default=noprint_wrappers=1", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # Parse ffprobe output
                for line in result_ffprobe.stdout.splitlines():
                    if "sample_rate=" in line:
                        result["sample_rate"] = int(line.split("=")[1])
                    elif "channels=" in line:
                        result["channels"] = int(line.split("=")[1])
                    elif "duration=" in line:
                        result["duration"] = float(line.split("=")[1])
                
                if result["sample_rate"] and result["duration"]:
                    result["valid"] = True
                    result["format"] = "unknown (ffprobe)"
            except Exception as ffprobe_error:
                result["error"] = f"soundfile error: {e}. ffprobe error: {ffprobe_error}"
        else:
            result["error"] = f"soundfile error: {e}. ffmpeg not available for fallback."
    
    return result


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio from one sample rate to another.
    
    Uses simple linear interpolation for speed. For production quality,
    consider using librosa or scipy.signal.resample.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    
    # Calculate new length
    new_length = int(len(audio) * ratio)
    
    # Use linear interpolation
    indices = np.linspace(0, len(audio) - 1, new_length)
    indices_floor = np.floor(indices).astype(np.int64)
    indices_ceil = np.minimum(indices_floor + 1, len(audio) - 1)
    fractions = indices - indices_floor
    
    resampled = audio[indices_floor] * (1 - fractions) + audio[indices_ceil] * fractions
    
    return resampled
