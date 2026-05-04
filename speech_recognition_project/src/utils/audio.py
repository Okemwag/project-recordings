"""Audio loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load mono audio, falling back to librosa for formats soundfile cannot decode."""
    audio_path = str(path)
    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        import librosa

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio.astype(np.float32), int(sr)

    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sr)
