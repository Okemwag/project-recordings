"""
silence_removal.py — Silence trimming for audio signals.

Trims leading and trailing silence from audio using librosa's
energy-based trimming, then validates the resulting duration.
"""

from __future__ import annotations

import numpy as np


class InsufficientAudioError(Exception):
    """Raised when audio is too short after silence trimming."""


class SilenceRemover:
    """Trim silence from the start and end of an audio signal."""

    def __init__(
        self,
        top_db: float = 20.0,
        min_duration: float = 0.5,
        max_duration: float = 3.0,
    ):
        """
        Args:
            top_db:        Threshold (dB below peak) below which frames are
                           considered silent.
            min_duration:  Minimum acceptable duration in seconds after trimming.
            max_duration:  Maximum acceptable duration in seconds; longer audio
                           is truncated.
        """
        self.top_db = top_db
        self.min_duration = min_duration
        self.max_duration = max_duration

    def trim(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Trim silence from audio and validate duration.

        Args:
            audio: 1D float32 numpy array.
            sr:    Sample rate.

        Returns:
            Trimmed (and possibly truncated) audio array.

        Raises:
            InsufficientAudioError: If duration after trimming < min_duration.
            ValueError:             If audio is empty or not 1D.
        """
        import librosa

        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1D audio array, got shape {audio.shape}"
            )
        if len(audio) == 0:
            raise ValueError("Audio array is empty.")

        trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)

        duration = len(trimmed) / sr

        if duration < self.min_duration:
            raise InsufficientAudioError(
                f"Audio too short after trimming: {duration:.3f}s "
                f"(minimum {self.min_duration}s required)"
            )

        # Truncate if too long
        max_samples = int(self.max_duration * sr)
        if len(trimmed) > max_samples:
            trimmed = trimmed[:max_samples]

        return trimmed.astype(np.float32)
