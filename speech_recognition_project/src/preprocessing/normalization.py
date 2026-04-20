"""
normalization.py — Amplitude normalization for audio signals.

Normalizes audio to a target peak amplitude and optionally resamples
to a target sample rate.
"""

from __future__ import annotations

import numpy as np


class AudioNormalizer:
    """Normalize audio amplitude and resample to a target sample rate."""

    def __init__(self, target_sr: int = 16000, target_peak: float = 0.95):
        """
        Args:
            target_sr:   Target sample rate in Hz.
            target_peak: Target peak amplitude (0.0–1.0). Defaults to 0.95
                         to leave a small headroom below clipping.
        """
        self.target_sr = target_sr
        self.target_peak = target_peak

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio to target_sr if needed.

        Args:
            audio:   1D float32 audio array.
            orig_sr: Original sample rate.

        Returns:
            Resampled audio array at self.target_sr.
        """
        if orig_sr == self.target_sr:
            return audio

        import librosa

        resampled = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=self.target_sr
        )
        return resampled.astype(np.float32)

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target peak amplitude.

        Args:
            audio: 1D float32 audio array.

        Returns:
            Amplitude-normalized audio array in [-target_peak, target_peak].

        Raises:
            ValueError: If audio is silent (all zeros).
        """
        peak = np.max(np.abs(audio))
        if peak == 0.0:
            raise ValueError(
                "Cannot normalize silent audio (all-zero signal)."
            )
        return (audio / peak * self.target_peak).astype(np.float32)

    def process(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample then normalize amplitude.

        Args:
            audio:   Raw audio array.
            orig_sr: Original sample rate.

        Returns:
            Resampled and normalized audio array.
        """
        audio = self.resample(audio, orig_sr)
        audio = self.normalize_amplitude(audio)
        return audio
