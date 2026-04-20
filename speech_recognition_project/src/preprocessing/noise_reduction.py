"""
noise_reduction.py — Spectral noise reduction for audio signals.

Uses the noisereduce library to apply non-stationary noise reduction
on raw audio arrays sampled at 16kHz.
"""

from __future__ import annotations

import numpy as np


class NoiseReducer:
    """Apply spectral subtraction-based noise reduction to an audio signal."""

    def __init__(self, stationary: bool = False, prop_decrease: float = 1.0):
        """
        Args:
            stationary:     If True, assume stationary noise (faster).
                            If False, use non-stationary reduction (better quality).
            prop_decrease:  Proportion of noise to reduce (0.0–1.0).
        """
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def reduce(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply noise reduction to an audio array.

        Args:
            audio: 1D float32 numpy array of audio samples.
            sr:    Sample rate (default 16000 Hz).

        Returns:
            Noise-reduced audio array of the same shape.

        Raises:
            ValueError: If audio is empty or not 1D.
        """
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1D audio array, got shape {audio.shape}"
            )
        if len(audio) == 0:
            raise ValueError("Audio array is empty.")

        try:
            import noisereduce as nr

            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
            )
            return reduced.astype(np.float32)
        except ImportError:
            # Graceful fallback: return audio unchanged if noisereduce not installed
            return audio
