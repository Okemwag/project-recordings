"""
mfcc_extraction.py — MFCC feature extraction with delta and delta-delta.

Extracts a fixed-size feature vector from a preprocessed audio signal:
  - 13 MFCC coefficients
  - 13 delta (Δ) coefficients
  - 13 delta-delta (ΔΔ) coefficients
  → 39-dimensional feature vector (mean across time frames)
"""

from __future__ import annotations

import numpy as np


class FeatureExtractor:
    """Extract MFCC + Δ + ΔΔ features from a preprocessed audio signal."""

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 40,
        sr: int = 16000,
    ):
        """
        Args:
            n_mfcc:      Number of MFCC coefficients (default 13).
            n_fft:       FFT window size in samples (default 512 = 32ms @ 16kHz).
            hop_length:  Hop length in samples (default 160 = 10ms @ 16kHz).
            n_mels:      Number of mel filterbanks (default 40).
            sr:          Expected sample rate (default 16000).
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr

    @property
    def feature_dim(self) -> int:
        """Total feature vector dimension: n_mfcc × 3 (MFCC + Δ + ΔΔ)."""
        return self.n_mfcc * 3

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute MFCC matrix.

        Args:
            audio: 1D float32 audio array at self.sr.

        Returns:
            MFCC matrix of shape (n_mfcc, T).
        """
        import librosa

        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def extract_delta(self, mfcc: np.ndarray) -> np.ndarray:
        """Compute first-order delta coefficients from MFCC matrix."""
        import librosa

        return librosa.feature.delta(mfcc, order=1)

    def extract_delta_delta(self, mfcc: np.ndarray) -> np.ndarray:
        """Compute second-order delta-delta coefficients from MFCC matrix."""
        import librosa

        return librosa.feature.delta(mfcc, order=2)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract the full 39-dimensional feature vector from audio.

        Computes MFCC + Δ + ΔΔ matrices, stacks them, then takes the
        mean across the time axis to produce a fixed-size vector.

        Args:
            audio: 1D float32 audio array at self.sr.

        Returns:
            Feature vector of shape (39,).

        Raises:
            ValueError: If audio is empty or contains NaN/Inf values.
        """
        if len(audio) == 0:
            raise ValueError("Cannot extract features from empty audio.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values.")

        mfcc = self.extract_mfcc(audio)
        delta = self.extract_delta(mfcc)
        delta2 = self.extract_delta_delta(mfcc)

        # Stack along coefficient axis: shape (39, T)
        combined = np.vstack([mfcc, delta, delta2])

        # Aggregate across time: mean per coefficient → shape (39,)
        feature_vector = np.mean(combined, axis=1)

        assert feature_vector.shape == (self.feature_dim,), (
            f"Expected shape ({self.feature_dim},), got {feature_vector.shape}"
        )
        assert np.isfinite(feature_vector).all(), (
            "Feature vector contains NaN or Inf after extraction."
        )

        return feature_vector.astype(np.float64)

    def extract_batch(self, audio_list: list[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of audio arrays.

        Args:
            audio_list: List of 1D float32 audio arrays.

        Returns:
            Feature matrix of shape (N, 39).
        """
        features = [self.extract(audio) for audio in audio_list]
        return np.array(features)
