"""
test_features.py — Unit and property-based tests for MFCC feature extraction.

Verifies:
- Output shape is always (39,)
- No NaN or Inf values in output
- Delta and delta-delta computation
- Batch extraction consistency
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.features.mfcc_extraction import FeatureExtractor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(n_mfcc=13, sr=16000)

    def test_output_shape(self):
        audio = make_sine(duration=1.0)
        fv = self.extractor.extract(audio)
        assert fv.shape == (39,)

    def test_no_nan_in_output(self):
        audio = make_sine(duration=1.0)
        fv = self.extractor.extract(audio)
        assert np.isfinite(fv).all()

    def test_feature_dim_property(self):
        assert self.extractor.feature_dim == 39

    def test_empty_audio_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.extractor.extract(np.array([], dtype=np.float32))

    def test_nan_audio_raises(self):
        audio = np.full(16000, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            self.extractor.extract(audio)

    def test_batch_extraction_shape(self):
        audios = [make_sine(duration=1.0) for _ in range(5)]
        X = self.extractor.extract_batch(audios)
        assert X.shape == (5, 39)

    def test_different_durations_same_shape(self):
        """Feature vector shape must be (39,) regardless of audio duration."""
        for duration in [0.5, 1.0, 2.0, 3.0]:
            audio = make_sine(duration=duration)
            fv = self.extractor.extract(audio)
            assert fv.shape == (39,), f"Failed for duration={duration}"

    def test_mfcc_matrix_shape(self):
        audio = make_sine(duration=1.0)
        mfcc = self.extractor.extract_mfcc(audio)
        assert mfcc.shape[0] == 13

    def test_delta_shape_matches_mfcc(self):
        audio = make_sine(duration=1.0)
        mfcc = self.extractor.extract_mfcc(audio)
        delta = self.extractor.extract_delta(mfcc)
        assert delta.shape == mfcc.shape

    def test_delta_delta_shape_matches_mfcc(self):
        audio = make_sine(duration=1.0)
        mfcc = self.extractor.extract_mfcc(audio)
        delta2 = self.extractor.extract_delta_delta(mfcc)
        assert delta2.shape == mfcc.shape


# ── Property-Based Tests ──────────────────────────────────────────────────────

@given(
    duration=st.floats(min_value=0.5, max_value=3.0),
    freq=st.floats(min_value=100.0, max_value=4000.0),
)
@settings(max_examples=20, deadline=5000)
def test_feature_shape_property(duration: float, freq: float):
    """
    Property: For any valid audio duration and frequency,
    the extracted feature vector always has shape (39,).
    """
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    extractor = FeatureExtractor(n_mfcc=13, sr=sr)
    fv = extractor.extract(audio)

    assert fv.shape == (39,)
    assert np.isfinite(fv).all()
