"""
test_preprocessing.py — Unit tests for preprocessing modules.

Tests noise reduction, silence trimming, amplitude normalization,
and resampling with known inputs and expected outputs.
"""

import numpy as np
import pytest

from src.preprocessing.noise_reduction import NoiseReducer
from src.preprocessing.normalization import AudioNormalizer
from src.preprocessing.silence_removal import InsufficientAudioError, SilenceRemover


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silent(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a silent (all-zero) audio array."""
    return np.zeros(int(sr * duration), dtype=np.float32)


# ── NoiseReducer ──────────────────────────────────────────────────────────────

class TestNoiseReducer:
    def setup_method(self):
        self.reducer = NoiseReducer(stationary=False, prop_decrease=1.0)

    def test_reduce_handles_1d_float32(self):
        """Verify NoiseReducer.reduce() handles 1D float32 arrays."""
        audio = make_sine(duration=1.0)
        result = self.reducer.reduce(audio, sr=16000)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert len(result) == len(audio)

    def test_reduce_empty_raises_valueerror(self):
        """Verify NoiseReducer.reduce() raises ValueError for empty input."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            self.reducer.reduce(audio, sr=16000)

    def test_reduce_non_1d_raises_valueerror(self):
        """Verify NoiseReducer.reduce() raises ValueError for non-1D input."""
        audio = np.zeros((2, 16000), dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            self.reducer.reduce(audio, sr=16000)

    def test_reduce_2d_single_channel_raises(self):
        """Verify NoiseReducer.reduce() raises ValueError for 2D arrays even with single channel."""
        audio = np.zeros((1, 16000), dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            self.reducer.reduce(audio, sr=16000)


# ── AudioNormalizer ───────────────────────────────────────────────────────────

class TestAudioNormalizer:
    def setup_method(self):
        self.normalizer = AudioNormalizer(target_sr=16000, target_peak=0.95)

    def test_normalize_amplitude_peak(self):
        audio = make_sine(duration=1.0)
        normalized = self.normalizer.normalize_amplitude(audio)
        assert np.max(np.abs(normalized)) == pytest.approx(0.95, abs=1e-5)

    def test_normalize_amplitude_range(self):
        audio = make_sine(duration=1.0)
        normalized = self.normalizer.normalize_amplitude(audio)
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_normalize_silent_raises(self):
        audio = make_silent()
        with pytest.raises(ValueError, match="silent"):
            self.normalizer.normalize_amplitude(audio)

    def test_resample_same_rate_unchanged(self):
        audio = make_sine(duration=0.5)
        result = self.normalizer.resample(audio, orig_sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_resample_different_rate(self):
        audio = make_sine(duration=1.0, sr=8000)
        result = self.normalizer.resample(audio, orig_sr=8000)
        # After resampling 8kHz → 16kHz, length should roughly double
        assert len(result) == pytest.approx(16000, rel=0.05)


# ── SilenceRemover ────────────────────────────────────────────────────────────

class TestSilenceRemover:
    def setup_method(self):
        self.remover = SilenceRemover(
            top_db=20.0, min_duration=0.5, max_duration=3.0
        )

    def test_trim_returns_shorter_or_equal(self):
        audio = make_sine(duration=1.0)
        trimmed = self.remover.trim(audio, sr=16000)
        assert len(trimmed) <= len(audio)

    def test_trim_too_short_raises(self):
        # 0.1s of audio — after trimming should be too short
        audio = make_sine(duration=0.1)
        with pytest.raises(InsufficientAudioError):
            self.remover.trim(audio, sr=16000)

    def test_trim_truncates_long_audio(self):
        audio = make_sine(duration=5.0)
        trimmed = self.remover.trim(audio, sr=16000)
        max_samples = int(3.0 * 16000)
        assert len(trimmed) <= max_samples

    def test_trim_empty_raises(self):
        with pytest.raises(ValueError):
            self.remover.trim(np.array([], dtype=np.float32), sr=16000)

    def test_trim_2d_raises(self):
        audio = np.zeros((2, 16000), dtype=np.float32)
        with pytest.raises(ValueError):
            self.remover.trim(audio, sr=16000)
