"""
test_inference.py — Unit tests for the InferenceEngine.

Verifies error handling for silent, short, and valid audio inputs
using a mock model and scaler.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from unittest.mock import MagicMock

from src.inference.predict import InferenceEngine, PredictionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silent(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def make_engine(vocabulary: list[str] | None = None) -> InferenceEngine:
    """Build an InferenceEngine with a mock model and fitted scaler/encoder."""
    vocab = vocabulary or ["maji", "chakula", "nyumba", "shule", "gari"]

    le = LabelEncoder()
    le.fit(vocab)

    scaler = StandardScaler()
    scaler.fit(np.random.randn(50, 39))

    # Mock model that always predicts class 0 with high confidence
    mock_model = MagicMock()
    n_classes = len(vocab)
    proba = np.zeros((1, n_classes))
    proba[0, 0] = 0.9
    proba[0, 1:] = 0.1 / (n_classes - 1)
    mock_model.predict_proba.return_value = proba

    return InferenceEngine(
        model=mock_model,
        scaler=scaler,
        label_encoder=le,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInferenceEngine:
    def setup_method(self):
        self.engine = make_engine()

    def test_valid_audio_returns_word(self):
        audio = make_sine(duration=1.0)
        # Inject audio directly via _run_pipeline
        result = self.engine._run_pipeline(audio)
        assert not result.is_error
        assert result.predicted_word in ["maji", "chakula", "nyumba", "shule", "gari"]

    def test_silent_audio_returns_error(self):
        audio = make_silent(duration=1.0)
        result = self.engine._run_pipeline(audio)
        assert result.is_error
        assert "No speech detected" in result.error

    def test_confidence_in_range(self):
        audio = make_sine(duration=1.0)
        result = self.engine._run_pipeline(audio)
        if not result.is_error:
            assert 0.0 <= result.confidence <= 1.0

    def test_top_k_length(self):
        audio = make_sine(duration=1.0)
        result = self.engine._run_pipeline(audio)
        if not result.is_error:
            assert len(result.top_k) <= self.engine.top_k

    def test_predicted_word_in_vocabulary(self):
        audio = make_sine(duration=1.0)
        result = self.engine._run_pipeline(audio)
        if not result.is_error:
            vocab = list(self.engine.label_encoder.classes_)
            assert result.predicted_word in vocab

    def test_prediction_result_is_error_property(self):
        ok = PredictionResult(predicted_word="maji", confidence=0.9)
        err = PredictionResult(error="No speech detected")
        assert not ok.is_error
        assert err.is_error
