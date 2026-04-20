"""
test_metrics.py — Unit tests for the Evaluator class.

Verifies accuracy, WER, per-accent breakdown, and full report
on known ground-truth / prediction pairs.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.evaluation.metrics import Evaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator():
    le = LabelEncoder()
    le.fit(["chakula", "maji", "nyumba", "shule", "gari"])
    return Evaluator(label_encoder=le)


@pytest.fixture
def perfect_preds(evaluator):
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_pred = y_true.copy()
    accents = np.array(
        ["coastal", "nairobi", "upcountry", "coastal", "nairobi",
         "upcountry", "coastal", "nairobi"]
    )
    return y_true, y_pred, accents


@pytest.fixture
def imperfect_preds(evaluator):
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 1, 0, 2])  # 2 errors
    accents = np.array(
        ["coastal", "nairobi", "upcountry", "coastal", "nairobi",
         "upcountry", "coastal", "nairobi"]
    )
    return y_true, y_pred, accents


# ── Accuracy ──────────────────────────────────────────────────────────────────

def test_perfect_accuracy(evaluator, perfect_preds):
    y_true, y_pred, _ = perfect_preds
    assert evaluator.accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_imperfect_accuracy(evaluator, imperfect_preds):
    y_true, y_pred, _ = imperfect_preds
    acc = evaluator.accuracy(y_true, y_pred)
    assert acc == pytest.approx(6 / 8)


# ── WER ───────────────────────────────────────────────────────────────────────

def test_wer_perfect(evaluator):
    refs = ["maji", "chakula", "nyumba"]
    hyps = ["maji", "chakula", "nyumba"]
    assert evaluator.word_error_rate(refs, hyps) == pytest.approx(0.0)


def test_wer_all_wrong(evaluator):
    refs = ["maji", "chakula", "nyumba"]
    hyps = ["gari", "shule", "gari"]
    assert evaluator.word_error_rate(refs, hyps) == pytest.approx(1.0)


def test_wer_partial(evaluator):
    refs = ["maji", "chakula", "nyumba", "gari"]
    hyps = ["maji", "shule", "nyumba", "gari"]  # 1 error
    assert evaluator.word_error_rate(refs, hyps) == pytest.approx(0.25)


def test_wer_mismatched_lengths_raises(evaluator):
    with pytest.raises(ValueError, match="same length"):
        evaluator.word_error_rate(["maji"], ["maji", "chakula"])


def test_wer_empty_raises(evaluator):
    with pytest.raises(ValueError, match="empty"):
        evaluator.word_error_rate([], [])


# ── Per-Accent Report ─────────────────────────────────────────────────────────

def test_per_accent_report_keys(evaluator, imperfect_preds):
    y_true, y_pred, accents = imperfect_preds
    report = evaluator.per_accent_report(y_true, y_pred, accents)
    assert set(report.keys()) == {"coastal", "nairobi", "upcountry"}


def test_per_accent_report_range(evaluator, imperfect_preds):
    y_true, y_pred, accents = imperfect_preds
    report = evaluator.per_accent_report(y_true, y_pred, accents)
    for acc_val in report.values():
        assert 0.0 <= acc_val <= 1.0


# ── Full Report ───────────────────────────────────────────────────────────────

def test_full_report_keys(evaluator, imperfect_preds):
    y_true, y_pred, accents = imperfect_preds
    report = evaluator.full_report(y_true, y_pred, accents)
    expected_keys = {
        "accuracy", "precision", "recall", "f1",
        "wer", "confusion_matrix", "per_accent", "classification_report"
    }
    assert expected_keys.issubset(set(report.keys()))


def test_full_report_accuracy_wer_consistent(evaluator, imperfect_preds):
    y_true, y_pred, accents = imperfect_preds
    report = evaluator.full_report(y_true, y_pred, accents)
    # For isolated-word: WER = 1 - accuracy
    assert report["wer"] == pytest.approx(1.0 - report["accuracy"], abs=1e-5)
