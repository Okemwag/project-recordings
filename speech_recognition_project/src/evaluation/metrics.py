"""
metrics.py — Evaluation metrics for the Kiswahili ASR system.

Computes accuracy, precision, recall, F1, confusion matrix,
Word Error Rate (WER), and per-accent breakdowns.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


class Evaluator:
    """Compute and aggregate evaluation metrics for ASR predictions."""

    def __init__(self, label_encoder: LabelEncoder):
        """
        Args:
            label_encoder: Fitted LabelEncoder used during training.
        """
        self.label_encoder = label_encoder

    def accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Overall classification accuracy."""
        return float(accuracy_score(y_true, y_pred))

    def precision_recall_f1(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """Macro-averaged precision, recall, and F1 score."""
        return {
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Confusion matrix with class labels."""
        labels = np.arange(len(self.label_encoder.classes_))
        return confusion_matrix(y_true, y_pred, labels=labels)

    def word_error_rate(
        self, references: list[str], hypotheses: list[str]
    ) -> float:
        """
        Compute Word Error Rate for isolated-word recognition.

        For isolated-word tasks, WER simplifies to:
            WER = number of incorrect predictions / total predictions

        Args:
            references:  List of ground-truth word strings.
            hypotheses:  List of predicted word strings.

        Returns:
            WER as a float in [0.0, 1.0].

        Raises:
            ValueError: If lists have different lengths or are empty.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"references ({len(references)}) and hypotheses "
                f"({len(hypotheses)}) must have the same length."
            )
        if len(references) == 0:
            raise ValueError("Cannot compute WER on empty lists.")

        errors = sum(r != h for r, h in zip(references, hypotheses))
        return errors / len(references)

    def per_accent_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        accent_labels: np.ndarray,
    ) -> dict:
        """
        Compute accuracy per accent group.

        Args:
            y_true:        Ground-truth integer labels.
            y_pred:        Predicted integer labels.
            accent_labels: Accent string for each sample.

        Returns:
            Dict mapping accent name → accuracy float.
        """
        report = {}
        for accent in np.unique(accent_labels):
            mask = accent_labels == accent
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y_true[mask], y_pred[mask])
            report[accent] = float(acc)
        return report

    def full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        accent_labels: np.ndarray,
    ) -> dict:
        """
        Generate a comprehensive evaluation report.

        Args:
            y_true:        Ground-truth integer labels.
            y_pred:        Predicted integer labels.
            accent_labels: Accent string for each sample.

        Returns:
            Dict with keys: accuracy, precision, recall, f1, wer,
            confusion_matrix, per_accent, classification_report.
        """
        prf = self.precision_recall_f1(y_true, y_pred)

        # Convert integer labels back to word strings for WER
        refs = self.label_encoder.inverse_transform(y_true).tolist()
        hyps = self.label_encoder.inverse_transform(y_pred).tolist()

        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "wer": self.word_error_rate(refs, hyps),
            "confusion_matrix": self.confusion_matrix(y_true, y_pred),
            "per_accent": self.per_accent_report(y_true, y_pred, accent_labels),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=np.arange(len(self.label_encoder.classes_)),
                target_names=self.label_encoder.classes_,
                zero_division=0,
            ),
        }
