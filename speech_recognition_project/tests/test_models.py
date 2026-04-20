"""
test_models.py — Unit tests for SVM and ANN model classes.

Uses synthetic data to verify train/predict interfaces,
error handling, and basic correctness.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.models.svm_model import SVMModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_synthetic_data(
    n_samples: int = 100,
    n_features: int = 39,
    n_classes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature matrix and integer labels."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float64)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


# ── SVMModel ──────────────────────────────────────────────────────────────────

class TestSVMModel:
    """Test suite for SVMModel — Task 6: SVM model verification and hardening."""
    
    def setup_method(self):
        self.model = SVMModel(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        self.X, self.y = make_synthetic_data()

    def test_6_1_train_fits_without_error(self):
        """Task 6.1: Verify SVMModel.train() fits the model on valid (X_train, y_train) without error."""
        # Should complete without raising any exception
        self.model.train(self.X, self.y)
        # Verify model is marked as trained
        assert self.model._is_trained is True
        # Verify predictions work after training
        preds = self.model.predict(self.X)
        assert preds.shape == (len(self.X),)

    def test_6_2_predict_returns_labels_within_training_set(self):
        """Task 6.2: Verify SVMModel.predict() returns labels within the training label set."""
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)
        
        # All predictions must be in the training label set
        training_labels = set(np.unique(self.y))
        assert all(p in training_labels for p in preds), \
            f"Found predictions outside training labels: {set(preds) - training_labels}"

    def test_6_3_predict_proba_rows_sum_to_one(self):
        """Task 6.3: Verify SVMModel.predict_proba() returns rows that sum to 1.0 within tolerance 1e-5."""
        self.model.train(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        
        # Verify shape
        n_classes = len(np.unique(self.y))
        assert proba.shape == (len(self.X), n_classes)
        
        # Verify each row sums to 1.0 within tolerance
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="Probability rows do not sum to 1.0")

    def test_6_4_predict_raises_runtime_error_before_train(self):
        """Task 6.4: Verify SVMModel.predict() raises RuntimeError when called before train()."""
        model = SVMModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(self.X)

    def test_6_4_predict_proba_raises_runtime_error_before_train(self):
        """Task 6.4: Verify SVMModel.predict_proba() raises RuntimeError when called before train()."""
        model = SVMModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict_proba(self.X)

    def test_6_4_save_raises_runtime_error_before_train(self):
        """Task 6.4: Verify SVMModel.save() raises RuntimeError when called before train()."""
        model = SVMModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.save("dummy_path.joblib")

    def test_6_5_train_raises_value_error_on_mismatched_lengths(self):
        """Task 6.5: Verify SVMModel.train() raises ValueError when X_train and y_train have mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            self.model.train(self.X, self.y[:50])
        
        # Also test the reverse case
        with pytest.raises(ValueError, match="same length"):
            self.model.train(self.X[:50], self.y)

    def test_6_6_save_load_roundtrip_identical_predictions(self, tmp_path):
        """Task 6.6: Verify SVMModel.save() and load() round-trip produces identical predictions."""
        # Train the original model
        self.model.train(self.X, self.y)
        
        # Get predictions from original model
        preds_original = self.model.predict(self.X)
        proba_original = self.model.predict_proba(self.X)
        
        # Save the model
        path = str(tmp_path / "svm_model.joblib")
        self.model.save(path)
        
        # Load into a new instance
        loaded = SVMModel()
        loaded.load(path)
        
        # Get predictions from loaded model
        preds_loaded = loaded.predict(self.X)
        proba_loaded = loaded.predict_proba(self.X)
        
        # Verify predictions are identical
        np.testing.assert_array_equal(preds_original, preds_loaded,
                                      err_msg="Predictions differ after save/load")
        np.testing.assert_allclose(proba_original, proba_loaded, atol=1e-10,
                                   err_msg="Probabilities differ after save/load")


# ── ANNModel (optional — requires TensorFlow) ─────────────────────────────────

def test_ann_model_basic():
    """Basic smoke test for ANNModel — skipped if TensorFlow not installed."""
    pytest.importorskip("tensorflow")

    from src.models.ann_model import ANNModel

    X, y = make_synthetic_data(n_samples=200, n_classes=5)
    model = ANNModel(
        input_dim=39,
        num_classes=5,
        hidden_units=[64, 32],
        dropout_rate=0.1,
        learning_rate=0.01,
        random_state=42,
    )
    model.train(X, y, epochs=3, batch_size=32, validation_split=0.1)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert all(0 <= p < 5 for p in preds)
