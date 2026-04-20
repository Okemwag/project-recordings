"""SVM classifier wrapper used by the training and inference pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC


class SVMModel:
    """Thin, serializable wrapper around sklearn's probability-enabled SVC."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        random_state: int = 42,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=random_state,
        )
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMModel":
        """Fit the SVM on feature vectors and integer labels."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_train.shape}.")
        if len(np.unique(y_train)) < 2:
            raise ValueError("SVM training requires at least two classes.")

        self.model.fit(X_train, y_train)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels."""
        self._require_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each row in X."""
        self._require_trained()
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def load(self, path: str | Path) -> "SVMModel":
        """Load a persisted model into this instance."""
        loaded = joblib.load(path)
        if isinstance(loaded, SVMModel):
            self.__dict__.update(loaded.__dict__)
        elif isinstance(loaded, SVC):
            self.model = loaded
            self._is_trained = True
        else:
            raise TypeError(f"Unsupported SVM artifact type: {type(loaded)!r}")
        return self

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("SVMModel has not been trained.")
