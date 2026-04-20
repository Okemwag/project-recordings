"""Optional TensorFlow/Keras ANN classifier wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ANNModel:
    """Feed-forward neural classifier with the same interface as SVMModel."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_units: list[int] | None = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units or [256, 128]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self._is_trained = False

    def _tensorflow(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for ANNModel. Install it with "
                "`uv sync --extra ann`."
            ) from exc
        return tf

    def _build(self):
        tf = self._tensorflow()
        tf.keras.utils.set_random_seed(self.random_state)

        layers = [tf.keras.layers.Input(shape=(self.input_dim,))]
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation="relu"))
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
        layers.append(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0,
    ) -> "ANNModel":
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_train.shape}.")

        self.model = self._build()
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X, verbose=0)

    def save(self, path: str | Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str | Path) -> "ANNModel":
        tf = self._tensorflow()
        self.model = tf.keras.models.load_model(path)
        self.input_dim = int(self.model.input_shape[-1])
        self.num_classes = int(self.model.output_shape[-1])
        self._is_trained = True
        return self

    def _require_trained(self) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError("ANNModel has not been trained.")
