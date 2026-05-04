"""End-to-end training pipeline for isolated-word Kiswahili ASR."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.evaluation.metrics import Evaluator
from src.features.mfcc_extraction import FeatureExtractor
from src.preprocessing.noise_reduction import NoiseReducer
from src.preprocessing.normalization import AudioNormalizer
from src.preprocessing.silence_removal import SilenceRemover
from src.utils.audio import load_audio
from src.utils.config import Config

logger = logging.getLogger(__name__)


REQUIRED_METADATA_COLUMNS = {
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
}


def train_pipeline(
    metadata_csv: str,
    data_dir: str,
    model_type: str = "svm",
    config: Config | None = None,
    save_dir: str = "models",
    target_column: str = "word_label",
):
    """Train a model, save artifacts, and return the fitted model plus report."""
    if target_column not in {"word_label", "accent_label"}:
        raise ValueError("target_column must be 'word_label' or 'accent_label'.")

    config = config or Config.default()
    metadata = _load_metadata(metadata_csv)
    _warn_on_accent_imbalance(metadata["accent_label"].to_numpy())

    X, labels, accents, splits = _build_feature_matrix(metadata, data_dir, config)
    if target_column == "accent_label":
        labels = accents

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_idx, test_idx = _split_indices(splits, y, config)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_accents = accents[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = _build_model(model_type, X.shape[1], len(label_encoder.classes_), config)
    if model_type == "ann":
        model.train(
            X_train_scaled,
            y_train,
            epochs=config.model.ann_epochs,
            batch_size=config.model.ann_batch_size,
            validation_split=0.1,
        )
    else:
        model.train(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = Evaluator(label_encoder).full_report(y_test, y_pred, test_accents)

    artifact_prefix = "accent_" if target_column == "accent_label" else ""
    _save_artifacts(model, scaler, label_encoder, save_dir, model_type, artifact_prefix)
    return model, report


def _load_metadata(metadata_csv: str) -> pd.DataFrame:
    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    metadata = pd.read_csv(path)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Metadata CSV does not contain any training rows.")
    return metadata


def _build_feature_matrix(
    metadata: pd.DataFrame,
    data_dir: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_cfg = config.features
    pre_cfg = config.preprocessing
    extractor = FeatureExtractor(
        n_mfcc=feature_cfg.n_mfcc,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        n_mels=feature_cfg.n_mels,
        sr=pre_cfg.target_sr,
    )
    normalizer = AudioNormalizer(target_sr=pre_cfg.target_sr)
    reducer = NoiseReducer()
    trimmer = SilenceRemover(
        top_db=pre_cfg.top_db,
        min_duration=pre_cfg.min_duration,
        max_duration=pre_cfg.max_duration,
    )

    features: list[np.ndarray] = []
    labels: list[str] = []
    accents: list[str] = []
    splits: list[str] = []
    skipped: list[str] = []

    for row in metadata.itertuples(index=False):
        audio_path = _resolve_audio_path(str(row.file_path), data_dir)
        try:
            audio, orig_sr = load_audio(audio_path)
            audio = normalizer.resample(audio, orig_sr)
            audio = reducer.reduce(audio, sr=pre_cfg.target_sr)
            audio = trimmer.trim(audio, sr=pre_cfg.target_sr)
            audio = normalizer.normalize_amplitude(audio)
            features.append(extractor.extract(audio))
            labels.append(str(row.word_label))
            accents.append(str(row.accent_label))
            splits.append(str(row.split).lower())
        except Exception as exc:
            skipped.append(f"{audio_path}: {exc}")

    if skipped:
        logger.warning("Skipped %d metadata rows that could not be processed.", len(skipped))
        for reason in skipped[:5]:
            logger.warning("Skipped row: %s", reason)
    if not features:
        raise ValueError("No usable audio rows were found for training.")

    return (
        np.vstack(features),
        np.asarray(labels),
        np.asarray(accents),
        np.asarray(splits),
    )


def _resolve_audio_path(file_path: str, data_dir: str) -> Path:
    path = Path(file_path)
    if path.is_absolute() or path.exists():
        return path
    data_root_path = Path(data_dir) / file_path
    if data_root_path.exists():
        return data_root_path
    return path


def _split_indices(
    splits: np.ndarray,
    y: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    splits = np.char.lower(splits.astype(str))
    if {"train", "test"}.issubset(set(splits)):
        train_idx = np.flatnonzero(splits == "train")
        test_idx = np.flatnonzero(splits == "test")
        return train_idx, test_idx

    indices = np.arange(len(y))
    stratify = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    return train_test_split(
        indices,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=stratify,
    )


def _build_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    config: Config,
):
    if model_type == "svm":
        from src.models.svm_model import SVMModel

        return SVMModel(
            kernel=config.model.svm_kernel,
            C=config.model.svm_C,
            gamma=config.model.svm_gamma,
            random_state=config.model.random_state,
        )
    if model_type == "ann":
        from src.models.ann_model import ANNModel

        return ANNModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_units=config.model.ann_hidden_units,
            dropout_rate=config.model.ann_dropout_rate,
            learning_rate=config.model.ann_learning_rate,
            random_state=config.model.random_state,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _save_artifacts(
    model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    save_dir: str,
    model_type: str,
    artifact_prefix: str = "",
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_suffix = "keras" if model_type == "ann" else "joblib"
    model.save(save_path / f"{artifact_prefix}{model_type}_model.{model_suffix}")
    joblib.dump(scaler, save_path / f"{artifact_prefix}scaler.joblib")
    joblib.dump(label_encoder, save_path / f"{artifact_prefix}label_encoder.joblib")


def _warn_on_accent_imbalance(accent_labels: np.ndarray) -> None:
    unique, counts = np.unique(accent_labels, return_counts=True)
    mean = counts.mean()
    if mean == 0:
        return
    for accent, count in zip(unique, counts):
        if abs(count - mean) / mean > 0.10:
            warnings.warn(
                f"Accent group '{accent}' count {count} differs from the mean "
                f"{mean:.1f} by more than 10%.",
                RuntimeWarning,
                stacklevel=2,
            )
