"""End-to-end tests for the training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf

from src.models.svm_model import SVMModel
from src.models.train import train_pipeline
from src.utils.config import Config


def _write_sine(path, freq: float, duration: float = 0.7, sr: int = 16000) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr)


def test_train_pipeline_svm_saves_artifacts(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = []
    samples = [
        ("maji", "coastal", 440.0, "train"),
        ("maji", "nairobi", 450.0, "train"),
        ("maji", "upcountry", 460.0, "test"),
        ("chakula", "coastal", 880.0, "train"),
        ("chakula", "nairobi", 890.0, "train"),
        ("chakula", "upcountry", 900.0, "test"),
    ]
    for idx, (word, accent, freq, split) in enumerate(samples):
        path = audio_dir / f"{idx}_{word}_{accent}.wav"
        _write_sine(path, freq)
        rows.append(
            {
                "file_path": str(path),
                "word_label": word,
                "accent_label": accent,
                "speaker_id": f"speaker_{idx}",
                "duration_sec": 0.7,
                "split": split,
            }
        )

    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)

    config = Config.default()
    config.preprocessing.min_duration = 0.2
    config.preprocessing.max_duration = 1.0

    save_dir = tmp_path / "models"
    model, report = train_pipeline(
        metadata_csv=str(metadata_csv),
        data_dir=str(audio_dir),
        model_type="svm",
        config=config,
        save_dir=str(save_dir),
    )

    assert isinstance(model, SVMModel)
    assert {"accuracy", "precision", "recall", "f1", "wer"}.issubset(report)
    assert (save_dir / "svm_model.joblib").exists()
    assert (save_dir / "scaler.joblib").exists()
    assert (save_dir / "label_encoder.joblib").exists()
