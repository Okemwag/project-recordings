"""
config.py — Configuration loader for the Kiswahili ASR system.

Loads YAML config files and exposes settings as a typed dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class PreprocessingConfig:
    target_sr: int = 16000
    top_db: float = 20.0
    min_duration: float = 0.5
    max_duration: float = 3.0


@dataclass
class FeatureConfig:
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 40
    use_delta: bool = True
    use_delta_delta: bool = True


@dataclass
class ModelConfig:
    type: str = "svm"          # "svm" or "ann"
    random_state: int = 42
    # SVM
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    # ANN
    ann_hidden_units: List[int] = field(default_factory=lambda: [256, 128])
    ann_dropout_rate: float = 0.3
    ann_learning_rate: float = 0.001
    ann_epochs: int = 100
    ann_batch_size: int = 32


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    accents: List[str] = field(
        default_factory=lambda: ["coastal", "nairobi", "upcountry"]
    )


@dataclass
class Config:
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_csv: str = "data/metadata.csv"
    models_dir: str = "models"

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()

        if "preprocessing" in raw:
            cfg.preprocessing = PreprocessingConfig(**raw["preprocessing"])
        if "features" in raw:
            cfg.features = FeatureConfig(**raw["features"])
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "training" in raw:
            cfg.training = TrainingConfig(**raw["training"])

        for key in ("data_dir", "processed_dir", "metadata_csv", "models_dir"):
            if key in raw:
                setattr(cfg, key, raw[key])

        return cfg

    @classmethod
    def default(cls) -> "Config":
        """Return a default configuration instance."""
        return cls()
