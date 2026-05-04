"""
main.py — CLI entry point for the Kiswahili ASR system.

Usage:
    # Train SVM model
    python main.py train --model svm

    # Train ANN model
    python main.py train --model ann

    # Predict from audio file
    python main.py predict --file path/to/audio.wav --model-path models/svm_model.joblib

    # Predict from microphone
    python main.py predict --mic --model-path models/svm_model.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    from src.models.train import train_pipeline
    from src.utils.config import Config

    config = Config.load(args.config) if args.config else Config.default()

    logger.info("Starting training pipeline (model=%s)...", args.model)
    target_column = "accent_label" if args.target == "accent" else "word_label"
    model, report = train_pipeline(
        metadata_csv=args.metadata,
        data_dir=args.data_dir,
        model_type=args.model,
        config=config,
        save_dir=args.save_dir,
        target_column=target_column,
    )

    print("\n── Evaluation Report ──────────────────────────────────────")
    print(f"  Accuracy  : {report['accuracy']:.4f}")
    print(f"  Precision : {report['precision']:.4f}")
    print(f"  Recall    : {report['recall']:.4f}")
    print(f"  F1        : {report['f1']:.4f}")
    print(f"  WER       : {report['wer']:.4f}")
    print("\n  Per-Accent Accuracy:")
    for accent, acc in report["per_accent"].items():
        print(f"    {accent:12s}: {acc:.4f}")
    print("\n" + report["classification_report"])


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference on a file or microphone input."""
    import joblib
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    from src.inference.predict import InferenceEngine
    from src.utils.config import Config

    config = Config.load(args.config) if args.config else Config.default()

    # Load model, scaler, and label encoder
    logger.info("Loading model from %s...", args.model_path)
    model_path = args.model_path
    scaler_path = args.scaler_path
    encoder_path = args.encoder_path

    if args.model_type == "svm":
        from src.models.svm_model import SVMModel
        model = SVMModel()
        model.load(model_path)
    else:
        from src.models.ann_model import ANNModel
        model = ANNModel(input_dim=39, num_classes=1)  # num_classes overridden by load
        model.load(model_path)

    scaler: StandardScaler = joblib.load(scaler_path)
    label_encoder: LabelEncoder = joblib.load(encoder_path)

    engine = InferenceEngine(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
    )

    if args.file:
        logger.info("Predicting from file: %s", args.file)
        result = engine.predict_from_file(args.file)
    elif args.mic:
        logger.info("Recording from microphone...")
        result = engine.predict_from_mic(duration=args.duration)
    else:
        logger.error("Specify --file or --mic for prediction.")
        sys.exit(1)

    if result.is_error:
        print(f"Error: {result.error}")
    else:
        print(f"\nPredicted word : {result.predicted_word}")
        print(f"Confidence     : {result.confidence:.4f}")
        print("\nTop predictions:")
        for word, prob in result.top_k:
            print(f"  {word:15s}: {prob:.4f}")


def cmd_dataset(args: argparse.Namespace) -> None:
    """Run dataset extraction or metadata generation."""
    from pathlib import Path

    from src.datasets.common_voice import (
        ArchiveExtractionError,
        build_common_voice_metadata,
        discover_common_voice_root,
        extract_archive,
        find_archives,
    )

    if args.dataset_command == "archives":
        archives = find_archives(args.raw_dir)
        if not archives:
            print(f"No supported archives found in {args.raw_dir}")
            return
        for archive in archives:
            print(archive)
        return

    if args.dataset_command == "extract":
        archive = args.archive
        if archive is None:
            archives = find_archives(args.raw_dir)
            if not archives:
                raise FileNotFoundError(f"No supported archives found in {args.raw_dir}")
            archive = str(archives[0])
        try:
            result = extract_archive(archive, destination=args.raw_dir, overwrite=args.overwrite)
        except ArchiveExtractionError as exc:
            print(f"Dataset extraction failed: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Extracted files : {result.extracted_files}")
        print(f"Destination     : {result.destination}")
        print(f"CommonVoice root: {result.common_voice_root or 'not found'}")
        return

    if args.dataset_command == "metadata":
        cv_root = (
            Path(args.common_voice_root)
            if args.common_voice_root
            else discover_common_voice_root(args.raw_dir)
        )
        if cv_root is None:
            raise FileNotFoundError(
                f"Could not find a Common Voice root under {args.raw_dir}. "
                "Run dataset extract first or pass --common-voice-root."
            )
        result = build_common_voice_metadata(
            common_voice_root=cv_root,
            output_csv=args.output,
            data_dir=args.raw_dir,
            label_strategy=args.label_strategy,
            max_rows=args.max_rows,
            single_word_only=args.single_word_only,
            min_label_count=args.min_label_count,
            max_labels=args.max_labels,
        )
        print(f"Metadata CSV : {result.output_csv}")
        print(f"Rows         : {result.rows}")
        print(f"Labels       : {result.labels}")
        print(f"Accents      : {result.accents}")
        print(f"Splits       : {result.splits}")
        print(f"Skipped rows : {result.skipped_rows}")
        return

    raise ValueError(f"Unsupported dataset command: {args.dataset_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiswahili Accent-Aware Speech Recognition System"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", choices=["svm", "ann"], default="svm",
        help="Model type to train (default: svm)"
    )
    train_parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root directory for raw audio files"
    )
    train_parser.add_argument(
        "--metadata", default="data/metadata.csv",
        help="Path to metadata CSV file"
    )
    train_parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file"
    )
    train_parser.add_argument(
        "--save-dir", default="models",
        help="Directory to save trained model"
    )
    train_parser.add_argument(
        "--target",
        choices=["word", "accent"],
        default="word",
        help="Train word recognition or accent classification artifacts",
    )

    # ── predict ────────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--file", default=None, help="Path to audio file")
    predict_parser.add_argument("--mic", action="store_true", help="Use microphone")
    predict_parser.add_argument(
        "--duration", type=float, default=2.0,
        help="Microphone recording duration in seconds"
    )
    predict_parser.add_argument(
        "--model-path", required=True, help="Path to saved model"
    )
    predict_parser.add_argument(
        "--scaler-path", required=True, help="Path to saved StandardScaler"
    )
    predict_parser.add_argument(
        "--encoder-path", required=True, help="Path to saved LabelEncoder"
    )
    predict_parser.add_argument(
        "--model-type", choices=["svm", "ann"], default="svm",
        help="Model type (must match saved model)"
    )
    predict_parser.add_argument(
        "--config", default=None, help="Path to YAML config file"
    )

    # ── dataset ────────────────────────────────────────────────────────────
    dataset_parser = subparsers.add_parser("dataset", help="Manage datasets")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    archives_parser = dataset_subparsers.add_parser("archives", help="List raw archives")
    archives_parser.add_argument("--raw-dir", default="data/raw")

    extract_parser = dataset_subparsers.add_parser("extract", help="Extract a dataset archive")
    extract_parser.add_argument(
        "--archive",
        default=None,
        help="Archive path; defaults to first archive in --raw-dir",
    )
    extract_parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files",
    )

    metadata_parser = dataset_subparsers.add_parser(
        "metadata",
        help="Build metadata.csv from Common Voice TSVs",
    )
    metadata_parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    metadata_parser.add_argument(
        "--common-voice-root",
        default=None,
        help="Extracted Common Voice language root",
    )
    metadata_parser.add_argument(
        "--output",
        default="data/metadata.csv",
        help="Metadata CSV output path",
    )
    metadata_parser.add_argument(
        "--label-strategy",
        choices=["sentence", "first_word"],
        default="sentence",
    )
    metadata_parser.add_argument("--single-word-only", action="store_true")
    metadata_parser.add_argument("--max-rows", type=int, default=None)
    metadata_parser.add_argument("--min-label-count", type=int, default=1)
    metadata_parser.add_argument("--max-labels", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "dataset":
        cmd_dataset(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
