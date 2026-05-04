"""Streamlit UI for dataset ingestion, training, and inference."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.datasets.common_voice import (
    ArchiveExtractionError,
    build_common_voice_metadata,
    discover_common_voice_root,
    extract_archive,
    find_archives,
)
from src.inference.predict import InferenceEngine
from src.models.ann_model import ANNModel
from src.models.svm_model import SVMModel
from src.models.train import train_pipeline
from src.utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ACCENT_GROUPS = ["coastal", "nairobi", "upcountry"]


st.set_page_config(page_title="Kiswahili ASR", layout="wide")
st.title("Accent-Aware Kiswahili Speech Recognition")


def _load_svm_engine(model_path: Path, scaler_path: Path, encoder_path: Path) -> InferenceEngine:
    model = SVMModel().load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return InferenceEngine(model=model, scaler=scaler, label_encoder=label_encoder)


def _artifact_exists(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def _accent_readiness(path: Path) -> tuple[bool, set[str]]:
    if not path.exists():
        return False, set()
    frame = pd.read_csv(path)
    accents = set(frame.get("accent_label", pd.Series(dtype=str)).dropna().astype(str))
    return set(ACCENT_GROUPS).issubset(accents), accents


def _word_prompts(metadata_path: Path, encoder_path: Path) -> list[str]:
    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
            return sorted(str(label) for label in encoder.classes_)
        except Exception:
            pass
    if metadata_path.exists():
        frame = pd.read_csv(metadata_path)
        if "word_label" in frame.columns:
            return sorted(frame["word_label"].dropna().astype(str).unique())
    return []


def _pipeline_steps() -> None:
    cols = st.columns(5)
    steps = [
        ("1", "Prompt word", "Choose the word the speaker must pronounce."),
        ("2", "Speaker audio", "Record or upload the speaker pronouncing that word."),
        ("3", "MFCC features", "Convert audio into a 39-value acoustic feature vector."),
        ("4", "Word text", "Check whether the model heard the target word."),
        ("5", "Accent group", "Classify speaker accent as Coastal, Nairobi, or Upcountry."),
    ]
    for col, (num, title, detail) in zip(cols, steps):
        with col:
            st.metric(num, title)
            st.caption(detail)


def _metadata_summary(path: Path) -> None:
    if not path.exists():
        st.info("No metadata CSV has been generated yet.")
        return
    frame = pd.read_csv(path)
    if frame.empty:
        st.warning("Metadata CSV exists but has no rows.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Labels", f"{frame['word_label'].nunique():,}")
    c3.metric("Accents", f"{frame['accent_label'].nunique():,}")
    c4.metric("Speakers", f"{frame['speaker_id'].nunique():,}")

    left, right = st.columns(2)
    with left:
        st.caption("Split counts")
        st.dataframe(frame["split"].value_counts().rename("rows"), use_container_width=True)
    with right:
        st.caption("Top labels")
        st.dataframe(frame["word_label"].value_counts().head(20).rename("rows"), use_container_width=True)

    low_counts = frame["word_label"].value_counts()
    if (low_counts < 2).any():
        st.warning(
            "Some labels have only one example. Training may fail or generalize poorly "
            "until the dataset has repeated examples per label."
        )


demo_tab, dataset_tab, train_tab = st.tabs(["Demo", "Dataset", "Train"])

with demo_tab:
    st.subheader("Speaker Audio to Text and Accent Group")
    _pipeline_steps()

    word_model_path = MODELS_DIR / "svm_model.joblib"
    word_scaler_path = MODELS_DIR / "scaler.joblib"
    word_encoder_path = MODELS_DIR / "label_encoder.joblib"
    accent_model_path = MODELS_DIR / "accent_svm_model.joblib"
    accent_scaler_path = MODELS_DIR / "accent_scaler.joblib"
    accent_encoder_path = MODELS_DIR / "accent_label_encoder.joblib"

    word_ready = _artifact_exists(word_model_path, word_scaler_path, word_encoder_path)
    accent_ready = _artifact_exists(accent_model_path, accent_scaler_path, accent_encoder_path)
    accent_data_ready, known_accents = _accent_readiness(METADATA_CSV)

    status_cols = st.columns(3)
    status_cols[0].metric("Word recognizer", "Ready" if word_ready else "Missing")
    status_cols[1].metric("Accent classifier", "Ready" if accent_ready else "Missing")
    status_cols[2].metric("Accent groups in data", f"{len(known_accents)}")

    if not accent_data_ready:
        st.warning(
            "Accent classification into Coastal, Nairobi, and Upcountry requires training data "
            "with those exact accent labels. The current metadata does not contain all three groups."
        )

    prompts = _word_prompts(METADATA_CSV, word_encoder_path)
    if prompts:
        selected_word = st.selectbox("Word for speaker to pronounce", prompts)
    else:
        selected_word = st.text_input("Word for speaker to pronounce", value="")

    st.markdown(f"### Pronounce: `{selected_word or 'choose a word'}`")
    st.caption("Ask the speaker to say this word once, clearly, then run recognition.")

    recorded_audio = None
    if hasattr(st, "audio_input"):
        recorded_audio = st.audio_input("Record speaker audio")
    uploaded_audio = st.file_uploader(
        "Or upload speaker audio",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Use a short recording of the speaker pronouncing the prompt word.",
    )
    audio_file = recorded_audio or uploaded_audio

    if st.button("Run Recognition", type="primary") and audio_file is not None:
        suffix = Path(getattr(audio_file, "name", "recording.wav")).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_file.getbuffer())
            tmp_path = tmp.name

        if not word_ready:
            st.error("Word model artifacts are missing. Train the word recognizer first.")
        else:
            try:
                word_engine = _load_svm_engine(word_model_path, word_scaler_path, word_encoder_path)
                word_result = word_engine.predict_from_file(tmp_path)
            except Exception as exc:
                word_result = None
                st.error(f"Word recognition failed: {exc}")

            if word_result and word_result.is_error:
                st.error(word_result.error)
            elif word_result:
                is_match = (
                    selected_word.strip().lower() == word_result.predicted_word.strip().lower()
                    if selected_word
                    else False
                )
                result_cols = st.columns(4)
                result_cols[0].metric("Prompt word", selected_word or "Not selected")
                result_cols[1].metric("Recognized text", word_result.predicted_word)
                result_cols[2].metric("Word confidence", f"{word_result.confidence:.3f}")
                result_cols[3].metric("Prompt match", "Yes" if is_match else "No")
                st.caption("Word alternatives")
                st.dataframe(
                    pd.DataFrame(word_result.top_k, columns=["word", "probability"]),
                    use_container_width=True,
                )

        st.divider()
        st.subheader("Accent Group")
        if accent_ready:
            try:
                accent_engine = _load_svm_engine(accent_model_path, accent_scaler_path, accent_encoder_path)
                accent_result = accent_engine.predict_from_file(tmp_path)
                if accent_result.is_error:
                    st.error(accent_result.error)
                else:
                    st.metric("Speaker accent group", accent_result.predicted_word)
                    probabilities = pd.DataFrame(
                        accent_result.top_k,
                        columns=["accent_group", "probability"],
                    )
                    st.bar_chart(probabilities.set_index("accent_group"))
            except Exception as exc:
                st.error(f"Accent classification failed: {exc}")
        else:
            st.info(
                "No trained accent classifier is available yet. Add examples labelled "
                "`coastal`, `nairobi`, and `upcountry`, then train target `accent`."
            )

with dataset_tab:
    st.subheader("Dataset Extraction")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archives = find_archives(RAW_DIR)
    archive_names = [archive.name for archive in archives]

    if archives:
        selected = st.selectbox("Archive", archive_names)
        archive_path = archives[archive_names.index(selected)]
        st.caption(str(archive_path))
        overwrite = st.checkbox("Overwrite existing extracted files", value=False)
        if st.button("Extract Archive", type="primary"):
            try:
                with st.spinner("Extracting dataset archive..."):
                    result = extract_archive(archive_path, RAW_DIR, overwrite=overwrite)
                st.success(f"Extracted {result.extracted_files:,} files.")
                if result.common_voice_root:
                    st.info(f"Common Voice root: {result.common_voice_root}")
            except ArchiveExtractionError as exc:
                st.error(str(exc))
    else:
        st.info("Place a Common Voice .tar, .tar.gz, .tgz, or .zip archive in data/raw.")

    st.divider()
    st.subheader("Metadata Generation")
    cv_root = discover_common_voice_root(RAW_DIR)
    cv_root_text = st.text_input("Common Voice root", value=str(cv_root or RAW_DIR))
    label_strategy = st.selectbox("Label strategy", ["sentence", "first_word"], index=0)
    single_word_only = st.checkbox("Keep single-word prompts only", value=False)
    min_label_count = st.number_input("Minimum examples per label", min_value=1, value=2, step=1)
    max_labels = st.number_input("Maximum labels", min_value=0, value=25, step=5)
    max_rows = st.number_input("Maximum rows", min_value=0, value=0, step=100)

    if st.button("Build Metadata CSV"):
        try:
            with st.spinner("Building metadata.csv..."):
                result = build_common_voice_metadata(
                    common_voice_root=cv_root_text,
                    output_csv=METADATA_CSV,
                    data_dir=RAW_DIR,
                    label_strategy=label_strategy,
                    max_rows=int(max_rows) or None,
                    single_word_only=single_word_only,
                    min_label_count=int(min_label_count),
                    max_labels=int(max_labels) or None,
                )
            st.success(
                f"Wrote {result.rows:,} rows, {result.labels:,} labels, "
                f"{result.accents:,} accent groups to {result.output_csv}."
            )
            if result.skipped_rows:
                st.warning(f"Skipped {result.skipped_rows:,} rows without usable audio or labels.")
        except Exception as exc:
            st.error(str(exc))

    _metadata_summary(METADATA_CSV)

with train_tab:
    st.subheader("Model Training")
    _metadata_summary(METADATA_CSV)
    target = st.selectbox("Target", ["word", "accent"], index=0)
    if target == "accent":
        ready, accents = _accent_readiness(METADATA_CSV)
        if ready:
            st.success("Metadata contains Coastal, Nairobi, and Upcountry labels.")
        else:
            st.warning(
                "Accent training needs all three labels: coastal, nairobi, upcountry. "
                f"Current labels: {', '.join(sorted(accents)) if accents else 'none'}"
            )
    model_type = st.selectbox("Model", ["svm", "ann"], index=0)
    config_path = st.text_input("Config path", value=str(PROJECT_ROOT / "config" / "default.yaml"))
    save_dir = st.text_input("Model output directory", value=str(MODELS_DIR))

    if st.button("Train Model", type="primary"):
        try:
            config = Config.load(config_path)
            with st.spinner("Training model..."):
                _, report = train_pipeline(
                    metadata_csv=str(METADATA_CSV),
                    data_dir=str(RAW_DIR),
                    model_type=model_type,
                    config=config,
                    save_dir=save_dir,
                    target_column="accent_label" if target == "accent" else "word_label",
                )
            st.success("Training complete.")
            metrics = {
                key: report[key]
                for key in ("accuracy", "precision", "recall", "f1", "wer")
            }
            st.json(metrics)
            st.caption("Per-accent accuracy")
            st.json(report["per_accent"])
        except Exception as exc:
            st.error(str(exc))
