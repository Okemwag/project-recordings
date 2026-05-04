# Accent-Aware Kiswahili Speech Recognition System

An isolated-word Kiswahili speech recognizer that handles three regional accent groups:
**Coastal**, **Nairobi**, and **Upcountry**.

Uses MFCC feature extraction (+ delta and delta-delta) with classical ML models (SVM and ANN).
Designed for CPU-only training on small datasets (~100–150 speakers, 80–100 words).

---

## Project Structure

```
speech_recognition_project/
├── config/
│   └── default.yaml          # System configuration
├── data/
│   ├── raw/                  # Raw WAV audio files (organized by accent)
│   ├── processed/            # Preprocessed audio cache
│   └── metadata.csv          # Dataset manifest
├── models/                   # Saved trained models
├── notebooks/
│   └── experiments.ipynb     # Exploratory analysis notebook
├── src/
│   ├── preprocessing/
│   │   ├── noise_reduction.py
│   │   ├── silence_removal.py
│   │   └── normalization.py
│   ├── features/
│   │   └── mfcc_extraction.py
│   ├── models/
│   │   ├── svm_model.py
│   │   ├── ann_model.py
│   │   └── train.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       └── config.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_inference.py
├── main.py
└── requirements.txt
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation

1. Place WAV files (16kHz, mono) in `data/raw/` organized by accent:
   ```
   data/raw/
   ├── coastal/
   ├── nairobi/
   └── upcountry/
   ```

2. Populate `data/metadata.csv` with columns:
   ```
   file_path, word_label, accent_label, speaker_id, duration_sec, split
   ```

### Common Voice Archives

This project can extract Mozilla Common Voice archives and build `data/metadata.csv`:

```bash
python main.py dataset archives --raw-dir data/raw
python main.py dataset extract --raw-dir data/raw
python main.py dataset metadata --raw-dir data/raw --output data/metadata.csv
```

For Common Voice sentence-level recognition, keep the default label strategy. For a smaller vocabulary, use:

```bash
python main.py dataset metadata --label-strategy first_word --min-label-count 20 --max-labels 25
```

If extraction reports `unexpected end of data`, the archive is incomplete and must be replaced with a full download before metadata generation or training can proceed.

## Streamlit UI

Run the local workbench:

```bash
streamlit run streamlit_app.py
```

The UI supports archive extraction, metadata generation, SVM/ANN training, and audio-file prediction.

The Demo tab shows the project flow:

```text
speaker audio -> preprocessing -> MFCC features -> word text -> accent group
```

Word recognition uses `models/svm_model.joblib`, `models/scaler.joblib`, and
`models/label_encoder.joblib`. Accent classification uses separate artifacts:
`models/accent_svm_model.joblib`, `models/accent_scaler.joblib`, and
`models/accent_label_encoder.joblib`.

To train the accent classifier, `data/metadata.csv` must contain examples with
these exact labels in the `accent_label` column:

```text
coastal
nairobi
upcountry
```

Then run:

```bash
python main.py train --target accent --model svm --data-dir data/raw --metadata data/metadata.csv --save-dir models
```

---

## Training

```bash
# Train SVM (recommended for small datasets)
python main.py train --model svm --data-dir data/raw --metadata data/metadata.csv

# Train ANN
python main.py train --model ann --save-dir models/
```

---

## Inference

```bash
# Predict from audio file
python main.py predict \
  --file data/raw/coastal/speaker_01_maji.wav \
  --model-path models/svm_model.joblib \
  --scaler-path models/scaler.joblib \
  --encoder-path models/label_encoder.joblib

# Predict from microphone (2 seconds)
python main.py predict \
  --mic --duration 2.0 \
  --model-path models/svm_model.joblib \
  --scaler-path models/scaler.joblib \
  --encoder-path models/label_encoder.joblib
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

Edit `config/default.yaml` to adjust:
- Audio preprocessing parameters (sample rate, silence threshold, duration limits)
- MFCC feature settings (n_mfcc, n_fft, hop_length)
- Model hyperparameters (SVM kernel/C/gamma, ANN architecture)
- Train/test split ratio and random seed

---

## Guardrails

- System is an **isolated-word recognizer** — not conversational ASR
- All random seeds fixed at 42 for reproducibility
- Test set is never used during training or normalization fitting
- Equal representation of all three accent groups is enforced
- All voice recordings require explicit speaker consent

---

## Supported Accents

| Accent | Description |
|---|---|
| `coastal` | Coastal Kiswahili (Mombasa region) |
| `nairobi` | Nairobi urban Kiswahili |
| `upcountry` | Upcountry / inland Kiswahili |
