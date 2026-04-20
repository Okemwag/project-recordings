# Accent-Aware Kiswahili Speech Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![Platform](https://img.shields.io/badge/Platform-CPU%20Only-lightgrey)

An isolated-word Kiswahili speech recognition system that accounts for regional accent variation across three Kenyan accent groups — **Coastal**, **Nairobi**, and **Upcountry**. The system uses MFCC-based feature extraction (with delta and delta-delta coefficients) fed into classical ML models (SVM and ANN), making it accessible for research on modest hardware without GPU infrastructure.

> **Scope**: This is an *isolated-word* recognizer. It classifies single spoken Kiswahili words. It is **not** a conversational or continuous speech ASR system.

---

## Table of Contents

1. [Background](#background)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Supported Accents](#supported-accents)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Data Preparation](#data-preparation)
8. [Configuration](#configuration)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Inference](#inference)
12. [Testing](#testing)
13. [Known Limitations](#known-limitations)
14. [Contributing](#contributing)
15. [License](#license)

---

## Background

Kiswahili is spoken by over 200 million people across East and Central Africa, yet it remains underrepresented in speech technology research. A key challenge is **accent variability**: speakers from the Kenyan coast, Nairobi, and upcountry regions pronounce the same words with measurably different phonetic patterns.

This project addresses that gap by:

- Building a balanced, multi-accent dataset of isolated Kiswahili words
- Extracting robust MFCC + Δ + ΔΔ features that capture spectral and temporal dynamics
- Training and comparing SVM (RBF kernel) and ANN (feedforward) classifiers
- Reporting per-accent accuracy breakdowns to surface model bias

The system is designed for telecom-adjacent use cases such as Interactive Voice Response (IVR) systems deployed in East Africa.

---

## System Architecture

```
Audio Input (WAV file or microphone)
         │
         ▼
┌─────────────────────────────┐
│     Preprocessing Module    │
│  • Resample → 16 kHz        │
│  • Spectral noise reduction │
│  • Silence trimming         │
│  • Amplitude normalization  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Feature Extraction Module │
│  • MFCC  (13 coefficients)  │
│  • Delta Δ  (13 coeff.)     │
│  • Delta-Delta ΔΔ (13 coeff)│
│  • Mean aggregation → (39,) │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Feature Normalization     │
│  • StandardScaler           │
│  • Fit on train only        │
└────────────┬────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────┐ ┌──────────┐
│   SVM    │ │   ANN    │
│ RBF kern │ │ Dense    │
│ sklearn  │ │ + Softmax│
└────┬─────┘ └────┬─────┘
     └──────┬─────┘
            ▼
┌─────────────────────────────┐
│     Evaluation Layer        │
│  • Accuracy / Precision     │
│  • Recall / F1 / WER        │
│  • Confusion Matrix         │
│  • Per-accent breakdown     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│     Inference Engine        │
│  • File or microphone input │
│  • Predicted word + top-K   │
│  • Confidence score         │
└─────────────────────────────┘
```

### Feature Vector

Each audio sample is reduced to a **39-dimensional feature vector**:

| Component     | Coefficients | Description                              |
|---------------|:------------:|------------------------------------------|
| MFCC          | 13           | Mel-frequency cepstral coefficients      |
| Delta (Δ)     | 13           | First-order temporal derivative of MFCC  |
| Delta-Delta (ΔΔ) | 13        | Second-order temporal derivative of MFCC |
| **Total**     | **39**       | Mean across all time frames              |

---

## Project Structure

```
kiswahili-asr/
├── data/
│   ├── raw/                    # Raw WAV files (gitignored — see Data Preparation)
│   ├── processed/              # Preprocessed audio cache (gitignored)
│   └── metadata.csv            # Dataset manifest (tracked in git)
│
├── notebooks/
│   └── experiments.ipynb       # Exploratory analysis and visualizations
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── noise_reduction.py  # Spectral noise reduction (noisereduce)
│   │   ├── silence_removal.py  # Silence trimming with duration guards
│   │   └── normalization.py    # Amplitude normalization and resampling
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── mfcc_extraction.py  # MFCC + Δ + ΔΔ feature extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svm_model.py        # SVM classifier (sklearn, RBF kernel)
│   │   ├── ann_model.py        # ANN classifier (TensorFlow/Keras)
│   │   └── train.py            # End-to-end training pipeline
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Accuracy, WER, confusion matrix, per-accent
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py          # InferenceEngine (file + microphone)
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py           # YAML config loader and defaults
│
├── config/
│   └── default.yaml            # All tunable hyperparameters
│
├── models/                     # Saved model artifacts (gitignored)
│   └── .gitkeep
│
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic audio, labels)
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_inference.py
│
├── .gitignore
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## Supported Accents

| Accent Label | Region | Characteristics |
|---|---|---|
| `coastal` | Kenyan Coast (Mombasa, Malindi, Lamu) | Swahili as a first language; softer consonants, distinct vowel length |
| `nairobi` | Nairobi metropolitan area | Urban Swahili influenced by Sheng and English; faster speech rate |
| `upcountry` | Central, Rift Valley, Western Kenya | Swahili as a second language; influenced by Kikuyu, Kalenjin, Luo phonology |

The training pipeline enforces balanced representation across all three accent groups. A dataset imbalance warning is raised if any accent group deviates more than 10% from the mean.

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 4 GB | 8 GB |
| Disk | 2 GB (code + models) | 10 GB (with audio data) |
| CPU | Any x86-64 | Multi-core (speeds up SVM training) |
| GPU | Not required | Not required |

### System Dependencies

**PortAudio** is required for microphone input via `pyaudio`. Install it before running `pip install`:

```bash
# Ubuntu / Debian
sudo apt-get install portaudio19-dev python3-dev

# macOS (Homebrew)
brew install portaudio

# Windows
# pyaudio ships with pre-built wheels on Windows — no extra step needed
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/kiswahili-asr.git
cd kiswahili-asr
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import librosa, sklearn, tensorflow; print('All dependencies OK')"
```

---

## Data Preparation

### Directory Layout

Place raw audio files under `data/raw/` organized by accent and speaker:

```
data/raw/
├── coastal/
│   ├── speaker_C001/
│   │   ├── maji_01.wav
│   │   ├── chakula_01.wav
│   │   └── ...
│   └── speaker_C002/
│       └── ...
├── nairobi/
│   └── speaker_N001/
│       └── ...
└── upcountry/
    └── speaker_U001/
        └── ...
```

### Audio Format Requirements

| Property | Required Value |
|---|---|
| Format | WAV (PCM) |
| Sample rate | Any (resampled to 16 kHz automatically) |
| Channels | Mono (stereo is downmixed automatically) |
| Bit depth | 16-bit recommended |
| Duration | 0.5 s – 3.0 s per word |

### metadata.csv

Every audio file must have a corresponding row in `data/metadata.csv`. This file is the single source of truth for the dataset.

```csv
file_path,word_label,accent_label,speaker_id,duration_sec,split
coastal/speaker_C001/maji_01.wav,maji,coastal,C001,1.2,train
coastal/speaker_C001/chakula_01.wav,chakula,coastal,C001,1.5,train
nairobi/speaker_N001/maji_01.wav,maji,nairobi,N001,0.9,train
upcountry/speaker_U001/maji_01.wav,maji,upcountry,U001,1.1,test
```

| Column | Type | Description |
|---|---|---|
| `file_path` | `str` | Path relative to `data/raw/` |
| `word_label` | `str` | Target Kiswahili word (e.g., `maji`, `chakula`) |
| `accent_label` | `str` | One of: `coastal`, `nairobi`, `upcountry` |
| `speaker_id` | `str` | Unique speaker identifier |
| `duration_sec` | `float` | Audio duration in seconds |
| `split` | `str` | `train` or `test` |

> **Tip**: If you don't have a pre-split dataset, omit the `split` column and the training pipeline will perform a stratified 80/20 split automatically.

### Data Scale Guidelines

| Scale | Speakers | Words | Samples | Notes |
|---|---|---|---|---|
| Minimum viable | 30 | 20 | ~1,800 | Proof of concept only |
| Recommended | 100–150 | 80–100 | ~40,000 | Good generalization |
| Production | 500+ | 100+ | 150,000+ | Requires data augmentation |

---

## Configuration

All hyperparameters are centralized in `config/default.yaml`. You can override any value without touching source code.

```yaml
# config/default.yaml

preprocessing:
  target_sr: 16000          # Target sample rate (Hz)
  top_db: 20                # Silence threshold for trimming (dB below peak)
  min_duration: 0.5         # Minimum audio duration after trimming (seconds)
  max_duration: 3.0         # Maximum audio duration — longer clips are truncated
  target_peak: 0.95         # Amplitude normalization target peak

features:
  n_mfcc: 13                # Number of MFCC coefficients
  n_fft: 512                # FFT window size
  hop_length: 160           # Hop length between frames (10 ms at 16 kHz)
  n_mels: 40                # Number of Mel filter banks

svm:
  kernel: rbf               # SVM kernel type
  C: 1.0                    # Regularization parameter
  gamma: scale              # Kernel coefficient

ann:
  hidden_units: [256, 128]  # Neurons per hidden layer
  dropout_rate: 0.3         # Dropout regularization rate
  learning_rate: 0.001      # Adam optimizer learning rate
  epochs: 100               # Maximum training epochs
  batch_size: 32            # Mini-batch size
  early_stopping_patience: 10  # Epochs without improvement before stopping

training:
  test_size: 0.2            # Fraction of data held out for testing
  random_state: 42          # Seed for reproducibility
  save_dir: models/         # Directory to save trained models

inference:
  top_k: 5                  # Number of top predictions to return
  silence_threshold_db: 40  # RMS threshold for silent input detection
  mic_duration: 2.0         # Microphone recording duration (seconds)
```

To use a custom config:

```bash
python main.py train --model svm --config config/my_experiment.yaml
```

---

## Training

### Train an SVM model

```bash
python main.py train \
  --model svm \
  --data-dir data/raw \
  --metadata data/metadata.csv \
  --config config/default.yaml \
  --save-dir models/
```

### Train an ANN model

```bash
python main.py train \
  --model ann \
  --data-dir data/raw \
  --metadata data/metadata.csv \
  --config config/default.yaml \
  --save-dir models/
```

### Train both models and compare

```bash
python main.py train --model svm --save-dir models/svm/
python main.py train --model ann --save-dir models/ann/
```

### Training Output

A successful training run prints:

```
Loading dataset...  1,200 samples across 3 accents
Preprocessing...    1,187 valid samples (13 skipped)
Extracting features...  done in 42.3s
Normalizing features... done

Training SVM (RBF, C=1.0)...
  Train accuracy: 0.9312
  Test  accuracy: 0.8741

Evaluation Report
─────────────────────────────────────────
  Accuracy  : 0.8741
  Precision : 0.8698
  Recall    : 0.8741
  F1 Score  : 0.8712
  WER       : 0.1259

Per-Accent Accuracy
  coastal    : 0.8523
  nairobi    : 0.8934
  upcountry  : 0.8766

Model saved → models/svm_model.joblib
Scaler saved → models/scaler.joblib
Encoder saved → models/label_encoder.joblib
```

---

## Evaluation

Evaluation runs automatically at the end of training. To re-evaluate a saved model on a test set:

```bash
python main.py evaluate \
  --model-path models/svm_model.joblib \
  --scaler-path models/scaler.joblib \
  --encoder-path models/label_encoder.joblib \
  --metadata data/metadata.csv \
  --data-dir data/raw \
  --split test
```

### Metrics Explained

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correctly classified words |
| **Precision** | Weighted average precision across all word classes |
| **Recall** | Weighted average recall across all word classes |
| **F1 Score** | Harmonic mean of precision and recall |
| **WER** | Word Error Rate — for isolated-word tasks, `WER = 1 - Accuracy` |
| **Confusion Matrix** | N×N matrix showing predicted vs. true word labels |
| **Per-Accent Accuracy** | Accuracy broken down by `coastal`, `nairobi`, `upcountry` |

---

## Inference

### Predict from an audio file

```bash
python main.py predict \
  --file data/raw/coastal/speaker_C001/maji_01.wav \
  --model-path models/svm_model.joblib \
  --scaler-path models/scaler.joblib \
  --encoder-path models/label_encoder.joblib
```

**Output:**

```
Predicted word : maji
Confidence     : 0.9312

Top-5 predictions:
  1. maji      0.9312
  2. maziwa    0.0421
  3. moto      0.0187
  4. mwili     0.0063
  5. meza      0.0017
```

### Predict from microphone

```bash
python main.py predict \
  --mic \
  --model-path models/svm_model.joblib \
  --scaler-path models/scaler.joblib \
  --encoder-path models/label_encoder.joblib
```

The system records for 2 seconds (configurable via `inference.mic_duration` in `config/default.yaml`), then prints the prediction.

### Programmatic usage

```python
from src.inference.predict import InferenceEngine

engine = InferenceEngine.from_saved(
    model_path="models/svm_model.joblib",
    scaler_path="models/scaler.joblib",
    encoder_path="models/label_encoder.joblib",
)

# From file
result = engine.predict_from_file("data/raw/coastal/speaker_C001/maji_01.wav")
if not result.is_error:
    print(result.predicted_word, result.confidence)
else:
    print("Error:", result.error)

# From microphone
result = engine.predict_from_mic(duration=2.0)
print(result.predicted_word)
```

### Error Responses

| Condition | Response |
|---|---|
| Silent input | `error: "No speech detected"` |
| Audio < 0.5 s | `error: "Audio too short (Xs)"` |
| Unreadable file | `error: "Could not load audio: <reason>"` |
| Audio > 3.0 s | Truncated to 3.0 s with a warning — prediction still returned |

---

## Testing

The test suite uses `pytest` with `hypothesis` for property-based tests.

### Run all tests

```bash
pytest tests/ -v
```

### Run with coverage report

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html in a browser for the full report
```

### Run a specific module's tests

```bash
pytest tests/test_features.py -v
pytest tests/test_models.py -v
pytest tests/test_inference.py -v
```

### Run only property-based tests

```bash
pytest tests/ -v -k "property"
```

### Test Structure

| Test File | What It Covers |
|---|---|
| `test_preprocessing.py` | Silence trimming, resampling, normalization, error cases |
| `test_features.py` | MFCC shape `(39,)`, no NaN, delta computation, batch extraction |
| `test_models.py` | SVM/ANN train/predict, save/load round-trip, untrained model errors |
| `test_metrics.py` | WER, accuracy, confusion matrix, per-accent report |
| `test_inference.py` | Silent audio, short audio, valid audio, file/mic prediction |

### Key Correctness Properties (Property-Based Tests)

The following invariants are verified with `hypothesis` across randomly generated inputs:

1. `extract(audio).shape == (39,)` for any valid audio duration in [0.5, 3.0] s
2. `predict(audio).predicted_word in vocabulary` for any non-silent audio
3. `wer(refs, hyps) >= 0.0` for any reference/hypothesis lists
4. `predict_proba(X).sum(axis=1) ≈ 1.0` for any feature matrix
5. `wer == 1.0 - accuracy` within tolerance 1e-5 for isolated-word predictions

---

## Known Limitations

| Limitation | Details |
|---|---|
| **Isolated words only** | The system classifies single spoken words. It cannot handle continuous speech, sentences, or code-switching. |
| **Closed vocabulary** | The model can only predict words it was trained on. Out-of-vocabulary words will be misclassified as the nearest known word. |
| **CPU training** | Training is designed for CPU. SVM on ~10,000 samples takes ~5 min; ANN takes ~10–20 min. No GPU optimization is implemented. |
| **Small dataset sensitivity** | With fewer than ~500 samples per class, both models are prone to overfitting. Use data augmentation (noise addition, time stretching) to mitigate. |
| **Microphone quality** | Inference quality degrades significantly with low-quality microphones or high ambient noise. |
| **No speaker adaptation** | The system does not adapt to individual speakers at inference time. |
| **No language model** | There is no language model or n-gram rescoring. Each word is predicted independently. |

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Write tests for any new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Follow PEP 8 style (enforced via `ruff`)
5. Submit a pull request with a clear description of the change

### Code Style

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/
```

### Ethical Guidelines

- All voice recordings must be collected with **explicit informed consent**
- Speaker IDs must be **anonymized** — no PII stored alongside audio
- The system is intended for **Kiswahili word recognition research only**
- Do not use this system for speaker identification, surveillance, or any purpose beyond its stated scope

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

- [librosa](https://librosa.org/) — audio analysis library
- [scikit-learn](https://scikit-learn.org/) — SVM and preprocessing utilities
- [TensorFlow / Keras](https://www.tensorflow.org/) — ANN implementation
- [noisereduce](https://github.com/timsainburg/noisereduce) — spectral noise reduction
- [hypothesis](https://hypothesis.readthedocs.io/) — property-based testing framework

---

*Built for East African speech technology research.*
