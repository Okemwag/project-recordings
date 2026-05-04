"""
predict.py — Inference engine for the Kiswahili ASR system.

Accepts audio from a file path or microphone, runs the full preprocessing
and feature extraction pipeline, and returns a predicted word with confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features.mfcc_extraction import FeatureExtractor
from src.preprocessing.noise_reduction import NoiseReducer
from src.preprocessing.normalization import AudioNormalizer
from src.preprocessing.silence_removal import InsufficientAudioError, SilenceRemover
from src.utils.audio import load_audio

logger = logging.getLogger(__name__)

VALID_ACCENTS = {"coastal", "nairobi", "upcountry"}


@dataclass
class PredictionResult:
    """Result of a single inference call."""

    predicted_word: str = ""
    confidence: float = 0.0
    top_k: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


class InferenceEngine:
    """
    End-to-end inference engine.

    Wraps preprocessing, feature extraction, normalization, and model
    prediction into a single predict() interface.
    """

    def __init__(
        self,
        model,
        scaler: StandardScaler,
        label_encoder: LabelEncoder,
        feature_extractor: FeatureExtractor | None = None,
        min_duration: float = 0.5,
        max_duration: float = 3.0,
        silence_threshold_db: float = 40.0,
        top_k: int = 5,
    ):
        """
        Args:
            model:               Trained SVMModel or ANNModel instance.
            scaler:              Fitted StandardScaler from training.
            label_encoder:       Fitted LabelEncoder from training.
            feature_extractor:   FeatureExtractor instance (default config if None).
            min_duration:        Minimum audio duration in seconds.
            max_duration:        Maximum audio duration in seconds.
            silence_threshold_db: RMS threshold (dB) for silence detection.
            top_k:               Number of top predictions to return.
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_threshold_db = silence_threshold_db
        self.top_k = top_k

        self._noise_reducer = NoiseReducer()
        self._silence_remover = SilenceRemover(
            min_duration=min_duration, max_duration=max_duration
        )
        self._normalizer = AudioNormalizer()

    def predict_from_file(self, path: str) -> PredictionResult:
        """
        Predict the word spoken in an audio file.

        Args:
            path: Path to a WAV audio file.

        Returns:
            PredictionResult with predicted word and confidence.
        """
        try:
            audio, orig_sr = load_audio(path)
        except Exception as e:
            return PredictionResult(error=f"Failed to load audio: {e}")

        audio = self._normalizer.resample(audio, orig_sr)
        return self._run_pipeline(audio)

    def predict_from_mic(self, duration: float = 2.0) -> PredictionResult:
        """
        Record from the microphone and predict the spoken word.

        Args:
            duration: Recording duration in seconds.

        Returns:
            PredictionResult with predicted word and confidence.
        """
        try:
            import pyaudio
            import struct

            sr = 16000
            chunk = 1024
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=chunk,
            )

            logger.info("Recording for %.1f seconds...", duration)
            frames = []
            num_chunks = int(sr / chunk * duration)
            for _ in range(num_chunks):
                data = stream.read(chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            pa.terminate()

            # Convert bytes to float32
            raw = b"".join(frames)
            samples = struct.unpack(f"{len(raw) // 2}h", raw)
            audio = np.array(samples, dtype=np.float32) / 32768.0

        except ImportError:
            return PredictionResult(
                error="pyaudio is not installed. Run: pip install pyaudio"
            )
        except Exception as e:
            return PredictionResult(error=f"Microphone error: {e}")

        return self._run_pipeline(audio)

    def _is_silent(self, audio: np.ndarray) -> bool:
        """Return True if audio RMS energy is below the silence threshold."""
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        return rms_db < -self.silence_threshold_db

    def _run_pipeline(self, audio: np.ndarray) -> PredictionResult:
        """
        Run the full inference pipeline on a preprocessed audio array.

        Args:
            audio: 1D float32 audio array at 16kHz.

        Returns:
            PredictionResult.
        """
        # Silence check
        if self._is_silent(audio):
            return PredictionResult(error="No speech detected")

        # Noise reduction + silence trimming + normalization
        try:
            audio = self._noise_reducer.reduce(audio)
            audio = self._silence_remover.trim(audio)
            audio = self._normalizer.normalize_amplitude(audio)
        except InsufficientAudioError as e:
            return PredictionResult(error=str(e))
        except ValueError as e:
            return PredictionResult(error=str(e))

        # Feature extraction
        try:
            fv = self.feature_extractor.extract(audio)
        except Exception as e:
            return PredictionResult(error=f"Feature extraction failed: {e}")

        # Normalize with training scaler
        fv_scaled = self.scaler.transform(fv.reshape(1, -1))

        # Predict
        proba = self.model.predict_proba(fv_scaled)[0]
        top_indices = np.argsort(proba)[::-1][: self.top_k]
        top_k_results = [
            (self.label_encoder.classes_[i], float(proba[i]))
            for i in top_indices
        ]

        return PredictionResult(
            predicted_word=top_k_results[0][0],
            confidence=top_k_results[0][1],
            top_k=top_k_results,
            error=None,
        )
