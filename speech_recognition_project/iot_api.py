"""HTTP API for IoT microphone inference.

Run with:
    uvicorn iot_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import tempfile
import wave
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from src.inference.predict import InferenceEngine, PredictionResult
from src.models.svm_model import SVMModel


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_WORD_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
DEFAULT_WORD_SCALER_PATH = MODELS_DIR / "scaler.joblib"
DEFAULT_WORD_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
DEFAULT_ACCENT_MODEL_PATH = MODELS_DIR / "accent_svm_model.joblib"
DEFAULT_ACCENT_SCALER_PATH = MODELS_DIR / "accent_scaler.joblib"
DEFAULT_ACCENT_ENCODER_PATH = MODELS_DIR / "accent_label_encoder.joblib"

SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/flac",
    "audio/ogg",
    "audio/mp4",
    "application/octet-stream",
}

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_VAD_THRESHOLD_DB = -40.0
DEFAULT_VAD_END_SILENCE_MS = 700
DEFAULT_VAD_MIN_SPEECH_MS = 250
DEFAULT_VAD_PRE_ROLL_MS = 250


class TopPrediction(BaseModel):
    label: str
    probability: float


class PredictionPayload(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopPrediction]


class MicrophonePredictionResponse(BaseModel):
    device_id: str | None = None
    expected_word: str | None = None
    word: PredictionPayload
    prompt_match: bool | None = None
    accent: PredictionPayload | None = None


class HealthResponse(BaseModel):
    status: str
    word_model_ready: bool
    accent_model_ready: bool


@dataclass
class VoiceActivityDetector:
    """Energy-based VAD for 16-bit PCM microphone chunks."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH
    speech_threshold_db: float = DEFAULT_VAD_THRESHOLD_DB
    end_silence_ms: int = DEFAULT_VAD_END_SILENCE_MS
    min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS
    pre_roll_ms: int = DEFAULT_VAD_PRE_ROLL_MS
    triggered: bool = False
    speech_bytes: bytearray = field(default_factory=bytearray)
    silence_bytes: int = 0
    pre_roll: deque[bytes] = field(default_factory=deque)
    _pre_roll_bytes: int = 0

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width

    @property
    def min_speech_bytes(self) -> int:
        return int(self.bytes_per_second * self.min_speech_ms / 1000)

    @property
    def end_silence_bytes(self) -> int:
        return int(self.bytes_per_second * self.end_silence_ms / 1000)

    @property
    def max_pre_roll_bytes(self) -> int:
        return int(self.bytes_per_second * self.pre_roll_ms / 1000)

    def add_chunk(self, chunk: bytes) -> str | None:
        if not chunk:
            return None

        is_speech = _pcm16le_rms_db(chunk) >= self.speech_threshold_db

        if not self.triggered:
            self._remember_pre_roll(chunk)
            if not is_speech:
                return None
            self.triggered = True
            for buffered in self.pre_roll:
                self.speech_bytes.extend(buffered)
            self.pre_roll.clear()
            self._pre_roll_bytes = 0
            self.silence_bytes = 0
            return "speech_started"

        self.speech_bytes.extend(chunk)
        if is_speech:
            self.silence_bytes = 0
            return None

        self.silence_bytes += len(chunk)
        if (
            self.silence_bytes >= self.end_silence_bytes
            and len(self.speech_bytes) >= self.min_speech_bytes
        ):
            return "speech_ended"
        return None

    def finalize(self) -> bytes | None:
        if not self.speech_bytes or len(self.speech_bytes) < self.min_speech_bytes:
            self.reset()
            return None

        utterance = bytes(self.speech_bytes)
        self.reset()
        return utterance

    def reset(self) -> None:
        self.triggered = False
        self.speech_bytes.clear()
        self.silence_bytes = 0
        self.pre_roll.clear()
        self._pre_roll_bytes = 0

    def _remember_pre_roll(self, chunk: bytes) -> None:
        self.pre_roll.append(chunk)
        self._pre_roll_bytes += len(chunk)
        while self._pre_roll_bytes > self.max_pre_roll_bytes and self.pre_roll:
            removed = self.pre_roll.popleft()
            self._pre_roll_bytes -= len(removed)


app = FastAPI(
    title="Kiswahili ASR IoT API",
    description="Endpoint for IoT microphone devices to submit short recordings.",
    version="0.1.0",
)


def _pcm16le_rms_db(chunk: bytes) -> float:
    if len(chunk) < 2:
        return -120.0
    usable_length = len(chunk) - (len(chunk) % 2)
    samples = np.frombuffer(chunk[:usable_length], dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return -120.0
    samples /= 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return 20.0 * float(np.log10(rms + 1e-9))


def _write_pcm16le_wav(
    path: str,
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _artifact_set_exists(model_path: Path, scaler_path: Path, encoder_path: Path) -> bool:
    return model_path.exists() and scaler_path.exists() and encoder_path.exists()


def _load_svm_engine(model_path: Path, scaler_path: Path, encoder_path: Path) -> InferenceEngine:
    missing = [str(path) for path in (model_path, scaler_path, encoder_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

    model = SVMModel().load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return InferenceEngine(model=model, scaler=scaler, label_encoder=label_encoder)


@lru_cache(maxsize=1)
def get_word_engine() -> InferenceEngine:
    return _load_svm_engine(
        _path_from_env("ASR_WORD_MODEL_PATH", DEFAULT_WORD_MODEL_PATH),
        _path_from_env("ASR_WORD_SCALER_PATH", DEFAULT_WORD_SCALER_PATH),
        _path_from_env("ASR_WORD_ENCODER_PATH", DEFAULT_WORD_ENCODER_PATH),
    )


@lru_cache(maxsize=1)
def get_accent_engine() -> InferenceEngine:
    return _load_svm_engine(
        _path_from_env("ASR_ACCENT_MODEL_PATH", DEFAULT_ACCENT_MODEL_PATH),
        _path_from_env("ASR_ACCENT_SCALER_PATH", DEFAULT_ACCENT_SCALER_PATH),
        _path_from_env("ASR_ACCENT_ENCODER_PATH", DEFAULT_ACCENT_ENCODER_PATH),
    )


def _prediction_to_payload(result: PredictionResult) -> PredictionPayload:
    return PredictionPayload(
        label=result.predicted_word,
        confidence=result.confidence,
        top_k=[
            TopPrediction(label=str(label), probability=probability)
            for label, probability in result.top_k
        ],
    )


def _run_prediction(engine: InferenceEngine, audio_path: str, label: str) -> PredictionPayload:
    result = engine.predict_from_file(audio_path)
    if result.is_error:
        raise HTTPException(status_code=422, detail={label: result.error})
    return _prediction_to_payload(result)


def _classify_audio_file(
    audio_path: str,
    *,
    device_id: str | None = None,
    expected_word: str | None = None,
    include_accent: bool = False,
) -> MicrophonePredictionResponse:
    word = _run_prediction(get_word_engine(), audio_path, "word")

    accent = None
    if include_accent:
        accent = _run_prediction(get_accent_engine(), audio_path, "accent")

    prompt_match = None
    if expected_word:
        prompt_match = expected_word.strip().lower() == word.label.strip().lower()

    return MicrophonePredictionResponse(
        device_id=device_id,
        expected_word=expected_word,
        word=word,
        prompt_match=prompt_match,
        accent=accent,
    )


def _validate_audio_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in SUPPORTED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio content type: {file.content_type}",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    word_paths = (
        _path_from_env("ASR_WORD_MODEL_PATH", DEFAULT_WORD_MODEL_PATH),
        _path_from_env("ASR_WORD_SCALER_PATH", DEFAULT_WORD_SCALER_PATH),
        _path_from_env("ASR_WORD_ENCODER_PATH", DEFAULT_WORD_ENCODER_PATH),
    )
    accent_paths = (
        _path_from_env("ASR_ACCENT_MODEL_PATH", DEFAULT_ACCENT_MODEL_PATH),
        _path_from_env("ASR_ACCENT_SCALER_PATH", DEFAULT_ACCENT_SCALER_PATH),
        _path_from_env("ASR_ACCENT_ENCODER_PATH", DEFAULT_ACCENT_ENCODER_PATH),
    )
    return HealthResponse(
        status="ok",
        word_model_ready=_artifact_set_exists(*word_paths),
        accent_model_ready=_artifact_set_exists(*accent_paths),
    )


@app.post("/iot/microphone", response_model=MicrophonePredictionResponse)
async def predict_iot_microphone(
    file: Annotated[UploadFile, File(description="Short microphone recording.")],
    device_id: Annotated[str | None, Form()] = None,
    expected_word: Annotated[str | None, Form()] = None,
    include_accent: Annotated[bool, Form()] = False,
) -> MicrophonePredictionResponse:
    """Accept microphone audio from an IoT device and return ASR predictions."""
    _validate_audio_upload(file)

    suffix = Path(file.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        try:
            return _classify_audio_file(
                tmp.name,
                device_id=device_id,
                expected_word=expected_word,
                include_accent=include_accent,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/iot/microphone/stream")
async def stream_iot_microphone(
    websocket: WebSocket,
    device_id: str | None = None,
    expected_word: str | None = None,
    include_accent: bool = False,
    vad_enabled: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    vad_threshold_db: float = DEFAULT_VAD_THRESHOLD_DB,
    vad_end_silence_ms: int = DEFAULT_VAD_END_SILENCE_MS,
) -> None:
    """Receive microphone audio chunks over WebSocket and classify utterances.

    With VAD enabled, binary messages must be raw 16-bit little-endian PCM. The
    server detects speech end automatically. With VAD disabled, binary messages
    are appended to one audio file until {"event": "stop"} is received.
    """
    await websocket.accept()
    await websocket.send_json(
        {
            "event": "ready",
            "vad_enabled": vad_enabled,
            "sample_rate": sample_rate,
            "message": (
                "send raw PCM chunks; prediction is automatic after speech ends"
                if vad_enabled
                else "send audio file bytes, then send {'event': 'stop'}"
            ),
        }
    )

    detector = VoiceActivityDetector(
        sample_rate=sample_rate,
        speech_threshold_db=vad_threshold_db,
        end_silence_ms=vad_end_silence_ms,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return

                if "bytes" in message and message["bytes"] is not None:
                    chunk = message["bytes"]
                    if vad_enabled:
                        vad_event = detector.add_chunk(chunk)
                        if vad_event == "speech_started":
                            await websocket.send_json({"event": "speech_started"})
                        elif vad_event == "speech_ended":
                            utterance = detector.finalize()
                            if utterance is None:
                                await websocket.send_json({"event": "speech_discarded"})
                                continue
                            _write_pcm16le_wav(
                                tmp.name,
                                utterance,
                                sample_rate=sample_rate,
                            )
                            try:
                                prediction = _classify_audio_file(
                                    tmp.name,
                                    device_id=device_id,
                                    expected_word=expected_word,
                                    include_accent=include_accent,
                                )
                            except FileNotFoundError as exc:
                                await websocket.send_json(
                                    {"event": "error", "detail": str(exc)}
                                )
                                continue
                            except HTTPException as exc:
                                await websocket.send_json(
                                    {"event": "error", "detail": exc.detail}
                                )
                                continue

                            await websocket.send_json(
                                {
                                    "event": "prediction",
                                    "result": prediction.model_dump(),
                                }
                            )
                        else:
                            await websocket.send_json(
                                {"event": "chunk_received", "bytes": len(chunk)}
                            )
                    else:
                        tmp.write(chunk)
                        await websocket.send_json(
                            {"event": "chunk_received", "bytes": len(chunk)}
                        )
                    continue

                if "text" not in message or message["text"] is None:
                    continue

                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"event": "error", "detail": "Text messages must be JSON."}
                    )
                    continue

                event = payload.get("event")
                if event == "stop":
                    if vad_enabled:
                        utterance = detector.finalize()
                        if utterance is None:
                            await websocket.send_json(
                                {"event": "error", "detail": "No complete speech detected."}
                            )
                            continue
                        _write_pcm16le_wav(tmp.name, utterance, sample_rate=sample_rate)
                    else:
                        tmp.flush()

                    try:
                        prediction = _classify_audio_file(
                            tmp.name,
                            device_id=payload.get("device_id", device_id),
                            expected_word=payload.get("expected_word", expected_word),
                            include_accent=bool(payload.get("include_accent", include_accent)),
                        )
                    except FileNotFoundError as exc:
                        await websocket.send_json({"event": "error", "detail": str(exc)})
                        continue
                    except HTTPException as exc:
                        await websocket.send_json({"event": "error", "detail": exc.detail})
                        continue

                    await websocket.send_json(
                        {
                            "event": "prediction",
                            "result": prediction.model_dump(),
                        }
                    )
                    if not vad_enabled:
                        break
                    detector.reset()
                    continue

                if event == "cancel":
                    detector.reset()
                    await websocket.send_json({"event": "cancelled"})
                    break

                await websocket.send_json(
                    {"event": "error", "detail": f"Unsupported event: {event}"}
                )
        except WebSocketDisconnect:
            return
