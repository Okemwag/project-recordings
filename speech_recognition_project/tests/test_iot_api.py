"""Tests for the IoT microphone HTTP API."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
import numpy as np

import iot_api
from src.inference.predict import PredictionResult


class FakeEngine:
    def predict_from_file(self, path: str) -> PredictionResult:
        return PredictionResult(
            predicted_word="maji",
            confidence=0.91,
            top_k=[("maji", 0.91), ("chakula", 0.09)],
        )


def _pcm_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    return np.zeros(int(duration_sec * sample_rate), dtype="<i2").tobytes()


def _pcm_tone(duration_sec: float, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2")
    return samples.tobytes()


def test_iot_microphone_endpoint_returns_prediction(monkeypatch):
    monkeypatch.setattr(iot_api, "get_word_engine", lambda: FakeEngine())
    client = TestClient(iot_api.app)

    response = client.post(
        "/iot/microphone",
        data={"device_id": "device-001", "expected_word": "maji"},
        files={"file": ("recording.wav", b"fake wav bytes", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["device_id"] == "device-001"
    assert payload["expected_word"] == "maji"
    assert payload["word"]["label"] == "maji"
    assert payload["word"]["confidence"] == pytest.approx(0.91)
    assert payload["prompt_match"] is True
    assert payload["accent"] is None


def test_iot_microphone_endpoint_rejects_unsupported_content_type():
    client = TestClient(iot_api.app)

    response = client.post(
        "/iot/microphone",
        files={"file": ("recording.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415


def test_iot_microphone_stream_returns_prediction(monkeypatch):
    monkeypatch.setattr(iot_api, "get_word_engine", lambda: FakeEngine())
    client = TestClient(iot_api.app)

    with client.websocket_connect(
        "/iot/microphone/stream?device_id=device-001&expected_word=maji&vad_enabled=false"
    ) as websocket:
        ready = websocket.receive_json()
        assert ready["event"] == "ready"
        assert ready["vad_enabled"] is False

        websocket.send_bytes(b"fake wav bytes")
        chunk_ack = websocket.receive_json()
        assert chunk_ack["event"] == "chunk_received"

        websocket.send_json({"event": "stop"})
        prediction = websocket.receive_json()

    assert prediction["event"] == "prediction"
    result = prediction["result"]
    assert result["device_id"] == "device-001"
    assert result["word"]["label"] == "maji"
    assert result["prompt_match"] is True


def test_iot_microphone_stream_vad_auto_classifies_after_silence(monkeypatch):
    monkeypatch.setattr(iot_api, "get_word_engine", lambda: FakeEngine())
    client = TestClient(iot_api.app)

    with client.websocket_connect(
        "/iot/microphone/stream?device_id=device-001&expected_word=maji"
    ) as websocket:
        ready = websocket.receive_json()
        assert ready["event"] == "ready"
        assert ready["vad_enabled"] is True

        websocket.send_bytes(_pcm_silence(0.2))
        silence_ack = websocket.receive_json()
        assert silence_ack["event"] == "chunk_received"

        websocket.send_bytes(_pcm_tone(0.4))
        started = websocket.receive_json()
        assert started["event"] == "speech_started"

        websocket.send_bytes(_pcm_silence(0.8))
        prediction = websocket.receive_json()

    assert prediction["event"] == "prediction"
    result = prediction["result"]
    assert result["device_id"] == "device-001"
    assert result["word"]["label"] == "maji"
    assert result["prompt_match"] is True
