"""Stream local microphone audio to the IoT WebSocket endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from urllib.parse import urlencode

import websockets


def build_url(args: argparse.Namespace) -> str:
    params = {
        "device_id": args.device_id,
        "expected_word": args.expected_word,
        "vad_enabled": "true",
        "sample_rate": str(args.sample_rate),
        "vad_threshold_db": str(args.vad_threshold_db),
        "vad_end_silence_ms": str(args.vad_end_silence_ms),
        "include_accent": str(args.include_accent).lower(),
    }
    return f"{args.server}/iot/microphone/stream?{urlencode(params)}"


async def receive_events(websocket) -> None:
    async for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(f"server: {message}")
            continue

        event = payload.get("event")
        if event == "prediction":
            print(json.dumps(payload, indent=2))
            return
        if event == "error":
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return
        print(json.dumps(payload))


async def stream_microphone(args: argparse.Namespace) -> None:
    try:
        import pyaudio
    except ImportError as exc:
        raise SystemExit("pyaudio is not installed. Install the mic extra or requirements.") from exc

    url = build_url(args)
    print(f"connecting: {url}")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=args.sample_rate,
            input=True,
            input_device_index=args.input_device,
            frames_per_buffer=args.frames_per_buffer,
        )

        async with websockets.connect(url, max_size=None) as websocket:
            receiver = asyncio.create_task(receive_events(websocket))
            deadline = time.monotonic() + args.max_seconds
            print("speak now; keep a short silence after the word...")

            while time.monotonic() < deadline and not receiver.done():
                data = stream.read(args.frames_per_buffer, exception_on_overflow=False)
                await websocket.send(data)
                await asyncio.sleep(0)

            if not receiver.done():
                await websocket.send(json.dumps({"event": "stop"}))
                await receiver
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio to Kiswahili ASR.")
    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:8001",
        help="WebSocket server base URL.",
    )
    parser.add_argument("--device-id", default="device-001")
    parser.add_argument("--expected-word", default="maji")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frames-per-buffer", type=int, default=1024)
    parser.add_argument("--vad-threshold-db", type=float, default=-40.0)
    parser.add_argument("--vad-end-silence-ms", type=int, default=700)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--include-accent", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(stream_microphone(parse_args()))


if __name__ == "__main__":
    main()
