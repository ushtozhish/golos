"""Запись с микрофона и воспроизведение (sounddevice)."""
from __future__ import annotations

import threading
import numpy as np


def input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd
    out = []
    try:
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                api = hostapis[d["hostapi"]]["name"]
                if "WDM-KS" in api:
                    continue
                out.append((i, f"{d['name']} — {api}"))
    except Exception:  # noqa: BLE001
        pass
    return out


def default_input() -> int | None:
    import sounddevice as sd
    try:
        return sd.default.device[0]
    except Exception:  # noqa: BLE001
        return None


class Recorder:
    def __init__(self, device: int | None, sr: int = 48000):
        self.device, self.sr = device, sr
        self.chunks: list[np.ndarray] = []
        self.level = 0.0
        self._stream = None
        self._lock = threading.Lock()

    def _cb(self, indata, frames, t, status):
        x = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        with self._lock:
            self.chunks.append(x)
        self.level = float(np.max(np.abs(x))) if len(x) else 0.0

    def start(self):
        import sounddevice as sd
        self.chunks = []
        try:
            self._stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sr,
                                          dtype="float32", callback=self._cb, blocksize=1024)
        except Exception:  # noqa: BLE001 — устройство не умеет 48 кГц
            info = sd.query_devices(self.device, "input")
            self.sr = int(info["default_samplerate"])
            self._stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sr,
                                          dtype="float32", callback=self._cb, blocksize=1024)
        self._stream.start()

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(c) for c in self.chunks) / self.sr

    def stop(self) -> tuple[np.ndarray, int]:
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        with self._lock:
            x = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        return x.astype(np.float32), self.sr


class Player:
    """Проигрывает массив; position() — текущая позиция в секундах от начала массива."""

    def __init__(self):
        self._stream = None
        self._data = None
        self._pos = 0
        self.sr = 48000
        self.on_finish = None

    def play(self, x: np.ndarray, sr: int, start_s: float = 0.0):
        import sounddevice as sd
        self.stop()
        self._data = x.astype(np.float32)
        self.sr = sr
        self._pos = int(max(0.0, start_s) * sr)

        def cb(outdata, frames, t, status):
            a = self._pos
            b = min(a + frames, len(self._data))
            n = b - a
            outdata[:n, 0] = self._data[a:b]
            if n < frames:
                outdata[n:, 0] = 0
                raise sd.CallbackStop
            self._pos = b

        self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", callback=cb,
                                       finished_callback=self._finished)
        self._stream.start()

    def _finished(self):
        if self.on_finish:
            self.on_finish()

    def position(self) -> float:
        return self._pos / self.sr

    @property
    def playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
