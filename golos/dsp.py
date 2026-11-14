"""Обработка звука: энергия, привязка резов к тишине, кроссфейды, шумодав, громкость."""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def rms_frames(x: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    """RMS по окнам hop_s секунд."""
    hop = max(1, int(sr * hop_s))
    n = len(x) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    f = x[: n * hop].reshape(n, hop)
    return np.sqrt((f * f).mean(axis=1) + 1e-12).astype(np.float32)


def snap_to_quiet(t: float, rms: np.ndarray, hop_s: float, window: float = 0.08) -> float:
    """Сдвигает точку реза к самому тихому месту в пределах ±window."""
    i = int(round(t / hop_s))
    w = int(round(window / hop_s))
    a, b = max(0, i - w), min(len(rms), i + w + 1)
    if b <= a:
        return t
    j = a + int(np.argmin(rms[a:b]))
    return j * hop_s


def speech_mask(rms: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    """Грубый детектор речи по энергии: порог = шумовой пол + 12 дБ."""
    db = 20 * np.log10(rms + 1e-9)
    if floor_db is None:
        floor_db = float(np.percentile(db, 10))
    thr = max(floor_db + 12.0, -55.0)
    return db > thr


def concat_crossfade(chunks: list[np.ndarray], sr: int, fade_s: float = 0.012) -> np.ndarray:
    """Склейка с равномощным кроссфейдом на каждом стыке."""
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].astype(np.float32).copy()
    n = int(sr * fade_s)
    for c in chunks[1:]:
        c = c.astype(np.float32)
        k = min(n, len(out), len(c))
        if k > 1:
            t = np.linspace(0, np.pi / 2, k, dtype=np.float32)
            mixed = out[-k:] * np.cos(t) + c[:k] * np.sin(t)
            out = np.concatenate([out[:-k], mixed, c[k:]])
        else:
            out = np.concatenate([out, c])
    return out


def fade_edges(x: np.ndarray, sr: int, fade_s: float = 0.005) -> np.ndarray:
    n = min(int(sr * fade_s), len(x) // 2)
    if n > 1:
        x = x.copy()
        r = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def denoise(x: np.ndarray, sr: int, strength: float = 0.8) -> np.ndarray:
    try:
        import noisereduce as nr
    except ImportError:
        return x
    y = nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=float(strength))
    return y.astype(np.float32)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -16.0,
                       peak_db: float = -1.0) -> np.ndarray:
    if len(x) < sr // 2:
        return x
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(x.astype(np.float64))
        if np.isfinite(loud):
            x = x * (10 ** ((target_lufs - loud) / 20.0))
    except ImportError:
        pass
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    lim = 10 ** (peak_db / 20.0)
    if peak > lim:
        x = soft_limit(x, lim)
    return x.astype(np.float32)


def soft_limit(x: np.ndarray, ceiling: float) -> np.ndarray:
    """Мягкий лимитер: tanh-кривая выше 70 % потолка."""
    knee = ceiling * 0.7
    y = x.copy()
    over = np.abs(y) > knee
    s = np.sign(y[over])
    a = np.abs(y[over]) - knee
    y[over] = s * (knee + (ceiling - knee) * np.tanh(a / (ceiling - knee)))
    return y


def time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 — быстрее (короче)."""
    if abs(rate - 1.0) < 0.005:
        return x
    import librosa
    return librosa.effects.time_stretch(x.astype(np.float32), rate=rate).astype(np.float32)


def peaks(x: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """min/max по корзинам — для отрисовки волны."""
    if len(x) == 0 or buckets <= 0:
        return np.zeros(0), np.zeros(0)
    n = len(x)
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    edges = np.maximum.accumulate(edges)
    idx = edges[:-1]
    idx = np.minimum(idx, n - 1)
    mx = np.maximum.reduceat(x, idx)
    mn = np.minimum.reduceat(x, idx)
    return mn, mx
