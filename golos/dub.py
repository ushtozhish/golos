"""Озвучка переводов своим голосом: варианты, проверка распознаванием, подгонка по времени, сборка дорожки."""
from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Callable

import numpy as np

from . import dsp
from .models import TTS, TTS_LANGS, Whisper, free_gpu
from .project import Project, DubPhrase, Variant, norm_text, SR

MAX_STRETCH = 1.12   # ускоряем не больше чем на 12 %
OUT_SR = 48000


def cer(ref: str, hyp: str) -> float:
    a, b = norm_text(ref).replace(" ", ""), norm_text(hyp).replace(" ", "")
    if not a:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def trim_silence(x: np.ndarray, sr: int, pad: float = 0.04) -> np.ndarray:
    if len(x) == 0:
        return x
    rms = dsp.rms_frames(x, sr, 0.01)
    m = dsp.speech_mask(rms)
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return x
    a = max(0, int((idx[0] * 0.01 - pad) * sr))
    b = min(len(x), int(((idx[-1] + 1) * 0.01 + pad) * sr))
    return x[a:b]


class Dubber:
    def __init__(self, project: Project, log: Callable[[str], None],
                 progress: Callable[[int, int], None] = lambda a, b: None,
                 cancelled: Callable[[], bool] = lambda: False,
                 precision: str = "fp16"):
        self.p = project
        self.log = log
        self.progress = progress
        self.cancelled = cancelled
        self.precision = precision

    def texts_for(self, lang: str) -> dict[int, str]:
        phrases = self.p.phrases()
        if lang == self.p.source_lang:
            return {ph.id: ph.text for ph in phrases}
        return self.p.translations.get(lang, {})

    def run(self, langs: list[str], n_variants: int = 3, verify: bool = True,
            only_ids: dict[str, set[int]] | None = None, temperature: float = 0.8):
        phrases = {ph.id: ph for ph in self.p.phrases()}
        ref_audio, ref_text = self.p.reference_clip()
        self.log(f"Образец голоса: {len(ref_audio) / 24000:.1f} с — «{ref_text[:80]}…»")
        by_model: dict[str, list[str]] = {}
        for lang in langs:
            by_model.setdefault(TTS_LANGS[lang][0], []).append(lang)

        jobs = []
        for lang in langs:
            texts = self.texts_for(lang)
            existing = {d.id: d for d in self.p.dub.get(lang, [])}
            lst = []
            for pid, ph in phrases.items():
                t = texts.get(pid, "").strip()
                d = existing.get(pid)
                if d is None or d.text != t:
                    d = DubPhrase(id=pid, text=t)
                lst.append(d)
            self.p.dub[lang] = lst
            for d in lst:
                want = only_ids is None or d.id in only_ids.get(lang, set())
                if d.text and want:
                    jobs.append((lang, d))
        total = len(jobs) * n_variants
        done = 0
        precision = self.precision
        for repo, model_langs in by_model.items():
            mjobs = [j for j in jobs if j[0] in model_langs]
            if not mjobs:
                continue
            tts = TTS(repo, self.log, precision)
            try:
                tts.set_reference(ref_audio, ref_text)
                for lang, d in mjobs:
                    if self.cancelled():
                        return
                    slot = phrases[d.id].dur
                    lname = TTS_LANGS[lang][1]
                    try:
                        wavs, sr = tts.generate([d.text] * n_variants, lname, temperature,
                                                max_seconds=max(6.0, slot * 2.5))
                    except (FloatingPointError, RuntimeError) as e:
                        if tts.precision == "fp32":
                            raise
                        self.log(f"fp16 дал сбой ({e}); перезагружаю модель в fp32")
                        tts.close()
                        precision = "fp32"
                        tts = TTS(repo, self.log, "fp32")
                        tts.set_reference(ref_audio, ref_text)
                        wavs, sr = tts.generate([d.text] * n_variants, lname, temperature,
                                                max_seconds=max(6.0, slot * 2.5))
                    d.variants = []
                    for vi, w in enumerate(wavs):
                        w = trim_silence(dsp.resample(w, sr, OUT_SR), OUT_SR)
                        f = self.p.dub_dir(lang) / f"p{d.id:03d}_v{vi + 1}_{int(time.time())}.wav"
                        _write(f, w)
                        d.variants.append(Variant(file=f.name, dur=len(w) / OUT_SR))
                        done += 1
                        self.progress(done, total)
                    d.chosen = -1
                    self.log(f"[{lang}] фраза {d.id}: {len(wavs)} вар.")
                    self.p.save()
            finally:
                tts.close()
        if verify and jobs:
            self.verify([j for j in jobs])
        for lang, d in jobs:
            self.auto_choose(d, phrases[d.id].dur)
        self.p.save()

    def verify(self, jobs):
        self.log("Проверяю варианты распознаванием (ищу пропуски и «галлюцинации»)…")
        wh = Whisper(self.log)
        try:
            import soundfile as sf
            for lang, d in jobs:
                for v in d.variants:
                    if self.cancelled():
                        return
                    x, sr = sf.read(str(self.p.dub_dir(lang) / v.file), dtype="float32")
                    try:
                        hyp = wh.text(x, sr, lang)
                        v.cer = round(cer(d.text, hyp), 3)
                    except Exception as e:  # noqa: BLE001
                        self.log(f"проверка не удалась: {e}")
        finally:
            wh.close()
        free_gpu()

    @staticmethod
    def score(v: Variant, slot: float) -> float:
        ratio = v.dur / slot if slot > 0 else 1.0
        c = v.cer if v.cer is not None else 0.1
        return c * 4.0 + max(0.0, ratio - 1.0) * 1.5 + abs(ratio - 1.0) * 0.3

    def auto_choose(self, d: DubPhrase, slot: float):
        if d.variants:
            d.chosen = min(range(len(d.variants)), key=lambda i: self.score(d.variants[i], slot))


def fit_status(v: Variant, slot: float) -> str:
    """ok — влезает; stretch — влезет с ускорением до 12 %; long — не влезает."""
    if slot <= 0:
        return "ok"
    r = v.dur / slot
    if r <= 1.0:
        return "ok"
    if r <= MAX_STRETCH:
        return "stretch"
    return "long"


def build_track(p: Project, lang: str) -> tuple[np.ndarray, list[dict], list[str]]:
    """Собирает дорожку языка: каждая фраза ставится на место русской фразы.
    Возвращает (аудио 48 кГц, тайминги, предупреждения)."""
    import soundfile as sf
    phrases = {ph.id: ph for ph in p.phrases()}
    items = sorted(p.dub.get(lang, []), key=lambda d: d.id)
    total = p.edited_duration()
    out = np.zeros(int((total + 1.0) * OUT_SR), dtype=np.float32)
    timings, warns = [], []
    cursor = 0.0
    for d in items:
        ph = phrases.get(d.id)
        if ph is None or d.chosen < 0 or d.chosen >= len(d.variants):
            if ph is not None:
                warns.append(f"фраза {d.id}: нет выбранного варианта")
            continue
        v = d.variants[d.chosen]
        x, sr = sf.read(str(p.dub_dir(lang) / v.file), dtype="float32")
        x = dsp.resample(dsp.to_mono(x), sr, OUT_SR)
        slot = ph.dur
        if slot > 0 and len(x) / OUT_SR > slot:
            rate = min(MAX_STRETCH, (len(x) / OUT_SR) / slot)
            x = dsp.time_stretch(x, rate)
            if len(x) / OUT_SR > slot * 1.01:
                warns.append(f"фраза {d.id}: длиннее оригинала на {len(x) / OUT_SR - slot:.2f} с — сдвигаю следующие")
        start = max(ph.start, cursor)
        a = int(start * OUT_SR)
        need = a + len(x)
        if need > len(out):
            out = np.concatenate([out, np.zeros(need - len(out) + OUT_SR, dtype=np.float32)])
        out[a:a + len(x)] += x
        cursor = start + len(x) / OUT_SR + 0.08
        timings.append({"id": d.id, "start": round(start, 3), "end": round(start + len(x) / OUT_SR, 3),
                        "text": d.text})
    end = max(total, cursor)
    out = out[: int(end * OUT_SR)]
    out = dsp.normalize_loudness(out, OUT_SR, p.edit.target_lufs)
    return out, timings, warns


def _write(path, x):
    import soundfile as sf
    sf.write(str(path), x, OUT_SR, subtype="PCM_24")
