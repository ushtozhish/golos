"""Озвучка переводов своим голосом: варианты, проверка распознаванием, подгонка по времени, сборка дорожки."""
from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Callable

import numpy as np

from . import dsp
from .models import TTS, TTS_LANGS, Whisper, free_gpu, get_tts, release_tts
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
                 precision: str = "fp16", stress_mode: str = "acute", mixed_mode: str = "auto"):
        self.p = project
        self.log = log
        self.progress = progress
        self.cancelled = cancelled
        self.precision = precision
        self.stress_mode = stress_mode
        self.mixed_mode = mixed_mode

    def texts_for(self, lang: str) -> dict[int, str]:
        phrases = self.p.phrases()
        if lang == self.p.source_lang:
            return {ph.id: ph.text for ph in phrases}
        return self.p.translations.get(lang, {})

    def run(self, langs: list[str], n_variants: int = 3, verify: bool = True,
            only_ids: dict[str, set[int]] | None = None, temperature: float = 0.8,
            replace_variant: tuple[str, int, int] | None = None, top_up: bool = False):
        """replace_variant=(язык, id фразы, № варианта) — пересоздать только один вариант.
        top_up=True — «Догенерировать»: оставить имеющиеся варианты и досоздать до n_variants актуальных."""
        if replace_variant:
            only_ids = {replace_variant[0]: {replace_variant[1]}}
        phrases = {ph.id: ph for ph in self.p.phrases()}
        from . import voices
        voice = voices.get(self.p.voice)
        ref_audio, ref_text = voices.reference(voice, self.p)
        self.log(f"Голос «{voice.name}»: образец {len(ref_audio) / 24000:.1f} с — «{ref_text[:80]}…»")
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
                if d is None:
                    d = DubPhrase(id=pid, text=t, voice=voice.id)
                for v in d.variants:              # старые проекты: вариант помнит текст и голос фразы
                    v.text = v.text or d.text
                    v.voice = v.voice or d.voice
                want = only_ids is None or pid in only_ids.get(lang, set())
                if want:
                    # варианты не удаляются: текст и голос фразы просто обновляются,
                    # старые варианты будут помечены «устарело»
                    d.text, d.voice = t, voice.id
                lst.append(d)
                if t and want:
                    jobs.append((lang, d))
            self.p.dub[lang] = lst
        single = replace_variant is not None

        def need(d) -> int:
            if single:
                return 1
            if not top_up:
                return n_variants
            fresh = sum(1 for v in d.variants if v.text == d.text and v.voice == d.voice)
            return max(0, n_variants - fresh)

        if top_up:
            jobs = [(lang, d) for lang, d in jobs if need(d) > 0]
            if not jobs:
                self.log("Догенерировать нечего: у всех фраз уже достаточно вариантов")
        total = sum(need(d) for _, d in jobs)
        done = 0
        precision = self.precision
        for repo, model_langs in by_model.items():
            mjobs = [j for j in jobs if j[0] in model_langs]
            if not mjobs:
                continue
            tts = get_tts(repo, self.log, precision)
            try:
                tts.set_reference(ref_audio, ref_text)
                for lang, d in mjobs:
                    if self.cancelled():
                        return
                    slot = phrases[d.id].dur
                    lname = TTS_LANGS[lang][1]
                    n = need(d)
                    if n <= 0:
                        continue
                    try:
                        wavs, sr = self._gen(tts, d.text, lang, n, temperature, max(6.0, slot * 2.5))
                    except (FloatingPointError, RuntimeError) as e:
                        if tts.precision == "fp32":
                            raise
                        self.log(f"fp16 дал сбой ({e}); перезагружаю модель в fp32")
                        release_tts()
                        precision = "fp32"
                        tts = get_tts(repo, self.log, "fp32")
                        tts.set_reference(ref_audio, ref_text)
                        wavs, sr = self._gen(tts, d.text, lang, n, temperature, max(6.0, slot * 2.5))
                    new = []
                    for vi, w in enumerate(wavs):
                        w = trim_silence(dsp.resample(w, sr, OUT_SR), OUT_SR)
                        f = self.p.dub_dir(lang) / f"p{d.id:03d}_v{vi + 1}_{int(time.time() * 1000)}.wav"
                        _write(f, w)
                        new.append(Variant(file=f.name, dur=len(w) / OUT_SR, text=d.text, voice=voice.id))
                        done += 1
                        self.progress(done, total)
                    if single:
                        k = replace_variant[2]
                        if 0 <= k < len(d.variants) and not d.variants[k].pinned:
                            d.variants[k] = new[0]
                        else:
                            d.variants.append(new[0])     # ★ не трогаем — новый вариант рядом
                        self.log(f"[{lang}] фраза {d.id}: вариант {k + 1} пересоздан")
                    elif top_up:
                        d.variants.extend(new)
                        self.log(f"[{lang}] фраза {d.id}: догенерировано {len(new)}, всего {len(d.variants)}")
                    else:
                        keep = [v for v in d.variants if v.pinned]
                        chosen_file = d.variants[d.chosen].file if 0 <= d.chosen < len(d.variants) else None
                        d.variants = keep + new
                        d.chosen = next((i for i, v in enumerate(d.variants) if v.file == chosen_file), -1)
                        self.log(f"[{lang}] фраза {d.id}: {len(new)} вар." +
                                 (f" (+ отмеченных ★: {len(keep)})" if keep else ""))
                    d.slot = round(slot, 3)
                    self.p.save()
            finally:
                pass                      # модель остаётся в памяти для следующих действий
        if verify and jobs:
            only = {replace_variant[2]} if single else None
            self.verify(jobs, only, skip_checked=top_up)
        if not single:
            for lang, d in jobs:
                if top_up and 0 <= d.chosen < len(d.variants) and d.variants[d.chosen].text == d.text:
                    continue                      # выбранный вариант остаётся выбранным
                self.auto_choose(d, phrases[d.id].dur)
        self.p.save()

    def _gen(self, tts, text: str, lang: str, n: int, temperature: float, max_s: float, mixed: str | None = None):
        """Генерация n вариантов фразы с учётом ударений, словаря произношения и двуязычия."""
        from . import stress
        mixed = mixed or self.mixed_mode
        mtext = stress.for_model(text, lang, self.stress_mode)
        lname = TTS_LANGS[lang][1]
        if lang == "ru" and stress.has_latin(mtext) and mixed != "plain":
            if mixed == "auto":
                try:
                    return tts.generate([mtext] * n, "auto", temperature, max_seconds=max_s)
                except FloatingPointError:
                    raise
                except Exception as e:  # noqa: BLE001
                    self.log(f"Режим «Авто» не поддержан моделью ({type(e).__name__}) — использую склейку")
            segs = stress.split_by_script(mtext)
            if len(segs) > 1 or segs[0][0] == "en":
                parts, sr = [], 24000
                for l, t in segs:
                    ln = "english" if l == "en" else "russian"
                    w, sr = tts.generate([t] * n, ln, temperature, max_seconds=max(4.0, len(t) * 0.15 + 3))
                    parts.append([trim_silence(np.asarray(x, np.float32), sr, pad=0.03) for x in w])
                gap = np.zeros(int(0.07 * sr), np.float32)
                out = []
                for i in range(n):
                    chunks = []
                    for k, seg in enumerate(parts):
                        if k:
                            chunks.append(gap)
                        chunks.append(seg[i])
                    out.append(dsp.concat_crossfade(chunks, sr, 0.01))
                return out, sr
        return tts.generate([mtext] * n, lname, temperature, max_seconds=max_s)

    def mixed_test(self, text: str, on_item=None) -> list[tuple[str, str]]:
        """Двуязычная фраза тремя способами — выбрать на слух."""
        from . import stress, voices
        voice = voices.get(self.p.voice)
        ref_audio, ref_text = voices.reference(voice, self.p)
        tts = get_tts(TTS_LANGS["ru"][0], self.log, self.precision)
        tts.set_reference(ref_audio, ref_text)
        d = self.p.folder / "stress_test"
        d.mkdir(exist_ok=True)
        out = []
        for i, mode in enumerate(stress.MIXED_MODES):
            if self.cancelled():
                break
            self.log(f"Двуязычие, способ «{stress.MIXED_NAMES[mode]}»")
            wavs, sr = self._gen(tts, text, "ru", 1, 0.6, 25.0, mixed=mode)
            w = trim_silence(dsp.resample(np.asarray(wavs[0], np.float32), sr, OUT_SR), OUT_SR)
            f = d / f"mixed_{mode}_{int(time.time() * 1000)}.wav"
            _write(f, w)
            out.append((mode, str(f)))
            if on_item:
                on_item(mode, str(f))
            self.progress(i + 1, len(stress.MIXED_MODES))
        return out

    def stress_test(self, text: str, lang: str = "ru", on_item=None) -> list[tuple[str, str]]:
        """Одна фраза тремя способами передачи ударения; on_item(mode, path) вызывается по мере готовности."""
        from . import stress, voices
        voice = voices.get(self.p.voice)
        ref_audio, ref_text = voices.reference(voice, self.p)
        repo, lname, _ = TTS_LANGS[lang]
        tts = get_tts(repo, self.log, self.precision)
        tts.set_reference(ref_audio, ref_text)
        d = self.p.folder / "stress_test"
        d.mkdir(exist_ok=True)
        out = []
        for i, mode in enumerate(stress.MODES):
            if self.cancelled():
                break
            mtext = stress.for_model(text, lang, mode)
            self.log(f"Ударение, способ «{stress.MODE_NAMES[mode]}»: {mtext}")
            wavs, sr = tts.generate([mtext], lname, 0.6, max_seconds=20)
            w = trim_silence(dsp.resample(wavs[0], sr, OUT_SR), OUT_SR)
            f = d / f"{mode}_{int(time.time() * 1000)}.wav"
            _write(f, w)
            out.append((mode, str(f)))
            if on_item:
                on_item(mode, str(f))
            self.progress(i + 1, len(stress.MODES))
        return out

    def verify(self, jobs, only_variants: set[int] | None = None, skip_checked: bool = False):
        self.log("Проверяю варианты распознаванием (ищу пропуски и «галлюцинации»)…")
        wh = Whisper(self.log)
        try:
            import soundfile as sf
            for lang, d in jobs:
                for vi, v in enumerate(d.variants):
                    if only_variants is not None and vi not in only_variants:
                        continue
                    if skip_checked and v.cer is not None:
                        continue
                    if self.cancelled():
                        return
                    x, sr = sf.read(str(self.p.dub_dir(lang) / v.file), dtype="float32")
                    try:
                        hyp = wh.text(x, sr, lang)
                        from . import stress
                        c1 = cer(stress.strip(d.text), hyp)
                        c2 = cer(stress.strip(stress.apply_pron(d.text)), hyp)
                        v.cer = round(min(c1, c2), 3)
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
        """Если выбран ★-вариант с актуальным текстом — оставляем его; иначе лучший по проверке и длительности."""
        if not d.variants:
            return
        if 0 <= d.chosen < len(d.variants):
            v = d.variants[d.chosen]
            if v.pinned and v.text == d.text:
                return
        fresh = [i for i, v in enumerate(d.variants) if v.text == d.text and v.voice == d.voice] or \
            list(range(len(d.variants)))
        d.chosen = min(fresh, key=lambda i: self.score(d.variants[i], slot))


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
    total = p.seq_duration()
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
