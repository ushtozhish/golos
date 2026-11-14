"""Модель проекта: исходная запись, слова, резы, фразы, переводы, варианты озвучки.

Монтаж неразрушающий: исходный raw.wav не меняется, результат вычисляется
из списка слов (вырезанные/оставленные), ручных резов и правил пауз.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import dsp

SR = 48000
HOP = 0.01

FILLERS = {"э", "ээ", "эээ", "эм", "мм", "ммм", "хм", "а-а", "э-э", "uh", "um", "erm", "hmm"}


@dataclass
class Word:
    text: str
    start: float
    end: float
    cut: bool = False      # вырезано (вручную или автоматически)
    auto: str = ""         # причина авто-реза: "dup" | "filler" | ""


@dataclass
class Gap:
    """Участок между словами, где есть речь, но Whisper её не распознал."""
    start: float
    end: float
    cut: bool = True


@dataclass
class EditSettings:
    shorten_pauses: bool = True
    max_pause: float = 0.45      # паузы длиннее — укорачиваются
    target_pause: float = 0.30   # до этой длины
    sentence_pause: float = 0.45 # пауза после точки
    head_tail: float = 0.15      # тишина в начале и конце
    pad: float = 0.06            # запас вокруг слова
    denoise: bool = True
    denoise_strength: float = 0.75
    loudness: bool = True
    target_lufs: float = -16.0


@dataclass
class Phrase:
    id: int
    text: str
    word_idx: list[int]
    start: float = 0.0   # на смонтированной шкале
    end: float = 0.0

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Variant:
    file: str
    dur: float
    cer: float | None = None
    stretched: float = 1.0


@dataclass
class DubPhrase:
    id: int
    text: str
    variants: list[Variant] = field(default_factory=list)
    chosen: int = -1


def norm_text(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def ends_sentence(word: str) -> bool:
    return bool(re.search(r"[.!?…]\W*$", word.strip()))


class Project:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.audio: np.ndarray = np.zeros(0, dtype=np.float32)
        self.words: list[Word] = []
        self.gaps: list[Gap] = []
        self.manual_cuts: list[list[float]] = []   # [start, end] на исходной шкале
        self.edit = EditSettings()
        self.source_lang = "ru"
        self.translations: dict[str, dict[int, str]] = {}
        self.dub: dict[str, list[DubPhrase]] = {}
        self.ref_override: list[float] | None = None
        self._rms: np.ndarray | None = None
        self._undo: list[str] = []
        self._redo: list[str] = []

    # ---------- файлы ----------
    @property
    def raw_path(self) -> Path:
        return self.folder / "raw.wav"

    @property
    def json_path(self) -> Path:
        return self.folder / "project.json"

    def dub_dir(self, lang: str) -> Path:
        d = self.folder / "dub" / lang
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def create(cls, folder: Path) -> "Project":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        p = cls(folder)
        p.save()
        return p

    @classmethod
    def open(cls, folder: Path) -> "Project":
        p = cls(folder)
        if p.json_path.exists():
            p._load_state(json.loads(p.json_path.read_text("utf-8")))
        if p.raw_path.exists():
            p.load_audio_file(p.raw_path, copy=False)
        return p

    def load_audio_file(self, path: Path, copy: bool = True):
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        x = dsp.resample(dsp.to_mono(x), sr, SR)
        self.set_audio(x, save=copy)

    def set_audio(self, x: np.ndarray, save: bool = True):
        self.audio = x.astype(np.float32)
        self._rms = None
        if save:
            import soundfile as sf
            sf.write(str(self.raw_path), self.audio, SR, subtype="PCM_24")
            self.words, self.gaps, self.manual_cuts = [], [], []
            self.dub, self.translations = {}, {}
            self._undo.clear(); self._redo.clear()
            self.save()

    @property
    def duration(self) -> float:
        return len(self.audio) / SR

    @property
    def rms(self) -> np.ndarray:
        if self._rms is None:
            self._rms = dsp.rms_frames(self.audio, SR, HOP)
        return self._rms

    def _state(self) -> dict:
        return {
            "version": 1,
            "source_lang": self.source_lang,
            "words": [asdict(w) for w in self.words],
            "gaps": [asdict(g) for g in self.gaps],
            "manual_cuts": self.manual_cuts,
            "edit": asdict(self.edit),
            "translations": {k: {str(i): t for i, t in v.items()} for k, v in self.translations.items()},
            "dub": {k: [{"id": d.id, "text": d.text, "chosen": d.chosen,
                         "variants": [asdict(v) for v in d.variants]} for d in v]
                    for k, v in self.dub.items()},
            "ref_override": self.ref_override,
        }

    def _load_state(self, s: dict):
        self.source_lang = s.get("source_lang", "ru")
        self.words = [Word(**w) for w in s.get("words", [])]
        self.gaps = [Gap(**g) for g in s.get("gaps", [])]
        self.manual_cuts = s.get("manual_cuts", [])
        e = EditSettings()
        for k, v in s.get("edit", {}).items():
            if hasattr(e, k):
                setattr(e, k, v)
        self.edit = e
        self.translations = {k: {int(i): t for i, t in v.items()} for k, v in s.get("translations", {}).items()}
        self.dub = {}
        for k, lst in s.get("dub", {}).items():
            self.dub[k] = [DubPhrase(id=d["id"], text=d["text"], chosen=d.get("chosen", -1),
                                     variants=[Variant(**v) for v in d.get("variants", [])]) for d in lst]
        self.ref_override = s.get("ref_override")

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state(), ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.json_path)

    # ---------- отмена ----------
    def _edit_snapshot(self) -> str:
        return json.dumps({"w": [[w.cut, w.auto] for w in self.words],
                           "g": [g.cut for g in self.gaps],
                           "m": self.manual_cuts, "e": asdict(self.edit)})

    def checkpoint(self):
        self._undo.append(self._edit_snapshot())
        if len(self._undo) > 300:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snap: str):
        s = json.loads(snap)
        for w, (c, a) in zip(self.words, s["w"]):
            w.cut, w.auto = c, a
        for g, c in zip(self.gaps, s["g"]):
            g.cut = c
        self.manual_cuts = s["m"]
        for k, v in s["e"].items():
            setattr(self.edit, k, v)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._edit_snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._edit_snapshot())
        self._restore(self._redo.pop())
        return True

    # ---------- распознавание → авторазметка ----------
    def set_words(self, words: list[Word]):
        dur = self.duration
        clean = []
        for w in words:
            t = w.text.strip()
            if not t:
                continue
            s = max(0.0, min(w.start, dur))
            e = max(s + 0.02, min(w.end if w.end else s + 0.3, dur))
            clean.append(Word(text=t, start=s, end=e))
        self.words = clean
        self._find_gaps()
        self.auto_mark()
        self.dub, self.translations = {}, {}
        self.save()

    def _find_gaps(self):
        """Ищет речь, которую Whisper пропустил (часто это повторы и оговорки)."""
        self.gaps = []
        mask = dsp.speech_mask(self.rms)
        bounds = [0.0] + [t for w in self.words for t in (w.start, w.end)] + [self.duration]
        for a, b in zip(bounds[0::2], bounds[1::2]):
            if b - a < 0.4:
                continue
            i0, i1 = int((a + 0.1) / HOP), int((b - 0.1) / HOP)
            seg = mask[i0:i1]
            if len(seg) and seg.sum() * HOP >= 0.25:
                idx = np.nonzero(seg)[0]
                s = (i0 + idx[0]) * HOP
                e = (i0 + idx[-1] + 1) * HOP
                self.gaps.append(Gap(start=s, end=e, cut=True))

    def raw_phrases(self) -> list[list[int]]:
        """Группы слов по знакам конца фразы и паузам — для поиска дублей."""
        groups, cur = [], []
        for i, w in enumerate(self.words):
            if cur:
                prev = self.words[cur[-1]]
                if w.start - prev.end > 0.6 or ends_sentence(prev.text) or prev.text.strip().endswith(","):
                    if w.start - prev.end > 0.35 or ends_sentence(prev.text):
                        groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        return groups

    def auto_mark(self):
        """Помечает дубли (ранние попытки фразы) и слова-паразиты."""
        for w in self.words:
            if w.auto:
                w.cut, w.auto = False, ""
        for w in self.words:
            if norm_text(w.text) in FILLERS:
                w.cut, w.auto = True, "filler"
        groups = self.raw_phrases()
        toks = [[norm_text(self.words[i].text) for i in g] for g in groups]
        for gi, a in enumerate(toks):
            a = [t for t in a if t]
            if not a:
                continue
            for gj in range(gi + 1, min(gi + 4, len(toks))):
                b = [t for t in toks[gj] if t]
                if not b:
                    continue
                if _is_retake(a, b):
                    for i in groups[gi]:
                        self.words[i].cut, self.words[i].auto = True, "dup"
                    break
            else:
                # фальстарт внутри одной группы: «клиент хочет — клиент хочет видеть»
                _mark_inner_restart(self.words, groups[gi])

    def clear_auto(self):
        for w in self.words:
            if w.auto:
                w.cut, w.auto = False, ""

    # ---------- вычисление монтажа ----------
    def keep_regions(self) -> list[tuple[float, float]]:
        """Участки исходника, которые идут в результат (исходная шкала)."""
        e = self.edit
        dur = self.duration
        if dur <= 0:
            return []
        if not self.words:
            regions = [(0.0, dur)]
        else:
            kept = [i for i, w in enumerate(self.words) if not w.cut]
            if not kept:
                return []
            first, last = self.words[kept[0]], self.words[kept[-1]]
            regions = []
            cur_s = max(0.0, first.start - e.head_tail)
            for k_prev, k in zip(kept, kept[1:]):
                wp, w = self.words[k_prev], self.words[k]
                gap = w.start - wp.end
                removed = (k - k_prev > 1) or self._gap_cut_between(wp.end, w.start)
                sent = ends_sentence(wp.text)
                target = e.sentence_pause if sent else e.target_pause
                limit = e.sentence_pause if sent else e.max_pause
                if removed or (e.shorten_pauses and gap > limit):
                    want = max(0.08, min(gap, target))
                    half = want / 2
                    a, b = wp.end + half, w.start - half
                    if b > a:
                        regions.append((cur_s, a))
                        cur_s = b
            regions.append((cur_s, min(dur, last.end + e.head_tail)))
        snapped = []
        for a, b in regions:
            a2 = dsp.snap_to_quiet(a, self.rms, HOP, 0.025) if a > 0 else a
            b2 = dsp.snap_to_quiet(b, self.rms, HOP, 0.025) if b < dur else b
            if b2 - a2 > 0.02:
                snapped.append((max(0.0, a2), min(dur, b2)))
        regions = _subtract(snapped, [tuple(c) for c in self.manual_cuts])
        return _merge(regions)

    def _gap_cut_between(self, a: float, b: float) -> bool:
        return any(g.cut and g.start >= a - 0.05 and g.end <= b + 0.05 for g in self.gaps)

    def edited_duration(self) -> float:
        return sum(b - a for a, b in self.keep_regions())

    def orig_to_edit(self, t: float, regions=None) -> float | None:
        regions = regions if regions is not None else self.keep_regions()
        acc = 0.0
        for a, b in regions:
            if t < a:
                return acc
            if t <= b:
                return acc + (t - a)
            acc += b - a
        return acc

    def edit_to_orig(self, t: float, regions=None) -> float:
        regions = regions if regions is not None else self.keep_regions()
        acc = 0.0
        for a, b in regions:
            if t <= acc + (b - a):
                return a + (t - acc)
            acc += b - a
        return regions[-1][1] if regions else 0.0

    def render(self, process: bool = True, regions=None) -> np.ndarray:
        regions = regions if regions is not None else self.keep_regions()
        chunks = [self.audio[int(a * SR): int(b * SR)] for a, b in regions]
        y = dsp.concat_crossfade(chunks, SR)
        if process and len(y):
            if self.edit.denoise:
                y = dsp.denoise(y, SR, self.edit.denoise_strength)
            if self.edit.loudness:
                y = dsp.normalize_loudness(y, SR, self.edit.target_lufs)
            y = dsp.fade_edges(y, SR)
        return y

    # ---------- фразы для перевода ----------
    def phrases(self) -> list[Phrase]:
        regions = self.keep_regions()
        kept = [i for i, w in enumerate(self.words) if not w.cut]
        out: list[Phrase] = []
        cur: list[int] = []

        def flush():
            if not cur:
                return
            text = " ".join(self.words[i].text.strip() for i in cur)
            text = re.sub(r"\s+([,.!?…:;])", r"\1", text)
            s = self.orig_to_edit(max(0.0, self.words[cur[0]].start - 0.03), regions)
            e = self.orig_to_edit(self.words[cur[-1]].end + 0.03, regions)
            out.append(Phrase(id=len(out) + 1, text=text, word_idx=list(cur), start=s, end=e))

        for k in kept:
            w = self.words[k]
            if cur:
                prev = self.words[cur[-1]]
                if ends_sentence(prev.text) or (w.start - prev.end > 0.9):
                    flush(); cur = []
            cur.append(k)
        flush()
        return out

    # ---------- образец голоса для клонирования ----------
    def reference_clip(self, min_s: float = 6.0, max_s: float = 14.0):
        """Возвращает (аудио 24 кГц, текст) — подряд идущие фразы без резов внутри."""
        if self.ref_override:
            a, b = self.ref_override
            words = [w for w in self.words if w.start >= a and w.end <= b and not w.cut]
            x = self.audio[int(a * SR): int(b * SR)]
            return dsp.resample(x, SR, 24000), " ".join(w.text for w in words)
        regions = self.keep_regions()
        phr = self.phrases()
        if not phr:
            raise RuntimeError("no speech")
        best = None
        for i in range(len(phr)):
            for j in range(i, len(phr)):
                total = phr[j].end - phr[i].start
                if total > max_s:
                    break
                if total >= min_s and (best is None or abs(total - 10) < abs(best[2] - 10)):
                    best = (i, j, total)
        if best is None:  # запись короче min_s — берём всё
            best = (0, len(phr) - 1, phr[-1].end - phr[0].start)
            if best[2] > max_s:
                best = (0, 0, phr[0].dur)
        i, j, _ = best
        y = self.render(process=False, regions=regions)
        a = max(0.0, phr[i].start - 0.05)
        b = phr[j].end + 0.1
        x = y[int(a * SR): int(b * SR)]
        if self.edit.denoise:
            x = dsp.denoise(x, SR, self.edit.denoise_strength)
        text = " ".join(ph.text for ph in phr[i:j + 1])
        return dsp.resample(x, SR, 24000), text


def _is_retake(a: list[str], b: list[str]) -> bool:
    """a — ранняя попытка b, если a совпадает с началом b (оборванный или полный дубль)."""
    if len(a) > len(b) + 2:
        return False
    head = b[: len(a) + 1]
    r = SequenceMatcher(None, a, head).ratio()
    if len(a) <= 2:
        return a == b[: len(a)]
    return r >= 0.72


def _mark_inner_restart(words: list[Word], group: list[int]):
    toks = [norm_text(words[i].text) for i in group]
    n = len(toks)
    for size in (4, 3, 2):
        for i in range(0, n - 2 * size + 1):
            if toks[i:i + size] == toks[i + size:i + 2 * size] and all(toks[i:i + size]):
                for k in group[i:i + size]:
                    if not words[k].cut:
                        words[k].cut, words[k].auto = True, "dup"
                return


def _merge(regs):
    regs = sorted((a, b) for a, b in regs if b > a)
    out = []
    for a, b in regs:
        if out and a <= out[-1][1] + 1e-4:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(regs, cuts):
    cuts = _merge(cuts)
    out = []
    for a, b in regs:
        segs = [(a, b)]
        for ca, cb in cuts:
            nxt = []
            for s, e in segs:
                if cb <= s or ca >= e:
                    nxt.append((s, e))
                else:
                    if ca > s:
                        nxt.append((s, ca))
                    if cb < e:
                        nxt.append((cb, e))
            segs = nxt
        out.extend(segs)
    return [(a, b) for a, b in out if b - a > 0.01]


def new_project_dir(root: Path) -> Path:
    return Path(root) / time.strftime("Озвучка %Y-%m-%d %H-%M")
