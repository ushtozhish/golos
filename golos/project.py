"""Модель проекта: исходная запись, ручной монтаж клипами, распознанный текст, переводы, варианты озвучки.

Монтаж неразрушающий: raw.wav не меняется. Результат — последовательность клипов
(отрезков исходника), стоящих встык. Шкала «sequence» — время на смонтированной дорожке.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import dsp

SR = 48000
HOP = 0.01


@dataclass
class Word:
    text: str
    start: float   # на смонтированной шкале
    end: float


@dataclass
class EditSettings:
    denoise: bool = True
    denoise_strength: float = 0.75
    loudness: bool = True
    target_lufs: float = -16.0
    pause_max: float = 0.6      # инструмент «Укоротить паузы»: паузы длиннее…
    pause_target: float = 0.35  # …сокращаются до


@dataclass
class Phrase:
    id: int
    text: str
    start: float = 0.0
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
    pinned: bool = False     # ★ отмечен как удачный — не удаляется при перегенерации
    text: str = ""           # текст, по которому сгенерирован
    voice: str = ""          # голос, которым сгенерирован


@dataclass
class DubPhrase:
    id: int
    text: str
    variants: list[Variant] = field(default_factory=list)
    chosen: int = -1
    voice: str = ""
    slot: float = 0.0       # длительность фразы, под которую генерировали


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
        self.clips: list[list[float]] = []       # [[a, b], ...] на исходной шкале, в порядке дорожки
        self.words: list[Word] = []              # распознано по смонтированной дорожке
        self.words_key: str = ""                 # монтаж, по которому распознано
        self.edit = EditSettings()
        self.source_lang = "ru"
        self.translations: dict[str, dict[int, str]] = {}
        self.phrase_texts: dict[int, str] = {}   # исправленный вручную текст фраз
        self.split_after: list[int] = []         # индексы слов, после которых фраза кончается (из вставленного текста)
        self.dur_override: dict[int, float] = {}  # длительность фразы, заданная вручную
        self.dub: dict[str, list[DubPhrase]] = {}
        self.voice = "me"                        # id голоса из библиотеки
        self.from_srt = False                    # проект открыт из SRT (звука начитки нет)
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
            import soundfile as sf
            x, sr = sf.read(str(p.raw_path), dtype="float32", always_2d=False)
            p.audio = dsp.resample(dsp.to_mono(x), sr, SR)
            if not p.clips:
                p.clips = [[0.0, p.duration]]
        return p

    def load_audio_file(self, path: Path):
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        self.set_audio(dsp.resample(dsp.to_mono(x), sr, SR))

    def set_audio(self, x: np.ndarray):
        """Новая запись: заменяет всё."""
        self.audio = x.astype(np.float32)
        self._rms = None
        self.clips = [[0.0, self.duration]] if len(x) else []
        self.words, self.words_key = [], ""
        self.dub, self.translations, self.phrase_texts = {}, {}, {}
        self._undo.clear(); self._redo.clear()
        self._write_raw()
        self.save()

    def append_audio(self, x: np.ndarray):
        """Дозапись: добавляет новый кусок в конец исходника и клипом в конец дорожки."""
        self.checkpoint()
        a = self.duration
        gap = np.zeros(int(0.05 * SR), np.float32)
        self.audio = np.concatenate([self.audio, gap, x.astype(np.float32)])
        self._rms = None
        self.clips.append([a + 0.05, self.duration])
        self._write_raw()
        self.save()

    def _write_raw(self):
        import soundfile as sf
        sf.write(str(self.raw_path), self.audio, SR, subtype="PCM_24")

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
            "version": 2,
            "source_lang": self.source_lang,
            "clips": self.clips,
            "words": [asdict(w) for w in self.words],
            "words_key": self.words_key,
            "edit": asdict(self.edit),
            "translations": {k: {str(i): t for i, t in v.items()} for k, v in self.translations.items()},
            "phrase_texts": {str(k): v for k, v in self.phrase_texts.items()},
            "split_after": self.split_after,
            "dur_override": {str(k): v for k, v in self.dur_override.items()},
            "dub": {k: [{"id": d.id, "text": d.text, "chosen": d.chosen, "voice": d.voice, "slot": d.slot,
                         "variants": [asdict(v) for v in d.variants]} for d in v]
                    for k, v in self.dub.items()},
            "voice": self.voice,
            "from_srt": self.from_srt,
        }

    def _load_state(self, s: dict):
        self.source_lang = s.get("source_lang", "ru")
        self.clips = [list(c) for c in s.get("clips", [])]
        self.words = [Word(**w) for w in s.get("words", [])]
        self.words_key = s.get("words_key", "")
        e = EditSettings()
        for k, v in s.get("edit", {}).items():
            if hasattr(e, k):
                setattr(e, k, v)
        self.edit = e
        self.translations = {k: {int(i): t for i, t in v.items()} for k, v in s.get("translations", {}).items()}
        self.phrase_texts = {int(k): v for k, v in s.get("phrase_texts", {}).items()}
        self.split_after = list(s.get("split_after", []))
        self.dur_override = {int(k): float(v) for k, v in s.get("dur_override", {}).items()}
        self.dub = {}
        for k, lst in s.get("dub", {}).items():
            self.dub[k] = [DubPhrase(id=d["id"], text=d["text"], chosen=d.get("chosen", -1),
                                     voice=d.get("voice", ""), slot=d.get("slot", 0.0),
                                     variants=[Variant(**v) for v in d.get("variants", [])]) for d in lst]
        self.voice = s.get("voice", "me")
        self.from_srt = bool(s.get("from_srt", False))

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state(), ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.json_path)

    # ---------- отмена ----------
    def checkpoint(self):
        self._undo.append(json.dumps(self.clips))
        if len(self._undo) > 500:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(json.dumps(self.clips))
        self.clips = json.loads(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(json.dumps(self.clips))
        self.clips = json.loads(self._redo.pop())
        return True

    # ---------- шкала дорожки ----------
    def clip_starts(self) -> list[float]:
        out, acc = [], 0.0
        for a, b in self.clips:
            out.append(acc)
            acc += b - a
        return out

    def seq_duration(self) -> float:
        return sum(b - a for a, b in self.clips)

    def clip_at(self, t: float) -> int:
        """Индекс клипа под временем t дорожки (-1, если вне дорожки)."""
        acc = 0.0
        for i, (a, b) in enumerate(self.clips):
            if acc <= t < acc + (b - a):
                return i
            acc += b - a
        return len(self.clips) - 1 if self.clips and abs(t - acc) < 1e-6 else -1

    def seq_to_orig(self, t: float) -> float:
        acc = 0.0
        for a, b in self.clips:
            if t <= acc + (b - a):
                return a + max(0.0, t - acc)
            acc += b - a
        return self.clips[-1][1] if self.clips else 0.0

    # ---------- инструменты монтажа ----------
    def split(self, t: float) -> bool:
        """T — разрезать клип в точке t дорожки."""
        i = self.clip_at(t)
        if i < 0:
            return False
        a, b = self.clips[i]
        o = a + (t - self.clip_starts()[i])
        if o - a < 0.01 or b - o < 0.01:
            return False
        self.checkpoint()
        self.clips[i:i + 1] = [[a, o], [o, b]]
        return True

    def delete_clips(self, idx: set[int]) -> bool:
        if not idx:
            return False
        self.checkpoint()
        self.clips = [c for i, c in enumerate(self.clips) if i not in idx]
        return True

    def delete_range(self, t0: float, t1: float, checkpoint: bool = True) -> bool:
        """Вырезать участок дорожки [t0, t1] со сдвигом (дырка закрывается)."""
        if t1 - t0 < 0.005:
            return False
        if checkpoint:
            self.checkpoint()
        out, acc = [], 0.0
        for a, b in self.clips:
            s, e = acc, acc + (b - a)
            acc = e
            if e <= t0 or s >= t1:
                out.append([a, b]); continue
            if s < t0:
                out.append([a, a + (t0 - s)])
            if e > t1:
                out.append([a + (t1 - s), b])
        self.clips = [c for c in out if c[1] - c[0] > 0.005]
        return True

    def trim_to(self, i: int, t: float, side: str) -> bool:
        """Q/W — обрезать начало/конец клипа i до точки t дорожки."""
        if not 0 <= i < len(self.clips):
            return False
        a, b = self.clips[i]
        s = self.clip_starts()[i]
        o = a + (t - s)
        if not a < o < b:
            return False
        self.checkpoint()
        self.clips[i] = [o, b] if side == "start" else [a, o]
        return True

    def set_clip_edge(self, i: int, side: str, orig_t: float, checkpoint: bool):
        """Перетаскивание края клипа мышью (в пределах исходника)."""
        if checkpoint:
            self.checkpoint()
        a, b = self.clips[i]
        if side == "start":
            self.clips[i][0] = float(np.clip(orig_t, 0.0, b - 0.02))
        else:
            self.clips[i][1] = float(np.clip(orig_t, a + 0.02, self.duration))

    def restore_all(self):
        self.checkpoint()
        self.clips = [[0.0, self.duration]]

    def silences(self, min_len: float) -> list[tuple[float, float]]:
        """Паузы на дорожке длиннее min_len (по энергии)."""
        mask = dsp.speech_mask(self.rms)
        out = []
        for (a, b), s in zip(self.clips, self.clip_starts()):
            i0, i1 = int(a / HOP), int(b / HOP)
            seg = ~mask[i0:i1]
            run = 0
            for k, quiet in enumerate(np.append(seg, False)):
                if quiet:
                    run += 1
                else:
                    if run * HOP >= 0.05:
                        st = s + (k - run) * HOP
                        en = s + k * HOP
                        out.append((st, en))
                    run = 0
        merged: list[tuple[float, float]] = []
        for a, b in sorted(out):              # пауза через стык клипов — одна пауза
            if merged and a - merged[-1][1] < 0.02:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return [(a, b) for a, b in merged if b - a >= min_len]

    def shorten_pauses(self, max_len: float, target: float) -> int:
        """Инструмент «Укоротить паузы»: каждую паузу длиннее max_len сокращает до target.
        Это обычные резы — их видно на дорожке, отменяются одним Ctrl+Z."""
        total = self.seq_duration()
        sil = self.silences(max_len)
        if not sil:
            return 0
        self.checkpoint()
        n = 0
        for a, b in sorted(sil, reverse=True):     # с конца, чтобы не сдвигать время
            edge = a <= 0.05 or b >= total - 0.05
            keep = min(target, 0.15) if edge else target
            if b - a <= keep + 0.01:
                continue
            if a <= 0.05:
                self.delete_range(a, b - keep, checkpoint=False)
            elif b >= total - 0.05:
                self.delete_range(a + keep, b, checkpoint=False)
            else:
                self.delete_range(a + keep / 2, b - keep / 2, checkpoint=False)
            n += 1
        if n == 0:
            self._undo.pop()
        return n

    # ---------- результат ----------
    def keep_regions(self) -> list[tuple[float, float]]:
        return [(a, b) for a, b in self.clips if b > a]

    def edit_key(self) -> str:
        return hashlib.md5(json.dumps([[round(a, 4), round(b, 4)] for a, b in self.clips]).encode()).hexdigest()

    @property
    def words_fresh(self) -> bool:
        return bool(self.words) and self.words_key == self.edit_key()

    def render(self, process: bool = True) -> np.ndarray:
        chunks = [self.audio[int(a * SR): int(b * SR)] for a, b in self.keep_regions()]
        y = dsp.concat_crossfade(chunks, SR, 0.008)
        if process and len(y):
            if self.edit.denoise:
                y = dsp.denoise(y, SR, self.edit.denoise_strength)
            if self.edit.loudness:
                y = dsp.normalize_loudness(y, SR, self.edit.target_lufs)
            y = dsp.fade_edges(y, SR)
        return y

    def set_words(self, words: list[Word]):
        dur = self.seq_duration()
        clean = []
        for w in words:
            t = w.text.strip()
            if not t:
                continue
            s = max(0.0, min(w.start, dur))
            e = max(s + 0.02, min(w.end if w.end else s + 0.3, dur))
            clean.append(Word(text=t, start=s, end=e))
        old = {ph.id: ph.text for ph in self.phrases()} if self.words else {}
        self.words = clean
        self.words_key = self.edit_key()
        new = {ph.id: ph.text for ph in self.phrases()}
        if old != new:
            self.phrase_texts = {}
            self.split_after = []
            self.dur_override = {}
        self.save()

    def load_cues(self, cues) -> int:
        """Проект из SRT: каждый титр — готовая фраза с таймингом (текст не распознаётся)."""
        dur = max((c.end for c in cues), default=0.0) + 1.0
        if self.duration < dur - 1.0:
            # звука нет или он короче субтитров — дополняем тишиной до конца последнего титра
            pad = np.zeros(int((dur - self.duration) * SR), np.float32)
            self.audio = np.concatenate([self.audio, pad])
            self._rms = None
            self._write_raw()
        self.clips = [[0.0, self.duration]]
        self.words = []
        self.set_words([Word(c.text, c.start, c.end) for c in cues])
        self.split_after = list(range(len(self.words)))
        self.phrase_texts = {}
        self.from_srt = True
        self.save()
        return len(self.words)

    # ---------- фразы ----------
    MAX_PHRASE = 12.0   # длиннее — делим по самой длинной паузе внутри

    def word_groups(self) -> list[list[int]]:
        """Слова, сгруппированные во фразы: по точкам, паузам > 0,9 с, границам из вставленного текста;
        фразы длиннее MAX_PHRASE делятся по самой длинной внутренней паузе."""
        cut = set(self.split_after)
        groups, cur = [], []
        for i, w in enumerate(self.words):
            if cur:
                prev = cur[-1]
                by_text = bool(cut) and prev in cut
                by_punct = not cut and ends_sentence(self.words[prev].text)
                if by_text or by_punct or w.start - self.words[prev].end > 0.9:
                    groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        out = []
        stack = list(reversed(groups))
        while stack:
            g = stack.pop()
            dur = self.words[g[-1]].end - self.words[g[0]].start
            if dur <= self.MAX_PHRASE or len(g) < 4:
                out.append(g); continue
            # лучшая точка деления: длинная пауза, ближе к середине, желательно после запятой
            best, score = None, -1.0
            for k in range(1, len(g)):
                gap = self.words[g[k]].start - self.words[g[k - 1]].end
                t = self.words[g[k]].start - self.words[g[0]].start
                mid = 1.0 - abs(t / dur - 0.5)
                comma = 0.15 if self.words[g[k - 1]].text.rstrip().endswith((",", ";", ":", "—")) else 0.0
                sc = gap + 0.25 * mid + comma
                if sc > score:
                    best, score = k, sc
            stack.append(g[best:]); stack.append(g[:best])
        return out

    def phrases(self) -> list[Phrase]:
        out: list[Phrase] = []
        for g in self.word_groups():
            ws = [self.words[i] for i in g]
            text = " ".join(w.text.strip() for w in ws)
            text = re.sub(r"\s+([,.!?…:;])", r"\1", text)
            pid = len(out) + 1
            st = max(0.0, ws[0].start - 0.03)
            en = st + self.dur_override[pid] if pid in self.dur_override else ws[-1].end + 0.03
            out.append(Phrase(id=pid, text=self.phrase_texts.get(pid, text), start=st, end=en))
        return out

    def apply_script(self, script: str) -> int:
        """Ваш текст вместо распознанного. Слова распознавания выравниваются со словами текста;
        фразы режутся по предложениям вашего текста, каждая фраза получает свой кусок текста."""
        from difflib import SequenceMatcher
        sw = script.split()
        if not sw or not self.words:
            return 0
        n = len(self.words)
        a = [norm_text(w.text) for w in self.words]
        b = [norm_text(w) for w in sw]
        amap: dict[int, int] = {}
        for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                amap[blk.a + k] = blk.b + k
        known = sorted(amap.items())

        def b_of(i: float) -> float:          # слово распознавания → позиция в тексте
            lo = max(((x, y) for x, y in known if x <= i), default=(-1, -1))
            hi = min(((x, y) for x, y in known if x >= i), default=(n, len(sw)))
            if lo[0] == hi[0]:
                return lo[1]
            return lo[1] + (hi[1] - lo[1]) * (i - lo[0]) / max(1, hi[0] - lo[0])

        def a_of(j: int) -> int:              # позиция в тексте → слово распознавания (последнее до неё)
            best = None
            for x, y in known:
                if y <= j:
                    best = x
            if best is not None:
                return best
            return int(round(j * n / max(1, len(sw))))

        # 1) границы фраз — концы предложений вашего текста
        ends = [j for j, w in enumerate(sw) if re.search(r"[.!?…]\W*$", w)]
        self.split_after = sorted({min(n - 1, max(0, a_of(j))) for j in ends if j < len(sw) - 1})
        groups = self.word_groups()
        # 2) кусок текста для каждой фразы
        starts = [0]
        for g in groups[1:]:
            j = int(round(b_of(g[0])))
            for d in (0, -1, 1, -2, 2, -3, 3):     # к ближайшему концу предложения/запятой
                jj = j + d
                if 0 < jj < len(sw) and re.search(r"[.!?…,;:—]\W*$", sw[jj - 1]):
                    j = jj
                    break
            starts.append(min(len(sw), max(starts[-1] + 1, j)))
        starts.append(len(sw))
        self.phrase_texts = {}
        for k in range(len(groups)):
            chunk = " ".join(sw[starts[k]:starts[k + 1]]).strip()
            if chunk:
                self.phrase_texts[k + 1] = chunk
        self.save()
        return len(groups)

    def recover_variants(self, lang: str, pid: int) -> int:
        """Возвращает в список варианты фразы, файлы которых остались в папке dub/<язык>."""
        import re as _re
        d = next((x for x in self.dub.get(lang, []) if x.id == pid), None)
        if d is None:
            d = DubPhrase(id=pid, text="")
            self.dub.setdefault(lang, []).append(d)
            self.dub[lang].sort(key=lambda x: x.id)
        known = {v.file for v in d.variants}
        n = 0
        folder = self.folder / "dub" / lang
        if not folder.exists():
            return 0
        import soundfile as sf
        for f in sorted(folder.glob(f"p{pid:03d}_v*.wav"), key=lambda q: q.stat().st_mtime):
            if f.name in known:
                continue
            try:
                dur = sf.info(str(f)).duration
            except Exception:  # noqa: BLE001
                continue
            d.variants.append(Variant(file=f.name, dur=dur, text="(восстановлен)", voice="?"))
            n += 1
        if n:
            self.save()
        return n

    # ---------- образец голоса ----------
    def reference_clip(self, min_s: float = 6.0, max_s: float = 14.0):
        """(аудио 24 кГц, текст) — подряд идущие фразы смонтированной дорожки, ~10 с."""
        phr = self.phrases()
        if not phr:
            raise RuntimeError("Сначала распознайте начитку")
        best = None
        for i in range(len(phr)):
            for j in range(i, len(phr)):
                total = phr[j].end - phr[i].start
                if total > max_s:
                    break
                if total >= min_s and (best is None or abs(total - 10) < abs(best[2] - 10)):
                    best = (i, j, total)
        if best is None:
            best = (0, len(phr) - 1, phr[-1].end - phr[0].start)
            if best[2] > max_s:
                best = (0, 0, phr[0].dur)
        i, j, _ = best
        y = self.render(process=False)
        a, b = max(0.0, phr[i].start - 0.05), phr[j].end + 0.1
        x = y[int(a * SR): int(b * SR)]
        if self.edit.denoise:
            x = dsp.denoise(x, SR, self.edit.denoise_strength)
        text = " ".join(ph.text for ph in phr[i:j + 1])
        return dsp.resample(x, SR, 24000), text


def new_project_dir(root: Path) -> Path:
    return Path(root) / time.strftime("Озвучка %Y-%m-%d %H-%M")
