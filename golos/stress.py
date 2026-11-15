"""Ударения: «+» перед ударной гласной (зам+ок), словарь ударений и подготовка текста для модели.

Как модель поймёт ударение, заранее неизвестно (у Qwen3-TTS нет документированной разметки),
поэтому есть три способа передачи — выбирается проверкой на слух («Проверить ударение»):
  acute   — знак ударения над гласной: замо́к
  caps    — заглавная ударная гласная: замОк
  respell — «как слышится»: безударное «о» → «а» в размеченных словах: зам+ок → замок, молок+о → малако
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
ACUTE = "́"
MODES = ("acute", "caps", "respell")
MODE_NAMES = {"acute": "Знак ударения (замо́к)", "caps": "Заглавная буква (замОк)",
              "respell": "Как слышится (безударное о → а)"}
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё+́-]+")


def strip(text: str) -> str:
    """Текст без пометок ударения — для перевода, проверки и показа."""
    return text.replace("+", "").replace(ACUTE, "")


def vowel_count(word: str) -> int:
    return sum(ch in VOWELS for ch in word)


# ---------------- словарь ----------------
def dict_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Golos"
    base.mkdir(parents=True, exist_ok=True)
    return base / "stress.json"


def load_dict() -> dict[str, str]:
    try:
        return json.loads(dict_path().read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_word(stressed: str):
    """Запоминает слово с ударением: {«замок»: «зам+ок»}."""
    stressed = stressed.strip()
    key = strip(stressed).lower()
    if "+" not in stressed or not key:
        return
    d = load_dict()
    d[key] = stressed.lower()
    dict_path().write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")


def remove_word(word: str):
    d = load_dict()
    d.pop(strip(word).lower(), None)
    dict_path().write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")


def _match_case(src: str, stressed: str) -> str:
    """Переносит регистр исходного слова на слово из словаря (с учётом «+»)."""
    out, i = [], 0
    for ch in stressed:
        if ch == "+":
            out.append(ch); continue
        s = src[i] if i < len(src) else ch
        out.append(ch.upper() if s.isupper() else ch)
        i += 1
    return "".join(out)


def apply_dict(text: str) -> str:
    """Расставляет ударения из словаря в словах, где их ещё нет."""
    d = load_dict()
    if not d:
        return text

    def rep(m):
        w = m.group(0)
        if "+" in w or ACUTE in w:
            return w
        st = d.get(w.lower())
        return _match_case(w, st) if st else w
    return _WORD.sub(rep, text)


# ---------------- автоматическая расстановка (RuAccent) ----------------
_ACC = None


def _accentizer(log=lambda m: None):
    global _ACC
    if _ACC is None:
        from ruaccent import RUAccent
        from .models import models_dir
        wd = models_dir() / "ruaccent"
        wd.mkdir(parents=True, exist_ok=True)
        log("RuAccent: загружаю модель ударений (один раз, на процессоре)…")
        a = RUAccent()
        a.load(omograph_model_size="turbo3.1", use_dictionary=True, custom_dict=load_dict(),
               device="CPU", workdir=str(wd))
        _ACC = a
    return _ACC


def auto_stress_word(phrase: str, word_index: int, log=lambda m: None) -> str | None:
    """Ставит ударение в одно слово фразы (с учётом контекста — для омографов). Возвращает новую фразу."""
    words = list(_WORD.finditer(phrase))
    if not 0 <= word_index < len(words):
        return None
    target = words[word_index]
    if vowel_count(target.group(0)) < 2 and "ё" not in target.group(0).lower():
        return None
    acc = _accentizer(log)
    res = acc.process_all(strip(phrase))
    rwords = [m.group(0) for m in _WORD.finditer(res)]
    # сопоставляем по порядку слов без пометок
    plain = [strip(w.group(0)).lower() for w in words]
    rplain = [strip(w).lower() for w in rwords]
    j = None
    if len(rplain) == len(plain) and rplain[word_index] == plain[word_index]:
        j = word_index
    else:
        cands = [k for k, w in enumerate(rplain) if w == plain[word_index]]
        if cands:
            j = min(cands, key=lambda k: abs(k - word_index))
    if j is None or "+" not in rwords[j]:
        return None
    stressed = _match_case(strip(target.group(0)), rwords[j])
    return phrase[:target.start()] + stressed + phrase[target.end():]


def words_of(phrase: str) -> list[str]:
    return [m.group(0) for m in _WORD.finditer(phrase)]


# ---------------- подготовка текста для модели ----------------
def for_model(text: str, lang: str, mode: str = "acute") -> str:
    """Текст, который уходит в нейросеть: ударения из словаря + выбранный способ передачи."""
    if lang != "ru":
        return strip(text)
    text = apply_pron(text)
    text = apply_dict(text)

    def conv(m):
        w = m.group(0)
        if "+" not in w:
            return w
        i = w.index("+")
        base = w.replace("+", "")
        # ударная — первая гласная после «+» (на случай, если плюс поставили перед согласной)
        k = next((n for n in range(i, len(base)) if base[n] in VOWELS), None)
        if k is None:
            return base
        if mode == "caps":
            return base[:k] + base[k].upper() + base[k + 1:]
        if mode == "respell":
            chars = list(base)
            for n, ch in enumerate(chars):
                if n != k and ch in "оО":
                    chars[n] = "а" if ch == "о" else "А"
            return "".join(chars)
        return base[:k + 1] + ACUTE + base[k + 1:]
    return _WORD.sub(conv, text)


# ---------------- ударение по записи голоса ----------------
def vowel_positions(word: str) -> list[int]:
    w = strip(word)
    return [i for i, ch in enumerate(w) if ch in VOWELS]


def set_stress(word: str, vowel_no: int) -> str:
    """Ставит «+» перед vowel_no-й гласной слова (счёт с 0)."""
    w = strip(word)
    pos = vowel_positions(w)
    if not 0 <= vowel_no < len(pos):
        return w
    i = pos[vowel_no]
    return w[:i] + "+" + w[i:]


def detect_from_audio(x, sr: int, word: str):
    """Находит ударный слог в записи одного слова. Возвращает (номер гласной, уверенность 0..1) или (None, 0).
    Признаки ударения в русском: ударный гласный длиннее и громче безударных (и обычно выше по тону)."""
    import numpy as np
    from scipy.signal import find_peaks, peak_widths
    from . import dsp
    n = len(vowel_positions(word))
    if n == 0:
        return None, 0.0
    if n == 1:
        return 0, 1.0
    x = np.asarray(x, dtype=np.float32)
    hop = 0.01
    rms = dsp.rms_frames(x, sr, hop)
    if len(rms) < 10 or rms.max() < 1e-3:
        return None, 0.0
    # сглаживание 40 мс и обрезка тишины
    k = 4
    env = np.convolve(rms, np.ones(k) / k, mode="same")
    thr = env.max() * 0.08
    voiced = np.nonzero(env > thr)[0]
    if len(voiced) < 5:
        return None, 0.0
    a, b = voiced[0], voiced[-1] + 1
    e = env[a:b]
    # слоговые ядра — пики энергии
    peaks, props = find_peaks(e, distance=max(1, int(0.07 / hop)), prominence=e.max() * 0.06)
    if len(peaks) != n:
        # не совпало число слогов — берём n самых заметных пиков или делим звучание поровну
        if len(peaks) > n:
            order = np.argsort(props["prominences"])[::-1][:n]
            peaks = np.sort(peaks[order])
        else:
            edges = np.linspace(0, len(e), n + 1).astype(int)
            peaks = np.array([s + int(np.argmax(e[s:t])) for s, t in zip(edges[:-1], edges[1:]) if t > s])
            if len(peaks) != n:
                return None, 0.0
    widths = peak_widths(e, peaks, rel_height=0.5)[0] * hop
    heights = e[peaks]
    # тон: средняя частота основного тона вокруг пика (если librosa есть)
    pitch = np.ones(n)
    try:
        import librosa
        seg = x[int(a * hop * sr): int(b * hop * sr)]
        f0, vflag, _ = librosa.pyin(seg, fmin=70, fmax=400, sr=sr, frame_length=1024,
                                    hop_length=int(hop * sr))
        for i, p in enumerate(peaks):
            w = f0[max(0, p - 3): p + 4]
            w = w[~np.isnan(w)] if w is not None else []
            if len(w):
                pitch[i] = float(np.mean(w))
        if np.all(pitch > 1):
            pitch = pitch / pitch.mean()
        else:
            pitch = np.ones(n)
    except Exception:  # noqa: BLE001
        pass
    score = (widths / widths.mean()) * 0.5 + (heights / heights.mean()) * 0.3 + pitch * 0.2
    best = int(np.argmax(score))
    srt = np.sort(score)[::-1]
    conf = float(min(1.0, (srt[0] - srt[1]) / max(1e-6, srt[0]) * 4)) if n > 1 else 1.0
    return best, conf


# ---------------- произношение слов (бренды, английские слова в русской речи) ----------------
def pron_path() -> Path:
    return dict_path().with_name("pronounce.json")


def load_pron() -> dict[str, str]:
    try:
        return json.loads(pron_path().read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_pron(word: str, how: str):
    d = load_pron()
    if how.strip():
        d[strip(word).lower()] = how.strip()
    else:
        d.pop(strip(word).lower(), None)
    pron_path().write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")


def apply_pron(text: str) -> str:
    d = load_pron()
    if not d:
        return text

    def rep(m):
        w = m.group(0)
        return d.get(strip(w).lower(), w)
    return _WORD.sub(rep, text)


# ---------------- двуязычные фразы ----------------
MIXED_MODES = ("auto", "split", "plain")
MIXED_NAMES = {"split": "Склейка: английские куски — английским произношением",
               "auto": "Авто: модель сама определяет язык",
               "plain": "Как пишется (всё одним языком)"}
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*(?:[\s,.:;!?—–-]+[A-Za-z][A-Za-z0-9'’\-]*)*")


def has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text))


def split_by_script(text: str) -> list[tuple[str, str]]:
    """Делит фразу на куски [(«ru»|«en», текст)]; соседние латинские слова идут одним куском."""
    out, pos = [], 0
    for m in _LATIN_RUN.finditer(text):
        if m.start() > pos and text[pos:m.start()].strip(" ,.;:!?…—–-«»\""):
            out.append(("ru", text[pos:m.start()].strip()))
        elif m.start() > pos and out:
            out[-1] = (out[-1][0], out[-1][1] + text[pos:m.start()].rstrip())
        out.append(("en", m.group(0).strip()))
        pos = m.end()
    if pos < len(text) and text[pos:].strip(" ,.;:!?…—–-«»\""):
        out.append(("ru", text[pos:].strip()))
    elif pos < len(text) and out:
        out[-1] = (out[-1][0], out[-1][1] + text[pos:].rstrip())
    return [(l, t) for l, t in out if t]
