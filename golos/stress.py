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
