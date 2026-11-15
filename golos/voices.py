"""Библиотека голосов. Любой голос хранится как образец ~10 с + его текст:
по нему модель клонирования (Qwen3-TTS Base / AIT-Syn) говорит на любом языке, включая казахский.

Виды голосов:
  me      — ваш голос из текущей начитки (образец берётся из проекта автоматически)
  file    — образец из аудиофайла (только с согласия владельца голоса)
  design  — голос по текстовому описанию (Qwen3-TTS VoiceDesign)
  preset  — готовый диктор (Qwen3-TTS CustomVoice)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import dsp

SAMPLE_TEXT = {
    "ru": "Добрый день! Это образец голоса. Сейчас я спокойно и чётко прочитаю небольшой текст, "
          "с естественными паузами и живой интонацией.",
    "en": "Hello! This is a voice sample. I am going to read a short text calmly and clearly, "
          "with natural pauses and a lively intonation.",
}


@dataclass
class Voice:
    id: str
    name: str
    kind: str              # me | file | design | preset | etalon
    text: str = ""         # текст образца
    description: str = ""  # для design
    speaker: str = ""      # для preset
    created: str = ""

    @property
    def wav(self) -> Path:
        return voices_dir() / self.id / "ref.wav"


def voices_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Golos" / "voices"
    base.mkdir(parents=True, exist_ok=True)
    return base


ME = Voice(id="me", name="Мой голос (из начитки)", kind="me")


def list_voices() -> list[Voice]:
    out = [ME]
    for d in sorted(voices_dir().iterdir()):
        f = d / "voice.json"
        if f.exists():
            try:
                out.append(Voice(**json.loads(f.read_text("utf-8"))))
            except Exception:  # noqa: BLE001
                pass
    return out


def get(vid: str) -> Voice:
    return next((v for v in list_voices() if v.id == vid), ME)


def _slug(name: str) -> str:
    s = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_")[:40] or "voice"
    return f"{s}_{int(time.time())}"


def save_voice(name: str, kind: str, audio: np.ndarray, sr: int, text: str, **extra) -> Voice:
    import soundfile as sf
    v = Voice(id=_slug(name), name=name, kind=kind, text=text, created=time.strftime("%Y-%m-%d %H:%M"), **extra)
    d = voices_dir() / v.id
    d.mkdir(parents=True, exist_ok=True)
    x = dsp.resample(dsp.to_mono(audio), sr, 24000)
    x = dsp.normalize_loudness(x, 24000, -18.0)
    sf.write(str(d / "ref.wav"), x, 24000, subtype="PCM_16")
    (d / "voice.json").write_text(json.dumps(asdict(v), ensure_ascii=False, indent=1), "utf-8")
    return v


def delete_voice(vid: str):
    import shutil
    if vid != "me":
        shutil.rmtree(voices_dir() / vid, ignore_errors=True)


def reference(v: Voice, project=None) -> tuple[np.ndarray, str]:
    """(аудио 24 кГц, текст образца) для клонирования."""
    if v.kind == "me":
        if project is None:
            raise RuntimeError("нет проекта")
        x, text = project.reference_clip()
        if len(x) == 0 or float(np.max(np.abs(x))) < 1e-3:
            raise RuntimeError("В проекте нет записи вашего голоса (открыт SRT без звука) — "
                               "выберите другой голос или откройте SRT вместе с видео/звуком начитки")
        return x, text
    import soundfile as sf
    x, sr = sf.read(str(v.wav), dtype="float32")
    return dsp.resample(dsp.to_mono(x), sr, 24000), v.text


def add_to_etalon(base: Voice, clip: np.ndarray, sr: int, text: str, project=None,
                  max_seconds: float = 15.0) -> Voice:
    """«★ Ориентироваться на этот вариант»: добавляет удачный вариант в образец голоса-эталона.
    Эталон — отдельный голос «<имя> — эталон»; образец = до 3 отмеченных вариантов (≤ 15 с).
    Если получается короче 4 с, в начало добавляется исходный образец голоса."""
    import json as _json
    import soundfile as sf
    from . import stress
    clip = dsp.resample(dsp.to_mono(clip), sr, 24000)
    text = stress.strip(text).strip()
    eid = base.id[:-7] if base.id.endswith("_etalon") else base.id
    root = base if not base.id.endswith("_etalon") else get(eid)
    et_id = f"{eid}_etalon"
    d = voices_dir() / et_id
    parts_file = d / "parts.json"
    parts = []
    if parts_file.exists():
        parts = _json.loads(parts_file.read_text("utf-8"))
    d.mkdir(parents=True, exist_ok=True)
    pf = d / f"part_{int(time.time() * 1000)}.wav"
    sf.write(str(pf), clip, 24000, subtype="PCM_16")
    parts.append({"file": pf.name, "text": text})
    # не больше max_seconds: выкидываем самые старые
    def total(ps):
        return sum(sf.info(str(d / q["file"])).duration for q in ps) + 0.3 * max(0, len(ps) - 1)
    while len(parts) > 1 and (len(parts) > 3 or total(parts) > max_seconds):
        parts.pop(0)
    parts_file.write_text(_json.dumps(parts, ensure_ascii=False, indent=1), "utf-8")
    gap = np.zeros(int(0.3 * 24000), np.float32)
    audio, texts = [], []
    if total(parts) < 4.0:
        try:
            x, t = reference(root, project)
            audio += [x, gap]; texts.append(t)
        except Exception:  # noqa: BLE001
            pass
    for q in parts:
        x, _ = sf.read(str(d / q["file"]), dtype="float32")
        audio += [x, gap]; texts.append(q["text"])
    ref = np.concatenate(audio[:-1]) if audio else clip
    name = f"{root.name if root.kind != 'me' else 'Мой голос'} — эталон"
    v = Voice(id=et_id, name=name, kind="etalon", text=" ".join(texts),
              created=time.strftime("%Y-%m-%d %H:%M"))
    x = dsp.normalize_loudness(ref, 24000, -18.0)
    sf.write(str(d / "ref.wav"), x, 24000, subtype="PCM_16")
    (d / "voice.json").write_text(_json.dumps(asdict(v), ensure_ascii=False, indent=1), "utf-8")
    return v


# ---------------- набор для дообучения ----------------
def dataset_dir() -> Path:
    d = Path(os.environ.get("APPDATA", Path.home())) / "Golos" / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_to_dataset(wav: Path, text: str, lang: str, voice_id: str, tag: str) -> float:
    """Кладёт пару «звук + текст» в набор для дообучения. Возвращает накопленные минуты."""
    import shutil
    from . import stress
    ld = dataset_dir() / lang
    ld.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w-]+", "_", tag, flags=re.UNICODE)[:60] + ".wav"
    shutil.copy(str(wav), str(ld / name))
    meta = dataset_dir() / "metadata.csv"
    line = f"{lang}/{name}|{stress.strip(text).replace('|', ' ')}|{lang}|{voice_id}\n"
    with open(meta, "a", encoding="utf-8") as f:
        f.write(line)
    return dataset_minutes()


def dataset_minutes() -> float:
    import soundfile as sf
    total = 0.0
    for f in dataset_dir().rglob("*.wav"):
        try:
            total += sf.info(str(f)).duration
        except Exception:  # noqa: BLE001
            pass
    return total / 60
