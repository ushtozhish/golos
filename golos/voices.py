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
    kind: str              # me | file | design | preset
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
        return project.reference_clip()
    import soundfile as sf
    x, sr = sf.read(str(v.wav), dtype="float32")
    return dsp.resample(dsp.to_mono(x), sr, 24000), v.text
