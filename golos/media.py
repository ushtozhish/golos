"""Чтение и запись SRT."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _t(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def read_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            m = _TS.search(l)
            if m:
                g = m.groups()
                body = " ".join(x.strip() for x in lines[i + 1:])
                body = re.sub(r"<[^>]+>", "", body).strip()
                if body:
                    cues.append(Cue(len(cues) + 1, _t(*g[:4]), _t(*g[4:]), body))
                break
    return cues


def merge_to_sentences(cues: list[Cue], max_len: float = 12.0) -> list[Cue]:
    """Склеивает мелкие титры (Resolve режет по длине строки) в целые предложения."""
    out: list[Cue] = []
    for c in cues:
        if out:
            p = out[-1]
            open_sentence = not re.search(r"[.!?…]\W*$", p.text)
            if open_sentence and c.start - p.end < 0.9 and c.end - p.start <= max_len:
                out[-1] = Cue(p.index, p.start, c.end, f"{p.text} {c.text}")
                continue
        out.append(Cue(len(out) + 1, c.start, c.end, c.text))
    return out


def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text: str, width: int = 42) -> str:
    """Разбивает длинный титр на две строки по ближайшему к середине пробелу."""
    if len(text) <= width:
        return text
    mid = len(text) // 2
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return text
    i = min(spaces, key=lambda k: abs(k - mid))
    return text[:i] + "\n" + text[i + 1:]


def write_srt(path: str | Path, items: list[dict], wrap_lines: bool = True):
    lines = []
    for i, it in enumerate(items, 1):
        txt = wrap(it["text"]) if wrap_lines else it["text"]
        lines += [str(i), f"{srt_time(it['start'])} --> {srt_time(it['end'])}", txt, ""]
    Path(path).write_text("\n".join(lines), "utf-8")
