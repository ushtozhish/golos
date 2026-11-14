"""Экспорт: файлы WAV/MP3, субтитры, тайминги и пакет для видеочата (zip)."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np

from .project import Project, SR
from . import dub


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, items: list[dict]):
    lines = []
    for i, it in enumerate(items, 1):
        lines += [str(i), f"{srt_time(it['start'])} --> {srt_time(it['end'])}", it["text"], ""]
    path.write_text("\n".join(lines), "utf-8")


def write_audio(base: Path, x: np.ndarray, sr: int, mp3: bool = True) -> list[Path]:
    import soundfile as sf
    files = [base.with_suffix(".wav")]
    sf.write(str(files[0]), x, sr, subtype="PCM_24")
    if mp3:
        try:
            sf.write(str(base.with_suffix(".mp3")), x, sr, format="MP3")
            files.append(base.with_suffix(".mp3"))
        except Exception:  # noqa: BLE001 — старый libsndfile без MP3
            pass
    return files


def export_source(p: Project, folder: Path) -> tuple[Path, list[dict]]:
    """Смонтированная начитка исходного языка."""
    y = p.render(process=True)
    base = folder / f"{p.source_lang}"
    write_audio(base, y, SR)
    items = [{"id": ph.id, "start": round(ph.start, 3), "end": round(ph.end, 3), "text": ph.text}
             for ph in p.phrases()]
    write_srt(folder / f"{p.source_lang}.srt", items)
    return base.with_suffix(".wav"), items


def export_package(p: Project, langs: list[str], log=print) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M")
    folder = p.folder / "export" / stamp
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"app": "Golos", "project": p.name, "created": stamp,
                "source_lang": p.source_lang, "sample_rate": 48000, "tracks": {}}
    _, src_items = export_source(p, folder)
    manifest["tracks"][p.source_lang] = {"file": f"{p.source_lang}.wav", "kind": "recording",
                                         "duration": round(p.seq_duration(), 3), "phrases": src_items}
    log(f"{p.source_lang}: начитка экспортирована")
    for lang in langs:
        if lang == p.source_lang and not p.dub.get(lang):
            continue
        if not p.dub.get(lang):
            continue
        x, items, warns = dub.build_track(p, lang)
        name = f"{lang}_voice" if lang == p.source_lang else lang
        write_audio(folder / name, x, dub.OUT_SR)
        write_srt(folder / f"{name}.srt", items)
        pdir = folder / f"{name}_phrases"
        pdir.mkdir(exist_ok=True)
        import soundfile as sf
        for it in items:
            a, b = int(it["start"] * dub.OUT_SR), int(it["end"] * dub.OUT_SR)
            sf.write(str(pdir / f"{it['id']:03d}.wav"), x[a:b], dub.OUT_SR, subtype="PCM_24")
        manifest["tracks"][name] = {"file": f"{name}.wav", "kind": "synth", "lang": lang,
                                    "duration": round(len(x) / dub.OUT_SR, 3), "phrases": items,
                                    "warnings": warns}
        for w in warns:
            log(f"[{lang}] {w}")
        log(f"{lang}: дорожка собрана")
    (folder / "timings.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    (folder / "README.txt").write_text(
        "Пакет озвучки Golos.\n"
        "timings.json — для каждой дорожки: фразы с началом и концом (секунды) и текстом.\n"
        "*.wav — 48 кГц / 24 бит, *.srt — субтитры, *_phrases — каждая фраза отдельным файлом.\n",
        "utf-8")
    zpath = folder.parent / f"{p.name}_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix != ".mp3":
                z.write(f, f.relative_to(folder))
    log(f"Пакет: {zpath}")
    return zpath
