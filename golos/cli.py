"""Golos без окна — озвучка текстов для других программ (например, YouTube Faceless Studio).

Запуск:  golos-cli.cmd <команда>   (в папке программы)   или   python -m golos.cli <команда>

Команды:
  info                         версия, видеокарта
  voices                       список голосов (JSON) — создаются в окне Golos → «Голоса»
  synth --job job.json --out DIR [--dry-run]
                               озвучить сегменты; на каждый — WAV + JSON с таймингами слов и проверкой
  test-en [--voice ID]         проверка английского диктора на трудном тексте (цифры, имена, термины)

Формат job.json:
  {"lang": "en", "voice": "<id голоса>", "variants": 3, "max_cer": 0.06, "temperature": 0.7,
   "pronounce": {"Nvidia": "en-VID-ee-uh"},
   "segments": [{"id": "seg_01", "text": "In 2008, a ship's anchor cut one cable."}]}

Что получается в DIR для каждого сегмента:
  <id>.wav   48 кГц, 24 бит, стерео, громкость −18 LUFS, 0,08 с тишины в начале и 0,3 с в конце
  <id>.json  {"id", "text", "spoken_text", "duration_sec", "cer", "status": "ok" | "needs_fix",
              "words": [{"word": исходное слово, "start_sec", "end_sec"}], "problems": [...]}
  result.json — сводка по всем сегментам.
Ход работы печатается в stdout строками JSON: {"event": "progress", ...}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

OUT_SR = 48000
LEAD_S = 0.08
TAIL_S = 0.30


def emit(event: str, **kw):
    print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)


# ============================================================ числа → слова (английский)
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
              "nine": "ninth", "twelve": "twelfth"}
_SUFFIX = {"k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion", "t": "trillion",
           "tn": "trillion"}


def int_words(n: int) -> str:
    if n < 0:
        return "minus " + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + int_words(r) if r else "")
    for val, name in _SCALES:
        if n >= val:
            q, r = divmod(n, val)
            return int_words(q) + " " + name + (" " + int_words(r) if r else "")
    return str(n)


def year_words(n: int) -> str:
    if 2000 <= n <= 2009:
        return "two thousand" + (" " + _ONES[n - 2000] if n > 2000 else "")
    hi, lo = divmod(n, 100)
    if lo == 0:
        return int_words(hi) + " hundred"
    return int_words(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else int_words(lo))


def ordinal_words(n: int) -> str:
    w = int_words(n)
    head, _, last = w.rpartition(" ")
    last_parts = last.split("-")
    tail = last_parts[-1]
    if tail in _ORD_IRREG:
        tail = _ORD_IRREG[tail]
    elif tail.endswith("y"):
        tail = tail[:-1] + "ieth"
    else:
        tail += "th"
    last_parts[-1] = tail
    last = "-".join(last_parts)
    return (head + " " + last).strip()


def number_words(num: str) -> str:
    """'1,250' → one thousand two hundred fifty; '3.5' → three point five; '2008' → two thousand eight."""
    had_comma = "," in num
    num = num.replace(",", "")
    if "." in num:
        a, b = num.split(".", 1)
        return int_words(int(a or 0)) + " point " + " ".join(_ONES[int(d)] for d in b if d.isdigit())
    n = int(num)
    if 1100 <= n <= 2099 and len(num) == 4 and not had_comma:
        return year_words(n)
    return int_words(n)


_TOKEN = re.compile(
    r"^(?P<pre>[^\w$€£]*)"
    r"(?P<cur>[$€£])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suf>(?:st|nd|rd|th|s|%|k|m|mn|b|bn|t|tn)?)"
    r"(?P<post>[^\w]*)$",
    re.IGNORECASE,
)
_CUR = {"$": "dollars", "€": "euros", "£": "pounds"}


def spell_token(tok: str) -> str:
    """Один токен текста → как его произнести. Не-числа возвращаются как есть."""
    m = _TOKEN.match(tok)
    if not m:
        if re.fullmatch(r"\W*\d+s\W*", tok):  # 1990s
            return tok
        return tok
    pre, cur, num, suf, post = m.group("pre"), m.group("cur"), m.group("num"), (m.group("suf") or ""), m.group("post")
    s = suf.lower()
    if s in ("st", "nd", "rd", "th") and "." not in num:
        words = ordinal_words(int(num.replace(",", "")))
    elif s == "s" and len(num) == 4:            # 1990s → nineteen nineties
        y = year_words(int(num))
        words = y[:-1] + "ies" if y.endswith("y") else y + "s"
    else:
        words = number_words(num)
        if s in _SUFFIX:
            words += " " + _SUFFIX[s]
        if s == "%":
            words += " percent"
    if cur:
        words += " " + _CUR[cur]
    return f"{pre}{words}{post}"


def build_spoken(text: str, pronounce: dict[str, str]) -> tuple[str, list[tuple[str, str, str]]]:
    """→ (текст для модели, [(исходный токен, произносимый текст, текст для сверки)])."""
    pr = {k.lower(): v for k, v in (pronounce or {}).items()}
    tokens = text.split()
    out, cmap = [], []
    for tok in tokens:
        spoken = spell_token(tok)
        core = re.sub(r"^\W+|\W+$", "", spoken)
        compare = spoken
        if core.lower() in pr:
            spoken = spoken.replace(core, pr[core.lower()])
        out.append(spoken)
        cmap.append((tok, spoken, compare))
    return " ".join(out), cmap


_CUR_SCALE = re.compile(r"([$€£])\s?(\d[\d,.]*)\s+(thousand|million|billion|trillion)\b", re.IGNORECASE)


def norm_words(s: str) -> list[str]:
    """Для сверки: нижний регистр, числа словами, без знаков.
    «$10 trillion» (как пишет Whisper) приводится к «10 trillion dollars» (как произносится)."""
    s = _CUR_SCALE.sub(lambda m: f"{m.group(2)} {m.group(3)} {_CUR[m.group(1)]}", s or "")
    s = re.sub(r"(?<!\w)([$€£])(?=\s|$)", lambda m: _CUR[m.group(1)], s)
    res = []
    for tok in s.split():
        sp = spell_token(tok).lower().replace("-", " ")
        sp = re.sub(r"[^\w\s']", " ", sp, flags=re.UNICODE).replace("'", "")
        res.extend(w for w in sp.split() if w)
    return res


def cer(ref_words: list[str], hyp_words: list[str]) -> float:
    a, b = "".join(ref_words), "".join(hyp_words)
    if not a:
        return 0.0
    return round(1.0 - SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)


def problems_of(ref_words: list[str], hyp_words: list[str]) -> list[str]:
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref_words, hyp_words, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        want, got = " ".join(ref_words[i1:i2]), " ".join(hyp_words[j1:j2])
        if tag == "delete":
            out.append(f"пропущено: «{want}»")
        elif tag == "insert":
            out.append(f"лишнее: «{got}»")
        else:
            out.append(f"«{want}» → услышано «{got}»")
    return out[:12]


def align_token_times(cmap, wh_words: list[tuple[str, float, float]], total: float) -> list[dict]:
    """Тайминги исходных токенов по словам Whisper (сопоставление последовательностей слов)."""
    exp, owner = [], []
    for ti, (_, _, compare) in enumerate(cmap):
        ws = norm_words(compare) or ["_"]
        exp.extend(ws)
        owner.extend([ti] * len(ws))
    got, gt = [], []
    for text, s, e in wh_words:
        ws = norm_words(text)
        if not ws:
            continue
        step = (e - s) / len(ws)
        for k, w in enumerate(ws):
            got.append(w)
            gt.append((s + k * step, s + (k + 1) * step))
    times: list[tuple[float, float] | None] = [None] * len(exp)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, exp, got, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                times[i1 + k] = gt[j1 + k]
    # пропуски — интерполяция между соседями
    known = [i for i, t in enumerate(times) if t is not None]
    for i in range(len(times)):
        if times[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        a = times[prev][1] if prev is not None else 0.0
        b = times[nxt][0] if nxt is not None else total
        lo = prev if prev is not None else -1
        hi = nxt if nxt is not None else len(times)
        frac0 = (i - lo) / (hi - lo)
        frac1 = (i - lo + 1) / (hi - lo)
        times[i] = (a + (b - a) * frac0 * 0.999, a + (b - a) * frac1 * 0.999)
    out = []
    for ti, (orig, _, _) in enumerate(cmap):
        idx = [k for k, o in enumerate(owner) if o == ti]
        s = min(times[k][0] for k in idx)
        e = max(times[k][1] for k in idx)
        out.append({"word": orig, "start_sec": round(s, 3), "end_sec": round(max(e, s + 0.02), 3)})
    # монотонность
    for k in range(1, len(out)):
        if out[k]["start_sec"] < out[k - 1]["start_sec"]:
            out[k]["start_sec"] = out[k - 1]["start_sec"]
        if out[k]["end_sec"] < out[k]["start_sec"]:
            out[k]["end_sec"] = out[k]["start_sec"] + 0.02
    return out


# ============================================================ синтез
def _write_wav(path: Path, x48: np.ndarray):
    import soundfile as sf
    st = np.stack([x48, x48], axis=1)
    sf.write(str(path), st, OUT_SR, subtype="PCM_24")


class Engine:
    """Модели держатся в видеопамяти между сегментами."""

    def __init__(self, lang: str, voice_id: str, log, precision: str = "fp16", dry_run: bool = False):
        self.lang, self.log, self.dry = lang, log, dry_run
        self.precision = precision
        if dry_run:
            self.voice_name = voice_id or "dry-run"
            return
        from . import voices
        from .models import TTS_LANGS
        if lang not in TTS_LANGS:
            raise SystemExit(f"Язык «{lang}» не поддерживается. Доступны: {', '.join(TTS_LANGS)}")
        v = voices.get(voice_id) if voice_id else None
        if v is None or v.id == "me":
            raise SystemExit("Укажите голос (voice) из списка `golos-cli voices`. Голос «из начитки» без окна недоступен.")
        self.voice = v
        self.voice_name = v.name
        self.repo, self.lname, _ = TTS_LANGS[lang]
        self.trained = v.kind == "trained"
        if self.trained:
            self.repo = v.description
        else:
            self.ref = voices.reference(v)

    def _tts(self):
        from .models import get_tts
        tts = get_tts(self.repo, self.log, self.precision)
        if not self.trained:
            tts.set_reference(*self.ref)
        return tts

    def generate(self, text: str, n: int, temperature: float, max_s: float) -> tuple[list[np.ndarray], int]:
        if self.dry:
            sr = 24000
            dur = max(0.6, len(text.split()) * 0.32)
            t = np.arange(int(dur * sr)) / sr
            return [(0.2 * np.sin(2 * np.pi * (180 + 20 * k) * t)).astype(np.float32) for k in range(n)], sr
        tts = self._tts()
        if self.lang == "ru":
            from . import stress
            text = stress.for_model(text, "ru", "acute")
        try:
            if self.trained:
                return tts.custom_many([text] * n, self.lname, self.voice.speaker, temperature, max_s)
            return tts.generate([text] * n, self.lname, temperature, max_seconds=max_s)
        except (FloatingPointError, RuntimeError) as e:
            if self.precision == "fp32":
                raise
            self.log(f"fp16 дал сбой ({e}); перезагружаю модель в fp32")
            from .models import release_tts
            release_tts()
            self.precision = "fp32"
            return self.generate(text, n, temperature, max_s)

    def words(self, x48: np.ndarray, compare_text: str) -> list[tuple[str, float, float]]:
        if self.dry:
            ws = compare_text.split()
            dur = len(x48) / OUT_SR
            step = dur / max(1, len(ws))
            return [(w, i * step, (i + 0.8) * step) for i, w in enumerate(ws)]
        from .models import get_whisper
        return get_whisper(self.log).words(x48, OUT_SR, self.lang)


def synth_segment(eng: Engine, seg: dict, job: dict, out_dir: Path) -> dict:
    from . import dsp
    from .dub import trim_silence
    sid = str(seg["id"])
    text = str(seg["text"]).strip()
    pron = dict(_load_pron_en() if eng.lang == "en" else {})
    pron.update(job.get("pronounce") or {})
    if eng.lang == "en":
        spoken, cmap = build_spoken(text, pron)
    else:
        spoken, cmap = text, [(t, t, t) for t in text.split()]
    compare_text = " ".join(c for _, _, c in cmap)
    ref_words = norm_words(compare_text)
    n = int(seg.get("variants", job.get("variants", 3)))
    temp = float(job.get("temperature", 0.7))
    max_cer = float(job.get("max_cer", 0.06))
    rounds = int(job.get("rounds", 2))
    max_s = max(6.0, len(spoken.split()) * 0.65 + 4.0)

    best = None
    tried = 0
    for rnd in range(rounds):
        wavs, sr = eng.generate(spoken, n, temp, max_s)
        for w in wavs:
            tried += 1
            x = dsp.resample(np.asarray(w, np.float32), sr, OUT_SR)
            x = trim_silence(x, OUT_SR, pad=0.03)
            if len(x) < int(0.2 * OUT_SR):
                continue
            ww = eng.words(x, compare_text)
            hyp = norm_words(" ".join(t for t, _, _ in ww))
            c = cer(ref_words, hyp)
            dur = len(x) / OUT_SR
            # при равной точности — без затянутых пауз (короче)
            score = c * 10 + dur * 0.01
            if best is None or score < best["score"]:
                best = {"score": score, "cer": c, "x": x, "ww": ww, "hyp": hyp}
        emit("progress", segment=sid, round=rnd + 1, best_cer=None if best is None else best["cer"])
        if best is not None and best["cer"] <= max_cer:
            break
        temp = max(0.45, temp - 0.12)
    if best is None:
        raise RuntimeError(f"Сегмент {sid}: модель не выдала звук")

    x = best["x"]
    if not eng.dry:
        x = dsp.normalize_loudness(x, OUT_SR, -18.0)
    x = np.concatenate([np.zeros(int(LEAD_S * OUT_SR), np.float32), x.astype(np.float32),
                        np.zeros(int(TAIL_S * OUT_SR), np.float32)])
    wav = out_dir / f"{sid}.wav"
    _write_wav(wav, x)
    total = len(x) / OUT_SR
    shifted = [(t, s + LEAD_S, e + LEAD_S) for t, s, e in best["ww"]]
    words = align_token_times(cmap, shifted, total)
    status = "ok" if best["cer"] <= max_cer else "needs_fix"
    meta = {
        "id": sid, "text": text, "spoken_text": spoken, "lang": eng.lang, "voice": eng.voice_name,
        "duration_sec": round(total, 3), "cer": best["cer"], "max_cer": max_cer, "status": status,
        "variants_tried": tried, "words": words,
        "problems": [] if status == "ok" else problems_of(ref_words, best["hyp"]),
        "heard": " ".join(best["hyp"]),
    }
    (out_dir / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return meta


def _load_pron_en() -> dict[str, str]:
    from . import stress
    p = stress.dict_path().with_name("pronounce_en.json")
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def run_job(job: dict, out_dir: Path, dry_run: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lang = job.get("lang", "en")
    log = lambda m: emit("log", message=m)  # noqa: E731
    t0 = time.time()
    eng = Engine(lang, job.get("voice", ""), log, job.get("precision", "fp16"), dry_run=dry_run)
    if lang == "kk" and not dry_run:
        emit("warning", message="Казахская модель AIT-Syn — лицензия CC BY-NC 4.0: нельзя для монетизируемых роликов.")
    results = []
    segs = job.get("segments") or []
    for i, seg in enumerate(segs, start=1):
        emit("progress", segment=str(seg.get("id")), index=i, total=len(segs), stage="start")
        try:
            meta = synth_segment(eng, seg, job, out_dir)
        except Exception as e:  # noqa: BLE001
            meta = {"id": str(seg.get("id")), "status": "error", "error": f"{type(e).__name__}: {e}"}
            emit("error", segment=meta["id"], message=meta["error"])
        results.append({k: meta.get(k) for k in ("id", "status", "cer", "duration_sec", "problems", "error")})
        emit("segment_done", **results[-1])
    from . import __version__
    summary = {"golos_version": __version__, "lang": lang, "voice": eng.voice_name,
               "seconds": round(time.time() - t0, 1),
               "ok": sum(r["status"] == "ok" for r in results),
               "needs_fix": sum(r["status"] == "needs_fix" for r in results),
               "errors": sum(r["status"] == "error" for r in results),
               "segments": results}
    (out_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    emit("done", ok=summary["ok"], needs_fix=summary["needs_fix"], errors=summary["errors"],
         result=str(out_dir / "result.json"))
    return summary


# ============================================================ проверка английского диктора
TEST_EN = [
    "In 2008, a single ship's anchor cut one undersea cable and knocked 75 million people offline.",
    "Today, more than 1.4 million kilometers of fiber carry about 99% of the world's internet traffic.",
    "Nvidia, TSMC and Qualcomm depend on a supply chain that runs through Taiwan, Malaysia and the Netherlands.",
    "By the 1990s, global debt was rising fast; in 2023 it passed $300T, roughly 3.5 times world GDP.",
    "Nobody planned for the 3rd failure. That is the part the official report never explains.",
]


def _pick_voice() -> str:
    from . import voices
    vs = [v for v in voices.list_voices() if v.id != "me"]
    if not vs:
        raise SystemExit("Голосов нет. Откройте Golos → «Голоса» → создайте английского диктора "
                         "(«Готовый диктор Qwen», например Ryan, язык English) и запустите проверку снова.")
    print("\nГолоса:")
    for i, v in enumerate(vs, 1):
        print(f"  {i}. {v.name}  ({v.kind})")
    while True:
        s = input("Номер голоса для проверки: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(vs):
            return vs[int(s) - 1].id


def test_en(voice: str | None, dry_run: bool = False):
    vid = voice or ("dry" if dry_run else _pick_voice())
    out = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / "Golos" / f"test-en-{time.strftime('%Y%m%d-%H%M%S')}"
    job = {"lang": "en", "voice": vid, "variants": 3, "max_cer": 0.06,
           "segments": [{"id": f"test_{i:02d}", "text": t} for i, t in enumerate(TEST_EN, 1)]}
    print(f"\nОзвучиваю {len(TEST_EN)} трудных фраз… Первый запуск скачивает модели (~8 ГБ) — это долго.\n")
    s = run_job(job, out, dry_run=dry_run)
    print("\n=== Итог ===")
    for r in s["segments"]:
        mark = {"ok": "ОК", "needs_fix": "ОШИБКИ", "error": "СБОЙ"}.get(r["status"], r["status"])
        print(f"  {r['id']}: {mark}  ошибок {100 * (r['cer'] or 0):.1f}%  {', '.join(r.get('problems') or [])}")
    print(f"\nФайлы: {out}\nПослушайте test_01.wav … test_05.wav. Сравнить с Edge можно в студии (A/B).")
    if os.name == "nt" and not dry_run:
        os.startfile(str(out))  # noqa: S606


# ============================================================ main
def main(argv=None):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="golos-cli", description="Golos без окна")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("voices")
    s = sub.add_parser("synth")
    s.add_argument("--job", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dry-run", action="store_true", help="без моделей: проверка формата (тон вместо голоса)")
    mv = sub.add_parser("make-voice", help="создать голос без окна: готовый диктор Qwen или по описанию")
    mv.add_argument("--name", required=True)
    mv.add_argument("--preset", help="диктор Qwen, например Ryan")
    mv.add_argument("--design", help="описание голоса по-английски")
    mv.add_argument("--style", default="")
    mv.add_argument("--lang", default="en")
    mv.add_argument("--sample-text", default="", help="фраза образца — задаёт манеру голоса (серьёзная/весёлая)")
    t = sub.add_parser("test-en")
    t.add_argument("--voice")
    t.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        from . import __version__
        from .models import torch_info
        print(json.dumps({"golos_version": __version__, "device": torch_info()}, ensure_ascii=False))
    elif a.cmd == "voices":
        from . import voices
        print(json.dumps([{"id": v.id, "name": v.name, "kind": v.kind, "speaker": v.speaker}
                          for v in voices.list_voices() if v.id != "me"], ensure_ascii=False, indent=1))
    elif a.cmd == "synth":
        job = json.loads(Path(a.job).read_text("utf-8-sig"))
        s = run_job(job, Path(a.out), dry_run=a.dry_run)
        sys.exit(0 if s["errors"] == 0 else 3)
    elif a.cmd == "make-voice":
        from . import models, voices
        if not (a.preset or a.design):
            raise SystemExit("Укажите --preset ДИКТОР или --design \"описание\"")
        log = lambda m: emit("log", message=m)  # noqa: E731
        lname = models.TTS_LANGS[a.lang][1]
        sample = a.sample_text.strip() or voices.SAMPLE_TEXT.get(a.lang, voices.SAMPLE_TEXT["en"])
        repo = models.TTS_PRESET_ID if a.preset else models.TTS_DESIGN_ID
        m = models.TTS(repo, log)
        try:
            if a.preset:
                x, sr = m.preset(sample, lname, a.preset, a.style)
                v = voices.save_voice(a.name, "preset", x, sr, sample, speaker=a.preset)
            else:
                x, sr = m.design(sample, lname, a.design)
                v = voices.save_voice(a.name, "design", x, sr, sample, description=a.design)
        finally:
            m.close()
        emit("voice_created", id=v.id, name=v.name)
    elif a.cmd == "test-en":
        try:
            test_en(a.voice, dry_run=a.dry_run)
        finally:
            if os.name == "nt" and not a.dry_run and sys.stdin and sys.stdin.isatty():
                input("\nНажмите Enter, чтобы закрыть окно")


if __name__ == "__main__":
    main()
