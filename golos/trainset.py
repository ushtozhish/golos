"""Набор для обучения своего голоса: фразы по блокам, дубли, автопроверка, экспорт пакета обучения,
подключение обученной модели.

Хранение: %APPDATA%\\Golos\\trainset\\<блок>\\<номер>_<время>.wav + takes.json
"""
from __future__ import annotations

import json
import os
import re
import time
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from . import dsp

SR = 48000
TRAIN_SR = 24000
BLOCK_NAMES = {
    "А": "А. Все звуки русского (нейтрально)",
    "Б": "Б. Интонации",
    "В": "В. Числа и английские слова",
    "Г1": "Г-1. Бодрая рекламная подача",
    "Г2": "Г-2. Спокойная подача",
    "Г3": "Г-3. Доверительная подача",
    "Г4": "Г-4. Деловая подача",
}


@dataclass
class Line:
    block: str
    no: int
    text: str        # что произносить
    hint: str = ""   # подсказка (например, какое слово выделить)
    emph: list = field(default_factory=list)   # слова, которые выделить голосом


@dataclass
class Take:
    file: str
    dur: float
    peak: float = 0.0
    level_db: float = -99.0
    snr_db: float = 0.0
    cer: float | None = None
    heard: str = ""
    ok: bool | None = None       # None — ещё не проверен
    problems: list[str] = field(default_factory=list)
    created: str = ""


# Инструкции из документа «Текст для записи своего голоса»
GENERAL_TIPS = [
    ("Комната", "Тихая, с мягкими вещами (шторы, диван, шкаф с одеждой). Без эха и гула кондиционера."),
    ("Микрофон", "Один и тот же во всех сессиях, на одном расстоянии: 15–20 см, чуть сбоку от рта."),
    ("Громкость", "Пики около −12 dB, без перегруза. Уровень не меняйте между сессиями."),
    ("Темп", "Обычный темп диктора, не медленнее. Читайте так, как хотите, чтобы звучала озвучка."),
    ("Одна фраза", "Одно предложение = один дубль. Паузы ~1 с до и после, без вдоха в самом начале."),
    ("Ошибка", "Остановите запись и запишите новый дубль целиком. Неудачный удалите или оставьте — "
               "в обучение идут только зелёные."),
    ("Сессии", "20–30 минут, потом перерыв: уставший голос садится, и модель запомнит усталость."),
    ("Самочувствие", "Не записывайте больным или после крика: модель выучит хрипоту."),
    ("Настроение", "Читайте блок одним настроением. Смешивать подачи внутри блока не нужно — "
                   "у каждой подачи свой блок Г."),
]

BLOCK_HOW = {
    "А": "Нейтрально, спокойно, как диктор новостей. Здесь собраны твёрдые и мягкие согласные, ж, ш, щ, ц, ч, й, "
         "ы и и, длинные согласные, скопления и безударные гласные.",
    "Б": "Здесь важна мелодика фразы, а не звуки. Читайте естественно, как сказали бы живому человеку.",
    "В": "Числа читайте словами так, как они должны звучать в ролике. Английские слова — так, как вы их реально "
         "произносите в русской речи: именно так модель и научится их читать.",
    "Г1": "Энергия, улыбка, темп чуть быстрее. Весь блок — в одном настроении.",
    "Г2": "Ровно, мягко, без спешки. Весь блок — в одном настроении.",
    "Г3": "Как другу, близко, чуть тише. Весь блок — в одном настроении.",
    "Г4": "Чётко, уверенно, как на презентации. Весь блок — в одном настроении.",
}

SECTIONS = [  # (первая, последняя фраза, раздел, как читать)
    (71, 78, "Общий вопрос", "Вопрос «да/нет»: тон поднимается на главном слове."),
    (79, 86, "Вопрос с вопросительным словом", "Тон — на вопросительном слове (где, как, почему…)."),
    (87, 94, "Восклицание", "С живой эмоцией, как в разговоре."),
    (95, 102, "Перечисления", "Тон держится на каждом пункте и опускается на последнем."),
    (103, 110, "Логическое ударение", "Одна фраза — разный смысл. Выделенное слово произносите с нажимом."),
    (111, 118, "Длинные фразы с паузами", "Запятая — короткая пауза, тире — длиннее."),
    (119, 130, "Короткие реплики", "Коротко и живо, как ответ собеседнику."),
    (131, 145, "Числа и даты", "Числа словами, как они должны звучать в ролике."),
    (146, 150, "Аббревиатуры", "Так, как произносите вслух (эм-пэ-четыре, кью-ар-код, пэ-дэ-эф)."),
    (151, 170, "Английские слова в русской речи", "Как реально говорите их в русской речи — так модель и научится."),
]


def section_of(no: int):
    for a, b, title, how in SECTIONS:
        if a <= no <= b:
            return title, how
    return "", ""


def load_script() -> list[Line]:
    out = []
    for raw in Path(__file__).with_name("trainscript.txt").read_text("utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        block, no, text = raw.split("|", 2)
        hint, emph = "", []
        m = re.search(r"\s*\(выделить:\s*([^)]+)\)\s*$", text)
        if m:
            emph = [w.strip() for w in m.group(1).split(";") if w.strip()]
            hint = "Выделите голосом: " + ", ".join("«" + w + "»" for w in emph)
            text = text[:m.start()]
        out.append(Line(block, int(no), text.strip(), hint, emph))
    return out


def root() -> Path:
    d = Path(os.environ.get("APPDATA", Path.home())) / "Golos" / "trainset"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TrainSet:
    def __init__(self):
        self.lines = load_script()
        self.by_no = {l.no: l for l in self.lines}
        self.takes: dict[int, list[Take]] = {}
        self.noise_db: float | None = None
        self.load()

    @property
    def meta(self) -> Path:
        return root() / "takes.json"

    def load(self):
        if self.meta.exists():
            s = json.loads(self.meta.read_text("utf-8"))
            self.takes = {int(k): [Take(**t) for t in v] for k, v in s.get("takes", {}).items()}
            self.noise_db = s.get("noise_db")

    def save(self):
        tmp = self.meta.with_suffix(".tmp")
        tmp.write_text(json.dumps({"takes": {str(k): [asdict(t) for t in v] for k, v in self.takes.items()},
                                   "noise_db": self.noise_db}, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.meta)

    def path(self, no: int, take: Take) -> Path:
        return root() / self.by_no[no].block / take.file

    # ----- запись дубля -----
    def add_take(self, no: int, x: np.ndarray, sr: int) -> Take:
        line = self.by_no[no]
        x = dsp.resample(dsp.to_mono(x), sr, SR)
        x = trim(x, SR)
        d = root() / line.block
        d.mkdir(parents=True, exist_ok=True)
        name = f"{no:03d}_{int(time.time() * 1000)}.wav"
        import soundfile as sf
        sf.write(str(d / name), x, SR, subtype="PCM_24")
        t = Take(file=name, dur=len(x) / SR, created=time.strftime("%Y-%m-%d %H:%M"))
        acoustic_check(t, x, SR, self.noise_db)
        self.takes.setdefault(no, []).append(t)
        self.save()
        return t

    def delete_take(self, no: int, k: int):
        lst = self.takes.get(no, [])
        if 0 <= k < len(lst):
            try:
                self.path(no, lst[k]).unlink()
            except OSError:
                pass
            lst.pop(k)
            self.save()

    # ----- статистика -----
    def good(self, no: int) -> list[Take]:
        return [t for t in self.takes.get(no, []) if t.ok]

    def stats(self) -> dict[str, dict]:
        out = {}
        for b in BLOCK_NAMES:
            lines = [l for l in self.lines if l.block == b]
            done = sum(1 for l in lines if self.good(l.no))
            mins = sum(t.dur for l in lines for t in self.good(l.no)) / 60
            out[b] = {"lines": len(lines), "done": done, "minutes": mins}
        return out

    def total_minutes(self) -> float:
        return sum(v["minutes"] for v in self.stats().values())


def trim(x: np.ndarray, sr: int, pad: float = 0.25) -> np.ndarray:
    """Обрезает тишину по краям, оставляя по pad секунд."""
    rms = dsp.rms_frames(x, sr, 0.01)
    m = dsp.speech_mask(rms)
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return x
    a = max(0, int((idx[0] * 0.01 - pad) * sr))
    b = min(len(x), int(((idx[-1] + 1) * 0.01 + pad) * sr))
    return x[a:b]


def acoustic_check(t: Take, x: np.ndarray, sr: int, noise_db: float | None):
    """Перегруз, громкость, шум (отношение сигнал/шум)."""
    t.problems = []
    t.peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = dsp.rms_frames(x, sr, 0.02)
    db = 20 * np.log10(rms + 1e-9)
    speech = db[db > np.percentile(db, 60)] if len(db) else db
    t.level_db = float(np.median(speech)) if len(speech) else -99.0
    floor = noise_db if noise_db is not None else float(np.percentile(db, 5)) if len(db) else -90.0
    t.snr_db = t.level_db - floor
    if t.peak > 0.985:
        t.problems.append("перегруз (клиппинг) — отодвиньтесь или убавьте усиление")
    if t.level_db < -38:
        t.problems.append("слишком тихо — придвиньтесь к микрофону")
    if t.snr_db < 25:
        t.problems.append(f"шумно: сигнал/шум {t.snr_db:.0f} дБ (нужно от 25)")
    if t.dur < 0.4:
        t.problems.append("слишком коротко")


def text_check(t: Take, expected: str, heard: str, lenient: bool = False):
    """lenient — для фраз с числами и английскими словами: распознавание пишет их цифрами/по-своему."""
    from .dub import cer
    t.heard = heard
    t.cer = round(cer(expected, heard), 3)
    if t.cer > (0.4 if lenient else 0.12):
        t.problems = [p for p in t.problems if not p.startswith("текст")]
        t.problems.append(f"текст не совпал на {t.cer * 100:.0f}% — услышано: «{heard}»")
    t.ok = not t.problems


def measure_noise(x: np.ndarray, sr: int) -> float:
    rms = dsp.rms_frames(x, sr, 0.02)
    return float(np.median(20 * np.log10(rms + 1e-9)))


# ----- пакет для обучения -----
RUN_SH = r'''#!/usr/bin/env bash
# Обучение голоса Golos на арендованной видеокарте (Linux, NVIDIA 24+ ГБ, Ampere и новее).
set -e
cd "$(dirname "$0")"
pip install -U qwen-tts accelerate tensorboard soundfile
[ -d Qwen3-TTS ] || git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git
cp -r Qwen3-TTS/finetuning/*.py .
python prepare_data.py --device cuda:0 --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl
python sft_12hz.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_model_path output \
  --train_jsonl train_with_codes.jsonl --batch_size 2 --lr 2e-5 --num_epochs 3 --speaker_name __SPEAKER__
python - <<'PY'
import glob, os, shutil
for d in sorted(glob.glob("output/checkpoint-epoch-*")):
    shutil.make_archive("golos_voice_" + os.path.basename(d), "zip", "output", os.path.basename(d))
PY
echo "ГОТОВО. Скачайте файл golos_voice_checkpoint-epoch-2.zip и подключите его в Golos (вкладка 4, шаг 5)."
'''

README = """Пакет обучения голоса Golos
==========================

1. Арендуйте видеокарту NVIDIA с 24+ ГБ памяти (RTX 3090/4090, A5000, A6000, L40S, A100) с образом PyTorch (Linux).
2. Загрузите этот zip на машину и распакуйте:  unzip golos_trainset.zip -d golos && cd golos
3. Запустите:  bash run_train.sh
   (подготовка данных ~10 мин, обучение 20–60 мин в зависимости от карты и объёма)
4. Скачайте получившийся golos_voice_checkpoint-epoch-2.zip (последняя эпоха) и выключите машину.
5. В Golos: вкладка «4. Обучение голоса» → «Подключить обученную модель…» → выберите этот zip.

Файлы: data/*.wav (24 кГц, моно), train_raw.jsonl (audio, text, ref_audio), ref.wav — образец голоса.
Тексты размечены знаком ударения над каждой ударной гласной (замо́к): после обучения модель слушается этого знака,
и Golos при озвучке обученным голосом расставляет его так же.
Скрипты обучения — официальные из репозитория QwenLM/Qwen3-TTS (папка finetuning).
"""


def build_ref(ts: TrainSet, seconds: float = 9.0):
    """Образец голоса из 2–3 лучших дублей (спокойная подача или блок А): (аудио 24 кГц, текст)."""
    import soundfile as sf
    items = [(l, t) for l in ts.lines for t in ts.good(l.no)]
    pref = [it for it in items if it[0].block == "Г2"] or [it for it in items if it[0].block == "А"] or items
    parts, texts, total = [], [], 0.0
    for line, t in sorted(pref, key=lambda it: -it[1].snr_db):
        if total > seconds:
            break
        x, sr = sf.read(str(ts.path(line.no, t)), dtype="float32")
        parts += [dsp.resample(x, sr, TRAIN_SR), np.zeros(int(0.3 * TRAIN_SR), np.float32)]
        texts.append(line.text)
        total += t.dur
    if not parts:
        return None, ""
    return np.concatenate(parts[:-1]), " ".join(texts)


def export_package(ts: TrainSet, speaker: str, log=lambda m: None, accent: bool = True) -> Path:
    import soundfile as sf
    out_dir = root().parent / "trainset_export"
    out_dir.mkdir(exist_ok=True)
    zpath = out_dir / f"golos_trainset_{time.strftime('%Y-%m-%d_%H-%M')}.zip"
    items, ref_parts, ref_text = [], [], []
    for line in ts.lines:
        for k, t in enumerate(ts.good(line.no)):
            items.append((line, k, t))
    if not items:
        raise RuntimeError("Нет проверенных дублей")
    # образец голоса: 2–3 хороших дубля спокойной подачи (или блока А), ~10 с
    pref = [it for it in items if it[0].block == "Г2"] or [it for it in items if it[0].block == "А"] or items
    total = 0.0
    for line, k, t in sorted(pref, key=lambda it: -it[2].snr_db):
        if total > 9:
            break
        x, sr = sf.read(str(ts.path(line.no, t)), dtype="float32")
        ref_parts += [dsp.resample(x, sr, TRAIN_SR), np.zeros(int(0.3 * TRAIN_SR), np.float32)]
        ref_text.append(line.text)
        total += t.dur
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        ref = np.concatenate(ref_parts[:-1])
        tmp = out_dir / "_ref.wav"
        sf.write(str(tmp), ref, TRAIN_SR, subtype="PCM_16")
        z.write(tmp, "ref.wav"); tmp.unlink()
        lines = []
        for i, (line, k, t) in enumerate(items):
            x, sr = sf.read(str(ts.path(line.no, t)), dtype="float32")
            x = dsp.normalize_loudness(dsp.resample(x, sr, TRAIN_SR), TRAIN_SR, -20.0)
            name = f"data/{line.no:03d}_{k + 1}.wav"
            tmp = out_dir / "_x.wav"
            sf.write(str(tmp), x, TRAIN_SR, subtype="PCM_16")
            z.write(tmp, name); tmp.unlink()
            text = line.text
            if accent:            # ударение в каждом слове: модель учится слушаться знака ударения
                from . import stress
                text = stress.for_model(stress.accent_full(text, log), "ru", "acute")
            lines.append(json.dumps({"audio": "./" + name, "text": text, "ref_audio": "./ref.wav"},
                                    ensure_ascii=False))
            if i % 20 == 0:
                log(f"Пакет: {i + 1}/{len(items)} дублей")
        z.writestr("train_raw.jsonl", "\n".join(lines) + "\n")
        z.writestr("run_train.sh", RUN_SH.replace("__SPEAKER__", speaker))
        z.writestr("README_RU.txt", README)
        z.writestr("ref_text.txt", " ".join(ref_text))
    log(f"Пакет обучения: {len(items)} дублей, {sum(t.dur for _, _, t in items) / 60:.1f} мин → {zpath}")
    return zpath


def install_trained(zip_or_dir: Path, name: str, speaker: str, log=lambda m: None):
    """Распаковывает обученную модель в папку моделей и создаёт голос «обученный»."""
    import shutil
    from .models import models_dir
    from . import voices
    target = models_dir() / f"trained--{re.sub(r'[^\w-]+', '_', name)}_{int(time.time())}"
    target.mkdir(parents=True)
    src = Path(zip_or_dir)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(target)
    else:
        shutil.copytree(src, target, dirs_exist_ok=True)
    # папка с config.json — сама модель
    cfg = next(target.rglob("config.json"), None)
    if cfg is None:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError("В архиве нет модели (не найден config.json)")
    model_dir = cfg.parent
    (model_dir / ".golos_complete").write_text("ok")
    ref, ref_text = build_ref(TrainSet())
    v = voices.Voice(id=f"trained_{int(time.time())}", name=name, kind="trained", speaker=speaker,
                     description=str(model_dir), text=ref_text, created=time.strftime("%Y-%m-%d %H:%M"))
    d = voices.voices_dir() / v.id
    d.mkdir(parents=True, exist_ok=True)
    if ref is not None:          # образец — для языков, которые обученная модель не знает (казахский)
        import soundfile as sf
        sf.write(str(d / "ref.wav"), ref, TRAIN_SR, subtype="PCM_16")
    (d / "voice.json").write_text(json.dumps(asdict(v), ensure_ascii=False, indent=1), "utf-8")
    log(f"Обученный голос «{name}» подключён: {model_dir}")
    return v
