"""Главное окно Golos."""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings, QUrl, QMimeData, QItemSelectionModel
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QDesktopServices, QIcon, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QProgressBar, QFileDialog, QMessageBox, QPlainTextEdit, QDockWidget,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSplitter, QDialog, QDialogButtonBox, QAbstractItemView, QToolButton,
    QListWidget, QListWidgetItem, QRadioButton, QStackedWidget, QTextEdit)

from . import __version__, APP_NAME
from .i18n import T, set_lang
from .project import Project, Word, SR, new_project_dir
from .widgets import TimelinePanel, PhraseEditor
from .workers import Task
from . import audio_io, bridge, models, voices, stress

ORG = "GolosApp"

HOTKEYS = [
    ("R", "запись / стоп", "record / stop"),
    ("Space", "воспроизведение / пауза", "play / pause"),
    ("T", "разрезать в точке курсора", "split at playhead"),
    ("Delete", "удалить выделенный клип или участок (дорожка сдвигается)", "ripple-delete selected clip or range"),
    ("Q", "обрезать начало клипа до курсора", "trim clip start to playhead"),
    ("W", "обрезать конец клипа от курсора", "trim clip end at playhead"),
    ("← / →", "курсор на 0,1 с (с Shift — на 1 с)", "nudge 0.1 s (Shift — 1 s)"),
    ("Home / End", "в начало / в конец", "go to start / end"),
    ("+ / − / 0", "зум / вся дорожка", "zoom / fit"),
    ("Ctrl+Z / Ctrl+Y", "отмена / повтор", "undo / redo"),
]


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


def default_root() -> Path:
    docs = Path(os.path.expanduser("~")) / "Documents"
    return (docs if docs.exists() else Path.home()) / "Golos"


# =====================================================================
class SettingsDialog(QDialog):
    def __init__(self, s: QSettings, parent=None):
        super().__init__(parent)
        self.s = s
        self.setWindowTitle(T("Настройки", "Settings"))
        f = QFormLayout(self)
        self.ui_lang = QComboBox(); self.ui_lang.addItem("Русский", "ru"); self.ui_lang.addItem("English", "en")
        self.ui_lang.setCurrentIndex(0 if s.value("ui_lang", "ru") == "ru" else 1)
        self.mic = QComboBox()
        self.mic.addItem(T("По умолчанию (Windows)", "Windows default"), -1)
        for i, name in audio_io.input_devices():
            self.mic.addItem(name, i)
        self.mic.setCurrentIndex(max(0, self.mic.findData(int(s.value("mic", -1)))))
        self.src = QComboBox()
        for code in ("ru", "kk", "en"):
            self.src.addItem(models.TTS_LANGS[code][2], code)
        self.src.setCurrentIndex(max(0, self.src.findData(s.value("source_lang", "ru"))))
        self.root = QLineEdit(str(s.value("root", str(default_root()))))
        b = QPushButton("…"); b.clicked.connect(self._pick)
        row = QHBoxLayout(); row.addWidget(self.root); row.addWidget(b)
        self.prec = QComboBox()
        self.prec.addItem("fp16 — " + T("быстро, для RTX 20xx", "fast, for RTX 20xx"), "fp16")
        self.prec.addItem("fp32 — " + T("медленнее, если fp16 даёт брак", "slower, if fp16 glitches"), "fp32")
        self.prec.setCurrentIndex(max(0, self.prec.findData(s.value("precision", "fp16"))))
        f.addRow(T("Язык интерфейса", "Interface language"), self.ui_lang)
        f.addRow(T("Микрофон", "Microphone"), self.mic)
        f.addRow(T("Язык начитки", "Recording language"), self.src)
        f.addRow(T("Папка проектов", "Projects folder"), row)
        f.addRow(T("Точность синтеза", "Synthesis precision"), self.prec)
        self.smode = QComboBox()
        for m_ in stress.MODES:
            self.smode.addItem(stress.MODE_NAMES[m_], m_)
        self.smode.setCurrentIndex(max(0, self.smode.findData(s.value("stress_mode", "acute"))))
        sd = QPushButton(T("Открыть словарь ударений", "Open stress dictionary"))
        sd.clicked.connect(lambda: (stress.dict_path().exists() or stress.dict_path().write_text("{}", "utf-8"),
                                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(stress.dict_path())))))
        f.addRow(T("Ударение для модели", "Stress for the model"), self.smode)
        self.mmode = QComboBox()
        for m_ in stress.MIXED_MODES:
            self.mmode.addItem(stress.MIXED_NAMES[m_], m_)
        self.mmode.setCurrentIndex(max(0, self.mmode.findData(s.value("mixed_mode", "split"))))
        f.addRow(T("Английские слова в русской фразе", "English words in Russian phrases"), self.mmode)
        pd = QPushButton(T("Открыть словарь произношения", "Open pronunciation dictionary"))
        pd.clicked.connect(lambda: (stress.pron_path().exists() or stress.pron_path().write_text("{}", "utf-8"),
                                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(stress.pron_path())))))
        f.addRow("", pd)
        f.addRow("", sd)
        f.addRow(QLabel(T("Видеокарта: ", "GPU: ") + models.torch_info()))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "", self.root.text())
        if d:
            self.root.setText(d)

    def accept(self):
        old = self.s.value("ui_lang", "ru")
        for k, v in [("ui_lang", self.ui_lang.currentData()), ("mic", self.mic.currentData()),
                     ("source_lang", self.src.currentData()), ("root", self.root.text()),
                     ("precision", self.prec.currentData()), ("stress_mode", self.smode.currentData()),
                     ("mixed_mode", self.mmode.currentData())]:
            self.s.setValue(k, v)
        if old != self.ui_lang.currentData():
            QMessageBox.information(self, APP_NAME, T("Язык интерфейса сменится после перезапуска.",
                                                      "Interface language changes after restart."))
        super().accept()


class NewVoiceDialog(QDialog):
    """Создание голоса: по описанию, готовый диктор или из файла."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(T("Новый голос", "New voice"))
        self.resize(560, 380)
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.name = QLineEdit(); self.name.setPlaceholderText(T("Например: Спокойный баритон", "e.g. Calm baritone"))
        f.addRow(T("Название", "Name"), self.name)
        v.addLayout(f)
        self.r_design = QRadioButton(T("По описанию (нейросеть придумает голос)", "From description (AI-designed voice)"))
        self.r_preset = QRadioButton(T("Готовый диктор Qwen", "Qwen preset speaker"))
        self.r_file = QRadioButton(T("Из аудиофайла (5–15 с чистой речи, только с согласия человека)",
                                     "From an audio file (5–15 s clean speech, with the person's consent)"))
        self.r_design.setChecked(True)
        for r in (self.r_design, self.r_preset, self.r_file):
            v.addWidget(r); r.toggled.connect(self._mode)
        self.stack = QStackedWidget()
        w1 = QWidget(); f1 = QFormLayout(w1)
        self.desc = QTextEdit(); self.desc.setMaximumHeight(80)
        self.desc.setPlaceholderText(T("Мужчина средних лет, тёплый низкий голос, спокойная уверенная подача диктора рекламы",
                                       "Middle-aged man, warm low voice, calm confident commercial narrator"))
        self.sample_lang = QComboBox(); self.sample_lang.addItem("Русский", "ru"); self.sample_lang.addItem("English", "en")
        f1.addRow(T("Описание", "Description"), self.desc)
        f1.addRow(T("Язык образца", "Sample language"), self.sample_lang)
        w2 = QWidget(); f2 = QFormLayout(w2)
        self.speaker = QComboBox(); self.speaker.addItems(models.PRESET_SPEAKERS)
        self.style = QLineEdit(); self.style.setPlaceholderText(T("Стиль (необязательно): спокойно, с улыбкой",
                                                                  "Style (optional): calm, smiling"))
        self.sample_lang2 = QComboBox(); self.sample_lang2.addItem("Русский", "ru"); self.sample_lang2.addItem("English", "en")
        f2.addRow(T("Диктор", "Speaker"), self.speaker)
        f2.addRow(T("Стиль", "Style"), self.style)
        f2.addRow(T("Язык образца", "Sample language"), self.sample_lang2)
        w3 = QWidget(); f3 = QFormLayout(w3)
        self.file = QLineEdit(); b = QPushButton("…"); b.clicked.connect(self._pick)
        row = QHBoxLayout(); row.addWidget(self.file); row.addWidget(b)
        self.file_text = QLineEdit(); self.file_text.setPlaceholderText(T("Текст образца (пусто — распознаю сам)",
                                                                          "Sample transcript (empty — auto)"))
        self.file_lang = QComboBox()
        for c in ("ru", "kk", "en"):
            self.file_lang.addItem(models.TTS_LANGS[c][2], c)
        f3.addRow(T("Файл", "File"), row)
        f3.addRow(T("Текст", "Text"), self.file_text)
        f3.addRow(T("Язык", "Language"), self.file_lang)
        for w in (w1, w2, w3):
            self.stack.addWidget(w)
        v.addWidget(self.stack)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _mode(self):
        self.stack.setCurrentIndex(0 if self.r_design.isChecked() else 1 if self.r_preset.isChecked() else 2)

    def _pick(self):
        f, _ = QFileDialog.getOpenFileName(self, "", "", "Audio (*.wav *.flac *.mp3 *.ogg)")
        if f:
            self.file.setText(f)

    def spec(self) -> dict:
        name = self.name.text().strip() or T("Голос", "Voice")
        if self.r_design.isChecked():
            return {"kind": "design", "name": name, "description": self.desc.toPlainText().strip(),
                    "lang": self.sample_lang.currentData()}
        if self.r_preset.isChecked():
            return {"kind": "preset", "name": name, "speaker": self.speaker.currentText(),
                    "style": self.style.text().strip(), "lang": self.sample_lang2.currentData()}
        return {"kind": "file", "name": name, "file": self.file.text().strip(),
                "text": self.file_text.text().strip(), "lang": self.file_lang.currentData()}


def safe_name(name: str) -> str:
    import re as _re
    name = _re.sub(r'[<>:"/\\|?*]+', " ", name).strip().rstrip(".")
    return name[:80] or "Проект"


def project_info(folder: Path) -> dict:
    """Короткая сводка проекта без загрузки звука."""
    import json as _json
    info = {"name": folder.name, "mtime": 0.0, "phrases": 0, "langs": [], "srt": False}
    f = folder / "project.json"
    try:
        info["mtime"] = f.stat().st_mtime
        st = _json.loads(f.read_text("utf-8"))
        words = st.get("words", [])
        info["srt"] = bool(st.get("from_srt"))
        if info["srt"]:
            info["phrases"] = len(words)
        else:
            info["phrases"] = sum(1 for w in words if str(w.get("text", "")).rstrip().endswith((".", "!", "?", "…"))) or \
                (1 if words else 0)
        info["langs"] = [k for k, v in st.get("dub", {}).items() if any(d.get("variants") for d in v)]
    except Exception:  # noqa: BLE001
        pass
    return info


class ProjectsDialog(QDialog):
    """Список проектов: открыть, новый, переименовать, дублировать, удалить."""

    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win = win
        self.choice: Path | None = None
        self.action = None
        self.setWindowTitle(T("Проекты", "Projects"))
        self.resize(820, 520)
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([T("Проект", "Project"), T("Изменён", "Modified"),
                                              T("Фраз", "Phrases"), T("Озвучено", "Voiced")])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._open)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in [(T("Открыть", "Open"), self._open), (T("Новый…", "New…"), self._new),
                         (T("Переименовать…", "Rename…"), self._rename), (T("Дублировать…", "Duplicate…"), self._dup),
                         (T("Удалить", "Delete"), self._delete), (T("Показать в папке", "Show in folder"), self._reveal),
                         (T("Открыть из другой папки…", "Open from another folder…"), self._browse)]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        v.addLayout(row)
        self.cb_last = QCheckBox(T("Открывать последний проект при запуске (без этого окна)",
                                   "Open the last project on start (skip this window)"))
        self.cb_last.setChecked(win.s.value("open_last", "false") == "true")
        self.cb_last.toggled.connect(lambda c: win.s.setValue("open_last", str(c).lower()))
        v.addWidget(self.cb_last)
        self.fill()

    def folders(self) -> list[Path]:
        root = self.win.root()
        out = [d for d in root.iterdir() if d.is_dir() and (d / "project.json").exists()]
        return sorted(out, key=lambda d: (d / "project.json").stat().st_mtime, reverse=True)

    def fill(self):
        import datetime as _dt
        self.items = self.folders()
        self.table.setRowCount(len(self.items))
        cur = self.win.p.folder if self.win.p else None
        for r, d in enumerate(self.items):
            info = project_info(d)
            name = info["name"] + (T("  (открыт)", "  (open)") if cur and d == cur else "")
            self.table.setItem(r, 0, QTableWidgetItem(name))
            self.table.setItem(r, 1, QTableWidgetItem(_dt.datetime.fromtimestamp(info["mtime"]).strftime("%d.%m.%Y %H:%M")))
            self.table.setItem(r, 2, QTableWidgetItem((str(info["phrases"]) if info["phrases"] else "—") +
                                                      (" · SRT" if info["srt"] else "")))
            self.table.setItem(r, 3, QTableWidgetItem(", ".join(info["langs"]) or "—"))
        if self.items:
            self.table.selectRow(0)

    def _sel(self) -> Path | None:
        r = self.table.currentRow()
        return self.items[r] if 0 <= r < len(self.items) else None

    def _open(self, *_):
        d = self._sel()
        if d:
            self.choice, self.action = d, "open"
            self.accept()

    def _new(self):
        self.action = "new"
        self.accept()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, T("Папка проекта", "Project folder"), str(self.win.root()))
        if d and Path(d, "project.json").exists():
            self.choice, self.action = Path(d), "open"
            self.accept()

    def _ask_name(self, title: str, default: str) -> str | None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, title, T("Название проекта:", "Project name:"), text=default)
        if not ok or not name.strip():
            return None
        return safe_name(name)

    def _rename(self):
        d = self._sel()
        if not d:
            return
        name = self._ask_name(T("Переименовать", "Rename"), d.name)
        if not name or name == d.name:
            return
        new = d.parent / name
        if new.exists():
            QMessageBox.warning(self, APP_NAME, T("Проект с таким названием уже есть.", "A project with this name exists.")); return
        is_open = self.win.p is not None and self.win.p.folder == d
        if is_open:
            self.win.p.save(); self.win.stop_play()
        try:
            d.rename(new)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, str(e)); return
        if is_open:
            self.win.open_project(new)
        self.fill()

    def _dup(self):
        import shutil
        d = self._sel()
        if not d:
            return
        name = self._ask_name(T("Дублировать", "Duplicate"), d.name + T(" (копия)", " (copy)"))
        if not name:
            return
        new = d.parent / name
        if new.exists():
            QMessageBox.warning(self, APP_NAME, T("Проект с таким названием уже есть.", "A project with this name exists.")); return
        if self.win.p is not None and self.win.p.folder == d:
            self.win.p.save()
        shutil.copytree(d, new, ignore=shutil.ignore_patterns("export", "stress_test"))
        self.fill()

    def _delete(self):
        import shutil
        import time as _t
        d = self._sel()
        if not d:
            return
        if QMessageBox.question(self, APP_NAME, T(f"Удалить проект «{d.name}»?\nОн переместится в папку «_Корзина» "
                                                  f"рядом с проектами — его можно будет вернуть.",
                                                  f"Delete «{d.name}»? It moves to «_Корзина» next to the projects.")) \
                != QMessageBox.Yes:
            return
        if self.win.p is not None and self.win.p.folder == d:
            self.win.stop_play(); self.win.p = None
            self.win.tl.view.set_project(None); self.win.text.set_project(None)
            self.win.fill_dub_table(); self.win._update_state()
        trash = d.parent / "_Корзина"
        trash.mkdir(exist_ok=True)
        shutil.move(str(d), str(trash / f"{d.name} {_t.strftime('%Y-%m-%d %H-%M-%S')}"))
        self.fill()

    def _reveal(self):
        d = self._sel()
        if d:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))


class StressWordDialog(QDialog):
    """Ударение в одном слове: клик по гласной, запись голоса (программа найдёт ударный слог) или RuAccent."""

    def __init__(self, win: "MainWindow", word: str, phrase: str, index: int):
        super().__init__(win)
        self.win, self.word, self.phrase, self.index = win, stress.strip(word), phrase, index
        self.choice: int | None = None
        self.rec = None
        self.audio = None
        self.setWindowTitle(T("Ударение: ", "Stress: ") + self.word)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(T("Кликните ударную гласную — или запишите, как слово должно звучать.",
                             "Click the stressed vowel — or record how the word should sound.")))
        row = QHBoxLayout(); row.addStretch(1)
        self.btns = []
        vpos = stress.vowel_positions(self.word)
        for i, ch in enumerate(self.word):
            if i in vpos:
                b = QPushButton(ch); b.setCheckable(True)
                b.setStyleSheet("QPushButton{font-size:22pt;min-width:34px;padding:4px}"
                                "QPushButton:checked{background:#d9a400;color:black}")
                b.clicked.connect(lambda _=False, k=vpos.index(i): self.pick(k))
                self.btns.append(b); row.addWidget(b)
            else:
                l = QLabel(ch); l.setStyleSheet("font-size:22pt"); row.addWidget(l)
        row.addStretch(1); v.addLayout(row)
        self.info = QLabel(""); self.info.setAlignment(Qt.AlignCenter); v.addWidget(self.info)
        r2 = QHBoxLayout()
        self.rec_btn = QPushButton("● " + T("Записать слово", "Record the word"))
        self.rec_btn.setStyleSheet("QPushButton{background:#a3242f;color:white;font-weight:bold;padding:8px}")
        self.rec_btn.clicked.connect(self.toggle_rec)
        self.play_btn = QPushButton("▶ " + T("Моя запись", "My recording")); self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(lambda: self.audio is not None and win.player.play(self.audio[0], self.audio[1]))
        auto = QPushButton(T("Подобрать автоматически", "Auto (RuAccent)"))
        auto.clicked.connect(self.auto)
        for b in (self.rec_btn, self.play_btn, auto):
            r2.addWidget(b)
        v.addLayout(r2)
        self.cb_regen = QCheckBox(T("Сразу перегенерировать фразу выбранным голосом", "Regenerate the phrase right away"))
        self.cb_regen.setChecked(True); v.addWidget(self.cb_regen)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); v.addWidget(bb)
        self.ok = bb.button(QDialogButtonBox.Ok); self.ok.setEnabled(False)
        known = stress.load_dict().get(self.word.lower())
        if known and "+" in known:
            k = sum(1 for ch in known[:known.index("+")] if ch in stress.VOWELS)
            self.pick(k, T("из словаря", "from dictionary"))
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

    def pick(self, k: int, why: str = ""):
        self.choice = k
        for i, b in enumerate(self.btns):
            b.setChecked(i == k)
        self.info.setText(stress.set_stress(self.word, k) + (f"   ({why})" if why else ""))
        self.ok.setEnabled(True)

    def toggle_rec(self):
        if self.rec is None:
            dev = int(self.win.s.value("mic", -1))
            self.rec = audio_io.Recorder(None if dev < 0 else dev)
            try:
                self.rec.start()
            except Exception as e:  # noqa: BLE001
                self.rec = None
                QMessageBox.warning(self, APP_NAME, str(e)); return
            self.rec_btn.setText("■ " + T("Стоп", "Stop"))
            self.info.setText(T("Говорите слово… (остановится само через 3 с)", "Say the word… (stops in 3 s)"))
            self.timer.start(100)
        else:
            self._stop()

    def _tick(self):
        if self.rec is not None and self.rec.seconds >= 3.0:
            self._stop()

    def _stop(self):
        self.timer.stop()
        x, sr = self.rec.stop()
        self.rec = None
        self.rec_btn.setText("● " + T("Записать ещё раз", "Record again"))
        if len(x) < sr * 0.2:
            self.info.setText(T("Слишком коротко — запишите ещё раз", "Too short — record again")); return
        self.audio = (x, sr); self.play_btn.setEnabled(True)
        k, conf = stress.detect_from_audio(x, sr, self.word)
        if k is None:
            self.info.setText(T("Не удалось разобрать слоги — кликните ударную гласную сами",
                                "Could not find syllables — click the stressed vowel"))
            return
        self.pick(k, T("по записи", "from recording") + ("" if conf > 0.3 else T(", не уверена — проверьте", ", unsure — check")))

    def auto(self):
        res = stress.auto_stress_word(self.phrase, self.index, self.win.log)
        if not res:
            self.info.setText(T("Не определилось — кликните гласную", "Not found — click the vowel")); return
        w = stress.words_of(res)[self.index]
        k = sum(1 for ch in w[:w.index("+")] if ch in stress.VOWELS)
        self.pick(k, "RuAccent")

    def result(self) -> str:
        return stress.set_stress(self.word, self.choice) if self.choice is not None else self.word

    def reject(self):
        if self.rec is not None:
            self.rec.stop(); self.rec = None
        super().reject()


# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.s = QSettings(ORG, APP_NAME)
        self.p: Project | None = None
        self.rec: audio_io.Recorder | None = None
        self.rec_append = False
        self.player = audio_io.Player()
        self.play_mode = None          # "seq" | "dub" | "variant"
        self.task: Task | None = None
        self._clip_ignore = ""
        self.last_zip: Path | None = None
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1400, 900)
        self._build()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(33)
        QApplication.clipboard().dataChanged.connect(self._clipboard_changed)
        last = self.s.value("last_project", "")
        if self.s.value("open_last", "false") == "true":
            if last and Path(last, "project.json").exists():
                try:
                    self.open_project(Path(last))
                except Exception as e:  # noqa: BLE001
                    self.log(f"Не удалось открыть последний проект: {e}")
        else:
            QTimer.singleShot(150, self.projects_dialog)
        self._update_state()

    # ================= интерфейс =================
    def _build(self):
        tb = self.addToolBar("main"); tb.setMovable(False)
        for text, key, fn in [
            (T("Проекты…", "Projects…"), QKeySequence.Open, self.projects_dialog),
            (T("Новый…", "New…"), QKeySequence.New, self.new_project_named),
            (T("Сохранить", "Save"), QKeySequence.Save, self.save_project),
            (T("Сохранить как…", "Save as…"), QKeySequence("Ctrl+Shift+S"), self.save_as),
            (T("Открыть SRT…", "Open SRT…"), None, self.open_srt),
            (T("Импорт аудио…", "Import audio…"), None, self.import_audio),
            (T("Папка проекта", "Project folder"), None, self.open_folder),
            (T("Настройки", "Settings"), None, self.settings),
            (T("Горячие клавиши", "Hotkeys"), None, self.show_hotkeys),
        ]:
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn); tb.addAction(a)
            if text.startswith(T("Сохранить как", "Save as")):
                from PySide6.QtWidgets import QMenu
                self.recent_btn = QToolButton(); self.recent_btn.setText(T("Недавние ▾", "Recent ▾"))
                self.recent_btn.setPopupMode(QToolButton.InstantPopup)
                self.recent_menu = QMenu(self); self.recent_btn.setMenu(self.recent_menu)
                self.recent_menu.aboutToShow.connect(self._fill_recent)
                tb.addWidget(self.recent_btn)
        self.title_lbl = QLabel(); tb.addSeparator(); tb.addWidget(self.title_lbl)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_edit_tab(), T("1. Запись и монтаж", "1. Record & edit"))
        self.tabs.addTab(self._build_dub_tab(), T("2. Перевод и озвучка", "2. Translate & voice"))
        self.tabs.addTab(self._build_export_tab(), T("3. Отправка в чат", "3. Send to chat"))
        self.setCentralWidget(self.tabs)

        dock = QDockWidget(T("Журнал", "Log"), self); dock.setObjectName("log")
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.prog = QProgressBar(); self.prog.setVisible(False)
        self.cancel_btn = QPushButton(T("Отмена", "Cancel")); self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(lambda: self.task and self.task.cancel())
        row.addWidget(self.prog, 1); row.addWidget(self.cancel_btn)
        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(3000)
        l.addLayout(row); l.addWidget(self.logbox)
        dock.setWidget(w)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [130], Qt.Vertical)

        tl = self.tl.view
        for seq, fn in [
            ("R", self.toggle_record), ("Space", self.toggle_play),
            ("T", self.split), ("Delete", self.delete_sel), ("Backspace", self.delete_sel),
            ("Q", lambda: self.trim("start")), ("W", lambda: self.trim("end")),
            ("Left", lambda: self.nudge(-0.1)), ("Right", lambda: self.nudge(0.1)),
            ("Shift+Left", lambda: self.nudge(-1)), ("Shift+Right", lambda: self.nudge(1)),
            ("Home", lambda: self.goto(0)), ("End", lambda: self.goto(1e9)),
            ("=", lambda: tl.zoom(0.7)), ("+", lambda: tl.zoom(0.7)), ("-", lambda: tl.zoom(1.4)),
            ("0", tl.zoom_all), ("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
        ]:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(lambda f=fn, k=seq: self._hotkey(f, k in ("Space", "Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z")))
        self.log(f"{APP_NAME} {__version__}. " + T("Видеокарта: ", "GPU: ") + models.torch_info())

    def _hotkey(self, fn, any_tab: bool = False):
        """Горячие клавиши не срабатывают, пока вы печатаете в поле ввода;
        клавиши монтажа работают только на вкладке «Запись и монтаж»."""
        if not any_tab and self.tabs.currentIndex() != 0:
            return
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit)) or \
                (isinstance(fw, QTextEdit) and not fw.isReadOnly()):
            return
        if fw is not None and fw.parent() is not None and isinstance(fw.parent().parent(), QTableWidget):
            return
        fn()

    def _build_edit_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.rec_btn = QPushButton("● " + T("Запись", "Record") + "  (R)")
        self.rec_btn.setStyleSheet("QPushButton{background:#a3242f;color:white;font-weight:bold;padding:8px 16px}")
        self.rec_btn.clicked.connect(lambda: self.toggle_record())
        self.append_btn = QPushButton("+ " + T("Дозаписать в конец", "Record more at the end"))
        self.append_btn.clicked.connect(lambda: self.toggle_record(append=True))
        self.level = QProgressBar(); self.level.setRange(0, 100); self.level.setTextVisible(False)
        self.level.setMaximumWidth(200)
        self.rec_time = QLabel("0:00.0"); self.rec_time.setMinimumWidth(60)
        self.play_btn = QPushButton("▶  (Space)"); self.play_btn.clicked.connect(self.toggle_play)
        self.pos_lbl = QLabel("0:00.0 / 0:00.0"); self.pos_lbl.setMinimumWidth(120)
        self.stats = QLabel()
        for x in (self.rec_btn, self.append_btn, self.level, self.rec_time, self.play_btn, self.pos_lbl):
            row.addWidget(x)
        row.addStretch(1); row.addWidget(self.stats)
        v.addLayout(row)

        self.tl = TimelinePanel()
        self.tl.view.seek.connect(self.on_seek)
        self.tl.view.edited.connect(self.edits_changed)
        tools = QHBoxLayout()
        for text, fn, tip in [
            (T("Разрезать", "Split") + " (T)", self.split, ""),
            (T("Удалить", "Delete") + " (Del)", self.delete_sel, ""),
            (T("Обрезать начало", "Trim start") + " (Q)", lambda: self.trim("start"), ""),
            (T("Обрезать конец", "Trim end") + " (W)", lambda: self.trim("end"), ""),
            (T("Отменить", "Undo") + " (Ctrl+Z)", self.undo, ""), (T("Повторить", "Redo") + " (Ctrl+Y)", self.redo, ""),
            (T("Вся дорожка", "Fit") + " (0)", lambda: self.tl.view.zoom_all(), ""),
        ]:
            b = QPushButton(text); b.clicked.connect(fn); b.setFocusPolicy(Qt.NoFocus)
            if tip:
                b.setToolTip(tip)
            tools.addWidget(b)
        tools.addStretch(1)
        hint = QLabel(T("Клик — выбрать клип, протяжка — выделить участок, края клипа тянутся мышью, "
                        "колесо — зум, Shift+колесо — прокрутка",
                        "Click — select clip, drag — select range, drag clip edges to trim, "
                        "wheel — zoom, Shift+wheel — scroll"))
        hint.setStyleSheet("color:#888"); tools.addWidget(hint)

        split = QSplitter(Qt.Vertical)
        top = QWidget(); tlay = QVBoxLayout(top); tlay.setContentsMargins(0, 0, 0, 0)
        tlay.addWidget(self.tl, 1); tlay.addLayout(tools)
        split.addWidget(top)

        bottom = QWidget(); bl = QHBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        self.text = PhraseEditor(); self.text.seek.connect(self.on_seek)
        self.text.text_edited.connect(self.on_phrase_edited)
        self.text.menu_requested.connect(lambda pid, gpos: self.phrase_menu(pid, gpos))
        bl.addWidget(self.text, 3)

        side = QVBoxLayout()
        g = QGroupBox(T("Паузы (по кнопке, не сами)", "Pauses (on demand)")); f = QFormLayout(g)
        self.sp_pmax = self._dspin(0.2, 3, 0.05, " s"); self.sp_ptgt = self._dspin(0.1, 2, 0.05, " s")
        f.addRow(T("Паузы длиннее", "Pauses longer than"), self.sp_pmax)
        f.addRow(T("сократить до", "shorten to"), self.sp_ptgt)
        b = QPushButton(T("Укоротить паузы", "Shorten pauses")); b.clicked.connect(self.shorten_pauses)
        f.addRow(b)
        side.addWidget(g)
        g2 = QGroupBox(T("Обработка при экспорте", "Processing on export")); f2 = QFormLayout(g2)
        self.cb_dn = QCheckBox(T("Шумодав", "Noise reduction")); self.sp_dn = self._dspin(0.1, 1.0, 0.05, "")
        self.cb_ln = QCheckBox(T("Громкость", "Loudness")); self.sp_ln = self._dspin(-30, -8, 1, " LUFS")
        f2.addRow(self.cb_dn, self.sp_dn); f2.addRow(self.cb_ln, self.sp_ln)
        side.addWidget(g2)
        for wdg in (self.cb_dn, self.cb_ln):
            wdg.toggled.connect(self.on_edit_settings)
        for wdg in (self.sp_pmax, self.sp_ptgt, self.sp_dn, self.sp_ln):
            wdg.valueChanged.connect(self.on_edit_settings)
        rb = QPushButton(T("Вернуть исходную запись целиком", "Restore the whole original"))
        rb.clicked.connect(self.restore_all); side.addWidget(rb)
        self.asr_btn = QPushButton(T("① Распознать текст", "① Recognize text"))
        self.asr_btn.clicked.connect(lambda: self.run_asr(None))
        side.addWidget(self.asr_btn)
        self.script_btn = QPushButton(T("Вставить свой текст…", "Paste my own text…"))
        self.script_btn.setToolTip(T("Вставьте текст, который вы читали: он заменит распознанный и разложится по фразам",
                                     "Paste the text you read: it replaces the recognized one, split into phrases"))
        self.script_btn.clicked.connect(self.paste_script)
        side.addWidget(self.script_btn)
        side.addStretch(1)
        self.send_btn = QPushButton(T("② Текст проверен — отправить на перевод  →",
                                      "② Text checked — send to translation  →"))
        self.send_btn.setStyleSheet("QPushButton{background:#2f6fd0;color:white;font-weight:bold;padding:12px}")
        self.send_btn.clicked.connect(self.send_to_translation)
        side.addWidget(self.send_btn)
        sw = QWidget(); sw.setLayout(side); sw.setMaximumWidth(360)
        bl.addWidget(sw, 1)
        split.addWidget(bottom)
        split.setSizes([430, 330])
        v.addWidget(split, 1)
        return w

    @staticmethod
    def _dspin(a, b, step, suffix):
        s = QDoubleSpinBox(); s.setRange(a, b); s.setSingleStep(step); s.setDecimals(2); s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        return s

    def _build_dub_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()

        g0 = QGroupBox(T("Голос", "Voice")); f0 = QVBoxLayout(g0)
        self.voice_box = QComboBox(); self.voice_box.currentIndexChanged.connect(self._voice_changed)
        r = QHBoxLayout()
        nb = QPushButton(T("Новый голос…", "New voice…")); nb.clicked.connect(self.new_voice)
        pb = QPushButton("▶"); pb.setToolTip(T("Послушать образец", "Play sample")); pb.clicked.connect(self.play_voice)
        db = QPushButton(T("Удалить", "Delete")); db.setToolTip(T("Удалить голос", "Delete voice")); db.clicked.connect(self.delete_voice)
        r.addWidget(nb); r.addWidget(pb); r.addWidget(db)
        f0.addWidget(self.voice_box); f0.addLayout(r)
        note = QLabel(T("Любой голос говорит на всех выбранных языках, включая казахский.",
                        "Any voice speaks all selected languages, including Kazakh."))
        note.setWordWrap(True); note.setStyleSheet("color:#999"); f0.addWidget(note)
        top.addWidget(g0, 1)

        g = QGroupBox(T("Языки", "Languages")); gl = QVBoxLayout(g)
        self.lang_list = QListWidget(); self.lang_list.setMaximumHeight(140)
        saved = self.s.value("dub_langs", "en,kk").split(",")
        for code, (_, _, label) in models.TTS_LANGS.items():
            extra = ("  — " + T("переозвучка моей начитки", "re-voice my recording")) if code == "ru" else ""
            it = QListWidgetItem(f"{label} ({code}){extra}")
            it.setData(Qt.UserRole, code)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if code in saved else Qt.Unchecked)
            self.lang_list.addItem(it)
        self.lang_list.itemChanged.connect(self._langs_changed)
        gl.addWidget(self.lang_list)
        top.addWidget(g, 1)

        g2 = QGroupBox(T("Перевод через чат с Claude", "Translate via Claude chat")); f = QVBoxLayout(g2)
        self.ctx = QLineEdit(); self.ctx.setPlaceholderText(T("О чём ролик (необязательно)", "What the video is about (optional)"))
        self.ctx.setText(self.s.value("ctx", ""))
        self.copy_req_btn = QPushButton(T("Скопировать запрос на перевод", "Copy translation request"))
        self.copy_req_btn.clicked.connect(self.copy_request)
        self.paste_btn = QPushButton(T("Вставить перевод из буфера", "Paste translation from clipboard"))
        self.paste_btn.clicked.connect(self.paste_translation)
        self.auto_paste = QCheckBox(T("Подхватывать перевод из буфера сам", "Pick up translation automatically"))
        self.auto_paste.setChecked(self.s.value("auto_paste", "true") == "true")
        self.auto_dub = QCheckBox(T("Сразу озвучивать после перевода", "Voice right after translation"))
        self.auto_dub.setChecked(self.s.value("auto_dub", "true") == "true")
        tip = QLabel(T("Ctrl+V в чат → под ответом «Копировать» → перевод и озвучка запустятся сами.",
                       "Ctrl+V in chat → Copy under the answer → translation and voicing start by themselves."))
        tip.setWordWrap(True); tip.setStyleSheet("color:#999")
        for x in (self.ctx, self.copy_req_btn, self.paste_btn, self.auto_paste, self.auto_dub, tip):
            f.addWidget(x)
        top.addWidget(g2, 2)

        g3 = QGroupBox(T("Озвучка", "Voicing")); f3 = QFormLayout(g3)
        self.sp_var = QSpinBox(); self.sp_var.setRange(1, 5); self.sp_var.setValue(int(self.s.value("variants", 3)))
        self.cb_verify = QCheckBox(T("Проверять варианты распознаванием", "Verify variants with ASR"))
        self.cb_verify.setChecked(self.s.value("verify", "true") == "true")
        self.sp_temp = self._dspin(0.3, 1.2, 0.05, ""); self.sp_temp.setValue(float(self.s.value("temp", 0.8)))
        self.dub_btn = QPushButton(T("Озвучить", "Generate voice-over"))
        self.dub_btn.setStyleSheet("QPushButton{font-weight:bold;padding:8px}")
        self.dub_btn.clicked.connect(lambda: self.run_dub(None))
        f3.addRow(T("Вариантов на фразу", "Variants per phrase"), self.sp_var)
        f3.addRow(T("Живость", "Expressiveness"), self.sp_temp)
        self.topup_btn = QPushButton(T("Догенерировать", "Top up variants"))
        self.topup_btn.setToolTip(T("Оставляет имеющиеся варианты и досоздаёт недостающие до «Вариантов на фразу».\n"
                                    "Выделены строки — только для них, иначе для всех фраз.",
                                    "Keeps existing takes and generates the missing ones up to «Variants per phrase».\n"
                                    "Selected rows only, or all phrases."))
        self.topup_btn.clicked.connect(self.top_up)
        f3.addRow(self.cb_verify); f3.addRow(self.dub_btn); f3.addRow(self.topup_btn)
        top.addWidget(g3, 1)
        v.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel(T("Язык:", "Language:")))
        self.view_lang = QComboBox(); self.view_lang.currentIndexChanged.connect(self.fill_dub_table)
        row.addWidget(self.view_lang)
        self.play_track_btn = QPushButton("▶ " + T("Проиграть выбранные", "Play chosen"))
        self.play_track_btn.setToolTip(T("Выбранные варианты всех фраз подряд, по таймингу оригинала (пробел — пауза).\n"
                                         "Выделена одна строка — с этой фразы до конца; несколько — от первой выделенной до последней.",
                                         "Chosen variants of all phrases in sequence, on the original timing (Space — pause).\n"
                                         "One row selected — from that phrase to the end; several — from the first to the last selected."))
        self.play_track_btn.clicked.connect(self.play_track)
        stop = QPushButton("■"); stop.clicked.connect(self.stop_play)
        self.regen_btn = QPushButton(T("Перегенерировать выделенные фразы", "Regenerate selected phrases"))
        self.regen_btn.clicked.connect(self.regen_selected)
        row.addWidget(self.play_track_btn); row.addWidget(stop); row.addWidget(self.regen_btn)
        row.addStretch(1)
        self.dub_info = QLabel(); row.addWidget(self.dub_info)
        v.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", T("Длит.", "Dur."), T("Оригинал (можно поправить)", "Original (editable)"),
                                              T("Перевод (можно поправить)", "Translation (editable)"),
                                              T("Варианты: клик — слушать и выбрать", "Variants: click to play & pick")])
        h = self.table.horizontalHeader()
        for c, mode in enumerate((QHeaderView.ResizeToContents, QHeaderView.ResizeToContents, QHeaderView.Stretch,
                                  QHeaderView.Stretch, QHeaderView.ResizeToContents)):
            h.setSectionResizeMode(c, mode)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self.on_table_edit)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.row_menu)
        v.addWidget(self.table, 1)
        self.reload_voices()
        return w

    def _build_export_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        info = QLabel(T(
            "Здесь собирается всё для чата с Claude, где делается ролик: ваша смонтированная начитка, "
            "дорожки озвучки, субтитры и тайминги каждой фразы — одним zip-файлом. "
            "Файл кладётся в буфер обмена: откройте чат и нажмите Ctrl+V.",
            "Everything for the Claude video chat in one zip: your edited recording, voice tracks, subtitles "
            "and per-phrase timings. The file goes to the clipboard: open the chat and press Ctrl+V."))
        info.setWordWrap(True); v.addWidget(info)
        row = QHBoxLayout()
        self.export_btn = QPushButton(T("Собрать и скопировать в буфер", "Build & copy to clipboard"))
        self.export_btn.setStyleSheet("QPushButton{background:#2f6fd0;color:white;font-weight:bold;padding:12px}")
        self.export_btn.clicked.connect(self.export_package)
        self.copy_zip_btn = QPushButton(T("Скопировать ещё раз", "Copy again"))
        self.copy_zip_btn.clicked.connect(lambda: self.copy_file(self.last_zip))
        self.open_exp_btn = QPushButton(T("Показать в папке", "Show in folder"))
        self.open_exp_btn.clicked.connect(lambda: self.reveal(self.last_zip))
        self.export_src_btn = QPushButton(T("Сохранить только начитку (WAV/MP3)…", "Save recording only (WAV/MP3)…"))
        self.export_src_btn.clicked.connect(self.export_source_only)
        for b in (self.export_btn, self.copy_zip_btn, self.open_exp_btn, self.export_src_btn):
            row.addWidget(b)
        row.addStretch(1); v.addLayout(row)
        self.export_lbl = QLabel(); self.export_lbl.setWordWrap(True); v.addWidget(self.export_lbl)
        tip = QLabel(T("Если чат не принял вставку — нажмите «Показать в папке» и перетащите zip в чат мышью.",
                       "If the chat doesn't accept the paste — click Show in folder and drag the zip into the chat."))
        tip.setWordWrap(True); tip.setStyleSheet("color:#999"); v.addWidget(tip)
        v.addStretch(1)
        return w

    # ================= общее =================
    def log(self, msg: str):
        self.logbox.appendPlainText(time.strftime("%H:%M:%S ") + msg)
        sb = self.logbox.verticalScrollBar()
        sb.setValue(sb.maximum())          # журнал всегда показывает последнее событие

    def show_hotkeys(self):
        rows = "".join(f"<tr><td><b>{k}</b>&nbsp;&nbsp;</td><td>{T(ru, en)}</td></tr>" for k, ru, en in HOTKEYS)
        QMessageBox.information(self, T("Горячие клавиши", "Hotkeys"), f"<table>{rows}</table>")

    def _busy(self) -> bool:
        return self.task is not None and self.task.isRunning()

    def _update_state(self):
        has = self.p is not None
        has_audio = has and bool(self.p.clips)
        has_words = has_audio and bool(self.p.words)
        busy = self._busy()
        self.play_btn.setEnabled(has_audio)
        self.append_btn.setEnabled(has_audio and self.rec is None)
        self.asr_btn.setEnabled(has_audio and not busy)
        self.script_btn.setEnabled(has_audio and not busy)
        self.send_btn.setEnabled(has_audio and not busy)
        for b in (self.copy_req_btn, self.paste_btn, self.dub_btn, self.topup_btn):
            b.setEnabled(has_words and not busy)
        self.export_btn.setEnabled(has_audio and not busy)
        self.export_src_btn.setEnabled(has_audio)
        self.copy_zip_btn.setEnabled(self.last_zip is not None)
        self.open_exp_btn.setEnabled(self.last_zip is not None)
        self.title_lbl.setText(f"  {self.p.name}" if has else T("  нет проекта", "  no project"))
        if has and len(self.p.audio):
            self.stats.setText(T("Записано ", "Recorded ") + fmt_time(self.p.duration) + "  →  " +
                               T("на дорожке ", "on track ") + fmt_time(self.p.seq_duration()) +
                               T(f"  ·  клипов: {len(self.p.clips)}", f"  ·  clips: {len(self.p.clips)}"))
        else:
            self.stats.setText("")

    def _load_edit_settings(self):
        e = self.p.edit
        for wdg, val in [(self.cb_dn, e.denoise), (self.cb_ln, e.loudness)]:
            wdg.blockSignals(True); wdg.setChecked(val); wdg.blockSignals(False)
        for wdg, val in [(self.sp_pmax, e.pause_max), (self.sp_ptgt, e.pause_target),
                         (self.sp_dn, e.denoise_strength), (self.sp_ln, e.target_lufs)]:
            wdg.blockSignals(True); wdg.setValue(val); wdg.blockSignals(False)

    def on_edit_settings(self, *_):
        if not self.p:
            return
        e = self.p.edit
        e.denoise, e.denoise_strength = self.cb_dn.isChecked(), self.sp_dn.value()
        e.loudness, e.target_lufs = self.cb_ln.isChecked(), self.sp_ln.value()
        e.pause_max, e.pause_target = self.sp_pmax.value(), self.sp_ptgt.value()
        self.p.save()

    def edits_changed(self):
        if not self.p:
            return
        if self.player.playing and self.play_mode == "seq":
            self.stop_play()
        self.p.save()
        self.tl.view.refresh()
        self.text.rebuild()
        self._update_state()

    # ================= проекты =================
    def root(self) -> Path:
        r = Path(self.s.value("root", str(default_root())))
        r.mkdir(parents=True, exist_ok=True)
        return r

    def new_project(self, name: str | None = None):
        self.stop_play()
        d = base = (self.root() / name) if name else new_project_dir(self.root())
        i = 2
        while d.exists():
            d = Path(f"{base} ({i})"); i += 1
        p = Project.create(d)
        p.source_lang = self.s.value("source_lang", "ru")
        p.voice = self.s.value("voice", "me")
        p.save()
        self.set_project(p)
        self.log(T("Новый проект: ", "New project: ") + str(d))

    def projects_dialog(self):
        dlg = ProjectsDialog(self)
        if not dlg.exec():
            return
        if dlg.action == "open" and dlg.choice:
            if self.p is not None and self.p.folder == dlg.choice:
                return
            self.open_project(dlg.choice)
        elif dlg.action == "new":
            self.new_project_named()

    def new_project_named(self):
        from PySide6.QtWidgets import QInputDialog
        default = new_project_dir(self.root()).name
        name, ok = QInputDialog.getText(self, T("Новый проект", "New project"), T("Название проекта:", "Project name:"),
                                        text=default)
        if not ok:
            return
        self.new_project(safe_name(name or default))

    def save_project(self):
        if self.p:
            self.p.save()
            self.log(T(f"Проект «{self.p.name}» сохранён (он и так сохраняется сам при каждом изменении)",
                       f"Project «{self.p.name}» saved (it also autosaves on every change)"))

    def save_as(self):
        import shutil
        from PySide6.QtWidgets import QInputDialog
        if not self.p:
            return
        name, ok = QInputDialog.getText(self, T("Сохранить как", "Save as"), T("Название копии:", "Copy name:"),
                                        text=self.p.name + T(" (копия)", " (copy)"))
        if not ok or not name.strip():
            return
        new = self.root() / safe_name(name)
        if new.exists():
            QMessageBox.warning(self, APP_NAME, T("Проект с таким названием уже есть.", "A project with this name exists.")); return
        self.p.save(); self.stop_play()
        shutil.copytree(self.p.folder, new, ignore=shutil.ignore_patterns("export", "stress_test"))
        self.open_project(new)

    def _recent(self) -> list[str]:
        r = [x for x in (self.s.value("recent", "") or "").split("|") if x]
        return [x for x in r if Path(x, "project.json").exists()][:10]

    def _fill_recent(self):
        self.recent_menu.clear()
        items = self._recent()
        if not items:
            a = self.recent_menu.addAction(T("(пусто)", "(empty)")); a.setEnabled(False)
        for x in items:
            a = self.recent_menu.addAction(Path(x).name)
            a.triggered.connect(lambda _=False, x=x: self.open_project(Path(x)))

    def open_dialog(self):
        d = QFileDialog.getExistingDirectory(self, T("Папка проекта", "Project folder"), str(self.root()))
        if d:
            self.open_project(Path(d))

    def open_project(self, d: Path):
        self.stop_play()
        self.set_project(Project.open(d))
        self.log(T("Открыт проект: ", "Opened: ") + str(d))

    def set_project(self, p: Project):
        self.p = p
        self.setWindowTitle(f"{p.name} — {APP_NAME} {__version__}")
        self.s.setValue("last_project", str(p.folder))
        rec = [str(p.folder)] + [x for x in (self.s.value("recent", "") or "").split("|") if x and x != str(p.folder)]
        self.s.setValue("recent", "|".join(rec[:10]))
        self.tl.view.set_project(p)
        self.text.set_project(p)
        self._load_edit_settings()
        self.reload_voices()
        self.refresh_dub_view()
        self._update_state()

    def open_folder(self):
        if self.p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.p.folder)))

    def settings(self):
        if SettingsDialog(self.s, self).exec() and self.p and not self.p.words:
            self.p.source_lang = self.s.value("source_lang", "ru"); self.p.save()

    def import_audio(self):
        f, _ = QFileDialog.getOpenFileName(self, T("Аудиофайл", "Audio file"), "",
                                           "Audio (*.wav *.flac *.mp3 *.ogg *.aiff *.aif)")
        if not f:
            return
        if self.p is None or len(self.p.audio):
            self.new_project()
        try:
            self.p.load_audio_file(Path(f))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e)); return
        self.set_project(self.p)
        self.log(T("Импортировано: ", "Imported: ") + f)

    def open_srt(self):
        """SRT с оригинальным текстом и таймкодами → озвучка по этому таймингу."""
        from . import media
        f, _ = QFileDialog.getOpenFileName(self, T("Субтитры с оригинальным текстом", "Subtitles with the original text"),
                                           "", "SRT (*.srt)")
        if not f:
            return
        try:
            raw = media.read_srt(f)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e)); return
        if not raw:
            QMessageBox.warning(self, APP_NAME, T("В файле нет титров.", "No cues in the file.")); return
        cues = media.merge_to_sentences(raw)
        self.stop_play()
        d = base = self.root() / Path(f).stem
        i = 2
        while d.exists():
            d = Path(f"{base} ({i})"); i += 1
        p = Project.create(d)
        p.source_lang = self.s.value("source_lang", "ru")
        p.load_cues(cues)
        if p.voice == "me":
            other = [v for v in voices.list_voices() if v.kind != "me"]
            if other:
                p.voice = other[0].id
        p.save()
        self.set_project(p)
        self.tabs.setCurrentIndex(1)
        merged = len(raw) - len(cues)
        self.log(T(f"Открыт SRT: {len(raw)} титров" + (f", склеено в {len(cues)} предложений" if merged else "") +
                   ". Дальше: «Скопировать запрос» → чат → «Копировать» под ответом → «Озвучить».",
                   f"SRT opened: {len(raw)} cues" + (f", merged into {len(cues)} sentences" if merged else "") +
                   ". Next: Copy request → chat → Copy the answer → Generate."))
        if p.voice == "me":
            self.log(T("Своего голоса в библиотеке пока нет: «Новый голос…» → «Из аудиофайла» (10 с вашей записи).",
                       "No own voice in the library yet: New voice… → From an audio file (10 s of your recording)."))

    # ================= запись =================
    def toggle_record(self, append: bool = False):
        if self.rec is None:
            self.stop_play()
            if not append and self.p is not None and len(self.p.audio):
                r = QMessageBox.question(
                    self, APP_NAME,
                    T("Начать новую запись в новом проекте?\n(Чтобы добавить к этой — «Дозаписать в конец».)",
                      "Start a new recording in a new project?\n(To add to this one use «Record more».)"))
                if r != QMessageBox.Yes:
                    return
            if not append and (self.p is None or len(self.p.audio)):
                self.new_project()
            dev = int(self.s.value("mic", -1))
            self.rec = audio_io.Recorder(None if dev < 0 else dev)
            self.rec_append = append
            try:
                self.rec.start()
            except Exception as e:  # noqa: BLE001
                self.rec = None
                QMessageBox.warning(self, APP_NAME, T("Микрофон не открылся: ", "Mic failed: ") + str(e))
                return
            self.rec_btn.setText("■ " + T("Стоп", "Stop") + "  (R)")
            self.log(T("Идёт запись… Остановить — кнопка «Стоп» или клавиша R.",
                       "Recording… Stop with the Stop button or R."))
        else:
            x, sr = self.rec.stop()
            self.rec = None
            self.rec_btn.setText("● " + T("Запись", "Record") + "  (R)")
            self.level.setValue(0)
            if len(x) < sr * 0.3:
                self.log(T("Запись слишком короткая", "Recording too short")); self._update_state(); return
            from .dsp import resample
            x = resample(x, sr, SR)
            if self.rec_append and len(self.p.audio):
                self.p.append_audio(x)
                self.tl.view.rebuild_env()
                self.tl.view.zoom_all()
                self.log(T("Дозаписано ", "Appended ") + fmt_time(len(x) / SR))
            else:
                self.p.set_audio(x)
                self.set_project(self.p)
                self.log(T("Записано ", "Recorded ") + fmt_time(self.p.duration) +
                         T(". Режьте: T — разрезать, Delete — удалить.", ". Edit: T — split, Delete — remove."))
            self.edits_changed()
        self._update_state()

    def _tick(self):
        if self.rec is not None:
            lv = self.rec.level
            self.level.setValue(int(min(1.0, lv) * 100))
            self.level.setStyleSheet("QProgressBar::chunk{background:%s}" % ("#e0433d" if lv > 0.89 else "#3fbf6f"))
            self.rec_time.setText(fmt_time(self.rec.seconds))
        if self.player.playing and self.play_mode == "seq":
            t = self.player.position()
            self.tl.view.set_head(t)
            self.text.highlight(t)
            self._show_pos(t)
        elif self.player.playing and self.play_mode == "dub":
            self._highlight_dub()
        elif self.play_mode and not self.player.playing:
            self.play_mode = None
            self.play_btn.setText("▶  (Space)")

    def _show_pos(self, t):
        if self.p:
            self.pos_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.p.seq_duration())}")

    # ================= воспроизведение =================
    def toggle_play(self):
        if self.player.playing:
            self.stop_play(); return
        if self.tabs.currentIndex() == 1:
            self.play_track(); return
        if not self.p or not self.p.clips:
            return
        t = self.tl.view.head
        if t >= self.p.seq_duration() - 0.05:
            t = 0.0
        self.player.play(self.p.render(process=False), SR, t)
        self.play_mode = "seq"
        self.play_btn.setText("❚❚  (Space)")

    def stop_play(self):
        self.player.stop()
        self.play_mode = None
        self.play_btn.setText("▶  (Space)")

    def on_seek(self, t: float):
        was = self.player.playing and self.play_mode == "seq"
        self.tl.view.set_head(t, follow=False)
        self.text.highlight(t)
        self._show_pos(t)
        if was:
            self.player.stop()
            self.player.play(self.p.render(process=False), SR, t)
            self.play_mode = "seq"

    def nudge(self, d: float):
        if self.p:
            self.on_seek(min(self.p.seq_duration(), max(0.0, self.tl.view.head + d)))
            self.tl.view.set_head(self.tl.view.head)

    def goto(self, t: float):
        if self.p:
            self.on_seek(min(self.p.seq_duration(), max(0.0, t)))
            self.tl.view.set_head(self.tl.view.head)

    # ================= монтаж =================
    def split(self):
        if self.p and self.p.split(self.tl.view.head):
            self.edits_changed()

    def delete_sel(self):
        if not self.p:
            return
        v = self.tl.view
        if v.range:
            a, b = v.range
            if self.p.delete_range(a, b):
                v.range = None
                self.on_seek(a); self.edits_changed()
        elif v.selected:
            first = min(v.selected)
            start = self.p.clip_starts()[first]
            if self.p.delete_clips(set(v.selected)):
                v.selected.clear()
                self.on_seek(min(start, self.p.seq_duration())); self.edits_changed()
        else:
            self.log(T("Выберите клип кликом или выделите участок протяжкой.",
                       "Click a clip or drag to select a range first."))

    def trim(self, side: str):
        if not self.p:
            return
        t = self.tl.view.head
        i = self.p.clip_at(t)
        if self.p.trim_to(i, t, side):
            if side == "start":
                self.on_seek(self.p.clip_starts()[i])
            self.edits_changed()

    def shorten_pauses(self):
        if not self.p:
            return
        n = self.p.shorten_pauses(self.sp_pmax.value(), self.sp_ptgt.value())
        self.log(T(f"Укорочено пауз: {n} (отменить — Ctrl+Z)", f"Pauses shortened: {n} (undo — Ctrl+Z)"))
        self.edits_changed()

    def restore_all(self):
        if self.p and len(self.p.audio):
            self.p.restore_all(); self.edits_changed()

    def undo(self):
        if self.p and self.p.undo():
            self.edits_changed()

    def redo(self):
        if self.p and self.p.redo():
            self.edits_changed()

    # ================= фоновые задачи =================
    def start_task(self, fn, done, *args, **kw):
        if self._busy():
            return False
        self.stop_play()
        t = Task(fn, *args, **kw)
        t.log.connect(self.log)
        t.progress.connect(self._on_progress)
        t.failed.connect(lambda m: (self.log("ОШИБКА: " + m), QMessageBox.warning(self, APP_NAME, m)))
        t.succeeded.connect(done)
        t.finished.connect(self._task_finished)

        self.task = t
        self.prog.setVisible(True); self.prog.setRange(0, 0); self.cancel_btn.setVisible(True)
        t.start()
        self._update_state()
        return True

    def _on_progress(self, a, b):
        self.prog.setRange(0, max(1, b)); self.prog.setValue(a)

    def _task_finished(self):
        self.prog.setVisible(False); self.cancel_btn.setVisible(False)
        self.task = None
        self._update_state()

    # ----- распознавание смонтированной дорожки -----
    def run_asr(self, then=None):
        if not self.p or not self.p.clips:
            return
        if self.p.words_fresh:
            if then:
                then()
            return
        audio, lang, key = self.p.render(process=False), self.p.source_lang, self.p.edit_key()

        def job(task: Task):
            task.log.emit(T("Распознаю текст смонтированной дорожки (Whisper)…", "Recognizing edited track (Whisper)…"))
            wh = models.Whisper(task.log.emit)
            try:
                return wh.words(audio, SR, lang)
            finally:
                wh.close()

        def done(words):
            if key != self.p.edit_key():
                self.log(T("Монтаж изменился во время распознавания — повторите.", "Edit changed during recognition — retry."))
                return
            self.p.set_words([Word(t, s, e) for t, s, e in words])
            self.log(T(f"Распознано фраз: {len(self.p.phrases())}. Проверьте текст и исправьте ошибки "
                       f"(двойной клик по фразе), затем «② Отправить на перевод».",
                       f"Phrases: {len(self.p.phrases())}. Check the text, fix errors (double-click), "
                       f"then «② Send to translation»."))
            self.text.rebuild(); self.tl.view.update(); self.refresh_dub_view()
            if then:
                QTimer.singleShot(0, then)

        self.start_task(job, done)

    def send_to_translation(self):
        """Если текст ещё не распознан по текущему монтажу — распознаём и даём его проверить.
        Если распознан — запрос (с вашими исправлениями) в буфер и вкладка 2."""
        if not self.p or not self.p.clips:
            return
        if not self.p.words_fresh:
            self.log(T("Сначала распознаю текст. Проверьте и исправьте его в списке фраз, "
                       "затем нажмите «② Отправить на перевод» ещё раз.",
                       "Recognizing first. Check and fix the phrases, then press «② Send» again."))
            self.run_asr(None)
            return
        self.copy_request()
        self.tabs.setCurrentIndex(1)

    def paste_script(self):
        """Свой текст вместо распознанного: распознавание даёт только границы и тайминги фраз."""
        if not self.p or not self.p.clips:
            return
        dlg = QDialog(self); dlg.setWindowTitle(T("Свой текст", "My own text")); dlg.resize(700, 480)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(T("Вставьте текст, который вы читали (Ctrl+V). Пропущенные или лишние слова не страшны — "
                             "программа сама сопоставит текст с записью и разложит по фразам. "
                             "Английские слова пишите как надо (GradeStream, DaVinci Resolve).",
                             "Paste the text you read (Ctrl+V). Missing or extra words are fine — it is matched "
                             "to the recording and split into phrases.")))
        ed = QPlainTextEdit()
        clip = QApplication.clipboard().text()
        if clip and not bridge.looks_like_answer(clip) and bridge.MARK not in clip and len(clip.split()) > 3:
            ed.setPlainText(clip)
        v.addWidget(ed, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
        if not dlg.exec() or not ed.toPlainText().strip():
            return
        script = ed.toPlainText()
        self.s.setValue("last_script", script)

        def apply():
            n = self.p.apply_script(script)
            self.text.rebuild(); self.fill_dub_table()
            self.log(T(f"Ваш текст разложен по {n} фразам. Проверьте границы и при необходимости поправьте "
                       f"(двойной клик по фразе), затем «② Отправить на перевод».",
                       f"Your text was split into {n} phrases. Check and fix if needed, then «② Send»."))

        if self.p.words_fresh:
            apply()
        else:
            self.log(T("Сначала распознаю запись, чтобы найти границы фраз…", "Recognizing to find phrase boundaries…"))
            self.run_asr(apply)

    def on_phrase_edited(self, pid: int, text: str):
        if not self.p:
            return
        orig = {ph.id: ph.text for ph in self.p.phrases()}
        self.p.phrase_texts[pid] = text
        for w in stress.words_of(text):
            if "+" in w:
                stress.save_word(w)
        self.p.save()
        if orig.get(pid) != text:
            self.log(T(f"Фраза {pid} исправлена", f"Phrase {pid} edited"))
        QTimer.singleShot(0, self.text.rebuild)
        self.fill_dub_table()

    # ================= перевод =================
    def selected_langs(self) -> list[str]:
        return [self.lang_list.item(i).data(Qt.UserRole) for i in range(self.lang_list.count())
                if self.lang_list.item(i).checkState() == Qt.Checked]

    def _langs_changed(self, *_):
        self.s.setValue("dub_langs", ",".join(self.selected_langs()))
        self.refresh_dub_view()

    def copy_request(self):
        if not self.p or not self.p.words:
            return
        langs = [l for l in self.selected_langs() if l != self.p.source_lang]
        if not langs:
            self.log(T("Языки перевода не выбраны — озвучка будет только на языке начитки.",
                       "No target languages — voicing the recording language only."))
            return
        self.s.setValue("ctx", self.ctx.text())
        text = bridge.build_request(self.p.phrases(), langs, self.p.source_lang, self.ctx.text())
        self._clip_ignore = text
        QApplication.clipboard().setText(text)
        self.log(T("Запрос на перевод в буфере. Вставьте в чат с Claude (Ctrl+V), под ответом нажмите «Копировать».",
                   "Request copied. Paste into the Claude chat (Ctrl+V), then press Copy under the answer."))

    def _clipboard_changed(self):
        if not self.p or not self.auto_paste.isChecked():
            return
        text = QApplication.clipboard().text()
        if text and text != self._clip_ignore and bridge.looks_like_answer(text):
            self._clip_ignore = text
            self.apply_translation(text, quiet=True)

    def paste_translation(self):
        self.apply_translation(QApplication.clipboard().text(), quiet=False)

    def apply_translation(self, text: str, quiet: bool):
        try:
            tr = bridge.parse_answer(text)
        except ValueError:
            if not quiet:
                QMessageBox.warning(self, APP_NAME, T("В буфере нет перевода. Скопируйте ответ Claude целиком.",
                                                      "No translation on the clipboard. Copy Claude's whole answer."))
            return
        ids = {ph.id for ph in self.p.phrases()}
        msgs = []
        for lang, items in tr.items():
            if lang not in models.TTS_LANGS:
                continue
            self.p.translations[lang] = {i: t for i, t in items.items() if i in ids}
            miss = sorted(ids - set(items))
            msgs.append(f"{lang}: {len(items)}" + (T(f", нет фраз {miss}", f", missing {miss}") if miss else ""))
            for i in range(self.lang_list.count()):
                it = self.lang_list.item(i)
                if it.data(Qt.UserRole) == lang:
                    it.setCheckState(Qt.Checked)
        self.p.save()
        self.refresh_dub_view()
        self.tabs.setCurrentIndex(1)
        self.log(T("Перевод получен: ", "Translation received: ") + "; ".join(msgs))
        self.raise_(); self.activateWindow()
        if self.auto_dub.isChecked():
            self.run_dub(None, ask=False)

    # ================= голоса =================
    def reload_voices(self):
        cur = self.p.voice if self.p else self.s.value("voice", "me")
        self.voice_box.blockSignals(True)
        self.voice_box.clear()
        for vc in voices.list_voices():
            label = T(vc.name, "My voice (from recording)") if vc.kind == "me" else vc.name
            kind = {"me": "", "design": T(" · по описанию", " · designed"), "preset": T(" · диктор", " · preset"),
                    "file": T(" · из файла", " · from file"), "etalon": " ★"}.get(vc.kind, "")
            self.voice_box.addItem(label + kind, vc.id)
        self.voice_box.setCurrentIndex(max(0, self.voice_box.findData(cur)))
        self.voice_box.blockSignals(False)

    def _voice_changed(self, *_):
        vid = self.voice_box.currentData() or "me"
        self.s.setValue("voice", vid)
        if self.p:
            self.p.voice = vid; self.p.save()
            self.fill_dub_table()

    def play_voice(self):
        vc = voices.get(self.voice_box.currentData() or "me")
        try:
            x, text = voices.reference(vc, self.p)
        except Exception as e:  # noqa: BLE001
            self.log(str(e)); return
        self.player.play(x, 24000); self.play_mode = "variant"
        self.log(T("Образец: «", "Sample: «") + text + "»")

    def delete_voice(self):
        vid = self.voice_box.currentData()
        if not vid or vid == "me":
            return
        if QMessageBox.question(self, APP_NAME, T("Удалить голос?", "Delete voice?")) == QMessageBox.Yes:
            voices.delete_voice(vid)
            if self.p and self.p.voice == vid:
                self.p.voice = "me"; self.p.save()
            self.reload_voices()

    def new_voice(self):
        dlg = NewVoiceDialog(self)
        if not dlg.exec():
            return
        spec = dlg.spec()
        prec = self.s.value("precision", "fp16")
        lname = models.TTS_LANGS[spec["lang"]][1]
        sample = voices.SAMPLE_TEXT.get(spec["lang"], voices.SAMPLE_TEXT["en"])
        if spec["kind"] == "design" and not spec["description"]:
            QMessageBox.information(self, APP_NAME, T("Опишите голос.", "Describe the voice.")); return
        if spec["kind"] == "file" and not Path(spec["file"]).exists():
            QMessageBox.information(self, APP_NAME, T("Выберите файл.", "Pick a file.")); return

        def job(task: Task):
            if spec["kind"] == "design":
                m = models.TTS(models.TTS_DESIGN_ID, task.log.emit, prec)
                try:
                    x, sr = m.design(sample, lname, spec["description"])
                finally:
                    m.close()
                return voices.save_voice(spec["name"], "design", x, sr, sample, description=spec["description"])
            if spec["kind"] == "preset":
                m = models.TTS(models.TTS_PRESET_ID, task.log.emit, prec)
                try:
                    x, sr = m.preset(sample, lname, spec["speaker"], spec["style"])
                finally:
                    m.close()
                return voices.save_voice(spec["name"], "preset", x, sr, sample, speaker=spec["speaker"])
            import soundfile as sf
            from . import dsp
            x, sr = sf.read(spec["file"], dtype="float32", always_2d=False)
            x = dsp.to_mono(x)
            if len(x) / sr > 16:
                x = x[: int(15 * sr)]
                task.log.emit(T("Файл обрезан до 15 с", "File trimmed to 15 s"))
            text = spec["text"]
            if not text:
                wh = models.Whisper(task.log.emit)
                try:
                    text = wh.text(x, sr, spec["lang"])
                finally:
                    wh.close()
            return voices.save_voice(spec["name"], "file", x, sr, text)

        def done(vc):
            self.reload_voices()
            self.voice_box.setCurrentIndex(max(0, self.voice_box.findData(vc.id)))
            self.log(T("Голос создан: ", "Voice created: ") + vc.name)
            self.play_voice()

        self.start_task(job, done)

    # ================= озвучка =================
    def refresh_dub_view(self):
        cur = self.view_lang.currentData()
        self.view_lang.blockSignals(True)
        self.view_lang.clear()
        for l in self.selected_langs():
            self.view_lang.addItem(models.TTS_LANGS[l][2], l)
        k = self.view_lang.findData(cur)
        self.view_lang.setCurrentIndex(k if k >= 0 else 0)
        self.view_lang.blockSignals(False)
        self.fill_dub_table()

    def fill_dub_table(self, *_):
        from .dub import fit_status
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        lang = self.view_lang.currentData()
        if not self.p or not self.p.words or not lang:
            self.table.blockSignals(False); self.dub_info.setText(""); return
        phrases = self.p.phrases()
        src = lang == self.p.source_lang
        texts = {ph.id: ph.text for ph in phrases} if src else self.p.translations.get(lang, {})
        dubs = {d.id: d for d in self.p.dub.get(lang, [])}
        self.table.setRowCount(len(phrases))
        n_chosen = n_long = 0
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        for r, ph in enumerate(phrases):
            it = QTableWidgetItem(str(ph.id)); it.setFlags(ro); it.setData(Qt.UserRole, ph.id)
            self.table.setItem(r, 0, it)
            it = QTableWidgetItem(f"{ph.dur:.1f}s")
            if ph.id in self.p.dur_override:
                it.setForeground(QColor("#8fd48f"))
                it.setToolTip(T("Длительность задана вручную (правый клик — сбросить)", "Manual duration (right-click to reset)"))
            else:
                it.setToolTip(T("Двойной клик — задать длительность, например 3.5", "Double-click to set duration, e.g. 3.5"))
            self.table.setItem(r, 1, it)
            self.table.setItem(r, 2, QTableWidgetItem(ph.text))
            it = QTableWidgetItem(texts.get(ph.id, "") if not src else T("(как в оригинале)", "(same as original)"))
            if src:
                it.setFlags(ro)
            self.table.setItem(r, 3, it)
            d = dubs.get(ph.id)
            cell = QWidget(); hl = QHBoxLayout(cell); hl.setContentsMargins(2, 2, 2, 2); hl.setSpacing(3)
            if d and d.variants:
                for vi, vv in enumerate(d.variants):
                    st = fit_status(vv, ph.dur)
                    col = {"ok": "#2e7d4f", "stretch": "#a0771a", "long": "#a3242f"}[st]
                    warn = " ⚠" if (vv.cer is not None and vv.cer > 0.25) else ""
                    vstale = (vv.text or d.text) != (ph.text if src else texts.get(ph.id, "")) or \
                        (vv.voice or d.voice) != self.p.voice
                    star = "★ " if vv.pinned else ""
                    b = QToolButton(); b.setText(f"{star}{vi + 1}: {vv.dur:.1f}s{warn}" + (" ⟲" if vstale else ""))
                    if vstale:
                        col = "#555a63"
                    sel = vi == d.chosen
                    b.setStyleSheet(f"QToolButton{{background:{col};color:white;padding:3px 6px;"
                                    f"border:{'2px solid #ffcc33' if sel else '1px solid #333'};"
                                    f"font-weight:{'bold' if sel else 'normal'}}}")
                    tip = T(f"{vv.dur:.2f} с при слоте {ph.dur:.2f} с", f"{vv.dur:.2f}s for a {ph.dur:.2f}s slot")
                    if vstale:
                        tip += T("\n⟲ сгенерирован со старым текстом или другим голосом", "\n⟲ made with older text or another voice")
                    if vv.pinned:
                        tip += T("\n★ отмечен — не удаляется при перегенерации", "\n★ starred — kept on regeneration")
                    if vv.cer is not None:
                        tip += T(f"\nРасхождение с текстом: {vv.cer * 100:.0f}%", f"\nText mismatch: {vv.cer * 100:.0f}%")
                    b.setToolTip(tip)
                    b.clicked.connect(lambda _=False, L=lang, pid=ph.id, k=vi: self.pick_variant(L, pid, k))
                    b.setContextMenuPolicy(Qt.CustomContextMenu)
                    b.customContextMenuRequested.connect(
                        lambda pos, btn=b, L=lang, pid=ph.id, k=vi: self.variant_menu(btn, pos, L, pid, k))
                    hl.addWidget(b)
                if d.chosen >= 0:
                    n_chosen += 1
                    n_long += fit_status(d.variants[d.chosen], ph.dur) == "long"
                cv = d.variants[d.chosen] if 0 <= d.chosen < len(d.variants) else None
                stale = cv is None or (cv.text or d.text) != (ph.text if src else texts.get(ph.id, "")) or \
                    (not cv.pinned and (cv.voice or d.voice) != self.p.voice) or \
                    (d.slot > 0 and abs(d.slot - ph.dur) > 0.05)
                if stale:
                    lbl = QLabel(T("устарело", "outdated")); lbl.setStyleSheet("color:#ffb300"); hl.addWidget(lbl)
            else:
                lbl = QLabel("—"); lbl.setStyleSheet("color:#777"); hl.addWidget(lbl)
            self.table.setCellWidget(r, 4, cell)
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)
        self.dub_info.setText(T(f"Выбрано: {n_chosen}/{len(phrases)} · не влезают: {n_long}",
                                f"Chosen: {n_chosen}/{len(phrases)} · too long: {n_long}"))

    def on_table_edit(self, item: QTableWidgetItem):
        lang = self.view_lang.currentData()
        pid = self.table.item(item.row(), 0).data(Qt.UserRole)
        if item.column() == 1:
            import re as _re
            m = _re.search(r"\d+(?:[.,]\d+)?", item.text())
            if m and 0.2 <= float(m.group().replace(",", ".")) <= 120:
                self.p.dur_override[pid] = round(float(m.group().replace(",", ".")), 2)
                self.log(T(f"Фраза {pid}: длительность {self.p.dur_override[pid]} с — перегенерируйте её (правый клик)",
                           f"Phrase {pid}: duration {self.p.dur_override[pid]} s — regenerate it (right-click)"))
            else:
                self.log(T("Длительность: введите число секунд, например 3.5", "Duration: enter seconds, e.g. 3.5"))
        elif item.column() == 2:
            self.p.phrase_texts[pid] = item.text().strip()
            for w in stress.words_of(item.text()):
                if "+" in w:
                    stress.save_word(w)
        elif item.column() == 3 and lang != self.p.source_lang:
            self.p.translations.setdefault(lang, {})[pid] = item.text().strip()
        else:
            return
        self.p.save()
        QTimer.singleShot(0, self.fill_dub_table)

    def variant_menu(self, btn, pos, lang: str, pid: int, k: int):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        a1 = m.addAction(T(f"Перегенерировать вариант {k + 1}", f"Regenerate variant {k + 1}"))
        a2 = m.addAction(T("Перегенерировать всю фразу", "Regenerate the whole phrase"))
        m.addSeparator()
        a4 = m.addAction(T("★ Ориентироваться на этот вариант", "★ Use this take as the voice reference"))
        a5 = m.addAction(T("В набор для дообучения", "Add to the fine-tuning set"))
        dd = next((x for x in self.p.dub.get(lang, []) if x.id == pid), None)
        a7 = None
        if dd and 0 <= k < len(dd.variants) and dd.variants[k].pinned:
            a7 = m.addAction(T("Снять ★ (можно будет удалить при перегенерации)", "Unstar"))
        a6 = m.addAction(T(f"Открыть папку набора ({voices.dataset_minutes():.1f} мин)",
                           f"Open the set folder ({voices.dataset_minutes():.1f} min)"))
        m.addSeparator()
        a3 = m.addAction(T(f"Удалить вариант {k + 1}", f"Delete variant {k + 1}"))
        act = m.exec(btn.mapToGlobal(pos))
        d = next((x for x in self.p.dub.get(lang, []) if x.id == pid), None)
        if act in (a4, a5) and d and 0 <= k < len(d.variants):
            wav = self.p.dub_dir(lang) / d.variants[k].file
            if act == a4:
                d.variants[k].pinned = True
                d.chosen = k
                self.p.save()
                self.make_etalon(wav, d, d.variants[k])
            else:
                mins = voices.add_to_dataset(wav, d.text, lang, d.voice, f"{self.p.name}_{lang}_{pid}_{k + 1}")
                self.log(T(f"Добавлено в набор для дообучения. Всего: {mins:.1f} мин (для дообучения нужно ~30–60).",
                           f"Added to the fine-tuning set. Total: {mins:.1f} min (~30–60 needed)."))
            return
        if act == a6:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(voices.dataset_dir())))
            return
        if a7 is not None and act == a7:
            d.variants[k].pinned = False
            self.p.save(); self.fill_dub_table()
            return
        if act == a1:
            self.run_dub(None, ask=False, replace_variant=(lang, pid, k))
        elif act == a2:
            self.run_dub({lang: {pid}}, ask=False)
        elif act == a3:
            d = next((x for x in self.p.dub.get(lang, []) if x.id == pid), None)
            if d and 0 <= k < len(d.variants):
                d.variants.pop(k)
                if d.chosen == k:
                    d.chosen = 0 if d.variants else -1
                elif d.chosen > k:
                    d.chosen -= 1
                self.p.save(); self.fill_dub_table()

    def row_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        lang = self.view_lang.currentData()
        it = self.table.itemAt(pos)
        if not lang or it is None:
            return
        pid = self.table.item(it.row(), 0).data(Qt.UserRole)
        rows = sorted({i.row() for i in self.table.selectedIndexes()} | {it.row()})
        ids = {self.table.item(r, 0).data(Qt.UserRole) for r in rows}
        m = QMenu(self)
        a1 = m.addAction(T(f"Перегенерировать фразу {pid}", f"Regenerate phrase {pid}") if len(ids) == 1 else
                         T(f"Перегенерировать выделенные фразы ({len(ids)})", f"Regenerate selected phrases ({len(ids)})"))
        a0 = m.addAction(T(f"Догенерировать до {self.sp_var.value()} вариантов", f"Top up to {self.sp_var.value()} variants"))
        a2 = m.addAction(T("Сбросить длительность", "Reset duration"))
        a2.setEnabled(any(i in self.p.dur_override for i in ids))
        a3 = m.addAction(T("Вернуть удалённые варианты (с диска)", "Recover lost variants (from disk)"))
        if len(ids) == 1:
            m.addSeparator()
            self._stress_items(m, pid)
        act = m.exec(self.table.viewport().mapToGlobal(pos))
        if act == a0:
            self.run_dub({lang: ids}, ask=False, top_up=True)
        elif act == a1:
            self.run_dub({lang: ids}, ask=False)
        elif act == a2:
            for i in ids:
                self.p.dur_override.pop(i, None)
            self.p.save(); self.fill_dub_table()
        elif act == a3:
            n = sum(self.p.recover_variants(lang, i) for i in ids)
            self.log(T(f"Возвращено вариантов: {n}. Они помечены ⟲ — послушайте и отметьте нужный ★.",
                       f"Recovered variants: {n}. Marked ⟲ — listen and star the one you need."))
            self.fill_dub_table()

    # ----- эталон -----
    def make_etalon(self, wav: Path, d, v):
        import soundfile as sf
        if v.cer is not None and v.cer > 0.15:
            if QMessageBox.question(self, APP_NAME, T(
                    f"В этом варианте модель отклонилась от текста ({v.cer * 100:.0f}%). "
                    "Образец с ошибками ухудшит следующие фразы. Всё равно взять?",
                    f"This take deviates from the text ({v.cer * 100:.0f}%). Use it anyway?")) != QMessageBox.Yes:
                return
        x, sr = sf.read(str(wav), dtype="float32")
        base = voices.get(self.p.voice)
        try:
            vc = voices.add_to_etalon(base, x, sr, d.text, self.p)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e)); return
        self.p.voice = vc.id; self.p.save()
        self.s.setValue("voice", vc.id)
        self.reload_voices()
        self.log(T(f"★ Голос «{vc.name}»: образец теперь из отмеченных вариантов. Перегенерируйте фразы — "
                   f"они будут ориентироваться на эту подачу.",
                   f"★ Voice «{vc.name}» now uses the starred takes as reference. Regenerate phrases."))
        self.fill_dub_table()

    # ----- ударения -----
    def phrase_menu(self, pid: int, gpos):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        self._stress_items(m, pid)
        if m.actions():
            m.exec(gpos)

    def _stress_items(self, m, pid: int):
        if not self.p or self.p.source_lang != "ru":
            return
        ph = next((x for x in self.p.phrases() if x.id == pid), None)
        if not ph:
            return
        sub = m.addMenu(T("Ударение в слове", "Stress in word"))
        for i, w in enumerate(stress.words_of(ph.text)):
            if stress.vowel_count(stress.strip(w)) >= 2:
                a = sub.addAction(w)
                a.triggered.connect(lambda _=False, pid=pid, i=i: self.stress_dialog(pid, i))
        if not sub.actions():
            sub.setEnabled(False)
        a = m.addAction(T("Убрать ударения из фразы", "Remove stress marks"))
        a.setEnabled("+" in ph.text)
        a.triggered.connect(lambda: self._set_phrase_text(pid, stress.strip(ph.text)))
        a = m.addAction(T("Проверить ударение на слух (3 способа)…", "Test stress by ear (3 ways)…"))
        a.setEnabled("+" in stress.apply_dict(ph.text))
        a.triggered.connect(lambda: self.test_stress(pid))
        latin = [w for w in stress.words_of(ph.text) if re.search(r"[A-Za-z]{2,}", w)]
        if latin:
            m.addSeparator()
            sub2 = m.addMenu(T("Как читать слово (английское, бренд)", "How to read a word"))
            for w in dict.fromkeys(latin):
                cur = stress.load_pron().get(w.lower())
                a = sub2.addAction(w + (f"  →  {cur}" if cur else ""))
                a.triggered.connect(lambda _=False, w=w: self.pronounce_word(w))
            a = m.addAction(T("Двуязычная фраза: проверить 3 способа на слух…", "Mixed-language phrase: test 3 ways…"))
            a.triggered.connect(lambda: self.test_mixed(pid))

    def _set_phrase_text(self, pid: int, text: str):
        self.p.phrase_texts[pid] = text
        for w in stress.words_of(text):
            if "+" in w:
                stress.save_word(w)
        self.p.save(); self.text.rebuild(); self.fill_dub_table()

    def stress_dialog(self, pid: int, i: int):
        ph = next((x for x in self.p.phrases() if x.id == pid), None)
        if not ph:
            return
        words = stress.words_of(ph.text)
        if not 0 <= i < len(words):
            return
        dlg = StressWordDialog(self, words[i], ph.text, i)
        if not dlg.exec():
            return
        new_word = dlg.result()
        # подставляем слово с ударением на место i-го слова фразы
        import re as _re
        ms = list(_re.finditer(r"[A-Za-zА-Яа-яЁё+\u0301-]+", ph.text))
        m_ = ms[i]
        src = stress.strip(m_.group(0))
        fixed = "".join(ch.upper() if (k < len(src) and src[k].isupper()) else ch
                        for k, ch in enumerate(new_word.replace("+", ""))) if new_word else src
        # переносим «+» обратно с учётом регистра
        pos = new_word.find("+")
        fixed = fixed[:pos] + "+" + fixed[pos:] if pos >= 0 else fixed
        self._set_phrase_text(pid, ph.text[:m_.start()] + fixed + ph.text[m_.end():])
        stress.save_word(fixed)
        self.log(T(f"Ударение: {fixed} — запомнено в словаре", f"Stress: {fixed} — saved to dictionary"))
        if dlg.cb_regen.isChecked():
            langs = [l for l in self.selected_langs() if l == "ru"]
            if "ru" in langs:
                self.run_dub({"ru": {pid}}, ask=False)
            else:
                self.log(T("Русская озвучка не отмечена в «Языках» — отметьте «Русский», чтобы перегенерировать.",
                           "Russian isn't ticked in Languages — tick it to regenerate."))

    def add_stress(self, pid: int, i: int):
        ph = next((x for x in self.p.phrases() if x.id == pid), None)
        if not ph:
            return
        text = ph.text

        def job(task: Task):
            return stress.auto_stress_word(text, i, task.log.emit)

        def done(res):
            if not res:
                self.log(T("Ударение не определилось — поставьте вручную: двойной клик по фразе и «+» перед ударной гласной.",
                           "Could not place stress — add «+» before the stressed vowel manually."))
                return
            self._set_phrase_text(pid, res)
            w = stress.words_of(res)[i]
            self.log(T(f"Ударение: {w} (если неверно — двойной клик по фразе и поправьте «+»)",
                       f"Stress: {w} (if wrong, double-click the phrase and move «+»)"))

        self.start_task(job, done)

    def test_stress(self, pid: int):
        self._ab_test(pid, T("Какой способ ударения сработал?", "Which stress method worked?"),
                      stress.MODES, stress.MODE_NAMES, "stress_mode", "acute", "stress_test")

    def test_mixed(self, pid: int):
        self._ab_test(pid, T("Как лучше читать английские слова?", "How to read the English words?"),
                      stress.MIXED_MODES, stress.MIXED_NAMES, "mixed_mode", "split", "mixed_test")

    def _ab_test(self, pid, title, modes, names, key, default, method):
        """Окно открывается сразу; каждый способ становится доступен, как только готов."""
        ph = next((x for x in self.p.phrases() if x.id == pid), None)
        if not ph or self._busy():
            return
        import soundfile as sf
        from .dub import Dubber
        p, prec, text = self.p, self.s.value("precision", "fp16"), ph.text
        smode, mmode = self.s.value("stress_mode", "acute"), self.s.value("mixed_mode", "split")
        dlg = QDialog(self); dlg.setWindowTitle(title); dlg.setModal(False)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(T("Варианты появляются по мере готовности. Послушайте и выберите лучший — "
                             "этот способ будет использоваться дальше.",
                             "Takes appear as they are ready. Pick the best — it will be used from now on.")))
        cur = self.s.value(key, default)
        rows = {}
        for mode in modes:
            row = QHBoxLayout()
            pb = QPushButton("⏳ " + names[mode] + T(" — готовлю…", " — generating…")); pb.setEnabled(False)
            cb = QPushButton(T("Выбрать", "Choose")); cb.setEnabled(False)
            cb.clicked.connect(lambda _=False, mode=mode: (self.s.setValue(key, mode), dlg.accept(),
                                                           self.log(T("Выбран способ: ", "Chosen: ") + names[mode])))
            row.addWidget(pb, 1); row.addWidget(cb); v.addLayout(row)
            rows[mode] = (pb, cb)
        dlg.show()

        def on_item(item):
            mode, f = item
            pb, cb = rows[mode]
            pb.setText("▶ " + names[mode] + (T("  (сейчас)", "  (current)") if mode == cur else ""))
            pb.clicked.connect(lambda _=False, f=f: self.player.play(*sf.read(f, dtype="float32")))
            pb.setEnabled(True); cb.setEnabled(True)
            if mode == modes[0]:
                self.player.play(*sf.read(f, dtype="float32"))

        def job(task: Task):
            dub = Dubber(p, task.log.emit, lambda a, b: task.progress.emit(a, b), task.is_cancelled, prec, smode, mmode)
            return getattr(dub, method)(text, on_item=lambda m, f: task.item.emit((m, f)))

        if self.start_task(job, lambda _: None):
            self.task.item.connect(on_item)
            dlg.rejected.connect(lambda: self.task and self.task.cancel())

    def pronounce_word(self, word: str):
        from PySide6.QtWidgets import QInputDialog
        cur = stress.load_pron().get(word.lower(), "")
        how, ok = QInputDialog.getText(
            self, T("Как читать слово", "How to read the word"),
            T(f"Как произносить «{word}» в русской речи (кириллицей, «+» перед ударной гласной).\n"
              f"Пусто — читать как написано.", f"How to pronounce «{word}» in Russian speech. Empty — as written."),
            text=cur or "")
        if not ok:
            return
        stress.save_pron(word, how)
        self.log(T(f"Произношение: {word} → {how or '(как написано)'} — для всех проектов. Перегенерируйте фразы с этим словом.",
                   f"Pronunciation: {word} → {how or '(as written)'}. Regenerate phrases with it."))
        self.fill_dub_table()

    def pick_variant(self, lang: str, pid: int, k: int):
        d = next((x for x in self.p.dub.get(lang, []) if x.id == pid), None)
        if not d:
            return
        d.chosen = k
        self.p.save()
        import soundfile as sf
        x, sr = sf.read(str(self.p.dub_dir(lang) / d.variants[k].file), dtype="float32")
        self.player.play(x, sr)
        self.play_mode = "variant"
        self.fill_dub_table()

    def top_up(self):
        lang = self.view_lang.currentData()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows and lang:
            self.run_dub({lang: {self.table.item(r, 0).data(Qt.UserRole) for r in rows}}, ask=False, top_up=True)
        else:
            self.run_dub(None, ask=False, top_up=True)

    def run_dub(self, only_ids, ask: bool = True, replace_variant=None, top_up: bool = False):
        if not self.p or not self.p.words:
            return
        langs = self.selected_langs()
        if replace_variant:
            langs = [replace_variant[0]]
        elif only_ids:
            langs = [l for l in langs if l in only_ids]
        missing = [l for l in langs if l != self.p.source_lang and not self.p.translations.get(l)]
        if missing and only_ids is None:
            if ask and QMessageBox.question(
                    self, APP_NAME, T(f"Нет перевода для: {', '.join(missing)}. Озвучить остальные?",
                                      f"No translation for: {', '.join(missing)}. Voice the rest?")) != QMessageBox.Yes:
                return
            langs = [l for l in langs if l not in missing]
        if not langs:
            return
        for k, val in [("variants", self.sp_var.value()), ("verify", str(self.cb_verify.isChecked()).lower()),
                       ("temp", self.sp_temp.value()), ("auto_dub", str(self.auto_dub.isChecked()).lower()),
                       ("auto_paste", str(self.auto_paste.isChecked()).lower())]:
            self.s.setValue(k, val)
        from .dub import Dubber
        p, nv, ver, temp = self.p, self.sp_var.value(), self.cb_verify.isChecked(), self.sp_temp.value()
        prec = self.s.value("precision", "fp16")
        smode = self.s.value("stress_mode", "acute")
        mmode = self.s.value("mixed_mode", "split")

        def job(task: Task):
            Dubber(p, task.log.emit, lambda a, b: task.progress.emit(a, b), task.is_cancelled, prec, smode, mmode).run(
                langs, nv, ver, only_ids, temp, replace_variant, top_up)

        def done(_):
            self.refresh_dub_view()
            self.log(T("Озвучка готова: послушайте варианты (клик по кнопке), лучшие уже выбраны.",
                       "Voice-over ready: listen to the variants; the best ones are pre-selected."))

        self.tabs.setCurrentIndex(1)
        self.start_task(job, done)

    def regen_selected(self):
        lang = self.view_lang.currentData()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not lang or not rows:
            QMessageBox.information(self, APP_NAME, T("Выделите строки фраз.", "Select phrase rows.")); return
        self.run_dub({lang: {self.table.item(r, 0).data(Qt.UserRole) for r in rows}})

    def play_track(self):
        lang = self.view_lang.currentData()
        if not lang or not self.p or not self.p.dub.get(lang):
            return
        from .dub import build_track, OUT_SR
        try:
            x, timings, warns = build_track(self.p, lang)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e)); return
        for w in warns:
            self.log(f"[{lang}] {w}")
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        start, end = 0.0, len(x) / OUT_SR
        if rows:
            ids = {self.table.item(r, 0).data(Qt.UserRole) for r in rows}
            sel = [t for t in timings if t["id"] in ids]
            if sel:
                start = max(0.0, min(t["start"] for t in sel) - 0.15)
                if len(rows) > 1:                       # несколько строк — от первой до последней выделенной
                    end = min(end, max(t["end"] for t in sel) + 0.3)
                # одна строка — с этой фразы до конца
        self.track_timings = timings
        self.track_offset = start
        self.player.play(x[int(start * OUT_SR): int(end * OUT_SR)], OUT_SR)
        self.play_mode = "dub"
        self._dub_row = -1
        n = len([t for t in timings if start <= t["start"] < end])
        self.log(T(f"Играю {n} фраз подряд ({end - start:.1f} с). Пробел — пауза.",
                   f"Playing {n} phrases ({end - start:.1f} s). Space — pause."))

    def _highlight_dub(self):
        t = self.track_offset + self.player.position()
        cur = next((it["id"] for it in self.track_timings if it["start"] - 0.05 <= t <= it["end"] + 0.1), None)
        if cur is None:
            return
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.UserRole) == cur:
                if r != self._dub_row:
                    self._dub_row = r
                    self.table.setCurrentCell(r, 2, QItemSelectionModel.NoUpdate)
                    for c in range(4):
                        cell = self.table.item(r, c)
                        if cell:
                            cell.setBackground(QColor(255, 204, 51, 60))
                    if hasattr(self, "_prev_dub_row") and self._prev_dub_row not in (None, r):
                        for c in range(4):
                            cell = self.table.item(self._prev_dub_row, c)
                            if cell:
                                cell.setBackground(QColor(0, 0, 0, 0))
                    self._prev_dub_row = r
                    self.table.scrollToItem(self.table.item(r, 2))
                break

    # ================= отправка в чат =================
    def export_package(self):
        if self.p and self.p.clips and not self.p.words_fresh:
            self.run_asr(self._export_package)     # тайминги фраз нужны по текущему монтажу
        else:
            self._export_package()

    def _export_package(self):
        from . import exporter
        langs = [l for l in self.selected_langs() if self.p.dub.get(l)]
        p = self.p

        def job(task: Task):
            return exporter.export_package(p, langs, task.log.emit)

        def done(z):
            self.last_zip = Path(z)
            self.copy_file(self.last_zip)
            self.export_lbl.setText(T("Готово, zip в буфере обмена — Ctrl+V в чате: ", "Done, zip is on the clipboard: ") + str(z))
            self._update_state()

        self.start_task(job, done)

    def export_source_only(self):
        f, _ = QFileDialog.getSaveFileName(self, "", str(self.p.folder / f"{self.p.name}.wav"), "WAV (*.wav)")
        if not f:
            return
        from . import exporter
        exporter.write_audio(Path(f).with_suffix(""), self.p.render(process=True), SR)
        self.log(T("Сохранено: ", "Saved: ") + f)

    def copy_file(self, path: Path | None):
        if not path or not Path(path).exists():
            return
        md = QMimeData()
        md.setUrls([QUrl.fromLocalFile(str(path))])
        QApplication.clipboard().setMimeData(md)
        self.log(T("Файл в буфере обмена — нажмите Ctrl+V в чате с Claude: ",
                   "File on clipboard — press Ctrl+V in the Claude chat: ") + Path(path).name)

    def reveal(self, path: Path | None):
        if not path:
            return
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def closeEvent(self, e):
        try:
            models.release_tts()
        except Exception:  # noqa: BLE001
            pass
        if self.rec is not None:
            self.toggle_record()
        self.player.stop()
        if self.p:
            self.p.save()
        super().closeEvent(e)


DARK = """
QWidget{background:#1e2026;color:#e6e6e6;font-size:10pt}
QPushButton,QToolButton{background:#2c2f36;border:1px solid #3a3d45;padding:5px 10px;border-radius:4px}
QPushButton:hover{background:#373b44} QPushButton:disabled{color:#666}
QLineEdit,QPlainTextEdit,QTextEdit,QTableWidget,QListWidget,QSpinBox,QDoubleSpinBox,QComboBox{
 background:#15171b;border:1px solid #33363d;selection-background-color:#3d5a80}
QHeaderView::section{background:#2a2d33;padding:4px;border:0}
QGroupBox{border:1px solid #33363d;margin-top:14px;padding-top:6px}
QGroupBox::title{subcontrol-origin:margin;left:8px}
QTabBar::tab{background:#2a2d33;padding:8px 18px} QTabBar::tab:selected{background:#3d5a80}
QProgressBar{border:1px solid #33363d;text-align:center} QProgressBar::chunk{background:#3d7dd8}
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME); app.setOrganizationName(ORG)
    s = QSettings(ORG, APP_NAME)
    set_lang(s.value("ui_lang", "ru"))
    app.setStyleSheet(DARK)
    icon = Path(__file__).with_name("icon.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow(); w.show()
    sys.exit(app.exec())
