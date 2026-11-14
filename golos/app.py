"""Главное окно Golos."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings, QUrl, QMimeData
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QDesktopServices, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QProgressBar, QFileDialog, QMessageBox, QPlainTextEdit, QDockWidget,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSplitter, QDialog, QDialogButtonBox, QAbstractItemView, QToolButton,
    QListWidget, QListWidgetItem)

from . import __version__, APP_NAME
from .i18n import T, set_lang
from .project import Project, Word, SR, new_project_dir
from .widgets import WaveformPanel, TranscriptView
from .workers import Task
from . import audio_io, bridge, models

ORG = "GolosApp"


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


def default_root() -> Path:
    docs = Path(os.path.expanduser("~")) / "Documents"
    return (docs if docs.exists() else Path.home()) / "Golos"


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
        cur = int(s.value("mic", -1))
        k = self.mic.findData(cur); self.mic.setCurrentIndex(max(0, k))
        self.src = QComboBox()
        for code in ("ru", "kk", "en"):
            self.src.addItem(models.TTS_LANGS[code][2], code)
        self.src.setCurrentIndex(max(0, self.src.findData(s.value("source_lang", "ru"))))
        self.root = QLineEdit(str(s.value("root", str(default_root()))))
        b = QPushButton("…"); b.clicked.connect(self._pick)
        row = QHBoxLayout(); row.addWidget(self.root); row.addWidget(b)
        self.prec = QComboBox()
        self.prec.addItem("fp16 (" + T("быстро, рекомендуется для RTX 20xx", "fast, recommended for RTX 20xx") + ")", "fp16")
        self.prec.addItem("fp32 (" + T("медленнее, если fp16 даёт брак", "slower, if fp16 glitches") + ")", "fp32")
        self.prec.setCurrentIndex(max(0, self.prec.findData(s.value("precision", "fp16"))))
        f.addRow(T("Язык интерфейса", "Interface language"), self.ui_lang)
        f.addRow(T("Микрофон", "Microphone"), self.mic)
        f.addRow(T("Язык начитки", "Recording language"), self.src)
        f.addRow(T("Папка проектов", "Projects folder"), row)
        f.addRow(T("Точность синтеза", "Synthesis precision"), self.prec)
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
        self.s.setValue("ui_lang", self.ui_lang.currentData())
        self.s.setValue("mic", self.mic.currentData())
        self.s.setValue("source_lang", self.src.currentData())
        self.s.setValue("root", self.root.text())
        self.s.setValue("precision", self.prec.currentData())
        if old != self.ui_lang.currentData():
            QMessageBox.information(self, APP_NAME, T("Язык интерфейса сменится после перезапуска.",
                                                      "Interface language changes after restart."))
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.s = QSettings(ORG, APP_NAME)
        self.p: Project | None = None
        self.rec: audio_io.Recorder | None = None
        self.player = audio_io.Player()
        self.play_mode = None          # "orig" | "edit" | ("dub", lang) | "variant"
        self.play_regions = None
        self.play_offset = 0.0
        self.task: Task | None = None
        self.cursor_t = 0.0
        self._clip_ignore = ""
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1400, 900)
        self._build()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(40)
        QApplication.clipboard().dataChanged.connect(self._clipboard_changed)
        last = self.s.value("last_project", "")
        if last and Path(last, "project.json").exists():
            try:
                self.open_project(Path(last))
            except Exception as e:  # noqa: BLE001
                self.log(f"Не удалось открыть последний проект: {e}")
        self._update_state()

    # ================= построение интерфейса =================
    def _build(self):
        tb = self.addToolBar("main"); tb.setMovable(False)
        for text, key, fn in [
            (T("Новый", "New"), QKeySequence.New, self.new_project),
            (T("Открыть…", "Open…"), QKeySequence.Open, self.open_dialog),
            (T("Импорт аудио…", "Import audio…"), None, self.import_audio),
            (T("Папка проекта", "Project folder"), None, self.open_folder),
            (T("Настройки", "Settings"), None, self.settings),
        ]:
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn); tb.addAction(a)
        self.title_lbl = QLabel(); tb.addSeparator(); tb.addWidget(self.title_lbl)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_edit_tab(), T("1. Запись и монтаж", "1. Record & edit"))
        self.tabs.addTab(self._build_dub_tab(), T("2. Перевод и озвучка", "2. Translate & voice"))
        self.tabs.addTab(self._build_export_tab(), T("3. Экспорт", "3. Export"))
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
        self.resizeDocks([dock], [150], Qt.Vertical)

        for seq, fn in [("Space", self.toggle_play), ("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo),
                        ("Ctrl+Shift+Z", self.redo),
                        ("Ctrl+=", lambda: self.wave.view.zoom(0.7)),
                        ("Ctrl+-", lambda: self.wave.view.zoom(1.4)),
                        ("Ctrl+0", lambda: self.wave.view.zoom_all())]:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)
        self.log(f"{APP_NAME} {__version__}. " + T("Видеокарта: ", "GPU: ") + models.torch_info())

    def _build_edit_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.rec_btn = QPushButton("● " + T("Запись", "Record"))
        self.rec_btn.setStyleSheet("QPushButton{background:#a3242f;color:white;font-weight:bold;padding:8px 16px}")
        self.rec_btn.clicked.connect(self.toggle_record)
        self.level = QProgressBar(); self.level.setRange(0, 100); self.level.setTextVisible(False)
        self.level.setMaximumWidth(220)
        self.rec_time = QLabel("0:00.0"); self.rec_time.setMinimumWidth(60)
        self.play_orig_btn = QPushButton("▶ " + T("Исходник", "Original"))
        self.play_orig_btn.clicked.connect(lambda: self.play("orig"))
        self.play_edit_btn = QPushButton("▶ " + T("Результат", "Result"))
        self.play_edit_btn.clicked.connect(lambda: self.play("edit"))
        self.stop_btn = QPushButton("■"); self.stop_btn.clicked.connect(self.stop_play)
        self.asr_btn = QPushButton(T("Распознать и найти дубли", "Recognize & find retakes"))
        self.asr_btn.clicked.connect(self.run_asr)
        self.stats = QLabel()
        for x in (self.rec_btn, self.level, self.rec_time, self.play_orig_btn, self.play_edit_btn,
                  self.stop_btn, self.asr_btn):
            row.addWidget(x)
        row.addStretch(1); row.addWidget(self.stats)
        v.addLayout(row)

        self.wave = WaveformPanel()
        self.wave.view.seek.connect(self.on_seek)
        self.wave.view.selection_changed.connect(self.on_selection)
        self.wave.view.cut_requested.connect(self.cut_selection)
        wrow = QHBoxLayout()
        for text, fn, tip in [
            ("✂ " + T("Вырезать выделение", "Cut selection"), self.cut_selection, "Delete"),
            ("↺ " + T("Вернуть выделение", "Restore selection"), self.restore_selection, ""),
            (T("Образец голоса = выделение", "Voice sample = selection"), self.set_ref, ""),
            ("🔍 " + T("К выделению", "Zoom to selection"), lambda: self.wave.view.zoom_sel(), ""),
            (T("Вся запись", "Fit all"), lambda: self.wave.view.zoom_all(), "Ctrl+0"),
            ("↶", self.undo, "Ctrl+Z"), ("↷", self.redo, "Ctrl+Y"),
        ]:
            b = QPushButton(text); b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            wrow.addWidget(b)
        wrow.addStretch(1)
        self.hint = QLabel(T("Колесо — зум, Shift+колесо — прокрутка, протяжка — выделение, пробел — play/pause.",
                             "Wheel — zoom, Shift+wheel — scroll, drag — select, Space — play/pause."))
        self.hint.setStyleSheet("color:#888"); wrow.addWidget(self.hint)

        split = QSplitter(Qt.Vertical)
        top = QWidget(); tl = QVBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.wave, 1); tl.addLayout(wrow)
        split.addWidget(top)

        bottom = QWidget(); bl = QHBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        self.trans = TranscriptView()
        self.trans.toggle_words.connect(self.toggle_words)
        self.trans.seek_word.connect(self.on_seek_word)
        tbox = QVBoxLayout(); tw = QWidget(); tw.setLayout(tbox); tbox.setContentsMargins(0, 0, 0, 0)
        legend = QLabel(T("Выделите слова и нажмите Delete, чтобы вырезать или вернуть. "
                          "<span style='color:#ff6b6b'>Красные</span> — найденные дубли, "
                          "<span style='color:#c792ea'>сиреневые</span> — «э-э», "
                          "<span style='color:#ffc400'>жёлтые</span> — речь, которую не удалось распознать (по умолчанию вырезана).",
                          "Select words and press Delete to cut or restore. "
                          "<span style='color:#ff6b6b'>Red</span> — detected retakes, "
                          "<span style='color:#c792ea'>purple</span> — fillers, "
                          "<span style='color:#ffc400'>yellow</span> — unrecognized speech (cut by default)."))
        legend.setWordWrap(True); legend.setTextFormat(Qt.RichText)
        tbox.addWidget(legend); tbox.addWidget(self.trans, 1)
        bl.addWidget(tw, 3)

        g = QGroupBox(T("Обработка", "Processing")); f = QFormLayout(g)
        self.cb_pauses = QCheckBox(T("Сокращать паузы", "Shorten pauses"))
        self.sp_max = self._dspin(0.2, 3, 0.05, " s"); self.sp_target = self._dspin(0.1, 2, 0.05, " s")
        self.sp_sent = self._dspin(0.1, 3, 0.05, " s")
        self.cb_dn = QCheckBox(T("Шумодав", "Noise reduction"))
        self.sp_dn = self._dspin(0.1, 1.0, 0.05, "")
        self.cb_ln = QCheckBox(T("Громкость (LUFS)", "Loudness (LUFS)"))
        self.sp_ln = self._dspin(-30, -8, 1, " LUFS")
        f.addRow(self.cb_pauses)
        f.addRow(T("Паузы длиннее", "Pauses longer than"), self.sp_max)
        f.addRow(T("укорачивать до", "shorten to"), self.sp_target)
        f.addRow(T("Пауза после точки", "Pause after sentence"), self.sp_sent)
        f.addRow(self.cb_dn, self.sp_dn)
        f.addRow(self.cb_ln, self.sp_ln)
        b1 = QPushButton(T("Найти дубли заново", "Re-detect retakes")); b1.clicked.connect(self.redetect)
        b2 = QPushButton(T("Вернуть всё вырезанное", "Restore everything")); b2.clicked.connect(self.restore_all)
        f.addRow(b1); f.addRow(b2)
        for wdg in (self.cb_pauses, self.cb_dn, self.cb_ln):
            wdg.toggled.connect(self.on_edit_settings)
        for wdg in (self.sp_max, self.sp_target, self.sp_sent, self.sp_dn, self.sp_ln):
            wdg.valueChanged.connect(self.on_edit_settings)
        bl.addWidget(g, 1)
        split.addWidget(bottom)
        split.setSizes([420, 380])
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
        g = QGroupBox(T("Языки озвучки", "Voice-over languages")); gl = QVBoxLayout(g)
        self.lang_list = QListWidget(); self.lang_list.setMaximumHeight(150)
        for code, (_, _, label) in models.TTS_LANGS.items():
            it = QListWidgetItem(f"{label} ({code})" + ("  — " + T("нейро-версия начитки", "neural re-voice")
                                                       if code == "ru" else ""))
            it.setData(Qt.UserRole, code)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            saved = self.s.value("dub_langs", "en,kk").split(",")
            it.setCheckState(Qt.Checked if code in saved else Qt.Unchecked)
            self.lang_list.addItem(it)
        self.lang_list.itemChanged.connect(self._langs_changed)
        gl.addWidget(self.lang_list)
        top.addWidget(g, 1)

        g2 = QGroupBox(T("Перевод через чат с Claude", "Translate via Claude chat")); f = QVBoxLayout(g2)
        self.ctx = QLineEdit(); self.ctx.setPlaceholderText(T("Контекст для перевода (необязательно): о чём ролик",
                                                              "Context (optional): what the video is about"))
        self.ctx.setText(self.s.value("ctx", ""))
        self.copy_req_btn = QPushButton("① " + T("Скопировать запрос на перевод", "Copy translation request"))
        self.copy_req_btn.clicked.connect(self.copy_request)
        self.paste_btn = QPushButton("② " + T("Вставить перевод из буфера", "Paste translation from clipboard"))
        self.paste_btn.clicked.connect(self.paste_translation)
        self.auto_paste = QCheckBox(T("Подхватывать перевод из буфера автоматически", "Pick up translation from clipboard automatically"))
        self.auto_paste.setChecked(self.s.value("auto_paste", "true") == "true")
        tip = QLabel(T("Вставьте запрос в чат (Ctrl+V), затем нажмите «Копировать» под ответом — перевод подставится сам.",
                       "Paste the request into the chat (Ctrl+V), then press Copy under the answer — it is applied automatically."))
        tip.setWordWrap(True); tip.setStyleSheet("color:#999")
        for x in (self.ctx, self.copy_req_btn, self.paste_btn, self.auto_paste, tip):
            f.addWidget(x)
        top.addWidget(g2, 2)

        g3 = QGroupBox(T("Озвучка своим голосом", "Voice with your voice")); f3 = QFormLayout(g3)
        self.sp_var = QSpinBox(); self.sp_var.setRange(1, 5); self.sp_var.setValue(int(self.s.value("variants", 3)))
        self.cb_verify = QCheckBox(T("Проверять варианты распознаванием", "Verify variants with ASR"))
        self.cb_verify.setChecked(self.s.value("verify", "true") == "true")
        self.sp_temp = self._dspin(0.3, 1.2, 0.05, ""); self.sp_temp.setValue(float(self.s.value("temp", 0.8)))
        self.dub_btn = QPushButton("③ " + T("Озвучить", "Generate voice-over"))
        self.dub_btn.setStyleSheet("QPushButton{font-weight:bold;padding:8px}")
        self.dub_btn.clicked.connect(lambda: self.run_dub(None))
        f3.addRow(T("Вариантов на фразу", "Variants per phrase"), self.sp_var)
        f3.addRow(T("Живость (temperature)", "Expressiveness"), self.sp_temp)
        f3.addRow(self.cb_verify)
        f3.addRow(self.dub_btn)
        top.addWidget(g3, 1)
        v.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel(T("Показать язык:", "Show language:")))
        self.view_lang = QComboBox(); self.view_lang.currentIndexChanged.connect(self.fill_dub_table)
        row.addWidget(self.view_lang)
        self.play_track_btn = QPushButton("▶ " + T("Слушать дорожку", "Play track"))
        self.play_track_btn.clicked.connect(self.play_track)
        self.regen_btn = QPushButton("↻ " + T("Перегенерировать выделенные фразы", "Regenerate selected phrases"))
        self.regen_btn.clicked.connect(self.regen_selected)
        stop = QPushButton("■"); stop.clicked.connect(self.stop_play)
        row.addWidget(self.play_track_btn); row.addWidget(stop); row.addWidget(self.regen_btn)
        row.addStretch(1)
        self.dub_info = QLabel(); row.addWidget(self.dub_info)
        v.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", T("Длит.", "Dur."), T("Оригинал", "Original"),
                                              T("Перевод (можно править)", "Translation (editable)"),
                                              T("Варианты — клик: слушать и выбрать", "Variants — click to play & pick")])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self.on_table_edit)
        v.addWidget(self.table, 1)
        return w

    def _build_export_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        info = QLabel(T(
            "Пакет для видеочата: смонтированная начитка, дорожки озвучки (WAV 48 кГц/24 бит и MP3), "
            "субтитры SRT, каждая фраза отдельным файлом и timings.json с началом и концом каждой фразы.",
            "Package for the video chat: edited recording, voice tracks (WAV 48 kHz/24-bit and MP3), "
            "SRT subtitles, one file per phrase and timings.json with every phrase's start/end."))
        info.setWordWrap(True); v.addWidget(info)
        row = QHBoxLayout()
        self.export_btn = QPushButton(T("Собрать пакет и скопировать в буфер", "Build package & copy to clipboard"))
        self.export_btn.setStyleSheet("QPushButton{font-weight:bold;padding:10px}")
        self.export_btn.clicked.connect(self.export_package)
        self.copy_zip_btn = QPushButton(T("Скопировать последний пакет ещё раз", "Copy last package again"))
        self.copy_zip_btn.clicked.connect(lambda: self.copy_file(self.last_zip))
        self.open_exp_btn = QPushButton(T("Показать в папке", "Show in folder"))
        self.open_exp_btn.clicked.connect(lambda: self.reveal(self.last_zip))
        self.export_src_btn = QPushButton(T("Только начитку: сохранить WAV/MP3…", "Recording only: save WAV/MP3…"))
        self.export_src_btn.clicked.connect(self.export_source_only)
        for b in (self.export_btn, self.copy_zip_btn, self.open_exp_btn, self.export_src_btn):
            row.addWidget(b)
        row.addStretch(1); v.addLayout(row)
        self.export_lbl = QLabel(); self.export_lbl.setWordWrap(True); v.addWidget(self.export_lbl)
        tip = QLabel(T("После «Собрать пакет» zip уже в буфере обмена: откройте чат с Claude и нажмите Ctrl+V. "
                       "Если чат не принял вставку — перетащите файл из папки.",
                       "After building, the zip is on the clipboard: open the Claude chat and press Ctrl+V. "
                       "If paste doesn't work, drag the file from the folder."))
        tip.setWordWrap(True); tip.setStyleSheet("color:#999"); v.addWidget(tip)
        v.addStretch(1)
        self.last_zip: Path | None = None
        return w

    # ================= общие =================
    def log(self, msg: str):
        self.logbox.appendPlainText(time.strftime("%H:%M:%S ") + msg)

    def _update_state(self):
        has = self.p is not None
        has_audio = has and len(self.p.audio) > 0
        has_words = has_audio and bool(self.p.words)
        busy = self.task is not None and self.task.isRunning()
        for b in (self.play_orig_btn, self.play_edit_btn):
            b.setEnabled(has_audio)
        self.asr_btn.setEnabled(has_audio and not busy)
        for b in (self.copy_req_btn, self.paste_btn, self.dub_btn, self.export_btn, self.export_src_btn):
            b.setEnabled(has_words and not busy)
        self.copy_zip_btn.setEnabled(self.last_zip is not None)
        self.open_exp_btn.setEnabled(self.last_zip is not None)
        self.title_lbl.setText(f"  {self.p.name}" if has else T("  нет проекта", "  no project"))
        if has_audio:
            self.stats.setText(T("Было ", "Before ") + fmt_time(self.p.duration) + " → " +
                               T("стало ", "after ") + fmt_time(self.p.edited_duration()))
        else:
            self.stats.setText("")

    def _load_edit_settings(self):
        e = self.p.edit
        for wdg, val in [(self.cb_pauses, e.shorten_pauses), (self.cb_dn, e.denoise), (self.cb_ln, e.loudness)]:
            wdg.blockSignals(True); wdg.setChecked(val); wdg.blockSignals(False)
        for wdg, val in [(self.sp_max, e.max_pause), (self.sp_target, e.target_pause), (self.sp_sent, e.sentence_pause),
                         (self.sp_dn, e.denoise_strength), (self.sp_ln, e.target_lufs)]:
            wdg.blockSignals(True); wdg.setValue(val); wdg.blockSignals(False)

    def on_edit_settings(self, *_):
        if not self.p:
            return
        self.p.checkpoint()
        e = self.p.edit
        e.shorten_pauses = self.cb_pauses.isChecked(); e.max_pause = self.sp_max.value()
        e.target_pause = self.sp_target.value(); e.sentence_pause = self.sp_sent.value()
        e.denoise = self.cb_dn.isChecked(); e.denoise_strength = self.sp_dn.value()
        e.loudness = self.cb_ln.isChecked(); e.target_lufs = self.sp_ln.value()
        self.edits_changed()

    def edits_changed(self, rebuild_text: bool = False):
        if not self.p:
            return
        self.p.save()
        self.wave.view.refresh()
        if rebuild_text:
            self.trans.rebuild()
        self._update_state()

    # ================= проекты =================
    def root(self) -> Path:
        r = Path(self.s.value("root", str(default_root())))
        r.mkdir(parents=True, exist_ok=True)
        return r

    def new_project(self):
        self.stop_play()
        d = new_project_dir(self.root())
        i = 2
        base = d
        while d.exists():
            d = Path(f"{base} ({i})"); i += 1
        self.set_project(Project.create(d))
        self.p.source_lang = self.s.value("source_lang", "ru")
        self.p.save()
        self.log(T("Новый проект: ", "New project: ") + str(d))

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
        self.s.setValue("last_project", str(p.folder))
        self.wave.view.set_project(p)
        self.trans.set_project(p)
        self._load_edit_settings()
        self.refresh_dub_view()
        self._update_state()

    def open_folder(self):
        if self.p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.p.folder)))

    def settings(self):
        if SettingsDialog(self.s, self).exec():
            if self.p and not self.p.words:
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
        self.run_asr()

    # ================= запись =================
    def toggle_record(self):
        if self.rec is None:
            self.stop_play()
            if self.p is not None and len(self.p.audio):
                r = QMessageBox.question(self, APP_NAME, T("Начать новую запись в новом проекте?",
                                                           "Start a new recording in a new project?"))
                if r != QMessageBox.Yes:
                    return
            if self.p is None or len(self.p.audio):
                self.new_project()
            dev = int(self.s.value("mic", -1))
            self.rec = audio_io.Recorder(None if dev < 0 else dev)
            try:
                self.rec.start()
            except Exception as e:  # noqa: BLE001
                self.rec = None
                QMessageBox.warning(self, APP_NAME, T("Микрофон не открылся: ", "Mic failed: ") + str(e))
                return
            self.rec_btn.setText("■ " + T("Стоп", "Stop"))
            self.log(T("Запись…", "Recording…"))
        else:
            x, sr = self.rec.stop()
            self.rec = None
            self.rec_btn.setText("● " + T("Запись", "Record"))
            self.level.setValue(0)
            if len(x) < sr * 0.5:
                self.log(T("Запись слишком короткая", "Recording too short")); return
            from .dsp import resample
            self.p.set_audio(resample(x, sr, SR))
            self.set_project(self.p)
            self.log(T("Записано ", "Recorded ") + fmt_time(self.p.duration))
            self.run_asr()

    def _tick(self):
        if self.rec is not None:
            lv = self.rec.level
            self.level.setValue(int(min(1.0, lv) * 100))
            self.level.setStyleSheet("QProgressBar::chunk{background:%s}" % ("#e0433d" if lv > 0.89 else "#3fbf6f"))
            self.rec_time.setText(fmt_time(self.rec.seconds))
        if self.player.playing and self.play_mode in ("orig", "edit"):
            t = self.player.position()
            orig = t if self.play_mode == "orig" else self.p.edit_to_orig(t, self.play_regions)
            self.wave.view.set_head(orig)
            self.cursor_t = orig
            self._highlight_at(orig)
        elif self.play_mode and not self.player.playing:
            self.play_mode = None

    def _highlight_at(self, t: float):
        if not self.p or not self.p.words:
            return
        i = next((k for k, w in enumerate(self.p.words) if w.start <= t <= w.end + 0.05), None)
        self.trans.highlight_word(i)

    # ================= воспроизведение =================
    def play(self, mode: str, start_orig: float | None = None):
        if not self.p or not len(self.p.audio):
            return
        start_orig = self.cursor_t if start_orig is None else start_orig
        if start_orig >= self.p.duration - 0.05:
            start_orig = 0.0
        if mode == "orig":
            self.player.play(self.p.audio, SR, start_orig)
        else:
            self.play_regions = self.p.keep_regions()
            y = self.p.render(process=False, regions=self.play_regions)
            self.player.play(y, SR, self.p.orig_to_edit(start_orig, self.play_regions) or 0.0)
        self.play_mode = mode

    def toggle_play(self):
        if isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox)):
            return
        if self.player.playing:
            self.stop_play()
        elif self.tabs.currentIndex() == 1:
            self.play_track()
        else:
            self.play(self.play_mode_pref())

    def play_mode_pref(self):
        return "edit" if self.p and self.p.words else "orig"

    def stop_play(self):
        self.player.stop()
        self.play_mode = None

    def on_seek(self, t: float):
        self.cursor_t = t
        self.wave.view.set_head(t)
        self._highlight_at(t)
        if self.player.playing and self.play_mode in ("orig", "edit"):
            self.play(self.play_mode, t)

    def on_seek_word(self, i: int, play: bool):
        t = self.p.words[i].start
        self.on_seek(t)
        if play:
            self.play("edit" if not self.p.words[i].cut else "orig", t)

    def on_selection(self, a: float, b: float):
        pass

    # ================= монтаж =================
    def cut_selection(self):
        sel = self.wave.view.sel
        if not self.p or not sel:
            return
        self.p.checkpoint()
        a, b = sel
        self.p.manual_cuts.append([a, b])
        for w in self.p.words:
            if w.start >= a - 0.02 and w.end <= b + 0.02:
                w.cut = True
        self.wave.view.sel = None
        self.edits_changed(rebuild_text=True)

    def restore_selection(self):
        sel = self.wave.view.sel
        if not self.p or not sel:
            return
        self.p.checkpoint()
        a, b = sel
        from .project import _subtract
        self.p.manual_cuts = [list(c) for c in _subtract([tuple(c) for c in self.p.manual_cuts], [(a, b)])]
        for w in self.p.words:
            if w.end > a and w.start < b:
                w.cut, w.auto = False, ""
        for g in self.p.gaps:
            if g.end > a and g.start < b:
                g.cut = False
        self.edits_changed(rebuild_text=True)

    def toggle_words(self, ws: list, gs: list):
        self.p.checkpoint()
        items = [self.p.words[i] for i in ws] + [self.p.gaps[i] for i in gs]
        new = not all(x.cut for x in items)
        for x in items:
            x.cut = new
            if isinstance(x, Word):
                x.auto = "" if not new else x.auto
        self.edits_changed(rebuild_text=True)

    def redetect(self):
        if not self.p:
            return
        self.p.checkpoint(); self.p.auto_mark()
        self.edits_changed(rebuild_text=True)

    def restore_all(self):
        if not self.p:
            return
        self.p.checkpoint()
        for w in self.p.words:
            w.cut, w.auto = False, ""
        for g in self.p.gaps:
            g.cut = False
        self.p.manual_cuts = []
        self.edits_changed(rebuild_text=True)

    def undo(self):
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        if self.p and self.p.undo():
            self._load_edit_settings(); self.edits_changed(rebuild_text=True)

    def redo(self):
        if self.p and self.p.redo():
            self._load_edit_settings(); self.edits_changed(rebuild_text=True)

    def set_ref(self):
        sel = self.wave.view.sel
        if not self.p:
            return
        if not sel:
            self.p.ref_override = None
            self.log(T("Образец голоса: автоматически", "Voice sample: automatic"))
        else:
            a, b = sel
            if not 4 <= b - a <= 20:
                QMessageBox.information(self, APP_NAME, T("Выделите 5–15 секунд чистой речи.",
                                                          "Select 5–15 seconds of clean speech."))
                return
            self.p.ref_override = [a, b]
            self.log(T("Образец голоса: ", "Voice sample: ") + f"{fmt_time(a)}–{fmt_time(b)}")
        self.p.save(); self.wave.view.update()

    # ================= фоновые задачи =================
    def start_task(self, fn, done, *args, **kw):
        if self.task and self.task.isRunning():
            return
        self.stop_play()
        t = Task(fn, *args, **kw)
        t.log.connect(self.log)
        t.progress.connect(self._on_progress)
        t.failed.connect(lambda m: (self.log("ОШИБКА: " + m),
                                    QMessageBox.warning(self, APP_NAME, m)))
        t.succeeded.connect(done)
        t.finished.connect(self._task_finished)
        self.task = t
        self.prog.setVisible(True); self.prog.setRange(0, 0); self.cancel_btn.setVisible(True)
        t.start()
        self._update_state()

    def _on_progress(self, a, b):
        self.prog.setRange(0, max(1, b)); self.prog.setValue(a)

    def _task_finished(self):
        self.prog.setVisible(False); self.cancel_btn.setVisible(False)
        self.task = None
        self._update_state()

    # ----- распознавание -----
    def run_asr(self):
        if not self.p or not len(self.p.audio):
            return
        audio, lang = self.p.audio.copy(), self.p.source_lang

        def job(task: Task):
            task.log.emit(T("Распознаю речь (Whisper)…", "Recognizing speech (Whisper)…"))
            wh = models.Whisper(task.log.emit)
            try:
                return wh.words(audio, SR, lang)
            finally:
                wh.close()

        def done(words):
            self.p.checkpoint()
            self.p.set_words([Word(t, s, e) for t, s, e in words])
            nd = sum(1 for w in self.p.words if w.auto == "dup")
            nf = sum(1 for w in self.p.words if w.auto == "filler")
            self.log(T(f"Слов: {len(self.p.words)}; вырезано дублей: {nd} сл., паразитов: {nf}; "
                       f"нераспознанных кусков: {len(self.p.gaps)}",
                       f"Words: {len(self.p.words)}; retake words cut: {nd}; fillers: {nf}; "
                       f"unrecognized pieces: {len(self.p.gaps)}"))
            self.wave.view.refresh(); self.trans.rebuild(); self.refresh_dub_view(); self._update_state()

        self.start_task(job, done)

    # ================= перевод =================
    def selected_langs(self) -> list[str]:
        out = []
        for i in range(self.lang_list.count()):
            it = self.lang_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _langs_changed(self, *_):
        self.s.setValue("dub_langs", ",".join(self.selected_langs()))
        self.refresh_dub_view()

    def copy_request(self):
        langs = [l for l in self.selected_langs() if l != self.p.source_lang]
        if not langs:
            QMessageBox.information(self, APP_NAME, T("Отметьте языки перевода.", "Tick target languages.")); return
        self.s.setValue("ctx", self.ctx.text())
        text = bridge.build_request(self.p.phrases(), langs, self.p.source_lang, self.ctx.text())
        self._clip_ignore = text
        QApplication.clipboard().setText(text)
        self.log(T("Запрос на перевод в буфере обмена — вставьте его в чат с Claude.",
                   "Translation request copied — paste it into the Claude chat."))

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
        except ValueError as e:
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
            miss = ids - set(items)
            msgs.append(f"{lang}: {len(items)}" + (T(f", нет фраз {sorted(miss)}", f", missing {sorted(miss)}") if miss else ""))
            for i in range(self.lang_list.count()):
                it = self.lang_list.item(i)
                if it.data(Qt.UserRole) == lang:
                    it.setCheckState(Qt.Checked)
        self.p.save()
        self.refresh_dub_view()
        self.tabs.setCurrentIndex(1)
        self.log(T("Перевод вставлен: ", "Translation applied: ") + "; ".join(msgs))
        self.raise_(); self.activateWindow()

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
            self.table.blockSignals(False); return
        phrases = self.p.phrases()
        texts = {ph.id: ph.text for ph in phrases} if lang == self.p.source_lang else self.p.translations.get(lang, {})
        dubs = {d.id: d for d in self.p.dub.get(lang, [])}
        self.table.setRowCount(len(phrases))
        n_ok = n_long = 0
        for r, ph in enumerate(phrases):
            it = QTableWidgetItem(str(ph.id)); it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setData(Qt.UserRole, ph.id)
            self.table.setItem(r, 0, it)
            it = QTableWidgetItem(f"{ph.dur:.1f}s"); it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(r, 1, it)
            it = QTableWidgetItem(ph.text); it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(r, 2, it)
            it = QTableWidgetItem(texts.get(ph.id, ""))
            if lang == self.p.source_lang:
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(r, 3, it)
            d = dubs.get(ph.id)
            cell = QWidget(); hl = QHBoxLayout(cell); hl.setContentsMargins(2, 2, 2, 2); hl.setSpacing(3)
            if d and d.variants:
                stale = d.text != texts.get(ph.id, "")
                for vi, v in enumerate(d.variants):
                    st = fit_status(v, ph.dur)
                    col = {"ok": "#2e7d4f", "stretch": "#a0771a", "long": "#a3242f"}[st]
                    warn = " ⚠" if (v.cer is not None and v.cer > 0.25) else ""
                    b = QToolButton(); b.setText(f"{vi + 1}: {v.dur:.1f}s{warn}")
                    sel = vi == d.chosen
                    b.setStyleSheet(f"QToolButton{{background:{col};color:white;padding:3px 6px;"
                                    f"border:{'2px solid #ffcc33' if sel else '1px solid #333'};"
                                    f"font-weight:{'bold' if sel else 'normal'}}}")
                    tip = T(f"Длительность {v.dur:.2f} с при слоте {ph.dur:.2f} с", f"{v.dur:.2f}s for a {ph.dur:.2f}s slot")
                    if v.cer is not None:
                        tip += T(f"\nРасхождение с текстом: {v.cer * 100:.0f}%", f"\nText mismatch: {v.cer * 100:.0f}%")
                    b.setToolTip(tip)
                    b.clicked.connect(lambda _=False, L=lang, pid=ph.id, k=vi: self.pick_variant(L, pid, k))
                    hl.addWidget(b)
                if d.chosen >= 0:
                    st = fit_status(d.variants[d.chosen], ph.dur)
                    n_ok += st != "long"; n_long += st == "long"
                if stale:
                    lbl = QLabel(T("текст изменён", "text changed")); lbl.setStyleSheet("color:#ffb300"); hl.addWidget(lbl)
            else:
                lbl = QLabel("—"); lbl.setStyleSheet("color:#777"); hl.addWidget(lbl)
            self.table.setCellWidget(r, 4, cell)
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)
        self.dub_info.setText(T(f"Выбрано: {n_ok + n_long}/{len(phrases)}, не влезают: {n_long}",
                                f"Chosen: {n_ok + n_long}/{len(phrases)}, too long: {n_long}"))

    def on_table_edit(self, item: QTableWidgetItem):
        if item.column() != 3:
            return
        lang = self.view_lang.currentData()
        pid = self.table.item(item.row(), 0).data(Qt.UserRole)
        self.p.translations.setdefault(lang, {})[pid] = item.text().strip()
        self.p.save()
        QTimer.singleShot(0, self.fill_dub_table)

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

    def run_dub(self, only_ids):
        langs = self.selected_langs()
        if not langs:
            return
        missing = [l for l in langs if l != self.p.source_lang and not self.p.translations.get(l)]
        if missing and only_ids is None:
            r = QMessageBox.question(self, APP_NAME, T(f"Нет перевода для: {', '.join(missing)}. Озвучить остальные?",
                                                       f"No translation for: {', '.join(missing)}. Voice the rest?"))
            if r != QMessageBox.Yes:
                return
            langs = [l for l in langs if l not in missing]
        if not langs:
            return
        self.s.setValue("variants", self.sp_var.value()); self.s.setValue("verify", str(self.cb_verify.isChecked()).lower())
        self.s.setValue("temp", self.sp_temp.value())
        from .dub import Dubber
        p, nv, ver, temp = self.p, self.sp_var.value(), self.cb_verify.isChecked(), self.sp_temp.value()
        prec = self.s.value("precision", "fp16")

        def job(task: Task):
            Dubber(p, task.log.emit, lambda a, b: task.progress.emit(a, b), task.is_cancelled, prec).run(
                langs, nv, ver, only_ids, temp)

        def done(_):
            self.refresh_dub_view()
            self.log(T("Озвучка готова. Послушайте варианты и выберите лучшие.",
                       "Voice-over ready. Listen and pick the best variants."))

        self.tabs.setCurrentIndex(1)
        self.start_task(job, done)

    def regen_selected(self):
        lang = self.view_lang.currentData()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not lang or not rows:
            QMessageBox.information(self, APP_NAME, T("Выделите строки фраз.", "Select phrase rows.")); return
        ids = {self.table.item(r, 0).data(Qt.UserRole) for r in rows}
        self.run_dub({lang: ids})

    def play_track(self):
        lang = self.view_lang.currentData()
        if not lang or not self.p.dub.get(lang):
            return
        from .dub import build_track, OUT_SR
        try:
            x, _, warns = build_track(self.p, lang)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e)); return
        for w in warns:
            self.log(f"[{lang}] {w}")
        self.player.play(x, OUT_SR)
        self.play_mode = ("dub", lang)

    # ================= экспорт =================
    def export_package(self):
        from . import exporter
        langs = self.selected_langs()
        p = self.p

        def job(task: Task):
            return exporter.export_package(p, langs, task.log.emit)

        def done(z):
            self.last_zip = Path(z)
            self.copy_file(self.last_zip)
            self.export_lbl.setText(T("Готово и скопировано в буфер: ", "Done and copied: ") + str(z))
            self._update_state()

        self.start_task(job, done)

    def export_source_only(self):
        f, _ = QFileDialog.getSaveFileName(self, "", str(self.p.folder / f"{self.p.name}.wav"), "WAV (*.wav)")
        if not f:
            return
        from . import exporter
        y = self.p.render(process=True)
        exporter.write_audio(Path(f).with_suffix(""), y, SR)
        self.log(T("Сохранено: ", "Saved: ") + f)

    def copy_file(self, path: Path | None):
        if not path or not Path(path).exists():
            return
        md = QMimeData()
        md.setUrls([QUrl.fromLocalFile(str(path))])
        self._clip_ignore = ""
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
