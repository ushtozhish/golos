"""Вкладка «4. Обучение голоса»: запись фраз по блокам, автопроверка дублей, пакет для обучения, подключение модели."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QListWidget,
                               QListWidgetItem, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QMessageBox, QCheckBox, QSplitter, QFileDialog, QInputDialog,
                               QGroupBox, QToolButton)

from .i18n import T
from . import audio_io, trainset


class TrainTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.ts = trainset.TrainSet()
        self.cur = self.ts.lines[0].no
        self.rec = None
        self.rec_mode = None          # "take" | "noise"
        self.queue: list[tuple[int, str]] = []   # (no, file) ждут проверки текста
        self._build()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(40)
        self.show_line()

    # ---------------- интерфейс ----------------
    def _build(self):
        v = QVBoxLayout(self)
        steps = QHBoxLayout()
        self.mic_btn = QPushButton(T("① Проверить микрофон и тишину комнаты", "① Check mic & room noise"))
        self.mic_btn.clicked.connect(self.noise_check)
        self.export_btn = QPushButton(T("④ Собрать пакет для обучения…", "④ Build training package…"))
        self.export_btn.clicked.connect(self.export)
        self.install_btn = QPushButton(T("⑤ Подключить обученную модель…", "⑤ Install trained model…"))
        self.install_btn.clicked.connect(self.install)
        self.open_btn = QPushButton(T("Папка записей", "Recordings folder"))
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(trainset.root()))))
        for b in (self.mic_btn, self.export_btn, self.install_btn, self.open_btn):
            steps.addWidget(b)
        steps.addStretch(1)
        self.total_lbl = QLabel(); steps.addWidget(self.total_lbl)
        v.addLayout(steps)
        self.noise_lbl = QLabel(); self.noise_lbl.setStyleSheet("color:#999"); v.addWidget(self.noise_lbl)

        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        self.block = QComboBox()
        for b, name in trainset.BLOCK_NAMES.items():
            self.block.addItem(name, b)
        self.block.currentIndexChanged.connect(self._block_changed)
        lv.addWidget(self.block)
        self.block_lbl = QLabel(); self.block_lbl.setStyleSheet("color:#999"); lv.addWidget(self.block_lbl)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._list_pick)
        lv.addWidget(self.list, 1)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        self.no_lbl = QLabel(); self.no_lbl.setStyleSheet("color:#999"); rv.addWidget(self.no_lbl)
        self.text_lbl = QLabel(); self.text_lbl.setWordWrap(True); self.text_lbl.setAlignment(Qt.AlignCenter)
        f = QFont(self.font()); f.setPointSizeF(22); self.text_lbl.setFont(f)
        self.text_lbl.setMinimumHeight(150)
        self.text_lbl.setStyleSheet("background:#15171b;color:#f2f2f2;border:1px solid #33363d;padding:16px")
        rv.addWidget(self.text_lbl)
        self.hint_lbl = QLabel(); self.hint_lbl.setAlignment(Qt.AlignCenter); self.hint_lbl.setStyleSheet("color:#ffcc33")
        rv.addWidget(self.hint_lbl)
        row = QHBoxLayout()
        self.prev_btn = QPushButton("◄ " + T("Назад (←)", "Back (←)")); self.prev_btn.clicked.connect(self.prev)
        self.rec_btn = QPushButton("● " + T("Записать дубль (R)", "Record take (R)"))
        self.rec_btn.setStyleSheet("QPushButton{background:#a3242f;color:white;font-weight:bold;padding:12px 24px}")
        self.rec_btn.clicked.connect(self.toggle_record)
        self.next_btn = QPushButton(T("Дальше (→)", "Next (→)") + " ►"); self.next_btn.clicked.connect(self.next)
        self.level = QProgressBar(); self.level.setRange(0, 100); self.level.setTextVisible(False); self.level.setMaximumWidth(180)
        self.rec_time = QLabel("0.0 s")
        for w in (self.prev_btn, self.rec_btn, self.next_btn, self.level, self.rec_time):
            row.addWidget(w)
        row.addStretch(1)
        rv.addLayout(row)
        self.auto_next = QCheckBox(T("После хорошего дубля сразу переходить к следующей фразе",
                                     "Go to the next phrase after a good take"))
        self.auto_next.setChecked(self.win.s.value("train_auto_next", "false") == "true")
        self.auto_next.toggled.connect(lambda c: self.win.s.setValue("train_auto_next", str(c).lower()))
        rv.addWidget(self.auto_next)
        rv.addWidget(QLabel(T("Дубли этой фразы (можно сколько угодно — в обучение идут только зелёные):",
                              "Takes of this phrase (any number — only green ones go to training):")))
        self.takes = QTableWidget(0, 4)
        self.takes.setHorizontalHeaderLabels(["", T("Длит.", "Dur."), T("Проверка", "Check"), ""])
        h = self.takes.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents); h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch); h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.takes.verticalHeader().setVisible(False)
        self.takes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.takes.setWordWrap(True)
        rv.addWidget(self.takes, 1)
        tip = QLabel(T("R — запись/стоп · → / ← — следующая/предыдущая фраза · пробел — послушать последний дубль. "
                       "Читайте каждый блок в одном настроении; паузу в 1 с до и после фразы программа обрежет сама.",
                       "R — record/stop · → / ← — next/previous · Space — play last take."))
        tip.setWordWrap(True); tip.setStyleSheet("color:#888"); rv.addWidget(tip)
        split.addWidget(right)
        split.setSizes([380, 900])
        v.addWidget(split, 1)

    # ---------------- список и фраза ----------------
    def _block_lines(self):
        b = self.block.currentData()
        return [l for l in self.ts.lines if l.block == b]

    def _block_changed(self, *_):
        lines = self._block_lines()
        if lines and self.ts.by_no[self.cur].block != self.block.currentData():
            self.cur = lines[0].no
        self.fill_list()
        self.show_line()

    def fill_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        sel = 0
        for i, l in enumerate(self._block_lines()):
            good = len(self.ts.good(l.no))
            bad = sum(1 for t in self.ts.takes.get(l.no, []) if t.ok is False)
            wait = sum(1 for t in self.ts.takes.get(l.no, []) if t.ok is None)
            mark = f"✓{good}" if good else ("✗" if bad else "·")
            it = QListWidgetItem(f"{mark:>3}  {l.no}. {l.text}")
            it.setForeground(QColor("#6fd08c") if good else QColor("#ff6b6b") if bad else QColor("#bbbbbb"))
            if wait:
                it.setText(it.text() + "  ⏳")
            self.list.addItem(it)
            if l.no == self.cur:
                sel = i
        self.list.setCurrentRow(sel)
        self.list.blockSignals(False)
        st = self.ts.stats()[self.block.currentData()]
        self.block_lbl.setText(T(f"Готово фраз: {st['done']}/{st['lines']} · хорошей записи: {st['minutes']:.1f} мин",
                                 f"Done: {st['done']}/{st['lines']} · good audio: {st['minutes']:.1f} min"))
        tot = self.ts.total_minutes()
        self.total_lbl.setText(T(f"Всего хорошей записи: {tot:.1f} мин (цель — 30–60)",
                                 f"Good audio total: {tot:.1f} min (goal 30–60)"))
        if self.ts.noise_db is not None:
            self.noise_lbl.setText(T(f"Шум комнаты: {self.ts.noise_db:.0f} дБ", f"Room noise: {self.ts.noise_db:.0f} dB"))

    def _list_pick(self, row):
        lines = self._block_lines()
        if 0 <= row < len(lines):
            self.cur = lines[row].no
            self.show_line()

    def show_line(self):
        l = self.ts.by_no[self.cur]
        k = self.block.findData(l.block)
        if k != self.block.currentIndex():
            self.block.blockSignals(True); self.block.setCurrentIndex(k); self.block.blockSignals(False)
            self.fill_list()
        self.no_lbl.setText(f"{trainset.BLOCK_NAMES[l.block]} · {T('фраза', 'phrase')} {l.no} / {len(self.ts.lines)}")
        self.text_lbl.setText(l.text)
        self.hint_lbl.setText(l.hint)
        self.fill_takes()

    def fill_takes(self):
        lst = self.ts.takes.get(self.cur, [])
        self.takes.setRowCount(len(lst))
        for r, t in enumerate(lst):
            play = QToolButton(); play.setText("▶")
            play.clicked.connect(lambda _=False, t=t: self.play(self.cur, t))
            self.takes.setCellWidget(r, 0, play)
            self.takes.setItem(r, 1, QTableWidgetItem(f"{t.dur:.1f} s"))
            if t.ok is None:
                txt, col = T("⏳ проверяю…", "⏳ checking…"), "#bbbbbb"
                if t.problems:
                    txt += "  " + "; ".join(t.problems)
            elif t.ok:
                txt, col = T(f"✓ годится (совпадение текста {100 - (t.cer or 0) * 100:.0f}%, шум {t.snr_db:.0f} дБ)",
                             f"✓ good"), "#6fd08c"
            else:
                txt, col = "✗ " + "; ".join(t.problems), "#ff6b6b"
            it = QTableWidgetItem(txt); it.setForeground(QColor(col)); it.setToolTip(t.heard)
            self.takes.setItem(r, 2, it)
            dl = QToolButton(); dl.setText("🗑" if False else T("Удалить", "Delete"))
            dl.clicked.connect(lambda _=False, k=r: self.delete_take(k))
            self.takes.setCellWidget(r, 3, dl)
        self.takes.resizeRowsToContents()

    def next(self):
        nos = [l.no for l in self.ts.lines]
        i = nos.index(self.cur)
        if i + 1 < len(nos):
            self.cur = nos[i + 1]; self.show_line(); self.fill_list()

    def prev(self):
        nos = [l.no for l in self.ts.lines]
        i = nos.index(self.cur)
        if i > 0:
            self.cur = nos[i - 1]; self.show_line(); self.fill_list()

    def play(self, no, t):
        import soundfile as sf
        x, sr = sf.read(str(self.ts.path(no, t)), dtype="float32")
        self.win.player.play(x, sr)

    def play_last(self):
        lst = self.ts.takes.get(self.cur, [])
        if lst:
            self.play(self.cur, lst[-1])

    def delete_take(self, k):
        self.ts.delete_take(self.cur, k)
        self.fill_takes(); self.fill_list()

    # ---------------- запись ----------------
    def _start(self, mode):
        dev = int(self.win.s.value("mic", -1))
        self.win.stop_play()
        self.rec = audio_io.Recorder(None if dev < 0 else dev)
        self.rec_mode = mode
        try:
            self.rec.start()
        except Exception as e:  # noqa: BLE001
            self.rec = None
            QMessageBox.warning(self, "Golos", T("Микрофон не открылся: ", "Mic failed: ") + str(e))
            return False
        return True

    def toggle_record(self):
        if self.rec is None:
            if self._start("take"):
                self.rec_btn.setText("■ " + T("Стоп (R)", "Stop (R)"))
        elif self.rec_mode == "take":
            x, sr = self.rec.stop()
            self.rec = None
            self.rec_btn.setText("● " + T("Записать дубль (R)", "Record take (R)"))
            self.level.setValue(0)
            if len(x) < sr * 0.5:
                return
            no = self.cur
            t = self.ts.add_take(no, x, sr)
            self.fill_takes(); self.fill_list()
            self.queue.append((no, t.file))
            self._run_checks()

    def noise_check(self):
        if self.rec is not None:
            return
        QMessageBox.information(self, "Golos", T("Сейчас 3 секунды записи тишины: молчите и не двигайтесь.",
                                                 "Recording 3 s of silence: stay quiet."))
        if self._start("noise"):
            self.mic_btn.setText(T("⏺ тишина…", "⏺ silence…"))
            QTimer.singleShot(3000, self._noise_done)

    def _noise_done(self):
        if self.rec is None or self.rec_mode != "noise":
            return
        x, sr = self.rec.stop()
        self.rec = None
        self.mic_btn.setText(T("① Проверить микрофон и тишину комнаты", "① Check mic & room noise"))
        self.ts.noise_db = trainset.measure_noise(x, sr)
        self.ts.save()
        n = self.ts.noise_db
        verdict = (T("отлично", "excellent") if n < -60 else T("нормально", "ok") if n < -50
                   else T("шумно — выключите вентиляторы и кондиционер, закройте окна", "noisy"))
        self.noise_lbl.setText(T(f"Шум комнаты: {n:.0f} дБ — {verdict}. Говорите обычным голосом: "
                                 f"индикатор должен доходить до середины, но не краснеть.",
                                 f"Room noise: {n:.0f} dB — {verdict}."))

    def _tick(self):
        if self.rec is not None:
            lv = self.rec.level
            self.level.setValue(int(min(1.0, lv) * 100))
            self.level.setStyleSheet("QProgressBar::chunk{background:%s}" % ("#e0433d" if lv > 0.89 else "#3fbf6f"))
            self.rec_time.setText(f"{self.rec.seconds:.1f} s")

    # ---------------- проверка текста (Whisper) ----------------
    def _run_checks(self):
        if not self.queue or self.win._busy():
            return
        jobs = list(self.queue)
        self.queue.clear()
        ts = self.ts

        def job(task):
            from . import models
            import soundfile as sf
            wh = models.get_whisper(task.log.emit)
            res = []
            for no, file in jobs:
                t = next((x for x in ts.takes.get(no, []) if x.file == file), None)
                if t is None:
                    continue
                x, sr = sf.read(str(ts.path(no, t)), dtype="float32")
                heard = wh.text(x, sr, "ru")
                res.append((no, file, heard))
            return res

        def done(res):
            for no, file, heard in res:
                t = next((x for x in self.ts.takes.get(no, []) if x.file == file), None)
                if t is None:
                    continue
                line = self.ts.by_no[no]
                lenient = line.block == "В" or any(ch.isascii() and ch.isalpha() for ch in line.text)
                trainset.text_check(t, line.text, heard, lenient=lenient)
                if t.ok and no == self.cur and self.auto_next.isChecked():
                    QTimer.singleShot(400, self.next)
            self.ts.save()
            self.fill_takes(); self.fill_list()
            QTimer.singleShot(100, self._run_checks)

        self.win.start_task(job, done)

    # ---------------- пакет и подключение ----------------
    def export(self):
        tot = self.ts.total_minutes()
        if tot < 1:
            QMessageBox.information(self, "Golos", T("Пока нет проверенных дублей.", "No checked takes yet.")); return
        if tot < 20 and QMessageBox.question(
                self, "Golos", T(f"Записано {tot:.1f} мин хорошей речи — для заметного результата нужно 30–60. "
                                 f"Всё равно собрать пакет?", f"Only {tot:.1f} min. Build anyway?")) != QMessageBox.Yes:
            return
        speaker = self.win.s.value("train_speaker", "my_voice")
        ts = self.ts

        def job(task):
            return trainset.export_package(ts, speaker, task.log.emit)

        def done(z):
            self.win.log(T("Пакет обучения готов. Откройте README_RU.txt внутри — там 5 шагов.",
                           "Training package ready. See README_RU.txt inside."))
            self.win.reveal(Path(z))

        self.win.start_task(job, done)

    def install(self):
        f, _ = QFileDialog.getOpenFileName(self, T("Обученная модель (zip)", "Trained model (zip)"), "", "ZIP (*.zip)")
        if not f:
            return
        name, ok = QInputDialog.getText(self, "Golos", T("Название голоса:", "Voice name:"),
                                        text=T("Мой голос (обученный)", "My voice (trained)"))
        if not ok:
            return
        speaker = self.win.s.value("train_speaker", "my_voice")

        def job(task):
            return trainset.install_trained(Path(f), name or "Мой голос (обученный)", speaker, task.log.emit)

        def done(v):
            self.win.reload_voices()
            self.win.log(T(f"Голос «{v.name}» добавлен в список голосов на вкладке 2.",
                           f"Voice «{v.name}» added to the voice list on tab 2."))

        self.win.start_task(job, done)
