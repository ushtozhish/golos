"""Таймлайн с клипами (ручной монтаж) и панель распознанного текста."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QPen, QFont, QTextCharFormat, QTextCursor,
                           QBrush, QFontMetrics, QCursor)
from PySide6.QtWidgets import (QWidget, QTextEdit, QScrollBar, QVBoxLayout, QTableWidget, QTableWidgetItem,
                               QHeaderView, QAbstractItemView, QLabel)

from .project import Project, SR, ends_sentence
from .i18n import T

BG = QColor("#16181d")
CLIP_A = QColor("#22324a")
CLIP_B = QColor("#283c58")
CLIP_SEL = QColor("#3a5578")
WAVE = QColor("#7cc4ff")
CUTLINE = QColor("#ff5a5a")
SEL = QColor(255, 255, 255, 45)
HEAD = QColor("#ffcc33")
RULER_H, WORDS_H = 20, 20
EDGE_PX = 6


class TimelineView(QWidget):
    seek = Signal(float)            # время дорожки
    edited = Signal()               # монтаж изменился
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.p: Project | None = None
        self.env = np.zeros(0, np.float32)
        self.view0, self.view1 = 0.0, 10.0
        self.selected: set[int] = set()
        self.range: tuple[float, float] | None = None
        self.head = 0.0
        self.scroll: QScrollBar | None = None
        self._press = None          # (kind, data)
        self._moved = False

    # ---------- данные ----------
    def set_project(self, p: Project | None):
        self.p = p
        self.rebuild_env()
        self.selected.clear(); self.range = None; self.head = 0.0
        self.zoom_all()

    def rebuild_env(self):
        p = self.p
        if p is not None and len(p.audio):
            hop = int(SR * 0.005)
            n = len(p.audio) // hop
            self.env = np.abs(p.audio[: n * hop]).reshape(n, hop).max(axis=1) if n else np.zeros(1)
        else:
            self.env = np.zeros(1, np.float32)

    @property
    def total(self) -> float:
        return self.p.seq_duration() if self.p else 0.0

    def refresh(self):
        if self.p:
            self.selected = {i for i in self.selected if i < len(self.p.clips)}
        self._sync_scroll()
        self.update()

    def set_head(self, t: float, follow: bool = True):
        self.head = max(0.0, t)
        if follow and self.p and not (self.view0 <= t <= self.view1):
            w = self.view1 - self.view0
            self.view0 = max(0.0, min(t - w * 0.1, max(0.0, self.total - w * 0.9)))
            self.view1 = self.view0 + w
            self._sync_scroll()
        self.update()

    # ---------- координаты и зум ----------
    def t2x(self, t: float) -> float:
        return (t - self.view0) / max(1e-6, self.view1 - self.view0) * self.width()

    def x2t(self, x: float) -> float:
        return self.view0 + x / max(1, self.width()) * (self.view1 - self.view0)

    def zoom(self, factor: float, center: float | None = None):
        total = max(1.0, self.total)
        c = self.head if center is None else center
        w = min(total * 1.05, max(0.15, (self.view1 - self.view0) * factor))
        k = (c - self.view0) / max(1e-6, self.view1 - self.view0)
        k = min(1.0, max(0.0, k))
        v0 = max(0.0, c - w * k)
        self.view0, self.view1 = v0, v0 + w
        self._sync_scroll(); self.update()

    def zoom_all(self):
        self.view0, self.view1 = 0.0, max(1.0, self.total * 1.02)
        self._sync_scroll(); self.update()

    def _sync_scroll(self):
        if not self.scroll:
            return
        total = max(1.0, self.total)
        w = self.view1 - self.view0
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, max(0, int((total - w) * 100)))
        self.scroll.setPageStep(max(1, int(w * 100)))
        self.scroll.setValue(int(self.view0 * 100))
        self.scroll.blockSignals(False)

    def on_scroll(self, v: int):
        w = self.view1 - self.view0
        self.view0 = v / 100.0
        self.view1 = self.view0 + w
        self.update()

    # ---------- поиск под мышью ----------
    def _edge_at(self, x: float):
        """('start'|'end', индекс клипа) если курсор у края клипа."""
        if not self.p:
            return None
        starts = self.p.clip_starts()
        best = None
        for i, ((a, b), s) in enumerate(zip(self.p.clips, starts)):
            for side, t in (("start", s), ("end", s + (b - a))):
                d = abs(self.t2x(t) - x)
                if d <= EDGE_PX:
                    # на стыке: левее линии — конец левого клипа, правее — начало правого
                    bias = (x - self.t2x(t)) * (1 if side == "start" else -1)
                    score = d - (0.5 if bias >= 0 else 0)
                    if best is None or score < best[0]:
                        best = (score, side, i)
        return (best[1], best[2]) if best else None

    # ---------- мышь ----------
    def wheelEvent(self, e):
        if not self.p:
            return
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ShiftModifier or e.angleDelta().x():
            w = self.view1 - self.view0
            d = -(dy or e.angleDelta().x()) / 120 * w * 0.15
            self.view0 = max(0.0, min(self.view0 + d, max(0.0, self.total - w * 0.5)))
            self.view1 = self.view0 + w
            self._sync_scroll(); self.update()
        else:
            self.zoom(0.8 if dy > 0 else 1.25, self.x2t(e.position().x()))

    def mousePressEvent(self, e):
        if not self.p or e.button() != Qt.LeftButton:
            return
        self.setFocus()
        x, y = e.position().x(), e.position().y()
        t = max(0.0, min(self.x2t(x), self.total))
        self._moved = False
        if y < RULER_H:
            self._press = ("scrub", None)
            self.seek.emit(t)
            return
        edge = self._edge_at(x)
        if edge:
            side, i = edge
            a, b = self.p.clips[i]
            self._press = ("edge", (side, i, t, a, b, False))
            return
        self._press = ("body", (t, self.p.clip_at(t), bool(e.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))))

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._press is None:
            if self.p and e.position().y() >= RULER_H and self._edge_at(x):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return
        kind, data = self._press
        t = max(0.0, min(self.x2t(x), self.total))
        if kind == "scrub":
            self.seek.emit(t); return
        if kind == "edge":
            side, i, t0, a, b, started = data
            d = self.x2t(x) - t0
            if abs(self.t2x(self.x2t(x)) - self.t2x(t0)) < 2 and not started:
                return
            self.p.set_clip_edge(i, side, (a + d) if side == "start" else (b + d), checkpoint=not started)
            self._press = ("edge", (side, i, t0, a, b, True))
            self._moved = True
            self.refresh()
            return
        if kind == "body":
            t0 = data[0]
            if abs(self.t2x(t) - self.t2x(t0)) > 4:
                self._moved = True
                a, b = sorted((t0, t))
                self.range = (a, b)
                self.selected.clear()
                self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        kind, data = self._press
        self._press = None
        if kind == "edge" and self._moved:
            self.edited.emit()
        elif kind == "body" and not self._moved:
            t, i, add = data
            self.range = None
            if i >= 0:
                if add:
                    self.selected ^= {i}
                else:
                    self.selected = {i}
            else:
                self.selected.clear()
            self.seek.emit(t)
            self.clicked.emit()
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.p:
            i = self.p.clip_at(self.x2t(e.position().x()))
            if i >= 0:
                s = self.p.clip_starts()[i]
                a, b = self.p.clips[i]
                self.view0, self.view1 = max(0.0, s - 0.3), s + (b - a) + 0.3
                self._sync_scroll(); self.update()

    # ---------- отрисовка ----------
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.fillRect(self.rect(), BG)
        W, H = self.width(), self.height()
        if not self.p or not self.p.clips:
            qp.setPen(QColor("#888"))
            qp.drawText(self.rect(), Qt.AlignCenter,
                        T("Нажмите «● Запись» (или R) и начитайте текст", "Press ● Record (or R) and read your text"))
            return
        show_words = self.p.words_fresh
        top = RULER_H + (WORDS_H if show_words else 0)
        mid = top + (H - top) / 2
        amp = (H - top) / 2 - 6
        span = self.view1 - self.view0
        starts = self.p.clip_starts()
        pen_w = QPen(WAVE)
        for i, ((a, b), s) in enumerate(zip(self.p.clips, starts)):
            e_ = s + (b - a)
            if e_ < self.view0 or s > self.view1:
                continue
            x0, x1 = self.t2x(s), self.t2x(e_)
            r = QRectF(x0, top + 2, max(1.0, x1 - x0), H - top - 4)
            col = CLIP_SEL if i in self.selected else (CLIP_A if i % 2 == 0 else CLIP_B)
            qp.fillRect(r, col)
            # волна клипа
            xa, xb = int(max(0, x0)), int(min(W, x1))
            if xb > xa:
                xs = np.arange(xa, xb)
                tseq = self.view0 + xs / W * span
                to = a + (tseq - s)
                dt = span / W
                i0 = np.clip((to / 0.005).astype(int), 0, len(self.env) - 1)
                i1 = np.clip(((to + dt) / 0.005).astype(int) + 1, 1, len(self.env))
                if dt / 0.005 > 1.5:
                    mx = np.array([self.env[u:max(u + 1, v)].max() for u, v in zip(i0, i1)])
                else:
                    mx = self.env[i0]
                qp.setPen(pen_w)
                for x, m in zip(xs, mx):
                    h = float(m) * amp
                    qp.drawLine(QPointF(x, mid - h), QPointF(x, mid + h))
            if i in self.selected:
                qp.setPen(QPen(HEAD, 2)); qp.drawRect(r)
            # рез
            if i > 0:
                qp.setPen(QPen(CUTLINE, 1.5))
                qp.drawLine(QPointF(x0, top), QPointF(x0, H))
        end_x = self.t2x(self.total)
        if end_x < W:
            qp.fillRect(QRectF(end_x, top, W - end_x, H - top), QColor("#101216"))
        # слова
        if show_words:
            f = QFont(self.font()); f.setPointSizeF(8.5); qp.setFont(f)
            fm = QFontMetrics(f)
            last = -1e9
            qp.setPen(QColor("#d8d8d8"))
            for w in self.p.words:
                if w.end < self.view0 or w.start > self.view1:
                    continue
                x = self.t2x(w.start)
                txt = w.text.strip()
                if x > last + 3:
                    qp.drawText(QPointF(x, RULER_H + 14), txt)
                    last = x + fm.horizontalAdvance(txt)
        # линейка
        qp.fillRect(0, 0, W, RULER_H, QColor("#0f1115"))
        pps = W / max(1e-6, span)
        step = next((s for s in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
                     if s * pps >= 70), 1800)
        qp.setPen(QColor("#8a8a8a"))
        f = QFont(self.font()); f.setPointSizeF(8); qp.setFont(f)
        t = (self.view0 // step) * step
        while t <= self.view1:
            x = self.t2x(t)
            qp.drawLine(QPointF(x, RULER_H - 6), QPointF(x, RULER_H))
            m, s = divmod(t, 60)
            qp.drawText(QPointF(x + 2, 13), f"{int(m)}:{s:05.2f}" if step < 1 else f"{int(m)}:{int(s):02d}")
            t += step
        if self.range:
            x0, x1 = self.t2x(self.range[0]), self.t2x(self.range[1])
            qp.fillRect(QRectF(x0, RULER_H, x1 - x0, H - RULER_H), SEL)
        x = self.t2x(self.head)
        qp.setPen(QPen(HEAD, 1.5))
        qp.drawLine(QPointF(x, 0), QPointF(x, H))
        qp.setBrush(HEAD)
        qp.drawPolygon([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 7)])
        qp.end()


class TimelinePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.view = TimelineView()
        self.bar = QScrollBar(Qt.Horizontal)
        self.view.scroll = self.bar
        self.bar.valueChanged.connect(self.view.on_scroll)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.bar)


class PhraseEditor(QTableWidget):
    """Распознанный текст по фразам. Текст фразы можно исправить (двойной клик или F2).
    Одинарный клик — перейти к фразе на дорожке."""
    seek = Signal(float)
    text_edited = Signal(int, str)      # id фразы, новый текст

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["#", T("Время", "Time"),
                                        T("Текст — исправьте ошибки распознавания (двойной клик)",
                                          "Text — fix recognition errors (double-click)")])
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setWordWrap(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        f = self.font(); f.setPointSizeF(11.5); self.setFont(f)
        self.p: Project | None = None
        self.phr = []
        self._cur = -1
        self.cellClicked.connect(self._clicked)
        self.itemChanged.connect(self._changed)
        self.note = QLabel(self)
        self.note.setWordWrap(True)
        self.note.setAlignment(Qt.AlignCenter)
        self.note.setStyleSheet("color:#999;background:transparent;padding:20px")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.note.setGeometry(0, 30, self.viewport().width(), 120)

    def set_project(self, p):
        self.p = p
        self.rebuild()

    def rebuild(self):
        self.blockSignals(True)
        self.setRowCount(0)
        self.phr = self.p.phrases() if self.p and self.p.words else []
        if not self.phr:
            self.note.setText(T("Текст появится после «Распознать текст». Монтировать можно на слух — "
                                "текст нужен для перевода и переозвучки.",
                                "Text appears after «Recognize text». You can edit by ear — "
                                "text is needed for translation and re-voicing."))
            self.note.show()
            self.blockSignals(False)
            return
        stale = not self.p.words_fresh
        self.note.setText(T("Монтаж изменился после распознавания — нажмите «Распознать текст» ещё раз.",
                            "Edit changed after recognition — press «Recognize text» again.") if stale else "")
        self.note.setVisible(stale)
        self.setRowCount(len(self.phr))
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        for r, ph in enumerate(self.phr):
            it = QTableWidgetItem(str(ph.id)); it.setFlags(ro); self.setItem(r, 0, it)
            m, sec = divmod(ph.start, 60)
            it = QTableWidgetItem(f"{int(m)}:{sec:04.1f}"); it.setFlags(ro); self.setItem(r, 1, it)
            it = QTableWidgetItem(ph.text)
            if ph.id in self.p.phrase_texts:
                it.setForeground(QColor("#8fd48f"))
                it.setToolTip(T("Исправлено вами", "Edited by you"))
            if stale:
                it.setFlags(ro); it.setForeground(QColor("#777"))
            self.setItem(r, 2, it)
        self.resizeRowsToContents()
        self.blockSignals(False)

    def _clicked(self, r, c):
        if 0 <= r < len(self.phr):
            self.seek.emit(self.phr[r].start)

    def _changed(self, item):
        if item.column() == 2 and 0 <= item.row() < len(self.phr):
            self.text_edited.emit(self.phr[item.row()].id, item.text().strip())

    def highlight(self, t: float):
        if not self.phr:
            return
        r = next((k for k, ph in enumerate(self.phr) if ph.start <= t <= ph.end + 0.3), -1)
        if r != self._cur and r >= 0 and self.state() != QAbstractItemView.EditingState:
            self._cur = r
            self.blockSignals(True)
            self.selectRow(r)
            self.blockSignals(False)
            self.scrollToItem(self.item(r, 2))
