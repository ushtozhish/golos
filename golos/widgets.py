"""Таймлайн с клипами (ручной монтаж) и панель распознанного текста."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QPen, QFont, QTextCharFormat, QTextCursor,
                           QBrush, QFontMetrics, QCursor)
from PySide6.QtWidgets import QWidget, QTextEdit, QScrollBar, QVBoxLayout

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


class TextPanel(QTextEdit):
    """Распознанный текст смонтированной дорожки. Клик по слову — перейти к нему."""
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        f = self.font(); f.setPointSizeF(12); self.setFont(f)
        self.spans: list[tuple[int, int, int]] = []
        self.p: Project | None = None

    def set_project(self, p):
        self.p = p
        self.rebuild()

    def rebuild(self):
        self.clear(); self.spans = []
        if not self.p or not self.p.words:
            self.setPlaceholderText(T("Текст появится после «Распознать» или «Отправить на перевод». "
                                      "Монтируйте на слух — текст не обязателен.",
                                      "Text appears after Recognize or Send to translation. "
                                      "Edit by ear — the text is optional."))
            return
        cur = self.textCursor()
        stale = not self.p.words_fresh
        fmt = QTextCharFormat(); fmt.setForeground(QColor("#7a7a7a" if stale else "#e8e8e8"))
        if stale:
            note = QTextCharFormat(); note.setForeground(QColor("#ffb300"))
            cur.insertText(T("Монтаж изменился — текст обновится при отправке.\n\n",
                             "Edit changed — text will refresh on send.\n\n"), note)
        for i, w in enumerate(self.p.words):
            p0 = cur.position()
            cur.insertText(w.text.strip(), fmt)
            self.spans.append((p0, cur.position(), i))
            cur.insertText("\n" if ends_sentence(w.text) else " ", QTextCharFormat())

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.p and self.p.words_fresh and not self.textCursor().hasSelection():
            c = self.cursorForPosition(e.position().toPoint()).position()
            for p0, p1, i in self.spans:
                if p0 <= c <= p1:
                    self.seek.emit(self.p.words[i].start)
                    break

    def highlight(self, t: float):
        if not self.p or not self.p.words_fresh:
            self.setExtraSelections([]); return
        i = next((k for k, w in enumerate(self.p.words) if w.start <= t <= w.end + 0.05), None)
        sel = []
        if i is not None and i < len(self.spans):
            p0, p1, _ = self.spans[i]
            s = QTextEdit.ExtraSelection()
            c = QTextCursor(self.document()); c.setPosition(p0); c.setPosition(p1, QTextCursor.KeepAnchor)
            s.cursor = c
            fmt = QTextCharFormat(); fmt.setBackground(QBrush(QColor(255, 204, 51, 90)))
            s.format = fmt
            sel.append(s)
        self.setExtraSelections(sel)
