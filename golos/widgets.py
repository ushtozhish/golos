"""Виджеты: волна с монтажом и текстовая расшифровка, по которой можно резать."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QPen, QFont, QTextCharFormat, QTextCursor,
                           QBrush, QFontMetrics)
from PySide6.QtWidgets import QWidget, QTextEdit, QScrollBar, QVBoxLayout

from .project import Project, SR, ends_sentence
from .i18n import T

BG = QColor("#16181d")
WAVE = QColor("#5fb3ff")
WAVE_CUT = QColor("#5a3a40")
CUT_SHADE = QColor(200, 40, 60, 70)
GAP = QColor(255, 200, 0, 110)
SEL = QColor(255, 255, 255, 45)
HEAD = QColor("#ffcc33")
REF = QColor(80, 220, 120, 60)


class WaveformView(QWidget):
    seek = Signal(float)              # время на исходной шкале
    selection_changed = Signal(float, float)
    cut_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.p: Project | None = None
        self.env = np.zeros(0, np.float32)
        self.view0, self.view1 = 0.0, 1.0
        self.regions: list[tuple[float, float]] = []
        self.sel: tuple[float, float] | None = None
        self.head: float | None = None
        self._drag0: float | None = None
        self._dragging = False
        self.scroll: QScrollBar | None = None

    # ----- данные -----
    def set_project(self, p: Project | None):
        self.p = p
        if p is not None and len(p.audio):
            hop = int(SR * 0.005)
            n = len(p.audio) // hop
            self.env = np.abs(p.audio[: n * hop]).reshape(n, hop).max(axis=1) if n else np.zeros(0)
            self.view0, self.view1 = 0.0, max(1.0, p.duration)
        else:
            self.env = np.zeros(0, np.float32)
            self.view0, self.view1 = 0.0, 1.0
        self.sel = None
        self.refresh()

    def refresh(self):
        self.regions = self.p.keep_regions() if self.p and len(self.p.audio) else []
        self._sync_scroll()
        self.update()

    def set_head(self, t: float | None):
        self.head = t
        if t is not None and self.p and not (self.view0 <= t <= self.view1):
            w = self.view1 - self.view0
            self.view0 = max(0.0, min(t - w * 0.1, self.p.duration - w))
            self.view1 = self.view0 + w
            self._sync_scroll()
        self.update()

    # ----- координаты -----
    def t2x(self, t: float) -> float:
        return (t - self.view0) / max(1e-6, self.view1 - self.view0) * self.width()

    def x2t(self, x: float) -> float:
        return self.view0 + x / max(1, self.width()) * (self.view1 - self.view0)

    def zoom(self, factor: float, center_t: float | None = None):
        if not self.p:
            return
        dur = max(1.0, self.p.duration)
        c = center_t if center_t is not None else (self.view0 + self.view1) / 2
        w = min(dur, max(0.2, (self.view1 - self.view0) * factor))
        k = (c - self.view0) / max(1e-6, self.view1 - self.view0)
        v0 = max(0.0, min(c - w * k, dur - w))
        self.view0, self.view1 = v0, v0 + w
        self._sync_scroll()
        self.update()

    def zoom_all(self):
        if self.p:
            self.view0, self.view1 = 0.0, max(1.0, self.p.duration)
            self._sync_scroll(); self.update()

    def zoom_sel(self):
        if self.sel and self.sel[1] - self.sel[0] > 0.1:
            pad = (self.sel[1] - self.sel[0]) * 0.1
            self.view0, self.view1 = max(0.0, self.sel[0] - pad), self.sel[1] + pad
            self._sync_scroll(); self.update()

    def _sync_scroll(self):
        if not self.scroll or not self.p:
            return
        dur = max(1.0, self.p.duration)
        w = self.view1 - self.view0
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, max(0, int((dur - w) * 100)))
        self.scroll.setPageStep(max(1, int(w * 100)))
        self.scroll.setValue(int(self.view0 * 100))
        self.scroll.blockSignals(False)

    def on_scroll(self, v: int):
        w = self.view1 - self.view0
        self.view0 = v / 100.0
        self.view1 = self.view0 + w
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_X):
            self.cut_requested.emit()
            return
        super().keyPressEvent(e)

    # ----- мышь -----
    def wheelEvent(self, e):
        if not self.p:
            return
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ShiftModifier or e.angleDelta().x():
            w = self.view1 - self.view0
            d = -(dy or e.angleDelta().x()) / 120 * w * 0.15
            self.view0 = max(0.0, min(self.view0 + d, self.p.duration - w))
            self.view1 = self.view0 + w
            self._sync_scroll(); self.update()
        else:
            self.zoom(0.8 if dy > 0 else 1.25, self.x2t(e.position().x()))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.p:
            self._drag0 = self.x2t(e.position().x())
            self._dragging = False
            self.setFocus()

    def mouseMoveEvent(self, e):
        if self._drag0 is not None and e.buttons() & Qt.LeftButton:
            t = self.x2t(e.position().x())
            if abs(self.t2x(t) - self.t2x(self._drag0)) > 3:
                self._dragging = True
                a, b = sorted((self._drag0, t))
                self.sel = (max(0.0, a), min(self.p.duration, b))
                self.update()

    def mouseReleaseEvent(self, e):
        if self._drag0 is None:
            return
        if self._dragging and self.sel:
            self.selection_changed.emit(*self.sel)
        else:
            self.sel = None
            self.seek.emit(max(0.0, self._drag0))
            self.selection_changed.emit(-1.0, -1.0)
        self._drag0 = None
        self._dragging = False
        self.update()

    # ----- отрисовка -----
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.fillRect(self.rect(), BG)
        W, H = self.width(), self.height()
        if not self.p or len(self.p.audio) == 0:
            qp.setPen(QColor("#888"))
            qp.drawText(self.rect(), Qt.AlignCenter,
                        T("Нажмите «Запись» или импортируйте файл", "Press Record or import a file"))
            return
        ruler_h, words_h = 18, 18
        top = ruler_h + words_h
        mid = top + (H - top) / 2
        amp = (H - top) / 2 - 4
        # волна
        spp = (self.view1 - self.view0) * SR / max(1, W)
        xs = np.arange(W)
        ts = self.view0 + xs / W * (self.view1 - self.view0)
        if spp > 240:
            idx0 = np.clip((ts / 0.005).astype(int), 0, len(self.env) - 1)
            idx1 = np.clip(((ts + (self.view1 - self.view0) / W) / 0.005).astype(int) + 1, 1, len(self.env))
            mx = np.array([self.env[a:max(a + 1, b)].max() for a, b in zip(idx0, idx1)])
            mn = -mx
        else:
            s0 = int(self.view0 * SR)
            s1 = min(len(self.p.audio), int(self.view1 * SR) + 1)
            seg = self.p.audio[s0:s1]
            if len(seg) < 2:
                return
            from .dsp import peaks
            mn, mx = peaks(seg, W)
        kept = np.zeros(W, bool)
        for a, b in self.regions:
            xa, xb = int(max(0, self.t2x(a))), int(min(W, self.t2x(b) + 1))
            if xb > xa:
                kept[xa:xb] = True
        pen_k, pen_c = QPen(WAVE), QPen(WAVE_CUT)
        for x in range(min(W, len(mx))):
            qp.setPen(pen_k if kept[x] else pen_c)
            qp.drawLine(QPointF(x, mid - mx[x] * amp), QPointF(x, mid - mn[x] * amp))
        # вырезанные участки
        prev = 0.0
        for a, b in self.regions + [(self.p.duration, self.p.duration)]:
            if a > prev:
                x0, x1 = self.t2x(prev), self.t2x(a)
                if x1 > 0 and x0 < W:
                    qp.fillRect(QRectF(x0, top, x1 - x0, H - top), CUT_SHADE)
            prev = b
        # нераспознанная речь
        qp.setPen(QPen(GAP, 1.5))
        for g in self.p.gaps:
            x0, x1 = self.t2x(g.start), self.t2x(g.end)
            if x1 > 0 and x0 < W:
                qp.drawRect(QRectF(x0, top + 2, x1 - x0, H - top - 4))
        # образец голоса
        if self.p.ref_override:
            a, b = self.p.ref_override
            qp.fillRect(QRectF(self.t2x(a), top, self.t2x(b) - self.t2x(a), H - top), REF)
        # слова
        pps = W / max(1e-6, self.view1 - self.view0)
        f = QFont(self.font()); f.setPointSizeF(8.5); qp.setFont(f)
        fm = QFontMetrics(f)
        if pps > 25:
            last_x = -1e9
            for w in self.p.words:
                if w.end < self.view0 or w.start > self.view1:
                    continue
                x = self.t2x(w.start)
                if x < last_x + 2:
                    continue
                col = QColor("#ff6b6b") if w.auto == "dup" else QColor("#777") if w.cut else QColor("#ddd")
                qp.setPen(col)
                txt = w.text.strip()
                qp.drawText(QPointF(x, ruler_h + 13), txt)
                if w.cut:
                    tw = fm.horizontalAdvance(txt)
                    qp.drawLine(QPointF(x, ruler_h + 9), QPointF(x + tw, ruler_h + 9))
                last_x = x + fm.horizontalAdvance(txt)
        # линейка
        qp.fillRect(0, 0, W, ruler_h, QColor("#0f1115"))
        span = self.view1 - self.view0
        step = next((s for s in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
                    if s * pps >= 70), 1800) if pps > 0 else 60
        qp.setPen(QColor("#888"))
        t = (self.view0 // step) * step
        while t <= self.view1:
            x = self.t2x(t)
            qp.drawLine(QPointF(x, ruler_h - 5), QPointF(x, ruler_h))
            m, s = divmod(t, 60)
            qp.drawText(QPointF(x + 2, 12), f"{int(m)}:{s:05.2f}" if step < 1 else f"{int(m)}:{int(s):02d}")
            t += step
        # выделение и курсор
        if self.sel:
            x0, x1 = self.t2x(self.sel[0]), self.t2x(self.sel[1])
            qp.fillRect(QRectF(x0, 0, x1 - x0, H), SEL)
        if self.head is not None:
            x = self.t2x(self.head)
            qp.setPen(QPen(HEAD, 1.5))
            qp.drawLine(QPointF(x, 0), QPointF(x, H))
        qp.end()


class WaveformPanel(QWidget):
    """Волна + горизонтальная прокрутка."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.view = WaveformView()
        self.bar = QScrollBar(Qt.Horizontal)
        self.view.scroll = self.bar
        self.bar.valueChanged.connect(self.view.on_scroll)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.bar)


class TranscriptView(QTextEdit):
    """Текст начитки. Выделите слова и нажмите Delete — они вырезаются (повторно — возвращаются).
    Клик по слову ставит курсор воспроизведения, двойной клик — играет с этого слова."""
    toggle_words = Signal(list, list)   # индексы слов, индексы пропусков
    seek_word = Signal(int, bool)       # индекс слова, играть ли

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        f = self.font(); f.setPointSizeF(12); self.setFont(f)
        self.spans: list[tuple[int, int, str, int]] = []  # (pos0, pos1, kind, index)
        self.p: Project | None = None

    def set_project(self, p: Project | None):
        self.p = p
        self.rebuild()

    def rebuild(self):
        sb = self.verticalScrollBar().value()
        self.clear()
        self.spans = []
        if not self.p or not self.p.words:
            self.setPlaceholderText(T("Здесь появится текст после распознавания.",
                                      "Transcript appears here after recognition."))
            return
        cur = self.textCursor()
        normal = QTextCharFormat(); normal.setForeground(QColor("#e8e8e8"))
        cut = QTextCharFormat(); cut.setForeground(QColor("#6d6d6d")); cut.setFontStrikeOut(True)
        dup = QTextCharFormat(); dup.setForeground(QColor("#ff6b6b")); dup.setFontStrikeOut(True)
        filler = QTextCharFormat(); filler.setForeground(QColor("#c792ea")); filler.setFontStrikeOut(True)
        gapf = QTextCharFormat(); gapf.setForeground(QColor("#ffc400"))
        gapc = QTextCharFormat(); gapc.setForeground(QColor("#8a7420")); gapc.setFontStrikeOut(True)
        space = QTextCharFormat()
        events = [(w.start, "w", i) for i, w in enumerate(self.p.words)] + \
                 [(g.start, "g", i) for i, g in enumerate(self.p.gaps)]
        events.sort()
        for _, kind, i in events:
            pos0 = cur.position()
            if kind == "w":
                w = self.p.words[i]
                fmt = normal if not w.cut else dup if w.auto == "dup" else filler if w.auto == "filler" else cut
                cur.insertText(w.text.strip(), fmt)
            else:
                g = self.p.gaps[i]
                cur.insertText(f"[{T('неразобрано', 'unrecognized')} {g.end - g.start:.1f}s]",
                               gapc if g.cut else gapf)
            self.spans.append((pos0, cur.position(), kind, i))
            txt = self.p.words[i].text if kind == "w" else ""
            cur.insertText("\n" if ends_sentence(txt) else " ", space)
        self.verticalScrollBar().setValue(sb)

    def _tokens_in(self, a: int, b: int):
        ws, gs = [], []
        for p0, p1, kind, i in self.spans:
            if p1 > a and p0 < b:
                (ws if kind == "w" else gs).append(i)
        return ws, gs

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            c = self.textCursor()
            a, b = c.selectionStart(), c.selectionEnd()
            if a == b:
                a, b = max(0, a - 1), a + 1
            ws, gs = self._tokens_in(a, b)
            if ws or gs:
                self.toggle_words.emit(ws, gs)
            return
        super().keyPressEvent(e)

    def _token_at(self, pos):
        c = self.cursorForPosition(pos).position()
        for p0, p1, kind, i in self.spans:
            if p0 <= c <= p1 and kind == "w":
                return i
        return None

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if not self.textCursor().hasSelection():
            i = self._token_at(e.position().toPoint())
            if i is not None:
                self.seek_word.emit(i, False)

    def mouseDoubleClickEvent(self, e):
        i = self._token_at(e.position().toPoint())
        if i is not None:
            self.seek_word.emit(i, True)
        super().mouseDoubleClickEvent(e)

    def highlight_word(self, i: int | None):
        sel = []
        if i is not None:
            for p0, p1, kind, k in self.spans:
                if kind == "w" and k == i:
                    s = QTextEdit.ExtraSelection()
                    c = QTextCursor(self.document()); c.setPosition(p0); c.setPosition(p1, QTextCursor.KeepAnchor)
                    s.cursor = c
                    fmt = QTextCharFormat(); fmt.setBackground(QBrush(QColor(255, 204, 51, 90)))
                    s.format = fmt
                    sel.append(s)
                    break
        self.setExtraSelections(sel)
