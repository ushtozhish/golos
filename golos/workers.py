"""Фоновые задачи (распознавание, синтез, экспорт) в отдельном потоке Qt."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class Task(QThread):
    log = Signal(str)
    progress = Signal(int, int)
    failed = Signal(str)
    succeeded = Signal(object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            res = self.fn(self, *self.args, **self.kwargs)
            self.succeeded.emit(res)
        except Exception as e:  # noqa: BLE001
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(e).__name__}: {e}")
