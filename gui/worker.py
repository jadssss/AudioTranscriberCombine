"""QThread wrapper around core.pipeline.run_pipeline."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from core.config import PipelineConfig
from core.pipeline import run_pipeline


class PipelineWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)     # done, total
    finished_ok = Signal(list)      # list[FileResult]
    failed = Signal(str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: D401 — Qt override
        try:
            results = run_pipeline(
                self._config,
                log=self.log.emit,
                on_progress=lambda d, t: self.progress.emit(d, t),
                stop_event=self._stop_event,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
