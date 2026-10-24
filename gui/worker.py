"""QThread wrapper. Supports three task types: main, lrc, embed."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from core.config import PipelineConfig


class PipelineWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int, str)   # done, total, stage
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, config: PipelineConfig, task: str = "main", parent=None):
        super().__init__(parent)
        self._config = config
        self._task = task
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            if self._task == "lrc":
                from core.lrc_pipeline import run_lrc_pipeline as runner
            elif self._task == "embed":
                from core.lrc_pipeline import run_embed_pipeline as runner
            else:
                from core.pipeline import run_pipeline as runner

            results = runner(
                self._config,
                log=self.log.emit,
                on_progress=lambda d, t, s: self.progress.emit(d, t, s),
                stop_event=self._stop_event,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
