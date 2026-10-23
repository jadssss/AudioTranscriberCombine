#!/usr/bin/env python3
"""Song Transcriber — Qt GUI entry point."""
from __future__ import annotations

import ctypes
import os
import sys

# Must be set BEFORE PySide6 is imported.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")


def preload_cudnn() -> None:
    try:
        import nvidia.cudnn
    except ImportError:
        return
    lib_dir = os.path.join(nvidia.cudnn.__path__[0], "lib")
    if not os.path.isdir(lib_dir):
        return
    for fname in sorted(os.listdir(lib_dir)):
        if ".so" in fname:
            try:
                ctypes.CDLL(os.path.join(lib_dir, fname), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def enable_tf32() -> None:
    """Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30/40 series)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass


def main() -> int:
    preload_cudnn()
    enable_tf32()

    from PySide6.QtWidgets import QApplication
    from gui.theme import DARK_STYLESHEET
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Transcriber")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
