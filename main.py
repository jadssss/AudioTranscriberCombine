#!/usr/bin/env python3
"""Song Transcriber v3 — Qt GUI entry point.

Run: python main.py
"""
from __future__ import annotations

import ctypes
import os
import sys


def preload_cudnn() -> None:
    """Preload cuDNN shared libraries so torch / onnxruntime can find them."""
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


def main() -> int:
    preload_cudnn()
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Transcriber")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
