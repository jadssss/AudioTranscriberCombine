"""Dark theme for the whole application (Catppuccin Macchiato-inspired)."""
from __future__ import annotations

DARK_STYLESHEET = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Inter', 'Segoe UI', 'Noto Sans', sans-serif;
    font-size: 10pt;
}
QMainWindow { background-color: #181825; }

QLabel { background: transparent; }
QLabel#appTitle {
    font-size: 18pt;
    font-weight: 700;
    color: #89b4fa;
    padding: 4px 0 10px 0;
}
QLabel#appSubtitle {
    color: #7f849c;
    font-size: 9pt;
    padding-bottom: 12px;
}
QLabel#deviceInfo {
    color: #a6adc8;
    font-size: 9pt;
    padding: 4px 8px;
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
}

QGroupBox {
    border: 1px solid #313244;
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    background-color: #1e1e2e;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}

QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    min-height: 22px;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #89b4fa;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    color: #585b70;
    background-color: #11111b;
}

QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #89b4fa;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #181825;
    border: 1px solid #313244;
    selection-background-color: #313244;
    selection-color: #89b4fa;
    outline: none;
}

/* ─── buttons: higher contrast ─────────────────────────────────── */
QPushButton {
    background-color: #3d4159;
    border: 1px solid #5c6080;
    border-radius: 6px;
    padding: 8px 16px;
    color: #ffffff;
    font-weight: 600;
    min-height: 22px;
}
QPushButton:hover  { background-color: #4a4e6b; border-color: #7a7fa0; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:focus  { border: 1px solid #a6c8ff; outline: none; }
QPushButton:disabled {
    background-color: #2a2d42;
    color: #6c7086;
    border-color: #3d4055;
}

QPushButton#primary {
    background-color: #89b4fa;
    color: #11111b;
    font-weight: 700;
    border: none;
    padding: 8px 20px;
}
QPushButton#primary:hover   { background-color: #a6c8ff; }
QPushButton#primary:pressed { background-color: #74a0e0; }
QPushButton#primary:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#danger {
    background-color: #f38ba8;
    color: #11111b;
    font-weight: 700;
    border: none;
    padding: 8px 20px;
}
QPushButton#danger:hover   { background-color: #f5a4b8; }
QPushButton#danger:pressed { background-color: #e07898; }
QPushButton#danger:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 10px;
    background-color: #1e1e2e;
    top: -1px;
}
QTabBar::tab {
    background-color: #181825;
    color: #7f849c;
    padding: 8px 20px;
    border: 1px solid #313244;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-color: #313244;
}
QTabBar::tab:hover:!selected { background-color: #252535; }

QCheckBox {
    spacing: 8px;
    background: transparent;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #5c6080;
    border-radius: 4px;
    background-color: #181825;
}
QCheckBox::indicator:hover { border-color: #89b4fa; }
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QProgressBar {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    text-align: center;
    color: #cdd6f4;
    min-height: 24px;
    max-height: 24px;
    font-weight: 500;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 7px;
    margin: 1px;
}

QScrollBar:vertical {
    background: #181825;
    width: 12px;
    border-radius: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 6px;
    min-height: 24px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QSplitter::handle { background-color: #313244; }

QPlainTextEdit#log {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 9.5pt;
    color: #cdd6f4;
}
"""
