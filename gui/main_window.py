"""Main application window."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from core.config import ConvertConfig, PipelineConfig, SeparateConfig, TranscribeConfig
from gui.worker import PipelineWorker


LANGUAGES = ["auto", "ru", "en", "de", "fr", "es", "it", "uk", "ja", "zh", "ko"]
SAMPLE_RATES = [16000, 22050, 44100, 48000]
ENGINES = ["whisperx", "gigaam"]


def _detect_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Song Transcriber v3")
        self.resize(1080, 760)
        self._worker: Optional[PipelineWorker] = None
        self._build_ui()
        self._apply_defaults()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- top: paths + options ---
        top_box = QGroupBox("Общие настройки")
        top_layout = QFormLayout(top_box)

        self.input_edit = QLineEdit()
        top_layout.addRow("Входная папка:", self._with_browse(self.input_edit, self._pick_input_dir))

        self.output_edit = QLineEdit()
        top_layout.addRow("Выходная папка:", self._with_browse(self.output_edit, self._pick_output_dir))

        opts_row = QHBoxLayout()
        self.recursive_cb = QCheckBox("Рекурсивный обход")
        self.keep_intermediate_cb = QCheckBox("Сохранять промежуточные файлы")
        self.overwrite_cb = QCheckBox("Перезаписывать")
        opts_row.addWidget(self.recursive_cb)
        opts_row.addWidget(self.keep_intermediate_cb)
        opts_row.addWidget(self.overwrite_cb)
        opts_row.addStretch(1)
        top_layout.addRow("Опции:", opts_row)

        root.addWidget(top_box)

        # --- stages tabs ---
        tabs = QTabWidget()
        tabs.addTab(self._build_convert_tab(), "Конвертация")
        tabs.addTab(self._build_separate_tab(), "Отделение вокала")
        tabs.addTab(self._build_transcribe_tab(), "Транскрибация")
        root.addWidget(tabs)

        # --- run row ---
        run_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Запустить")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_out_btn = QPushButton("📂 Открыть выходную папку")
        self.open_out_btn.clicked.connect(self._open_output_dir)
        self.save_preset_btn = QPushButton("💾 Сохранить пресет")
        self.save_preset_btn.clicked.connect(self._save_preset)
        self.load_preset_btn = QPushButton("📁 Загрузить пресет")
        self.load_preset_btn.clicked.connect(self._load_preset)

        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch(1)
        run_row.addWidget(self.open_out_btn)
        run_row.addWidget(self.save_preset_btn)
        run_row.addWidget(self.load_preset_btn)
        root.addLayout(run_row)

        # --- progress ---
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        root.addWidget(self.progress)

        # --- log ---
        splitter = QSplitter(Qt.Vertical)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Здесь будет лог выполнения…")
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter, 1)

    def _with_browse(self, line_edit: QLineEdit, picker) -> QWidget:
        wrapper = QWidget()
        h = QHBoxLayout(wrapper)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line_edit)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(picker)
        h.addWidget(btn)
        return wrapper

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.convert_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.convert_enabled_cb)

        self.convert_sr = QComboBox()
        for sr in SAMPLE_RATES:
            self.convert_sr.addItem(str(sr), sr)
        form.addRow("Sample rate:", self.convert_sr)

        self.convert_ch = QComboBox()
        self.convert_ch.addItem("Моно", 1)
        self.convert_ch.addItem("Стерео", 2)
        form.addRow("Каналы:", self.convert_ch)

        self.convert_ffmpeg = QLineEdit()
        form.addRow("ffmpeg:", self.convert_ffmpeg)
        return w

    def _build_separate_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.separate_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.separate_enabled_cb)

        self.separate_model = QLineEdit()
        form.addRow("Модель:", self.separate_model)

        self.separate_stem = QLineEdit()
        form.addRow("Isolate stem:", self.separate_stem)

        self.separate_unblend = QLineEdit()
        form.addRow("unblend CLI:", self.separate_unblend)

        self.separate_kernels_cb = QCheckBox("Использовать кастомные ядра")
        form.addRow("", self.separate_kernels_cb)
        return w

    def _build_transcribe_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.transcribe_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.transcribe_enabled_cb)

        self.engine_cb = QComboBox()
        self.engine_cb.addItems(ENGINES)
        form.addRow("Движок:", self.engine_cb)

        self.model_edit = QLineEdit()
        form.addRow("Модель:", self.model_edit)

        self.lang_cb = QComboBox()
        self.lang_cb.setEditable(True)
        self.lang_cb.addItems(LANGUAGES)
        form.addRow("Язык:", self.lang_cb)

        self.device_cb = QComboBox()
        self.device_cb.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device:", self.device_cb)

        self.compute_type_cb = QComboBox()
        self.compute_type_cb.addItems(["auto", "int8_float16", "int8", "float16", "float32"])
        form.addRow("Compute type:", self.compute_type_cb)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        form.addRow("Batch size:", self.batch_spin)

        self.gigaam_provider_cb = QComboBox()
        self.gigaam_provider_cb.addItems(["auto", "cuda", "cpu"])
        form.addRow("GigaAM provider:", self.gigaam_provider_cb)
        return w

    # ---------------- defaults ----------------
    def _apply_defaults(self) -> None:
        cfg = PipelineConfig()
        self.output_edit.setText(str(Path(cfg.output_dir).absolute()))
        self.recursive_cb.setChecked(cfg.recursive)
        self.keep_intermediate_cb.setChecked(cfg.keep_intermediate)
        self.overwrite_cb.setChecked(cfg.overwrite)

        self.convert_enabled_cb.setChecked(cfg.convert.enabled)
        idx = self.convert_sr.findData(cfg.convert.sample_rate)
        if idx >= 0:
            self.convert_sr.setCurrentIndex(idx)
        idx = self.convert_ch.findData(cfg.convert.channels)
        if idx >= 0:
            self.convert_ch.setCurrentIndex(idx)
        self.convert_ffmpeg.setText(cfg.convert.ffmpeg_path)

        self.separate_enabled_cb.setChecked(cfg.separate.enabled)
        self.separate_model.setText(cfg.separate.model)
        self.separate_stem.setText(cfg.separate.isolate_stem)
        self.separate_unblend.setText(cfg.separate.unblend_path)
        self.separate_kernels_cb.setChecked(cfg.separate.custom_kernels)

        self.transcribe_enabled_cb.setChecked(cfg.transcribe.enabled)
        self.engine_cb.setCurrentText(cfg.transcribe.engine)
        self.model_edit.setText(cfg.transcribe.model)
        self.lang_cb.setCurrentText(cfg.transcribe.language or "auto")
        self.device_cb.setCurrentText(cfg.transcribe.device)
        self.compute_type_cb.setCurrentText(cfg.transcribe.compute_type or "auto")
        self.batch_spin.setValue(cfg.transcribe.batch_size)
        self.gigaam_provider_cb.setCurrentText(cfg.transcribe.gigaam_provider)

    # ---------------- config <-> ui ----------------
    def _collect_config(self) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.input_dir = self.input_edit.text().strip()
        cfg.output_dir = self.output_edit.text().strip() or "transcribed"
        cfg.recursive = self.recursive_cb.isChecked()
        cfg.keep_intermediate = self.keep_intermediate_cb.isChecked()
        cfg.overwrite = self.overwrite_cb.isChecked()

        cfg.convert = ConvertConfig(
            enabled=self.convert_enabled_cb.isChecked(),
            sample_rate=int(self.convert_sr.currentData()),
            channels=int(self.convert_ch.currentData()),
            ffmpeg_path=self.convert_ffmpeg.text().strip() or "ffmpeg",
        )
        cfg.separate = SeparateConfig(
            enabled=self.separate_enabled_cb.isChecked(),
            model=self.separate_model.text().strip(),
            isolate_stem=self.separate_stem.text().strip() or "vocals",
            unblend_path=self.separate_unblend.text().strip() or "unblend",
            custom_kernels=self.separate_kernels_cb.isChecked(),
        )

        lang = self.lang_cb.currentText().strip()
        compute = self.compute_type_cb.currentText().strip()
        cfg.transcribe = TranscribeConfig(
            enabled=self.transcribe_enabled_cb.isChecked(),
            engine=self.engine_cb.currentText(),
            model=self.model_edit.text().strip() or "large-v3",
            language=None if lang in ("", "auto") else lang,
            device=self.device_cb.currentText(),
            compute_type=None if compute == "auto" else compute,
            batch_size=self.batch_spin.value(),
            gigaam_provider=self.gigaam_provider_cb.currentText(),
        )
        return cfg

    def _apply_config(self, cfg: PipelineConfig) -> None:
        self.input_edit.setText(cfg.input_dir)
        self.output_edit.setText(cfg.output_dir)
        self.recursive_cb.setChecked(cfg.recursive)
        self.keep_intermediate_cb.setChecked(cfg.keep_intermediate)
        self.overwrite_cb.setChecked(cfg.overwrite)

        self.convert_enabled_cb.setChecked(cfg.convert.enabled)
        i = self.convert_sr.findData(cfg.convert.sample_rate)
        if i >= 0:
            self.convert_sr.setCurrentIndex(i)
        i = self.convert_ch.findData(cfg.convert.channels)
        if i >= 0:
            self.convert_ch.setCurrentIndex(i)
        self.convert_ffmpeg.setText(cfg.convert.ffmpeg_path)

        self.separate_enabled_cb.setChecked(cfg.separate.enabled)
        self.separate_model.setText(cfg.separate.model)
        self.separate_stem.setText(cfg.separate.isolate_stem)
        self.separate_unblend.setText(cfg.separate.unblend_path)
        self.separate_kernels_cb.setChecked(cfg.separate.custom_kernels)

        self.transcribe_enabled_cb.setChecked(cfg.transcribe.enabled)
        self.engine_cb.setCurrentText(cfg.transcribe.engine)
        self.model_edit.setText(cfg.transcribe.model)
        self.lang_cb.setCurrentText(cfg.transcribe.language or "auto")
        self.device_cb.setCurrentText(cfg.transcribe.device)
        self.compute_type_cb.setCurrentText(cfg.transcribe.compute_type or "auto")
        self.batch_spin.setValue(cfg.transcribe.batch_size)
        self.gigaam_provider_cb.setCurrentText(cfg.transcribe.gigaam_provider)

    # ---------------- handlers ----------------
    def _pick_input_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Выберите входную папку")
        if d:
            self.input_edit.setText(d)

    def _pick_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Выберите выходную папку")
        if d:
            self.output_edit.setText(d)

    def _open_output_dir(self) -> None:
        p = Path(self.output_edit.text().strip() or "transcribed").absolute()
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить пресет", "preset.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._collect_config().save(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить пресет", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._apply_config(PipelineConfig.load(Path(path)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _on_start(self) -> None:
        cfg = self._collect_config()
        if not cfg.input_dir or not Path(cfg.input_dir).is_dir():
            QMessageBox.warning(self, "Нет входной папки", "Выберите существующую входную папку.")
            return
        if not (cfg.convert.enabled or cfg.separate.enabled or cfg.transcribe.enabled):
            QMessageBox.warning(self, "Нет этапов", "Включите хотя бы один этап.")
            return

        # sanity: ffmpeg / unblend availability
        if cfg.convert.enabled and not shutil.which(cfg.convert.ffmpeg_path):
            QMessageBox.warning(
                self, "ffmpeg не найден",
                f"Команда '{cfg.convert.ffmpeg_path}' не найдена в PATH.",
            )
            return
        if cfg.separate.enabled and not shutil.which(cfg.separate.unblend_path):
            QMessageBox.warning(
                self, "unblend не найден",
                f"Команда '{cfg.separate.unblend_path}' не найдена в PATH.",
            )
            return

        self.log_view.clear()
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = PipelineWorker(cfg)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._append_log("⏹ Запрошена остановка…")

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, results: list) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        ok = sum(1 for r in results if r.error is None)
        fail = len(results) - ok
        self._append_log(f"\n✅ Готово. Успешно: {ok}, с ошибкой: {fail}.")

    def _on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"\n✗ Ошибка пайплайна: {message}")
        QMessageBox.critical(self, "Ошибка", message)
