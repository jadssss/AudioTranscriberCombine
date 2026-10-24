"""Main application window — dark theme, mode selector, staged/sequential pipelines."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget,
)

from core.config import (
    ConvertConfig, LyricsConfig, PipelineConfig, SeparateConfig,
    ShazamConfig, TranscribeConfig,
)
from gui.worker import PipelineWorker


LANGUAGES = ["auto", "ru", "en", "de", "fr", "es", "it", "uk", "ja", "zh", "ko"]
SAMPLE_RATES = [16000, 22050, 44100, 48000]
ENGINES = ["whisperx", "gigaam"]

STAGE_LABELS = {
    "file": "Файлы",
    "identify": "Идентификация",
    "convert": "Конвертация",
    "separate": "Отделение",
    "transcribe": "Транскрибация",
}

MODE_LABELS = {
    "sequential": "Последовательно (файл за файлом)",
    "staged": "Пакетно (по этапам)",
}


def _detect_device_info() -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "⚠ PyTorch не установлен", "#f38ba8"

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"✓ CUDA: {name} · {vram:.1f} GB VRAM", "#a6e3a1"
        except Exception:  # noqa: BLE001
            return "✓ CUDA доступна", "#a6e3a1"

    return "⚠ CUDA не обнаружена — расчёт пойдёт на CPU (медленно)", "#f9e2af"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Song Transcriber")
        self.resize(1140, 880)
        self._worker: Optional[PipelineWorker] = None
        self._build_ui()
        self._apply_defaults()
        self._sync_tab_indicators()
        self._sync_engine_fields()
        self._log_device_info()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("🎵 Song Transcriber")
        title.setObjectName("appTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.device_label = QLabel("")
        self.device_label.setObjectName("deviceInfo")
        header.addWidget(self.device_label)
        root.addLayout(header)

        top_box = QGroupBox("Общие настройки")
        top_layout = QFormLayout(top_box)
        top_layout.setSpacing(10)

        self.input_edit = QLineEdit()
        top_layout.addRow("Входная папка:", self._with_browse(self.input_edit, self._pick_input_dir))

        self.output_edit = QLineEdit()
        top_layout.addRow("Выходная папка:", self._with_browse(self.output_edit, self._pick_output_dir))

        self.mode_cb = QComboBox()
        for key, label in MODE_LABELS.items():
            self.mode_cb.addItem(label, key)
        top_layout.addRow("Режим:", self.mode_cb)

        opts = QHBoxLayout()
        self.recursive_cb = QCheckBox("Рекурсивный обход")
        self.keep_intermediate_cb = QCheckBox("Сохранять промежуточные")
        self.overwrite_cb = QCheckBox("Перезаписывать")
        opts.addWidget(self.recursive_cb)
        opts.addWidget(self.keep_intermediate_cb)
        opts.addWidget(self.overwrite_cb)
        opts.addStretch(1)
        top_layout.addRow("Опции:", opts)

        root.addWidget(top_box)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_shazam_tab(), "Shazam")
        self.tabs.addTab(self._build_lrclib_tab(), "LRCLIB")
        self.tabs.addTab(self._build_convert_tab(), "Конвертация")
        self.tabs.addTab(self._build_separate_tab(), "Отделение вокала")
        self.tabs.addTab(self._build_transcribe_tab(), "Транскрибация")
        root.addWidget(self.tabs)

        self.shazam_enabled_cb.toggled.connect(self._sync_tab_indicators)
        self.lyrics_enabled_cb.toggled.connect(self._sync_tab_indicators)
        self.convert_enabled_cb.toggled.connect(self._sync_tab_indicators)
        self.separate_enabled_cb.toggled.connect(self._sync_tab_indicators)
        self.transcribe_enabled_cb.toggled.connect(self._sync_tab_indicators)
        self.engine_cb.currentTextChanged.connect(self._sync_engine_fields)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.start_btn = QPushButton("▶  Запустить")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("⏹  Стоп")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_out_btn = QPushButton("📂  Открыть результат")
        self.open_out_btn.clicked.connect(self._open_output_dir)
        self.save_preset_btn = QPushButton("💾  Сохранить пресет")
        self.save_preset_btn.clicked.connect(self._save_preset)
        self.load_preset_btn = QPushButton("📁  Загрузить пресет")
        self.load_preset_btn.clicked.connect(self._load_preset)

        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        row.addWidget(self.open_out_btn)
        row.addWidget(self.save_preset_btn)
        row.addWidget(self.load_preset_btn)
        root.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setFormat("Ожидание")
        root.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("log")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Здесь появится лог выполнения…")
        root.addWidget(self.log_view, 1)

    def _with_browse(self, edit: QLineEdit, picker) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(edit)
        btn = QPushButton("…")
        btn.setFixedWidth(36)
        btn.clicked.connect(picker)
        h.addWidget(btn)
        return w

    def _build_shazam_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self.shazam_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.shazam_enabled_cb)

        info = QLabel(
            "Идентификация трека через Shazam (unofficial API).\n"
            "При совпадении сохраняются метаданные и (если возможно) текст из LRCLIB.\n"
            "Требует ffmpeg (для конвертации в 16 кГц моно)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6adc8;")
        form.addRow(info)

        return w

    def _build_lrclib_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self.lyrics_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.lyrics_enabled_cb)

        self.lyrics_user_agent = QLineEdit()
        form.addRow("User-Agent:", self.lyrics_user_agent)

        self.lyrics_use_tags_cb = QCheckBox("Искать по тегам (artist + title)")
        form.addRow("", self.lyrics_use_tags_cb)

        self.lyrics_use_shazam_cb = QCheckBox("Искать по результату Shazam")
        form.addRow("", self.lyrics_use_shazam_cb)

        self.lyrics_after_asr_cb = QCheckBox("Проверять после ASR (fuzzy-match)")
        form.addRow("", self.lyrics_after_asr_cb)

        self.lyrics_transcribe_unresolved_cb = QCheckBox(
            "Запускать ASR, если Shazam опознал, но текста нет"
        )
        form.addRow("", self.lyrics_transcribe_unresolved_cb)

        self.lyrics_threshold = QDoubleSpinBox()
        self.lyrics_threshold.setRange(0.0, 1.0)
        self.lyrics_threshold.setSingleStep(0.05)
        self.lyrics_threshold.setDecimals(2)
        self.lyrics_threshold.setToolTip(
            "Насколько должен совпасть распознанный ASR-текст с текстом из LRCLIB,\n"
            "чтобы считать это тем же треком. 1.0 = точное совпадение, 0.0 = любое."
        )
        form.addRow("Порог fuzzy:", self.lyrics_threshold)

        return w

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self.convert_enabled_cb = QCheckBox("Включён")
        form.addRow("", self.convert_enabled_cb)

        self.convert_sr = QComboBox()
        for sr in SAMPLE_RATES:
            self.convert_sr.addItem(f"{sr} Hz", sr)
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
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

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
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

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
        self.lang_cb.setToolTip(
            "'auto' — WhisperX определяет язык на каждом файле.\n"
            "Если знаешь язык — выбери его явно, будет быстрее."
        )
        form.addRow("Язык:", self.lang_cb)

        self.device_cb = QComboBox()
        self.device_cb.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device:", self.device_cb)

        self.compute_type_cb = QComboBox()
        self.compute_type_cb.addItems(["auto", "int8_float16", "int8", "float16", "float32"])
        form.addRow("Compute type:", self.compute_type_cb)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setToolTip(
            "Размер батча сегментов ВНУТРИ одного файла (не файлов!).\n"
            "8–16 для 8 ГБ VRAM, 32–64 для 24 ГБ."
        )
        form.addRow("Batch size:", self.batch_spin)

        self.gigaam_provider_cb = QComboBox()
        self.gigaam_provider_cb.addItems(["auto", "cuda", "cpu"])
        form.addRow("GigaAM provider:", self.gigaam_provider_cb)

        self.local_files_only_cb = QCheckBox("Не проверять HuggingFace (offline)")
        self.local_files_only_cb.setToolTip(
            "Если модель уже скачана — включите, чтобы не ждать сетевую проверку."
        )
        form.addRow("", self.local_files_only_cb)
        return w

    # ---------------- visual sync ----------------

    def _log_device_info(self) -> None:
        text, color = _detect_device_info()
        self.device_label.setText(text)
        self.device_label.setStyleSheet(f"color: {color};")
        self._append_log(text)

    def _sync_tab_indicators(self) -> None:
        active = QColor("#89b4fa")
        inactive = QColor("#585b70")
        bar = self.tabs.tabBar()
        bar.setTabTextColor(0, active if self.shazam_enabled_cb.isChecked() else inactive)
        bar.setTabTextColor(1, active if self.lyrics_enabled_cb.isChecked() else inactive)
        bar.setTabTextColor(2, active if self.convert_enabled_cb.isChecked() else inactive)
        bar.setTabTextColor(3, active if self.separate_enabled_cb.isChecked() else inactive)
        bar.setTabTextColor(4, active if self.transcribe_enabled_cb.isChecked() else inactive)

    def _sync_engine_fields(self) -> None:
        is_whisperx = self.engine_cb.currentText() == "whisperx"
        for w in (self.lang_cb, self.device_cb, self.compute_type_cb, self.batch_spin):
            w.setEnabled(is_whisperx)
        self.gigaam_provider_cb.setEnabled(not is_whisperx)

    # ---------------- defaults & config <-> UI ----------------

    def _apply_defaults(self) -> None:
        self._apply_config(PipelineConfig())

    def _apply_config(self, cfg: PipelineConfig) -> None:
        self.input_edit.setText(cfg.input_dir)
        self.output_edit.setText(str(Path(cfg.output_dir).absolute()))
        self.recursive_cb.setChecked(cfg.recursive)
        self.keep_intermediate_cb.setChecked(cfg.keep_intermediate)
        self.overwrite_cb.setChecked(cfg.overwrite)

        i = self.mode_cb.findData(cfg.mode)
        if i >= 0:
            self.mode_cb.setCurrentIndex(i)

        self.shazam_enabled_cb.setChecked(cfg.shazam.enabled)

        self.lyrics_enabled_cb.setChecked(cfg.lyrics.enabled)
        self.lyrics_user_agent.setText(cfg.lyrics.user_agent)
        self.lyrics_use_tags_cb.setChecked(cfg.lyrics.use_tags_first)
        self.lyrics_use_shazam_cb.setChecked(cfg.lyrics.use_shazam_result)
        self.lyrics_after_asr_cb.setChecked(cfg.lyrics.try_after_asr)
        self.lyrics_transcribe_unresolved_cb.setChecked(
            cfg.lyrics.transcribe_if_identified_no_lyrics
        )
        self.lyrics_threshold.setValue(cfg.lyrics.fuzzy_threshold)

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
        self.local_files_only_cb.setChecked(cfg.transcribe.local_files_only)

    def _collect_config(self) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.input_dir = self.input_edit.text().strip()
        cfg.output_dir = self.output_edit.text().strip() or "transcribed"
        cfg.recursive = self.recursive_cb.isChecked()
        cfg.keep_intermediate = self.keep_intermediate_cb.isChecked()
        cfg.overwrite = self.overwrite_cb.isChecked()
        cfg.mode = self.mode_cb.currentData()

        cfg.shazam = ShazamConfig(enabled=self.shazam_enabled_cb.isChecked())

        cfg.lyrics = LyricsConfig(
            enabled=self.lyrics_enabled_cb.isChecked(),
            user_agent=self.lyrics_user_agent.text().strip() or "audio-transcriber/1.0",
            use_tags_first=self.lyrics_use_tags_cb.isChecked(),
            use_shazam_result=self.lyrics_use_shazam_cb.isChecked(),
            try_after_asr=self.lyrics_after_asr_cb.isChecked(),
            transcribe_if_identified_no_lyrics=self.lyrics_transcribe_unresolved_cb.isChecked(),
            fuzzy_threshold=float(self.lyrics_threshold.value()),
        )

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
            local_files_only=self.local_files_only_cb.isChecked(),
        )
        return cfg

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
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить пресет", "preset.json", "JSON (*.json)")
        if not path:
            return
        try:
            self._collect_config().save(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить пресет", "", "JSON (*.json)")
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
        has_work = any([
            cfg.shazam.enabled, cfg.lyrics.enabled,
            cfg.convert.enabled, cfg.separate.enabled, cfg.transcribe.enabled,
        ])
        if not has_work:
            QMessageBox.warning(self, "Нет этапов", "Включите хотя бы один этап.")
            return
        needs_ffmpeg = cfg.convert.enabled or cfg.shazam.enabled
        if needs_ffmpeg and not shutil.which(cfg.convert.ffmpeg_path):
            QMessageBox.warning(self, "ffmpeg не найден",
                                f"Команда '{cfg.convert.ffmpeg_path}' не найдена в PATH.")
            return
        if cfg.separate.enabled and not shutil.which(cfg.separate.unblend_path):
            QMessageBox.warning(self, "unblend не найден",
                                f"Команда '{cfg.separate.unblend_path}' не найдена в PATH.")
            return

        self.log_view.clear()
        self._log_device_info()
        if cfg.transcribe.enabled and cfg.transcribe.language is None:
            self._append_log("⚠ Язык = auto: WhisperX определит язык на каждом файле.")

        self.progress.setValue(0)
        self.progress.setFormat("Запуск…")
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

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        label = STAGE_LABELS.get(stage, stage)
        self.progress.setFormat(f"{label} — %v / %m")

    def _on_finished(self, results: list) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        ok = sum(1 for r in results if r.error is None)
        fail = len(results) - ok
        self.progress.setFormat(f"Готово — успешно {ok}, с ошибкой {fail}")
        self._append_log(f"\n✅ Готово. Успешно: {ok}, с ошибкой: {fail}.")

    def _on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setFormat("Ошибка")
        self._append_log(f"\n✗ Ошибка пайплайна: {message}")
        QMessageBox.critical(self, "Ошибка", message)
