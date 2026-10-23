"""Orchestrate convert -> separate -> transcribe over a directory of audio files."""
from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import PipelineConfig
from .convert import convert_to_wav
from .separate import separate_vocals
from .transcribe import transcribe


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
}

LogFn = Callable[[str], None]
StageFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int], None]


@dataclass
class FileResult:
    input_path: Path
    text: Optional[str]
    output_dir: Optional[Path]
    error: Optional[str] = None


class Cancelled(Exception):
    """Raised internally when the stop event is set."""


def collect_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _check_cancel(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise Cancelled()


def _final_stages(config: PipelineConfig) -> set[str]:
    """Which stage produces the final artifact that must survive cleanup."""
    if config.transcribe.enabled:
        return {"transcribe"}
    if config.separate.enabled:
        return {"separate"}
    if config.convert.enabled:
        return {"convert"}
    return set()


_ARTIFACT_NAMES = {
    "convert": "convert.wav",
    "separate": "vocals.wav",
    "transcribe": "text.txt",
}


def run_pipeline(
    config: PipelineConfig,
    log: LogFn = print,
    on_stage: Optional[StageFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
) -> list[FileResult]:
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(input_dir, config.recursive)
    total = len(files)
    results: list[FileResult] = []

    if total == 0:
        log("Аудиофайлы не найдены.")
        return results

    log(f"Найдено треков: {total}")

    for idx, input_path in enumerate(files, start=1):
        _check_cancel(stop_event)
        on_progress and on_progress(idx - 1, total)
        log(f"\n▶ [{idx}/{total}] {input_path.name}")
        try:
            results.append(_process_one(input_path, config, log, on_stage, stop_event))
        except Cancelled:
            log("⏹ Отменено пользователем")
            break
        except Exception as exc:  # noqa: BLE001 — we log per-file and continue
            log(f"  ✗ Ошибка: {exc}")
            results.append(FileResult(input_path, None, None, str(exc)))
        finally:
            on_progress and on_progress(idx, total)

    return results


def _process_one(
    input_path: Path,
    config: PipelineConfig,
    log: LogFn,
    on_stage: Optional[StageFn],
    stop_event: Optional[threading.Event],
) -> FileResult:
    track_stem = input_path.stem
    track_out = Path(config.output_dir) / track_stem
    track_out.mkdir(parents=True, exist_ok=True)

    tmp_work = Path(tempfile.mkdtemp(prefix=f"at_{track_stem}_"))
    finals = _final_stages(config)

    try:
        # ---------- 1. convert ----------
        source_wav: Path = input_path
        if config.convert.enabled:
            _check_cancel(stop_event)
            on_stage and on_stage("convert", input_path.name)
            log("  [1] Конвертация в WAV...")
            target = (
                track_out / "convert.wav"
                if ("convert" in finals or config.keep_intermediate)
                else tmp_work / "convert.wav"
            )
            if config.overwrite or not target.exists():
                convert_to_wav(
                    input_path, target,
                    sample_rate=config.convert.sample_rate,
                    channels=config.convert.channels,
                    ffmpeg_path=config.convert.ffmpeg_path,
                )
            source_wav = target
            log(f"  ✓ {target.name}")

        # ---------- 2. separate ----------
        vocal_path: Path = source_wav
        if config.separate.enabled:
            _check_cancel(stop_event)
            on_stage and on_stage("separate", input_path.name)
            log("  [2] Отделение вокала...")
            raw_vocal = separate_vocals(
                source_wav, tmp_work,
                model=config.separate.model,
                isolate_stem=config.separate.isolate_stem,
                unblend_path=config.separate.unblend_path,
                custom_kernels=config.separate.custom_kernels,
            )
            if "separate" in finals or config.keep_intermediate:
                vocal_path = track_out / "vocals.wav"
                shutil.move(str(raw_vocal), str(vocal_path))
            else:
                vocal_path = raw_vocal  # stays in tmp_work
            log(f"  ✓ {vocal_path.name}")

        # ---------- 3. transcribe ----------
        text: Optional[str] = None
        if config.transcribe.enabled:
            _check_cancel(stop_event)
            on_stage and on_stage("transcribe", input_path.name)
            log("  [3] Транскрибация...")
            text = transcribe(
                vocal_path,
                engine=config.transcribe.engine,
                model=config.transcribe.model,
                language=config.transcribe.language,
                device=config.transcribe.device,
                compute_type=config.transcribe.compute_type,
                batch_size=config.transcribe.batch_size,
                gigaam_provider=config.transcribe.gigaam_provider,
            )
            (track_out / "text.txt").write_text(text, encoding="utf-8")
            log(f"  ✓ text.txt ({len(text)} символов)")

        if not config.keep_intermediate:
            _cleanup_keeping(track_out, finals)

        return FileResult(input_path, text, track_out)

    finally:
        shutil.rmtree(tmp_work, ignore_errors=True)


def _cleanup_keeping(track_out: Path, finals: set[str]) -> None:
    if not track_out.exists():
        return
    keep = {_ARTIFACT_NAMES[s] for s in finals if s in _ARTIFACT_NAMES}
    for p in track_out.iterdir():
        if p.name in keep:
            continue
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
