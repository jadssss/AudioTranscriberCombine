"""Orchestrate convert -> separate -> transcribe.

Two modes:
  - sequential: for each file, run all enabled stages before moving to the next.
  - staged:     run stage 1 over all files, then stage 2 over all files, then stage 3.
"""
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
ProgressFn = Callable[[int, int, str], None]   # done, total, stage


@dataclass
class FileResult:
    input_path: Path
    text: Optional[str]
    output_dir: Optional[Path]
    error: Optional[str] = None


class Cancelled(Exception):
    """Raised when the stop event is set."""


# ---------------------------------------------------------------- helpers

_ARTIFACT_NAMES = {
    "convert": "convert.wav",
    "separate": "vocals.wav",
    "transcribe": "text.txt",
}


def collect_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _final_stages(config: PipelineConfig) -> set[str]:
    if config.transcribe.enabled:
        return {"transcribe"}
    if config.separate.enabled:
        return {"separate"}
    if config.convert.enabled:
        return {"convert"}
    return set()


def _check_cancel(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise Cancelled()


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


# ---------------------------------------------------------------- entry

def run_pipeline(
    config: PipelineConfig,
    log: LogFn = print,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
) -> list[FileResult]:
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(input_dir, config.recursive)
    if not files:
        log("Аудиофайлы не найдены.")
        return []

    mode = (config.mode or "sequential").lower()
    log(f"Найдено треков: {len(files)}   |   режим: {mode}")

    if mode == "staged":
        return _run_staged(config, files, log, on_progress, stop_event)
    return _run_sequential(config, files, log, on_progress, stop_event)


# ---------------------------------------------------------------- sequential

def _run_sequential(
    config: PipelineConfig,
    files: list[Path],
    log: LogFn,
    on_progress: Optional[ProgressFn],
    stop_event: Optional[threading.Event],
) -> list[FileResult]:
    total = len(files)
    finals = _final_stages(config)
    results: list[FileResult] = []

    for idx, f in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break
        if on_progress:
            on_progress(idx - 1, total, "file")
        log(f"\n▶ [{idx}/{total}] {f.name}")

        track_out = Path(config.output_dir) / f.stem
        track_out.mkdir(parents=True, exist_ok=True)

        audio_path: Path = f
        text: Optional[str] = None
        error: Optional[str] = None

        try:
            if config.convert.enabled:
                _check_cancel(stop_event)
                log("  [1] Конвертация в WAV...")
                target = track_out / "convert.wav"
                if config.overwrite or not target.exists():
                    convert_to_wav(
                        f, target,
                        sample_rate=config.convert.sample_rate,
                        channels=config.convert.channels,
                        ffmpeg_path=config.convert.ffmpeg_path,
                    )
                audio_path = target
                log("  ✓ convert.wav")

            if config.separate.enabled:
                _check_cancel(stop_event)
                log("  [2] Отделение вокала...")
                target = track_out / "vocals.wav"
                with tempfile.TemporaryDirectory() as tmp:
                    raw = separate_vocals(
                        audio_path, Path(tmp),
                        model=config.separate.model,
                        isolate_stem=config.separate.isolate_stem,
                        unblend_path=config.separate.unblend_path,
                        custom_kernels=config.separate.custom_kernels,
                    )
                    shutil.move(str(raw), str(target))
                audio_path = target
                log("  ✓ vocals.wav")

            if config.transcribe.enabled:
                _check_cancel(stop_event)
                log("  [3] Транскрибация...")
                text = transcribe(
                    audio_path,
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

        except Cancelled:
            log("⏹ Отменено")
            break
        except Exception as e:  # noqa: BLE001 — per-file isolation
            error = f"{type(e).__name__}: {e}"
            log(f"  ✗ Ошибка: {error}")
        finally:
            if on_progress:
                on_progress(idx, total, "file")
            results.append(FileResult(f, text, track_out if not error else None, error))

    return results


# ---------------------------------------------------------------- staged

def _run_staged(
    config: PipelineConfig,
    files: list[Path],
    log: LogFn,
    on_progress: Optional[ProgressFn],
    stop_event: Optional[threading.Event],
) -> list[FileResult]:
    output_root = Path(config.output_dir)
    total = len(files)

    # per-file state
    state: dict[Path, dict] = {
        f: {"audio_path": f, "track_out": None, "text": None, "error": None}
        for f in files
    }

    def track_out_for(f: Path) -> Path:
        s = state[f]
        if s["track_out"] is None:
            p = output_root / f.stem
            p.mkdir(parents=True, exist_ok=True)
            s["track_out"] = p
        return s["track_out"]

    def active() -> list[Path]:
        return [f for f in files if state[f]["error"] is None]

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    # ---- Stage 1: convert ----
    if config.convert.enabled:
        log(f"\n══ Этап 1/3 · Конвертация ({total} файлов) ══")
        for i, f in enumerate(files, 1):
            if cancelled():
                log("⏹ Отменено")
                break
            if on_progress:
                on_progress(i, total, "convert")
            try:
                target = track_out_for(f) / "convert.wav"
                if config.overwrite or not target.exists():
                    convert_to_wav(
                        f, target,
                        sample_rate=config.convert.sample_rate,
                        channels=config.convert.channels,
                        ffmpeg_path=config.convert.ffmpeg_path,
                    )
                state[f]["audio_path"] = target
                log(f"  ✓ {f.name}")
            except Exception as e:  # noqa: BLE001
                state[f]["error"] = f"{type(e).__name__}: {e}"
                log(f"  ✗ {f.name}: {e}")

    # ---- Stage 2: separate ----
    if config.separate.enabled and not cancelled():
        todo = active()
        log(f"\n══ Этап 2/3 · Отделение вокала ({len(todo)} файлов) ══")
        for i, f in enumerate(todo, 1):
            if cancelled():
                log("⏹ Отменено")
                break
            if on_progress:
                on_progress(i, len(todo), "separate")
            try:
                target = track_out_for(f) / "vocals.wav"
                with tempfile.TemporaryDirectory() as tmp:
                    raw = separate_vocals(
                        state[f]["audio_path"], Path(tmp),
                        model=config.separate.model,
                        isolate_stem=config.separate.isolate_stem,
                        unblend_path=config.separate.unblend_path,
                        custom_kernels=config.separate.custom_kernels,
                    )
                    shutil.move(str(raw), str(target))
                state[f]["audio_path"] = target
                log(f"  ✓ {f.name}")
            except Exception as e:  # noqa: BLE001
                state[f]["error"] = f"{type(e).__name__}: {e}"
                log(f"  ✗ {f.name}: {e}")

    # ---- Stage 3: transcribe ----
    if config.transcribe.enabled and not cancelled():
        todo = active()
        log(f"\n══ Этап 3/3 · Транскрибация ({len(todo)} файлов) ══")
        for i, f in enumerate(todo, 1):
            if cancelled():
                log("⏹ Отменено")
                break
            if on_progress:
                on_progress(i, len(todo), "transcribe")
            try:
                text = transcribe(
                    state[f]["audio_path"],
                    engine=config.transcribe.engine,
                    model=config.transcribe.model,
                    language=config.transcribe.language,
                    device=config.transcribe.device,
                    compute_type=config.transcribe.compute_type,
                    batch_size=config.transcribe.batch_size,
                    gigaam_provider=config.transcribe.gigaam_provider,
                )
                (track_out_for(f) / "text.txt").write_text(text, encoding="utf-8")
                state[f]["text"] = text
                log(f"  ✓ {f.name} ({len(text)} символов)")
            except Exception as e:  # noqa: BLE001
                state[f]["error"] = f"{type(e).__name__}: {e}"
                log(f"  ✗ {f.name}: {e}")

    # ---- Cleanup ----
    if not config.keep_intermediate:
        finals = _final_stages(config)
        for f in files:
            s = state[f]
            if s["error"] is None and s["track_out"] is not None:
                _cleanup_keeping(s["track_out"], finals)

    return [
        FileResult(f, state[f]["text"], state[f]["track_out"], state[f]["error"])
        for f in files
    ]
