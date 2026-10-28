"""LRC-generation and LRC-embedding pipelines."""
from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from .config import PipelineConfig
from .convert import convert_to_wav
from .lrc import embed_lrc, is_lrc, read_embedded_lyrics, segments_to_lrc
from .metadata import read_metadata
from .pipeline import (
    FileResult, _artifact_belongs_to, _save_unrecognized, collect_audio_files,
)
from .transcribe import CudaOOMError, transcribe_aligned


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]  # stage, current, total
OverallFn = Callable[[int, int], None]        # completed_files, total_files


def _pick_audio_source(
    f: Path, output_dir: Path, config: PipelineConfig, work_dir: Path, log: LogFn,
) -> Path:
    mode = (config.lrc.audio_source or "auto").lower()
    vocals = output_dir / f.stem / "vocals.wav"
    convert = output_dir / f.stem / "convert.wav"
    if mode == "vocals":
        return vocals
    if mode == "convert":
        return convert
    if mode == "input":
        return f
    if vocals.exists():
        return vocals
    if convert.exists():
        return convert
    target = work_dir / "input_16k.wav"
    log(f"  [convert] {f.name} -> 16k mono WAV (для align)")
    convert_to_wav(
        f, target, sample_rate=16000, channels=1,
        ffmpeg_path=config.convert.ffmpeg_path,
    )
    return target


def _ready_lrc(track_out: Path, source: Path) -> Optional[str]:
    """Return existing valid LRC text for this exact source track, if any."""
    if not _artifact_belongs_to(track_out, source):
        return None
    lrc_path = track_out / "text.lrc"
    if not lrc_path.is_file():
        return None
    try:
        content = lrc_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return content if content.strip() and is_lrc(content) else None


def run_lrc_pipeline(
    config: PipelineConfig,
    log: LogFn = print,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
    on_overall_progress: Optional[OverallFn] = None,
) -> list[FileResult]:
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.lrc.enabled:
        log("LRC: генерация отключена — пропуск этапа")
        return []

    files = collect_audio_files(input_dir, config.recursive)
    if not files:
        log("Аудиофайлы не найдены.")
        return []

    lang = config.lrc.language or config.transcribe.language
    log(f"LRC: {len(files)} файлов   |   язык={lang or 'auto'}   |   "
        f"источник={config.lrc.audio_source}")

    results: list[FileResult] = []
    total = len(files)
    if on_progress:
        on_progress("lrc", 0, total)
    if on_overall_progress:
        on_overall_progress(0, total)

    for idx, f in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break
        log(f"\n▶ [{idx}/{total}] {f.name}")

        track_out = output_dir / f.stem
        track_out.mkdir(parents=True, exist_ok=True)
        lrc_path = track_out / "text.lrc"
        error: Optional[str] = None

        if config.lrc.skip_existing:
            ready = _ready_lrc(track_out, f)
            if ready is not None:
                log("  ⊘ text.lrc уже готов — пропуск")
                results.append(FileResult(f, None, track_out, None, "existing+lrc"))
                if on_progress:
                    on_progress("lrc", idx, total)
                if on_overall_progress:
                    on_overall_progress(idx, total)
                continue

        tmp = Path(tempfile.mkdtemp(prefix=f"at_lrc_{f.stem}_"))
        try:
            embedded = read_embedded_lyrics(f) if config.lyrics.use_embedded_lyrics else None
            if embedded and is_lrc(embedded):
                lrc_path.write_text(embedded, encoding="utf-8")
                log("  ✓ text.lrc (взят из тегов аудио)")
                results.append(FileResult(f, None, track_out, None, "embedded+lrc"))
                continue

            audio_path = _pick_audio_source(f, output_dir, config, tmp, log)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Источник аудио не найден: {audio_path}. "
                    "Включите conversion/separation в основном пайплайне "
                    "или выберите audio_source=input."
                )
            log(f"  [align] {audio_path.name}")

            segments, detected = transcribe_aligned(
                audio_path,
                model=config.transcribe.model,
                language=lang,
                device=config.lrc.device,
                compute_type=config.transcribe.compute_type,
                batch_size=config.transcribe.batch_size,
                align_model=config.lrc.align_model,
                local_files_only=config.transcribe.local_files_only,
            )

            lrc_content = segments_to_lrc(segments)
            if not lrc_content.strip():
                log(f"  ⚠ {f.name}: текст не извлечён (пустой результат ASR)")
                results.append(FileResult(
                    f, None, None,
                    "EmptyResult: ASR returned no text", "lrc+empty",
                ))
                continue
            lrc_path.write_text(lrc_content, encoding="utf-8")
            log(f"  ✓ text.lrc ({len(segments)} строк, язык: {detected})")

            # NEW: save to JSONL for LRCLIB submission
            _save_unrecognized(
                config, f, lrc_content, source="lrc",
                metadata=read_metadata(f),
            )
            results.append(FileResult(f, None, track_out, None, "lrc"))
        except CudaOOMError as e:
            error = f"CudaOOMError: {e}"
            log(f"  ⊘ {f.name}: OOM — пропущен, идём дальше")
            results.append(FileResult(f, None, None, error, "lrc+oom"))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            log(f"  ✗ Ошибка: {error}")
            results.append(FileResult(f, None, None, error, "lrc"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if on_progress:
                on_progress("lrc", idx, total)
            if on_overall_progress:
                on_overall_progress(idx, total)

    return results


def run_embed_pipeline(
    config: PipelineConfig,
    log: LogFn = print,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
    on_overall_progress: Optional[OverallFn] = None,
) -> list[FileResult]:
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.lrc.embed_enabled:
        log("LRC embed: встраивание отключено — пропуск этапа")
        return []

    files = collect_audio_files(input_dir, config.recursive)
    if not files:
        log("Аудиофайлы не найдены.")
        return []

    log(f"Embed LRC: {len(files)} файлов   |   "
        f"режим={'in-place' if config.lrc.embed_in_place else 'копия в output'}")

    results: list[FileResult] = []
    total = len(files)
    if on_progress:
        on_progress("embed", 0, total)
    if on_overall_progress:
        on_overall_progress(0, total)

    for idx, f in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break

        lrc_path = output_dir / f.stem / "text.lrc"
        error: Optional[str] = None

        try:
            if not lrc_path.exists():
                log(f"  ⊘ {f.name}: text.lrc отсутствует, пропуск")
                results.append(FileResult(f, None, None, "no LRC", "embed"))
                continue

            lrc_content = lrc_path.read_text(encoding="utf-8")

            if config.lrc.embed_in_place:
                target = f
            else:
                target = output_dir / f.stem / f.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or config.overwrite:
                    shutil.copy2(f, target)

            embed_lrc(target, lrc_content)
            log(f"  ✓ {f.name} → {target if config.lrc.embed_in_place else target.name}")
            results.append(FileResult(f, None, target.parent, None, "embed"))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            log(f"  ✗ {f.name}: {error}")
            results.append(FileResult(f, None, None, error, "embed"))
        finally:
            if on_progress:
                on_progress("embed", idx, total)
            if on_overall_progress:
                on_overall_progress(idx, total)

    return results
