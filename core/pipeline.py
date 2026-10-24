"""Orchestrate identify -> convert -> separate -> transcribe -> lyrics-match.

Pipeline order per track (short-circuits as soon as text is found):
  1. metadata tags -> LRCLIB search (if `use_tags_first`)
  2. Shazam fingerprint (if enabled)
  3. Shazam result -> LRCLIB search (if `use_shazam_result`)
  4. full pipeline: convert -> separate -> transcribe
  5. post-ASR: LRCLIB search by recognized text (fuzzy match)

Two modes:
  - sequential: run all of the above per file
  - staged:     identification pass first, then convert/separate/transcribe
                passes only for unresolved files, then post-ASR lyrics pass.
"""
from __future__ import annotations

import difflib
import json
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import PipelineConfig
from .convert import convert_to_wav
from .lyrics import LyricsMatch, search_by_identity, search_by_text
from .metadata import TrackMetadata, read_metadata
from .recognize import ShazamResult, recognize
from .separate import separate_vocals, separate_vocals_batch
from .transcribe import transcribe


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
}

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]


@dataclass
class FileResult:
    input_path: Path
    text: Optional[str]
    output_dir: Optional[Path]
    error: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[dict] = field(default=None)


class Cancelled(Exception):
    """Raised when the stop event is set."""


# ---------------------------------------------------------------- helpers

def collect_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _common_parent(paths: list[Path]) -> Path:
    """Return the deepest common parent directory of the given paths."""
    if not paths:
        raise ValueError("empty paths")
    parts = [p.resolve().parts for p in paths]
    common = []
    for chunk in zip(*parts):
        if all(c == chunk[0] for c in chunk):
            common.append(chunk[0])
        else:
            break
    if not common:
        return paths[0].parent
    return Path(*common)


def _check_cancel(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise Cancelled()


def _text_head(text: str, n_words: int = 8) -> str:
    return " ".join(text.split()[:n_words])


def _build_metadata_dict(
    metadata: TrackMetadata,
    shazam: Optional[ShazamResult],
    source: Optional[str],
) -> Optional[dict]:
    d: dict = {}
    if metadata.has_identity:
        d["tags"] = {
            "title": metadata.title,
            "artist": metadata.artist,
            "album": metadata.album,
            "duration": metadata.duration,
        }
    if shazam:
        d["shazam"] = {
            "title": shazam.title,
            "artist": shazam.artist,
            "album": shazam.album,
            "year": shazam.year,
        }
    if source:
        d["source"] = source
    return d if d else None


def _write_metadata_json(path: Path, data: Optional[dict]) -> None:
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dataclass
class IdentifyResult:
    metadata: TrackMetadata
    shazam: Optional[ShazamResult] = None
    text: Optional[str] = None
    source: Optional[str] = None
    needs_asr: bool = True


def _identify_file(
    input_path: Path,
    work_dir: Path,
    config: PipelineConfig,
    log: LogFn,
) -> IdentifyResult:
    """Run cheap identification steps. May populate .text and .source,
    and set .needs_asr=False if the track is fully resolved."""
    result = IdentifyResult(metadata=read_metadata(input_path))

    # --- tags -> LRCLIB ---
    if (
        config.lyrics.enabled
        and config.lyrics.use_tags_first
        and result.metadata.has_identity
    ):
        log("    LRCLIB по тегам...")
        match = search_by_identity(
            title=result.metadata.title,
            artist=result.metadata.artist,
            album=result.metadata.album,
            duration=result.metadata.duration,
            user_agent=config.lyrics.user_agent,
            timeout=config.lyrics.timeout,
        )
        if match:
            result.text = match.lyrics
            result.source = "tag+lrclib"
            result.needs_asr = False
            log(f"    ✓ {match.artist} — {match.title}")
            return result

    # --- Shazam ---
    if config.shazam.enabled:
        log("    Shazam...")
        shazam_wav = work_dir / "shazam_16k.wav"
        try:
            convert_to_wav(
                input_path, shazam_wav,
                sample_rate=16000, channels=1,
                ffmpeg_path=config.convert.ffmpeg_path,
            )
            result.shazam = recognize(shazam_wav)
        except Exception as e:  # noqa: BLE001
            log(f"    ⚠ Shazam error: {e}")
            result.shazam = None

        if result.shazam:
            log(f"    ✓ {result.shazam.artist} — {result.shazam.title}")

            if config.lyrics.enabled and config.lyrics.use_shazam_result:
                match = search_by_identity(
                    title=result.shazam.title,
                    artist=result.shazam.artist,
                    album=result.shazam.album,
                    duration=result.metadata.duration,
                    user_agent=config.lyrics.user_agent,
                    timeout=config.lyrics.timeout,
                )
                if match:
                    result.text = match.lyrics
                    result.source = "shazam+lrclib"
                    result.needs_asr = False
                    log("    ✓ текст найден через LRCLIB")
                    return result

            # Identified, but no lyrics
            result.source = "shazam"
            if not config.lyrics.transcribe_if_identified_no_lyrics:
                result.needs_asr = False
            log("    ⚠ текста нет")
        else:
            log("    ⚠ не распознан")

    return result


def _try_post_asr_match(
    text: str,
    config: PipelineConfig,
    log: LogFn,
) -> Optional[str]:
    if not text:
        return None

    # Filter out common ASR watermarks/credits
    ignore_phrases = ["субтитры", "редактор", "перевод", "сделал", "озвучка"]
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p in line.lower() for p in ignore_phrases):
            continue
        lines.append(line)

    if not lines:
        return None

    # Take the first line with at least 3 words, strip punctuation
    query = ""
    for line in lines:
        words = line.split()
        if len(words) >= 3:
            clean_words = [re.sub(r'[^\w\s]', '', w) for w in words]
            query = " ".join(clean_words[:6])
            break

    if not query:
        return None

    log(f"    LRCLIB по тексту: {query!r}...")

    try:
        candidates = search_by_text(
            query, limit=5,
            user_agent=config.lyrics.user_agent,
            timeout=config.lyrics.timeout,
        )
    except Exception as e:  # noqa: BLE001
        log(f"    ⚠ LRCLIB error: {e}")
        return None

    if not candidates:
        log("    ⚠ нет кандидатов")
        return None

    best: Optional[LyricsMatch] = None
    best_ratio = 0.0
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, text.lower(), c.lyrics.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = c

    if best is not None and best_ratio >= config.lyrics.fuzzy_threshold:
        log(f"    ✓ совпадение {best_ratio:.2f}: {best.artist} — {best.title}")
        return best.lyrics

    log(f"    ⚠ лучшее совпадение {best_ratio:.2f} < {config.lyrics.fuzzy_threshold}")
    return None


def _persist_artifacts(
    work_dir: Path,
    persist_dir: Path,
    config: PipelineConfig,
    has_text: bool,
    has_metadata: bool,
    log: LogFn,
) -> None:
    """Move wanted artifacts from work_dir to persist_dir."""
    persist_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    if has_metadata and (work_dir / "metadata.json").exists():
        names.append("metadata.json")
    if has_text and (work_dir / "text.txt").exists():
        names.append("text.txt")

    if config.keep_intermediate:
        for n in ("convert.wav", "vocals.wav"):
            if (work_dir / n).exists():
                names.append(n)
    elif not has_text:
        if config.separate.enabled and (work_dir / "vocals.wav").exists():
            names.append("vocals.wav")
        elif config.convert.enabled and (work_dir / "convert.wav").exists():
            names.append("convert.wav")

    for name in names:
        src = work_dir / name
        dst = persist_dir / name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        log(f"    [persist] {name}")


# ---------------------------------------------------------------- entry

def _run_identification_parallel(
    config: PipelineConfig,
    files: list[Path],
    log: LogFn,
    on_progress: Optional[ProgressFn],
    stop_event: Optional[threading.Event],
    work_root: Path,
) -> dict[Path, Optional[IdentifyResult]]:
    """Parallel identification pass (tags → LRCLIB, Shazam, etc.)."""
    if not (config.shazam.enabled or config.lyrics.enabled):
        return {}

    concurrency = max(1, int(config.shazam.identify_concurrency))
    total = len(files)

    if concurrency == 1 or total == 1:
        results: dict[Path, Optional[IdentifyResult]] = {}
        for i, f in enumerate(files, 1):
            if stop_event is not None and stop_event.is_set():
                break
            if on_progress:
                on_progress(i, total, "identify")
            try:
                results[f] = _identify_file(f, work_root / f.stem, config, log)
            except Exception as e:  # noqa: BLE001
                log(f"  ✗ {f.name} (identify): {e}")
                results[f] = None
        return results

    log(f"  Идентификация: {total} файлов, конкурентность {concurrency}")

    def worker(f: Path) -> tuple[Path, Optional[IdentifyResult]]:
        if stop_event is not None and stop_event.is_set():
            return f, None

        def prefixed(msg: str) -> None:
            log(f"  [{f.name}] {msg.strip()}")

        try:
            ident = _identify_file(f, work_root / f.stem, config, prefixed)
            return f, ident
        except Exception as e:  # noqa: BLE001
            log(f"  ✗ {f.name} (identify): {e}")
            return f, None

    results: dict[Path, Optional[IdentifyResult]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                _, ident = fut.result()
                results[f] = ident
            except Exception as e:  # noqa: BLE001
                log(f"  ✗ {f.name} (identify): {e}")
                results[f] = None
            done += 1
            if on_progress:
                on_progress(done, total, "identify")

    return results


def _run_convert_parallel(
    config: PipelineConfig,
    files: list[Path],
    state: dict,
    log: LogFn,
    on_progress: Optional[ProgressFn],
    stop_event: Optional[threading.Event],
) -> None:
    """Parallel convert stage. Writes state[f]['audio_path'] or state[f]['error']."""
    concurrency = max(1, int(config.convert.convert_concurrency))
    total = len(files)

    def worker(f: Path) -> tuple[Path, Optional[Path], Optional[str]]:
        if stop_event is not None and stop_event.is_set():
            return f, None, None
        try:
            target = Path(tempfile.mkdtemp(prefix=f"at_{f.stem}_")) / "convert.wav"
            if config.overwrite or not target.exists():
                convert_to_wav(
                    f, target,
                    sample_rate=config.convert.sample_rate,
                    channels=config.convert.channels,
                    ffmpeg_path=config.convert.ffmpeg_path,
                )
            return f, target, None
        except Exception as e:  # noqa: BLE001
            return f, None, f"{type(e).__name__}: {e}"

    if concurrency == 1 or total == 1:
        for i, f in enumerate(files, 1):
            if stop_event is not None and stop_event.is_set():
                break
            if on_progress:
                on_progress(i, total, "convert")
            f2, target, err = worker(f)
            if err:
                state[f2]["error"] = err
                log(f"  ✗ {f2.name}: {err}")
            else:
                state[f2]["audio_path"] = target
                log(f"  ✓ {f2.name}")
        return

    log(f"  Конвертация: {total} файлов, конкурентность {concurrency}")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            f2, target, err = fut.result()
            if err:
                state[f2]["error"] = err
                log(f"  ✗ {f2.name}: {err}")
            else:
                state[f2]["audio_path"] = target
                log(f"  ✓ {f2.name}")
            done += 1
            if on_progress:
                on_progress(done, total, "convert")

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
    log(
        f"Найдено треков: {len(files)}   |   режим: {mode}   |   "
        f"keep_intermediate={config.keep_intermediate}"
    )
    if config.shazam.enabled:
        log("  Shazam: включён")
    if config.lyrics.enabled:
        log(f"  LRCLIB: включён (порог fuzzy={config.lyrics.fuzzy_threshold:.2f})")

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
    results: list[FileResult] = []
    output_root = Path(config.output_dir)

    for idx, f in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break
        if on_progress:
            on_progress(idx - 1, total, "file")
        log(f"\n▶ [{idx}/{total}] {f.name}")

        tmp_root = Path(tempfile.mkdtemp(prefix=f"at_{f.stem}_"))
        track_out = output_root / f.stem
        track_out.mkdir(parents=True, exist_ok=True)

        text: Optional[str] = None
        source: Optional[str] = None
        error: Optional[str] = None
        metadata_dict: Optional[dict] = None

        try:
            # --- Phase 1: identification ---
            ident = _identify_file(f, tmp_root, config, log)

            if not ident.needs_asr:
                text = ident.text
                source = ident.source
                metadata_dict = _build_metadata_dict(ident.metadata, ident.shazam, source)
                if text:
                    (tmp_root / "text.txt").write_text(text, encoding="utf-8")
                    log(f"  ✓ text.txt ({len(text)} символов, {source})")
                if metadata_dict:
                    _write_metadata_json(tmp_root / "metadata.json", metadata_dict)
                    log("  ✓ metadata.json")
                _persist_artifacts(
                    tmp_root, track_out, config,
                    has_text=text is not None,
                    has_metadata=bool(metadata_dict),
                    log=log,
                )
                results.append(FileResult(f, text, track_out, None, source, metadata_dict))
                continue

            # --- Phase 2: full pipeline ---
            audio_path: Path = f
            if config.convert.enabled:
                _check_cancel(stop_event)
                log("  [1] Конвертация в WAV...")
                target = tmp_root / "convert.wav"
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
                target = tmp_root / "vocals.wav"
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
                    local_files_only=config.transcribe.local_files_only,
                )
                source = config.transcribe.engine
                log(f"  ✓ распознано {len(text)} символов")

                if config.lyrics.enabled and config.lyrics.try_after_asr:
                    matched = _try_post_asr_match(text, config, log)
                    if matched:
                        text = matched
                        source = f"{source}+lrclib"

                (tmp_root / "text.txt").write_text(text, encoding="utf-8")

            metadata_dict = _build_metadata_dict(ident.metadata, ident.shazam, source)
            if metadata_dict:
                _write_metadata_json(tmp_root / "metadata.json", metadata_dict)

            _persist_artifacts(
                tmp_root, track_out, config,
                has_text=text is not None,
                has_metadata=bool(metadata_dict),
                log=log,
            )

        except Cancelled:
            log("⏹ Отменено")
            break
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            log(f"  ✗ Ошибка: {error}")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
            if on_progress:
                on_progress(idx, total, "file")
            results.append(
                FileResult(f, text, track_out if not error else None, error, source, metadata_dict)
            )

    _log_summary(results, log)
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

    state: dict[Path, dict] = {
        f: {
            "audio_path": f,
            "ident": None,
            "text": None,
            "source": None,
            "metadata_dict": None,
            "error": None,
            "resolved": False,
        }
        for f in files
    }

    def active() -> list[Path]:
        return [f for f in files if state[f]["error"] is None and not state[f]["resolved"]]

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    # --- Stage 0: identification ---
    if config.shazam.enabled or config.lyrics.enabled:
        log(f"\n══ Этап 0 · Идентификация ({total} файлов) ══")
        ident_tmp = Path(tempfile.mkdtemp(prefix="at_ident_"))
        try:
            for i, f in enumerate(files, 1):
                if cancelled():
                    log("⏹ Отменено")
                    break
                if on_progress:
                    on_progress(i, total, "identify")
                try:
                    ident = _identify_file(f, ident_tmp / f.stem, config, log)
                    state[f]["ident"] = ident

                    if not ident.needs_asr:
                        state[f]["resolved"] = True
                        state[f]["text"] = ident.text
                        state[f]["source"] = ident.source
                        state[f]["metadata_dict"] = _build_metadata_dict(
                            ident.metadata, ident.shazam, ident.source,
                        )
                        # Persist resolved tracks right away
                        track_out = output_root / f.stem
                        track_out.mkdir(parents=True, exist_ok=True)
                        if ident.text:
                            (track_out / "text.txt").write_text(ident.text, encoding="utf-8")
                        if state[f]["metadata_dict"]:
                            _write_metadata_json(track_out / "metadata.json", state[f]["metadata_dict"])
                        log(f"  ✓ {f.name} [{ident.source}]")
                    else:
                        log(f"  → {f.name} (нужен полный пайплайн)")
                except Exception as e:  # noqa: BLE001
                    state[f]["error"] = f"{type(e).__name__}: {e}"
                    log(f"  ✗ {f.name}: {e}")
        finally:
            shutil.rmtree(ident_tmp, ignore_errors=True)

    # --- Stage 1: convert ---
    if config.convert.enabled and not cancelled():
        todo = active()
        log(f"\n══ Этап 1 · Конвертация ({len(todo)} файлов) ══")
        for i, f in enumerate(todo, 1):
            if cancelled():
                log("⏹ Отменено")
                break
            if on_progress:
                on_progress(i, len(todo), "convert")
            try:
                target = Path(tempfile.mkdtemp(prefix=f"at_{f.stem}_")) / "convert.wav"
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

    # ---- Stage 2: separate (batched) ----
    if config.separate.enabled and not cancelled():
        todo = active()
        log(f"\n══ Этап 2/3 · Отделение вокала ({len(todo)} файлов) ══")
        if not todo:
            pass
        elif len(todo) == 1:
            # Single file — no batching advantage
            f = todo[0]
            if on_progress:
                on_progress(1, 1, "separate")
            try:
                src = state[f]["audio_path"]
                target = src.parent / "vocals.wav"
                with tempfile.TemporaryDirectory() as tmp:
                    raw = separate_vocals(
                        src, Path(tmp),
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
        else:
            # Batch: one unblend invocation, model loaded once
            input_wavs = [state[f]["audio_path"] for f in todo]
            common_root = _common_parent(input_wavs)
            log(f"  Батч: {len(todo)} файлов, один вызов unblend (модель загрузится один раз)")
            if on_progress:
                on_progress(0, len(todo), "separate")
            try:
                mapping = separate_vocals_batch(
                    input_wavs,
                    common_root,
                    model=config.separate.model,
                    isolate_stem=config.separate.isolate_stem,
                    unblend_path=config.separate.unblend_path,
                    custom_kernels=config.separate.custom_kernels,
                    progress_callback=lambda p: on_progress(p, len(todo), "separate") if on_progress else None,
                )
                for f in todo:
                    src = state[f]["audio_path"]
                    vocal = mapping.get(src)
                    if vocal is None:
                        state[f]["error"] = "batch produced no output"
                        log(f"  ✗ {f.name}: нет результата в батче")
                        continue
                    state[f]["audio_path"] = vocal
                    log(f"  ✓ {f.name}")
                if on_progress:
                    on_progress(len(todo), len(todo), "separate")
            except Exception as e:  # noqa: BLE001
                log(f"  ✗ Батч упал: {e}")
                log("  ↻ Fallback: по одному...")
                for i, f in enumerate(todo, 1):
                    if cancelled():
                        break
                    if on_progress:
                        on_progress(i, len(todo), "separate")
                    try:
                        src = state[f]["audio_path"]
                        target = src.parent / "vocals.wav"
                        with tempfile.TemporaryDirectory() as tmp:
                            raw = separate_vocals(
                                src, Path(tmp),
                                model=config.separate.model,
                                isolate_stem=config.separate.isolate_stem,
                                unblend_path=config.separate.unblend_path,
                                custom_kernels=config.separate.custom_kernels,
                            )
                            shutil.move(str(raw), str(target))
                        state[f]["audio_path"] = target
                        log(f"  ✓ {f.name}")
                    except Exception as inner:  # noqa: BLE001
                        state[f]["error"] = f"{type(inner).__name__}: {inner}"
                        log(f"  ✗ {f.name}: {inner}")

    # --- Stage 3: transcribe ---
    if config.transcribe.enabled and not cancelled():
        todo = active()
        log(f"\n══ Этап 3 · Транскрибация ({len(todo)} файлов) ══")
        for i, f in enumerate(todo, 1):
            if cancelled():
                log("⏹ Отменено")
                break
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
                    local_files_only=config.transcribe.local_files_only,
                )
                state[f]["text"] = text
                state[f]["source"] = config.transcribe.engine
                log(f"  ✓ {f.name} ({len(text)} символов)")
            except Exception as e:  # noqa: BLE001
                state[f]["error"] = f"{type(e).__name__}: {e}"
                log(f"  ✗ {f.name}: {e}")
            finally:
                if on_progress:
                    on_progress(i, len(todo), "transcribe")

    # --- Stage 4: post-ASR lyrics match + persist ---
    if not cancelled():
        log("\n══ Финализация ══")
        for f in files:
            s = state[f]
            if s["error"] is not None:
                continue
            if s["resolved"]:
                continue  # already persisted in identification pass

            text = s["text"]
            source = s["source"]

            if text and config.lyrics.enabled and config.lyrics.try_after_asr:
                matched = _try_post_asr_match(text, config, log)
                if matched:
                    text = matched
                    source = f"{source}+lrclib"
                    s["text"] = text
                    s["source"] = source

            ident = s["ident"]
            metadata_dict = _build_metadata_dict(
                ident.metadata, ident.shazam, source,
            ) if ident else None
            s["metadata_dict"] = metadata_dict

            track_out = output_root / f.stem
            track_out.mkdir(parents=True, exist_ok=True)

            audio_work = s["audio_path"]
            work_dir = audio_work.parent if isinstance(audio_work, Path) else track_out
            if text:
                (work_dir / "text.txt").write_text(text, encoding="utf-8")
            if metadata_dict:
                _write_metadata_json(work_dir / "metadata.json", metadata_dict)

            _persist_artifacts(
                work_dir, track_out, config,
                has_text=text is not None,
                has_metadata=bool(metadata_dict),
                log=log,
            )
            log(f"  ✓ {f.name}")

    # cleanup temp dirs
    for f in files:
        audio_path = state[f]["audio_path"]
        if isinstance(audio_path, Path) and audio_path.parent.name.startswith(f"at_{f.stem}"):
            shutil.rmtree(audio_path.parent, ignore_errors=True)

    results = [
        FileResult(
            f,
            state[f]["text"],
            output_root / f.stem if state[f]["error"] is None else None,
            state[f]["error"],
            state[f]["source"],
            state[f]["metadata_dict"],
        )
        for f in files
    ]
    _log_summary(results, log)
    return results


# ---------------------------------------------------------------- summary

def _log_summary(results: list[FileResult], log: LogFn) -> None:
    from collections import Counter
    counts = Counter(r.source or ("error" if r.error else "no_result") for r in results)
    log("\n══ Итог ══")
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        log(f"  {src}: {n}")
