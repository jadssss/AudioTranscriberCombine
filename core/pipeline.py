"""Orchestrate identify -> convert -> separate -> transcribe -> lyrics-match."""
from __future__ import annotations

import difflib
import json
import shutil
import sys
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
from .transcribe import CudaOOMError, transcribe
from .lrc import is_lrc, read_embedded_lyrics


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
}

LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]  # stage, current, total_for_stage
OverallFn = Callable[[int, int], None]        # completed_files, total_files


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

def _free_memory() -> None:
    """Free unreferenced RAM and VRAM. Keeps model caches alive.

    Never import torch here: if the current run does not use ASR, importing it
    would initialize a CUDA context and steal VRAM from separation. If torch is
    already loaded by the ASR stage, we can safely empty its cache.
    """
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — never fail cleanup on a broken CUDA context
        pass


def collect_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _common_parent(paths: list[Path]) -> Path:
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
    input_path: Optional[Path] = None,
) -> Optional[dict]:
    d: dict = {}
    if input_path is not None:
        try:
            d["path"] = str(input_path.resolve())
        except (OSError, RuntimeError):
            d["path"] = str(input_path)
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


def _artifact_belongs_to(track_out: Path, input_path: Path) -> bool:
    """Check that an output folder was created for this exact input file.

    ``metadata.json`` now stores the source path. If an output folder with the
    same stem belongs to a different file (e.g. two albums both contain
    ``01 - Intro.mp3``), its artifacts must not be treated as "ready" for this
    track. Legacy folders without a ``path`` key are still accepted.
    """
    meta_file = track_out / "metadata.json"
    if not meta_file.is_file():
        return True
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    stored = data.get("path") if isinstance(data, dict) else None
    if not stored:
        return True
    try:
        return Path(str(stored)).resolve() == input_path.resolve()
    except (OSError, RuntimeError):
        return Path(str(stored)) == input_path


def _write_metadata_json(path: Path, data: Optional[dict]) -> None:
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _existing_artifacts(
    output_root: Path, input_path: Path,
) -> tuple[bool, bool, bool]:
    """Return (has_text, has_vocals, has_convert) for the matching output folder.

    Artifacts are ignored when ``metadata.json`` says the folder belongs to a
    different input file with the same stem.
    """
    track_out = output_root / input_path.stem
    if not _artifact_belongs_to(track_out, input_path):
        return False, False, False
    return (
        (track_out / "text.txt").is_file(),
        (track_out / "vocals.wav").is_file(),
        (track_out / "convert.wav").is_file(),
    )


def _needs_work(config: PipelineConfig, output_root: Path, f: Path) -> bool:
    """True if at least one enabled stage still has to do something for ``f``.

    This mirrors the resume rules used by the staged/sequential pipelines:
      * text.txt + skip_existing        -> resolved, nothing to do
      * vocals.wav + resume_partial     -> convert/separate are skipped
      * convert.wav + resume_partial    -> convert is skipped
    Files with no pending work are filtered out before batching, so batches
    stay full instead of being filled by already-finished tracks.
    """
    has_text, has_vocals, has_convert = _existing_artifacts(output_root, f)

    if config.skip_existing and has_text:
        return False  # resolved by the same rule the pipelines use

    vocals_ready = config.resume_partial and has_vocals
    convert_ready = config.resume_partial and (has_vocals or has_convert)

    if config.shazam.enabled or config.lyrics.enabled:
        return True  # identification still has to run (no resolved text)
    if config.convert.enabled and not convert_ready:
        return True
    if config.separate.enabled and not vocals_ready:
        return True
    if config.transcribe.enabled:
        return True  # text is missing or skip_existing is disabled
    return False


def _save_unrecognized(
    config: PipelineConfig,
    f: Path,
    text: Optional[str],
    source: Optional[str],
    metadata: Optional[TrackMetadata] = None,
    lrc_path: Optional[Path] = None,
) -> None:
    """Append a record to the JSONL file for texts NOT confirmed by LRCLIB."""
    if not config.lyrics.save_unrecognized or not text:
        return
    if source and "+lrclib" in source:
        return

    if config.lyrics.unrecognized_path:
        out_path = Path(config.lyrics.unrecognized_path)
    else:
        out_path = Path(config.output_dir) / "unrecognized_lyrics.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "file": str(f),
        "trackName": (metadata.title if metadata else None) or f.stem,
        "artistName": metadata.artist if metadata else None,
        "albumName": metadata.album if metadata else None,
        "duration": metadata.duration if metadata else None,
        "plainLyrics": text if not is_lrc(text) else None,
        "syncedLyrics": text if is_lrc(text) else None,
        "source": source,
    }
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


@dataclass
class IdentifyResult:
    metadata: TrackMetadata
    shazam: Optional[ShazamResult] = None
    text: Optional[str] = None
    source: Optional[str] = None
    needs_asr: bool = True
    lrclib_attempted: bool = False  # author/title LRCLIB search was attempted


def _identify_file(
    input_path: Path,
    work_dir: Path,
    config: PipelineConfig,
    log: LogFn,
) -> IdentifyResult:
    """Run cheap identification steps. May populate .text and .source,
    and set .needs_asr=False if the track is fully resolved.

    Order:
      0. embedded lyrics in file tags (USLT/LYRICS/©lyr) — cheapest, no network
      1. Shazam fingerprint
      2. LRCLIB identity search — exactly once:
         Shazam result if available, otherwise artist/title from file tags
    """
    result = IdentifyResult(metadata=read_metadata(input_path))

    # --- 0) embedded lyrics в самом файле ---
    if config.lyrics.enabled and config.lyrics.use_embedded_lyrics:
        embedded = read_embedded_lyrics(input_path)
        if embedded and embedded.strip():
            if is_lrc(embedded):
                result.text = embedded
                result.source = "embedded+lrc"
                log("    ✓ встроенный LRC найден")
            else:
                result.text = embedded
                result.source = "embedded"
                log("    ✓ встроенный текст найден")
            result.needs_asr = False
            return result

    # --- 1) Shazam ---
    if config.shazam.enabled:
        log("    Shazam...")
        shazam_wav = work_dir / "shazam_16k.wav"
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            convert_to_wav(
                input_path, shazam_wav,
                sample_rate=16000, channels=1,
                ffmpeg_path=config.convert.ffmpeg_path,
            )
            result.shazam = recognize(shazam_wav)
        except Exception as e:  # noqa: BLE001
            log(f"    ⚠ Shazam error: {e}")
            result.shazam = None
        finally:
            # 16 kHz WAV для Shazam больше не нужен — удаляем сразу,
            # чтобы не копить десятки гигабайт в /tmp на длинных батчах.
            try:
                shazam_wav.unlink(missing_ok=True)
            except OSError:
                pass

        if result.shazam:
            log(f"    ✓ {result.shazam.artist} — {result.shazam.title}")
        else:
            log("    ⚠ не распознан")

    # --- 2) LRCLIB: ровно один поиск по автору/названию ---
    # Приоритет — результат Shazam. Если Shazam ничего не нашёл (или его
    # результат не используется), берём artist/title из тегов файла.
    if config.lyrics.enabled:
        if result.shazam and config.lyrics.use_shazam_result:
            log("    LRCLIB по Shazam...")
            result.lrclib_attempted = True
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
                log(f"    ✓ {match.artist} — {match.title}")
                return result
            log("    ⚠ LRCLIB: не найдено")
        elif config.lyrics.use_tags_first and result.metadata.has_identity:
            log("    LRCLIB по тегам...")
            result.lrclib_attempted = True
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
            log("    ⚠ LRCLIB: не найдено")

    # --- 3) Identified, but no lyrics ---
    if result.shazam:
        result.source = "shazam"
        if not config.lyrics.transcribe_if_identified_no_lyrics:
            result.needs_asr = False
        if not result.lrclib_attempted:
            log("    ⚠ текста нет")

    return result


def _try_post_asr_match(text: str, config: PipelineConfig, log: LogFn) -> Optional[str]:
    if not text:
        return None
    ignore_phrases = ["субтитры", "редактор", "перевод", "сделал", "озвучка"]
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or any(p in line.lower() for p in ignore_phrases):
            continue
        lines.append(line)
    if not lines:
        return None
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
            query, limit=5, user_agent=config.lyrics.user_agent,
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
    if work_dir.resolve() == persist_dir.resolve():
        return  # nothing to move — same directory

    persist_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    if has_metadata and (work_dir / "metadata.json").exists():
        names.append("metadata.json")
    if has_text and (work_dir / "text.txt").exists():
        names.append("text.txt")
    if has_text and (work_dir / "text.lrc").exists():
        names.append("text.lrc")

    if config.keep_intermediate:
        for n in ("convert.wav", "vocals.wav"):
            if (work_dir / n).exists():
                names.append(n)
    elif not has_text:
        if config.separate.enabled and (work_dir / "vocals.wav").exists():
            names.append("vocals.wav")
        elif config.convert.enabled and (work_dir / "convert.wav").exists():
            names.append("convert.wav")

    if config.separate.save_vocals_copy and (work_dir / "vocals.wav").exists():
        if "vocals.wav" not in names:
            names.append("vocals.wav")

    for name in names:
        src = work_dir / name
        dst = persist_dir / name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        log(f"    [persist] {name}")


# ---------------------------------------------------------------- entry

def _run_identification_parallel(
    config: PipelineConfig, files: list[Path], log: LogFn,
    on_progress: Optional[ProgressFn], stop_event: Optional[threading.Event],
    work_root: Path,
) -> dict[Path, Optional[IdentifyResult]]:
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
                on_progress("identify", i, total)
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
            return f, _identify_file(f, work_root / f.stem, config, prefixed)
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
                on_progress("identify", done, total)
    return results


def _run_convert_parallel(
    config: PipelineConfig, files: list[Path], state: dict, log: LogFn,
    on_progress: Optional[ProgressFn], stop_event: Optional[threading.Event],
    temp_registry: Optional[list[Path]] = None,
) -> None:
    concurrency = max(1, int(config.convert.convert_concurrency))
    total = len(files)

    def worker(f: Path) -> tuple[Path, Optional[Path], Optional[str]]:
        if stop_event is not None and stop_event.is_set():
            return f, None, None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"at_{f.stem}_"))
            if temp_registry is not None:
                temp_registry.append(temp_dir)
            target = temp_dir / "convert.wav"
            if config.overwrite or not target.exists():
                convert_to_wav(
                    f, target, sample_rate=config.convert.sample_rate,
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
                on_progress("convert", i, total)
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
                on_progress("convert", done, total)


def run_pipeline(
    config: PipelineConfig, log: LogFn = print,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
    on_overall_progress: Optional[OverallFn] = None,
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
    total_all = len(files)
    log(f"Найдено треков: {total_all}   |   режим: {mode}   |   "
        f"keep_intermediate={config.keep_intermediate}")
    if config.separate.save_vocals_copy:
        log("  ✓ Вокал будет сохраняться всегда (save_vocals_copy=True)")
    if config.shazam.enabled:
        log("  Shazam: включён")
    if config.lyrics.enabled:
        log(f"  LRCLIB: включён (порог fuzzy={config.lyrics.fuzzy_threshold:.2f})")
    if config.skip_existing:
        log("  Resume: готовые файлы (text.txt) будут пропущены")
    if config.resume_partial:
        log("  Resume: существующие vocals.wav / convert.wav будут переиспользованы")

    # Pre-scan before chunking: remove files that are already complete for the
    # current set of enabled stages. Without this, skipped tracks consumed
    # batch slots and a staged run produced many half-empty batches.
    pending = [f for f in files if _needs_work(config, output_dir, f)]
    already_done = total_all - len(pending)
    if already_done:
        log(f"  ✓ Уже готово: {already_done} — в батчи не попадут")
        log(f"  → К обработке: {len(pending)}")
    files = pending
    if not files:
        log("  Все файлы уже готовы — работа не требуется.")
        if on_overall_progress:
            on_overall_progress(total_all, total_all)
        return []

    if on_overall_progress:
        on_overall_progress(already_done, total_all)
    if mode == "staged":
        return _run_staged(
            config, files, log, on_progress, stop_event, on_overall_progress,
            overall_base=already_done, overall_total=total_all,
        )
    return _run_sequential(
        config, files, log, on_progress, stop_event, on_overall_progress,
        overall_base=already_done, overall_total=total_all,
    )


def _run_sequential(
    config: PipelineConfig, files: list[Path], log: LogFn,
    on_progress: Optional[ProgressFn], stop_event: Optional[threading.Event],
    on_overall_progress: Optional[OverallFn] = None,
    overall_base: int = 0,
    overall_total: Optional[int] = None,
) -> list[FileResult]:
    total = len(files)
    if overall_total is None:
        overall_total = total
    results: list[FileResult] = []
    output_root = Path(config.output_dir)
    ident_tmp = Path(tempfile.mkdtemp(prefix="at_ident_"))
    ident_files = files
    if config.skip_existing:
        ident_files = [
            f for f in files
            if not (output_root / f.stem / "text.txt").is_file()
        ]
    try:
        ident_map = _run_identification_parallel(
            config, ident_files, log, on_progress, stop_event, ident_tmp,
        )
    finally:
        shutil.rmtree(ident_tmp, ignore_errors=True)

    for idx, f in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break
        if on_progress:
            on_progress("file", idx - 1, total)
        log(f"\n▶ [{idx}/{total}] {f.name}")

        track_out = output_root / f.stem
        track_out.mkdir(parents=True, exist_ok=True)
        belongs = _artifact_belongs_to(track_out, f)

        if config.skip_existing:
            existing_text = track_out / "text.txt"
            if existing_text.is_file() and belongs:
                try:
                    existing = existing_text.read_text(encoding="utf-8")
                except OSError:
                    existing = None
                log("  ⊘ text.txt уже есть — пропуск")
                if on_progress:
                    on_progress("file", idx, total)
                if on_overall_progress:
                    on_overall_progress(
                        min(overall_base + idx, overall_total), overall_total,
                    )
                results.append(FileResult(f, existing, track_out, None, "existing"))
                continue

        tmp_root = Path(tempfile.mkdtemp(prefix=f"at_{f.stem}_"))

        text: Optional[str] = None
        source: Optional[str] = None
        error: Optional[str] = None
        metadata_dict: Optional[dict] = None
        appended = False
        was_cancelled = False

        try:
            ident = ident_map.get(f) or IdentifyResult(metadata=TrackMetadata())

            if not ident.needs_asr:
                text = ident.text
                source = ident.source
                metadata_dict = _build_metadata_dict(
                    ident.metadata, ident.shazam, source, input_path=f,
                )
                if text:
                    (tmp_root / "text.txt").write_text(text, encoding="utf-8")
                if metadata_dict:
                    _write_metadata_json(tmp_root / "metadata.json", metadata_dict)
                _persist_artifacts(
                    tmp_root, track_out, config,
                    has_text=text is not None,
                    has_metadata=bool(metadata_dict),
                    log=log,
                )
                _save_unrecognized(
                    config, f, text, source,
                    metadata=ident.metadata,
                )
                log(f"  ✓ {f.name} [{source}]")
                results.append(FileResult(f, text, track_out, None, source, metadata_dict))
                appended = True
                continue

            audio_path: Path = f
            skip_convert = False
            skip_separate = False
            if config.resume_partial and belongs:
                vocals = track_out / "vocals.wav"
                convert = track_out / "convert.wav"
                if vocals.is_file():
                    audio_path = vocals
                    skip_convert = True
                    skip_separate = True
                    log("  ↻ используем готовый vocals.wav")
                elif convert.is_file():
                    audio_path = convert
                    skip_convert = True
                    log("  ↻ используем готовый convert.wav")

            if config.convert.enabled and not skip_convert:
                _check_cancel(stop_event)
                log("  [1] Конвертация в WAV...")
                target = tmp_root / "convert.wav"
                if config.overwrite or not target.exists():
                    convert_to_wav(
                        f, target, sample_rate=config.convert.sample_rate,
                        channels=config.convert.channels,
                        ffmpeg_path=config.convert.ffmpeg_path,
                    )
                audio_path = target
                log("  ✓ convert.wav")

            if config.separate.enabled and not skip_separate:
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
                    audio_path, engine=config.transcribe.engine,
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
                if (config.lyrics.enabled and config.lyrics.try_after_asr):
                    matched = _try_post_asr_match(text, config, log)
                    if matched:
                        text = matched
                        source = f"{source}+lrclib"
                (tmp_root / "text.txt").write_text(text, encoding="utf-8")

            metadata_dict = _build_metadata_dict(
                ident.metadata, ident.shazam, source, input_path=f,
            )
            if metadata_dict:
                _write_metadata_json(tmp_root / "metadata.json", metadata_dict)

            _persist_artifacts(
                tmp_root, track_out, config,
                has_text=text is not None,
                has_metadata=bool(metadata_dict),
                log=log,
            )
            _save_unrecognized(
                config, f, text, source,
                metadata=ident.metadata,
            )

        except Cancelled:
            log("⏹ Отменено")
            was_cancelled = True
            break
        except CudaOOMError as e:
            error = f"CudaOOMError: {e}"
            source = "oom_skipped"
            log(f"  ⊘ {f.name}: OOM после 3 попыток — пропущен, идём дальше")
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            log(f"  ✗ Ошибка: {error}")
        finally:
            # Even on error — preserve vocals if the user asked for it.
            if error is not None and config.separate.save_vocals_copy:
                vocal_src = tmp_root / "vocals.wav"
                if vocal_src.exists():
                    try:
                        dst = track_out / "vocals.wav"
                        if dst.exists():
                            dst.unlink()
                        shutil.move(str(vocal_src), str(dst))
                        log(f"  ✓ {f.name} (vocals сохранён, несмотря на ошибку)")
                    except OSError:
                        pass
            shutil.rmtree(tmp_root, ignore_errors=True)
            if was_cancelled:
                # The file was not processed to completion — do not count it
                # in the overall bar and do not report a fake "success" row.
                pass
            else:
                if not appended:
                    results.append(FileResult(
                        f, text, track_out if not error else None, error, source, metadata_dict,
                    ))
                if on_progress:
                    on_progress("file", idx, total)
                if on_overall_progress:
                    on_overall_progress(
                        min(overall_base + idx, overall_total), overall_total,
                    )

    _log_summary(results, log)
    return results


def _run_staged(
    config: PipelineConfig, files: list[Path], log: LogFn,
    on_progress: Optional[ProgressFn], stop_event: Optional[threading.Event],
    on_overall_progress: Optional[OverallFn] = None,
    overall_base: int = 0,
    overall_total: Optional[int] = None,
) -> list[FileResult]:
    """Chunked staged mode: process `chunk_size` files, free memory, repeat."""
    output_root = Path(config.output_dir)
    total = len(files)
    if overall_total is None:
        overall_total = overall_base + total
    chunk_size = max(1, int(config.chunk_size or total))

    if chunk_size >= total:
        log(f"Один батч: {total} файлов (chunk_size={chunk_size})")
        results = _run_staged_batch(
            config, files, output_root, log, on_progress, stop_event,
            overall_offset=overall_base, overall_total=overall_total,
            on_overall_progress=on_overall_progress,
            overall_counter=[overall_base],
        )
        _free_memory()
        _log_summary(results, log)
        return results

    n_batches = (total + chunk_size - 1) // chunk_size
    log(f"Батчей: {n_batches} × {chunk_size} файлов (всего {total})")

    all_results: list[FileResult] = []
    overall_counter = [overall_base]
    for b in range(n_batches):
        if stop_event is not None and stop_event.is_set():
            log("⏹ Отменено")
            break
        start = b * chunk_size
        end = min(start + chunk_size, total)
        chunk = files[start:end]
        log(f"\n╔═══════════════════════════════════════════════════════╗")
        log(f"║  Батч {b+1}/{n_batches}: файлы {start+1}–{end} из {total}")
        log(f"╚═══════════════════════════════════════════════════════╝")

        try:
            batch_results = _run_staged_batch(
                config, chunk, output_root, log, on_progress, stop_event,
                overall_offset=overall_base + start, overall_total=overall_total,
                on_overall_progress=on_overall_progress,
                overall_counter=overall_counter,
            )
            all_results.extend(batch_results)
        except Exception as e:  # noqa: BLE001
            log(f"  ✗ Батч {b+1} упал: {type(e).__name__}: {e}")

        _free_memory()
        log(f"  [batch] память очищена")

    _log_summary(all_results, log)
    return all_results


def _run_staged_batch(
    config: PipelineConfig,
    batch_files: list[Path],
    output_root: Path,
    log: LogFn,
    on_progress: Optional[ProgressFn],
    stop_event: Optional[threading.Event],
    overall_offset: int = 0,
    overall_total: Optional[int] = None,
    on_overall_progress: Optional[OverallFn] = None,
    overall_counter: Optional[list[int]] = None,
) -> list[FileResult]:
    """One chunk of the staged pipeline. All temps are tracked and removed
    in the `finally` block at the end of this call."""
    if overall_total is None:
        overall_total = len(batch_files)

    if overall_counter is None:
        overall_counter = [overall_offset]

    def mark_overall(complete: bool) -> None:
        if not complete:
            return
        overall_counter[0] += 1
        if on_overall_progress:
            on_overall_progress(
                min(overall_counter[0], overall_total),
                overall_total,
            )

    def is_complete(s: dict) -> bool:
        return bool(
            s["resolved"]
            or s["error"] is not None
            or s["text"] is not None
            or not config.transcribe.enabled
        )

    def progress(stage: str, done: int, total: int) -> None:
        """Report progress for the current stage only.

        The GUI bar is reset for every stage, so the counters never have to be
        mixed across stages (identify sees all files, transcribe only the ones
        that actually need ASR, etc.).
        """
        if on_progress:
            on_progress(stage, done, total)

    temp_registry: list[Path] = []

    def _mkdtemp(prefix: str) -> Path:
        p = Path(tempfile.mkdtemp(prefix=prefix))
        temp_registry.append(p)
        return p

    def active() -> list[Path]:
        return [f for f in batch_files if state[f]["error"] is None and not state[f]["resolved"]]

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    # Seed the state from artifacts already present in output_root/<stem>.
    # This is what makes "Resume" work: a file that already has text.txt is
    # resolved, a file with vocals.wav skips convert+separate, and a file with
    # only convert.wav skips just the conversion step.
    state: dict[Path, dict] = {}
    skipped_existing = 0
    resumed_partial = 0
    mismatched_folders = 0
    for f in batch_files:
        track_out = output_root / f.stem
        audio_path: Path = f
        skip_convert = False
        skip_separate = False
        text: Optional[str] = None
        source: Optional[str] = None
        resolved = False

        belongs = _artifact_belongs_to(track_out, f)
        if not belongs:
            has_artifacts = any(
                (track_out / name).exists()
                for name in ("text.txt", "vocals.wav", "convert.wav")
            )
            if has_artifacts:
                mismatched_folders += 1
                log(f"  ⚠ {f.name}: папка {track_out.name} относится к другому треку — обрабатываю заново")

        text_file = track_out / "text.txt"
        if config.skip_existing and belongs and text_file.is_file():
            resolved = True
            skipped_existing += 1
            source = "existing"
            try:
                text = text_file.read_text(encoding="utf-8")
            except OSError:
                text = None
        elif config.resume_partial and belongs:
            vocals = track_out / "vocals.wav"
            convert = track_out / "convert.wav"
            if vocals.is_file():
                audio_path = vocals
                skip_convert = True
                skip_separate = True
                resumed_partial += 1
            elif convert.is_file():
                audio_path = convert
                skip_convert = True
                resumed_partial += 1

        state[f] = {
            "audio_path": audio_path, "ident": None, "text": text,
            "source": source, "metadata_dict": None, "error": None,
            "resolved": resolved, "skip_convert": skip_convert,
            "skip_separate": skip_separate,
        }

    if skipped_existing:
        log(f"  ⊘ Пропущено готовых (text.txt): {skipped_existing}")
    if resumed_partial:
        log(f"  ↻ Подхвачено промежуточных (vocals/convert): {resumed_partial}")
    if mismatched_folders:
        log(f"  ⚠ Папок с чужим path: {mismatched_folders} (пересчитаны заново)")
    if on_overall_progress:
        on_overall_progress(min(overall_counter[0], overall_total), overall_total)

    try:
        # --- Stage 0: identification ---
        if config.shazam.enabled or config.lyrics.enabled:
            to_identify = [f for f in batch_files if not state[f]["resolved"]]
            if to_identify:
                log(f"\n══ Этап 0 · Идентификация ({len(to_identify)} файлов) ══")
                ident_tmp = _mkdtemp("at_ident_")
                try:
                    ident_map = _run_identification_parallel(
                        config, to_identify, log, progress, stop_event, ident_tmp,
                    )
                finally:
                    shutil.rmtree(ident_tmp, ignore_errors=True)
                    try:
                        temp_registry.remove(ident_tmp)
                    except ValueError:
                        pass

                for f in to_identify:
                    ident = ident_map.get(f)
                    if ident is None:
                        state[f]["ident"] = IdentifyResult(metadata=TrackMetadata())
                        log(f"  → {f.name} (identification failed)")
                        continue
                    state[f]["ident"] = ident
                    if not ident.needs_asr:
                        state[f]["resolved"] = True
                        state[f]["text"] = ident.text
                        state[f]["source"] = ident.source
                        state[f]["metadata_dict"] = _build_metadata_dict(
                            ident.metadata, ident.shazam, ident.source,
                            input_path=f,
                        )
                        track_out = output_root / f.stem
                        track_out.mkdir(parents=True, exist_ok=True)
                        if ident.text:
                            (track_out / "text.txt").write_text(ident.text, encoding="utf-8")
                        if state[f]["metadata_dict"]:
                            _write_metadata_json(track_out / "metadata.json", state[f]["metadata_dict"])
                        _save_unrecognized(
                            config, f, ident.text, ident.source,
                            metadata=ident.metadata,
                        )
                        log(f"  ✓ {f.name} [{ident.source}]")
                    else:
                        log(f"  → {f.name} (нужен полный пайплайн)")

        # --- Stage 1: convert ---
        if config.convert.enabled and not cancelled():
            todo = [f for f in active() if not state[f]["skip_convert"]]
            log(f"\n══ Этап 1 · Конвертация ({len(todo)} файлов) ══")
            progress("convert", 0, len(todo))
            _run_convert_parallel(
                config, todo, state, log, progress, stop_event, temp_registry,
            )

        # --- Stage 2: separate ---
        if config.separate.enabled and not cancelled():
            todo = [f for f in active() if not state[f]["skip_separate"]]
            log(f"\n══ Этап 2 · Отделение вокала ({len(todo)} файлов) ══")
            if todo:
                def separate_single(f: Path) -> None:
                    try:
                        src = state[f]["audio_path"]
                        # Never write next to the original input: with
                        # convert.enabled=False src == f, so every track would
                        # overwrite input_dir/vocals.wav. Only reuse the track
                        # output folder when the source already lives there.
                        track_dir = output_root / f.stem
                        if src.parent == track_dir:
                            # Already in the persistent track folder (resume).
                            target = src.parent / "vocals.wav"
                        elif src.parent == f.parent:
                            # src is the original input file (convert disabled).
                            # Never write next to it: all tracks share input_dir
                            # and would overwrite input_dir/vocals.wav.
                            target = _mkdtemp("at_sep_") / "vocals.wav"
                        else:
                            # Freshly converted file in its own unique temp dir.
                            target = src.parent / "vocals.wav"
                        with tempfile.TemporaryDirectory() as tmp:
                            raw = separate_vocals(
                                src, Path(tmp), model=config.separate.model,
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

                # Resumed convert.wav files already live in output_root/<stem>,
                # while freshly converted files live in unique temp dirs.
                # Original inputs (convert disabled) must also go through the
                # one-by-one path: batching them shares input_dir and can
                # collide on vocals.wav.
                # unblend's `{parent}` batch layout would mix those roots, so
                # separate resumed/original files one by one and batch the rest.
                resumed = [
                    f for f in todo
                    if state[f]["audio_path"].parent == output_root / f.stem
                    or state[f]["audio_path"].parent == f.parent
                ]
                resumed_set = set(resumed)
                batchable = [f for f in todo if f not in resumed_set]
                done = 0
                progress("separate", 0, len(todo))

                if len(batchable) == 1:
                    separate_single(batchable[0])
                    done = 1
                    progress("separate", done, len(todo))
                elif batchable:
                    input_wavs = [state[f]["audio_path"] for f in batchable]
                    common_root = _common_parent(input_wavs)
                    log(f"  Батч: {len(batchable)} файлов, один вызов unblend")
                    try:
                        mapping = separate_vocals_batch(
                            input_wavs, common_root,
                            model=config.separate.model,
                            isolate_stem=config.separate.isolate_stem,
                            unblend_path=config.separate.unblend_path,
                            custom_kernels=config.separate.custom_kernels,
                            progress_callback=lambda p: progress(
                                "separate", p, len(todo),
                            ),
                        )
                        for f in batchable:
                            src = state[f]["audio_path"]
                            vocal = mapping.get(src)
                            if vocal is None:
                                state[f]["error"] = "batch produced no output"
                                log(f"  ✗ {f.name}: нет результата в батче")
                                continue
                            state[f]["audio_path"] = vocal
                            log(f"  ✓ {f.name}")
                        done = len(batchable)
                        progress("separate", done, len(todo))
                    except Exception as e:  # noqa: BLE001
                        log(f"  ✗ Батч упал: {e}")
                        log("  ↻ Fallback: по одному...")
                        for i, f in enumerate(batchable, 1):
                            if cancelled():
                                break
                            progress("separate", i - 1, len(todo))
                            separate_single(f)
                            done = i
                            progress("separate", i, len(todo))

                for f in resumed:
                    if cancelled():
                        break
                    done += 1
                    progress("separate", done, len(todo))
                    separate_single(f)
            else:
                progress("separate", 0, 0)

        # --- Stage 3: transcribe ---
        if config.transcribe.enabled and not cancelled():
            todo = active()
            log(f"\n══ Этап 3 · Транскрибация ({len(todo)} файлов) ══")
            progress("transcribe", 0, len(todo))
            for i, f in enumerate(todo, 1):
                if cancelled():
                    log("⏹ Отменено")
                    break
                try:
                    text = transcribe(
                        state[f]["audio_path"], engine=config.transcribe.engine,
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
                except CudaOOMError as e:
                    state[f]["error"] = f"CudaOOMError: {e}"
                    state[f]["source"] = "oom_skipped"
                    log(f"  ⊘ {f.name}: OOM — пропущен")
                except Exception as e:  # noqa: BLE001
                    state[f]["error"] = f"{type(e).__name__}: {e}"
                    log(f"  ✗ {f.name}: {e}")
                finally:
                    progress("transcribe", i, len(todo))

        # --- Stage 4: finalize ---
        # NOTE: runs even if cancelled — we don't want to lose texts that
        # were already transcribed before the user hit "Stop". Only the
        # network call to LRCLIB (fuzzy match) is skipped when cancelled.
        log("\n══ Финализация ══")
        for f in batch_files:
            s = state[f]

            # Errored file: no text, but if separate succeeded we may still
            # have a vocals.wav worth keeping (user asked for save_vocals_copy).
            if s["error"] is not None:
                if config.separate.save_vocals_copy:
                    src_audio = s["audio_path"]
                    if isinstance(src_audio, Path):
                        vocal = src_audio.parent / "vocals.wav"
                        if vocal.exists():
                            track_out = output_root / f.stem
                            track_out.mkdir(parents=True, exist_ok=True)
                            dst = track_out / "vocals.wav"
                            if dst.exists():
                                dst.unlink()
                            try:
                                shutil.move(str(vocal), str(dst))
                                log(f"  ✓ {f.name} (vocals сохранён, несмотря на ошибку)")
                            except OSError:
                                pass
                mark_overall(True)
                continue

            if s["resolved"]:
                mark_overall(True)
                continue

            text = s["text"]
            source = s["source"]
            ident = s["ident"]

            # LRCLIB fuzzy-match is the only slow/network step here —
            # skip it on cancel, but still save the text we already have.
            # This is a *text* search, not a duplicate author/title search.
            if (text and config.lyrics.enabled and config.lyrics.try_after_asr
                    and not cancelled()):
                matched = _try_post_asr_match(text, config, log)
                if matched:
                    text = matched
                    source = f"{source}+lrclib"
                    s["text"] = text
                    s["source"] = source

            metadata_dict = _build_metadata_dict(
                ident.metadata if ident else TrackMetadata(),
                ident.shazam if ident else None,
                source,
                input_path=f,
            )
            s["metadata_dict"] = metadata_dict

            track_out = output_root / f.stem
            track_out.mkdir(parents=True, exist_ok=True)

            audio_work = s["audio_path"]
            if isinstance(audio_work, Path) and audio_work != f:
                work_dir = audio_work.parent
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
            else:
                if text:
                    (track_out / "text.txt").write_text(text, encoding="utf-8")
                if metadata_dict:
                    _write_metadata_json(track_out / "metadata.json", metadata_dict)

            _save_unrecognized(
                config, f, text, source,
                metadata=(ident.metadata if ident else None),
            )
            log(f"  ✓ {f.name}")
            mark_overall(is_complete(s))

    finally:
        # Clean every temp dir created in this batch, regardless of success/error
        for p in temp_registry:
            try:
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
        log(f"  [batch] удалено временных папок: {len(temp_registry)}")

    return [
        FileResult(
            f, state[f]["text"],
            output_root / f.stem if state[f]["error"] is None else None,
            state[f]["error"], state[f]["source"], state[f]["metadata_dict"],
        )
        for f in batch_files
    ]


def _log_summary(results: list[FileResult], log: LogFn) -> None:
    from collections import Counter
    counts = Counter(r.source or ("error" if r.error else "no_result") for r in results)
    log("\n══ Итог ══")
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        log(f"  {src}: {n}")
