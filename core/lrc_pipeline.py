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
    FileResult, _artifact_belongs_to, _output_dir_for, _save_unrecognized,
    collect_audio_files,
)
from .transcribe import (
    CudaOOMError, align_text_to_audio, transcribe, transcribe_aligned,
)


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]  # stage, current, total
OverallFn = Callable[[int, int], None]        # completed_files, total_files


def _pick_audio_source(
    f: Path, input_root: Path, output_dir: Path, config: PipelineConfig,
    work_dir: Path, log: LogFn,
) -> Path:
    mode = (config.lrc.audio_source or "auto").lower()
    track_out = _output_dir_for(f, input_root, output_dir)
    vocals = track_out / "vocals.wav"
    convert = track_out / "convert.wav"
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

    # sanity: если ASR и конвертация/отделение выключены — alignment имеет смысл
    # только для треков, у которых УЖЕ есть text.txt + vocals/convert.wav.
    # Иначе почти всё упадёт. Предупредить заранее.
    if not config.transcribe.enabled and not config.convert.enabled and not config.separate.enabled:
        log("  ⚠ ASR, конвертация и отделение выключены. "
            "LRC будет построен только для треков, где уже есть text.txt + vocals.wav "
            "(или convert.wav). Остальные упадут с ошибкой.")

    # `None` means "auto" — the language will be auto-detected during ASR
    # (transcribe_aligned) or by the alignment model (align_text_to_audio).
    lang = config.lrc.language
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

        track_out = _output_dir_for(f, input_dir, output_dir)
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

        track_out = _output_dir_for(f, input_dir, output_dir)
        track_out.mkdir(parents=True, exist_ok=True)
        lrc_path = track_out / "text.lrc"
        text_file = track_out / "text.txt"
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
            # 0) embedded LRC from tags — бесплатно
            embedded = read_embedded_lyrics(f) if config.lyrics.use_embedded_lyrics else None
            if embedded and is_lrc(embedded):
                lrc_path.write_text(embedded, encoding="utf-8")
                log("  ✓ text.lrc (взят из тегов аудио)")
                results.append(FileResult(f, None, track_out, None, "embedded+lrc"))
                continue

            # 1) готовый текст из text.txt
            text: Optional[str] = None
            if text_file.is_file():
                try:
                    text = text_file.read_text(encoding="utf-8").strip()
                except OSError as e:
                    raise RuntimeError(f"не читается text.txt: {e}") from e
            if not text:
                text = None

            # 1a) text.txt уже LRC → писать его как есть, alignment не нужен
            if text is not None and is_lrc(text):
                lrc_path.write_text(text, encoding="utf-8")
                log("  ✓ text.lrc (text.txt уже содержит LRC-таймкоды)")
                results.append(FileResult(f, None, track_out, None, "text_already_lrc"))
                continue

            # 2) аудио для alignment
            audio_path = _pick_audio_source(
                f, input_dir, output_dir, config, tmp, log,
            )
            if not audio_path.exists():
                # Если convert/separate выключены и vocals/convert.wav отсутствуют —
                # понятная ошибка вместо «файл не найден»
                if not config.convert.enabled and not config.separate.enabled:
                    raise FileNotFoundError(
                        f"для alignment нет аудио: {audio_path}. "
                        "Включите конвертацию/отделение в основном пайплайне, "
                        "либо запустите их один раз, либо положите vocals.wav/convert.wav."
                    )
                raise FileNotFoundError(
                    f"источник аудио не найден: {audio_path}. "
                    f"Источник={config.lrc.audio_source}, "
                    f"convert.enabled={config.convert.enabled}, "
                    f"separate.enabled={config.separate.enabled}"
                )
            log(f"  [align] {audio_path.name}")

            # 3) готовый текст → forced alignment (быстро, без ASR)
            if text is not None:
                log(f"  [align-only] text.txt найден ({len(text)} символов)")
                segments, detected = align_text_to_audio(
                    audio_path,
                    text,
                    language=lang,
                    model=config.transcribe.model,
                    device=config.lrc.device,
                    compute_type=config.transcribe.compute_type,
                    local_files_only=config.transcribe.local_files_only,
                )
            else:
                # 3a) нет text.txt — ASR разрешён только если transcribe.enabled
                if not config.transcribe.enabled:
                    raise FileNotFoundError(
                        "text.txt отсутствует, а транскрибация выключена. "
                        "Включите транскрибацию в основном пайплайне или "
                        "положите text.txt в папку трека вручную."
                    )
                log("  [asr+align] text.txt нет — выполняю ASR заново")
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
                log(f"  ⚠ {f.name}: пустой LRC (нет сегментов)")
                results.append(FileResult(
                    f, None, None,
                    "EmptyResult: alignment returned no text", "lrc+empty",
                ))
                continue
            lrc_path.write_text(lrc_content, encoding="utf-8")
            log(f"  ✓ text.lrc ({len(segments)} строк, язык: {detected})")

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

        track_out = _output_dir_for(f, input_dir, output_dir)
        lrc_path = track_out / "text.lrc"
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
                target = track_out / f.name
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
