"""Speech recognition via WhisperX or GigaAM (onnx-asr).

Also exposes `transcribe_aligned()` for word-level alignment (used by LRC).
Models are cached; aggressive VRAM cleanup happens after each file.
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Optional


# The model caches below are process-wide. They are intentionally never
# cleared between files or batches: reloading WhisperX takes minutes and is
# what made runs look like they reloaded the model every batch. The lock makes
# the "load once, reuse forever" guarantee safe even if ASR is ever called
# from multiple threads.
_MODEL_LOCK = threading.RLock()
_REUSE_LOGGED: set = set()

# CUDA context failures that cannot be recovered by retrying or empty_cache().
# They require restarting the app (or switching to CPU).
_DEAD_CUDA_MARKERS = (
    "invalid device ordinal",
    "cudaerrorinvaliddevice",
    "device-side assert",
    "unspecified launch failure",
)


def _is_dead_cuda(exc: BaseException) -> bool:
    err = str(exc).lower()
    return any(m in err for m in _DEAD_CUDA_MARKERS)


    _WhisperXModels._cache.clear()
    _AlignModels._cache.clear()
    _GigaAMModels._cache.clear()
    _StableTSModels._cache.clear()
    _REUSE_LOGGED.clear()


# Phrases that are strong signals of ASR hallucination (end credits)
_HALLUCINATION_PHRASES = (
    "продолжение следует",
    "спасибо за внимание",
    "спасибо за просмотр",
    "спасибо за подписку",
    "подписывайтесь на канал",
    "ставьте лайки",
    "не забудьте подписаться",
    "субтитры создавал",
    "субтитры сделал",
    "субтитры подготовил",
    "редактор субтитров",
    "корректор",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
)


def _is_hallucination(text: str) -> bool:
    """Return True if the text looks like an ASR hallucination.

    Checks:
      - known call-to-action / promo phrases;
      - repeated-word tic ("Музыка Музыка").
    """
    if not text:
        return False
    t = " ".join(text.lower().split()).strip(" .!?…,;:\"'«»")
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in t:
            return True
    words = t.split()
    if len(words) == 2 and words[0] == words[1]:
        return True
    return False


def _detect_language(model_obj, audio_or_path, fallback: str = "ru") -> str:
    """Detect the spoken language from audio using the ASR model itself.

    Returns a lowercase ISO code or `fallback` on failure.
    """
    try:
        lang, _prob = model_obj.detect_language(audio_or_path)
        if lang:
            return lang.lower()
    except Exception:
        pass
    return fallback


class _StableTSModels:
    _cache: dict = {}

    @classmethod
    def get(cls, model_name: str, device: str, compute_type: Optional[str],
            local_files_only: bool = False) -> Any:
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        key = (model_name, device, compute_type, local_files_only)
        with _MODEL_LOCK:
            if key not in cls._cache:
                import stable_whisper

                print(f"  [ALIGN] loading stable-ts {model_name} on {device} "
                      f"({compute_type}, local_files_only={local_files_only})...")

                def load() -> 'faster_whisper.WhisperModel':
                    return stable_whisper.load_faster_whisper(
                        model_name, device=device, compute_type=compute_type,
                        local_files_only=local_files_only,
                    )

                cls._cache[key] = _retry(load)
                print("  [ALIGN] stable-ts model loaded once — reuse.")
            return cls._cache[key]


def _clear_model_caches() -> None:
    _WhisperXModels._cache.clear()
    _AlignModels._cache.clear()
    _GigaAMModels._cache.clear()
    _StableTSModels._cache.clear()
    _REUSE_LOGGED.clear()


class CudaOOMError(RuntimeError):
    """CUDA OOM, переживший все стратегии понижения нагрузки."""


def _is_cuda_error(exc: BaseException) -> bool:
    err = str(exc).lower()
    return (
        "out of memory" in err
        or "cuda" in err
        or "device ordinal" in err
        or "cublas" in err
        or "cudnn" in err
    )


def _resolve_device(device: str) -> str:
    if device in ("cuda", "cpu"):
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _retry(fn, attempts: int = 3, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last


def _free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — cleanup must never mask the original error
        pass


def _enable_tf32() -> None:
    """Enable TF32 matmul on Ampere+ GPUs (RTX 30/40).

    Called lazily right before a WhisperX model is loaded, so runs that do not
    use ASR never import Torch or create a CUDA context at startup.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:  # noqa: BLE001 — purely an optimization
        pass


def _model_transcribe_with_retry(
    model_obj, audio, language, batch_size, name="<audio>",
):
    """Try progressively lighter settings so one long file can still finish.

    The loaded model is intentionally kept in the cache between attempts:
    reloading it costs minutes and does not help with an OOM caused by a long
    audio file. VAD is not touched here — WhisperX expects it at model-load
    time (``vad_model``/``vad_options``), not per transcription call. A dead
    CUDA context is never retried: caches are dropped and CudaOOMError is
    raised immediately. If even batch=1 fails with a normal OOM, raise
    CudaOOMError so the caller can skip this file and continue with the rest
    of the batch.
    """
    attempts = [batch_size, 1]
    last: Optional[Exception] = None
    for i, bs in enumerate(attempts, 1):
        try:
            return model_obj.transcribe(
                audio, batch_size=bs, language=language,
            )
        except Exception as e:  # noqa: BLE001
            if _is_dead_cuda(e):
                _clear_model_caches()
                raise CudaOOMError(
                    f"CUDA context died on {name}: {e}. "
                    "GPU state is unrecoverable — restart the app or switch to CPU."
                ) from e
            if not _is_cuda_error(e):
                raise
            last = e
            print(f"  [ASR] OOM on {name} (attempt {i}/{len(attempts)}, bs={bs}): {e}")
            _free_vram()
    raise CudaOOMError(
        f"OOM after {len(attempts)} attempts on {name}: {last}"
    ) from last


class _WhisperXModels:
    _cache: dict = {}

    @classmethod
    def get(cls, model_name: str, device: str, compute_type: Optional[str],
            local_files_only: bool = False):
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        key = (model_name, device, compute_type, local_files_only)
        with _MODEL_LOCK:
            if key not in cls._cache:
                if device == "cuda":
                    _enable_tf32()
                import whisperx
                print(f"  [ASR] loading WhisperX {model_name} on {device} "
                      f"({compute_type}, local_files_only={local_files_only})...")

                def load():
                    return whisperx.load_model(
                        model_name, device,
                        compute_type=compute_type,
                        local_files_only=local_files_only,
                    )

                cls._cache[key] = _retry(load)
                print("  [ASR] model loaded once — subsequent files will reuse it.")
            elif ("whisperx",) + key not in _REUSE_LOGGED:
                _REUSE_LOGGED.add(("whisperx",) + key)
                print(f"  [ASR] reusing loaded WhisperX {model_name} ({device}).")
            return cls._cache[key]


class _AlignModels:
    _cache: dict = {}

    @classmethod
    def get(cls, language_code: str, device: str, model_name: Optional[str] = None):
        key = (language_code, device, model_name or "")
        with _MODEL_LOCK:
            if key not in cls._cache:
                import whisperx
                print(f"  [ALIGN] loading align model for {language_code} "
                      f"({model_name or 'default'}) on {device}...")

                def load():
                    kwargs = {"language_code": language_code, "device": device}
                    if model_name:
                        kwargs["model_name"] = model_name
                    return whisperx.load_align_model(**kwargs)

                cls._cache[key] = _retry(load)
                print("  [ALIGN] align model loaded once — reusing it.")
            elif ("align",) + key not in _REUSE_LOGGED:
                _REUSE_LOGGED.add(("align",) + key)
                print(f"  [ALIGN] reusing loaded align model for {language_code}.")
            return cls._cache[key]


class _GigaAMModels:
    _cache: dict = {}

    @classmethod
    def get(cls, provider: str):
        providers = {
            "cuda": ["CUDAExecutionProvider"],
            "cpu": ["CPUExecutionProvider"],
            "auto": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }.get(provider, ["CPUExecutionProvider"])
        key = tuple(providers)
        with _MODEL_LOCK:
            if key not in cls._cache:
                import onnx_asr

                def load():
                    return onnx_asr.load_model("gigaam-v3-ctc", providers=list(providers))

                cls._cache[key] = _retry(load)
            return cls._cache[key]


def preload_transcription_model(
    engine: str = "whisperx",
    model: str = "large-v3",
    device: str = "auto",
    compute_type: Optional[str] = None,
    gigaam_provider: str = "auto",
    local_files_only: bool = False,
) -> None:
    """Load the ASR model once so the per-file loop only reuses it.

    All model caches are process-wide and are never cleared between files,
    batches or pipeline runs. This helper just makes the first load explicit.
    """
    device = _resolve_device(device)
    if engine == "whisperx":
        _WhisperXModels.get(model, device, compute_type, local_files_only)
    elif engine == "gigaam":
        _GigaAMModels.get(gigaam_provider)
    else:
        raise ValueError(f"Unknown ASR engine: {engine!r}")


def transcribe(
    audio_path: Path,
    engine: str = "whisperx",
    model: str = "large-v3",
    language: Optional[str] = "ru",
    device: str = "auto",
    compute_type: Optional[str] = None,
    batch_size: int = 8,
    gigaam_provider: str = "auto",
    local_files_only: bool = False,
    vad_filter: bool = True,
) -> str:
    device = _resolve_device(device)

    if engine == "whisperx":
        return _transcribe_whisperx(
            audio_path, model, language, device, compute_type,
            batch_size, local_files_only, vad_filter,
        )

    if engine == "gigaam":
        return _transcribe_gigaam(audio_path, gigaam_provider)
    raise ValueError(f"Unknown ASR engine: {engine!r}")


def _transcribe_whisperx(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: str,
    compute_type: Optional[str],
    batch_size: int,
    local_files_only: bool,
    vad_filter: bool,
) -> str:
    import whisperx
    model = _WhisperXModels.get(model_name, device, compute_type, local_files_only)
    audio = whisperx.load_audio(str(audio_path))
    result = _model_transcribe_with_retry(
        model, audio, language, batch_size, name=audio_path.name,
    )

    # Filter out hallucinated / spammy segments.
    segments = result.get("segments") or []
    kept = []
    for seg in segments:
        t = (seg.get("text") or "").strip()
        if not t or _is_hallucination(t):
            continue
        if seg.get("avg_logprob", 0.0) < -1.2:
            continue
        kept.append(seg)
    text = " ".join(s["text"].strip() for s in kept).strip()
    if _is_hallucination(text):
        text = ""

    del audio, result
    _free_vram()
    return text


def _transcribe_gigaam(audio_path: Path, provider: str) -> str:
    model = _GigaAMModels.get(provider)
    return model.recognize(str(audio_path), channel="mean").strip()


def transcribe_aligned(
    audio_path: Path,
    model: str = "large-v3",
    language: Optional[str] = "ru",
    device: str = "auto",
    compute_type: Optional[str] = None,
    batch_size: int = 8,
    align_model: Optional[str] = None,
    local_files_only: bool = False,
    vad_filter: bool = True,
) -> tuple[list[dict], str]:
    """Transcribe + word-level align. Returns (aligned_segments, detected_language).

    Each segment has 'start', 'end', 'text', and optionally 'words'.
    Requires whisperx engine.
    """
    import whisperx
    device = _resolve_device(device)
    model_obj = _WhisperXModels.get(model, device, compute_type, local_files_only)
    audio = whisperx.load_audio(str(audio_path))
    result = _model_transcribe_with_retry(
        model_obj, audio, language, batch_size, name=audio_path.name,
    )

    detected = result.get("language") or language or "en"

    try:
        align_model_obj, align_metadata = _AlignModels.get(detected, device, align_model)
        aligned = whisperx.align(
            result["segments"], align_model_obj, align_metadata,
            audio, device, return_char_alignments=False,
        )
        segments = aligned["segments"]
    except Exception as e:
        print(f"  [ALIGN] failed: {e}. Falling back to raw WhisperX segments.")
        segments = result["segments"]

    # Filter out hallucinated segments.
    segments = [s for s in segments if not _is_hallucination(s.get("text", ""))]

    del audio, result
    _free_vram()
    return segments, detected


def align_text_to_audio(
    audio_path: Path,
    text: str,
    language: Optional[str] = "ru",
    model: str = "large-v3",
    device: str = "auto",
    compute_type: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple[list[dict], str]:
    """Forced alignment of EXISTING text to audio. No ASR.

    When `language` is None, the language is auto-detected from the audio
    using the alignment model itself (which is a faster-whisper model and
    exposes `detect_language`).
    """
    import stable_whisper as _sw
    device = _resolve_device(device)
    model_obj = _StableTSModels.get(model, device, compute_type, local_files_only)

    if language:
        lang = language
    else:
        lang = _detect_language(model_obj, str(audio_path), fallback="ru")

    try:
        result = model_obj.align(str(audio_path), text, language=lang)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"align failed on {audio_path.name}: {e}") from e

    segments = [
        {"start": float(seg.start), "end": float(seg.end),
         "text": (seg.text or "").strip()}
        for seg in result.segments
        if (seg.text or "").strip()
    ]
    return segments, lang
