"""Speech recognition via WhisperX or GigaAM (onnx-asr).

Also exposes `transcribe_aligned()` for word-level alignment (used by LRC).
Models are cached; aggressive VRAM cleanup happens after each file.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional


# Set permanently once the CUDA context has died (e.g. after an unrecoverable
# OOM). Every later CUDA transcription fails instantly instead of re-loading
# the model forever and hanging the pipeline.
_CUDA_BROKEN = False


def _is_cuda_error(exc: BaseException) -> bool:
    err = str(exc).lower()
    return (
        "out of memory" in err
        or "cuda" in err
        or "device ordinal" in err
        or "cublas" in err
        or "cudnn" in err
    )


def _check_cuda_usable(device: str) -> None:
    if device == "cuda" and _CUDA_BROKEN:
        raise RuntimeError(
            "CUDA context is broken (previous OOM was unrecoverable). "
            "Restart the app or switch device to CPU."
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


class _WhisperXModels:
    _cache: dict = {}

    @classmethod
    def get(cls, model_name: str, device: str, compute_type: Optional[str],
            local_files_only: bool = False):
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        key = (model_name, device, compute_type, local_files_only)
        if key not in cls._cache:
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
            print("  [ASR] model loaded.")
        return cls._cache[key]


class _AlignModels:
    _cache: dict = {}

    @classmethod
    def get(cls, language_code: str, device: str, model_name: Optional[str] = None):
        key = (language_code, device, model_name or "")
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
            print("  [ALIGN] align model loaded.")
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
        if key not in cls._cache:
            import onnx_asr

            def load():
                return onnx_asr.load_model("gigaam-v3-ctc", providers=list(providers))

            cls._cache[key] = _retry(load)
        return cls._cache[key]


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
    global _CUDA_BROKEN
    device = _resolve_device(device)
    _check_cuda_usable(device)

    if engine == "whisperx":
        try:
            return _transcribe_whisperx(
                audio_path, model, language, device, compute_type,
                batch_size, local_files_only, vad_filter,
            )
        except Exception as e:
            if not _is_cuda_error(e):
                raise
            print(f"  [ASR] CUDA error: {e}. Clearing model cache and retrying...")
            _WhisperXModels._cache.clear()
            _free_vram()
            try:
                return _transcribe_whisperx(
                    audio_path, model, language, device, compute_type,
                    batch_size, local_files_only, vad_filter,
                )
            except Exception as retry_error:
                if not _is_cuda_error(retry_error):
                    raise
                _CUDA_BROKEN = True
                print("  [ASR] CUDA context is unrecoverable — "
                      "GPU disabled for the rest of this run.")
                raise RuntimeError(
                    "CUDA context is broken after OOM recovery failed. "
                    "Restart the app or switch device to CPU."
                ) from retry_error

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
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    text = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
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
) -> tuple[list[dict], str]:
    """Transcribe + word-level align. Returns (aligned_segments, detected_language).

    Each segment has 'start', 'end', 'text', and optionally 'words'.
    Requires whisperx engine.
    """
    import whisperx
    device = _resolve_device(device)
    _check_cuda_usable(device)
    model_obj = _WhisperXModels.get(model, device, compute_type, local_files_only)
    audio = whisperx.load_audio(str(audio_path))
    result = model_obj.transcribe(audio, batch_size=batch_size, language=language)

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

    del audio, result
    _free_vram()
    return segments, detected
