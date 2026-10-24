"""Speech recognition via WhisperX or GigaAM (onnx-asr).

Includes aggressive VRAM cleanup to prevent OOM on long batches,
and automatic model reload if the CUDA context gets corrupted.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional


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
    device = _resolve_device(device)

    if engine == "whisperx":
        try:
            return _transcribe_whisperx(
                audio_path, model, language, device, compute_type,
                batch_size, local_files_only, vad_filter,
            )
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or "cuda" in err or "device ordinal" in err:
                print(f"  [ASR] CUDA error: {e}. Clearing model cache and retrying...")
                _WhisperXModels._cache.clear()
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                return _transcribe_whisperx(
                    audio_path, model, language, device, compute_type,
                    batch_size, local_files_only, vad_filter,
                )
            raise

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
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )
    
    text = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
    
    # Free VRAM aggressively to prevent fragmentation/OOM on long batches
    del audio
    del result
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    return text


def _transcribe_gigaam(audio_path: Path, provider: str) -> str:
    model = _GigaAMModels.get(provider)
    return model.recognize(str(audio_path), channel="mean").strip()
