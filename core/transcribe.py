"""Speech recognition via WhisperX or GigaAM (onnx-asr).

Models are cached per (name, device, compute_type) so loading happens once
per process, not once per file. Loading is retried a few times to survive
transient HuggingFace / SSL hiccups.
"""
from __future__ import annotations

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
        except Exception as e:  # noqa: BLE001 — HF download / SSL are broad
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last


class _WhisperXModels:
    _cache: dict = {}

    @classmethod
    def get(cls, model_name: str, device: str, compute_type: Optional[str]):
        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        key = (model_name, device, compute_type)
        if key not in cls._cache:
            import whisperx

            def load():
                return whisperx.load_model(model_name, device, compute_type=compute_type)

            cls._cache[key] = _retry(load)
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
) -> str:
    device = _resolve_device(device)
    if engine == "whisperx":
        return _transcribe_whisperx(audio_path, model, language, device, compute_type, batch_size)
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
) -> str:
    import whisperx
    model = _WhisperXModels.get(model_name, device, compute_type)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    return " ".join(seg["text"].strip() for seg in result["segments"]).strip()


def _transcribe_gigaam(audio_path: Path, provider: str) -> str:
    model = _GigaAMModels.get(provider)
    return model.recognize(str(audio_path), channel="mean").strip()
