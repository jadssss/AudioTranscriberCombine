"""Vocal separation via unblend (Mel-Band RoFormer).

Two entry points:

- separate_vocals()       single file, subprocess per call (sequential mode)
- separate_vocals_batch() many files in ONE subprocess, model loaded once
                          (staged mode — big speedup on batches)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _build_env(custom_kernels: bool) -> dict:
    env = os.environ.copy()
    env["UNBLEND_CUSTOM_KERNELS"] = "1" if custom_kernels else "0"
    return env


def _check_result(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"unblend failed ({what}):\n{result.stderr.strip()[:1200]}"
        )


def separate_vocals(
    input_wav: Path,
    work_dir: Path,
    model: str = "bs_roformer_anvuew",
    isolate_stem: str = "vocals",
    unblend_path: str = "unblend",
    custom_kernels: bool = False,
    device: str = "cuda",
) -> Path:
    """Separate one file. Returns path to the isolated stem WAV."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = work_dir / "{track}" / "{stem}.wav"
    cmd = [
        unblend_path, "separate", str(input_wav),
        "-m", model,
        "-o", str(output_pattern),
        "--isolate-stem", isolate_stem,
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_build_env(custom_kernels))
    _check_result(result, input_wav.name)

    candidates = sorted(work_dir.rglob(f"{isolate_stem}.wav"))
    if not candidates:
        raise RuntimeError(
            f"unblend finished but stem '{isolate_stem}.wav' was not found in {work_dir}"
        )
    return candidates[0]


def separate_vocals_batch(
    input_wavs: list[Path],
    work_root: Path,
    model: str = "bs_roformer_anvuew",
    isolate_stem: str = "vocals",
    unblend_path: str = "unblend",
    custom_kernels: bool = False,
    device: str = "cuda",
) -> dict[Path, Path]:
    """Separate many files in one unblend invocation.

    Input WAVs are expected to live in distinct directories
    (e.g. <work_root>/<stem>/convert.wav). The output pattern uses {parent}
    so each result lands next to its input as <parent>/vocals.wav.

    Returns: {input_path: vocal_path}
    """
    if not input_wavs:
        return {}

    work_root.mkdir(parents=True, exist_ok=True)
    output_pattern = work_root / "{parent}" / "{stem}.wav"

    cmd = [
        unblend_path, "separate",
        *[str(p) for p in input_wavs],
        "-m", model,
        "-o", str(output_pattern),
        "--isolate-stem", isolate_stem,
        "--device", device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_build_env(custom_kernels))
    _check_result(result, f"{len(input_wavs)} files")

    mapping: dict[Path, Path] = {}
    missing: list[Path] = []
    for src in input_wavs:
        candidate = src.parent / f"{isolate_stem}.wav"
        if candidate.exists():
            mapping[src] = candidate
        else:
            missing.append(src)

    if missing:
        # Fallback: search anywhere under work_root
        found = sorted(work_root.rglob(f"{isolate_stem}.wav"))
        if len(found) == len(input_wavs):
            for src, dst in zip(input_wavs, found):
                mapping[src] = dst
        else:
            names = ", ".join(p.name for p in missing[:3])
            raise RuntimeError(
                f"unblend finished but outputs missing for {len(missing)} file(s): {names}"
            )

    return mapping
