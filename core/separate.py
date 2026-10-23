"""Vocal separation via unblend (Mel-Band RoFormer)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def separate_vocals(
    input_wav: Path,
    work_dir: Path,
    model: str = "bs_roformer_anvuew",
    isolate_stem: str = "vocals",
    unblend_path: str = "unblend",
    custom_kernels: bool = False,
) -> Path:
    """Separate the isolated stem from input_wav. Returns path to the stem WAV."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = work_dir / "{track}" / "{stem}.wav"
    cmd = [
        unblend_path, "separate", str(input_wav),
        "-m", model,
        "-o", str(output_pattern),
        "--isolate-stem", isolate_stem,
    ]
    env = os.environ.copy()
    env["UNBLEND_CUSTOM_KERNELS"] = "1" if custom_kernels else "0"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"unblend failed for {input_wav.name}:\n{result.stderr.strip()[:800]}"
        )

    candidates = sorted(work_dir.rglob(f"{isolate_stem}.wav"))
    if not candidates:
        raise RuntimeError(
            f"unblend finished but stem '{isolate_stem}.wav' was not found in {work_dir}"
        )
    return candidates[0]
