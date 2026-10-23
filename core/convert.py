"""Convert any audio file to a WAV with chosen sample rate and channels via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path


def convert_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 44100,
    channels: int = 2,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Run ffmpeg to produce a WAV file. Raises RuntimeError on ffmpeg failure."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path.name}:\n{result.stderr.strip()[:800]}"
        )
