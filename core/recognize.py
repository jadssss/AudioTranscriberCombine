"""Audio fingerprinting via Shazam (unofficial, no API key needed).

Wraps shazamlite. If Shazam can't identify the track — return None.
Never raises on 'not found', only on missing dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ShazamResult:
    title: str
    artist: str
    album: Optional[str] = None
    year: Optional[str] = None


def recognize(audio_wav: Path) -> Optional[ShazamResult]:
    """Run Shazam fingerprinting on a WAV file. Returns None if not found."""
    try:
        from shazamlite import Shazam
    except ImportError as e:
        raise RuntimeError(
            "shazamlite не установлен. Установите: pip install shazamlite numpy soundfile"
        ) from e

    shazam = Shazam()
    try:
        track = shazam.recognize(str(audio_wav))
    except Exception:
        return None

    if track is None:
        return None

    title = getattr(track, "title", None) or getattr(track, "track", None)
    artist = getattr(track, "artist", None) or getattr(track, "subtitle", None)
    album = getattr(track, "album", None)
    year = getattr(track, "year", None) or getattr(track, "released", None)

    if not title or not artist:
        return None

    return ShazamResult(
        title=str(title),
        artist=str(artist),
        album=str(album) if album else None,
        year=str(year) if year else None,
    )
