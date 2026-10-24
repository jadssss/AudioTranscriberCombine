"""Read audio file tags via mutagen. Always safe — never raises."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrackMetadata:
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[float] = None
    has_identity: bool = False


def read_metadata(path: Path) -> TrackMetadata:
    """Extract title/artist/album/duration from tags. Never raises."""
    try:
        from mutagen import File
    except ImportError:
        return TrackMetadata()

    try:
        audio = File(str(path), easy=True)
    except Exception:
        return TrackMetadata()

    if audio is None:
        return TrackMetadata()

    def _first(key: str) -> Optional[str]:
        v = audio.get(key)
        if isinstance(v, list):
            return v[0] if v else None
        if isinstance(v, str):
            return v
        return None

    title = _first("title")
    artist = _first("artist")
    album = _first("album")

    duration: Optional[float] = None
    try:
        if audio.info is not None:
            duration = float(audio.info.length)
    except Exception:
        pass

    return TrackMetadata(
        title=title,
        artist=artist,
        album=album,
        duration=duration,
        has_identity=bool(title and artist),
    )
