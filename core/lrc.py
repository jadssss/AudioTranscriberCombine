"""LRC generation, parsing, embedding, and reading embedded lyrics.

Reading embedded lyrics checks all tag locations for a given audio format:
  - MP3       -> ID3 USLT / SYLT
  - FLAC/OGG  -> LYRICS, UNSYNCEDLYRICS (Vorbis comments)
  - M4A/MP4   -> `©lyr` atom
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_LRC_LINE_RE = re.compile(r"^\s*\[\d{1,3}:\d{2}(?:\.\d{1,3})?\]")
_LRC_PARSE_RE = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)$")


@dataclass
class LrcLine:
    start: float
    text: str


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def segments_to_lrc(segments: list[dict]) -> str:
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        lines.append(f"{format_timestamp(start)}{text}")
    return "\n".join(lines) + "\n"


def parse_lrc(content: str) -> list[LrcLine]:
    out: list[LrcLine] = []
    for line in content.splitlines():
        m = _LRC_PARSE_RE.match(line.strip())
        if not m:
            continue
        mm, ss, frac, text = m.groups()
        frac_sec = float(f"0.{frac}") if frac else 0.0
        out.append(LrcLine(start=int(mm) * 60 + int(ss) + frac_sec, text=text.strip()))
    return out


def is_lrc(content: str) -> bool:
    """True if content contains at least one LRC-style timestamp line."""
    if not content:
        return False
    for line in content.splitlines():
        if _LRC_LINE_RE.match(line):
            return True
    return False


# ---------------------------------------------------------------- read embedded

def read_embedded_lyrics(audio_path: Path) -> Optional[str]:
    """Return lyrics stored in the file's tags, or None.

    Order: prefer synced (LRC-like) -> plain. Never raises.
    """
    suffix = audio_path.suffix.lower()
    try:
        if suffix in (".mp3", ".mp2"):
            return _read_mp3(audio_path)
        if suffix in (".flac", ".ogg", ".opus"):
            return _read_vorbis(audio_path)
        if suffix in (".m4a", ".mp4", ".m4b", ".aac"):
            return _read_mp4(audio_path)
    except Exception:  # noqa: BLE001 — never break the pipeline on tag issues
        return None
    return None


def _read_mp3(path: Path) -> Optional[str]:
    from mutagen.id3 import ID3, ID3NoHeaderError
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        return None

    # SYLT = synced lyrics (rare); USLT = unsynced (common, may contain LRC text)
    sylt = tags.getall("SYLT")
    for tag in sylt:
        if tag.text:
            try:
                content = "\n".join(
                    f"{format_timestamp(ms / 1000.0)}{txt}"
                    for txt, ms in tag.text
                )
                if content.strip():
                    return content
            except Exception:  # noqa: BLE001
                continue

    uslt = tags.getall("USLT")
    for tag in uslt:
        if tag.text and str(tag.text).strip():
            return str(tag.text)

    return None


def _read_vorbis(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".flac":
        from mutagen.flac import FLAC as Audio
    elif suffix == ".ogg":
        from mutagen.oggvorbis import OggVorbis as Audio
    else:
        from mutagen.oggopus import OggOpus as Audio
    audio = Audio(str(path))
    for key in ("LYRICS", "UNSYNCEDLYRICS", "LYRIC", "SYNCEDLYRICS"):
        if key in audio:
            values = audio[key]
            if isinstance(values, list) and values:
                text = str(values[0])
            else:
                text = str(values)
            if text.strip():
                return text
    return None


def _read_mp4(path: Path) -> Optional[str]:
    from mutagen.mp4 import MP4
    audio = MP4(str(path))
    for key in ("\xa9lyr", "©lyr"):
        if key in audio:
            values = audio[key]
            if isinstance(values, list) and values:
                text = str(values[0])
            else:
                text = str(values)
            if text.strip():
                return text
    return None


# ---------------------------------------------------------------- embedding

def embed_lrc(audio_path: Path, lrc_content: str) -> None:
    suffix = audio_path.suffix.lower()
    if suffix in (".mp3", ".mp2"):
        _embed_mp3(audio_path, lrc_content)
    elif suffix in (".flac", ".ogg", ".opus"):
        _embed_vorbis(audio_path, lrc_content)
    elif suffix in (".m4a", ".mp4", ".m4b", ".aac"):
        _embed_mp4(audio_path, lrc_content)
    else:
        raise ValueError(f"Unsupported format for LRC embedding: {suffix}")


def _embed_mp3(path: Path, lrc_content: str) -> None:
    from mutagen.id3 import ID3, USLT, ID3NoHeaderError
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("USLT")
    tags.add(USLT(encoding=3, lang="rus", desc="", text=lrc_content))
    tags.save(str(path), v2_version=3)


def _embed_vorbis(path: Path, lrc_content: str) -> None:
    suffix = path.suffix.lower()
    if suffix == ".flac":
        from mutagen.flac import FLAC as Audio
    elif suffix == ".ogg":
        from mutagen.oggvorbis import OggVorbis as Audio
    else:
        from mutagen.oggopus import OggOpus as Audio
    audio = Audio(str(path))
    audio["LYRICS"] = lrc_content
    audio["UNSYNCEDLYRICS"] = lrc_content
    audio.save()


def _embed_mp4(path: Path, lrc_content: str) -> None:
    from mutagen.mp4 import MP4
    audio = MP4(str(path))
    audio["\xa9lyr"] = [lrc_content]
    audio.save()
