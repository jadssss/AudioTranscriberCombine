"""Lyrics lookup via LRCLIB (https://lrclib.net) — free, no API key.

Rate limit: LRCLIB allows ~90 req/min with a proper User-Agent,
~45 without. We throttle to 0.4s between requests (150/min theoretical,
well within limits) and always send a UA.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests


DEFAULT_USER_AGENT = "audio-transcriber/1.0"
_API_BASE = "https://lrclib.net/api"

_throttle_lock = threading.Lock()
_last_request_at = 0.0
_MIN_INTERVAL = 0.4  # seconds


@dataclass
class LyricsMatch:
    title: str
    artist: str
    lyrics: str
    duration: Optional[float] = None


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        now = time.monotonic()
        wait = _last_request_at + _MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _get(url: str, params: dict, user_agent: str, timeout: float = 10.0) -> Optional[dict]:
    _throttle()
    try:
        resp = requests.get(
            url, params=params,
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _make_match(item: dict) -> Optional[LyricsMatch]:
    lyrics = item.get("plainLyrics") or item.get("syncedLyrics")
    if not lyrics:
        return None
    return LyricsMatch(
        title=item.get("trackName", ""),
        artist=item.get("artistName", ""),
        lyrics=lyrics.strip(),
        duration=item.get("duration"),
    )


def search_by_identity(
    title: Optional[str],
    artist: Optional[str],
    album: Optional[str] = None,
    duration: Optional[float] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 10.0,
) -> Optional[LyricsMatch]:
    """Search LRCLIB by track metadata. Works even with messy tags."""
    if not title and not artist:
        return None

    query = " ".join(filter(None, [artist or "", title or ""])).strip()
    if not query:
        return None

    data = _get(f"{_API_BASE}/search", {"q": query}, user_agent, timeout)
    if not isinstance(data, list) or not data:
        return None

    for item in data[:5]:
        match = _make_match(item)
        if match:
            return match
    return None


def search_by_text(
    query: str,
    limit: int = 5,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 10.0,
) -> list[LyricsMatch]:
    """Search LRCLIB by plain text (e.g. first words of recognized speech)."""
    if not query or not query.strip():
        return []
    data = _get(f"{_API_BASE}/search", {"q": query.strip()}, user_agent, timeout)
    if not isinstance(data, list):
        return []
    matches: list[LyricsMatch] = []
    for item in data[:limit]:
        match = _make_match(item)
        if match:
            matches.append(match)
    return matches
