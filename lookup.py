"""
Song lookups for Chordy.

- identify_audio: who's playing? Uses AudD (optional, needs AUDD_API_TOKEN).
- describe_link:  what is this link? YouTube, Spotify and SoundCloud links give
                  a title and artist through their public oEmbed endpoints; a
                  direct link to an audio file is downloaded and analysed.
"""

from __future__ import annotations

import os
from urllib.parse import quote, urlparse

import aiohttp

AUDD_TOKEN = os.getenv("AUDD_API_TOKEN", "").strip()
MAX_DOWNLOAD = 10 * 1024 * 1024
UA = {"User-Agent": "Chordy/0.1 (music demo on a Synthesia avatar)"}

OEMBED = {
    "youtube.com": "https://www.youtube.com/oembed?format=json&url=",
    "youtu.be": "https://www.youtube.com/oembed?format=json&url=",
    "music.youtube.com": "https://www.youtube.com/oembed?format=json&url=",
    "open.spotify.com": "https://open.spotify.com/oembed?url=",
    "soundcloud.com": "https://soundcloud.com/oembed?format=json&url=",
}


def song_card(result: dict) -> dict:
    art = (result.get("apple_music") or {}).get("artwork") or {}
    artwork = art.get("url", "").replace("{w}", "300").replace("{h}", "300") if art else ""
    if not artwork:
        imgs = ((result.get("spotify") or {}).get("album") or {}).get("images") or []
        artwork = imgs[0]["url"] if imgs else ""
    return {
        "kind": "song",
        "title": result.get("title"),
        "artist": result.get("artist"),
        "album": result.get("album"),
        "year": (result.get("release_date") or "")[:4],
        "link": result.get("song_link"),
        "artwork": artwork,
        "source": "Identified by AudD",
    }


async def identify_audio(wav: bytes) -> dict | None:
    """Return a song card, or None if no match / no token."""
    if not AUDD_TOKEN:
        return None
    form = aiohttp.FormData()
    form.add_field("api_token", AUDD_TOKEN)
    form.add_field("return", "apple_music,spotify")
    form.add_field("file", wav, filename="clip.wav", content_type="audio/wav")
    async with aiohttp.ClientSession(headers=UA) as s:
        async with s.post("https://api.audd.io/", data=form, timeout=aiohttp.ClientTimeout(total=20)) as r:
            data = await r.json(content_type=None)
    result = data.get("result") if data.get("status") == "success" else None
    return song_card(result) if result else None


async def describe_link(url: str) -> dict:
    """Returns {"card": {...}} for a page link, or {"audio": bytes, "name": str} for an audio file."""
    host = (urlparse(url).hostname or "").removeprefix("www.").removeprefix("m.")
    async with aiohttp.ClientSession(headers=UA) as s:
        for domain, endpoint in OEMBED.items():
            if host == domain or host.endswith("." + domain):
                async with s.get(endpoint + quote(url, safe=""), timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        raise ValueError(f"{domain} wouldn't describe that link (status {r.status})")
                    d = await r.json(content_type=None)
                return {"card": {
                    "kind": "link", "title": d.get("title"), "artist": d.get("author_name"),
                    "artwork": d.get("thumbnail_url"), "link": url, "source": d.get("provider_name") or domain,
                }}
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            ctype = r.headers.get("Content-Type", "")
            if r.status != 200:
                raise ValueError(f"the link returned status {r.status}")
            if not (ctype.startswith("audio/") or url.lower().split("?")[0].endswith((".mp3", ".wav", ".ogg", ".flac"))):
                raise ValueError("that link isn't a song page I know (YouTube, Spotify, SoundCloud) or an audio file")
            data = await r.content.read(MAX_DOWNLOAD + 1)
            if len(data) > MAX_DOWNLOAD:
                raise ValueError("that audio file is over 10 MB")
            return {"audio": data, "name": os.path.basename(urlparse(url).path) or "audio"}
