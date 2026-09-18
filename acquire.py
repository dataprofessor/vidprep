"""Media acquisition: either a local file upload or a YouTube URL via yt-dlp.

Produces a normalized `AcquiredMedia` object that downstream transcription
and generation modules consume, regardless of source.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import yt_dlp
import imageio_ffmpeg

from config import TMP_DIR, YOUTUBE_COOKIES_FILE

os.makedirs(TMP_DIR, exist_ok=True)

# Make sure pip-installed console-script binaries (e.g. the `deno` JS runtime
# yt-dlp needs to solve YouTube's signature challenges) are on PATH — the
# container runtime doesn't always put the interpreter's own bin/ dir there.
_bin_dir = os.path.dirname(sys.executable)
if _bin_dir and _bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

# AI_TRANSCRIBE-supported containers (video ones can be sent to AI_TRANSCRIBE
# directly with no local audio extraction needed).
_SUPPORTED_VIDEO_EXTS = {"mp4", "mov", "mkv", "webm", "ogv"}
_SUPPORTED_AUDIO_EXTS = {"aac", "flac", "m4a", "mp3", "mp4", "ogg", "wav", "webm"}


@dataclass
class ExistingChapter:
    start_seconds: float
    title: str


@dataclass
class AcquiredMedia:
    source_type: str  # "upload" | "youtube"
    local_media_path: str  # path to the file that will be sent to AI_TRANSCRIBE
    media_kind: str  # "video" | "audio"
    cache_key: str  # stable key for caching transcript/generation results
    title: Optional[str] = None
    existing_description: Optional[str] = None
    existing_chapters: list[ExistingChapter] = field(default_factory=list)
    captions_vtt_path: Optional[str] = None  # set if usable YouTube captions were found
    captions_source: Optional[str] = None  # "manual" | "automatic"
    video_id: Optional[str] = None
    local_preview_path: Optional[str] = None  # small local video file for the UI preview player
    error: Optional[str] = None


class AcquisitionError(Exception):
    """Raised when acquisition fails in a way the UI should surface directly."""


def _base_ydl_opts() -> dict:
    """Common yt-dlp options. Uses a real logged-in session's cookies (if
    present) to get past YouTube's bot-detection on datacenter/cloud IPs
    (e.g. Snowflake SPCS) — this requires a JS runtime (deno, installed as a
    dependency) to solve YouTube's signature challenges for the default web
    client. Note: don't override player_client here — clients like
    'android'/'tv_embedded' explicitly refuse to use cookies, which would
    silently break authenticated access."""
    opts: dict = {}
    if YOUTUBE_COOKIES_FILE and os.path.exists(YOUTUBE_COOKIES_FILE):
        opts["cookiefile"] = YOUTUBE_COOKIES_FILE
    return opts


def _file_hash(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def acquire_from_upload(uploaded_file) -> AcquiredMedia:
    """Save a Streamlit UploadedFile to disk and wrap it as AcquiredMedia."""
    ext = os.path.splitext(uploaded_file.name)[1].lstrip(".").lower()
    if ext not in _SUPPORTED_VIDEO_EXTS and ext not in _SUPPORTED_AUDIO_EXTS:
        raise AcquisitionError(
            f"Unsupported file type '.{ext}'. AI_TRANSCRIBE supports: "
            f"{sorted(_SUPPORTED_VIDEO_EXTS | _SUPPORTED_AUDIO_EXTS)}"
        )

    dest_path = os.path.join(TMP_DIR, uploaded_file.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_hash = _file_hash(dest_path)
    media_kind = "video" if ext in _SUPPORTED_VIDEO_EXTS else "audio"

    return AcquiredMedia(
        source_type="upload",
        local_media_path=dest_path,
        media_kind=media_kind,
        cache_key=f"upload_{file_hash}",
        title=os.path.splitext(uploaded_file.name)[0],
    )


def _extract_youtube_id(url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return match.group(1) if match else None


def _pick_caption_track(info: dict) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """Prefer manual subtitles over auto-generated captions, English first."""
    for source, label in (("subtitles", "manual"), ("automatic_captions", "automatic")):
        tracks = info.get(source) or {}
        if not tracks:
            continue
        for lang in ("en", "en-US", "en-GB"):
            if lang in tracks:
                return tracks[lang], label, lang
        # fall back to first available language track
        first_lang = next(iter(tracks), None)
        if first_lang:
            return tracks[first_lang], label, first_lang
    return None, None, None


def acquire_from_youtube(url: str, prefer_captions: bool = True) -> AcquiredMedia:
    """Fetch metadata (and captions, if usable) then download audio-only media."""
    video_id = _extract_youtube_id(url) or "unknown"

    metadata_opts = {**_base_ydl_opts(), "quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(metadata_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise AcquisitionError(
            f"Could not fetch this YouTube video's info: {e}. "
            "The video may be private, age-restricted, geo-blocked, or removed. "
            "Try uploading the file directly instead."
        ) from e

    title = info.get("title")
    existing_description = info.get("description") or None
    existing_chapters = [
        ExistingChapter(start_seconds=c["start_time"], title=c.get("title", ""))
        for c in (info.get("chapters") or [])
        if c.get("start_time") is not None
    ]

    captions_vtt_path = None
    captions_source = None
    if prefer_captions:
        track, captions_source, lang = _pick_caption_track(info)
        if track:
            caption_opts = {
                **_base_ydl_opts(),
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": captions_source == "manual",
                "writeautomaticsub": captions_source == "automatic",
                "subtitleslangs": [lang],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(TMP_DIR, f"{video_id}.%(ext)s"),
            }
            try:
                with yt_dlp.YoutubeDL(caption_opts) as ydl:
                    ydl.download([url])
                candidate = os.path.join(TMP_DIR, f"{video_id}.{lang}.vtt")
                if os.path.exists(candidate):
                    captions_vtt_path = candidate
            except yt_dlp.utils.DownloadError:
                captions_vtt_path = None  # fall back to audio transcription

    # Always download audio (used either for AI_TRANSCRIBE fallback, or if
    # the caller opts out of caption-based timing entirely).
    audio_template = os.path.join(TMP_DIR, f"{video_id}.%(ext)s")
    audio_opts = {
        **_base_ydl_opts(),
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": audio_template,
    }
    try:
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            local_media_path = ydl.prepare_filename(result)
    except yt_dlp.utils.DownloadError as e:
        raise AcquisitionError(
            f"Could not download audio for this video: {e}. "
            "Try uploading the file directly instead."
        ) from e

    ext = os.path.splitext(local_media_path)[1].lstrip(".").lower()
    if ext not in _SUPPORTED_AUDIO_EXTS:
        raise AcquisitionError(
            f"Downloaded audio format '.{ext}' isn't supported by AI_TRANSCRIBE."
        )

    # Also download a small, low-resolution video+audio stream purely for the
    # UI preview player. Streamlit's CSP blocks embedding third-party iframes
    # (e.g. a YouTube embed), but a local media file served from the app's own
    # origin plays fine, so this is the only way to show a real video preview
    # instead of audio-only. Most videos only have a single-file (progressive)
    # option at 360p+, so to actually get down to a small size we pick a
    # low-res video-only DASH stream and mux it with a small audio-only stream
    # via ffmpeg, falling back to whatever's smallest if that's unavailable.
    local_preview_path = None
    preview_template = os.path.join(TMP_DIR, f"{video_id}_preview.%(ext)s")
    preview_opts = {
        **_base_ydl_opts(),
        "quiet": True,
        "no_warnings": True,
        "format": (
            "bestvideo[height<=240][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=240]+bestaudio/"
            "best[height<=240]/worst[ext=mp4]/worst"
        ),
        "merge_output_format": "mp4",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "outtmpl": preview_template,
    }
    try:
        with yt_dlp.YoutubeDL(preview_opts) as ydl:
            preview_result = ydl.extract_info(url, download=True)
            local_preview_path = ydl.prepare_filename(preview_result)
    except yt_dlp.utils.DownloadError:
        local_preview_path = None  # preview is best-effort; app still works without it

    return AcquiredMedia(
        source_type="youtube",
        local_media_path=local_media_path,
        media_kind="audio",
        cache_key=f"yt_{video_id}",
        title=title,
        existing_description=existing_description,
        existing_chapters=existing_chapters,
        captions_vtt_path=captions_vtt_path,
        captions_source=captions_source,
        video_id=video_id,
        local_preview_path=local_preview_path,
    )
