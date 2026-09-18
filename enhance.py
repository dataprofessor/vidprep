"""Publishing-prep extras: title/category/SEO suggestions, thumbnail frames,
pull quotes, caption file export, and FAQ generation — all layered on top of
the core transcript + chapters/description already produced by generate.py.

Timestamps here follow the same rule as the rest of the app: never guessed by
the model. Quotes are built verbatim from real transcript word spans, and FAQ
items are tied to a real chapter index (resolved to that chapter's real start
time in code), not any timestamp the model invents.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

import imageio_ffmpeg

import cache
from acquire import AcquiredMedia
from config import MIN_CHAPTERS, TMP_DIR
from generate import (
    CHAPTER_SCAN_CHUNK_WORDS,
    CHAPTER_SCAN_OVERLAP_WORDS,
    GeneratedResult,
    _ai_complete_json,
    _build_indexed_text,
    _windows,
)
from transcribe import Transcript

YOUTUBE_CATEGORIES = [
    "Film & Animation",
    "Autos & Vehicles",
    "Music",
    "Pets & Animals",
    "Sports",
    "Travel & Events",
    "Gaming",
    "People & Blogs",
    "Comedy",
    "Entertainment",
    "News & Politics",
    "How-to & Style",
    "Education",
    "Science & Technology",
    "Nonprofits & Activism",
]

_CTA_PHRASES = (
    "subscribe", "like", "comment", "follow", "link", "check out",
    "join", "sign up", "download", "click",
)


# ---------------------------------------------------------------------------
# Titles & SEO
# ---------------------------------------------------------------------------


@dataclass
class TitleSeoResult:
    titles: list[str] = field(default_factory=list)
    category: str = ""
    category_reason: str = ""
    endscreen_seconds: float = 0.0
    endscreen_suggestion: str = ""

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


_TITLE_SEO_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string"},
        "category_reason": {"type": "string"},
        "end_screen_suggestion": {"type": "string"},
    },
    "required": ["titles", "category", "category_reason", "end_screen_suggestion"],
}


def generate_titles_seo(media: AcquiredMedia, result: GeneratedResult, duration: float) -> TitleSeoResult:
    cache_key = f"{media.cache_key}_titles_seo"
    cached = cache.get("titles_seo", cache_key)
    if cached:
        return TitleSeoResult.from_dict(cached)

    chapter_titles = "\n".join(f"- {c.title}" for c in result.chapters)
    categories_list = "\n".join(f"- {c}" for c in YOUTUBE_CATEGORIES)
    prompt = f"""You are preparing a video for publishing on YouTube. Here is the video's
description and chapter list:

Description:
{result.description}

Chapters:
{chapter_titles}

Tasks:
1. Suggest 5-8 candidate video titles. Mix descriptive/SEO-friendly titles with
   a couple punchier options. Each title must be 100 characters or fewer
   (YouTube's title limit). No clickbait that misrepresents the content.
2. Pick the single best-fitting category for this video from EXACTLY this list
   (return the text exactly as written here):
{categories_list}
   Also give a one-sentence reason for that pick.
3. Suggest one short end-screen call-to-action (what to promote/link/ask the
   viewer to do in the last ~20 seconds), based on the content.

Return only the structured JSON.
"""
    raw = _ai_complete_json(prompt, _TITLE_SEO_SCHEMA)

    titles = [t.strip() for t in (raw.get("titles") or []) if t and t.strip()]
    category = (raw.get("category") or "").strip()
    if category not in YOUTUBE_CATEGORIES:
        # Model drifted from the allowed list — fall back to the closest
        # case-insensitive match, else leave unset rather than show junk.
        lowered = {c.lower(): c for c in YOUTUBE_CATEGORIES}
        category = lowered.get(category.lower(), "")

    last_chapter_start = result.chapters[-1].start_seconds if result.chapters else 0.0
    endscreen_seconds = max(last_chapter_start, duration - 20.0)
    endscreen_seconds = max(0.0, min(endscreen_seconds, duration))

    seo_result = TitleSeoResult(
        titles=titles,
        category=category,
        category_reason=(raw.get("category_reason") or "").strip(),
        endscreen_seconds=endscreen_seconds,
        endscreen_suggestion=(raw.get("end_screen_suggestion") or "").strip(),
    )
    cache.set("titles_seo", cache_key, seo_result.to_dict())
    return seo_result


def seo_checklist(result: GeneratedResult, titles: list[str]) -> list[dict]:
    """Pure-Python, no-AI checklist. Recomputed on every render (free)."""
    desc = result.description or ""
    desc_len = len(desc)
    desc_lower = desc.lower()

    keyword_hits = sum(1 for k in result.keywords if k.lower() in desc_lower)
    has_cta = any(phrase in desc_lower for phrase in _CTA_PHRASES)
    has_short_title = any(len(t) <= 100 for t in titles) if titles else False

    return [
        {
            "label": "Description length (150-5000 chars)",
            "passed": 150 <= desc_len <= 5000,
            "detail": f"{desc_len} characters",
        },
        {
            "label": "Keywords used in description (3+)",
            "passed": keyword_hits >= 3,
            "detail": f"{keyword_hits} of {len(result.keywords)} keywords appear in the description",
        },
        {
            "label": "Description has a call-to-action",
            "passed": has_cta,
            "detail": "found a CTA phrase (subscribe/like/comment/...)" if has_cta else "no CTA phrase found",
        },
        {
            "label": "At least one title within YouTube's 100-char limit",
            "passed": has_short_title,
            "detail": "ok" if has_short_title else "no title generated yet, or all are too long",
        },
        {
            "label": f"At least {MIN_CHAPTERS} chapters",
            "passed": len(result.chapters) >= MIN_CHAPTERS,
            "detail": f"{len(result.chapters)} chapters",
        },
    ]


# ---------------------------------------------------------------------------
# Thumbnails & Clips
# ---------------------------------------------------------------------------


@dataclass
class ThumbnailCandidate:
    seconds: float
    image_path: str
    chapter_title: str


def _preview_video_path(media: AcquiredMedia) -> str | None:
    if media.source_type == "youtube":
        return media.local_preview_path
    if media.media_kind == "video":
        return media.local_media_path
    return None


def generate_thumbnails(media: AcquiredMedia, result: GeneratedResult, max_candidates: int = 5) -> list[ThumbnailCandidate]:
    video_path = _preview_video_path(media)
    if not video_path or not os.path.exists(video_path) or not result.chapters:
        return []

    chapters = result.chapters
    if len(chapters) > max_candidates:
        step = len(chapters) / max_candidates
        chapters = [chapters[int(i * step)] for i in range(max_candidates)]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    candidates: list[ThumbnailCandidate] = []
    base = os.path.splitext(os.path.basename(video_path))[0]
    for c in chapters:
        seconds = c.start_seconds + 2.0
        out_path = os.path.join(TMP_DIR, f"{base}_thumb_{int(seconds)}.jpg")
        try:
            subprocess.run(
                [
                    ffmpeg_exe, "-y", "-ss", str(seconds), "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", out_path,
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            candidates.append(ThumbnailCandidate(seconds=seconds, image_path=out_path, chapter_title=c.title))
    return candidates


@dataclass
class Quote:
    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


_QUOTE_SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_word_index": {"type": "number"},
                    "end_word_index": {"type": "number"},
                },
                "required": ["start_word_index", "end_word_index"],
            },
        },
    },
    "required": ["quotes"],
}


def _scan_chunk_for_quotes(words, start_idx: int, end_idx: int) -> list[dict]:
    indexed_text = _build_indexed_text(words, start_idx, end_idx)
    prompt = f"""You are scanning a video transcript for short, quotable moments suitable
for social media teasers or thumbnail text. Each word below is tagged with its
index in the format [INDEX]word — indices range from {start_idx} to {end_idx - 1}.

Transcript excerpt:
{indexed_text}

Task: Pick up to 2 short quotable spans (roughly 5-20 words each) from this
excerpt — punchy, self-contained statements, not generic filler. Return each
as start_word_index and end_word_index (both integers that MUST fall between
{start_idx} and {end_idx - 1} inclusive, and MUST match [INDEX] tags above —
never invent or estimate an index). Return only the structured JSON.
"""
    raw = _ai_complete_json(prompt, _QUOTE_SCAN_SCHEMA)
    return raw.get("quotes") or []


def generate_quotes(media: AcquiredMedia, transcript: Transcript, max_quotes: int = 6) -> list[Quote]:
    cache_key = f"{media.cache_key}_quotes"
    cached = cache.get("quotes", cache_key)
    if cached:
        return [Quote.from_dict(d) for d in cached]

    words = transcript.words
    n = len(words)
    all_spans: list[tuple[int, int]] = []
    for start_idx, end_idx in _windows(words, CHAPTER_SCAN_CHUNK_WORDS, CHAPTER_SCAN_OVERLAP_WORDS):
        for q in _scan_chunk_for_quotes(words, start_idx, end_idx):
            try:
                s, e = int(q["start_word_index"]), int(q["end_word_index"])
            except (KeyError, TypeError, ValueError):
                continue
            s = max(0, min(s, n - 1))
            e = max(s, min(e, n - 1))
            if e - s <= 60:  # sanity cap on quote length
                all_spans.append((s, e))

    all_spans.sort(key=lambda se: se[0])
    if len(all_spans) > max_quotes:
        step = len(all_spans) / max_quotes
        all_spans = [all_spans[int(i * step)] for i in range(max_quotes)]

    quotes = [
        Quote(
            start_seconds=words[s].start,
            end_seconds=words[e].end,
            text=" ".join(w.text for w in words[s : e + 1]),
        )
        for s, e in all_spans
    ]
    cache.set("quotes", cache_key, [q.to_dict() for q in quotes])
    return quotes


# ---------------------------------------------------------------------------
# Captions & FAQ
# ---------------------------------------------------------------------------


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    return _srt_timestamp(seconds).replace(",", ".")


def _build_cues(transcript: Transcript, max_words: int = 10, max_chars: int = 42, max_gap: float = 0.6):
    """Group word-level transcript entries into caption cues."""
    cues: list[tuple[float, float, str]] = []
    current: list = []
    for w in transcript.words:
        if current:
            gap = w.start - current[-1].end
            joined = " ".join(x.text for x in current + [w])
            if gap > max_gap or len(current) >= max_words or len(joined) > max_chars:
                cues.append((current[0].start, current[-1].end, " ".join(x.text for x in current)))
                current = []
        current.append(w)
    if current:
        cues.append((current[0].start, current[-1].end, " ".join(x.text for x in current)))
    return cues


def build_srt(transcript: Transcript) -> str:
    cues = _build_cues(transcript)
    lines = []
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def build_vtt(transcript: Transcript) -> str:
    cues = _build_cues(transcript)
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        lines.append(f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


@dataclass
class FaqItem:
    question: str
    answer: str
    start_seconds: float
    chapter_title: str

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


_FAQ_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_index": {"type": "number"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["chapter_index", "question", "answer"],
            },
        },
    },
    "required": ["items"],
}


def generate_faq(media: AcquiredMedia, result: GeneratedResult) -> list[FaqItem]:
    cache_key = f"{media.cache_key}_faq"
    cached = cache.get("faq", cache_key)
    if cached:
        return [FaqItem.from_dict(d) for d in cached]

    if not result.chapters:
        return []

    numbered_chapters = "\n".join(f"{i}: {c.title}" for i, c in enumerate(result.chapters))
    prompt = f"""Here is a video's description and its chapters (numbered):

Description:
{result.description}

Chapters:
{numbered_chapters}

For each chapter, phrase it as a natural viewer question (e.g. "How do I
...?", "What is ...?") with a short 1-2 sentence answer based on the
description and chapter title. Return each item with the exact chapter_index
(the integer shown before the colon above — never invent a new one), a
question, and an answer. Return only the structured JSON.
"""
    raw = _ai_complete_json(prompt, _FAQ_SCHEMA)
    items = []
    for entry in raw.get("items") or []:
        try:
            idx = int(entry["chapter_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(result.chapters)):
            continue
        question = (entry.get("question") or "").strip()
        answer = (entry.get("answer") or "").strip()
        if not question or not answer:
            continue
        items.append(
            FaqItem(
                question=question,
                answer=answer,
                start_seconds=result.chapters[idx].start_seconds,
                chapter_title=result.chapters[idx].title,
            )
        )

    cache.set("faq", cache_key, [i.to_dict() for i in items])
    return items
