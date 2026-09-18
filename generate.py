"""Chapter + description generation via AI_COMPLETE.

Core correctness rule: the LLM NEVER emits a timestamp. It only ever picks a
`word_index` into the transcript's word-level array (built in transcribe.py).
All actual timestamps are resolved in code by looking up
`transcript.words[word_index].start` — so every emitted timestamp is a real,
transcribed moment, never a model guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cache
from acquire import AcquiredMedia, ExistingChapter
from config import (
    COMPLETE_MAX_TOKENS,
    COMPLETE_MODEL,
    MAX_CHAPTERS,
    MIN_CHAPTER_GAP_SECONDS,
    MIN_CHAPTERS,
)
from snowflake_conn import run_query
from transcribe import Transcript

# Words per LLM chunk when scanning for chapter boundaries (keeps prompts
# within a reasonable token budget for long videos, and — importantly —
# forces multiple independent LLM passes even for medium-length videos, since
# a single big call tends to under-cover the tail of a long transcript) and
# overlap between consecutive chunks so a topic change right at a chunk edge
# isn't missed.
CHAPTER_SCAN_CHUNK_WORDS = 1200
CHAPTER_SCAN_OVERLAP_WORDS = 150

# If the transcript is shorter than this, summarize it directly for the
# description step instead of chunk-summarizing first.
DIRECT_SUMMARY_WORD_LIMIT = 4000


class GenerationError(Exception):
    pass


@dataclass
class Chapter:
    start_seconds: float
    title: str


@dataclass
class GeneratedResult:
    description: str
    chapters: list[Chapter] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "description": self.description,
            "chapters": [c.__dict__ for c in self.chapters],
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            description=d["description"],
            chapters=[Chapter(**c) for c in d["chapters"]],
            keywords=d.get("keywords", []),
        )


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _ai_complete_json(prompt: str, schema: dict) -> dict:
    sql = "SELECT AI_COMPLETE(?, ?, PARSE_JSON(?), PARSE_JSON(?))"
    model_params = json.dumps({"temperature": 0.2, "max_tokens": COMPLETE_MAX_TOKENS})
    response_format = json.dumps({"type": "json", "schema": schema})
    rows = run_query(sql, params=(COMPLETE_MODEL, prompt, model_params, response_format))
    if not rows or rows[0][0] is None:
        raise GenerationError("AI_COMPLETE returned no result.")
    raw = rows[0][0]
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    # AI_COMPLETE with response_format returns the structured object directly.
    return parsed


def _windows(words: list, chunk_size: int, overlap: int):
    """Yield (start_idx, end_idx_exclusive) windows over a word list."""
    n = len(words)
    if n == 0:
        return
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end
        if end == n:
            break
        start += step


def _build_indexed_text(words: list, start_idx: int, end_idx: int) -> str:
    parts = [f"[{i}]{words[i].text}" for i in range(start_idx, end_idx)]
    return " ".join(parts)


_CHAPTER_SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "chapter_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word_index": {"type": "number"},
                    "title": {"type": "string"},
                },
                "required": ["word_index", "title"],
            },
        },
        "chunk_summary": {"type": "string"},
    },
    "required": ["chapter_candidates", "chunk_summary"],
}

_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "keywords"],
}


def _scan_chunk_for_chapters(words, start_idx: int, end_idx: int, is_first_chunk: bool) -> dict:
    indexed_text = _build_indexed_text(words, start_idx, end_idx)
    prompt = f"""You are analyzing a transcript of a video/tutorial to find natural topic or
chapter boundaries. Each word below is tagged with its index in the format
[INDEX]word — indices range from {start_idx} to {end_idx - 1} in this excerpt.

Transcript excerpt:
{indexed_text}

Task:
1. Identify natural chapter/topic boundaries in this excerpt (e.g. moving from
   intro to a new topic, a demo starting, a new section beginning).
2. For each boundary, return the exact word_index (an integer that MUST be
   between {start_idx} and {end_idx - 1} inclusive, and MUST be a value that
   appears in the [INDEX] tags above — never invent or estimate an index) of
   the FIRST word of that new chapter, plus a short, punchy chapter title
   (3-6 words, sentence case, no trailing punctuation).
{"3. Include a boundary at word_index " + str(start_idx) + " for the opening chapter." if is_first_chunk else ""}
4. Also return a 2-3 sentence summary of what this excerpt covers, for later
   use in writing an overall video description.

Return only the structured JSON.
"""
    return _ai_complete_json(prompt, _CHAPTER_SCAN_SCHEMA)


def _resolve_and_clean_chapters(
    words, raw_candidates: list[dict]
) -> list[Chapter]:
    n = len(words)
    resolved: list[tuple[int, str]] = []
    for c in raw_candidates:
        try:
            idx = int(c["word_index"])
        except (KeyError, TypeError, ValueError):
            continue
        idx = max(0, min(idx, n - 1))
        title = (c.get("title") or "").strip()
        if title:
            resolved.append((idx, title))

    resolved.sort(key=lambda x: x[0])

    # Always start at the true beginning.
    if not resolved or resolved[0][0] != 0:
        resolved.insert(0, (0, "Introduction"))

    cleaned: list[Chapter] = []
    last_start_seconds = None
    for idx, title in resolved:
        start_seconds = words[idx].start
        if last_start_seconds is not None and (start_seconds - last_start_seconds) < MIN_CHAPTER_GAP_SECONDS:
            continue  # too close to the previous kept chapter — drop
        cleaned.append(Chapter(start_seconds=start_seconds, title=title))
        last_start_seconds = start_seconds

    if len(cleaned) > MAX_CHAPTERS:
        # Keep the first chapter and evenly sample the rest to cap the count.
        rest = cleaned[1:]
        keep_count = MAX_CHAPTERS - 1
        step = len(rest) / keep_count
        sampled = [rest[int(i * step)] for i in range(keep_count)]
        cleaned = [cleaned[0]] + sampled

    return cleaned


def generate(media: AcquiredMedia, transcript: Transcript) -> GeneratedResult:
    cache_key = f"{media.cache_key}_{COMPLETE_MODEL}"
    cached = cache.get("generated", cache_key)
    if cached:
        return GeneratedResult.from_dict(cached)

    words = transcript.words
    if not words:
        raise GenerationError("Cannot generate chapters/description from an empty transcript.")

    chunk_summaries: list[str] = []
    all_candidates: list[dict] = []
    for i, (start_idx, end_idx) in enumerate(
        _windows(words, CHAPTER_SCAN_CHUNK_WORDS, CHAPTER_SCAN_OVERLAP_WORDS)
    ):
        result = _scan_chunk_for_chapters(words, start_idx, end_idx, is_first_chunk=(i == 0))
        all_candidates.extend(result.get("chapter_candidates") or [])
        if result.get("chunk_summary"):
            chunk_summaries.append(result["chunk_summary"])

    chapters = _resolve_and_clean_chapters(words, all_candidates)

    combined_summary = " ".join(chunk_summaries)
    existing_desc_block = (
        f"\n\nExisting YouTube description (refine and improve this, don't discard it "
        f"or write something unrelated):\n{media.existing_description}"
        if media.existing_description
        else ""
    )
    desc_prompt = f"""Write a compelling 2-4 sentence video description for a tutorial/video
based on this content summary:

{combined_summary}
{existing_desc_block}

Write in second person ("you'll..."), engaging but factual, no hashtags, no
emoji, no clickbait.

Also suggest 10-15 SEO keywords/tags for this video, suitable for pasting into
YouTube's video tags field and for improving search discoverability. Prefer
specific, high-intent multi-word phrases (e.g. "snowflake cortex ai tutorial")
over single generic words (e.g. "tutorial"), based on the topics actually
covered in the content summary above. Return only the structured JSON.
"""
    desc_result = _ai_complete_json(desc_prompt, _DESCRIPTION_SCHEMA)
    description = desc_result.get("description", "").strip()
    keywords = [k.strip() for k in (desc_result.get("keywords") or []) if k and k.strip()]

    result = GeneratedResult(description=description, chapters=chapters, keywords=keywords)
    cache.set("generated", cache_key, result.to_dict())
    return result
