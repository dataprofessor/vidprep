"""Central configuration constants for the video-automation app."""

# Fully-qualified internal stage used to stage audio/video files for AI_TRANSCRIBE.
STAGE_FQN = "CHANINN_DEMO_DATA.APPS.VIDEO_AUTOMATION_STAGE"

# Model used for AI_COMPLETE description/chapter generation.
COMPLETE_MODEL = "claude-4-sonnet"

# Max output tokens for AI_COMPLETE calls (generous to avoid truncation).
COMPLETE_MAX_TOKENS = 8192

# AI_TRANSCRIBE limits (seconds) — chunk longer media to stay safely under the
# documented 60-minute cap when using word-level timestamp granularity.
MAX_CHUNK_SECONDS = 55 * 60

# Target number of chapters to aim for in generated output.
MIN_CHAPTERS = 6
MAX_CHAPTERS = 12

# Minimum gap (seconds) enforced between two consecutive chapter start times.
MIN_CHAPTER_GAP_SECONDS = 10

# Local cache directory for transcripts and generated results.
CACHE_DIR = ".cache"

# Local scratch directory for downloaded/uploaded media before staging.
TMP_DIR = ".cache/tmp"

# Optional Netscape-format cookies file (exported from a real logged-in
# YouTube session) used to work around YouTube's bot-detection on
# datacenter/cloud IPs (e.g. Snowflake SPCS). Only used if the file exists.
YOUTUBE_COOKIES_FILE = "youtube_cookies.txt"
