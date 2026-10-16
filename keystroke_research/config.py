"""
Configuration constants for the Keystroke Dynamics Research application.

Centralizing configuration here makes it easy to adjust paths, the
typing prompt, and thresholds without touching business logic in other
modules, and keeps the project easy to extend with future data
modalities (e.g. speech, wearable sensors) without restructuring.
"""

from pathlib import Path

# --- Directory layout ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "keystroke"

PARTICIPANTS_FILE = DATA_DIR / "participants.csv"

RAW_KEYSTROKES_DIR = DATA_DIR / "raw_keystrokes"
FEATURES_DIR = DATA_DIR / "features"

# Reserved for future modalities. Keeping the top-level data layout
# stable now means new collectors can be plugged in later without
# reorganizing existing participant data.
RAW_SPEECH_DIR = DATA_DIR / "raw_speech"
RAW_WEARABLE_DIR = DATA_DIR / "raw_wearable"

# --- Typing task -------------------------------------------------------

SAMPLE_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the sun sets "
    "behind the distant hills. Typing patterns reveal a great deal "
    "about how a person thinks and moves, and researchers study these "
    "small rhythms to understand attention, fatigue, and skill."
)

# A gap between the release of one key and the press of the next that
# exceeds this many seconds is counted as a "pause" during feature
# extraction.
PAUSE_THRESHOLD_SECONDS = 1.0

# --- Session types -------------------------------------------------------
#
# Each session is collected under one of three conditions. "stress_level"
# is stored as a ground-truth condition label alongside the extracted
# features (low/medium/high) -- it is not measured from the participant,
# it is the condition they were assigned to. This label is intended for
# later analysis or model training done outside this application; this
# app does not build or run any model itself.
#
# "time_limit_sec" drives an in-session countdown for the Timed Typing
# and Stress Task conditions. Relaxed has no limit.

SESSION_TYPES = {
    "1": {
        "key": "relaxed",
        "label": "Relaxed",
        "stress_level": "low",
        "time_limit_sec": None,
        "intro": "Take your time. There is no time limit for this session.",
    },
    "2": {
        "key": "timed",
        "label": "Timed Typing",
        "stress_level": "medium",
        "time_limit_sec": 45,
        "intro": "Try to finish typing the paragraph within 45 seconds.",
    },
    "3": {
        "key": "stress",
        "label": "Stress Task",
        "stress_level": "high",
        "time_limit_sec": 20,
        "intro": (
            "You have only 20 seconds. Type as quickly and accurately as "
            "possible - your speed and accuracy are being scored."
        ),
    },
}

# --- Misc ----------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"