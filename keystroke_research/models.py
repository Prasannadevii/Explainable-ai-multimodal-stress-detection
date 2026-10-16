"""
Data models shared across the keystroke dynamics application.

Using dataclasses keeps raw event records and derived feature records
explicit and typed, making it straightforward to add new fields later
(e.g. pressure or accelerometer readings from a wearable sensor
collector) without breaking existing consumers of this data.
"""

from dataclasses import dataclass


@dataclass
class KeyEvent:
    """A single key press or key release event."""

    session_id: str
    participant_id: str
    key: str          # normalized key name, e.g. "a", "space", "backspace"
    event_type: str    # "press" or "release"
    timestamp: float   # seconds, from time.perf_counter()


@dataclass
class SessionFeatures:
    """Aggregate keystroke dynamics features for one typing session."""

    session_id: str
    participant_id: str
    session_type: str        # "relaxed", "timed", or "stress"
    stress_level: str        # "low", "medium", or "high" -- condition label
    total_typing_time_sec: float
    total_keystrokes: int
    correct_characters: int
    backspace_count: int
    words_per_minute: float
    avg_dwell_time_ms: float
    avg_flight_time_ms: float
    pause_count: int
    total_pause_time_sec: float
    avg_pause_duration_sec: float