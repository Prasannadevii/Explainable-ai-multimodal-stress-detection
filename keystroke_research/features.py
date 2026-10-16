"""
Feature extraction for keystroke dynamics.

Given the ordered list of raw KeyEvent records collected for a typing
session, this module derives standard keystroke dynamics features:

- dwell time:   press -> release of the SAME key
- flight time:  release of one key -> press of the next key
- typing speed: words per minute (WPM)
- total typing time, backspace count, and pause durations
"""

from collections import defaultdict, deque
from typing import List, Tuple

from . import config
from .models import KeyEvent, SessionFeatures

# (key, press_timestamp, release_timestamp)
KeyPair = Tuple[str, float, float]


def _pair_press_release(events: List[KeyEvent]) -> List[KeyPair]:
    """
    Pair each press event with its matching release event (FIFO per key).

    Returns pairs sorted by press timestamp. A release with no prior,
    unmatched press for that key (which can occasionally happen with
    OS-level key repeat/hook quirks) is ignored rather than guessed at.
    """
    pending = defaultdict(deque)
    pairs: List[KeyPair] = []

    for e in sorted(events, key=lambda ev: ev.timestamp):
        if e.event_type == "press":
            pending[e.key].append(e.timestamp)
        elif e.event_type == "release" and pending[e.key]:
            press_ts = pending[e.key].popleft()
            pairs.append((e.key, press_ts, e.timestamp))

    pairs.sort(key=lambda p: p[1])
    return pairs


def extract_features(
    session_id: str,
    participant_id: str,
    events: List[KeyEvent],
    typed_text: str,
    session_type: str = "relaxed",
    stress_level: str = "low",
) -> SessionFeatures:
    """
    Compute SessionFeatures from raw key events and the text produced.

    typed_text is the actual string the participant typed (as captured
    by the terminal input), used for WPM and character-count reporting.

    session_type / stress_level are condition labels for the session
    (e.g. "relaxed"/"low", "timed"/"medium", "stress"/"high") -- they
    describe which task condition the participant was assigned to, not
    a measurement derived from their behavior.
    """
    pairs = _pair_press_release(events)

    if not pairs:
        return SessionFeatures(
            session_id=session_id,
            participant_id=participant_id,
            session_type=session_type,
            stress_level=stress_level,
            total_typing_time_sec=0.0,
            total_keystrokes=0,
            correct_characters=len(typed_text),
            backspace_count=0,
            words_per_minute=0.0,
            avg_dwell_time_ms=0.0,
            avg_flight_time_ms=0.0,
            pause_count=0,
            total_pause_time_sec=0.0,
            avg_pause_duration_sec=0.0,
        )

    # --- Dwell time: press -> release of the same key ---
    dwell_times = [release - press for _, press, release in pairs]
    avg_dwell_ms = (sum(dwell_times) / len(dwell_times)) * 1000

    # --- Flight time: release of key n -> press of key n+1 ---
    flight_times = []
    for i in range(1, len(pairs)):
        prev_release = pairs[i - 1][2]
        curr_press = pairs[i][1]
        gap = curr_press - prev_release
        if gap >= 0:
            flight_times.append(gap)
    avg_flight_ms = (sum(flight_times) / len(flight_times) * 1000) if flight_times else 0.0

    # --- Pauses: flight gaps at or beyond the configured threshold ---
    pauses = [g for g in flight_times if g >= config.PAUSE_THRESHOLD_SECONDS]
    pause_count = len(pauses)
    total_pause_time = sum(pauses)
    avg_pause_duration = (total_pause_time / pause_count) if pause_count else 0.0

    # --- Total typing time: first press to last release ---
    first_press = pairs[0][1]
    last_release = pairs[-1][2]
    total_typing_time = max(last_release - first_press, 0.0)

    # --- Backspace count ---
    backspace_count = sum(1 for key, _, _ in pairs if key == "backspace")

    # --- Typing speed (WPM), standard convention: 5 characters = 1 word ---
    minutes = total_typing_time / 60.0
    words_typed = len(typed_text) / 5.0
    wpm = (words_typed / minutes) if minutes > 0 else 0.0

    return SessionFeatures(
        session_id=session_id,
        participant_id=participant_id,
        session_type=session_type,
        stress_level=stress_level,
        total_typing_time_sec=round(total_typing_time, 3),
        total_keystrokes=len(pairs),
        correct_characters=len(typed_text),
        backspace_count=backspace_count,
        words_per_minute=round(wpm, 2),
        avg_dwell_time_ms=round(avg_dwell_ms, 2),
        avg_flight_time_ms=round(avg_flight_ms, 2),
        pause_count=pause_count,
        total_pause_time_sec=round(total_pause_time, 3),
        avg_pause_duration_sec=round(avg_pause_duration, 3),
    )