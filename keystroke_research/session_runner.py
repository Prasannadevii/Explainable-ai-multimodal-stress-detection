"""
Orchestrates a single typing-task data collection session:
display the prompt, capture keystrokes while the participant types it,
extract features, and persist raw + feature data.

Sessions run under one of three conditions (see config.SESSION_TYPES):
Relaxed (no time limit), Timed Typing (moderate time pressure), or
Stress Task (tight time limit + evaluative pressure). Each condition
carries a fixed "stress_level" label (low/medium/high) that is stored
alongside the extracted features as ground truth for later analysis.
"""

import threading
import time
import uuid
from datetime import datetime

from . import config, storage
from .capture import KeystrokeCapture
from .features import extract_features


class _CountdownAnnouncer(threading.Thread):
    """
    Background thread that prints time-remaining warnings during a
    timed session, without interrupting the blocking input() call
    that is capturing what the participant types.
    """

    def __init__(self, duration_sec: float, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.duration_sec = duration_sec
        self.stop_event = stop_event

    def run(self) -> None:
        start = time.perf_counter()
        announced = set()
        half = int(self.duration_sec * 0.5)
        

        while not self.stop_event.is_set():
            remaining = self.duration_sec - (time.perf_counter() - start)
            if remaining <= 0:
                print("\n[TIME'S UP] Please finish typing now.")
                return
            remaining_int = int(remaining)
            if  remaining_int not in announced:
                announced.add(remaining_int)
            time.sleep(0.2)


def run_typing_session(participant_id: str, session_type_key: str, session_id: str = None):
    """Run one full typing session for the given participant.

    session_type_key must be one of config.SESSION_TYPES' keys ("1", "2", "3").

    session_id: optional externally-supplied ID (e.g. from a combined
    multimodal run) so this session's rows line up with mouse/voice rows
    collected for the same task instance. If omitted, one is generated.

    Returns the SessionFeatures for the session, or None if no keystrokes
    were captured.
    """
    session_info = config.SESSION_TYPES[session_type_key]
    session_type = session_info["key"]
    stress_level = session_info["stress_level"]
    time_limit = session_info["time_limit_sec"]

    session_id = session_id or (
        f"{participant_id}_"
        f"{datetime.now().strftime(config.TIMESTAMP_FORMAT)}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    print("\n" + "=" * 70)
    print(f"Session type: {session_info['label']}  (stress level: {stress_level})")
    print(session_info["intro"])
    print("Type the paragraph below exactly as shown, then press Enter.")
    print("=" * 70)
    print(f"\n{config.SAMPLE_PARAGRAPH}\n")
    print("=" * 70)
    input("Press Enter when you are ready to begin...")
    print()
    if time_limit:
        print(f"[Timer starts now - you have {time_limit} seconds]\n")

    capture = KeystrokeCapture(session_id=session_id, participant_id=participant_id)
    capture.start()

    stop_event = threading.Event()
    announcer = None
    if time_limit:
        announcer = _CountdownAnnouncer(time_limit, stop_event)
        announcer.start()

    try:
        typed_text = input("> ")
    finally:
        events = capture.stop()
        stop_event.set()
        if announcer is not None:
            announcer.join(timeout=1)

    if not events:
        print("No keystrokes were recorded; session discarded.")
        return None

    raw_path = storage.save_raw_events(events)
    features = extract_features(
        session_id=session_id,
        participant_id=participant_id,
        events=events,
        typed_text=typed_text,
        session_type=session_type,
        stress_level=stress_level,
    )
    features_path = storage.save_features(features)

    print("\nSession complete.")
    print(f"  Raw keystrokes saved to:  {raw_path}")
    print(f"  Features saved to:       {features_path}")
    print("\n--- Summary ---")
    print(f"  Session type:      {session_info['label']} (stress: {stress_level})")
    print(f"  Total typing time: {features.total_typing_time_sec:.2f} s")
    print(f"  Words per minute:  {features.words_per_minute:.2f}")
    print(f"  Backspaces:        {features.backspace_count}")
    print(f"  Avg dwell time:    {features.avg_dwell_time_ms:.2f} ms")
    print(f"  Avg flight time:   {features.avg_flight_time_ms:.2f} ms")
    print(f"  Pauses (>{config.PAUSE_THRESHOLD_SECONDS:.1f}s): {features.pause_count}")

    return features