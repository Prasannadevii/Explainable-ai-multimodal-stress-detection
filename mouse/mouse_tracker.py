#!/usr/bin/env python3
"""
mouse_stress_tracker.py

Companion instrument to your keystroke stress-detection dataset.

Passively records EVERY mouse move, click, and scroll for the duration
of a session. The GUI (mouse_gui.py) layers a click-the-target game on
top of this (see target_game.py) as the actual task participants do;
this module still records raw movement underneath that regardless, and
extracts a feature row from it at session end.

Stress level is inferred with a rule-based heuristic, not a trained
model -- see classify_stress_game() for the game (the primary path:
based on hit rate, stray clicks, and reaction time/variability) and
classify_stress() for plain free-form movement (only used for
"timed"/"relaxed" sessions started from the CLI below, not from the
GUI). Once you have enough labeled sessions (ideally paired with the
ECG recorded alongside), swap either out for a proper trained
classifier.

Column names are kept aligned with your keystroke CSV so the two datasets
join cleanly on session_id / participant_id / session_type / stress_level.

Install:
    pip install pynput

Run (interactive):
    python mouse_stress_tracker.py

Run (non-interactive):
    python mouse_stress_tracker.py --participant 354c862d --session-type timed \
        --duration 120 --output mouse_stress_sessions.csv

Stop early any time by pressing ESC (works in both timed and relaxed mode).

Platform notes:
    - macOS: System Settings -> Privacy & Security -> Accessibility AND
      Input Monitoring must both allow your terminal/Python.
    - Linux: needs an X11 session (Wayland restricts global input hooks
      for most compositors).
    - Windows: works out of the box.
"""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from statistics import mean, pstdev

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOUSE_DATA_DIR = PROJECT_ROOT / "data" / "mouse"
DEFAULT_MOUSE_CSV = DEFAULT_MOUSE_DATA_DIR / "mouse_stress_sessions.csv"

try:
    from pynput import mouse, keyboard
except ImportError:
    sys.exit(
        "Missing dependency. Install it first:\n\n    pip install pynput\n"
    )

# ---------------------------------------------------------------------------
# Column set. Kept aligned with the keystroke CSV's naming style:
#   - shared join keys: session_id, participant_id, session_type, stress_level
#   - pause_count / total_pause_time_sec / avg_pause_duration_sec reuse the
#     exact same names+semantics as the keystroke file (idle-gap detection)
#   - everything mouse-specific is prefixed mouse_/click_ so a merge on the
#     join keys never collides with a keystroke column
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "session_id",
    "participant_id",
    "session_type",
    "stress_level",
    "total_tracking_time_sec",
    "total_mouse_events",
    "total_distance_px",
    "mouse_avg_speed_px_s",
    "mouse_max_speed_px_s",
    "mouse_speed_std_px_s",
    "mouse_avg_acceleration_px_s2",
    "mouse_avg_jerk_px_s3",
    "mouse_path_straightness_ratio",
    "mouse_direction_change_count",
    "click_count",
    "left_click_count",
    "right_click_count",
    "mouse_avg_click_dwell_time_ms",
    "mouse_avg_time_between_clicks_ms",
    "mouse_scroll_count",
    "pause_count",
    "total_pause_time_sec",
    "avg_pause_duration_sec",
    "mouse_stress_score",
    # -- click-the-target game (session_type == "game") only; blank for
    # timed/relaxed sessions where no target task was run. See
    # target_game.py -- this is a structured task (fixed circles to
    # click before they vanish) layered on top of the same free-form
    # movement recording above, giving reaction-time and precision
    # signals that passive tracking alone can't.
    "game_targets_shown",
    "game_hits",
    "game_misses",
    "game_hit_rate",
    "game_stray_clicks",
    "game_avg_reaction_time_ms",
    "game_reaction_time_std_ms",
    "game_avg_click_offset_px",
]

GAME_FIELDNAMES = [f for f in FIELDNAMES if f.startswith("game_")]


class MouseSessionRecorder:
    """Global mouse listener that logs every move/click/scroll with timestamps."""

    def __init__(self, pause_threshold=0.3):
        self.pause_threshold = pause_threshold
        self.events = []          # {'type': move|down|up|scroll, 'x','y','t', ['button']}
        self.click_records = []   # {'t_down','t_up','x','y','button','dwell_ms'}
        self._pending_press = {}  # button -> (t, x, y)
        self.t_start = None
        self.t_end = None
        self._stop_event = None

    # -- pynput callbacks ---------------------------------------------------
    def _on_move(self, x, y):
        self.events.append({"type": "move", "x": x, "y": y, "t": time.perf_counter()})

    def _on_click(self, x, y, button, pressed):
        t = time.perf_counter()
        self.events.append(
            {"type": "down" if pressed else "up", "x": x, "y": y, "t": t, "button": str(button)}
        )
        if pressed:
            self._pending_press[button] = (t, x, y)
        else:
            if button in self._pending_press:
                t_down, dx, dy = self._pending_press.pop(button)
                self.click_records.append(
                    {
                        "t_down": t_down,
                        "t_up": t,
                        "x": x,
                        "y": y,
                        "button": str(button),
                        "dwell_ms": (t - t_down) * 1000,
                    }
                )

    def _on_scroll(self, x, y, dx, dy):
        self.events.append({"type": "scroll", "x": x, "y": y, "t": time.perf_counter()})

    def _on_key_press(self, key):
        if key == keyboard.Key.esc:
            self._stop_event.set()
            return False  # stop the keyboard listener

    # -- non-blocking start/stop, for GUI-driven sessions ------------------
    def start(self):
        """Start listening in the background. Call stop() later to finish.
        Unlike record(), this never blocks -- meant for use inside a
        Tkinter mainloop (see mouse_gui.py)."""
        import threading

        self._stop_event = threading.Event()
        self._kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self._m_listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll
        )
        self._kb_listener.start()
        self._m_listener.start()
        self.t_start = time.perf_counter()

    def stop(self):
        """Stop listening. Safe to call once; returns nothing, just tears
        down the listeners started by start()."""
        self.t_end = time.perf_counter()
        if getattr(self, "_m_listener", None) is not None:
            self._m_listener.stop()
        if getattr(self, "_kb_listener", None) is not None and self._kb_listener.running:
            self._kb_listener.stop()

    def esc_was_pressed(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    # -- recording loop -------------------------------------------------
    def record(self, session_type, duration=None):
        """
        Blocks until the session ends.
        session_type == 'timed'   -> stops automatically after `duration` seconds
                                      (or earlier if ESC is pressed)
        session_type == 'relaxed' -> runs until ESC is pressed (no time limit)
        """
        import threading

        self._stop_event = threading.Event()

        kb_listener = keyboard.Listener(on_press=self._on_key_press)
        m_listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll
        )

        kb_listener.start()
        m_listener.start()
        self.t_start = time.perf_counter()

        print("\n● Recording... move your mouse and work as you normally would.")
        print("  Press ESC at any time to stop early.\n")

        try:
            if session_type == "timed":
                end_at = self.t_start + duration
                last_print = -1
                while time.perf_counter() < end_at and not self._stop_event.is_set():
                    remaining = end_at - time.perf_counter()
                    if int(remaining) != last_print and int(remaining) % 5 == 0:
                        print(f"  ...{int(remaining):>3d}s remaining  "
                              f"({len(self.events)} events captured)")
                        last_print = int(remaining)
                    time.sleep(0.1)
            else:  # relaxed: wait indefinitely for ESC
                while not self._stop_event.is_set():
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass

        self.t_end = time.perf_counter()
        m_listener.stop()
        if kb_listener.running:
            kb_listener.stop()
        print(f"\n■ Recording stopped. {len(self.events)} raw events captured "
              f"over {self.t_end - self.t_start:.1f}s.\n")


# ---------------------------------------------------------------------------
# Stress inference -- rule-based heuristic over the extracted features.
#
# Rationale (well-supported directions in HCI/affective-computing literature,
# even though exact thresholds below are reasonable defaults, not fitted):
#   - higher speed variability      -> more erratic control      -> more stress
#   - more direction reversals/sec  -> more corrective micro-moves -> more stress
#   - higher jerk                   -> less smooth motor control -> more stress
#   - more pauses per minute        -> more hesitation/checking   -> more stress
#   - higher path-straightness ratio (wandering vs direct)        -> more stress
#
# Each indicator is normalized to 0-1 against a plausible working range, then
# averaged into a single composite `mouse_stress_score`. This is a heuristic
# starting point -- replace with a model trained on your own labeled data
# (e.g. paired with EDA/HRV ground truth) once you have enough sessions.
# ---------------------------------------------------------------------------
def classify_stress(speed_std, dir_change_rate, avg_jerk, pause_rate, straightness):
    def clamp01(v):
        return max(0.0, min(1.0, v))

    speed_std_n = clamp01(speed_std / 400.0)          # px/s std -- 400+ considered high
    dir_change_n = clamp01(dir_change_rate / 3.0)      # reversals/sec -- 3+ considered high
    jerk_n = clamp01(abs(avg_jerk) / 50000.0)          # px/s^3 -- scale is rough by design
    pause_rate_n = clamp01(pause_rate / 10.0)          # pauses/min -- 10+ considered high
    straightness_n = clamp01((straightness - 1.0) / 1.5)  # 1.0 = dead straight

    score = mean([speed_std_n, dir_change_n, jerk_n, pause_rate_n, straightness_n])

    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"

    breakdown = {
        "speed_std": speed_std_n,
        "direction_change_rate": dir_change_n,
        "jerk": jerk_n,
        "pause_rate": pause_rate_n,
        "path_straightness": straightness_n,
    }
    return level, round(score, 3), breakdown


# ---------------------------------------------------------------------------
# Stress inference for the click-target game specifically.
#
# classify_stress() above was calibrated for "use your mouse normally"
# free-form tracking, where fast/jerky/reversal-heavy movement is itself
# a sign of agitation. That assumption breaks for the game: chasing a
# circle around the screen *requires* fast corrective movement and lots
# of direction changes even when you're calm and doing fine -- so
# classify_stress() reads a good, focused round as "high stress" just
# because it was fast, which is exactly the false positive this
# function replaces.
#
# For the game, task PERFORMANCE is the more meaningful signal:
#   - lower hit rate                      -> more misses/overload -> more stress
#   - more stray clicks per target shown  -> impulsive/frustrated -> more stress
#   - higher reaction-time variability    -> inconsistent, less controlled -> more stress
#   - slower average reaction time        -> hesitation/overload -> more stress
#
# Same caveat as classify_stress(): normalization ranges below are
# reasonable starting points, not fitted to real labeled data. Swap
# them out once you have enough sessions (ideally paired with the ECG
# recorded alongside each game round) to calibrate against.
# ---------------------------------------------------------------------------
def classify_stress_game(game_summary: dict):
    def clamp01(v):
        return max(0.0, min(1.0, v))

    hit_rate = game_summary.get("game_hit_rate", 0.0)
    targets_shown = game_summary.get("game_targets_shown", 0) or 1
    stray_clicks = game_summary.get("game_stray_clicks", 0)
    avg_reaction_ms = game_summary.get("game_avg_reaction_time_ms", 0.0)
    reaction_std_ms = game_summary.get("game_reaction_time_std_ms", 0.0)

    miss_n = clamp01(1.0 - hit_rate)                          # 0 misses -> 0, all misses -> 1
    stray_rate_n = clamp01((stray_clicks / targets_shown) / 0.5)  # 0.5+ stray clicks/target -> high
    reaction_n = clamp01((avg_reaction_ms - 350.0) / 450.0)    # ~350ms calm baseline, 800ms+ -> high
    variability_n = clamp01(reaction_std_ms / 250.0)           # inconsistent timing -> high

    score = mean([miss_n, stray_rate_n, reaction_n, variability_n])

    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"

    breakdown = {
        "miss_rate": miss_n,
        "stray_click_rate": stray_rate_n,
        "reaction_time": reaction_n,
        "reaction_time_variability": variability_n,
    }
    return level, round(score, 3), breakdown


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def compute_features(rec: MouseSessionRecorder, participant_id, session_type,
                      session_id=None, game_summary=None):
    moves = [e for e in rec.events if e["type"] == "move"]
    moves.sort(key=lambda e: e["t"])

    total_dur = (rec.t_end - rec.t_start) if (rec.t_start and rec.t_end) else 0.0

    distances, speeds, accels, jerks = [], [], [], []
    dir_changes = 0
    prev_speed = prev_accel = prev_angle = None

    for a, b in zip(moves, moves[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        d = (dx * dx + dy * dy) ** 0.5
        if dt > rec.pause_threshold:
            continue  # this gap is a pause, not motion -- handled separately
        distances.append(d)
        speed = d / dt
        speeds.append(speed)
        if prev_speed is not None:
            accel = (speed - prev_speed) / dt
            accels.append(accel)
            if prev_accel is not None:
                jerks.append((accel - prev_accel) / dt)
            prev_accel = accel
        prev_speed = speed
        if d > 2:
            import math
            angle = math.atan2(dy, dx)
            if prev_angle is not None:
                diff = abs(angle - prev_angle)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                if diff > (math.pi / 6):
                    dir_changes += 1
            prev_angle = angle

    total_distance = sum(distances)

    # -- pause detection across ALL events (move/click/scroll), plus the
    #    boundary gaps at the very start/end of the session
    all_t = sorted(e["t"] for e in rec.events)
    if rec.t_start is not None:
        all_t = [rec.t_start] + all_t
    if rec.t_end is not None:
        all_t = all_t + [rec.t_end]
    pause_gaps = [b - a for a, b in zip(all_t, all_t[1:]) if (b - a) > rec.pause_threshold]

    # -- path straightness: ratio of actual path length to straight-line
    #    distance, computed per inter-click segment (falls back to whole
    #    session if there were no clicks)
    click_times = sorted(c["t_down"] for c in rec.click_records)
    marks = [moves[0]["t"]] if moves else []
    marks += click_times
    marks = sorted(set(marks))
    straight_ratios = []
    for s, e in zip(marks, marks[1:]):
        seg = [m for m in moves if s <= m["t"] <= e]
        if len(seg) < 2:
            continue
        seg_dist = sum(
            ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
            for a, b in zip(seg, seg[1:])
        )
        straight = ((seg[-1]["x"] - seg[0]["x"]) ** 2 + (seg[-1]["y"] - seg[0]["y"]) ** 2) ** 0.5
        if straight > 5:
            straight_ratios.append(seg_dist / straight)

    dwells = [c["dwell_ms"] for c in rec.click_records]
    click_ts = sorted(c["t_down"] for c in rec.click_records)
    inter_click = [(b - a) * 1000 for a, b in zip(click_ts, click_ts[1:])]

    left_clicks = sum(1 for c in rec.click_records if "left" in c["button"].lower())
    right_clicks = sum(1 for c in rec.click_records if "right" in c["button"].lower())

    scroll_count = sum(1 for e in rec.events if e["type"] == "scroll")

    def safe_mean(v):
        return round(mean(v), 3) if v else 0.0

    def safe_std(v):
        return round(pstdev(v), 3) if len(v) > 1 else 0.0

    speed_std_val = safe_std(speeds)
    avg_jerk_val = safe_mean(jerks)
    straightness_val = safe_mean(straight_ratios) or 1.0
    dir_change_rate = dir_changes / total_dur if total_dur > 0 else 0.0
    pause_rate_per_min = (len(pause_gaps) / total_dur * 60) if total_dur > 0 else 0.0

    # Movement-based heuristic still computed and stored (mouse_stress_score
    # + breakdown below) since the raw movement features are useful signal
    # for a downstream trained model either way -- it's just no longer what
    # decides the stress_level LABEL for a game round, since it's tuned for
    # free-form movement and reads normal fast gameplay as "high" stress
    # (see classify_stress_game() above for why). When a game was played,
    # its own performance-based classifier decides the label instead.
    inferred_level, stress_score, breakdown = classify_stress(
        speed_std_val, dir_change_rate, avg_jerk_val, pause_rate_per_min, straightness_val
    )
    if game_summary:
        inferred_level, stress_score, game_breakdown = classify_stress_game(game_summary)
        breakdown = game_breakdown

    row = {
        "session_id": session_id or _make_session_id(participant_id),
        "participant_id": participant_id,
        "session_type": session_type,
        "stress_level": inferred_level,
        "total_tracking_time_sec": round(total_dur, 3),
        "total_mouse_events": len(rec.events),
        "total_distance_px": round(total_distance, 1),
        "mouse_avg_speed_px_s": safe_mean(speeds),
        "mouse_max_speed_px_s": round(max(speeds), 2) if speeds else 0.0,
        "mouse_speed_std_px_s": safe_std(speeds),
        "mouse_avg_acceleration_px_s2": safe_mean(accels),
        "mouse_avg_jerk_px_s3": safe_mean(jerks),
        "mouse_path_straightness_ratio": safe_mean(straight_ratios),
        "mouse_direction_change_count": dir_changes,
        "click_count": len(rec.click_records),
        "left_click_count": left_clicks,
        "right_click_count": right_clicks,
        "mouse_avg_click_dwell_time_ms": safe_mean(dwells),
        "mouse_avg_time_between_clicks_ms": safe_mean(inter_click),
        "mouse_scroll_count": scroll_count,
        "pause_count": len(pause_gaps),
        "total_pause_time_sec": round(sum(pause_gaps), 3),
        "avg_pause_duration_sec": safe_mean(pause_gaps),
        "mouse_stress_score": stress_score,
    }

    # game_summary comes from target_game.ClickTargetGame.summary() when
    # session_type == "game" -- merge in whichever of its keys are actual
    # CSV columns (it also carries a "results" list, which isn't). Left
    # blank (not zero) for non-game sessions -- "no target task was run"
    # is a different fact from "zero hits".
    for f in GAME_FIELDNAMES:
        row[f] = ""
    if game_summary:
        for f in GAME_FIELDNAMES:
            if f in game_summary:
                row[f] = game_summary[f]

    return row, breakdown


def _make_session_id(participant_id):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"P-{participant_id}_{ts}_{rand}"


def _load_and_normalize_rows(path, fieldnames):
    """Read every existing row in the mouse CSV and normalize it to the
    CURRENT schema (fieldnames), regardless of which schema version it
    was originally written under -- same approach as
    keystroke_research/storage.py, so adding a new column (like the
    game_* fields) never corrupts rows written before that column
    existed. Missing columns on old rows are filled blank, not
    "unknown" -- these are numeric feature columns, and blank is what
    the rest of the pipeline already treats as "not collected"."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        all_lines = list(csv.reader(f))

    if not all_lines:
        return []

    header, data_lines = all_lines[0], all_lines[1:]
    normalized = []
    for values in data_lines:
        if not values:
            continue
        if len(values) == len(fieldnames):
            normalized.append(dict(zip(fieldnames, values)))
        elif len(values) == len(header):
            old_row = dict(zip(header, values))
            normalized.append({f: old_row.get(f, "") for f in fieldnames})
        else:
            partial = dict(zip(fieldnames, values))
            for f in fieldnames:
                partial.setdefault(f, "")
            normalized.append(partial)

    return normalized


def save_row(row, output_path):
    """Append one session's row to output_path, rewriting the file with
    the current FIELDNAMES header each time and normalizing any
    pre-existing rows to match (see _load_and_normalize_rows)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if output_path.exists() and output_path.stat().st_size > 0:
        rows = _load_and_normalize_rows(output_path, FIELDNAMES)
    rows.append({f: row.get(f, "") for f in FIELDNAMES})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Open-ended mouse stress-tracking session recorder.")
    p.add_argument("--participant", help="Participant ID, e.g. 354c862d")
    p.add_argument("--session-type", choices=["timed", "relaxed"], help="timed = auto-stops after --duration; relaxed = runs until ESC")
    p.add_argument("--duration", type=float, default=60.0, help="Duration in seconds for timed sessions (default 60)")
    p.add_argument("--session-id", help="Reuse an existing session_id (e.g. from your keystroke tool) to force an exact join on that row")
    p.add_argument("--output", default=str(DEFAULT_MOUSE_CSV), help="CSV file to append the session's feature row to")
    p.add_argument("--pause-threshold", type=float, default=0.3, help="Seconds of inactivity that counts as a pause (default 0.3, matches keystroke tool)")
    return p.parse_args()


def run_mouse_session(participant_id, session_type="timed", duration=60.0,
                       session_id=None, output_path=None,
                       pause_threshold=0.3, verbose=True):
    """
    Programmatic entry point: records one mouse session and appends the
    extracted feature row to output_path. Used both by this module's CLI
    and by run_combined_session.py so a keystroke/mouse/voice trio can
    share one session_id.

    Returns (row, breakdown).
    """
    output_path = output_path or str(DEFAULT_MOUSE_CSV)
    rec = MouseSessionRecorder(pause_threshold=pause_threshold)
    rec.record(session_type=session_type, duration=duration)

    row, breakdown = compute_features(rec, participant_id, session_type, session_id=session_id)
    save_row(row, output_path)

    if verbose:
        print("Extracted features:")
        for k, v in row.items():
            print(f"  {k:35s} {v}")
        print(f"\nInferred stress level: {row['stress_level'].upper()}  "
              f"(composite score {row['mouse_stress_score']})")
        print("Indicator breakdown (0 = calm pattern, 1 = highly stressed pattern):")
        for k, v in breakdown.items():
            print(f"  {k:25s} {v:.2f}")
        print(f"\nAppended to: {os.path.abspath(output_path)}")

    return row, breakdown


def main():
    args = parse_args()

    participant_id = args.participant or input("Participant ID: ").strip() or "unknown"
    session_type = args.session_type or (input("Session type [timed/relaxed]: ").strip() or "timed")
    duration = args.duration
    if session_type == "timed" and args.duration == 60.0 and args.session_type is None:
        try:
            duration = float(input("Duration in seconds [60]: ").strip() or 60)
        except ValueError:
            duration = 60.0

    run_mouse_session(
        participant_id=participant_id,
        session_type=session_type,
        duration=duration,
        session_id=args.session_id,
        output_path=args.output,
        pause_threshold=args.pause_threshold,
    )

    print("Join this file with your keystroke CSV on session_id / participant_id / "
          "session_type / stress_level to get one combined feature row per session.")
    print("\nNote: stress_level here is a rule-based estimate from movement patterns, "
          "not a trained model. Treat it as a starting heuristic and recalibrate the "
          "thresholds in classify_stress() once you have real labeled data.")


if __name__ == "__main__":
    main()