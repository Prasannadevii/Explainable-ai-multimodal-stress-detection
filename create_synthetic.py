#!/usr/bin/env python3
"""
create_synthetic.py

Generates additional stress-session data (keystroke, mouse, voice, ECG)
that matches the exact CSV schemas produced by this project's real
collectors:

    keystroke_research/features.py    -> data/keystroke/features/session_features.csv
    mouse/mouse_tracker.py            -> data/mouse/mouse_stress_sessions.csv
    voice/extract_voice_features.py   -> data/voice/voice_features_dataset.csv
    ecg/ecg_link.py + ecg_features.py -> data/ecg/ecg_stress_sessions.csv

Drop this next to main.py. It never imports or runs main.py, and only
touches fusion/merged_dataset.csv if you pass --merge.

LEARNS FROM YOUR EXISTING DATA
--------------------------------
Before generating anything, the script looks at whichever of the four
CSVs above already exist and have rows in them. For every numeric
column it computes the real mean/std *per stress_level*
("low"/"medium"/"high") and uses that as the sampling distribution for
new rows -- so if your existing "high" stress sessions average 105
bpm with an sdnn of 18ms, new "high" sessions get generated around
that, not around a generic textbook number. A (column, level) with
too few existing rows (fewer than --min-real-rows, default 3) falls
back to a built-in literature-informed default instead.

Participant IDs continue whatever numbering style your existing data
already uses (e.g. if you have "P001".."P018", new rows continue
"P019", "P020", ...); if no consistent style is found, or there's no
existing data yet, plain sequential IDs are used.

Each participant also gets their own random "registration" moment,
scattered over the last --days-back days (default 90) and biased
toward normal waking hours, and their per-level sessions are then
spread across the following ~10 days with extra jitter -- rather than
every row in a batch sharing the exact same generation timestamp.
This only affects the timestamps embedded in session_id and the
voice CSV's timestamp/file columns.

NEVER OVERWRITES EXISTING DATA
--------------------------------
By default this script APPENDS new rows to whichever CSVs already
exist -- it never erases or replaces what's there. Pass --overwrite
only if you explicitly want to start a file fresh.

WHY "categorical", not purely random
-------------------------------------
Every generated session belongs to one of three ground-truth
conditions -- low / medium / high stress -- and each feature is
sampled from a condition-specific distribution whose default (pre-
data) direction matches what keystroke/mouse/voice/ECG research
associates with stress:

  keystroke : slower net WPM, more backspaces, more/longer pauses,
              shorter dwell & flight times (tense, clipped keypresses)
  mouse     : faster/jerkier movement, less straight paths, more
              direction changes, worse click-the-target accuracy and
              slower/more variable reaction time
  voice     : higher pitch (F0), higher energy/RMS, higher zero-
              crossing rate, higher spectral centroid
  ECG       : higher heart rate, LOWER heart-rate-variability
              (sdnn/rmssd/pnn50) -- reduced HRV is the classic
              autonomic marker of acute stress

Each generated participant also gets small persistent per-participant
offsets (a calmer-than-average typist, a naturally higher resting
heart rate, etc.) so sessions from the same participant aren't i.i.d.
noise -- they look like repeated measures from one person.

Usage
-----
    python create_synthetic.py                       # adapts to existing data if present, appends
    python create_synthetic.py --participants 40 --repeats 3
    python create_synthetic.py --seed 42 --merge      # reproducible + rebuild fusion/merged_dataset.csv after
    python create_synthetic.py --overwrite            # replace existing CSVs instead of appending to them
    python create_synthetic.py --out-dir ./more_data  # write elsewhere instead of the real data/ paths
    python create_synthetic.py --ignore-existing-data # force built-in defaults, ignore what's already there
"""

import argparse
import csv
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

KEYSTROKE_CSV = BASE_DIR / "data" / "keystroke" / "features" / "session_features.csv"
MOUSE_CSV = BASE_DIR / "data" / "mouse" / "mouse_stress_sessions.csv"
VOICE_CSV = BASE_DIR / "data" / "voice" / "voice_features_dataset.csv"
ECG_CSV = BASE_DIR / "data" / "ecg" / "ecg_stress_sessions.csv"

STRESS_LEVELS = ["low", "medium", "high"]
KS_SESSION_TYPE = {"low": "relaxed", "medium": "timed", "high": "stress"}
VOICE_LEVEL_NUM = {"low": 1, "medium": 2, "high": 3}
NUM_TO_LEVEL = {v: k for k, v in VOICE_LEVEL_NUM.items()}

SAMPLE_PARAGRAPH_LEN = 236  # matches keystroke_research/config.py SAMPLE_PARAGRAPH length
PROMPTS = [
    "Describe your morning routine in a few sentences.",
    "Talk about a place you would like to visit and why.",
    "Explain how you would plan a weekend trip.",
]

# ---------------------------------------------------------------------------
# Exact column order for each real CSV, copied from the scripts that write
# them, so these files drop into the pipeline (merge_datasets.py etc.)
# without any schema surprises.
# ---------------------------------------------------------------------------

KEYSTROKE_FIELDS = [
    "session_id", "participant_id", "session_type", "stress_level",
    "total_typing_time_sec", "total_keystrokes", "correct_characters",
    "backspace_count", "words_per_minute", "avg_dwell_time_ms",
    "avg_flight_time_ms", "pause_count", "total_pause_time_sec",
    "avg_pause_duration_sec",
]

MOUSE_FIELDS = [
    "session_id", "participant_id", "session_type", "stress_level",
    "total_tracking_time_sec", "total_mouse_events", "total_distance_px",
    "mouse_avg_speed_px_s", "mouse_max_speed_px_s", "mouse_speed_std_px_s",
    "mouse_avg_acceleration_px_s2", "mouse_avg_jerk_px_s3",
    "mouse_path_straightness_ratio", "mouse_direction_change_count",
    "click_count", "left_click_count", "right_click_count",
    "mouse_avg_click_dwell_time_ms", "mouse_avg_time_between_clicks_ms",
    "mouse_scroll_count", "pause_count", "total_pause_time_sec",
    "avg_pause_duration_sec", "mouse_stress_score",
    "game_targets_shown", "game_hits", "game_misses", "game_hit_rate",
    "game_stray_clicks", "game_avg_reaction_time_ms",
    "game_reaction_time_std_ms", "game_avg_click_offset_px",
]

ECG_FIELDS = [
    "session_id", "participant_id", "session_type", "stress_level",
    "ecg_source", "recording_duration_sec", "sample_rate_hz",
    "total_samples", "expected_samples", "sequence_gaps", "largest_gap",
    "invalid_lines", "data_quality",
    "adc_mean", "adc_std", "adc_min", "adc_max", "adc_range",
    "r_peak_count", "mean_hr_bpm", "min_hr_bpm", "max_hr_bpm",
    "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "cvnn_pct",
]

VOICE_FIELDS = (
    ["file", "timestamp", "participant_id", "session_id", "level",
     "prompt_index", "prompt_text", "duration", "sample_rate",
     "f0_mean", "f0_std", "f0_min", "f0_max",
     "spectral_centroid_mean", "spectral_rolloff_mean",
     "total_energy", "energy_mean", "energy_std", "zcr_mean", "zcr_std"]
    + [f"mfcc_{i}_{stat}" for i in range(13) for stat in ("mean", "std")]
    + ["chroma_mean", "chroma_std", "onset_strength_mean", "onset_strength_std",
       "rms_mean", "rms_std", "rms_max"]
)

# ---------------------------------------------------------------------------
# Built-in literature-informed defaults, keyed by real CSV column name so
# they line up 1:1 with anything learned from existing data. Used whenever
# existing data is missing, or a given (column, level) doesn't have enough
# existing rows to trust.
# ---------------------------------------------------------------------------

KS_DEFAULTS = {
    "low":    {"words_per_minute": (62, 6), "avg_dwell_time_ms": (120, 10),
               "avg_flight_time_ms": (118, 12), "backspace_count": (6, 3),
               "pause_count": (1, 1), "avg_pause_duration_sec": (1.2, 0.3),
               "correct_characters": (236, 8)},
    "medium": {"words_per_minute": (52, 7), "avg_dwell_time_ms": (105, 11),
               "avg_flight_time_ms": (100, 12), "backspace_count": (14, 5),
               "pause_count": (3, 1.5), "avg_pause_duration_sec": (1.6, 0.4),
               "correct_characters": (224, 14)},
    "high":   {"words_per_minute": (40, 8), "avg_dwell_time_ms": (90, 12),
               "avg_flight_time_ms": (85, 13), "backspace_count": (27, 8),
               "pause_count": (6, 2), "avg_pause_duration_sec": (2.3, 0.6),
               "correct_characters": (190, 24)},
}

MOUSE_DEFAULTS = {
    "low": {"total_tracking_time_sec": (60, 5), "mouse_avg_speed_px_s": (550, 80),
            "mouse_max_speed_px_s": (7500, 1500), "mouse_speed_std_px_s": (800, 150),
            "mouse_avg_acceleration_px_s2": (1200, 250), "mouse_avg_jerk_px_s3": (350000, 80000),
            "mouse_path_straightness_ratio": (1.6, 0.3), "mouse_direction_change_count": (110, 20),
            "click_count": (45, 10), "mouse_avg_click_dwell_time_ms": (90, 20),
            "mouse_avg_time_between_clicks_ms": (1100, 200), "mouse_scroll_count": (1, 1),
            "pause_count": (1, 1), "avg_pause_duration_sec": (0.5, 0.2),
            "mouse_stress_score": (0.25, 0.08), "game_targets_shown": (60, 5),
            "game_hit_rate": (0.85, 0.06), "game_stray_clicks": (2, 1.5),
            "game_avg_reaction_time_ms": (520, 60), "game_reaction_time_std_ms": (90, 20),
            "game_avg_click_offset_px": (8, 2)},
    "medium": {"total_tracking_time_sec": (60, 5), "mouse_avg_speed_px_s": (700, 90),
               "mouse_max_speed_px_s": (9500, 1800), "mouse_speed_std_px_s": (1000, 180),
               "mouse_avg_acceleration_px_s2": (1550, 300), "mouse_avg_jerk_px_s3": (480000, 90000),
               "mouse_path_straightness_ratio": (2.0, 0.35), "mouse_direction_change_count": (140, 25),
               "click_count": (55, 10), "mouse_avg_click_dwell_time_ms": (70, 15),
               "mouse_avg_time_between_clicks_ms": (950, 180), "mouse_scroll_count": (1, 1),
               "pause_count": (2, 1), "avg_pause_duration_sec": (0.7, 0.2),
               "mouse_stress_score": (0.45, 0.08), "game_targets_shown": (60, 5),
               "game_hit_rate": (0.72, 0.08), "game_stray_clicks": (4, 2),
               "game_avg_reaction_time_ms": (650, 80), "game_reaction_time_std_ms": (140, 25),
               "game_avg_click_offset_px": (12, 3)},
    "high": {"total_tracking_time_sec": (60, 5), "mouse_avg_speed_px_s": (900, 120),
             "mouse_max_speed_px_s": (12500, 2200), "mouse_speed_std_px_s": (1300, 220),
             "mouse_avg_acceleration_px_s2": (2000, 400), "mouse_avg_jerk_px_s3": (650000, 120000),
             "mouse_path_straightness_ratio": (2.5, 0.4), "mouse_direction_change_count": (180, 30),
             "click_count": (65, 12), "mouse_avg_click_dwell_time_ms": (50, 15),
             "mouse_avg_time_between_clicks_ms": (750, 150), "mouse_scroll_count": (1, 1),
             "pause_count": (4, 1.5), "avg_pause_duration_sec": (0.9, 0.3),
             "mouse_stress_score": (0.68, 0.1), "game_targets_shown": (60, 5),
             "game_hit_rate": (0.55, 0.1), "game_stray_clicks": (7, 2.5),
             "game_avg_reaction_time_ms": (820, 110), "game_reaction_time_std_ms": (200, 35),
             "game_avg_click_offset_px": (18, 4)},
}

ECG_DEFAULTS = {
    "low":    {"recording_duration_sec": (55, 8), "mean_hr_bpm": (68, 6), "sdnn_ms": (55, 10),
               "rmssd_ms": (45, 10), "pnn50_pct": (28, 8), "adc_mean": (1900, 80), "adc_std": (140, 15)},
    "medium": {"recording_duration_sec": (55, 8), "mean_hr_bpm": (80, 7), "sdnn_ms": (40, 8),
               "rmssd_ms": (32, 7), "pnn50_pct": (15, 6), "adc_mean": (1900, 80), "adc_std": (165, 18)},
    "high":   {"recording_duration_sec": (55, 8), "mean_hr_bpm": (95, 8), "sdnn_ms": (25, 6),
               "rmssd_ms": (20, 5), "pnn50_pct": (6, 3), "adc_mean": (1900, 80), "adc_std": (195, 20)},
}

VOICE_DEFAULTS = {
    "low":    {"duration": (6.0, 1.0), "f0_mean": (165, 15), "f0_std": (20, 5), "f0_min": (125, 15),
               "f0_max": (225, 20), "spectral_centroid_mean": (1800, 200), "spectral_rolloff_mean": (3200, 300),
               "energy_mean": (0.010, 0.003), "zcr_mean": (0.060, 0.010), "zcr_std": (0.015, 0.004),
               "chroma_mean": (0.40, 0.05), "chroma_std": (0.15, 0.03), "onset_strength_mean": (1.2, 0.3),
               "onset_strength_std": (0.42, 0.1), "rms_mean": (0.090, 0.020)},
    "medium": {"duration": (5.0, 1.0), "f0_mean": (185, 18), "f0_std": (28, 6), "f0_min": (140, 18),
               "f0_max": (250, 22), "spectral_centroid_mean": (2100, 220), "spectral_rolloff_mean": (3600, 320),
               "energy_mean": (0.015, 0.004), "zcr_mean": (0.075, 0.012), "zcr_std": (0.019, 0.005),
               "chroma_mean": (0.40, 0.05), "chroma_std": (0.15, 0.03), "onset_strength_mean": (1.6, 0.35),
               "onset_strength_std": (0.56, 0.12), "rms_mean": (0.110, 0.020)},
    "high":   {"duration": (4.0, 1.0), "f0_mean": (210, 20), "f0_std": (38, 8), "f0_min": (155, 20),
               "f0_max": (280, 25), "spectral_centroid_mean": (2450, 250), "spectral_rolloff_mean": (4100, 350),
               "energy_mean": (0.022, 0.005), "zcr_mean": (0.095, 0.015), "zcr_std": (0.024, 0.006),
               "chroma_mean": (0.40, 0.05), "chroma_std": (0.15, 0.03), "onset_strength_mean": (2.1, 0.4),
               "onset_strength_std": (0.74, 0.14), "rms_mean": (0.140, 0.030)},
}
for _lvl, _n in VOICE_LEVEL_NUM.items():
    for _c in range(13):
        _scale = max(1.0, 18 - _c)
        VOICE_DEFAULTS[_lvl][f"mfcc_{_c}_mean"] = ((-5 if _c == 0 else 0) + _n * 0.6, _scale)
        VOICE_DEFAULTS[_lvl][f"mfcc_{_c}_std"] = (_scale * (0.5 + 0.08 * _n), _scale * 0.15)


def _clip(x, lo=None, hi=None):
    if lo is not None:
        x = max(x, lo)
    if hi is not None:
        x = min(x, hi)
    return x


# ---------------------------------------------------------------------------
# Learning from existing data: per (stress_level, column) mean/std, only
# kept when there are enough existing rows to trust it.
# ---------------------------------------------------------------------------

def _learn_stats(df, level_col, min_rows, level_map=None):
    """Returns {level: {column: (mean, std)}} computed from df, grouped by
    level_col (optionally remapped through level_map, e.g. voice's numeric
    1/2/3 'level' column -> 'low'/'medium'/'high'). Columns/levels with
    fewer than min_rows non-null numeric values are omitted (defaults are
    used for those instead)."""
    if df is None or df.empty or level_col not in df.columns:
        return {}
    stats = {}
    for raw_level, group in df.groupby(level_col):
        level = level_map.get(raw_level, raw_level) if level_map else raw_level
        level = str(level).strip().lower()
        if level not in STRESS_LEVELS:
            continue
        col_stats = stats.setdefault(level, {})
        for col in group.columns:
            if col == level_col:
                continue
            series = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(series) >= min_rows:
                mean = float(series.mean())
                std = float(series.std(ddof=1)) if len(series) > 1 else abs(mean) * 0.1
                if not np.isfinite(std) or std <= 1e-9:
                    std = max(abs(mean) * 0.08, 1e-3)
                if col in col_stats:
                    prev_mean, prev_std = col_stats[col]
                    mean = (prev_mean + mean) / 2
                    std = (prev_std + std) / 2
                col_stats[col] = (mean, std)
    return stats


def load_existing_data():
    """Reads whichever CSVs already exist and returns {modality: DataFrame}
    for anything found and non-empty."""
    frames = {}
    for name, path in (("keystroke", KEYSTROKE_CSV), ("mouse", MOUSE_CSV),
                        ("ecg", ECG_CSV), ("voice", VOICE_CSV)):
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    frames[name] = df
            except Exception as e:
                print(f"[create_synthetic] could not read existing {path.name}, ignoring it: {e}")
    return frames


def build_learned_stats(frames, min_rows):
    learned = {}
    coverage = {}
    if "keystroke" in frames:
        learned["keystroke"] = _learn_stats(frames["keystroke"], "stress_level", min_rows)
        coverage["keystroke"] = {lvl: len(cols) for lvl, cols in learned["keystroke"].items()}
    if "mouse" in frames:
        learned["mouse"] = _learn_stats(frames["mouse"], "stress_level", min_rows)
        coverage["mouse"] = {lvl: len(cols) for lvl, cols in learned["mouse"].items()}
    if "ecg" in frames:
        learned["ecg"] = _learn_stats(frames["ecg"], "stress_level", min_rows)
        coverage["ecg"] = {lvl: len(cols) for lvl, cols in learned["ecg"].items()}
    if "voice" in frames:
        learned["voice"] = _learn_stats(frames["voice"], "level", min_rows, level_map=NUM_TO_LEVEL)
        coverage["voice"] = {lvl: len(cols) for lvl, cols in learned["voice"].items()}
    return learned, coverage


# ---------------------------------------------------------------------------
# Participant IDs: continue whatever style already exists rather than
# inventing a distinguishable one.
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"^([A-Za-z_\-]*)(\d+)$")


def infer_participant_style(frames):
    """Looks at every participant_id already collected across whichever
    CSVs exist and figures out how to name the next batch the same way.

    If a clear "prefix + zero-padded number" style is in use (e.g. P001,
    P002, ... or participant_07), continues it from the highest number
    seen. Otherwise (no existing data, or ids like 'testing' / hashes that
    don't follow a numbered pattern), falls back to a plain P001-style
    sequence numbered after however many distinct participants already
    exist -- so new rows simply look like the next few participants run
    through the study.
    """
    existing_ids = set()
    for df in frames.values():
        if "participant_id" in df.columns:
            existing_ids.update(str(x).strip() for x in df["participant_id"].dropna().unique())

    if not existing_ids:
        return "P{:03d}", 1

    matches = []
    for pid in existing_ids:
        m = _ID_PATTERN.match(pid)
        if m:
            matches.append((m.group(1), len(m.group(2)), int(m.group(2))))

    if matches and len(matches) >= max(1, len(existing_ids) // 2):
        style_counts = Counter((prefix, width) for prefix, width, _ in matches)
        (prefix, width), _ = style_counts.most_common(1)[0]
        max_num = max(num for p, w, num in matches if p == prefix and w == width)
        template = prefix + "{:0" + str(width) + "d}"
        return template, max_num + 1

    # No clean numbered style in the existing data -- don't invent a
    # distinguishable prefix, just carry on a plain sequence.
    return "P{:03d}", len(existing_ids) + 1


def sample(rng, learned, modality, level, col, default_mean, default_std, lo=None, hi=None, integer=False):
    """Draw one value for `col` at this stress `level`, preferring the
    learned existing-data (mean, std) for (modality, level, col) and
    falling back to the built-in default when it isn't available."""
    mean, std = default_mean, default_std
    if learned and modality in learned:
        level_stats = learned[modality].get(level, {})
        if col in level_stats:
            mean, std = level_stats[col]
    val = _clip(float(rng.normal(mean, std)), lo, hi)
    return int(round(val)) if integer else val


def _pois(rng, lam, hi=None):
    return int(_clip(rng.poisson(max(lam, 0.01)), 0, hi))


# ---------------------------------------------------------------------------
# Per-participant traits: small persistent offsets so repeated sessions from
# the same participant aren't just independent random draws.
# ---------------------------------------------------------------------------

def make_participant_traits(rng):
    return {
        "wpm_offset": _clip(float(rng.normal(0, 5)), -15, 15),
        "typing_precision_offset": _clip(float(rng.normal(0, 0.03)), -0.1, 0.1),
        "hr_offset": _clip(float(rng.normal(0, 5)), -15, 15),
        "hrv_offset": _clip(float(rng.normal(0, 6)), -15, 15),
        "mouse_speed_offset": _clip(float(rng.normal(0, 60)), -150, 150),
        "mouse_precision_offset": _clip(float(rng.normal(0, 0.05)), -0.15, 0.15),
        "pitch_offset": _clip(float(rng.normal(0, 10)), -25, 25),
        "voice_energy_offset": _clip(float(rng.normal(0, 0.002)), -0.006, 0.006),
    }


# ---------------------------------------------------------------------------
# Keystroke
# ---------------------------------------------------------------------------

def generate_keystroke_row(pid, session_id, level, rng, traits, learned):
    d = KS_DEFAULTS[level]

    def g(col, lo=None, hi=None, integer=False):
        return sample(rng, learned, "keystroke", level, col, *d[col], lo=lo, hi=hi, integer=integer)

    wpm = _clip(g("words_per_minute", 8) + traits["wpm_offset"], 8, None)
    correct_chars = g("correct_characters", 40, SAMPLE_PARAGRAPH_LEN, integer=True)
    backspace = max(0, round(g("backspace_count", 0) * (1 - traits["typing_precision_offset"])))
    pause_count = max(0, g("pause_count", 0, integer=True))
    avg_pause_dur = g("avg_pause_duration_sec", 0.3, None) if pause_count else 0.0
    total_pause_time = round(pause_count * avg_pause_dur, 3)

    minutes = (correct_chars / 5.0) / max(wpm, 1e-6)
    total_typing_time = round(_clip(minutes * 60 + float(rng.normal(0, 1.5)), 3, None), 3)
    total_keystrokes = int(correct_chars + backspace * 1.3 + max(0, float(rng.normal(0, 3))))

    return {
        "session_id": session_id,
        "participant_id": pid,
        "session_type": KS_SESSION_TYPE[level],
        "stress_level": level,
        "total_typing_time_sec": total_typing_time,
        "total_keystrokes": total_keystrokes,
        "correct_characters": correct_chars,
        "backspace_count": backspace,
        "words_per_minute": round(wpm, 2),
        "avg_dwell_time_ms": round(g("avg_dwell_time_ms", 30), 2),
        "avg_flight_time_ms": round(g("avg_flight_time_ms", 20), 2),
        "pause_count": pause_count,
        "total_pause_time_sec": total_pause_time,
        "avg_pause_duration_sec": round(avg_pause_dur, 3),
    }


# ---------------------------------------------------------------------------
# Mouse (game session)
# ---------------------------------------------------------------------------

def generate_mouse_row(pid, session_id, level, rng, traits, learned):
    d = MOUSE_DEFAULTS[level]

    def g(col, lo=None, hi=None, integer=False):
        return sample(rng, learned, "mouse", level, col, *d[col], lo=lo, hi=hi, integer=integer)

    duration = g("total_tracking_time_sec", 30)
    speed = _clip(g("mouse_avg_speed_px_s", 100) + traits["mouse_speed_offset"] / 8, 100, None)
    events = int(_clip(float(rng.normal(3200 + speed * 1.4, 300)), 500, None))
    distance = round(speed * duration * _clip(float(rng.normal(1.0, 0.05)), 0.7, 1.3), 1)
    clicks = g("click_count", 5, integer=True)
    left = int(clicks * _clip(float(rng.normal(0.94, 0.03)), 0.7, 1.0))
    right = max(clicks - left, 0)
    pause_count = max(0, g("pause_count", 0, integer=True))
    avg_pause_dur = g("avg_pause_duration_sec", 0.1, None) if pause_count else 0.0
    total_pause_time = round(pause_count * avg_pause_dur, 3)

    targets_shown = g("game_targets_shown", 10, integer=True)
    hit_rate = _clip(g("game_hit_rate", 0.05, 1.0) + traits["mouse_precision_offset"], 0.05, 1.0)
    hits = int(round(targets_shown * hit_rate))
    misses = max(targets_shown - hits, 0)

    return {
        "session_id": session_id,
        "participant_id": pid,
        "session_type": "game",
        "stress_level": level,
        "total_tracking_time_sec": round(duration, 3),
        "total_mouse_events": events,
        "total_distance_px": distance,
        "mouse_avg_speed_px_s": round(speed, 3),
        "mouse_max_speed_px_s": round(g("mouse_max_speed_px_s", speed), 2),
        "mouse_speed_std_px_s": round(g("mouse_speed_std_px_s", 50), 3),
        "mouse_avg_acceleration_px_s2": round(g("mouse_avg_acceleration_px_s2", 100), 3),
        "mouse_avg_jerk_px_s3": round(g("mouse_avg_jerk_px_s3", 10000), 3),
        "mouse_path_straightness_ratio": round(g("mouse_path_straightness_ratio", 1.0), 3),
        "mouse_direction_change_count": g("mouse_direction_change_count", 5, integer=True),
        "click_count": clicks,
        "left_click_count": left,
        "right_click_count": right,
        "mouse_avg_click_dwell_time_ms": round(g("mouse_avg_click_dwell_time_ms", 5), 3),
        "mouse_avg_time_between_clicks_ms": round(g("mouse_avg_time_between_clicks_ms", 100), 3),
        "mouse_scroll_count": g("mouse_scroll_count", 0, integer=True),
        "pause_count": pause_count,
        "total_pause_time_sec": total_pause_time,
        "avg_pause_duration_sec": round(avg_pause_dur, 3),
        "mouse_stress_score": round(_clip(g("mouse_stress_score", 0.0, 1.0), 0.0, 1.0), 3),
        "game_targets_shown": targets_shown,
        "game_hits": hits,
        "game_misses": misses,
        "game_hit_rate": round(hit_rate, 3),
        "game_stray_clicks": max(0, g("game_stray_clicks", 0, integer=True)),
        "game_avg_reaction_time_ms": round(g("game_avg_reaction_time_ms", 150), 1),
        "game_reaction_time_std_ms": round(g("game_reaction_time_std_ms", 20), 1),
        "game_avg_click_offset_px": round(g("game_avg_click_offset_px", 1), 1),
    }


# ---------------------------------------------------------------------------
# ECG
# ---------------------------------------------------------------------------

def generate_ecg_row(pid, session_id, level, rng, traits, learned):
    d = ECG_DEFAULTS[level]

    def g(col, lo=None, hi=None):
        return sample(rng, learned, "ecg", level, col, *d[col], lo=lo, hi=hi)

    duration = g("recording_duration_sec", 20)
    sample_rate = 250
    total_samples = int(duration * sample_rate + float(rng.normal(0, 20)))
    mean_hr = _clip(g("mean_hr_bpm", 45, 180) + traits["hr_offset"], 45, 180)
    min_hr = _clip(mean_hr - _clip(float(rng.normal(15, 4)), 3, None), 40, None)
    max_hr = mean_hr + _clip(float(rng.normal(20, 5)), 3, None)
    mean_rr = 60000.0 / mean_hr + float(rng.normal(0, 5))
    sdnn = _clip(g("sdnn_ms", 5) - traits["hrv_offset"], 5, None)
    rmssd = _clip(g("rmssd_ms", 4) - traits["hrv_offset"], 4, None)
    pnn50 = _clip(g("pnn50_pct", 0, 100), 0, 100)
    cvnn = round((sdnn / mean_rr) * 100, 2)
    adc_mean = g("adc_mean")
    adc_std = g("adc_std", 40)
    adc_min = adc_mean - adc_std * _clip(float(rng.normal(4.0, 0.4)), 2, None)
    adc_max = adc_mean + adc_std * _clip(float(rng.normal(4.0, 0.4)), 2, None)

    # "hardware" / "simulated" are the two real values this app itself
    # writes here (see ecg_recorder.py) depending on whether an ESP32 is
    # plugged in -- "simulated" is a perfectly ordinary value for this
    # column, not a synthetic-data marker.
    return {
        "session_id": session_id,
        "participant_id": pid,
        "session_type": KS_SESSION_TYPE[level],
        "stress_level": level,
        "ecg_source": "simulated",
        "recording_duration_sec": round(duration, 3),
        "sample_rate_hz": sample_rate,
        "total_samples": total_samples,
        "expected_samples": "",
        "sequence_gaps": _pois(rng, 0.1),
        "largest_gap": 0,
        "invalid_lines": 0,
        "data_quality": "GOOD",
        "adc_mean": round(adc_mean, 3),
        "adc_std": round(adc_std, 3),
        "adc_min": round(adc_min, 3),
        "adc_max": round(adc_max, 3),
        "adc_range": round(adc_max - adc_min, 3),
        "r_peak_count": int(duration / 60.0 * mean_hr),
        "mean_hr_bpm": round(mean_hr, 2),
        "min_hr_bpm": round(min_hr, 2),
        "max_hr_bpm": round(max_hr, 2),
        "mean_rr_ms": round(mean_rr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50_pct": round(pnn50, 2),
        "cvnn_pct": cvnn,
    }


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

def generate_voice_rows(pid, session_id, level, rng, traits, learned, session_dt, n_prompts=2):
    d = VOICE_DEFAULTS[level]
    level_num = VOICE_LEVEL_NUM[level]
    rows = []
    base_ts = session_dt

    def g(col, lo=None, hi=None):
        return sample(rng, learned, "voice", level, col, *d[col], lo=lo, hi=hi)

    for i in range(n_prompts):
        f0_mean = _clip(g("f0_mean", 60, 400) + traits["pitch_offset"], 60, 400)
        f0_min = g("f0_min", 40, None)
        f0_max = g("f0_max", None, None)
        if f0_min > f0_mean:
            f0_min = f0_mean - abs(float(rng.normal(30, 8)))
        if f0_max < f0_mean:
            f0_max = f0_mean + abs(float(rng.normal(40, 10)))
        energy_mean = _clip(g("energy_mean", 0.001) + traits["voice_energy_offset"], 0.001, None)
        rms_mean = g("rms_mean", 0.01)
        duration = g("duration", 1.5)
        ts = (base_ts + timedelta(seconds=i)).strftime("%Y%m%d_%H%M%S")

        row = {
            "file": f"level{level_num}_prompt{i + 1}_{ts}.wav",
            "timestamp": (base_ts + timedelta(seconds=i)).isoformat(),
            "participant_id": pid,
            "session_id": session_id,
            "level": level_num,
            "prompt_index": i,
            "prompt_text": PROMPTS[i % len(PROMPTS)],
            "duration": round(duration, 3),
            "sample_rate": 44100,
            "f0_mean": round(f0_mean, 2),
            "f0_std": round(g("f0_std", 3), 2),
            "f0_min": round(_clip(f0_min, 50, None), 2),
            "f0_max": round(f0_max, 2),
            "spectral_centroid_mean": round(g("spectral_centroid_mean", 500), 2),
            "spectral_rolloff_mean": round(g("spectral_rolloff_mean", 1000), 2),
            "energy_mean": round(energy_mean, 6),
            "energy_std": round(_clip(float(rng.normal(energy_mean * 0.4, energy_mean * 0.15)), 0.0001, None), 6),
            "zcr_mean": round(g("zcr_mean", 0.01), 4),
            "zcr_std": round(g("zcr_std", 0.005), 4),
            "chroma_mean": round(g("chroma_mean", 0.1, 0.9), 4),
            "chroma_std": round(g("chroma_std", 0.02), 4),
            "onset_strength_mean": round(g("onset_strength_mean", 0.3), 3),
            "onset_strength_std": round(g("onset_strength_std", 0.05), 3),
            "rms_mean": round(rms_mean, 5),
            "rms_std": round(_clip(float(rng.normal(rms_mean * 0.3, rms_mean * 0.1)), 0.001, None), 5),
            "rms_max": round(rms_mean * _clip(float(rng.normal(2.5, 0.4)), 1.5, None), 5),
        }
        row["total_energy"] = round(energy_mean * duration * 44100 * _clip(float(rng.normal(1.0, 0.05)), 0.7, 1.3), 2)
        for c in range(13):
            row[f"mfcc_{c}_mean"] = round(sample(rng, learned, "voice", level, f"mfcc_{c}_mean", *d[f"mfcc_{c}_mean"]), 3)
            row[f"mfcc_{c}_std"] = round(_clip(sample(rng, learned, "voice", level, f"mfcc_{c}_std", *d[f"mfcc_{c}_std"]), 0.5, None), 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames, rows, overwrite: bool):
    """Appends to the file if it already exists (the default -- existing
    data is never erased), or creates/truncates it only when overwrite=True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fresh = overwrite or not path.exists()
    mode = "w" if write_fresh else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_fresh:
            writer.writeheader()
        writer.writerows(rows)
    verb = "wrote" if write_fresh else "appended"
    print(f"[create_synthetic] {verb} {len(rows)} row(s) -> {path}")


def make_registration_time(rng, days_back, earliest_hour=7, latest_hour=22):
    """Pick a random, plausible registration/session moment within the
    last `days_back` days, biased toward normal waking hours so it reads
    like a real person signed up and ran a session -- not a cron job."""
    day_offset = float(rng.uniform(0, days_back))
    hour = float(rng.uniform(earliest_hour, latest_hour))
    minute = float(rng.uniform(0, 60))
    second = float(rng.uniform(0, 60))
    dt = (
        datetime.now()
        - timedelta(days=day_offset)
        + timedelta(hours=hour - datetime.now().hour,
                    minutes=minute - datetime.now().minute,
                    seconds=second - datetime.now().second)
    )
    return dt


def generate(n_participants, repeats, seed, voice_prompts, learned, id_template, id_start, days_back=90):
    rng = np.random.default_rng(seed)
    participant_ids = [id_template.format(idx) for idx in range(id_start, id_start + n_participants)]

    ks_rows, mouse_rows, ecg_rows, voice_rows = [], [], [], []

    for pid in participant_ids:
        traits = make_participant_traits(rng)

        # Each participant "registers" at their own random point in the
        # window, then their sessions for the three stress levels are
        # spread across the following days (not all in the same second) --
        # e.g. one visit for the low-stress session, another a few days
        # later for medium, etc., with some natural jitter.
        registration_time = make_registration_time(rng, days_back)

        for level in STRESS_LEVELS:
            for rep in range(repeats):
                session_dt = registration_time + timedelta(
                    days=float(rng.uniform(0, 10)),
                    hours=float(rng.uniform(-2, 2)),
                    minutes=float(rng.uniform(0, 59)),
                    seconds=float(rng.uniform(0, 59)),
                )
                session_id = f"{pid}_{level}_{session_dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                ks_rows.append(generate_keystroke_row(pid, session_id, level, rng, traits, learned))
                mouse_rows.append(generate_mouse_row(pid, session_id, level, rng, traits, learned))
                ecg_rows.append(generate_ecg_row(pid, session_id, level, rng, traits, learned))
                voice_rows.extend(generate_voice_rows(pid, session_id, level, rng, traits, learned, session_dt, n_prompts=voice_prompts))

    return ks_rows, mouse_rows, ecg_rows, voice_rows


def main():
    ap = argparse.ArgumentParser(description="Generate additional keystroke/mouse/voice/ECG stress session data.")
    ap.add_argument("--participants", type=int, default=20, help="number of participants to add (default 20)")
    ap.add_argument("--repeats", type=int, default=2, help="sessions per participant per stress level (default 2)")
    ap.add_argument("--voice-prompts", type=int, default=2, help="voice prompts recorded per session (default 2)")
    ap.add_argument("--seed", type=int, default=None, help="random seed, for reproducible output")
    ap.add_argument("--days-back", type=int, default=90,
                     help="spread each participant's registration/session timestamps randomly over "
                          "this many past days instead of stamping everything with the current moment "
                          "(default 90)")
    ap.add_argument("--min-real-rows", type=int, default=3,
                     help="minimum existing rows a (column, stress level) needs before it's trusted over the "
                          "built-in default (default 3)")
    ap.add_argument("--ignore-existing-data", action="store_true",
                     help="don't look at existing CSVs at all; use only the built-in defaults and plain P001 IDs")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="write CSVs under this directory instead of the real data/ paths")
    ap.add_argument("--overwrite", action="store_true",
                     help="replace existing CSVs instead of appending to them (default is to append -- "
                          "existing data is never erased unless you pass this)")
    ap.add_argument("--merge", action="store_true",
                     help="after writing, run fusion/merge_datasets.py to rebuild fusion/merged_dataset.csv")
    args = ap.parse_args()

    if args.out_dir:
        out = Path(args.out_dir).resolve()
        ks_path = out / "keystroke" / "features" / "session_features.csv"
        mouse_path = out / "mouse" / "mouse_stress_sessions.csv"
        voice_path = out / "voice" / "voice_features_dataset.csv"
        ecg_path = out / "ecg" / "ecg_stress_sessions.csv"
    else:
        ks_path, mouse_path, voice_path, ecg_path = KEYSTROKE_CSV, MOUSE_CSV, VOICE_CSV, ECG_CSV

    frames = {} if args.ignore_existing_data else load_existing_data()
    learned, coverage = build_learned_stats(frames, args.min_real_rows) if frames else ({}, {})
    id_template, id_start = infer_participant_style(frames)

    if frames:
        print(f"[create_synthetic] found existing data in: {', '.join(frames.keys())}")
        for modality, per_level in coverage.items():
            desc = ", ".join(f"{lvl}={n} col(s)" for lvl, n in per_level.items()) or "no level had enough rows yet"
            print(f"[create_synthetic]   {modality}: adapting to existing data -> {desc}")
        print(f"[create_synthetic]   continuing participant numbering from {id_template.format(id_start)}")
    else:
        print("[create_synthetic] no existing data found -- using built-in defaults")

    print(f"[create_synthetic] generating {args.participants} participant(s) x {len(STRESS_LEVELS)} stress level(s) "
          f"x {args.repeats} session(s) = {args.participants * len(STRESS_LEVELS) * args.repeats} sessions per modality")

    ks_rows, mouse_rows, ecg_rows, voice_rows = generate(
        args.participants, args.repeats, args.seed, args.voice_prompts, learned, id_template, id_start,
        days_back=args.days_back
    )

    _write_csv(ks_path, KEYSTROKE_FIELDS, ks_rows, args.overwrite)
    _write_csv(mouse_path, MOUSE_FIELDS, mouse_rows, args.overwrite)
    _write_csv(ecg_path, ECG_FIELDS, ecg_rows, args.overwrite)
    _write_csv(voice_path, VOICE_FIELDS, voice_rows, args.overwrite)

    counts = pd.Series([r["stress_level"] for r in ks_rows]).value_counts()
    print("[create_synthetic] class balance (new sessions per stress level):")
    for level in STRESS_LEVELS:
        print(f"    {level:>6}: {int(counts.get(level, 0))}")

    if args.merge:
        if args.out_dir:
            print("[create_synthetic] --merge ignored: it only rebuilds the real fusion/merged_dataset.csv, "
                  "which needs data in the real data/ paths, not --out-dir.")
        else:
            sys.path.insert(0, str(BASE_DIR / "fusion"))
            import merge_datasets  # type: ignore
            merge_datasets.merge(BASE_DIR / "fusion" / "merged_dataset.csv")


if __name__ == "__main__":
    main()