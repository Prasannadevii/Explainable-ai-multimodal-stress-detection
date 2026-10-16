"""
Configuration constants for the ECG data-collection module.

Centralizing configuration here mirrors keystroke_research/config.py so
the four modalities (keystroke, mouse, voice, ECG) stay easy to reason
about as one family of tools.
"""

import os
from pathlib import Path

# --- Directory layout ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "ecg"

RAW_DIR = DATA_DIR / "raw"
FEATURES_CSV = DATA_DIR / "ecg_stress_sessions.csv"

# --- Serial / hardware -------------------------------------------------
#
# Overridable via environment variables so different machines/OSes don't
# require editing source: e.g.
#   ECG_PORT=/dev/ttyUSB0 ECG_BAUD=921600 python run_combined_session.py

PORT = os.environ.get("ECG_PORT", "COM7")
BAUD = int(os.environ.get("ECG_BAUD", "921600"))
SAMPLE_RATE = int(os.environ.get("ECG_SAMPLE_RATE", "250"))

# How long to wait for the ESP32 to settle after opening the serial port.
SERIAL_WARMUP_SEC = 1.0

# --- Session conditions --------------------------------------------------
#
# Mirrors keystroke_research's SESSION_TYPES: each ECG condition carries a
# ground-truth "stress_level" label (low/medium/high) for the condition
# the participant was assigned to, plus a default duration. These are
# used by the standalone ECG GUI/CLI. When ECG recording is triggered
# automatically alongside a keystroke/mouse/voice session (see
# run_combined_session.py), the recording instead runs for exactly as
# long as that other tool's window stays open, and the condition/
# stress_level are backfilled from that tool's own saved row when
# possible.

CONDITIONS = {
    "1": {
        "key": "baseline",
        "label": "Baseline (Rest)",
        "stress_level": "low",
        "default_duration_sec": 300,
        "intro": "Sit comfortably. Relax. Do not talk. Avoid unnecessary movement.",
    },
    "2": {
        "key": "monkeytype",
        "label": "Timed Monkeytype",
        "stress_level": "high",
        "default_duration_sec": 300,
        "intro": "Perform the timed Monkeytype task.",
    },
    "3": {
        "key": "recovery",
        "label": "Recovery",
        "stress_level": "medium",
        "default_duration_sec": 300,
        "intro": "Sit comfortably. Relax after the task.",
    },
}

# --- ADC / signal validation ---------------------------------------------

ADC_MIN = 0
ADC_MAX = 4095

# --- Misc ------------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
