"""
One-time repair script for an existing session_features.csv that was
corrupted by a schema change (columns added after some rows were
already written under an older header).

Place this file directly in the same folder as main.py, config.py,
etc. (the keystroke_research folder), then run it from there:

    python fix_features_csv.py

This is a standalone convenience script with no dependency on the rest
of the package. You don't strictly need it -- the next time the app
itself calls save_features() (i.e. the next time you complete a
session), the file is repaired automatically by storage.py. Run this
only if you want the fix applied immediately without running a new
session first.
"""

import csv
from pathlib import Path

# Current SessionFeatures schema, in order. Keep this in sync with
# models.py if new fields are ever added.
CURRENT_FIELDNAMES = [
    "session_id",
    "participant_id",
    "session_type",
    "stress_level",
    "total_typing_time_sec",
    "total_keystrokes",
    "correct_characters",
    "backspace_count",
    "words_per_minute",
    "avg_dwell_time_ms",
    "avg_flight_time_ms",
    "pause_count",
    "total_pause_time_sec",
    "avg_pause_duration_sec",
]

FEATURES_PATH = Path(__file__).resolve().parent / "data" / "features" / "session_features.csv"


def _load_and_normalize_rows(path: Path, fieldnames: list) -> list:
    """
    Read every existing row and normalize it to the CURRENT schema,
    regardless of which schema version it was originally written under.
    """
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
            normalized.append({f: old_row.get(f, "unknown") for f in fieldnames})
        else:
            partial = dict(zip(fieldnames, values))
            for f in fieldnames:
                partial.setdefault(f, "unknown")
            normalized.append(partial)

    return normalized


def main() -> None:
    if not FEATURES_PATH.exists():
        print(f"No file found at {FEATURES_PATH} -- nothing to repair.")
        return

    rows = _load_and_normalize_rows(FEATURES_PATH, CURRENT_FIELDNAMES)

    if not rows:
        print("File exists but has no data rows -- nothing to repair.")
        return

    with open(FEATURES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURRENT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Repaired {len(rows)} row(s) in {FEATURES_PATH}")
    print(f"Header is now: {', '.join(CURRENT_FIELDNAMES)}")
    print(
        "Rows written before session_type/stress_level existed were "
        "filled with 'unknown' for those two columns."
    )


if __name__ == "__main__":
    main()