"""
CSV-based storage for raw keystroke events and derived session features.

Kept behind small, explicit functions so SQLite (or any other backend)
could be swapped in later without changing calling code in
session_runner.py.

save_features is self-healing with respect to schema changes: if the
SessionFeatures dataclass gains or loses fields over time (as it did
when session_type/stress_level were added), old rows already on disk
are normalized to the CURRENT schema every time a new row is saved,
rather than being appended under a stale header and corrupting the
file. Any column that didn't exist when an old row was written is
filled with "unknown".
"""

import csv
from dataclasses import asdict
from typing import Dict, List

from . import config
from .models import KeyEvent, SessionFeatures


def _ensure_dirs() -> None:
    config.RAW_KEYSTROKES_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_events(events: List[KeyEvent]) -> str:
    """Write raw key events for one session to their own CSV file.

    Returns the path to the written file.
    """
    if not events:
        raise ValueError("No events to save.")

    _ensure_dirs()
    session_id = events[0].session_id
    path = config.RAW_KEYSTROKES_DIR / f"{session_id}_raw.csv"

    fieldnames = list(asdict(events[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            writer.writerow(asdict(e))

    return str(path)


def _load_and_normalize_rows(path, fieldnames: List[str]) -> List[Dict[str, str]]:
    """
    Read every existing row in the features CSV and normalize it to the
    CURRENT schema (fieldnames), regardless of which schema version it
    was originally written under.

    Handles three cases per data row:
    - Row already has the current number of columns -> read as-is.
    - Row matches the file's OLD header (fewer/different columns) ->
      map by the old header's names, filling any new columns with
      "unknown".
    - Anything else unexpected -> best-effort positional match, filling
      remaining columns with "unknown" rather than dropping the row.
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        all_lines = list(csv.reader(f))

    if not all_lines:
        return []

    header, data_lines = all_lines[0], all_lines[1:]
    normalized: List[Dict[str, str]] = []

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


def save_features(features: SessionFeatures) -> str:
    """Save one session's derived features to the shared features CSV.

    Rewrites the file with a consistent header each time, normalizing
    any pre-existing rows to the current schema first (see
    _load_and_normalize_rows). Returns the path to the features file.
    """
    _ensure_dirs()
    path = config.FEATURES_DIR / "session_features.csv"

    new_row = asdict(features)
    fieldnames = list(new_row.keys())

    rows: List[Dict[str, str]] = []
    if path.exists():
        rows = _load_and_normalize_rows(path, fieldnames)

    rows.append(new_row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(path)