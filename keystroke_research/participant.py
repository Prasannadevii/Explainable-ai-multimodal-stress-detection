"""
Participant management.

Handles creation and lookup of participant IDs, stored in a simple CSV
file so the project has no external database dependency by default.
Swapping this out for SQLite later would only require changing the
functions in this module.
"""

import csv
import uuid
from datetime import datetime
from typing import List, Optional

from . import config


def _ensure_participants_file() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.PARTICIPANTS_FILE.exists():
        with open(config.PARTICIPANTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["participant_id", "created_at"])


def create_participant(participant_id: Optional[str] = None) -> str:
    """
    Register a new participant and return their participant_id.

    If participant_id is not supplied, a short unique ID is generated.
    If the given ID already exists, it is reused rather than duplicated.
    """
    _ensure_participants_file()

    if not participant_id:
        participant_id = f"P-{uuid.uuid4().hex[:8]}"

    if participant_exists(participant_id):
        return participant_id

    with open(config.PARTICIPANTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([participant_id, datetime.now().isoformat(timespec="seconds")])

    return participant_id


def participant_exists(participant_id: str) -> bool:
    """Return True if participant_id is already registered."""
    _ensure_participants_file()
    with open(config.PARTICIPANTS_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(row["participant_id"] == participant_id for row in reader)


def list_participants() -> List[dict]:
    """Return all registered participants as a list of row dicts."""
    _ensure_participants_file()
    with open(config.PARTICIPANTS_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)