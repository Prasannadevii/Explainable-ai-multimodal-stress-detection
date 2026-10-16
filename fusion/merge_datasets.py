"""
merge_datasets.py

Joins the four modalities into a single multimodal feature table:

  - keystroke_research/data/features/session_features.csv
  - mouse/mouse_stress_sessions.csv
  - voice/voice_features_dataset.csv
  - ecg/data/ecg/ecg_stress_sessions.csv

Two join strategies are used, in order of preference:

1. EXACT session_id join (best). If you collected data with
   run_combined_session.py, keystroke/mouse/ECG rows for the same task
   instance share one session_id (ECG is recorded automatically in the
   background for every keystroke/mouse/voice session -- see
   run_combined_session.py), so they join directly on it. Voice rows
   share session_id too when the voice GUI was launched with
   --session-id from the same combined run.

2. FALLBACK join on (normalized participant_id, stress_level). Older /
   independently-collected data (like the sample CSVs shipped with this
   project) usually was NOT collected with a shared session_id -- e.g.
   keystroke IDs look like "P-354c862d", mouse IDs look like "1001",
   voice IDs look like "1". For those rows, this script normalizes each
   ID to its trailing alphanumeric token and groups by
   (participant_id, stress_level), averaging numeric features within
   each group before joining across modalities. This produces one
   fused row per participant per condition rather than a true
   moment-by-moment join -- treat it as a coarse fallback, not as
   precise as the session_id join.

Output: fusion/merged_dataset.csv (one row per fused session/group,
with a `source_modalities` column telling you which of
keystroke/mouse/voice/ecg contributed to that row).

Usage:
    python merge_datasets.py
    python merge_datasets.py --output my_merged.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
KEYSTROKE_CSV = BASE_DIR / "data" / "keystroke" / "features" / "session_features.csv"
MOUSE_CSV = BASE_DIR / "data" / "mouse" / "mouse_stress_sessions.csv"
VOICE_CSV = BASE_DIR / "data" / "voice" / "voice_features_dataset.csv"
ECG_CSV = BASE_DIR / "data" / "ecg" / "ecg_stress_sessions.csv"

LEVEL_TO_STRESS = {1: "low", 2: "medium", 3: "high", "1": "low", "2": "medium", "3": "high"}

ID_COLUMNS = {"session_id", "participant_id", "session_type", "stress_level"}


def _norm_participant(pid) -> str:
    """Normalize a participant id like 'P-354c862d', '1001', 1 -> a bare token."""
    s = str(pid).strip()
    s = re.sub(r"^[Pp][_-]?", "", s)
    return s.lower()


def _load_keystroke():
    if not KEYSTROKE_CSV.exists():
        print(f"[merge] keystroke file not found, skipping: {KEYSTROKE_CSV}")
        return None
    df = pd.read_csv(KEYSTROKE_CSV)
    if df.empty:
        return None
    df["participant_norm"] = df["participant_id"].map(_norm_participant)
    df = df.add_prefix("ks_")
    df = df.rename(columns={
        "ks_session_id": "session_id",
        "ks_participant_id": "participant_id",
        "ks_stress_level": "stress_level",
        "ks_participant_norm": "participant_norm",
    })
    return df


def _load_mouse():
    if not MOUSE_CSV.exists():
        print(f"[merge] mouse file not found, skipping: {MOUSE_CSV}")
        return None
    df = pd.read_csv(MOUSE_CSV)
    if df.empty:
        return None
    df["participant_norm"] = df["participant_id"].map(_norm_participant)
    df = df.add_prefix("mouse_")
    df = df.rename(columns={
        "mouse_session_id": "session_id",
        "mouse_participant_id": "participant_id",
        "mouse_stress_level": "stress_level",
        "mouse_participant_norm": "participant_norm",
    })
    return df


def _load_voice():
    if not VOICE_CSV.exists():
        print(f"[merge] voice file not found, skipping: {VOICE_CSV}")
        return None
    df = pd.read_csv(VOICE_CSV)
    if df.empty:
        return None

    df["stress_level"] = df["level"].map(LEVEL_TO_STRESS)
    df["participant_norm"] = df["participant_id"].map(_norm_participant)

    # Multiple prompts can exist per (participant, session_id, level); average
    # the numeric acoustic features down to one row per group first.
    group_cols = ["participant_id", "participant_norm", "session_id", "level", "stress_level"]
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in group_cols]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    agg = agg.add_prefix("voice_")
    agg = agg.rename(columns={
        "voice_session_id": "session_id",
        "voice_participant_id": "participant_id",
        "voice_stress_level": "stress_level",
        "voice_participant_norm": "participant_norm",
        "voice_level": "voice_level",
    })
    return agg


def _load_ecg():
    if not ECG_CSV.exists():
        print(f"[merge] ecg file not found, skipping: {ECG_CSV}")
        return None
    df = pd.read_csv(ECG_CSV)
    if df.empty:
        return None
    df["participant_norm"] = df["participant_id"].map(_norm_participant)
    df = df.add_prefix("ecg_")
    df = df.rename(columns={
        "ecg_session_id": "session_id",
        "ecg_participant_id": "participant_id",
        "ecg_stress_level": "stress_level",
        "ecg_participant_norm": "participant_norm",
    })
    return df


def _join_on_session_id(ks, mouse, voice, ecg=None):
    """Try the precise join first: exact session_id match across modalities."""
    frames = [f for f in (ks, mouse, voice, ecg) if f is not None]
    if not frames:
        return None

    shared_id_cols = {"participant_id", "participant_norm", "stress_level", "session_type"}

    merged = frames[0]
    for f in frames[1:]:
        # keep these identifier columns only from the left (first-seen) frame
        f = f.drop(columns=[c for c in shared_id_cols if c in f.columns], errors="ignore")
        merged = merged.merge(f, on="session_id", how="inner")
    return merged if not merged.empty else None


def _join_on_participant_condition(ks, mouse, voice, ecg=None):
    """Fallback join: group each modality by (participant_norm, stress_level)
    and average, then outer-join across modalities on that composite key."""
    frames = {}
    if ks is not None:
        frames["keystroke"] = ks
    if mouse is not None:
        frames["mouse"] = mouse
    if voice is not None:
        frames["voice"] = voice
    if ecg is not None:
        frames["ecg"] = ecg

    if not frames:
        return None

    grouped = {}
    for name, df in frames.items():
        numeric_cols = [c for c in df.select_dtypes(include="number").columns]
        g = df.groupby(["participant_norm", "stress_level"], dropna=False)[numeric_cols].mean().reset_index()
        grouped[name] = g

    names = list(grouped.keys())
    merged = grouped[names[0]]
    merged["source_modalities"] = names[0]
    for name in names[1:]:
        merged = merged.merge(
            grouped[name], on=["participant_norm", "stress_level"],
            how="outer", suffixes=("", f"_{name}")
        )
    return merged


def merge(output_path: Path):
    ks = _load_keystroke()
    mouse = _load_mouse()
    voice = _load_voice()
    ecg = _load_ecg()

    if ks is None and mouse is None and voice is None and ecg is None:
        print("[merge] No modality data found at all. Nothing to merge.")
        return

    precise = _join_on_session_id(ks, mouse, voice, ecg)
    if precise is not None and len(precise) > 0:
        print(f"[merge] Exact session_id join produced {len(precise)} row(s).")
        precise["source_modalities"] = "+".join(
            n for n, f in (("keystroke", ks), ("mouse", mouse), ("voice", voice), ("ecg", ecg))
            if f is not None
        )
        result = precise
    else:
        print("[merge] No overlapping session_id found across modalities.")
        print("[merge] Falling back to (participant_id, stress_level) grouping.")
        result = _join_on_participant_condition(ks, mouse, voice, ecg)

    if result is None or result.empty:
        print("[merge] Could not produce any merged rows.")
        return

    # stress_level is the training target -- drop rows where it's missing
    result = result.dropna(subset=["stress_level"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[merge] Wrote {len(result)} row(s), {len(result.columns)} column(s) -> {output_path}")
    print("[merge] Columns:", list(result.columns))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge keystroke + mouse + voice feature CSVs.")
    p.add_argument("--output", default=str(BASE_DIR / "fusion" / "merged_dataset.csv"))
    args = p.parse_args()
    merge(Path(args.output))
