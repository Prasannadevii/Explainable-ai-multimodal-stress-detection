"""
modality_utils.py

Shared helpers used by both fusion approaches (decision_fusion.py /
train_tabnet_model.py) and the explainability layer
(explain.py). Centralizes the one thing that has to stay consistent
everywhere: which merged_dataset.csv column belongs to which modality.

merge_datasets.py prefixes every feature column with its source
modality (ks_ / mouse_ / voice_ / ecg_), so modality membership is
just a prefix check -- but keeping that mapping in one place means
decision_fusion.py, explain.py, and consolidated_report.py can never
drift out of sync with each other.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_CSV = BASE_DIR / "fusion" / "merged_dataset.csv"

# Display name -> column prefix in merged_dataset.csv. Order here is the
# canonical modality order used throughout the fusion/explainability code
# (e.g. Approach A trains one model per entry, in this order).
MODALITY_PREFIXES = {
    "keystroke": "ks_",
    "mouse": "mouse_",
    "voice": "voice_",
    "ecg": "ecg_",
}

# Columns that are identifiers/labels/metadata, never model features,
# regardless of which modality (or none) they came from. Kept identical
# to train_tabnet_model.py's NON_FEATURE_COLUMNS so Approach A and Approach B
# always agree on what counts as a feature.
NON_FEATURE_COLUMNS = {
    "session_id", "participant_id", "participant_norm", "stress_level",
    "session_type", "source_modalities", "voice_file", "voice_timestamp",
    "voice_prompt_text", "voice_level", "ecg_source", "ecg_data_quality",
    "voice_level_ecg",
}


def load_merged(merged_csv: Path = None) -> pd.DataFrame:
    merged_csv = Path(merged_csv or MERGED_CSV)
    if not merged_csv.exists():
        raise SystemExit(f"{merged_csv} not found. Run merge_datasets.py first to build it.")
    df = pd.read_csv(merged_csv)
    return df.dropna(subset=["stress_level"])


def all_feature_columns(df: pd.DataFrame):
    """Every numeric, non-identifier column -- i.e. Approach B's full
    concatenated feature vector.

    Uses pd.api.types.is_numeric_dtype rather than `dtype == object` to
    detect non-feature text columns: modern pandas (with the pyarrow
    string backend, or future.infer_string) reports string columns as a
    dedicated StringDtype whose repr is "str", NOT "object" -- a plain
    `== object` check silently lets those columns through and crashes
    downstream at `.to_numpy(dtype=float)`."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def modality_feature_columns(df: pd.DataFrame, modality: str):
    """Numeric feature columns belonging to a single modality (by its
    ks_/mouse_/voice_/ecg_ prefix) -- i.e. one Approach A submodel's
    input space."""
    prefix = MODALITY_PREFIXES[modality]
    return [c for c in all_feature_columns(df) if c.startswith(prefix)]


def modality_of_column(col: str):
    """Reverse lookup: which modality a merged_dataset.csv feature column
    belongs to, or None if it's not a recognized modality-prefixed
    feature. Used by explain.py to roll per-feature SHAP/LIME values up
    into a per-modality contribution breakdown."""
    for name, prefix in MODALITY_PREFIXES.items():
        if col.startswith(prefix):
            return name
    return None


def available_modalities(df: pd.DataFrame):
    """Modalities that actually have at least one non-empty feature
    column in this merged dataset (a modality you haven't collected yet
    won't have survived merge_datasets.py's all-NaN column drop, or
    train_tabnet_model.py's, but check directly here too)."""
    out = []
    for name in MODALITY_PREFIXES:
        cols = modality_feature_columns(df, name)
        cols = [c for c in cols if not df[c].isna().all()]
        if cols:
            out.append(name)
    return out
