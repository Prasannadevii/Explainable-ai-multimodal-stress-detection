"""
predict.py

Loads the trained TabNet model from fusion/model_artifacts_tabnet/
(trained by train_tabnet_model.py) and predicts a stress level
(low / medium / high) for new session data.

Two ways to use it:

1. From a CSV of new merged rows (same shape as merged_dataset.csv):
       python predict.py --csv new_sessions.csv

2. Directly on the most recent row of each raw source file (keystroke /
   mouse / voice / ECG), doing a quick on-the-fly merge for just that
   latest session (handy right after running run_combined_session.py):
       python predict.py --latest

This script only runs Approach B (the single feature-fusion TabNet
model). For Approach A (decision-level fusion, one model per modality)
use decision_fusion.py's own prediction path. For a WHY-flavored
answer (SHAP + LIME, both approaches side by side) use explain.py or
consolidated_report.py instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TABNET_ARTIFACTS_DIR = BASE_DIR / "fusion" / "model_artifacts_tabnet"


class _TabNetModel:
    def __init__(self, artifacts_dir: Path):
        import joblib
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.imputer = joblib.load(artifacts_dir / "imputer.joblib")
        self.scaler = joblib.load(artifacts_dir / "scaler.joblib")
        self.clf = TabNetClassifier()
        self.clf.load_model(str(artifacts_dir / "stress_model.zip"))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.clf.predict_proba(X)


def load_model(artifacts_dir: Path = TABNET_ARTIFACTS_DIR):
    cols_path = artifacts_dir / "feature_columns.json"
    classes_path = artifacts_dir / "label_classes.json"
    model_file = artifacts_dir / "stress_model.zip"

    if not model_file.exists():
        raise SystemExit(f"No trained model found at {model_file}. "
                          f"Run merge_datasets.py then train_tabnet_model.py first.")

    model = _TabNetModel(artifacts_dir)
    feature_cols = json.loads(cols_path.read_text())
    classes = json.loads(classes_path.read_text())
    return model, feature_cols, classes


def predict_rows(df: pd.DataFrame, model, feature_cols, classes):
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan  # let the imputer fill columns this session didn't have

    X = df[feature_cols].to_numpy(dtype=float)
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)

    results = []
    for i in range(len(df)):
        row_result = {classes[j]: round(float(proba[i][j]), 3) for j in range(len(classes))}
        results.append({
            "predicted_stress_level": classes[pred_idx[i]],
            "probabilities": row_result,
        })
    return results


def predict_from_csv(csv_path: Path, artifacts_dir: Path = TABNET_ARTIFACTS_DIR):
    model, feature_cols, classes = load_model(artifacts_dir)
    df = pd.read_csv(csv_path)
    results = predict_rows(df, model, feature_cols, classes)
    for i, r in enumerate(results):
        print(f"\nRow {i}: predicted stress level = {r['predicted_stress_level'].upper()}")
        print("  probabilities:", r["probabilities"])


def get_latest_row():
    """Re-runs the merge for just the newest session across all
    modalities and returns it as a pandas Series. Shared by
    predict_latest() below and by explain.py / consolidated_report.py
    so "the latest session" always means the same row everywhere."""
    import merge_datasets as md

    ks = md._load_keystroke()
    mouse = md._load_mouse()
    voice = md._load_voice()
    ecg = md._load_ecg()

    frames = [f for f in (ks, mouse, voice, ecg) if f is not None]
    if not frames:
        raise SystemExit("No session data found in keystroke/mouse/voice/ecg CSVs yet.")

    precise = md._join_on_session_id(ks, mouse, voice, ecg)
    if precise is not None and len(precise) > 0:
        return precise.iloc[-1]

    fallback = md._join_on_participant_condition(ks, mouse, voice, ecg)
    if fallback is None or fallback.empty:
        raise SystemExit("Could not build a row for the latest session.")
    return fallback.iloc[-1]


def predict_latest(artifacts_dir: Path = TABNET_ARTIFACTS_DIR):
    """Re-run the merge for just the newest session and score it."""
    latest = get_latest_row()
    model, feature_cols, classes = load_model(artifacts_dir)
    results = predict_rows(pd.DataFrame([latest]), model, feature_cols, classes)
    r = results[0]
    print(f"\nLatest session -> predicted stress level: {r['predicted_stress_level'].upper()}")
    print("Probabilities:", r["probabilities"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict stress level from session features (TabNet).")
    p.add_argument("--csv", help="Path to a CSV of merged session rows to score")
    p.add_argument("--latest", action="store_true", help="Score the most recent session across all modalities")
    p.add_argument("--artifacts-dir", default=str(TABNET_ARTIFACTS_DIR))
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    if args.csv:
        predict_from_csv(Path(args.csv), artifacts_dir)
    elif args.latest:
        predict_latest(artifacts_dir)
    else:
        raise SystemExit("Specify --csv <file> or --latest.")
