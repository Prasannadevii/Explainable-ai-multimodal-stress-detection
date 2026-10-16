"""
decision_fusion.py

Approach A from the architecture diagram: Decision-Level Fusion
(Independent Models).

    Typing / Speech / Mouse / Sensor(ECG)
            |  (each trained separately, on only its own features)
       Model: Typing   Model: Speech   Model: Mouse   Model: Sensor
            \\_______________|_______________|_______________/
                                    |
                    Combine Individual Model Outputs
                                    |
        Output A: Stress Classification + Modality Contribution

Contrast with train_tabnet_model.py (Approach B:
Feature-Level Fusion), which concatenate every modality's features into
one vector and train a single model on the combined table.

Usage:
    python merge_datasets.py     # build fusion/merged_dataset.csv first
    python decision_fusion.py    # trains one model per modality + combines them

Saves to fusion/model_artifacts_decision/:
    <modality>/stress_model.joblib, feature_columns.json, label_classes.json,
        metrics.json   -- one submodel per modality (only for modalities
                           with enough data to train + cross-validate)
    metrics.json        -- combined-fusion accuracy/report + each
                           modality's contribution weight (its own CV
                           accuracy, normalized) -- read by
                           consolidated_report.py and used by explain.py
                           as the Output-A modality-contribution figure
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                      cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import modality_utils as mu

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_CSV = BASE_DIR / "fusion" / "merged_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "fusion" / "model_artifacts_decision"


def build_pipeline(n_estimators=300, random_state=42):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def _subset_for_modality(df: pd.DataFrame, modality: str):
    """Rows + columns for one modality: drop columns that are entirely
    NaN within this modality, then drop rows that have no data left for
    any of the remaining columns (i.e. this modality wasn't collected
    for that row)."""
    cols = mu.modality_feature_columns(df, modality)
    cols = [c for c in cols if not df[c].isna().all()]
    if not cols:
        return None, []
    sub = df[["stress_level"] + cols]
    row_mask = ~sub[cols].isna().all(axis=1)
    sub = sub[row_mask]
    return sub, cols


def _cross_validated_report(pipeline, X, y, class_names):
    """Same small-vs-big-dataset CV strategy as train_tabnet_model.py, returns
    (accuracy, report_dict, cv_method, y_pred)."""
    n_samples = len(y)
    min_class_count = pd.Series(y).value_counts().min()
    if n_samples < 8 or min_class_count < 2:
        cv_method = "leave-one-out"
        y_pred = cross_val_predict(pipeline, X, y, cv=LeaveOneOut())
    else:
        n_splits = min(5, min_class_count)
        cv_method = f"{n_splits}-fold stratified cross-validation"
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, labels=list(range(len(class_names))),
                                    target_names=class_names, zero_division=0,
                                    output_dict=True)
    return acc, report, cv_method, y_pred


def train_one_modality(df: pd.DataFrame, modality: str, global_encoder: LabelEncoder,
                        artifacts_dir: Path):
    """Trains and saves one modality's submodel. Returns a dict with its
    own metrics (used both to weight it in the combined decision and to
    report its standalone performance), or None if there wasn't enough
    data for this modality to train on."""
    sub, feature_cols = _subset_for_modality(df, modality)
    if sub is None or len(sub) < 4:
        print(f"[decision] {modality}: not enough data to train a submodel "
              f"(need >=4 rows with this modality's features present, "
              f"have {0 if sub is None else len(sub)}) -- skipping.")
        return None

    y = global_encoder.transform(sub["stress_level"].astype(str))
    class_counts = pd.Series(y).value_counts()
    if class_counts.min() < 1 or len(class_counts) < 2:
        print(f"[decision] {modality}: only {len(class_counts)} class(es) present in its "
              f"{len(sub)} available rows -- need at least 2 to train a classifier, skipping.")
        return None

    X = sub[feature_cols].to_numpy(dtype=float)
    pipeline = build_pipeline()
    acc, report, cv_method, y_pred = _cross_validated_report(
        pipeline, X, y, list(global_encoder.classes_)
    )
    print(f"[decision] {modality}: {len(sub)} samples, {len(feature_cols)} features, "
          f"{cv_method} accuracy = {acc:.2f}")

    pipeline.fit(X, y)  # final fit on all available rows for this modality

    modality_dir = artifacts_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, modality_dir / "stress_model.joblib")
    with open(modality_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(modality_dir / "label_classes.json", "w") as f:
        json.dump(list(global_encoder.classes_), f, indent=2)

    # Out-of-fold probabilities, keyed by original row index, ONLY when
    # every global class is present in this modality's subset -- that's
    # what makes cross_val_predict's predict_proba columns line up with
    # the global class order consistently across folds. When a class is
    # entirely missing from this modality's data, we still train + save
    # the submodel (it's useful on its own), we just can't fairly fold
    # it into the row-by-row combined-accuracy evaluation below.
    oof_proba_by_row = {}
    if len(class_counts) == len(global_encoder.classes_) and class_counts.min() >= 2 \
            and len(sub) >= len(global_encoder.classes_) * 2:
        n_splits = min(5, int(class_counts.min()))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")
        for row_id, proba in zip(sub.index, oof_proba):
            oof_proba_by_row[row_id] = proba

    metrics = {
        "modality": modality,
        "n_samples": len(sub),
        "n_features": len(feature_cols),
        "cv_method": cv_method,
        "cv_accuracy": round(float(acc), 4),
        "classification_report": report,
        "feature_columns": feature_cols,
    }
    with open(modality_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {"metrics": metrics, "oof_proba_by_row": oof_proba_by_row}


def train(merged_csv: Path = MERGED_CSV, artifacts_dir: Path = ARTIFACTS_DIR):
    df = mu.load_merged(merged_csv)
    if df.empty:
        raise SystemExit("Merged dataset has no labeled rows to train on.")

    global_encoder = LabelEncoder()
    global_encoder.fit(df["stress_level"].astype(str))
    class_names = list(global_encoder.classes_)
    print(f"[decision] Global classes: {class_names}")

    modalities = mu.available_modalities(df)
    if not modalities:
        raise SystemExit("No modality has any usable feature columns in the merged dataset.")

    per_modality = {}
    for modality in modalities:
        result = train_one_modality(df, modality, global_encoder, artifacts_dir)
        if result is not None:
            per_modality[modality] = result

    if not per_modality:
        raise SystemExit("No modality had enough data to train a submodel. "
                          "Collect more sessions before running decision_fusion.py.")

    # -- combine: weighted average of each available modality's
    # out-of-fold probabilities per row, weight = that modality's own
    # cross-validated accuracy (renormalized across whichever modalities
    # actually have an entry for that specific row).
    weights = {m: per_modality[m]["metrics"]["cv_accuracy"] for m in per_modality}
    n_classes = len(class_names)

    combined_true, combined_pred, evaluated_rows = [], [], 0
    for row_id, true_label in zip(df.index, global_encoder.transform(df["stress_level"].astype(str))):
        contributions = []
        for m, result in per_modality.items():
            proba = result["oof_proba_by_row"].get(row_id)
            if proba is not None:
                contributions.append((weights[m], proba))
        if not contributions:
            continue
        total_w = sum(w for w, _ in contributions) or 1.0
        combined_proba = np.zeros(n_classes)
        for w, proba in contributions:
            combined_proba += (w / total_w) * proba
        combined_true.append(true_label)
        combined_pred.append(int(np.argmax(combined_proba)))
        evaluated_rows += 1

    if evaluated_rows > 0:
        combined_acc = accuracy_score(combined_true, combined_pred)
        combined_report = classification_report(
            combined_true, combined_pred, labels=list(range(n_classes)),
            target_names=class_names, zero_division=0, output_dict=True
        )
        print(f"\n[decision] Combined (fused) accuracy across {evaluated_rows} row(s) "
              f"with cross-validated out-of-fold predictions available: {combined_acc:.2f}")
        print(classification_report(combined_true, combined_pred, labels=list(range(n_classes)),
                                     target_names=class_names, zero_division=0))
    else:
        combined_acc, combined_report = None, None
        print("\n[decision] Not enough data yet for a fair row-by-row combined-accuracy "
              "estimate (needs modalities whose subsets cover every class with >=2 "
              "samples each). Per-modality submodels above are still trained and usable; "
              "collect more sessions per condition to unlock this comparison.")

    total_weight = sum(weights.values()) or 1.0
    modality_contribution = {m: round(w / total_weight, 4) for m, w in
                              sorted(weights.items(), key=lambda t: -t[1])}
    print("\n[decision] Modality contribution (normalized submodel CV accuracy -- "
          "how much each modality's vote counts in the combined decision):")
    for m, share in modality_contribution.items():
        print(f"  {m:12s} {share:.1%}  (own accuracy: {weights[m]:.2f})")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    overall_metrics = {
        "approach": "A: Decision-Level Fusion (independent per-modality models, combined)",
        "trained_modalities": list(per_modality.keys()),
        "classes": class_names,
        "per_modality_cv_accuracy": {m: per_modality[m]["metrics"]["cv_accuracy"] for m in per_modality},
        "modality_contribution": modality_contribution,
        "combined_rows_evaluated": evaluated_rows,
        "combined_accuracy": round(float(combined_acc), 4) if combined_acc is not None else None,
        "combined_classification_report": combined_report,
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"\n[decision] Saved {len(per_modality)} submodel(s) + combined metrics to {artifacts_dir}/")


# ---------------------------------------------------------------------------
# Inference: combine individual modality models' outputs for one new row.
# ---------------------------------------------------------------------------
def load_submodels(artifacts_dir: Path = ARTIFACTS_DIR):
    """Loads every trained modality submodel + the combined-decision
    weights. Returns (submodels dict, weights dict, class_names)."""
    overall_path = artifacts_dir / "metrics.json"
    if not overall_path.exists():
        raise SystemExit(f"No trained decision-fusion models found at {artifacts_dir}. "
                          "Run decision_fusion.py first.")
    overall = json.loads(overall_path.read_text())
    class_names = overall["classes"]
    weights = overall["modality_contribution"]

    submodels = {}
    for modality in overall["trained_modalities"]:
        mdir = artifacts_dir / modality
        pipeline = joblib.load(mdir / "stress_model.joblib")
        feature_cols = json.loads((mdir / "feature_columns.json").read_text())
        submodels[modality] = {"pipeline": pipeline, "feature_columns": feature_cols}
    return submodels, weights, class_names


def combine_predict_row(row: pd.Series, submodels: dict, weights: dict, class_names: list):
    """Runs every modality's own submodel on `row` (a pandas Series /
    dict-like with merged_dataset.csv-style columns), then combines
    their individual predictions into one decision. Returns a dict with
    the combined result AND each modality's own individual output --
    exactly "Combine Individual Model Outputs" -> "Output A" from the
    architecture diagram."""
    n_classes = len(class_names)
    per_modality_output = {}
    contributions = []

    for modality, sm in submodels.items():
        cols = sm["feature_columns"]
        values = pd.Series({c: row.get(c, np.nan) for c in cols}, dtype=float)
        if values.isna().all():
            per_modality_output[modality] = None  # this modality wasn't collected for this row
            continue
        X = values.to_numpy(dtype=float).reshape(1, -1)
        raw_proba = sm["pipeline"].predict_proba(X)[0]

        # A submodel may have been fit on a subset of rows that didn't
        # cover every global class (small datasets especially), so its
        # predict_proba columns only cover the classes it actually saw --
        # sm["pipeline"].named_steps["clf"].classes_ gives their global
        # label indices in the order predict_proba returns them. Expand
        # back out to the full global class vector (0 for classes this
        # submodel never saw) so every modality's output lines up on the
        # same n_classes axis before we combine or index into class_names.
        clf_classes = sm["pipeline"].named_steps["clf"].classes_
        proba = np.zeros(n_classes)
        for local_idx, global_idx in enumerate(clf_classes):
            proba[int(global_idx)] = raw_proba[local_idx]

        pred_idx = int(np.argmax(proba))
        per_modality_output[modality] = {
            "predicted_stress_level": class_names[pred_idx],
            "probabilities": {class_names[i]: round(float(proba[i]), 3) for i in range(n_classes)},
        }
        w = weights.get(modality, 0.0)
        contributions.append((modality, w, proba))

    if not contributions:
        return {
            "predicted_stress_level": None,
            "probabilities": None,
            "modality_outputs": per_modality_output,
            "modality_weights_used": {},
        }

    total_w = sum(w for _, w, _ in contributions) or 1.0
    combined_proba = np.zeros(n_classes)
    weights_used = {}
    for modality, w, proba in contributions:
        norm_w = w / total_w
        combined_proba += norm_w * proba
        weights_used[modality] = round(norm_w, 4)

    pred_idx = int(np.argmax(combined_proba))
    return {
        "predicted_stress_level": class_names[pred_idx],
        "probabilities": {class_names[i]: round(float(combined_proba[i]), 3) for i in range(n_classes)},
        "modality_outputs": per_modality_output,
        "modality_weights_used": weights_used,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Approach A: Decision-Level Fusion "
                                             "(one model per modality, combined).")
    p.add_argument("--merged-csv", default=str(MERGED_CSV))
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    args = p.parse_args()
    train(Path(args.merged_csv), Path(args.artifacts_dir))
