"""
train_tabnet_model.py

Trains a TabNet stress-level classifier (low / medium / high) on the
merged multimodal feature table produced by merge_datasets.py. TabNet
(Arik & Pfister, 2019) is a deep attentive-transformer architecture
built for exactly this kind of tabular, multi-source feature table, and
its per-sample feature masks (`explain()`) double as an interpretable
"which modality drove this prediction" view alongside the Random
Forest's feature_importances_.

Usage:
    python merge_datasets.py          # build fusion/merged_dataset.csv first
    python train_tabnet_model.py      # then train

Saves to fusion/model_artifacts_tabnet/:
    - stress_model.zip       (pytorch-tabnet TabNetClassifier.save_model())
    - feature_columns.json   (exact column order the model expects)
    - label_classes.json     (class name <-> index mapping)

Requires: pip install torch pytorch-tabnet
On a tiny dataset (the norm early in data collection -- see
the same caveat) TabNet is trained with a small batch
size / few epochs and no held-out validation split; treat its accuracy
number as illustrative, not a real generalization estimate, until you
have dozens of labeled sessions per class.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import modality_utils as mu

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_CSV = BASE_DIR / "fusion" / "merged_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "fusion" / "model_artifacts_tabnet"

# Kept identical across modalities so the model sees the same feature
# space and are directly comparable.
NON_FEATURE_COLUMNS = mu.NON_FEATURE_COLUMNS


def _select_feature_columns(df: pd.DataFrame):
    return mu.all_feature_columns(df)


def _import_tabnet():
    try:
        import torch  # noqa: F401
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as e:
        raise SystemExit(
            "TabNet requires torch + pytorch-tabnet. Install them with:\n\n"
            "    pip install torch pytorch-tabnet\n\n"
            f"(import failed with: {e})"
        )
    return TabNetClassifier


def train(merged_csv: Path, artifacts_dir: Path,
          max_epochs: int = 200, patience: int = 20, random_state: int = 42):
    TabNetClassifier = _import_tabnet()

    if not merged_csv.exists():
        raise SystemExit(
            f"{merged_csv} not found. Run merge_datasets.py first to build it."
        )

    df = pd.read_csv(merged_csv)
    df = df.dropna(subset=["stress_level"])
    if df.empty:
        raise SystemExit("Merged dataset has no labeled rows to train on.")

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise SystemExit("No numeric feature columns found in the merged dataset.")

    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"[tabnet] Dropping {len(all_nan_cols)} column(s) with no observed values "
              f"in this dataset (e.g. a modality you haven't collected yet): {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    if not feature_cols:
        raise SystemExit("No usable feature columns remain after dropping empty ones.")

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y_raw = df["stress_level"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    n_samples = len(df)
    n_classes = len(encoder.classes_)
    print(f"[tabnet] {n_samples} samples, {len(feature_cols)} features, "
          f"{n_classes} classes: {list(encoder.classes_)}")

    # TabNet doesn't have a built-in imputer/scaler like the sklearn
    # Pipeline used for Random Forest, so we do it explicitly and save
    # the fitted imputer/scaler alongside the model for use at inference.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(X_raw)
    X = scaler.fit_transform(X)

    min_class_count = pd.Series(y).value_counts().min()
    # A plain 80/20 train_test_split can fail outright on small multi-class
    # datasets (e.g. 9 samples / 3 classes -> a 20% test split is smaller
    # than the number of classes, which sklearn rejects for a stratified
    # split). Use k-fold cross-validation instead whenever there isn't
    # comfortably enough data per class for a held-out split to make sense.
    small_dataset = n_samples < 8 or min_class_count < 2
    can_kfold = min_class_count >= 2

    def _batch_sizes(n):
        # TabNet's batch/virtual-batch sizes must be <= the number of
        # training rows. virtual_batch_size must also evenly divide
        # batch_size -- otherwise Ghost Batch Norm hands its last chunk
        # (whatever's left over) to BatchNorm, and a chunk of size 1
        # makes BatchNorm raise ("Expected more than 1 value per
        # channel"). Try common divisors first; falling back to
        # virtual_batch_size == batch_size (no chunking at all) is
        # always safe as long as batch_size >= 2, which matters a lot
        # for the tiny datasets this tool sees early in data collection.
        batch = max(2, min(256, n))
        vbatch = batch
        for cand in (128, 64, 32, 16, 8, 4, 2):
            if cand <= batch and batch % cand == 0:
                vbatch = cand
                break
        return batch, vbatch

    def _new_clf():
        return TabNetClassifier(seed=random_state, verbose=0)

    if small_dataset or not can_kfold:
        print("[tabnet] Very small dataset -- training on all data with no held-out "
              "validation split. Treat the reported training accuracy as illustrative "
              "only, not a generalization estimate; collect more labeled sessions "
              "before trusting this number.")
        cv_method = "none (trained + evaluated on all data -- illustrative only)"
        batch_size, virtual_batch_size = _batch_sizes(n_samples)
        clf = _new_clf()
        clf.fit(
            X, y,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        )
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"[tabnet] Training accuracy (not cross-validated): {acc:.2f}")
        report_dict = classification_report(y, y_pred, target_names=encoder.classes_,
                                             zero_division=0, output_dict=True)
        print(classification_report(y, y_pred, target_names=encoder.classes_, zero_division=0))
    else:
        # k-fold cross-validation (same idea as decision_fusion.py's
        # StratifiedKFold + cross_val_predict), done as an explicit loop
        # since a fresh TabNet needs training per fold.
        n_splits = min(5, min_class_count)
        cv_method = f"{n_splits}-fold stratified cross-validation"
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_pred = np.empty_like(y)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            batch_size, virtual_batch_size = _batch_sizes(len(X_tr))
            fold_clf = _new_clf()
            fold_clf.fit(
                X_tr, y_tr,
                max_epochs=max_epochs, patience=patience,
                batch_size=batch_size, virtual_batch_size=virtual_batch_size,
            )
            y_pred[test_idx] = fold_clf.predict(X_te)
            print(f"[tabnet] fold {fold_i}/{n_splits} done")

        acc = accuracy_score(y, y_pred)
        print(f"[tabnet] {n_splits}-fold cross-validated accuracy: {acc:.2f}")
        report_dict = classification_report(y, y_pred, target_names=encoder.classes_,
                                             zero_division=0, output_dict=True)
        print(classification_report(y, y_pred, target_names=encoder.classes_, zero_division=0))

        # Refit on all data for the deployed model.
        batch_size, virtual_batch_size = _batch_sizes(n_samples)
        clf = _new_clf()
        clf.fit(X, y, max_epochs=max_epochs, patience=patience,
                batch_size=batch_size, virtual_batch_size=virtual_batch_size)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(artifacts_dir / "stress_model"))  # writes stress_model.zip
    with open(artifacts_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(artifacts_dir / "label_classes.json", "w") as f:
        json.dump(list(encoder.classes_), f, indent=2)

    import joblib
    joblib.dump(imputer, artifacts_dir / "imputer.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")

    modality_contribution = {}
    try:
        importances = clf.feature_importances_
        top = sorted(zip(feature_cols, importances), key=lambda t: -t[1])[:15]
        print("\n[tabnet] Top features by importance:")
        for name, imp in top:
            print(f"  {name:40s} {imp:.4f}")

        modality_totals = defaultdict(float)
        for name, imp in zip(feature_cols, importances):
            modality = mu.modality_of_column(name) or "other"
            modality_totals[modality] += float(imp)
        total = sum(modality_totals.values()) or 1.0
        modality_contribution = {k: round(v / total, 4) for k, v in
                                  sorted(modality_totals.items(), key=lambda t: -t[1])}
        print("\n[tabnet] Modality contribution (share of total feature importance):")
        for name, share in modality_contribution.items():
            print(f"  {name:12s} {share:.1%}")
    except Exception:
        pass

    metrics = {
        "approach": "B: Feature-Level Fusion (single model on concatenated features)",
        "model_type": "tabnet",
        "n_samples": n_samples,
        "n_classes": n_classes,
        "classes": list(encoder.classes_),
        "cv_method": cv_method,
        "cv_accuracy": round(float(acc), 4),
        "classification_report": report_dict,
        "modality_contribution": modality_contribution,
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[tabnet] Saved model + metadata to {artifacts_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the multimodal stress-level classifier with TabNet.")
    p.add_argument("--merged-csv", default=str(MERGED_CSV))
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()
    train(Path(args.merged_csv), Path(args.artifacts_dir),
          max_epochs=args.max_epochs, patience=args.patience)
