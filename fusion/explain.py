"""
explain.py

The "Model Interpretability (SHAP / LIME)" step from the architecture
diagram. Takes a prediction from either approach and explains WHY the
model reached that decision -- per-feature and per-modality
contributions, plus a plain-language narrative -- instead of just
handing back a class label.

Two explainers, used together (not either/or):
    - SHAP: exact TreeExplainer for the Random Forest models used by
      Approach A's per-modality submodels; model-agnostic
      KernelExplainer (bounded sample count, so it stays fast) for the
      Approach B TabNet model, since TabNet has no dedicated fast
      SHAP explainer.
    - LIME: `lime.lime_tabular.LimeTabularExplainer`, model-agnostic,
      used identically for every model type via its `predict_proba`.

Both are computed per-instance (i.e. for one specific session), which
is what makes them different from -- and complementary to -- the
global `feature_importances_`-based "modality contribution" that
train_tabnet_model.py / decision_fusion.py compute once at training time.

Requires: pip install shap lime
If either library is missing, or an explainer fails for any reason
(e.g. TabNet's KernelExplainer timing out on a very wide feature
space), that half of the explanation is skipped with a clear error
message rather than crashing the whole prediction -- you still get the
class label and probabilities either way.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import modality_utils as mu

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fill_row(row, background: pd.DataFrame, feature_cols):
    """Turns a (possibly NaN-containing) session row into a fully numeric
    vector over `feature_cols`, filling any gaps with that column's
    median across the background dataset -- SHAP/LIME both need a
    concrete numeric instance to explain, not a missing value."""
    medians = background[feature_cols].median(numeric_only=True)
    values = []
    for c in feature_cols:
        v = row.get(c, np.nan) if hasattr(row, "get") else row[c]
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if pd.isna(v):
            v = float(medians.get(c, 0.0)) if not pd.isna(medians.get(c, np.nan)) else 0.0
        values.append(v)
    return np.array(values, dtype=float)


def _background_matrix(background: pd.DataFrame, feature_cols, max_rows=25, random_state=42):
    bg = background[feature_cols].apply(lambda col: col.fillna(col.median()))
    bg = bg.fillna(0.0)  # columns that are all-NaN in the background: last resort
    if len(bg) > max_rows:
        bg = bg.sample(n=max_rows, random_state=random_state)
    if len(bg) == 0:
        return np.zeros((1, len(feature_cols)))
    return bg.to_numpy(dtype=float)


def _extract_class_shap(raw_shap, class_idx):
    """Normalizes the several shapes shap's API has returned across
    versions for multiclass output down to one 1-D array over features
    for the requested class."""
    if isinstance(raw_shap, list):
        arr = np.asarray(raw_shap[class_idx])
        return arr[0] if arr.ndim == 2 else np.asarray(arr).ravel()
    arr = np.asarray(raw_shap)
    if arr.ndim == 3:       # (n_samples, n_features, n_classes)
        return arr[0, :, class_idx]
    if arr.ndim == 2:       # (n_samples, n_features) -- single-output case
        return arr[0]
    return arr.ravel()


def _narrative(approach_label, pred_class, shap_row: dict, top_n=5):
    lines = [f"Predicted stress level ({approach_label}): {str(pred_class).upper()}."]
    if not shap_row:
        lines.append("(Per-feature explanation unavailable for this session -- see shap_error.)")
        return "\n".join(lines)
    ranked = sorted(shap_row.items(), key=lambda t: -abs(t[1]))[:top_n]
    lines.append("Top factors driving this decision (SHAP):")
    for name, val in ranked:
        modality = mu.modality_of_column(name) or "other"
        direction = f"pushed TOWARD {pred_class}" if val > 0 else f"pushed AWAY FROM {pred_class}"
        lines.append(f"  - [{modality}] {name}: {direction} (SHAP={val:+.3f})")
    return "\n".join(lines)


def _modality_contribution_from_shap(shap_row: dict):
    if not shap_row:
        return {}
    totals = defaultdict(float)
    for col, val in shap_row.items():
        totals[mu.modality_of_column(col) or "other"] += abs(val)
    total = sum(totals.values()) or 1.0
    return {m: round(v / total, 4) for m, v in sorted(totals.items(), key=lambda t: -t[1])}


# ---------------------------------------------------------------------------
# Approach B (feature-level fusion): explain the single unified model
# ---------------------------------------------------------------------------
def explain_feature_fusion(row, artifacts_dir=None, background_csv=None, top_n=10):
    import predict as pr

    artifacts_dir = Path(artifacts_dir) if artifacts_dir else pr.TABNET_ARTIFACTS_DIR
    model, feature_cols, classes = pr.load_model(artifacts_dir)
    background = mu.load_merged(background_csv)

    x_row = _fill_row(row, background, feature_cols).reshape(1, -1)
    proba = model.predict_proba(x_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = classes[pred_idx]
    probabilities = {classes[i]: round(float(proba[i]), 3) for i in range(len(classes))}

    bg_matrix = _background_matrix(background, feature_cols)

    shap_row, shap_error = None, None
    try:
        import shap
        explainer = shap.KernelExplainer(model.predict_proba, bg_matrix)
        raw = explainer.shap_values(x_row, nsamples=100, silent=True)
        shap_row = {c: float(v) for c, v in zip(feature_cols, _extract_class_shap(raw, pred_idx))}
    except Exception as e:
        shap_error = f"{type(e).__name__}: {e}"

    lime_result, lime_error = None, None
    try:
        import lime.lime_tabular
        bg_full = background[feature_cols].apply(lambda col: col.fillna(col.median())).fillna(0.0)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            bg_full.to_numpy(dtype=float), feature_names=feature_cols,
            class_names=classes, mode="classification", discretize_continuous=True,
        )
        exp = explainer.explain_instance(
            x_row[0], model.predict_proba,
            num_features=min(top_n, len(feature_cols)), labels=(pred_idx,),
        )
        lime_result = exp.as_list(label=pred_idx)
    except Exception as e:
        lime_error = f"{type(e).__name__}: {e}"

    return {
        "approach": "B: Feature-Level Fusion",
        "model_type": "tabnet",
        "predicted_stress_level": pred_class,
        "probabilities": probabilities,
        "shap_values": shap_row,
        "shap_error": shap_error,
        "modality_contribution_shap": _modality_contribution_from_shap(shap_row),
        "lime_result": lime_result,
        "lime_error": lime_error,
        "narrative": _narrative("Approach B - Feature-Level Fusion", pred_class, shap_row),
    }


# ---------------------------------------------------------------------------
# Approach A (decision-level fusion): explain each modality submodel,
# then show how they were weighted into the combined decision.
# ---------------------------------------------------------------------------
def explain_decision_fusion(row, artifacts_dir=None, background_csv=None, top_n=8):
    import decision_fusion as dfmod
    import shap  # imported eagerly here so a missing install fails once, clearly

    artifacts_dir = Path(artifacts_dir) if artifacts_dir else dfmod.ARTIFACTS_DIR
    submodels, weights, classes = dfmod.load_submodels(artifacts_dir)
    background = mu.load_merged(background_csv)
    combined = dfmod.combine_predict_row(row, submodels, weights, classes)

    shap_per_modality = {}
    lime_per_modality = {}

    for modality, sm in submodels.items():
        out = combined["modality_outputs"].get(modality)
        if out is None:
            continue
        cols = sm["feature_columns"]
        pred_idx = classes.index(out["predicted_stress_level"])  # global class index
        x_row = _fill_row(row, background, cols).reshape(1, -1)
        bg_matrix = _background_matrix(background, cols)
        pipeline = sm["pipeline"]

        # This submodel may only have been trained on a subset of the
        # global classes (see decision_fusion.combine_predict_row), so
        # its own predict_proba / SHAP output is indexed by *local*
        # position within clf.classes_, not the global class index.
        # Map pred_idx into that local space before pulling SHAP/LIME
        # values for it.
        clf_classes = list(pipeline.named_steps["clf"].classes_)
        if pred_idx not in clf_classes:
            # Shouldn't happen (pred_idx came from this same submodel's
            # own argmax), but skip cleanly instead of crashing if it does.
            shap_per_modality[modality] = {
                "__error__": f"predicted class not in this submodel's trained classes {clf_classes}"
            }
            lime_per_modality[modality] = [("__error__", "predicted class unavailable for this submodel")]
            continue
        local_idx = clf_classes.index(pred_idx)

        try:
            x_t_row = pipeline.named_steps["scale"].transform(
                pipeline.named_steps["impute"].transform(x_row))
            x_t_bg = pipeline.named_steps["scale"].transform(
                pipeline.named_steps["impute"].transform(bg_matrix))
            explainer = shap.TreeExplainer(pipeline.named_steps["clf"], x_t_bg)
            raw = explainer.shap_values(x_t_row)
            shap_per_modality[modality] = {
                c: float(v) for c, v in zip(cols, _extract_class_shap(raw, local_idx))
            }
        except Exception as e:
            shap_per_modality[modality] = {"__error__": f"{type(e).__name__}: {e}"}

        try:
            import lime.lime_tabular
            bg_full = background[cols].apply(lambda col: col.fillna(col.median())).fillna(0.0)
            local_class_names = [classes[i] for i in clf_classes]
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                bg_full.to_numpy(dtype=float), feature_names=cols,
                class_names=local_class_names, mode="classification", discretize_continuous=True,
            )
            exp = lime_explainer.explain_instance(
                x_row[0], pipeline.predict_proba,
                num_features=min(top_n, len(cols)), labels=(local_idx,),
            )
            lime_per_modality[modality] = exp.as_list(label=local_idx)
        except Exception as e:
            lime_per_modality[modality] = [("__error__", f"{type(e).__name__}: {e}")]

    narrative_lines = [
        f"Predicted stress level (Approach A - Decision-Level Fusion): "
        f"{str(combined['predicted_stress_level']).upper()}.",
        "Each modality voted independently (its own model, its own features), "
        "weighted by that modality's own cross-validated accuracy:",
    ]
    for modality, out in combined["modality_outputs"].items():
        if out is None:
            narrative_lines.append(f"  - {modality}: not available for this session")
            continue
        w = combined["modality_weights_used"].get(modality, 0.0)
        narrative_lines.append(
            f"  - {modality}: voted {out['predicted_stress_level'].upper()} "
            f"(vote weight {w:.0%}, own accuracy {weights.get(modality, 0):.0%})"
        )
        shap_row = shap_per_modality.get(modality, {})
        if shap_row and "__error__" not in shap_row:
            ranked = sorted(shap_row.items(), key=lambda t: -abs(t[1]))[:3]
            feat_str = ", ".join(f"{n} ({v:+.2f})" for n, v in ranked)
            narrative_lines.append(f"      top factors: {feat_str}")

    return {
        "approach": "A: Decision-Level Fusion",
        "predicted_stress_level": combined["predicted_stress_level"],
        "probabilities": combined["probabilities"],
        "modality_outputs": combined["modality_outputs"],
        "modality_weights_used": combined["modality_weights_used"],
        "shap_per_modality": shap_per_modality,
        "lime_per_modality": lime_per_modality,
        "narrative": "\n".join(narrative_lines),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Explain one session's prediction with SHAP + LIME.")
    p.add_argument("--csv", help="CSV with one merged row to explain (uses the last row)")
    p.add_argument("--approach", choices=["a", "b", "both"], default="both")
    args = p.parse_args()

    if not args.csv:
        raise SystemExit("Specify --csv <file with one merged row> (see predict.py --latest "
                          "for the equivalent 'most recent session' lookup used by "
                          "consolidated_report.py).")

    df = pd.read_csv(args.csv)
    row = df.iloc[-1]

    if args.approach in ("b", "both"):
        result_b = explain_feature_fusion(row)
        print(result_b["narrative"])
        if result_b["lime_result"]:
            print("\nLIME (Approach B):")
            for feat, w in result_b["lime_result"]:
                print(f"  {feat}: {w:+.3f}")
        print()

    if args.approach in ("a", "both"):
        result_a = explain_decision_fusion(row)
        print(result_a["narrative"])
