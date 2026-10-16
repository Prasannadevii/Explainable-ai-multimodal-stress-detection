"""
consolidated_report.py

The final "Consolidated Findings" step from the architecture diagram:
runs a session through BOTH approaches --

    Approach A: Decision-Level Fusion  (decision_fusion.py)
    Approach B: Feature-Level Fusion   (train_tabnet_model.py)

-- explains each with SHAP + LIME (explain.py), and writes one report
comparing classification performance and each modality's relative
contribution to the stress prediction under both approaches, plus a
plain-language explanation of why each approach reached its verdict.

Usage:
    # after collecting sessions + merge_datasets.py + training both
    # decision_fusion.py and train_tabnet_model.py:
    python consolidated_report.py --latest
    python consolidated_report.py --csv one_session.csv

Prints the report and saves it to fusion/reports/<session_id>_report.md.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import decision_fusion as dfmod
import explain as ex
import predict as pr
import modality_utils as mu

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "fusion" / "reports"


def _load_metrics(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _probs_to_sentence(probs: dict) -> str:
    if not probs:
        return ""
    parts = [f"{level} {p:.0%}" for level, p in probs.items()]
    return ", ".join(parts)


def _shap_to_sentence(shap_row, pred_class, top_n=5):
    if not shap_row:
        return ""
    ranked = sorted(shap_row.items(), key=lambda t: -abs(t[1]))[:top_n]
    bits = []
    for name, val in ranked:
        modality = mu.modality_of_column(name) or "other"
        direction = "toward" if val > 0 else "away from"
        bits.append(f"{name} ({modality} data, {direction} {pred_class})")
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + ", and " + bits[-1]


def generate_report(row: pd.Series) -> str:
    session_id = row.get("session_id", "unknown")
    participant_id = row.get("participant_id", "unknown")
    when = datetime.now().strftime("%B %d, %Y at %I:%M %p").lstrip("0").replace(" 0", " ")

    paragraphs = []
    paragraphs.append(
        f"Stress report for session {session_id}, participant {participant_id}. "
        f"Generated {when}."
    )

    # -- Feature-level fusion (single TabNet model) --------------------------
    b_artifacts = pr.TABNET_ARTIFACTS_DIR
    b_metrics = _load_metrics(b_artifacts / "metrics.json")
    b_explained = None
    if b_metrics is None:
        paragraphs.append(
            "The feature-level model (a single TabNet classifier trained on all "
            f"modalities' features concatenated together) hasn't been trained yet -- "
            f"no artifacts were found in {b_artifacts}. Run the merge step and then "
            "train the feature fusion model before this section can be filled in."
        )
    else:
        b_explained = ex.explain_feature_fusion(row)
        contrib = b_metrics.get("modality_contribution", {})
        contrib_sentence = ", ".join(f"{m} {s:.0%}" for m, s in contrib.items()) if contrib else "not available"
        paragraphs.append(
            f"The feature-level model is a {b_metrics.get('model_type', 'tabnet')} classifier "
            f"trained on {b_metrics.get('n_samples')} sessions across {b_metrics.get('n_classes')} "
            f"stress classes, with a training accuracy of {b_metrics.get('cv_accuracy'):.2f} "
            f"({b_metrics.get('cv_method')}). Across all the training data, the modalities it "
            f"leaned on most were {contrib_sentence}."
        )
        pred_b = b_explained["predicted_stress_level"]
        probs_sentence = _probs_to_sentence(b_explained["probabilities"])
        paragraphs.append(
            f"For this specific session, that model predicted {pred_b} stress "
            f"(probabilities: {probs_sentence})."
        )
        shap_sentence = _shap_to_sentence(b_explained.get("shap_values"), pred_b)
        if shap_sentence:
            shap_contrib = b_explained.get("modality_contribution_shap", {})
            shap_contrib_sentence = ", ".join(f"{m} {s:.0%}" for m, s in shap_contrib.items())
            paragraphs.append(
                f"Looking at this session alone rather than the training set as a whole, "
                f"the modalities that actually swayed this particular prediction were "
                f"{shap_contrib_sentence}. The individual features that mattered most were "
                f"{shap_sentence}."
            )
        elif b_explained.get("shap_error"):
            paragraphs.append(
                f"A per-session breakdown of which features drove this prediction wasn't "
                f"available this time ({b_explained['shap_error']})."
            )
        if b_explained.get("lime_result"):
            lime_bits = ", ".join(f"{feat} ({weight:+.2f})" for feat, weight in b_explained["lime_result"][:5])
            paragraphs.append(f"A second, independent check (LIME) pointed to similar features: {lime_bits}.")

    # -- Decision-level fusion (independent per-modality models) -------------
    a_metrics = _load_metrics(dfmod.ARTIFACTS_DIR / "metrics.json")
    a_explained = None
    if a_metrics is None:
        paragraphs.append(
            "The decision-level model, which trains a separate model per modality and "
            f"then combines their votes, hasn't been trained yet -- no artifacts were "
            f"found in {dfmod.ARTIFACTS_DIR}. Run the merge step and then train decision "
            "fusion before this section can be filled in."
        )
    else:
        trained = ", ".join(a_metrics.get("trained_modalities", [])) or "none"
        per_mod_acc = a_metrics.get("per_modality_cv_accuracy", {})
        acc_sentence = ", ".join(f"{m} {a:.2f}" for m, a in per_mod_acc.items()) if per_mod_acc else "not available"
        combined_acc = a_metrics.get("combined_accuracy")
        combined_sentence = f"{combined_acc:.2f}" if combined_acc is not None else "not enough data yet to measure"
        paragraphs.append(
            f"The decision-level approach trained separate models per modality ({trained}), "
            f"each scoring {acc_sentence} on its own, and combines them into an ensemble "
            f"accuracy of {combined_sentence}."
        )
        a_explained = ex.explain_decision_fusion(row)
        pred_a = a_explained["predicted_stress_level"]
        if pred_a:
            probs_sentence_a = _probs_to_sentence(a_explained["probabilities"])
            paragraphs.append(
                f"For this session, the combined vote came out to {pred_a} stress "
                f"(probabilities: {probs_sentence_a})."
            )
            vote_bits = []
            for modality, out in a_explained["modality_outputs"].items():
                if out is None:
                    vote_bits.append(f"{modality} had no data for this session")
                    continue
                w = a_explained["modality_weights_used"].get(modality, 0.0)
                vote_bits.append(f"{modality} voted {out['predicted_stress_level']} (weighted {w:.0%})")
            if vote_bits:
                paragraphs.append("Per modality: " + "; ".join(vote_bits) + ".")
        else:
            paragraphs.append("This session didn't have enough data for the combined model to produce a vote.")

    # -- Bottom line -----------------------------------------------------------
    if b_explained is None or a_explained is None or a_explained.get("predicted_stress_level") is None:
        paragraphs.append(
            "Only one of the two approaches could be evaluated for this session (see above) -- "
            "train the other model and re-run to get a full comparison."
        )
    else:
        pred_b = b_explained["predicted_stress_level"]
        pred_a = a_explained["predicted_stress_level"]
        if pred_a == pred_b:
            paragraphs.append(f"Both approaches agree on this one: {pred_b} stress.")
        else:
            paragraphs.append(
                f"The two approaches don't agree here -- feature-level fusion says {pred_b}, "
                f"decision-level fusion says {pred_a}. Worth treating this session's reading with "
                f"some caution; a single noisy or missing modality is usually the reason for a "
                f"split like this, so it's worth checking the per-modality breakdown above."
            )

    return "\n\n".join(paragraphs)


def save_report(text: str, session_id) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = str(session_id).replace("/", "_").replace("\\", "_")
    path = REPORTS_DIR / f"{safe_id}_report.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate the consolidated Approach A + Approach B + SHAP/LIME report."
    )
    p.add_argument("--latest", action="store_true", help="Use the most recent session across all modalities")
    p.add_argument("--csv", help="CSV with one merged row to report on (uses the last row)")
    args = p.parse_args()

    if args.csv:
        row = pd.read_csv(args.csv).iloc[-1]
    elif args.latest:
        row = pr.get_latest_row()
    else:
        raise SystemExit("Specify --latest or --csv <file>.")

    report_text = generate_report(row)
    print(report_text)
    saved_path = save_report(report_text, row.get("session_id", "session"))
    print(f"\n[report] Saved to {saved_path}")
