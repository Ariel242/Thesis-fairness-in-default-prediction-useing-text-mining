# -*- coding: utf-8 -*-
"""
05_compute_interrater_reliability.py
======================================
Step 3 (validation half) of the Distress Lexicon protocol (see
Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf, section 2,
"Human-in-the-Loop Validation", and section 4, "Scientific Reproducibility &
Audit Tracking" -- "Confusion matrix, Cohen's kappa, and Precision scores
comparing LLM labels vs. human expert sample.").

Compares Ariel's blind manual labels (Dictionarys/results/human_validation_blind.csv,
filled in by hand after 03_sample_for_human_validation.py) against the LLM's
own labels for the same terms (Dictionarys/results/human_validation_key.csv,
kept separate specifically so labeling stayed blind).

Outputs:
  Dictionarys/results/interrater_reliability.md
    -- Cohen's kappa (5-category), 5x5 confusion matrix, per-category
       precision/recall, and overall binary (is_distress vs not) precision/
       recall/F1 treating the human label as ground truth.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, precision_recall_fscore_support,
    precision_score, recall_score, f1_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent

BLIND_CSV = DICT_DIR / "results" / "human_validation_blind.csv"
KEY_CSV   = DICT_DIR / "results" / "human_validation_key.csv"
OUT_MD    = DICT_DIR / "results" / "interrater_reliability.md"

CATEGORIES = [
    "DEBT_PRESSURE", "LIQUIDITY_SHORTAGE", "SHOCK_EMERGENCY",
    "HIGH_RISK_REFINANCING", "NEUTRAL_BENIGN",
]


def main() -> None:
    blind = pd.read_csv(BLIND_CSV)
    key = pd.read_csv(KEY_CSV)

    merged = blind.merge(key, on="token", how="inner", validate="one_to_one")
    if len(merged) != len(blind):
        print(f"WARNING: {len(blind) - len(merged)} tokens in the blind file "
              f"did not match the key file -- excluded from the comparison.")

    missing_labels = merged["human_label"].isna() | (merged["human_label"].astype(str).str.strip() == "")
    if missing_labels.any():
        print(f"WARNING: {missing_labels.sum()} rows have no human_label -- excluded.")
        merged = merged[~missing_labels].copy()

    merged["human_label"] = merged["human_label"].astype(str).str.strip().str.upper()
    merged["llm_category"] = merged["llm_category"].astype(str).str.strip().str.upper()

    unknown = set(merged["human_label"]) - set(CATEGORIES)
    if unknown:
        print(f"WARNING: human_label contains categories outside the rubric's 5: {unknown}")

    n = len(merged)
    print(f"Comparing {n} labeled terms...")

    y_human = merged["human_label"]
    y_llm = merged["llm_category"]

    labels_present = sorted(set(y_human) | set(y_llm), key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 99)

    kappa = cohen_kappa_score(y_human, y_llm, labels=labels_present)
    agreement = (y_human == y_llm).mean()

    cm = confusion_matrix(y_human, y_llm, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=[f"human:{c}" for c in labels_present],
                          columns=[f"llm:{c}" for c in labels_present])

    precision, recall, f1, support = precision_recall_fscore_support(
        y_human, y_llm, labels=labels_present, zero_division=0)
    per_cat = pd.DataFrame({
        "category": labels_present, "precision": precision, "recall": recall,
        "f1": f1, "support (human n)": support,
    })

    # Binary is_distress view: human ground truth vs LLM prediction
    y_human_bin = (y_human != "NEUTRAL_BENIGN").astype(int)
    y_llm_bin = (y_llm != "NEUTRAL_BENIGN").astype(int)
    bin_precision = precision_score(y_human_bin, y_llm_bin, zero_division=0)
    bin_recall = recall_score(y_human_bin, y_llm_bin, zero_division=0)
    bin_f1 = f1_score(y_human_bin, y_llm_bin, zero_division=0)
    bin_agreement = (y_human_bin == y_llm_bin).mean()

    print(f"Cohen's kappa (5-category): {kappa:.3f}")
    print(f"Raw agreement (5-category): {agreement:.3f}")
    print(f"Binary is_distress -- precision={bin_precision:.3f} recall={bin_recall:.3f} f1={bin_f1:.3f}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Inter-Rater Reliability -- LLM vs Human Expert (Ariel)\n\n")
        f.write(f"- Executed: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Sample size compared: {n}\n")
        f.write(f"- Blind labels: `{BLIND_CSV.relative_to(DICT_DIR.parent)}`\n")
        f.write(f"- LLM labels (key, not seen during labeling): `{KEY_CSV.relative_to(DICT_DIR.parent)}`\n\n")

        f.write("## 5-category agreement\n\n")
        f.write(f"- **Cohen's kappa: {kappa:.3f}**\n")
        f.write(f"- Raw agreement: {agreement:.3f}\n\n")

        f.write("### Confusion matrix (rows = human label, columns = LLM label)\n\n")
        f.write(cm_df.to_markdown())
        f.write("\n\n")

        f.write("### Per-category precision / recall / F1 (human label = ground truth)\n\n")
        f.write(per_cat.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")

        f.write("## Binary is_distress view (NEUTRAL_BENIGN vs any distress category)\n\n")
        f.write(f"- Precision: {bin_precision:.3f}\n")
        f.write(f"- Recall: {bin_recall:.3f}\n")
        f.write(f"- F1: {bin_f1:.3f}\n")
        f.write(f"- Agreement: {bin_agreement:.3f}\n")

    print(f"Output: {OUT_MD}")


if __name__ == "__main__":
    main()
