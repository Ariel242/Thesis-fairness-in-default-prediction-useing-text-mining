# -*- coding: utf-8 -*-
"""
08_fairness_metrics_black_ses.py
====================================
STAGE 8 (first fairness look) of the ZIP3 grouping pipeline -- implements doc section
14, "הקשר למדדי ההוגנות במחקר", for the two "cleanest" comparison bases identified in
Stage 5's balance diagnostics: `BlackConcentration` and `SES_group` (lowest
cross-dimensional SMD of all 7 groupings -- see census/scripts/05_balance_diagnostics.py's
output). Not run automatically by run_pipeline.py, same reasoning as Stage 7: this is a
downstream analysis step, not part of group *construction*.

SCOPE OF THIS SCRIPT (explicit, so results aren't over-read)
------------------------------------------------------------
Only 2 of the 7 groupings (BlackConcentration, SES_group) and only the "structured"
(no text at all) model are covered here, on explicit request. The other 5 groupings
(Hispanic/Asian/White/ForeignBorn/LimitedEnglishConcentration) and the text-vs-no-text
comparison the rest of this thesis is actually about are NOT computed here -- extending
this script to loop over every grouping x every representation is a small change (see
the PREDICTIONS_SOURCE / GROUPINGS constants below) but was deliberately not done yet.

PREDICTIONS SOURCE (a real methodological choice -- documented, not hidden)
------------------------------------------------------------------------------
Uses `results/strict_temporal_v2/predictions.csv` -- the leak-fixed walk-forward run
(see docs/strict_temporal_v2_changes.md), filtered to `representation == "structured"`,
i.e. the baseline model with NO text signal at all. This is the right starting point for
a first fairness look for two reasons: (1) it is the most methodologically rigorous
prediction set in this repo (per-fold structured preprocessing, no leakage), and (2) it
isolates "does the credit model itself already treat these ZIP3 groups differently"
before asking whether adding text changes that picture. Rows from all 14 walk-forward
folds are pooled together (each fold's test period is disjoint, so pooling is a valid
"whole test period" summary, not double-counting).

CLASSIFICATION THRESHOLD -- 0.5 (also a documented, revisitable choice)
----------------------------------------------------------------------
No fixed threshold is defined anywhere else in this repo for TPR/FPR (the
`THRESHOLDS = [0.5, 0.6, 0.7]` constant that appears in several other scripts is
explicitly marked dead/unused code in each of them). 0.5 is used here as the standard
default. AUC (threshold-free) is reported alongside TPR/FPR precisely so a reader is not
stuck trusting one arbitrary cutoff.

METRICS (doc section 14)
-------------------------
Per group level (Low/Medium/High), per model:
  - TPR (True Positive Rate / sensitivity/recall)  = TP / (TP + FN), among actual defaults
  - FPR (False Positive Rate)                       = FP / (FP + TN), among actual non-defaults
  - PPR (Predicted Positive Rate)                   = (TP + FP) / n   -- for a Demographic Parity read
  - AUC                                             -- threshold-free ranking quality
Then pairwise GAPS (Low-Medium, Low-High, Medium-High) on each of TPR/FPR/PPR/AUC --
this is the Equalized-Odds-style comparison doc section 14 describes
(`FPR_Low, FPR_Medium, FPR_High` -> gaps).

INPUT:  results/strict_temporal_v2/predictions.csv  (repo root, NOT under census/)
        census/results/06_loan_level_zip3_group_labels.csv
OUTPUT: census/results/08_fairness_by_group.csv        (one row per model x grouping x level)
        census/results/08_fairness_gaps.csv             (one row per model x grouping x pairwise comparison)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
BASE_DIR = CENSUS_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

PREDICTIONS_CSV = BASE_DIR / "results" / "strict_temporal_v2" / "predictions.csv"
LABELS_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"

BY_GROUP_CSV = RESULTS_DIR / "08_fairness_by_group.csv"
GAPS_CSV = RESULTS_DIR / "08_fairness_gaps.csv"

PREDICTIONS_SOURCE_REPRESENTATION = "structured"  # baseline, no text -- see docstring
THRESHOLD = 0.5
MODELS = ["Logistic", "XGBoost"]
GROUPINGS = ["BlackConcentration", "SES_group"]  # the 2 requested, lowest-SMD groupings
GROUP_LEVELS = ["Low", "Medium", "High"]


def load_merged_predictions() -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds[preds["representation"] == PREDICTIONS_SOURCE_REPRESENTATION].copy()

    labels = pd.read_csv(LABELS_CSV, dtype={"id": str})
    labels["id"] = labels["id"].astype("int64")  # match predictions.csv's id dtype

    merged = preds.merge(labels[["id"] + GROUPINGS], on="id", how="inner", validate="many_to_one")
    merged["y_pred"] = (merged["y_prob"] >= THRESHOLD).astype(int)
    return merged


def group_metrics(df: pd.DataFrame, group_col: str, model: str) -> pd.DataFrame:
    rows = []
    for level in GROUP_LEVELS:
        sub = df[(df[group_col] == level) & (df["model"] == model)]
        y_true, y_pred, y_prob = sub["y_true"], sub["y_pred"], sub["y_prob"]

        actual_pos = y_true == 1
        actual_neg = y_true == 0
        tp = ((y_pred == 1) & actual_pos).sum()
        fn = ((y_pred == 0) & actual_pos).sum()
        fp = ((y_pred == 1) & actual_neg).sum()
        tn = ((y_pred == 0) & actual_neg).sum()

        rows.append({
            "grouping_variable": group_col, "group_level": level, "model": model,
            "n": len(sub), "n_defaults": int(actual_pos.sum()),
            "TPR": round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan"),
            "FPR": round(fp / (fp + tn), 4) if (fp + tn) > 0 else float("nan"),
            "PPR": round((tp + fp) / len(sub), 4) if len(sub) > 0 else float("nan"),
            "AUC": round(roc_auc_score(y_true, y_prob), 4) if actual_pos.any() and actual_neg.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def gap_table(by_group: pd.DataFrame, group_col: str, model: str) -> pd.DataFrame:
    sub = by_group[(by_group["grouping_variable"] == group_col) & (by_group["model"] == model)]
    sub = sub.set_index("group_level")
    rows = []
    for level_a, level_b in combinations(GROUP_LEVELS, 2):
        row = {"grouping_variable": group_col, "model": model, "comparison": f"{level_a}-{level_b}"}
        for metric in ["TPR", "FPR", "PPR", "AUC"]:
            row[f"gap_{metric}"] = round(sub.loc[level_a, metric] - sub.loc[level_b, metric], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_merged_predictions()
    print(f"Merged predictions (representation='{PREDICTIONS_SOURCE_REPRESENTATION}', "
          f"threshold={THRESHOLD}): {len(df):,} rows "
          f"({df['model'].value_counts().to_dict()}), pooled across all 14 folds.")

    by_group_tables, gap_tables = [], []
    for group_col in GROUPINGS:
        for model in MODELS:
            bg = group_metrics(df, group_col, model)
            by_group_tables.append(bg)
            gap_tables.append(gap_table(bg, group_col, model))

    by_group = pd.concat(by_group_tables, ignore_index=True)
    gaps = pd.concat(gap_tables, ignore_index=True)

    print("\n-- Per-group fairness metrics --")
    print(by_group.to_string(index=False))
    print("\n-- Pairwise gaps (Equalized-Odds-style) --")
    print(gaps.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    by_group.to_csv(BY_GROUP_CSV, index=False)
    gaps.to_csv(GAPS_CSV, index=False)
    print(f"\nSaved: {BY_GROUP_CSV}")
    print(f"Saved: {GAPS_CSV}")


if __name__ == "__main__":
    main()
