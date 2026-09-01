# -*- coding: utf-8 -*-
"""
12_fairness_by_fold_with_text.py
====================================
Extends Stage 8 in the two directions Ariel asked for on 2026-08-30, after confirming
Stages 8-11 used ONLY the text-free `structured` baseline, pooled across all 14
walk-forward folds:

  1. ADD TEXT: also evaluate `tfidf_full` and `finbert_pca50` -- the two
     best-performing text representations per docs/strict_temporal_v2_changes.md
     (Logistic mean AUC 0.6971 and 0.6956 respectively, vs. 0.6937 for structured) --
     to see whether adding text signal changes the fairness picture found on the
     no-text baseline.
  2. PER-FOLD, NOT POOLED: Stage 8 pooled all 14 folds' test predictions into one
     comparison. This script instead computes the fairness gap SEPARATELY per fold and
     aggregates (mean gap, count of folds significant, BH-FDR-adjusted count) --
     exactly the pattern already used everywhere else in this project for the DeLong
     AUC tests (see e.g. docs/strict_temporal_v2_changes.md's own finding that BH
     correction turned an 8/14-significant TF-IDF result into 0/14 -- pooling across
     chronologically distinct folds can hide exactly that kind of instability, since
     an early-period effect and a late-period effect of opposite sign can average out
     or reinforce each other in a pooled comparison in a way a per-fold breakdown
     would expose).

SCOPE (a deliberate narrowing, confirmed before building)
------------------------------------------------------------
Only the Low-vs-High contrast (doc section 11.4) is computed here, matching Stages
9-11's own scope -- not the 3-way Low/Medium/High comparison Stage 8 used. This keeps
the output focused on the exact question Stages 9-11 already investigated (does the
raw gap survive scrutiny), now checked across folds and with text added, rather than
re-opening the broader 3-group comparison. No CEM/entropy-balance matching is redone
here -- that would require rebuilding Stages 9-11 for every fold x representation
combination, a much larger undertaking not requested.

METRICS AND SIGNIFICANCE
---------------------------
Per fold x representation x model x grouping (BlackConcentration, SES_group): TPR,
FPR, PPR (each with a two-proportion z-test, Low vs High) and AUC (point estimate
only, no test -- DeLong would be the right test for an AUC gap but is out of scope
here). Then, across the 14 folds, for each metric: mean gap, count of folds with raw
p<0.05, and count of folds with BH-FDR-adjusted q<0.05 (BH applied within each
representation x model x grouping x metric family of 14 p-values, mirroring
strict_temporal_v2/run_full.py's own `finalize_delong()`).

INPUT:  results/strict_temporal_v2/predictions.csv   (repo root)
        census/results/06_loan_level_zip3_group_labels.csv
OUTPUT: census/results/12_fairness_by_fold.csv           (one row per fold x representation x model x grouping x metric)
        census/results/12_fairness_by_fold_summary.csv   (one row per representation x model x grouping x metric, aggregated across folds)
"""

import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
BASE_DIR = CENSUS_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

_cli = argparse.ArgumentParser()
_cli.add_argument("--predictions-csv", type=Path, default=None)
_cli.add_argument("--suffix", type=str, default="",
                  help='e.g. "_tuned" -- appended to every output filename.')
_args = _cli.parse_args()

PREDICTIONS_CSV = _args.predictions_csv or (BASE_DIR / "results" / "strict_temporal_v2" / "predictions.csv")
LABELS_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"
SUFFIX = _args.suffix

BY_FOLD_CSV = RESULTS_DIR / f"12_fairness_by_fold{SUFFIX}.csv"
SUMMARY_CSV = RESULTS_DIR / f"12_fairness_by_fold_summary{SUFFIX}.csv"

THRESHOLD = 0.5
REPRESENTATIONS = ["structured", "tfidf_full", "finbert_pca50"]
MODELS = ["Logistic", "XGBoost"]
GROUPINGS = ["BlackConcentration", "SES_group"]
N_FOLDS = 14


def two_proportion_test(x1, n1, x2, n2):
    """Standard two-proportion z-test. Returns (p1, p2, z, p_value); NaN z/p if either
    group has zero observations."""
    if n1 == 0 or n2 == 0:
        return (x1 / n1 if n1 else np.nan), (x2 / n2 if n2 else np.nan), np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1, p2, np.nan, np.nan
    z = (p1 - p2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1, p2, z, pval


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """Standard BH-FDR adjustment (same formula as strict_temporal_v2/metrics.py's
    add_bh_correction, reimplemented here per this repo's no-shared-module
    convention). NaN p-values are passed through as NaN q-values."""
    valid = pvals.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=pvals.index)
    ranks = valid.rank(method="first")
    n = len(valid)
    q = (valid.values * n / ranks.values)
    order = np.argsort(-ranks.values)  # descending rank -> enforce monotonicity
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted)
    q_final = np.empty(n)
    q_final[order] = np.clip(q_sorted, 0, 1)
    out = pd.Series(np.nan, index=pvals.index)
    out.loc[valid.index] = q_final
    return out


def fold_metrics(df: pd.DataFrame, group_col: str) -> dict:
    """TPR/FPR/PPR (with Low-vs-High tests) + AUC gap, for one fold x representation x
    model x grouping slice of predictions already merged with group labels."""
    low = df[df[group_col] == "Low"]
    high = df[df[group_col] == "High"]

    out = {}
    for label, side in [("Low", low), ("High", high)]:
        actual_pos = side["y_true"] == 1
        actual_neg = side["y_true"] == 0
        out[f"n_{label}"] = len(side)
        out[f"AUC_{label}"] = (
            roc_auc_score(side["y_true"], side["y_prob"]) if actual_pos.any() and actual_neg.any() else np.nan
        )

    tp_l = ((low["y_pred"] == 1) & (low["y_true"] == 1)).sum()
    fn_l = ((low["y_pred"] == 0) & (low["y_true"] == 1)).sum()
    fp_l = ((low["y_pred"] == 1) & (low["y_true"] == 0)).sum()
    tn_l = ((low["y_pred"] == 0) & (low["y_true"] == 0)).sum()
    tp_h = ((high["y_pred"] == 1) & (high["y_true"] == 1)).sum()
    fn_h = ((high["y_pred"] == 0) & (high["y_true"] == 1)).sum()
    fp_h = ((high["y_pred"] == 1) & (high["y_true"] == 0)).sum()
    tn_h = ((high["y_pred"] == 0) & (high["y_true"] == 0)).sum()

    for metric, (x1, n1, x2, n2) in {
        "TPR": (tp_l, tp_l + fn_l, tp_h, tp_h + fn_h),
        "FPR": (fp_l, fp_l + tn_l, fp_h, fp_h + tn_h),
        "PPR": ((low["y_pred"] == 1).sum(), len(low), (high["y_pred"] == 1).sum(), len(high)),
    }.items():
        p_low, p_high, z, pval = two_proportion_test(x1, n1, x2, n2)
        out[f"{metric}_Low"] = p_low
        out[f"{metric}_High"] = p_high
        out[f"gap_{metric}"] = p_high - p_low
        out[f"z_{metric}"] = z
        out[f"p_{metric}"] = pval

    out["gap_AUC"] = out["AUC_High"] - out["AUC_Low"]
    return out


def main() -> None:
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds[preds["representation"].isin(REPRESENTATIONS)].copy()
    preds["y_pred"] = (preds["y_prob"] >= THRESHOLD).astype(int)

    labels = pd.read_csv(LABELS_CSV, dtype={"id": str, "zip3": str})
    labels["id"] = labels["id"].astype("int64")
    df = preds.merge(labels[["id"] + GROUPINGS], on="id", how="inner", validate="many_to_one")

    global MODELS
    MODELS = sorted(df["model"].unique().tolist())
    print(f"Models found in predictions file: {MODELS}")

    rows = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            for grouping in GROUPINGS:
                for fold in range(1, N_FOLDS + 1):
                    sub = df[
                        (df["representation"] == representation)
                        & (df["model"] == model)
                        & (df["fold"] == fold)
                    ]
                    m = fold_metrics(sub, grouping)
                    rows.append({
                        "representation": representation, "model": model,
                        "grouping": grouping, "fold": fold, **m,
                    })

    by_fold = pd.DataFrame(rows)

    # -- BH-FDR correction within each (representation, model, grouping, metric) family --
    for metric in ["TPR", "FPR", "PPR"]:
        by_fold[f"q_{metric}"] = (
            by_fold.groupby(["representation", "model", "grouping"])[f"p_{metric}"]
            .transform(benjamini_hochberg)
        )

    summary_rows = []
    for (representation, model, grouping), grp in by_fold.groupby(["representation", "model", "grouping"]):
        for metric in ["TPR", "FPR", "PPR", "AUC"]:
            row = {
                "representation": representation, "model": model, "grouping": grouping,
                "metric": metric, "mean_gap": round(grp[f"gap_{metric}"].mean(), 4),
                "n_folds": grp["fold"].nunique(),
            }
            if metric != "AUC":
                row["n_sig_raw_p05"] = int((grp[f"p_{metric}"] < 0.05).sum())
                row["n_sig_BH_q05"] = int((grp[f"q_{metric}"] < 0.05).sum())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    print("-- Per-representation x model x grouping summary (mean gap across 14 folds, "
          "Low->High; FPR/TPR/PPR include significance counts) --")
    print(summary.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    by_fold.round(4).to_csv(BY_FOLD_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {BY_FOLD_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
