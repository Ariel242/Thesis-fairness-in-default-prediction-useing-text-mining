# -*- coding: utf-8 -*-
"""
14_fairness_full_grid.py
====================================
Closes four coverage gaps left open by Stages 08 and 12, agreed with Ariel 2026-08-31:

  (2) ALL 7 GROUPINGS, not just BlackConcentration and SES_group. Stage 04 built six
      demographic concentration dimensions and Stage 03 built the SES grouping, but
      fairness was only ever evaluated on two of the seven.
  (3) ALL 7 REPRESENTATIONS, not just structured / tfidf_full / finbert_pca50 --
      adding lexicon, tfidf_chi2, finbert_768 and structured_has_desc. finbert_768
      matters most here: it is the worst-performing representation for prediction
      (see docs/strict_temporal_v2_changes.md), so it is the most plausible candidate
      for a fairness cost that the better representations do not show.
  (4) THRESHOLD SENSITIVITY. Every earlier fairness stage used a single hard-coded
      0.5 cutoff. TPR/FPR/PPR all depend on that arbitrary choice, so the whole grid is
      recomputed at several thresholds -- this is a robustness check on the existing
      conclusions, NOT four times as many independent findings.
  (5) EQUALIZED ODDS as a single combined measure (doc section 14 names it explicitly;
      earlier stages reported its TPR and FPR components separately but never the
      combined quantity).

WHAT "EQUALIZED ODDS DIFFERENCE" MEANS HERE
---------------------------------------------
For the Low-vs-High contrast of a grouping, the standard definition is used:

    EO_diff = max(|TPR_High - TPR_Low|, |FPR_High - FPR_Low|)

i.e. the worse of the two component violations. It is always non-negative and has no
direction, so unlike the signed TPR/FPR gaps it says "how unequal", not "which way" --
both components are therefore kept in the output alongside it, since EO_diff alone
cannot tell you whether the High group is over- or under-predicted.

SMALL-CELL GUARD (self-check added after reviewing the risk)
--------------------------------------------------------------
Splitting 14 folds x 7 groupings means some fold x group cells are thin, especially in
early folds (fold 1's test set is ~1,000 loans, split three ways, of which only the
Low/High thirds are used, of which only the actual-defaults subset drives TPR). A TPR
computed on a handful of defaults is noise, not signal. Any cell with fewer than
MIN_CELL_DEFAULTS actual defaults or MIN_CELL_NEGATIVES actual non-defaults on EITHER
side of the contrast is therefore recorded with NaN metrics and flagged in
`insufficient_data`, rather than silently contributing a meaningless extreme value to
the fold counts and means. The summary reports how many folds were usable per row
(`n_folds_used`) so a "0/14 significant" line can be told apart from a "only 3 folds
had enough data" line.

MULTIPLE COMPARISONS: BH-FDR is applied across the 14 folds within each
(representation, model, grouping, metric, threshold) family -- identical convention to
Stage 12, so the two stages' significance counts are directly comparable. No correction
is applied ACROSS thresholds, because the thresholds are not independent hypotheses;
they are the same hypothesis re-asked under a different operating point.

RUNTIME: reads the full 1.05M-row predictions file once and slices it in memory.
Expect a couple of minutes, no model fitting.

INPUT:  results/strict_temporal_v2/predictions.csv   (repo root)
        census/results/06_loan_level_zip3_group_labels.csv
OUTPUT: census/results/14_fairness_full_grid.csv          (per fold x repr x model x grouping x threshold)
        census/results/14_fairness_full_grid_summary.csv  (aggregated across folds)
"""

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

PREDICTIONS_CSV = BASE_DIR / "results" / "strict_temporal_v2" / "predictions.csv"
LABELS_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"

GRID_CSV = RESULTS_DIR / "14_fairness_full_grid.csv"
SUMMARY_CSV = RESULTS_DIR / "14_fairness_full_grid_summary.csv"

REPRESENTATIONS = [
    "structured", "structured_has_desc", "lexicon",
    "tfidf_full", "tfidf_chi2", "finbert_768", "finbert_pca50",
]
MODELS = ["Logistic", "XGBoost"]
GROUPINGS = [
    "SES_group",
    "BlackConcentration", "HispanicConcentration", "AsianConcentration",
    "WhiteConcentration", "ForeignBornConcentration", "LimitedEnglishConcentration",
]
# Full sweep 0.2-0.9. Note these models are fit with class_weight="balanced"
# (and XGBoost with scale_pos_weight), which inflates predicted probabilities well
# above the ~15% base rate -- the mean predicted probability sits near 0.45. So the
# "natural" operating region here is the middle of this range, and the high end (0.8,
# 0.9) is a deliberately conservative regime where very few loans are flagged at all.
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_FOLDS = 14

MIN_CELL_DEFAULTS = 30    # actual positives needed on each side for a usable TPR
MIN_CELL_NEGATIVES = 30   # actual negatives needed on each side for a usable FPR


def two_proportion_test(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1, p2, np.nan, np.nan
    z = (p1 - p2) / se
    return p1, p2, z, 2 * (1 - stats.norm.cdf(abs(z)))


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH-FDR (same formula as Stages 12-13 and strict_temporal_v2/metrics.py, kept as
    a local copy per this repo's no-shared-module convention)."""
    valid = pvals.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=pvals.index)
    ranks = valid.rank(method="first")
    n = len(valid)
    q = valid.values * n / ranks.values
    order = np.argsort(-ranks.values)
    q_sorted = np.minimum.accumulate(q[order])
    q_final = np.empty(n)
    q_final[order] = np.clip(q_sorted, 0, 1)
    out = pd.Series(np.nan, index=pvals.index)
    out.loc[valid.index] = q_final
    return out


def cell_metrics(low: pd.DataFrame, high: pd.DataFrame, threshold: float) -> dict:
    """All fairness metrics for one Low-vs-High contrast at one threshold, or NaNs
    plus insufficient_data=True if either side is too thin to be meaningful."""
    out = {"n_Low": len(low), "n_High": len(high),
           "n_defaults_Low": int((low["y_true"] == 1).sum()),
           "n_defaults_High": int((high["y_true"] == 1).sum())}

    neg_low = int((low["y_true"] == 0).sum())
    neg_high = int((high["y_true"] == 0).sum())

    insufficient = (
        out["n_defaults_Low"] < MIN_CELL_DEFAULTS or out["n_defaults_High"] < MIN_CELL_DEFAULTS
        or neg_low < MIN_CELL_NEGATIVES or neg_high < MIN_CELL_NEGATIVES
    )
    out["insufficient_data"] = insufficient
    if insufficient:
        for m in ["TPR", "FPR", "PPR"]:
            out.update({f"gap_{m}": np.nan, f"{m}_Low": np.nan,
                        f"{m}_High": np.nan, f"p_{m}": np.nan})
        out.update({"gap_AUC": np.nan, "AUC_Low": np.nan, "AUC_High": np.nan,
                    "EO_diff": np.nan})
        return out

    pred_low = (low["y_prob"] >= threshold).astype(int)
    pred_high = (high["y_prob"] >= threshold).astype(int)

    counts = {
        "TPR": (int(((pred_low == 1) & (low["y_true"] == 1)).sum()), out["n_defaults_Low"],
                int(((pred_high == 1) & (high["y_true"] == 1)).sum()), out["n_defaults_High"]),
        "FPR": (int(((pred_low == 1) & (low["y_true"] == 0)).sum()), neg_low,
                int(((pred_high == 1) & (high["y_true"] == 0)).sum()), neg_high),
        "PPR": (int((pred_low == 1).sum()), len(low),
                int((pred_high == 1).sum()), len(high)),
    }

    for metric, (x1, n1, x2, n2) in counts.items():
        p_low, p_high, _z, pval = two_proportion_test(x1, n1, x2, n2)
        out[f"{metric}_Low"] = p_low
        out[f"{metric}_High"] = p_high
        out[f"gap_{metric}"] = p_high - p_low
        out[f"p_{metric}"] = pval

    # AUC is threshold-free; it is recomputed per threshold row only so every row is
    # self-contained, and is identical across thresholds by construction.
    out["AUC_Low"] = roc_auc_score(low["y_true"], low["y_prob"])
    out["AUC_High"] = roc_auc_score(high["y_true"], high["y_prob"])
    out["gap_AUC"] = out["AUC_High"] - out["AUC_Low"]

    out["EO_diff"] = max(abs(out["gap_TPR"]), abs(out["gap_FPR"]))
    return out


def main() -> None:
    print("Loading predictions...")
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds[preds["representation"].isin(REPRESENTATIONS)]

    labels = pd.read_csv(LABELS_CSV, dtype={"id": str, "zip3": str})
    labels["id"] = labels["id"].astype("int64")
    df = preds.merge(labels[["id"] + GROUPINGS], on="id", how="inner", validate="many_to_one")
    print(f"  {len(df):,} prediction rows across "
          f"{df['representation'].nunique()} representations x {df['model'].nunique()} models.")

    rows = []
    for representation in REPRESENTATIONS:
        rep_df = df[df["representation"] == representation]
        for model in MODELS:
            model_df = rep_df[rep_df["model"] == model]
            for fold in range(1, N_FOLDS + 1):
                fold_df = model_df[model_df["fold"] == fold]
                for grouping in GROUPINGS:
                    low = fold_df[fold_df[grouping] == "Low"]
                    high = fold_df[fold_df[grouping] == "High"]
                    for threshold in THRESHOLDS:
                        rows.append({
                            "representation": representation, "model": model,
                            "grouping": grouping, "fold": fold, "threshold": threshold,
                            **cell_metrics(low, high, threshold),
                        })
        print(f"  done: {representation}")

    grid = pd.DataFrame(rows)

    family = ["representation", "model", "grouping", "threshold"]
    for metric in ["TPR", "FPR", "PPR"]:
        grid[f"q_{metric}"] = grid.groupby(family)[f"p_{metric}"].transform(benjamini_hochberg)

    summary_rows = []
    for keys, grp in grid.groupby(family):
        usable = grp[~grp["insufficient_data"]]
        base = dict(zip(family, keys))
        base["n_folds_used"] = len(usable)
        base["n_folds_insufficient"] = int(grp["insufficient_data"].sum())
        for metric in ["TPR", "FPR", "PPR", "AUC", "EO_diff"]:
            row = dict(base)
            row["metric"] = metric
            col = "EO_diff" if metric == "EO_diff" else f"gap_{metric}"
            row["mean_value"] = round(usable[col].mean(), 4) if len(usable) else np.nan
            if metric in ("TPR", "FPR", "PPR"):
                row["n_sig_raw_p05"] = int((usable[f"p_{metric}"] < 0.05).sum())
                row["n_sig_BH_q05"] = int((usable[f"q_{metric}"] < 0.05).sum())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    n_insufficient = int(grid["insufficient_data"].sum())
    print(f"\nCells skipped for insufficient data: {n_insufficient:,} of {len(grid):,} "
          f"({n_insufficient / len(grid):.1%}) -- see `insufficient_data` column.")

    print("\n-- FPR gap at threshold 0.5, all groupings x representations (Logistic) --")
    view = summary[(summary["metric"] == "FPR") & (summary["threshold"] == 0.5)
                   & (summary["model"] == "Logistic")]
    print(view.pivot(index="grouping", columns="representation", values="n_sig_BH_q05").to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    grid.round(4).to_csv(GRID_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {GRID_CSV} ({len(grid):,} rows)")
    print(f"Saved: {SUMMARY_CSV} ({len(summary):,} rows)")


if __name__ == "__main__":
    main()
