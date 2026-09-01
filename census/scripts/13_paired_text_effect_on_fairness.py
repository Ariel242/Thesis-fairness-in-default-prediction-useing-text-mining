# -*- coding: utf-8 -*-
"""
13_paired_text_effect_on_fairness.py
====================================
Tests the thesis's actual central question, which no earlier stage tested formally:
DOES ADDING TEXT CHANGE THE FAIRNESS GAP?

Stage 12 computed the fairness gap per fold for three representations (structured,
tfidf_full, finbert_pca50) and reported their mean gaps side by side (e.g. SES_group
FPR: 0.0607 structured vs. 0.0583 tfidf_full vs. 0.0536 finbert_pca50). But comparing
those three means BY EYE is not a statistical test -- it cannot say whether a 0.0024
difference is a real effect of adding TF-IDF or just fold-to-fold noise. This script
runs the actual test.

THE PAIRING (why this is the right design)
--------------------------------------------
The three representations were evaluated on THE SAME 14 walk-forward folds, i.e. on
identical test sets with identical ZIP3 group assignments and identical y_true. The
only thing that differs is which features the model saw. That makes the fold the
natural pairing unit: for each fold f, compute

    delta_f = gap(text_representation, f) - gap(structured, f)

and test whether the 14 delta_f values are centered on zero. Pairing removes all
between-fold variance (some folds are simply harder, or have a wider raw gap, for
reasons unrelated to text) -- exactly the same logic that makes DeLong's paired test
the right tool for comparing two AUCs on one test set, as used throughout this repo.

TWO TESTS PER COMPARISON (reported together, deliberately)
-------------------------------------------------------------
  - Paired t-test (scipy.stats.ttest_rel): parametric, higher power, assumes the 14
    per-fold deltas are roughly normal.
  - Wilcoxon signed-rank (scipy.stats.wilcoxon): non-parametric, no normality
    assumption, robust to an outlier fold -- with n=14 this matters.
Both are reported for every comparison rather than picking one, so a reader can see
whether the conclusion depends on the distributional assumption. Choosing whichever
test gives the smaller p-value after seeing both would be exactly the
outcome-dependent selection the methodology doc's section 8 forbids -- so the rule
fixed here in advance is: BOTH are reported, and a conclusion is called "significant"
only if BOTH agree at the 5% level (recorded in the `both_agree_sig` output column).

EFFECT SIZE: mean paired delta plus Cohen's d_z (mean(delta) / sd(delta)) -- the
standard paired-design effect size. With n=14 a null result is likely to be
"underpowered" rather than "proven zero", so `mean_delta` and its 95% CI are reported
alongside every p-value: a tight CI around zero is meaningfully different from a wide
CI that merely fails to exclude zero, and only the former supports "text does not
change fairness".

COMPARISONS RUN: for each model (Logistic, XGBoost) x grouping (BlackConcentration,
SES_group) x metric (TPR, FPR, PPR, AUC):
    tfidf_full     vs structured
    finbert_pca50  vs structured
    finbert_pca50  vs tfidf_full     (text-vs-text, not just text-vs-none)

MULTIPLE COMPARISONS: BH-FDR is applied across the full family of paired tests within
each (model, grouping) pair -- 12 tests each (3 comparisons x 4 metrics) -- mirroring
how strict_temporal_v2 corrects its own DeLong families.

INPUT:  census/results/12_fairness_by_fold.csv   (per-fold gaps produced by Stage 12)
OUTPUT: census/results/13_paired_text_effect.csv        (one row per comparison x model x grouping x metric)
        census/results/13_paired_text_effect_deltas.csv (the raw per-fold deltas behind every test)
"""

import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

# --suffix reads Stage 12's correspondingly-suffixed output (e.g. "_tuned") and writes
# this stage's own output under the same suffix, so a tuned-vs-untuned pair of runs
# never overwrites each other.
_cli = argparse.ArgumentParser()
_cli.add_argument("--suffix", type=str, default="",
                  help='e.g. "_tuned" -- must match the suffix Stage 12 was run with.')
_args = _cli.parse_args()
SUFFIX = _args.suffix

BY_FOLD_CSV = RESULTS_DIR / f"12_fairness_by_fold{SUFFIX}.csv"
OUTPUT_CSV = RESULTS_DIR / f"13_paired_text_effect{SUFFIX}.csv"
DELTAS_CSV = RESULTS_DIR / f"13_paired_text_effect_deltas{SUFFIX}.csv"

MODELS = ["Logistic", "XGBoost"]
GROUPINGS = ["BlackConcentration", "SES_group"]
METRICS = ["TPR", "FPR", "PPR", "AUC"]

# (representation_B, representation_A) -> tests B - A
COMPARISONS = [
    ("tfidf_full", "structured"),
    ("finbert_pca50", "structured"),
    ("finbert_pca50", "tfidf_full"),
]


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH-FDR adjustment (same formula as Stage 12's own copy and
    strict_temporal_v2/metrics.py, per this repo's no-shared-module convention)."""
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


def paired_test(deltas: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon signed-rank on the per-fold deltas, with effect size
    and a 95% CI for the mean delta."""
    deltas = deltas[~np.isnan(deltas)]
    n = len(deltas)
    if n < 3:
        return {k: np.nan for k in
                ["n_folds", "mean_delta", "sd_delta", "ci_lo", "ci_hi",
                 "cohens_dz", "t_stat", "p_ttest", "w_stat", "p_wilcoxon"]}

    mean_d = deltas.mean()
    sd_d = deltas.std(ddof=1)
    se = sd_d / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)

    t_stat, p_ttest = stats.ttest_rel(deltas, np.zeros(n))

    # Wilcoxon errors out if every delta is exactly zero; treat that as "no effect".
    if np.allclose(deltas, 0):
        w_stat, p_wilcoxon = np.nan, 1.0
    else:
        w_stat, p_wilcoxon = stats.wilcoxon(deltas)

    return {
        "n_folds": n,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "ci_lo": mean_d - t_crit * se,
        "ci_hi": mean_d + t_crit * se,
        "cohens_dz": mean_d / sd_d if sd_d > 0 else np.nan,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "w_stat": w_stat,
        "p_wilcoxon": p_wilcoxon,
    }


def main() -> None:
    by_fold = pd.read_csv(BY_FOLD_CSV)

    global MODELS
    MODELS = sorted(by_fold["model"].unique().tolist())
    print(f"Models found in {BY_FOLD_CSV.name}: {MODELS}")

    rows, delta_rows = [], []
    for model in MODELS:
        for grouping in GROUPINGS:
            for repr_b, repr_a in COMPARISONS:
                for metric in METRICS:
                    gap_col = f"gap_{metric}"

                    a = (by_fold[(by_fold["representation"] == repr_a)
                                 & (by_fold["model"] == model)
                                 & (by_fold["grouping"] == grouping)]
                         .set_index("fold")[gap_col].sort_index())
                    b = (by_fold[(by_fold["representation"] == repr_b)
                                 & (by_fold["model"] == model)
                                 & (by_fold["grouping"] == grouping)]
                         .set_index("fold")[gap_col].sort_index())

                    assert a.index.equals(b.index), \
                        f"Fold sets differ between {repr_a} and {repr_b} -- pairing would be invalid."

                    deltas = (b - a).values
                    result = paired_test(deltas)

                    rows.append({
                        "model": model, "grouping": grouping, "metric": metric,
                        "comparison": f"{repr_b} - {repr_a}",
                        f"mean_gap_{repr_a}": round(a.mean(), 4),
                        f"mean_gap_{repr_b}": round(b.mean(), 4),
                        **{k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
                           for k, v in result.items()},
                    })

                    for fold, d in zip(a.index, deltas):
                        delta_rows.append({
                            "model": model, "grouping": grouping, "metric": metric,
                            "comparison": f"{repr_b} - {repr_a}", "fold": fold,
                            f"gap_{repr_a}": round(a.loc[fold], 4),
                            f"gap_{repr_b}": round(b.loc[fold], 4),
                            "delta": round(d, 4),
                        })

    out = pd.DataFrame(rows)

    # BH-FDR within each (model, grouping) family of 12 tests, for each test separately.
    for p_col, q_col in [("p_ttest", "q_ttest"), ("p_wilcoxon", "q_wilcoxon")]:
        out[q_col] = out.groupby(["model", "grouping"])[p_col].transform(benjamini_hochberg).round(4)

    out["both_agree_sig"] = (out["p_ttest"] < 0.05) & (out["p_wilcoxon"] < 0.05)
    out["both_agree_sig_BH"] = (out["q_ttest"] < 0.05) & (out["q_wilcoxon"] < 0.05)

    deltas_out = pd.DataFrame(delta_rows)

    display_cols = ["model", "grouping", "metric", "comparison", "mean_delta",
                    "ci_lo", "ci_hi", "cohens_dz", "p_ttest", "p_wilcoxon",
                    "both_agree_sig", "both_agree_sig_BH"]
    print("-- Paired per-fold test: does adding text change the fairness gap? --")
    print("   (mean_delta > 0 means the text representation has a WIDER gap than the baseline)")
    print(out[display_cols].to_string(index=False))

    n_sig = int(out["both_agree_sig"].sum())
    n_sig_bh = int(out["both_agree_sig_BH"].sum())
    print(f"\nComparisons where BOTH tests agree at p<0.05: {n_sig} of {len(out)}")
    print(f"                     ... and survive BH-FDR: {n_sig_bh} of {len(out)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    deltas_out.to_csv(DELTAS_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {DELTAS_CSV}")


if __name__ == "__main__":
    main()
