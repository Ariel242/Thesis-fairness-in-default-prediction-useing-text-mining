# -*- coding: utf-8 -*-
"""
10_matched_sensitivity_analysis_deciles.py
====================================
Finer-grained follow-up to 09_matched_sensitivity_analysis.py, per doc section 7's
open-ended invitation to check whether conclusions hold under a different, reasonable
operationalization (the same spirit as the section-11 robustness checks, applied to the
matching step itself rather than to group construction).

WHY A SECOND VERSION
----------------------
Stage 9 matched on the CATEGORICAL 3-level group (SES_group / BlackConcentration) that
Stages 3-4 already built. That worked well for one covariate at a time but, being only
3 strata, left several of the 7 raw SES covariates poorly balanced for the
BlackConcentration direction (median_income_approx and share_bachelor_plus actually got
WORSE after matching, not better -- see Stage 9's own printed output). The likely reason:
two ZIP3s in the same SES_group tertile can still differ a lot on any ONE underlying
variable, since SES_group is already a lossy compression of 7 variables into 3 buckets.

This script repeats the exact same CEM logic, but stratifies on DECILES (up to 10 bins,
fewer if percentile ties force merges -- see `decile_strata()`) of the underlying
CONTINUOUS score instead of the 3-level categorical group:

    A) BlackConcentration (High vs Low), matched on deciles of SESDisadvantage
    B) SES_group (High vs Low), matched on deciles of share_black_nh

Finer strata should balance the underlying covariates more precisely, at the cost of
more strata being at risk of pruning (a decile might end up with only High or only Low
ZIP3s, especially in the tails where the two groups are most different -- this script
prints exactly how many ZIP3s get pruned, so that cost is visible, not hidden).

Everything else -- the CEM weight formula, the weighted-SMD balance check, the weighted
fairness metrics with Kish-ESS significance testing -- is identical to Stage 9 (see that
script's docstring for the full methodological explanation); only the stratification
granularity changes. Not run by run_pipeline.py, same as Stages 7-9.

INPUT:  census/results/01_census_data_validated.csv        (raw covariates + share_black_nh)
        census/results/02_ses_disadvantage_index.csv        (continuous SESDisadvantage)
        census/results/03_ses_groups.csv                    (SES_group, High/Low labels)
        census/results/04_demographic_concentration_groups.csv (BlackConcentration, High/Low labels)
        census/results/06_loan_level_zip3_group_labels.csv
        results/strict_temporal_v2/predictions.csv            (repo root)
OUTPUT: census/results/10_cem_weights_deciles.csv
        census/results/10_balance_before_after_matching_deciles.csv
        census/results/10_fairness_before_after_matching_deciles.csv
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
BASE_DIR = CENSUS_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

VALIDATED_CSV = RESULTS_DIR / "01_census_data_validated.csv"
SES_INDEX_CSV = RESULTS_DIR / "02_ses_disadvantage_index.csv"
SES_GROUPS_CSV = RESULTS_DIR / "03_ses_groups.csv"
DEMO_GROUPS_CSV = RESULTS_DIR / "04_demographic_concentration_groups.csv"
LABELS_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"

_cli = argparse.ArgumentParser()
_cli.add_argument("--predictions-csv", type=Path, default=None)
_cli.add_argument("--suffix", type=str, default="",
                  help='e.g. "_tuned" -- appended to every output filename.')
_args = _cli.parse_args()

PREDICTIONS_CSV = _args.predictions_csv or (BASE_DIR / "results" / "strict_temporal_v2" / "predictions.csv")
SUFFIX = _args.suffix

WEIGHTS_CSV = RESULTS_DIR / f"10_cem_weights_deciles{SUFFIX}.csv"
BALANCE_CSV = RESULTS_DIR / f"10_balance_before_after_matching_deciles{SUFFIX}.csv"
FAIRNESS_CSV = RESULTS_DIR / f"10_fairness_before_after_matching_deciles{SUFFIX}.csv"

THRESHOLD = 0.5
MODELS = ["Logistic", "XGBoost"]
N_DECILES = 10

SES_COVARIATES = [
    "poverty_rate", "unemployment_rate", "share_bachelor_plus",
    "homeownership_rate", "uninsured_rate",
    "median_income_approx", "median_home_value_approx",
]
DIRECTIONS = [
    {
        "label": "BlackConcentration (matched on SESDisadvantage deciles)",
        "treatment_col": "BlackConcentration",
        "stratify_score_col": "SESDisadvantage",
        "balance_covariates": SES_COVARIATES,
    },
    {
        "label": "SES_group (matched on share_black_nh deciles)",
        "treatment_col": "SES_group",
        "stratify_score_col": "share_black_nh",
        "balance_covariates": ["share_black_nh"],
    },
]


def decile_strata(score: pd.Series, n_bins: int = N_DECILES) -> pd.Series:
    """Tie-safe decile bins -- same `<=` cutpoint logic as the tertile split in
    03_build_ses_tertiles.py, generalized to n_bins instead of 3. If heavy ties force
    fewer than n_bins distinct cutpoints (e.g. many ZIP3s sharing the same low
    share_black_nh value), fewer, wider strata are produced rather than splitting tied
    ZIP3s apart -- `np.unique` collapses duplicate percentile edges."""
    edges = np.unique(np.percentile(score, np.linspace(0, 100, n_bins + 1)))
    bin_idx = np.searchsorted(edges[1:-1], score, side="left")
    return pd.Series(bin_idx, index=score.index)


def compute_cem_weights(df: pd.DataFrame, treatment_col: str, stratum: pd.Series) -> pd.DataFrame:
    """Identical CEM weight formula to 09_matched_sensitivity_analysis.py's version,
    generalized to take a precomputed stratum label Series (here, decile bins) instead
    of reading a stratifying column by name."""
    work = df[[  "zip3", treatment_col]].copy()
    work["stratum"] = stratum.values
    work = work[work[treatment_col].isin(["Low", "High"])]

    strata_both = work.groupby("stratum")[treatment_col].nunique().pipe(lambda s: s[s == 2].index)
    n_pruned = work[~work["stratum"].isin(strata_both)]["zip3"].nunique()
    work = work[work["stratum"].isin(strata_both)].copy()

    t_total = (work[treatment_col] == "High").sum()
    c_total = (work[treatment_col] == "Low").sum()

    weights = []
    for s, grp in work.groupby("stratum"):
        t_s = (grp[treatment_col] == "High").sum()
        c_s = (grp[treatment_col] == "Low").sum()
        for _, row in grp.iterrows():
            w = 1.0 if row[treatment_col] == "High" else (t_s / t_total) / (c_s / c_total)
            weights.append({"zip3": row["zip3"], treatment_col: row[treatment_col], "weight": w})

    n_strata_used = work["stratum"].nunique()
    print(f"  Strata used: {n_strata_used} (of up to {N_DECILES} requested). "
          f"Pruned: {n_pruned} ZIP3s (stratum missing High or Low side).")
    return pd.DataFrame(weights)


def weighted_mean_var(x: pd.Series, w: pd.Series) -> tuple:
    mean = np.average(x, weights=w)
    var = np.average((x - mean) ** 2, weights=w)
    return mean, var


def weighted_smd(group_a: pd.DataFrame, group_b: pd.DataFrame, covariate: str) -> float:
    mean_a, var_a = weighted_mean_var(group_a[covariate], group_a["weight"])
    mean_b, var_b = weighted_mean_var(group_b[covariate], group_b["weight"])
    pooled_sd = ((var_a + var_b) / 2) ** 0.5
    return 0.0 if pooled_sd == 0 else (mean_a - mean_b) / pooled_sd


def kish_ess(w: np.ndarray) -> float:
    return (w.sum() ** 2) / (w ** 2).sum()


def weighted_two_proportion_test(y1, w1, y2, w2):
    p1 = np.average(y1, weights=w1)
    p2 = np.average(y2, weights=w2)
    n1_eff = kish_ess(w1)
    n2_eff = kish_ess(w2)
    p_pool = (p1 * n1_eff + p2 * n2_eff) / (n1_eff + n2_eff)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1_eff + 1 / n2_eff))
    z = (p1 - p2) / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    return p1, p2, n1_eff, n2_eff, z, pval


def load_loan_predictions() -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds[preds["representation"] == "structured"].copy()
    preds["y_pred"] = (preds["y_prob"] >= THRESHOLD).astype(int)

    labels = pd.read_csv(LABELS_CSV, dtype={"id": str, "zip3": str})
    labels["id"] = labels["id"].astype("int64")
    return preds.merge(labels[["id", "zip3"]], on="id", how="inner", validate="many_to_one")


def main() -> None:
    validated = pd.read_csv(VALIDATED_CSV, dtype={"zip3": str})
    ses_index = pd.read_csv(SES_INDEX_CSV, dtype={"zip3": str})[["zip3", "SESDisadvantage"]]
    ses_groups = pd.read_csv(SES_GROUPS_CSV, dtype={"zip3": str})[["zip3", "SES_group"]]
    demo_groups = pd.read_csv(DEMO_GROUPS_CSV, dtype={"zip3": str})

    groups = (
        validated.merge(ses_index, on="zip3")
        .merge(ses_groups, on="zip3")
        .merge(demo_groups[["zip3", "BlackConcentration"]], on="zip3")
    )

    loan_preds = load_loan_predictions()

    global MODELS
    MODELS = sorted(loan_preds["model"].unique().tolist())
    print(f"Models found in predictions file: {MODELS}")

    all_weights, balance_rows, fairness_rows = [], [], []

    for direction in DIRECTIONS:
        label = direction["label"]
        treatment_col = direction["treatment_col"]
        score_col = direction["stratify_score_col"]

        print(f"\n{'='*70}\nDIRECTION: {label}\n{'='*70}")
        strata = decile_strata(groups[score_col])
        weights = compute_cem_weights(groups, treatment_col, strata)
        weights["direction"] = label
        all_weights.append(weights)

        zip3_info = weights.merge(groups[["zip3"] + direction["balance_covariates"]], on="zip3")

        for covariate in direction["balance_covariates"]:
            treated = zip3_info[zip3_info[treatment_col] == "High"]
            control = zip3_info[zip3_info[treatment_col] == "Low"]

            unweighted = weighted_smd(treated.assign(weight=1.0), control.assign(weight=1.0), covariate)
            matched = weighted_smd(treated, control, covariate)

            balance_rows.append({
                "direction": label, "covariate": covariate,
                "SMD_before_matching": round(unweighted, 4),
                "SMD_after_matching": round(matched, 4),
            })

        df = loan_preds.merge(weights[["zip3", "weight"]], on="zip3", how="inner")
        df = df.merge(groups[["zip3", treatment_col]], on="zip3")

        for model in MODELS:
            sub = df[df["model"] == model]
            for weighted in [False, True]:
                w_col = sub["weight"] if weighted else pd.Series(1.0, index=sub.index)

                neg = sub[sub["y_true"] == 0]
                w_neg = w_col.loc[neg.index]
                low_mask = neg[treatment_col] == "Low"
                high_mask = neg[treatment_col] == "High"

                p_low, p_high, n_low_eff, n_high_eff, z, pval = weighted_two_proportion_test(
                    neg.loc[low_mask, "y_pred"], w_neg.loc[low_mask],
                    neg.loc[high_mask, "y_pred"], w_neg.loc[high_mask],
                )
                fairness_rows.append({
                    "direction": label, "model": model,
                    "metric": "FPR", "weighted": weighted,
                    "Low": round(p_low, 4), "High": round(p_high, 4),
                    "gap_High_minus_Low": round(p_high - p_low, 4),
                    "n_eff_Low": round(n_low_eff, 1), "n_eff_High": round(n_high_eff, 1),
                    "z": round(z, 2), "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
                })

    weights_out = pd.concat(all_weights, ignore_index=True)
    balance_out = pd.DataFrame(balance_rows)
    fairness_out = pd.DataFrame(fairness_rows)

    print("\n-- Balance: SMD before vs. after decile-level CEM matching --")
    print(balance_out.to_string(index=False))
    print("\n-- FPR gap: unweighted vs. decile-CEM-weighted --")
    print(fairness_out.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    weights_out.to_csv(WEIGHTS_CSV, index=False)
    balance_out.to_csv(BALANCE_CSV, index=False)
    fairness_out.to_csv(FAIRNESS_CSV, index=False)
    print(f"\nSaved: {WEIGHTS_CSV}")
    print(f"Saved: {BALANCE_CSV}")
    print(f"Saved: {FAIRNESS_CSV}")


if __name__ == "__main__":
    main()
