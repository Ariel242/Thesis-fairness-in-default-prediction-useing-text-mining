# -*- coding: utf-8 -*-
"""
09_matched_sensitivity_analysis.py
====================================
STAGE 9 -- the "balance second" half of doc section 7 ("טיפול בהבדלים בממד האחר") and
section 5's "Define first, balance second" principle. Not run automatically by
run_pipeline.py (same reasoning as Stages 7 and 8: downstream analysis, not group
construction).

WHY THIS SCRIPT EXISTS
-----------------------
Stage 8 found a statistically significant FPR/TPR gap for both BlackConcentration
(High vs Low) and SES_group (High vs Low) on the structured baseline model. But Stage 5
already showed BlackConcentration is confounded with SES (SMD 0.20-0.37) and SES_group
is confounded with demographics -- so part of Stage 8's gap could be "really" a SES
effect wearing a demographic-concentration label, or vice versa. Doc section 6 is
explicit that raw group comparisons cannot answer this on their own: "לא ניתן להסיק
באופן אוטומטי שההבדל קשור להרכב הגזעי עצמו... הפתרון אינו להשתמש ב-SES כדי להגדיר את
הקבוצות הדמוגרפיות" -- so the fix is NOT to change how the groups are defined
(Stages 3/4 stay exactly as built), it is to ask a narrower follow-up question:
"does the Stage 8 gap survive once we equalize the OTHER dimension's distribution
between the two groups being compared?"

METHOD: Coarsened Exact Matching, Iacus/King/Porro (2012) -- cited in the doc's own
methodological sources
------------------------------------------------------------------------------------
CEM reweights the CONTROL group so its distribution across strata of a covariate
matches the TREATMENT group's distribution, instead of discarding data. Concretely,
for treatment/control groups T and C, stratified by a covariate into strata s:

    weight(control unit in stratum s) = (T_s / T_total) / (C_s / C_total)
    weight(treatment unit)            = 1

where T_s / C_s are the treated/control counts IN stratum s, and T_total / C_total are
the totals across all strata that contain both groups (strata with only one group
present are "pruned" -- excluded entirely, since there is nothing to match them to).
After reweighting, the weighted control group has, by construction, the same stratum
distribution as the treatment group.

The two coarsening covariates used here are the categorical groups this pipeline
already built (Stages 3-4), which keeps every stratum well-populated (829 ZIP3s / 3
strata, not fragmented into dozens of near-empty cells the way coarsening all 7 raw SES
variables at once would be -- see doc section 6's own note that CEM "מתאים ... ליצור
weights ... כך שהקבוצות יהיו מאוזנות ביחס למומנטים מוגדרים של covariates", it does not
mandate a specific coarsening scheme):

    A) BlackConcentration (High=treatment, Low=control), stratified by SES_group
    B) SES_group (High=treatment, Low=control), stratified by BlackConcentration

The Medium level of the TREATMENT dimension is dropped before matching (not part of a
High-vs-Low contrast; matches doc section 11.4's own recommended High-vs-Low
robustness variant). The Medium level of the STRATIFYING dimension is kept (it is a
valid third stratum).

WEIGHTED FAIRNESS METRICS
---------------------------
Every loan inherits its ZIP3's CEM weight (matching happens at the ZIP3 level, per
Stage 6/doc section 9 -- the ZIP3 stays the unit of analysis). TPR/FPR/PPR become
weighted proportions: e.g. weighted_FPR = sum(w * 1[pred=1, true=0]) / sum(w * 1[true=0]).
Significance uses a survey-style weighted two-proportion test with Kish's effective
sample size (ESS = (sum w)^2 / sum(w^2)), the standard correction for the fact that
unequal weights reduce a weighted sample's effective information content below its raw
row count.

INPUT:  census/results/03_ses_groups.csv
        census/results/04_demographic_concentration_groups.csv
        census/results/01_census_data_validated.csv          (raw SES covariates, for the balance check)
        census/results/06_loan_level_zip3_group_labels.csv
        results/strict_temporal_v2/predictions.csv             (repo root -- same source as Stage 8)
OUTPUT: census/results/09_cem_weights.csv                     (zip3, direction, weight)
        census/results/09_balance_before_after_matching.csv   (does matching actually reduce the SMD?)
        census/results/09_fairness_before_after_matching.csv  (does the Stage-8 gap survive?)
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

SES_GROUPS_CSV = RESULTS_DIR / "03_ses_groups.csv"
DEMO_GROUPS_CSV = RESULTS_DIR / "04_demographic_concentration_groups.csv"
VALIDATED_CSV = RESULTS_DIR / "01_census_data_validated.csv"
LABELS_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"
PREDICTIONS_CSV = BASE_DIR / "results" / "strict_temporal_v2" / "predictions.csv"

WEIGHTS_CSV = RESULTS_DIR / "09_cem_weights.csv"
BALANCE_CSV = RESULTS_DIR / "09_balance_before_after_matching.csv"
FAIRNESS_CSV = RESULTS_DIR / "09_fairness_before_after_matching.csv"

THRESHOLD = 0.5
MODELS = ["Logistic", "XGBoost"]

# Each direction: (treatment column, its "high" level, its "low" level,
#                  stratifying column, raw covariates to check balance on)
SES_COVARIATES = [
    "poverty_rate", "unemployment_rate", "share_bachelor_plus",
    "homeownership_rate", "uninsured_rate",
    "median_income_approx", "median_home_value_approx",
]
DIRECTIONS = [
    {
        "label": "BlackConcentration (matched on SES_group)",
        "treatment_col": "BlackConcentration",
        "stratify_col": "SES_group",
        "balance_covariates": SES_COVARIATES,
    },
    {
        "label": "SES_group (matched on BlackConcentration)",
        "treatment_col": "SES_group",
        "stratify_col": "BlackConcentration",
        "balance_covariates": ["share_black_nh"],
    },
]


def compute_cem_weights(df: pd.DataFrame, treatment_col: str, stratify_col: str) -> pd.DataFrame:
    """Returns a zip3-level DataFrame (zip3, <treatment_col>, weight) for the High vs
    Low contrast of `treatment_col`, reweighted so the Low ("control") group's
    distribution across `stratify_col` matches the High ("treatment") group's."""
    sub = df[df[treatment_col].isin(["Low", "High"])].copy()

    strata_present_both = (
        sub.groupby(stratify_col)[treatment_col].nunique().pipe(lambda s: s[s == 2].index)
    )
    n_pruned = sub[~sub[stratify_col].isin(strata_present_both)]["zip3"].nunique()
    sub = sub[sub[stratify_col].isin(strata_present_both)].copy()

    t_total = (sub[treatment_col] == "High").sum()
    c_total = (sub[treatment_col] == "Low").sum()

    weights = []
    for stratum, grp in sub.groupby(stratify_col):
        t_s = (grp[treatment_col] == "High").sum()
        c_s = (grp[treatment_col] == "Low").sum()
        for _, row in grp.iterrows():
            if row[treatment_col] == "High":
                w = 1.0
            else:
                w = (t_s / t_total) / (c_s / c_total)
            weights.append({"zip3": row["zip3"], treatment_col: row[treatment_col], "weight": w})

    print(f"  Pruned strata: {n_pruned} ZIP3s dropped (stratum missing High or Low side).")
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
    """Kish's effective sample size: how many EQUALLY-weighted rows the weighted
    sample carries as much information as."""
    return (w.sum() ** 2) / (w ** 2).sum()


def weighted_two_proportion_test(y1, w1, y2, w2):
    """Survey-style weighted two-proportion z-test, using Kish ESS in place of the raw
    row count in the standard-error formula."""
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
    ses_groups = pd.read_csv(SES_GROUPS_CSV, dtype={"zip3": str})
    demo_groups = pd.read_csv(DEMO_GROUPS_CSV, dtype={"zip3": str})
    validated = pd.read_csv(VALIDATED_CSV, dtype={"zip3": str})
    groups = ses_groups.merge(demo_groups, on="zip3").merge(validated, on="zip3", suffixes=("", "_raw"))

    loan_preds = load_loan_predictions()

    all_weights, balance_rows, fairness_rows = [], [], []

    for direction in DIRECTIONS:
        label = direction["label"]
        treatment_col = direction["treatment_col"]
        stratify_col = direction["stratify_col"]

        print(f"\n{'='*70}\nDIRECTION: {label}\n{'='*70}")
        weights = compute_cem_weights(groups, treatment_col, stratify_col)
        weights["direction"] = label
        all_weights.append(weights)

        zip3_info = weights.merge(groups[["zip3"] + direction["balance_covariates"]], on="zip3")

        # -- balance check: weighted SMD vs. the original (unweighted) SMD ------------
        for covariate in direction["balance_covariates"]:
            treated = zip3_info[zip3_info[treatment_col] == "High"]
            control = zip3_info[zip3_info[treatment_col] == "Low"]

            unweighted = treated.assign(weight=1.0).pipe(
                lambda d: weighted_smd(d, control.assign(weight=1.0), covariate))
            matched = weighted_smd(treated, control, covariate)

            balance_rows.append({
                "direction": label, "covariate": covariate,
                "SMD_before_matching": round(unweighted, 4),
                "SMD_after_matching": round(matched, 4),
            })

        # -- fairness metrics: unweighted (Stage 8-style) vs. CEM-weighted ------------
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

    print("\n-- Balance: SMD before vs. after CEM matching --")
    print(balance_out.to_string(index=False))
    print("\n-- FPR gap: unweighted (raw, Stage 8) vs. CEM-weighted (matched) --")
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
