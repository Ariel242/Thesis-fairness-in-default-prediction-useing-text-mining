# -*- coding: utf-8 -*-
"""
11_entropy_balanced_sensitivity_analysis.py
====================================
Multivariate follow-up to Stages 9-10, using the SECOND matching method the doc names
in section 7: Entropy Balancing (Hainmueller, 2012 -- cited in the doc's own
methodological sources). Not run by run_pipeline.py, same reasoning as Stages 7-10.

WHY THIS SCRIPT EXISTS
-----------------------
Stages 9 and 10 both matched on ONE stratifying score at a time (SES_group's 3
tertiles, then SESDisadvantage's 10 deciles) -- and both left several of the 7 raw SES
covariates poorly balanced for the BlackConcentration direction (education, income and
home value got WORSE, not better, after matching). The diagnosis in Stage 10's own
docstring: SESDisadvantage is a MEAN of 7 percentile ranks, so two ZIP3s can share the
same average score with very different underlying profiles -- matching on the average
alone cannot fix that.

Entropy Balancing solves a different, more direct problem: instead of stratifying on
one score and hoping it drags the other 6 covariates along, it solves DIRECTLY for
control-group weights such that the WEIGHTED MEAN of every one of the 7 (or 6, for the
SES_group direction) covariates simultaneously matches the treatment group's mean, to
near-numerical-precision -- multivariate mean balance by construction, not as a
side-effect of a single stratification variable.

METHOD (Hainmueller, 2012)
----------------------------
For control-group covariates X (n x k, standardized) and treatment-group target means
m (a k-vector), entropy balancing finds weights w minimizing the (negative) entropy
sum(w_i * log(w_i)) subject to sum(w_i * X_i) = m and sum(w_i) = n. The solution has a
closed dual form: w_i(lambda) = exp(-lambda . (X_i - m)) / sum_j exp(-lambda . (X_j - m))
for a lambda in R^k found by minimizing the convex "log partition" objective

    A(lambda) = logsumexp(-lambda . (X_i - m))

(implemented with scipy's numerically-stable `logsumexp`). At A's minimum, the gradient
-- the weighted mean of (X_i - m) -- is exactly zero, i.e. the weighted covariate means
equal the target means exactly. This is solved here with `scipy.optimize.minimize`
(BFGS), no external causal-inference package required.

Covariates are standardized (zero mean, unit variance, computed once on the pooled
matched sample) before balancing, purely for optimizer numerical stability -- the
resulting weights and the reported SMDs are unaffected by this rescaling (SMD itself is
already scale-free).

Applied to BOTH directions, symmetric with Stage 5/9/10's own covariate lists:

    A) BlackConcentration (High vs Low), entropy-balanced on all 7 SES_COVARIATES
    B) SES_group (High vs Low), entropy-balanced on all 6 DEMOGRAPHIC_COVARIATES
       (a stricter check than Stage 9/10's single-covariate share_black_nh match)

WEIGHT DIAGNOSTICS
--------------------
Unlike CEM, entropy balancing never prunes units -- every control ZIP3 gets a weight
in the reweighted sample. But weights can become extreme when treatment/control overlap
is poor on some covariate, which inflates variance without the CEM diagnostic (pruned
count) to flag it. This script instead prints the weight distribution (min/median/max)
and Kish's effective sample size (the same metric Stages 9-10 already use for
significance testing) so that cost is visible here too.

INPUT:  census/results/01_census_data_validated.csv
        census/results/03_ses_groups.csv
        census/results/04_demographic_concentration_groups.csv
        census/results/06_loan_level_zip3_group_labels.csv
        results/strict_temporal_v2/predictions.csv   (repo root)
OUTPUT: census/results/11_entropy_weights.csv
        census/results/11_balance_before_after_entropy.csv
        census/results/11_fairness_before_after_entropy.csv
"""

import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
BASE_DIR = CENSUS_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

VALIDATED_CSV = RESULTS_DIR / "01_census_data_validated.csv"
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

WEIGHTS_CSV = RESULTS_DIR / f"11_entropy_weights{SUFFIX}.csv"
BALANCE_CSV = RESULTS_DIR / f"11_balance_before_after_entropy{SUFFIX}.csv"
FAIRNESS_CSV = RESULTS_DIR / f"11_fairness_before_after_entropy{SUFFIX}.csv"

THRESHOLD = 0.5
MODELS = ["Logistic", "XGBoost"]

SES_COVARIATES = [
    "poverty_rate", "unemployment_rate", "share_bachelor_plus",
    "homeownership_rate", "uninsured_rate",
    "median_income_approx", "median_home_value_approx",
]
DEMOGRAPHIC_COVARIATES = [
    "share_black_nh", "share_hispanic", "share_asian_nh",
    "share_white_nh", "share_foreign_born", "share_ltd_english_hh",
]
DIRECTIONS = [
    {
        "label": "BlackConcentration (entropy-balanced on 7 SES covariates)",
        "treatment_col": "BlackConcentration",
        "covariates": SES_COVARIATES,
    },
    {
        "label": "SES_group (entropy-balanced on 6 demographic covariates)",
        "treatment_col": "SES_group",
        "covariates": DEMOGRAPHIC_COVARIATES,
    },
]


def entropy_balance_weights(X_control: np.ndarray, target_mean: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Returns weights for the control group (length = n_control, summing to
    n_control) such that the weighted mean of every column of X_control equals
    target_mean, via the Hainmueller (2012) dual formulation.

    RIDGE TERM (why this exists): the unregularized dual objective is unbounded
    whenever target_mean sits outside what the control group can actually reach on
    some covariate (or the covariates are collinear/skewed enough that lambda has to
    grow without limit to get arbitrarily close) -- BFGS then diverges to +-inf and
    returns NaN weights, exactly what happened here on first attempt with the 6
    skewed demographic covariates. Adding `0.5 * ridge * ||lambda||^2` keeps the
    objective strictly convex and bounded everywhere, at the cost of only
    approximate (not exact) mean balance when the unregularized target was already
    unreachable. When exact balance IS reachable (as for the 7 SES covariates below),
    a small ridge changes the solution negligibly."""
    n, k = X_control.shape
    centered = X_control - target_mean  # (n, k)

    def objective(lam):
        return logsumexp(-centered @ lam) + 0.5 * ridge * (lam @ lam)

    def grad(lam):
        z = -centered @ lam
        log_w = z - logsumexp(z)  # log of normalized (sum-to-1) weights
        w = np.exp(log_w)
        return -(centered * w[:, None]).sum(axis=0) + ridge * lam

    result = minimize(objective, x0=np.zeros(k), jac=grad, method="BFGS")
    if not result.success:
        print(f"  WARNING: entropy-balance optimizer did not converge ({result.message}).")
    z = -centered @ result.x
    w = np.exp(z - logsumexp(z))  # sums to 1
    return w * n  # rescale so weights sum to n_control (matches Stage 9/10 convention)


def weighted_mean_var(x: pd.Series, w: np.ndarray) -> tuple:
    mean = np.average(x, weights=w)
    var = np.average((x - mean) ** 2, weights=w)
    return mean, var


def weighted_smd(a_vals, a_w, b_vals, b_w) -> float:
    mean_a, var_a = weighted_mean_var(a_vals, a_w)
    mean_b, var_b = weighted_mean_var(b_vals, b_w)
    pooled_sd = ((var_a + var_b) / 2) ** 0.5
    return 0.0 if pooled_sd == 0 else (mean_a - mean_b) / pooled_sd


def kish_ess(w: np.ndarray) -> float:
    return (w.sum() ** 2) / (w ** 2).sum()


def weighted_two_proportion_test(y1, w1, y2, w2):
    p1 = np.average(y1, weights=w1)
    p2 = np.average(y2, weights=w2)
    n1_eff = kish_ess(np.asarray(w1))
    n2_eff = kish_ess(np.asarray(w2))
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
    ses_groups = pd.read_csv(SES_GROUPS_CSV, dtype={"zip3": str})[["zip3", "SES_group"]]
    demo_groups = pd.read_csv(DEMO_GROUPS_CSV, dtype={"zip3": str})[["zip3", "BlackConcentration"]]
    groups = validated.merge(ses_groups, on="zip3").merge(demo_groups, on="zip3")

    loan_preds = load_loan_predictions()

    global MODELS
    MODELS = sorted(loan_preds["model"].unique().tolist())
    print(f"Models found in predictions file: {MODELS}")

    all_weights, balance_rows, fairness_rows = [], [], []

    for direction in DIRECTIONS:
        label = direction["label"]
        treatment_col = direction["treatment_col"]
        covariates = direction["covariates"]

        print(f"\n{'='*70}\nDIRECTION: {label}\n{'='*70}")
        sub = groups[groups[treatment_col].isin(["Low", "High"])].copy()
        treated = sub[sub[treatment_col] == "High"]
        control = sub[sub[treatment_col] == "Low"]

        # standardize on the pooled matched sample, for optimizer stability only
        pooled_mean = sub[covariates].mean()
        pooled_std = sub[covariates].std()
        X_treated = ((treated[covariates] - pooled_mean) / pooled_std).values
        X_control = ((control[covariates] - pooled_mean) / pooled_std).values
        target_mean = X_treated.mean(axis=0)

        w_control = entropy_balance_weights(X_control, target_mean)
        w_treated = np.ones(len(treated))

        ess_control = kish_ess(w_control)
        print(f"  Control weights: n={len(w_control)}, "
              f"min={w_control.min():.3f}, median={np.median(w_control):.3f}, max={w_control.max():.3f}. "
              f"Kish effective n: {ess_control:.1f} (of {len(w_control)} raw).")

        direction_weights = pd.concat([
            pd.DataFrame({"zip3": treated["zip3"].values, treatment_col: "High", "weight": w_treated}),
            pd.DataFrame({"zip3": control["zip3"].values, treatment_col: "Low", "weight": w_control}),
        ], ignore_index=True)
        direction_weights["direction"] = label
        all_weights.append(direction_weights)

        # -- balance check: every covariate at once ------------------------------------
        for covariate in covariates:
            smd_before = weighted_smd(
                treated[covariate].values, np.ones(len(treated)),
                control[covariate].values, np.ones(len(control)),
            )
            smd_after = weighted_smd(
                treated[covariate].values, w_treated,
                control[covariate].values, w_control,
            )
            balance_rows.append({
                "direction": label, "covariate": covariate,
                "SMD_before_matching": round(smd_before, 4),
                "SMD_after_matching": round(smd_after, 4),
            })

        # -- fairness metrics: unweighted vs. entropy-weighted -------------------------
        df = loan_preds.merge(direction_weights[["zip3", "weight"]], on="zip3", how="inner")
        df = df.merge(groups[["zip3", treatment_col]], on="zip3")

        for model in MODELS:
            msub = df[df["model"] == model]
            for weighted in [False, True]:
                w_col = msub["weight"] if weighted else pd.Series(1.0, index=msub.index)

                neg = msub[msub["y_true"] == 0]
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

    print("\n-- Balance: SMD before vs. after entropy balancing (all covariates at once) --")
    print(balance_out.to_string(index=False))
    print("\n-- FPR gap: unweighted vs. entropy-weighted --")
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
