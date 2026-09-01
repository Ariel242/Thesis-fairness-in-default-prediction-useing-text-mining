# -*- coding: utf-8 -*-
"""
15_calibration_by_group.py
====================================
The test that distinguishes "the gap reflects a real base-rate difference" from
"the model systematically over-predicts risk in disadvantaged areas". Stages 08-14
established that a robust FPR/TPR gap exists for SES_group and does not exist for the
demographic groupings -- but a gap alone cannot tell those two explanations apart,
because both produce one.

WHY RAW CALIBRATION IS THE WRONG QUESTION HERE
------------------------------------------------
All models in this project are fit with `class_weight="balanced"` (Logistic) or
`scale_pos_weight` (XGBoost), which deliberately inflates predicted probabilities far
above the ~15% base rate -- the mean predicted probability is around 0.45. So the model
is badly calibrated in ABSOLUTE terms for every group, by construction, and asking "is
p=0.3 really a 30% default rate" would only re-measure that intentional reweighting.

The fairness-relevant question is whether that distortion is THE SAME across groups.
That is the `sufficiency` criterion: P(Y=1 | score=s, group=a) == P(Y=1 | score=s,
group=b). If a Low-SES loan and a High-SES loan carrying the same score really do
default at the same rate, then a common threshold treats them consistently and the
observed FPR gap is fully attributable to the score distribution being shifted (a
base-rate effect). If, at the same score, High-SES loans actually default LESS often
than Low-SES ones, the model is over-stating their risk -- that is a genuine,
model-attributable disparity, not an artifact of differing base rates.

THE FOUR ANALYSES (per the plan agreed 2026-08-31)
---------------------------------------------------
(a) CALIBRATION CURVES -- predictions binned into deciles of predicted probability
    (bin edges computed once per fold on the whole fold, so both groups share identical
    edges and are directly comparable); per bin x group, mean predicted probability vs.
    observed default rate. This is the reliability-diagram data, in tabular form.

(b) CALIBRATION SLOPE AND INTERCEPT -- a logistic regression of y_true on logit(y_prob),
    fit separately per group. A perfectly calibrated model gives intercept 0, slope 1.
    Here the absolute values are meaningless (see above); what matters is the DIFFERENCE
    between groups. Intercept difference = systematic over/under-prediction of one group
    at every score. Slope difference = one group's scores separate risk better than the
    other's.

(c) ECE -- Expected Calibration Error per group: the bin-count-weighted mean absolute
    gap between predicted and observed rates. A single summary number, again meaningful
    here as a between-group comparison, not as an absolute.

(d) SUFFICIENCY TEST -- one pooled logistic regression per fold with an interaction:
        y_true ~ logit(p) + is_high + logit(p):is_high
    The `is_high` coefficient tests an intercept shift (the key quantity: does group
    membership change the default rate at a fixed score?), and the interaction tests a
    slope difference. A significant NEGATIVE `is_high` coefficient means High-group
    loans default LESS than their score implies -- i.e. the model over-predicts their
    risk, the finding that would constitute genuine model-attributable unfairness.

PER-FOLD, THEN AGGREGATED: every quantity is computed within each walk-forward fold and
then summarized across folds (mean, and count of folds significant with BH-FDR applied
across the 14 folds) -- never on pooled predictions, since mixing chronologically
distinct periods with different base rates would itself create apparent miscalibration.
Runs for all 7 representations, since text could affect calibration even where it did
not affect the gap (Stage 13).

Groupings covered: SES_group and BlackConcentration -- the two that Stages 08-11
examined in depth. The other five demographic groupings showed no gap to explain and
lose 4-7 of 14 folds to the small-cell guard, so they are not carried here.

INPUT:  results/strict_temporal_v2/predictions.csv   (repo root)
        census/results/06_loan_level_zip3_group_labels.csv
OUTPUT: census/results/15_calibration_curves.csv     (decile x group x fold x repr x model)
        census/results/15_calibration_fit.csv        (slope/intercept/ECE per group x fold)
        census/results/15_sufficiency_test.csv       (interaction test per fold)
        census/results/15_calibration_summary.csv    (aggregated across folds)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

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

CURVES_CSV = RESULTS_DIR / f"15_calibration_curves{SUFFIX}.csv"
FIT_CSV = RESULTS_DIR / f"15_calibration_fit{SUFFIX}.csv"
SUFFICIENCY_CSV = RESULTS_DIR / f"15_sufficiency_test{SUFFIX}.csv"
SUMMARY_CSV = RESULTS_DIR / f"15_calibration_summary{SUFFIX}.csv"

REPRESENTATIONS = [
    "structured", "structured_has_desc", "lexicon",
    "tfidf_full", "tfidf_chi2", "finbert_768", "finbert_pca50",
]
MODELS = ["Logistic", "XGBoost"]
GROUPINGS = ["SES_group", "BlackConcentration"]
N_FOLDS = 14
N_BINS = 10
MIN_GROUP_DEFAULTS = 30   # same guard rationale as Stage 14
EPS = 1e-6                # keeps logit() finite at p exactly 0 or 1


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
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


def calibration_fit(y: np.ndarray, p: np.ndarray) -> dict:
    """(b) Logistic recalibration: y ~ 1 + logit(p). Returns intercept and slope."""
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"intercept": np.nan, "slope": np.nan}
    X = sm.add_constant(logit(p), has_constant="add")
    try:
        fit = sm.Logit(y, X).fit(disp=0)
        return {"intercept": float(fit.params[0]), "slope": float(fit.params[1])}
    except Exception:
        return {"intercept": np.nan, "slope": np.nan}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> float:
    """(c) ECE: bin-count-weighted mean |observed - predicted| across shared bins."""
    if len(y) == 0:
        return np.nan
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)
    total, ece = len(y), 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(y[mask].mean() - p[mask].mean())
    return ece


def sufficiency_test(df_low: pd.DataFrame, df_high: pd.DataFrame) -> dict:
    """(d) Pooled fit with an interaction:
           y ~ logit(p) + is_high + logit(p):is_high
    `coef_is_high` < 0 means High-group loans default LESS than their score implies,
    i.e. the model over-predicts their risk at a fixed score."""
    pooled = pd.concat([df_low.assign(is_high=0), df_high.assign(is_high=1)])
    if pooled["y_true"].nunique() < 2:
        return {k: np.nan for k in
                ["coef_is_high", "p_is_high", "coef_interaction", "p_interaction"]}
    lp = logit(pooled["y_prob"].values)
    X = pd.DataFrame({
        "logit_p": lp,
        "is_high": pooled["is_high"].values,
        "interaction": lp * pooled["is_high"].values,
    })
    X = sm.add_constant(X, has_constant="add")
    try:
        fit = sm.Logit(pooled["y_true"].values, X).fit(disp=0)
        return {
            "coef_is_high": float(fit.params["is_high"]),
            "p_is_high": float(fit.pvalues["is_high"]),
            "coef_interaction": float(fit.params["interaction"]),
            "p_interaction": float(fit.pvalues["interaction"]),
        }
    except Exception:
        return {k: np.nan for k in
                ["coef_is_high", "p_is_high", "coef_interaction", "p_interaction"]}


def main() -> None:
    print("Loading predictions...")
    preds = pd.read_csv(PREDICTIONS_CSV)

    global REPRESENTATIONS, MODELS
    available_reps = set(preds["representation"].unique())
    skipped = [r for r in REPRESENTATIONS if r not in available_reps]
    REPRESENTATIONS = [r for r in REPRESENTATIONS if r in available_reps]
    if skipped:
        print(f"  Not present in this predictions file, skipped: {skipped}")
    preds = preds[preds["representation"].isin(REPRESENTATIONS)]

    labels = pd.read_csv(LABELS_CSV, dtype={"id": str, "zip3": str})
    labels["id"] = labels["id"].astype("int64")
    df = preds.merge(labels[["id"] + GROUPINGS], on="id", how="inner", validate="many_to_one")
    MODELS = sorted(df["model"].unique().tolist())
    print(f"  {len(df):,} rows. Models: {MODELS}")

    curve_rows, fit_rows, suff_rows = [], [], []

    for representation in REPRESENTATIONS:
        rep_df = df[df["representation"] == representation]
        for model in MODELS:
            model_df = rep_df[rep_df["model"] == model]
            for fold in range(1, N_FOLDS + 1):
                fold_df = model_df[model_df["fold"] == fold]
                if fold_df.empty:
                    continue

                # Shared bin edges from the WHOLE fold, so both groups are comparable.
                edges = np.unique(np.percentile(
                    fold_df["y_prob"], np.linspace(0, 100, N_BINS + 1)))
                if len(edges) < 3:
                    continue

                for grouping in GROUPINGS:
                    low = fold_df[fold_df[grouping] == "Low"]
                    high = fold_df[fold_df[grouping] == "High"]
                    if (low["y_true"].sum() < MIN_GROUP_DEFAULTS
                            or high["y_true"].sum() < MIN_GROUP_DEFAULTS):
                        continue

                    base = {"representation": representation, "model": model,
                            "grouping": grouping, "fold": fold}

                    for level, side in [("Low", low), ("High", high)]:
                        y = side["y_true"].values
                        p = side["y_prob"].values

                        # (a) curves
                        idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"),
                                      0, len(edges) - 2)
                        for b in range(len(edges) - 1):
                            mask = idx == b
                            if mask.sum() == 0:
                                continue
                            curve_rows.append({
                                **base, "group_level": level, "bin": b,
                                "n": int(mask.sum()),
                                "mean_pred": round(float(p[mask].mean()), 4),
                                "observed_rate": round(float(y[mask].mean()), 4),
                            })

                        # (b) + (c)
                        fit_rows.append({
                            **base, "group_level": level, "n": len(side),
                            "n_defaults": int(y.sum()),
                            **{k: (round(v, 4) if v == v else np.nan)
                               for k, v in calibration_fit(y, p).items()},
                            "ECE": round(expected_calibration_error(y, p, edges), 4),
                        })

                    # (d)
                    suff_rows.append({**base, **{
                        k: (round(v, 5) if isinstance(v, float) and v == v else v)
                        for k, v in sufficiency_test(low, high).items()}})

        print(f"  done: {representation}")

    curves = pd.DataFrame(curve_rows)
    fits = pd.DataFrame(fit_rows)
    suff = pd.DataFrame(suff_rows)

    fam = ["representation", "model", "grouping"]
    for p_col, q_col in [("p_is_high", "q_is_high"), ("p_interaction", "q_interaction")]:
        suff[q_col] = suff.groupby(fam)[p_col].transform(benjamini_hochberg)

    # -- summary: between-group differences, aggregated across folds ------------------
    wide = fits.pivot_table(index=fam + ["fold"], columns="group_level",
                            values=["intercept", "slope", "ECE"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["intercept_diff"] = wide["intercept_High"] - wide["intercept_Low"]
    wide["slope_diff"] = wide["slope_High"] - wide["slope_Low"]
    wide["ECE_diff"] = wide["ECE_High"] - wide["ECE_Low"]

    summary_rows = []
    for keys, grp in wide.groupby(fam):
        s = suff[(suff["representation"] == keys[0]) & (suff["model"] == keys[1])
                 & (suff["grouping"] == keys[2])]
        d = grp["intercept_diff"].dropna()
        t_p = stats.ttest_1samp(d, 0).pvalue if len(d) > 2 else np.nan
        summary_rows.append({
            "representation": keys[0], "model": keys[1], "grouping": keys[2],
            "n_folds": len(grp),
            "mean_intercept_diff": round(grp["intercept_diff"].mean(), 4),
            "p_intercept_diff_vs0": round(t_p, 4) if t_p == t_p else np.nan,
            "mean_slope_diff": round(grp["slope_diff"].mean(), 4),
            "mean_ECE_Low": round(grp["ECE_Low"].mean(), 4),
            "mean_ECE_High": round(grp["ECE_High"].mean(), 4),
            "mean_coef_is_high": round(s["coef_is_high"].mean(), 4),
            "n_folds_sufficiency_sig_raw": int((s["p_is_high"] < 0.05).sum()),
            "n_folds_sufficiency_sig_BH": int((s["q_is_high"] < 0.05).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    print("\n-- Sufficiency: is_high coefficient (negative = model OVER-predicts risk "
          "for the High group at a fixed score) --")
    print(summary[summary["grouping"] == "SES_group"][
        ["representation", "model", "n_folds", "mean_coef_is_high",
         "n_folds_sufficiency_sig_raw", "n_folds_sufficiency_sig_BH",
         "mean_intercept_diff", "mean_ECE_Low", "mean_ECE_High"]].to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    curves.to_csv(CURVES_CSV, index=False)
    fits.to_csv(FIT_CSV, index=False)
    suff.round(5).to_csv(SUFFICIENCY_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    for path, obj in [(CURVES_CSV, curves), (FIT_CSV, fits),
                      (SUFFICIENCY_CSV, suff), (SUMMARY_CSV, summary)]:
        print(f"Saved: {path} ({len(obj):,} rows)")


if __name__ == "__main__":
    main()
