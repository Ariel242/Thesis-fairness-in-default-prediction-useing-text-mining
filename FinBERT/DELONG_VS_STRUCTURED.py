# -*- coding: utf-8 -*-
"""
DeLong Test: Structured vs Structured+FinBERT (768-dim and PCA-50), fixed AND
grid-searched hyperparameters — same test set, per fold
=============================================================================================
Fills the gap flagged by Dictionarys/scripts/08_three_tier_comparison_report.py: the
FinBERT baseline/gridsearch scripts (BASELINE_768.py, BASELINE_PCA50.py, GRIDSEARCH_768.py,
GRIDSEARCH_PCA50.py) never ran a DeLong test against the Structured baseline (only
FinBERT-768 vs FinBERT-PCA50 was tested directly, in DELONG_768_VS_PCA50.py), so those
four rows in Dictionarys/results/three_tier_master_comparison.csv read "not run".

APPROACH: per fold, fit the Structured-only arm AND both FinBERT arms (768-dim raw,
PCA-50) side by side on the identical test set, for BOTH configs:
  - "fixed"       -- analysis/BASELINE.py's fixed hyperparameters (identical values
                     verified against BASELINE_768.py / BASELINE_PCA50.py).
  - "grid search" -- per-fold winning hyperparameters, REUSED from the already-completed
                     runs rather than re-searched (no leakage risk either way, since the
                     search itself never touches the outer test fold):
                       results/walk_forward_gridsearch/wf_best_params.csv       (Structured)
                       FinBERT/results/gridsearch_768/wf_best_params.csv        (FinBERT-768)
                       FinBERT/results/gridsearch_pca50/wf_best_params.csv      (FinBERT-PCA50)
                     Each arm uses its OWN winning config (the config that was actually
                     reported for that arm in the master comparison table) -- this is the
                     fair comparison: "each representation's best config vs the other's".

Structured block (STRUCT_COLS, EXCLUDE_COLS, impute+scale) and both FinBERT blocks
(768 raw + scaler; PCA-50 fit train-only + scaler) are built exactly as in
BASELINE_768.py / BASELINE_PCA50.py -- verified line-for-line against those files.
The Structured-only scaled block is fit ONCE per fold and reused across all three arms
(mathematically identical to each script independently refitting the same imputer/scaler
on the same STRUCT_COLS/train rows -- just avoids redundant computation).

Self-contained: does not import or modify BASELINE.py / BASELINE_768.py / BASELINE_PCA50.py
/ GRIDSEARCH_768.py / GRIDSEARCH_PCA50.py / analysis/GRIDSEARCH.py. Reuses their SAVED
wf_best_params.csv outputs only.

CHECKPOINTED like strict_temporal_v2/run_full.py: each fold's rows are appended to disk
immediately, with a completed_folds.txt marker. Safe to kill and re-run; already-completed
folds are skipped. RF n_jobs capped at 4 throughout (documented OOM history on this
13.7GB-RAM machine at n_jobs=-1 on wide/late folds -- see BASELINE_768.py's MEMORY NOTE).

OUTPUT (written under FinBERT/results/delong_vs_structured/ ONLY):
  fold_metrics.csv     -- AUC/PR-AUC/Brier/GINI per fold x arm x config x model
  delong_raw.csv       -- per fold x model x config x representation, DeLong result
                            (Arm_A=Structured, Arm_B=Structured+FinBERT-768 / PCA50)
  completed_folds.txt  -- resume marker

Usage:
  python -m FinBERT.DELONG_VS_STRUCTURED             # run/resume all 14 folds
  python -m FinBERT.DELONG_VS_STRUCTURED --restart    # ignore checkpoint, start clean
"""

import argparse
import gc
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../F-TM-CR/FinBERT
BASE_DIR   = SCRIPT_DIR.parent                         # .../F-TM-CR

PATH_CSV     = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
PATH_FINBERT = SCRIPT_DIR / "data" / "finbert_desc_embeddings.parquet"

BEST_PARAMS_STRUCTURED = BASE_DIR / "results" / "walk_forward_gridsearch" / "wf_best_params.csv"
BEST_PARAMS_FB768      = SCRIPT_DIR / "results" / "gridsearch_768" / "wf_best_params.csv"
BEST_PARAMS_PCA50      = SCRIPT_DIR / "results" / "gridsearch_pca50" / "wf_best_params.csv"

OUT_DIR       = SCRIPT_DIR / "results" / "delong_vs_structured"
METRICS_CSV   = OUT_DIR / "fold_metrics.csv"
DELONG_CSV    = OUT_DIR / "delong_raw.csv"
MARKER_FILE   = OUT_DIR / "completed_folds.txt"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"
ZIP_COL    = "zip3"

MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

RANDOM_STATE     = 242
N_PCA_COMPONENTS = 50
SEM_COLS         = [f"sem_{i:03d}" for i in range(768)]

# Fixed hyperparameters -- verified identical to analysis/BASELINE.py / BASELINE_768.py / BASELINE_PCA50.py
LR_FIXED_PARAMS = dict(penalty="l2", C=0.3, solver="saga", max_iter=1000,
                        class_weight="balanced", random_state=RANDOM_STATE)
RF_FIXED_PARAMS = dict(n_estimators=500, min_samples_leaf=20, class_weight="balanced",
                        n_jobs=4, random_state=RANDOM_STATE)
XGB_FIXED_KW = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                     eval_metric="auc", use_label_encoder=False, verbosity=0,
                     n_jobs=4, random_state=RANDOM_STATE)

# Grid-search "FIXED" halves -- verified identical to analysis/GRIDSEARCH.py / GRIDSEARCH_768.py / GRIDSEARCH_PCA50.py
LR_GRID_FIXED  = dict(penalty="l2", solver="saga", max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
XGB_GRID_FIXED = dict(n_estimators=300, subsample=0.8, eval_metric="auc",
                       use_label_encoder=False, verbosity=0, n_jobs=4, random_state=RANDOM_STATE)
RF_GRID_FIXED  = dict(n_estimators=500, class_weight="balanced", n_jobs=4, random_state=RANDOM_STATE)

MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]
CONFIGS = ["fixed", "grid"]
REPRESENTATIONS = ["fb768", "pca50"]  # each compared against "structured" (Arm_A)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def _append_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _read_completed_folds() -> set:
    if not MARKER_FILE.exists():
        return set()
    return {int(l.strip()) for l in MARKER_FILE.read_text().splitlines() if l.strip()}


def _mark_fold_complete(fold_num: int) -> None:
    with open(MARKER_FILE, "a", encoding="utf-8") as f:
        f.write(f"{fold_num}\n")


# ============================================================
# DELONG TEST (copied verbatim from Dictionarys/scripts/07_lexicon_structured_ablation.py)
# ============================================================
def delong_auc_test(y_true, p1, p2):
    def auc_and_kernel(y, p):
        pos = p[y == 1]; neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([np.mean(pi > neg) + 0.5 * np.mean(pi == neg) for pi in pos])
        V01 = np.array([np.mean(pj < pos) + 0.5 * np.mean(pj == pos) for pj in neg])
        return V10.mean(), V10, V01, n1, n0

    y = np.asarray(y_true).astype(int)
    auc1, V10_1, V01_1, n1, n0 = auc_and_kernel(y, np.asarray(p1))
    auc2, V10_2, V01_2, _, _ = auc_and_kernel(y, np.asarray(p2))

    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)
    var_diff = (S10[0, 0] / n1 + S01[0, 0] / n0) + (S10[1, 1] / n1 + S01[1, 1] / n0) \
             - 2 * (S10[0, 1] / n1 + S01[0, 1] / n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan, np.nan
    diff = auc1 - auc2
    se = np.sqrt(var_diff)
    z = diff / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval, diff - 1.96 * se, diff + 1.96 * se


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    print("Loading structured data...")
    df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

    GRADE_ORDER = list("ABCDEFG")
    SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]
    if "grade" in df.columns:
        df["grade_ord"] = df["grade"].astype(str).str.strip().map(
            {g: i + 1 for i, g in enumerate(GRADE_ORDER)}).astype("float32")
    if "sub_grade" in df.columns:
        df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map(
            {s: i + 1 for i, s in enumerate(SUBGRADE_ORDER)}).astype("float32")

    global STRUCT_COLS
    EXCLUDE_COLS = {TARGET_COL, DATE_COL, ZIP_COL,
                     "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                     "zip_code", "emp_title", "title", "desc", "funded_ratio",
                     "grade", "sub_grade",
                     "last_fico_range_high", "last_fico_range_low"}
    STRUCT_COLS = [c for c in df.columns
                   if c not in EXCLUDE_COLS
                   and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                       np.int8, "Int64", "float32", "float64"]]
    print(f"  Structured features: {len(STRUCT_COLS)}")

    print("Loading FinBERT embeddings...")
    df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
    for c in SEM_COLS:
        df_fb[c] = df_fb[c].astype(np.float32)
    n_before = len(df)
    df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
    assert len(df) == n_before, "Merge changed row count -- id is not a clean 1:1 key"
    assert df["has_desc"].isna().sum() == 0, "Unmatched rows after FinBERT merge"
    df["has_desc"] = df["has_desc"].astype(np.float32)
    print(f"  Merged embeddings for {len(df):,} rows ({df['has_desc'].mean():.1%} have a description)")
    return df


def build_folds(df):
    months = df[DATE_COL].dt.to_period("M")
    all_periods = sorted(months.unique())
    folds = []
    fold_start_idx = MIN_TRAIN_MONTHS
    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]
        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, TARGET_COL].sum() >= MIN_TEST_DEFAULTS:
                break
            test_end_idx += 1
        if test_end_idx >= len(all_periods):
            break
        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)
        folds.append({
            "fold": len(folds) + 1, "train_cutoff": str(train_cutoff), "test_end": str(test_period),
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_test_defaults": int(df.loc[te_mask, TARGET_COL].sum()),
            "tr_idx": df.index[tr_mask].tolist(), "te_idx": df.index[te_mask].tolist(),
        })
        fold_start_idx += STEP_MONTHS
    assert len(folds) == 14, f"Expected 14 folds, got {len(folds)}."
    return folds


def build_matrices(df_tr, df_te, y_tr):
    """Returns X_structured (train/test) and both FinBERT-augmented matrices,
    built exactly as in BASELINE_768.py / BASELINE_PCA50.py."""
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_te_s = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))
    scaler_s = StandardScaler()
    X_tr_s = scaler_s.fit_transform(X_tr_s)
    X_te_s = scaler_s.transform(X_te_s)

    # -- FinBERT-768 raw --
    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), df_tr[SEM_COLS].values.astype(np.float32)])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), df_te[SEM_COLS].values.astype(np.float32)])
    scaler_fb768 = StandardScaler()
    X_tr_fb768 = np.hstack([X_tr_s, scaler_fb768.fit_transform(fb_tr_raw)])
    X_te_fb768 = np.hstack([X_te_s, scaler_fb768.transform(fb_te_raw)])

    # -- FinBERT-PCA50 --
    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver="randomized", random_state=RANDOM_STATE)
    pca_tr = pca.fit_transform(df_tr[SEM_COLS].values.astype(np.float32))
    pca_te = pca.transform(df_te[SEM_COLS].values.astype(np.float32))
    pca50_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
    pca50_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
    scaler_pca50 = StandardScaler()
    X_tr_pca50 = np.hstack([X_tr_s, scaler_pca50.fit_transform(pca50_tr_raw)])
    X_te_pca50 = np.hstack([X_te_s, scaler_pca50.transform(pca50_te_raw)])

    return {
        "structured": (X_tr_s, X_te_s),
        "fb768": (X_tr_fb768, X_te_fb768),
        "pca50": (X_tr_pca50, X_te_pca50),
    }


def fit_predict(arm, config, model_name, X_tr, y_tr, X_te, best_params_row):
    if model_name == "Logistic":
        C = 0.3 if config == "fixed" else float(best_params_row["C"])
        params = LR_FIXED_PARAMS if config == "fixed" else {**LR_GRID_FIXED, "C": C}
        model = LogisticRegression(**params)
    elif model_name == "XGBoost":
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        if config == "fixed":
            model = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_FIXED_KW)
        else:
            model = xgb.XGBClassifier(max_depth=int(best_params_row["max_depth"]),
                                       learning_rate=float(best_params_row["learning_rate"]),
                                       scale_pos_weight=spw, **XGB_GRID_FIXED)
    else:  # RandomForest
        if config == "fixed":
            model = RandomForestClassifier(**RF_FIXED_PARAMS)
        else:
            model = RandomForestClassifier(min_samples_leaf=int(best_params_row["min_samples_leaf"]), **RF_GRID_FIXED)

    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    del model
    return p


def run_fold(df, fold, best_params, fold_num_for_lookup):
    tr_idx, te_idx = fold["tr_idx"], fold["te_idx"]
    df_tr, df_te = df.loc[tr_idx], df.loc[te_idx]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    matrices = build_matrices(df_tr, df_te, y_tr)

    metric_rows, delong_rows = [], []
    fold_preds = {}  # (arm, config, model) -> p

    for arm in ["structured", "fb768", "pca50"]:
        X_tr, X_te = matrices[arm]
        for config in CONFIGS:
            for model_name in MODEL_NAMES:
                bp_row = None
                if config == "grid":
                    bp_df = best_params[arm]
                    bp_row = bp_df[(bp_df["fold"] == fold_num_for_lookup) & (bp_df["Model"] == model_name)].iloc[0]
                p = fit_predict(arm, config, model_name, X_tr, y_tr, X_te, bp_row)
                fold_preds[(arm, config, model_name)] = p

                auc = roc_auc_score(y_te, p)
                metric_rows.append({
                    "fold": fold["fold"], "arm": arm, "config": config, "Model": model_name,
                    "n_train": fold["n_train"], "n_test": fold["n_test"], "n_defaults": fold["n_test_defaults"],
                    "AUC": round(auc, 4),
                    "PR_AUC": round(average_precision_score(y_te, p), 4),
                    "Brier": round(brier_score_loss(y_te, p), 4),
                    "GINI": round(2 * auc - 1, 4),
                })
        del X_tr, X_te

    for config in CONFIGS:
        for model_name in MODEL_NAMES:
            p_struct = fold_preds[("structured", config, model_name)]
            for repr_name in REPRESENTATIONS:
                p_repr = fold_preds[(repr_name, config, model_name)]
                auc_a, auc_b, z, pval, ci_lo, ci_hi = delong_auc_test(y_te, p_struct, p_repr)
                delong_rows.append({
                    "fold": fold["fold"], "Model": model_name, "config": config,
                    "Arm_A": "Structured",
                    "Arm_B": "Structured+FinBERT-768" if repr_name == "fb768" else "Structured+FinBERT-PCA50",
                    "AUC_A": round(auc_a, 4), "AUC_B": round(auc_b, 4),
                    "Delta_AUC": round(auc_b - auc_a, 4),
                    "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
                    "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
                    "CI_95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                    "CI_95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    _append_csv(METRICS_CSV, metric_rows)
    _append_csv(DELONG_CSV, delong_rows)
    _mark_fold_complete(fold["fold"])

    del matrices, fold_preds, df_tr, df_te
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.restart:
        for p in [METRICS_CSV, DELONG_CSV, MARKER_FILE]:
            if p.exists():
                p.unlink()
        print("--restart: cleared checkpoint.")

    completed = _read_completed_folds()
    if completed:
        print(f"Resuming: {len(completed)} fold(s) already completed ({sorted(completed)}) -- skipping.")

    best_params = {
        "structured": pd.read_csv(BEST_PARAMS_STRUCTURED),
        "fb768": pd.read_csv(BEST_PARAMS_FB768),
        "pca50": pd.read_csv(BEST_PARAMS_PCA50),
    }

    df = load_data()
    folds = build_folds(df)
    print(f"Fold count verified: {len(folds)}")

    run_start = time.perf_counter()
    for fold in folds:
        if fold["fold"] in completed:
            print(f"Fold {fold['fold']}: already completed -- skipping.")
            continue
        t0 = time.perf_counter()
        run_fold(df, fold, best_params, fold["fold"])
        print(f"Fold {fold['fold']} done in {time.perf_counter() - t0:.1f}s "
              f"({(time.perf_counter() - run_start) / 60:.1f}min elapsed total). "
              f"(checkpointed -- safe to stop and resume)")

    print(f"\nAll folds complete. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
