"""
DeLong Test: FinBERT-768 (raw) vs. FinBERT-PCA50 — same test set, per fold
=============================================================================================
Companion to BASELINE_768.py and BASELINE_PCA50.py. Neither of those scripts
saved raw per-row predicted probabilities (only aggregate fold metrics), so
a paired DeLong test needs a fresh run: per fold, build BOTH the raw
768-dim and PCA-50 FinBERT feature matrices (structured block is identical
and shared), fit each of the three models with the SAME fixed
hyperparameters as BASELINE_768.py / BASELINE_PCA50.py (not the tuned grid
versions — this isolates the dimensionality question from the tuning
question), and compare the two resulting AUCs on the same held-out test
set with DeLong's test (DeLong et al. 1988) — the standard paired test for
two correlated AUCs measured on identical samples.

No leakage: imputer/scaler (structured block), FinBERT scaler, and PCA are
all fit fresh per fold, on that fold's TRAIN rows only — identical
discipline to BASELINE_768.py / BASELINE_PCA50.py.

OUTPUT (written under FinBERT/ only):
  results/delong_768_vs_pca50/delong_per_fold.csv
  results/delong_768_vs_pca50/delong_summary.csv
"""

import os
import sys
import time
import gc
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
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

PATH_CSV     = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
PATH_FINBERT = SCRIPT_DIR / "data" / "finbert_desc_embeddings.parquet"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"
ZIP_COL    = "zip3"

MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

RANDOM_STATE = 242
SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
N_PCA_COMPONENTS = 50

LR_PARAMS = dict(penalty="l2", C=0.3, solver="saga", max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=20, class_weight="balanced", n_jobs=4, random_state=RANDOM_STATE)

# ============================================================
# 1. LOAD + MERGE (identical to BASELINE_768.py / BASELINE_PCA50.py)
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

GRADE_ORDER = list("ABCDEFG"); SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]
df["grade_ord"] = df["grade"].astype(str).str.strip().map({g: i+1 for i, g in enumerate(GRADE_ORDER)}).astype("float32")
df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map({s: i+1 for i, s in enumerate(SUBGRADE_ORDER)}).astype("float32")

EXCLUDE_COLS = {TARGET_COL, DATE_COL, ZIP_COL, "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "emp_title", "title", "desc", "funded_ratio", "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low"}
STRUCT_COLS = [c for c in df.columns if c not in EXCLUDE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features: {len(STRUCT_COLS)}")

print("Loading FinBERT embeddings...")
df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
for c in SEM_COLS:
    df_fb[c] = df_fb[c].astype(np.float32)
n_before = len(df)
df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == n_before
assert df["has_desc"].isna().sum() == 0
df["has_desc"] = df["has_desc"].astype(np.float32)
print(f"  Merged embeddings for {len(df):,} rows ({df['has_desc'].mean():.1%} have a non-empty description)")

# ============================================================
# 2. DELONG TEST
# ============================================================
def delong_auc_test(y_true, p1, p2):
    """DeLong et al. (1988) test for two correlated AUCs on the same test set."""
    def auc_and_kernel(y, p):
        pos = p[y == 1]; neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([np.mean(pi > neg) + 0.5*np.mean(pi == neg) for pi in pos])
        V01 = np.array([np.mean(pj < pos) + 0.5*np.mean(pj == pos) for pj in neg])
        return V10.mean(), V10, V01, n1, n0

    y = np.asarray(y_true).astype(int)
    auc1, V10_1, V01_1, n1, n0 = auc_and_kernel(y, np.asarray(p1))
    auc2, V10_2, V01_2, _, _  = auc_and_kernel(y, np.asarray(p2))
    S10 = np.cov(V10_1, V10_2); S01 = np.cov(V01_1, V01_2)
    var_diff = (S10[0,0]/n1 + S01[0,0]/n0) + (S10[1,1]/n1 + S01[1,1]/n0) - 2*(S10[0,1]/n1 + S01[0,1]/n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan, np.nan
    diff = auc1 - auc2
    se = np.sqrt(var_diff)
    z = diff / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval, diff - 1.96*se, diff + 1.96*se

# ============================================================
# 3. WALK-FORWARD FOLDS
# ============================================================
def build_folds(df, date_col, min_train_months, step_months, min_test_defaults):
    months = df[date_col].dt.to_period("M"); all_periods = sorted(months.unique())
    folds = []; fold_start_idx = min_train_months
    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]
        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, TARGET_COL].sum() >= min_test_defaults: break
            test_end_idx += 1
        if test_end_idx >= len(all_periods): break
        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)
        folds.append({"fold": len(folds)+1, "train_cutoff": str(train_cutoff), "test_end": str(test_period),
                      "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                      "tr_idx": df.index[tr_mask].tolist(), "te_idx": df.index[te_mask].tolist()})
        fold_start_idx += step_months
    return folds

print("\nBuilding Walk-Forward folds...")
folds = build_folds(df, DATE_COL, MIN_TRAIN_MONTHS, STEP_MONTHS, MIN_TEST_DEFAULTS)
assert len(folds) == 14
print(f"  Total folds: {len(folds)}")

# ============================================================
# 4. FEATURE MATRICES  (structured block shared; two FinBERT variants)
# ============================================================
def build_structured(df_tr, df_te):
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_te = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te

def build_fb_768(df_tr, df_te):
    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), df_tr[SEM_COLS].values.astype(np.float32)])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), df_te[SEM_COLS].values.astype(np.float32)])
    scaler = StandardScaler()
    return scaler.fit_transform(fb_tr_raw), scaler.transform(fb_te_raw)

def build_fb_pca50(df_tr, df_te):
    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver='randomized', random_state=RANDOM_STATE)
    pca_tr = pca.fit_transform(df_tr[SEM_COLS].values.astype(np.float32))
    pca_te = pca.transform(df_te[SEM_COLS].values.astype(np.float32))
    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
    scaler = StandardScaler()
    return scaler.fit_transform(fb_tr_raw), scaler.transform(fb_te_raw)

def make_models(y_tr):
    return {
        "Logistic": LogisticRegression(**LR_PARAMS),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(), eval_metric="auc", use_label_encoder=False,
            verbosity=0, n_jobs=4, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }

# ============================================================
# 5. MAIN LOOP
# ============================================================
delong_rows = []
run_start = time.perf_counter()

for f in folds:
    t0 = time.perf_counter()
    df_tr, df_te = df.loc[f["tr_idx"]], df.loc[f["te_idx"]]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    X_tr_s, X_te_s = build_structured(df_tr, df_te)

    # --- Phase A: 768-dim variant. Fit all 3 models, keep only the small
    # prediction vectors, then free the (large) matrices + fitted models
    # before Phase B starts, so peak memory never holds both variants at
    # once (this is what killed the first attempt at fold 11/14). ---
    X_tr_768, X_te_768 = build_fb_768(df_tr, df_te)
    X_tr_768_full = np.hstack([X_tr_s, X_tr_768])
    X_te_768_full = np.hstack([X_te_s, X_te_768])
    preds_768 = {}
    for mname, model in make_models(y_tr).items():
        model.fit(X_tr_768_full, y_tr)
        preds_768[mname] = model.predict_proba(X_te_768_full)[:, 1]
        del model
    del X_tr_768, X_te_768, X_tr_768_full, X_te_768_full
    gc.collect()

    # --- Phase B: PCA-50 variant. Same pattern. ---
    X_tr_pca, X_te_pca = build_fb_pca50(df_tr, df_te)
    X_tr_pca_full = np.hstack([X_tr_s, X_tr_pca])
    X_te_pca_full = np.hstack([X_te_s, X_te_pca])
    preds_pca = {}
    for mname, model in make_models(y_tr).items():
        model.fit(X_tr_pca_full, y_tr)
        preds_pca[mname] = model.predict_proba(X_te_pca_full)[:, 1]
        del model
    del X_tr_pca, X_te_pca, X_tr_pca_full, X_te_pca_full, X_tr_s, X_te_s
    gc.collect()

    for mname in preds_768:
        auc_768, auc_pca, z, pval, ci_lo, ci_hi = delong_auc_test(y_te, preds_768[mname], preds_pca[mname])
        delong_rows.append({
            "fold": f["fold"], "Model": mname,
            "AUC_768": round(auc_768, 4), "AUC_PCA50": round(auc_pca, 4),
            "Delta_AUC_PCA50_minus_768": round(auc_pca - auc_768, 4),
            "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
            "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "CI_95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "CI_95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
        })

    del preds_768, preds_pca, df_tr, df_te
    gc.collect()

    dt = time.perf_counter() - t0
    total = time.perf_counter() - run_start
    print(f"  Fold {f['fold']} done. ({dt:.1f}s this fold, {total/60:.1f}min elapsed total)")

# ============================================================
# 6. SUMMARY
# ============================================================
df_delong = pd.DataFrame(delong_rows)
print("\n" + "="*70)
print("DELONG TEST PER FOLD: FinBERT-PCA50 vs FinBERT-768 (same test set, fixed hyperparameters)")
print("="*70)
print(df_delong.to_string(index=False))

print("\n" + "="*70)
print("SUMMARY — mean across folds, and count of folds with p < 0.05")
print("="*70)
summary_rows = []
for m in df_delong["Model"].unique():
    sub = df_delong[df_delong["Model"] == m]
    n_sig = (sub["p_value"] < 0.05).sum()
    n_sig_favor_pca = ((sub["p_value"] < 0.05) & (sub["Delta_AUC_PCA50_minus_768"] > 0)).sum()
    n_sig_favor_768 = ((sub["p_value"] < 0.05) & (sub["Delta_AUC_PCA50_minus_768"] < 0)).sum()
    row = {
        "Model": m,
        "mean_Delta_AUC_PCA50_minus_768": round(sub["Delta_AUC_PCA50_minus_768"].mean(), 4),
        "folds_significant_p<0.05": int(n_sig),
        "of_which_favor_PCA50": int(n_sig_favor_pca),
        "of_which_favor_768": int(n_sig_favor_768),
        "total_folds": len(sub),
    }
    summary_rows.append(row)
    print(f"  [{m}] mean ΔAUC(PCA50-768)={row['mean_Delta_AUC_PCA50_minus_768']:+.4f} | "
          f"significant in {n_sig}/{len(sub)} folds ({n_sig_favor_pca} favor PCA50, {n_sig_favor_768} favor 768)")

df_summary = pd.DataFrame(summary_rows)

# ============================================================
# 7. SAVE
# ============================================================
out_dir = SCRIPT_DIR / "results" / "delong_768_vs_pca50"
os.makedirs(out_dir, exist_ok=True)
df_delong.to_csv(out_dir / "delong_per_fold.csv", index=False)
df_summary.to_csv(out_dir / "delong_summary.csv", index=False)
print(f"\nSaved results to: {out_dir}")

total_min = (time.perf_counter() - run_start) / 60
print(f"\nTotal run time: {total_min:.1f} minutes")
