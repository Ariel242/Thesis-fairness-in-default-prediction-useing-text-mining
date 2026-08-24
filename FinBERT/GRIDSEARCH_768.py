"""
Walk-Forward Cross-Validation for Default Prediction — with Grid Search
Structured + Full FinBERT (768-dim)
=============================================================================================
Companion to analysis/GRIDSEARCH.py and to this folder's BASELINE_768.py.
Same feature matrix as BASELINE_768.py (structured block + has_desc + raw
768-dim sem_* FinBERT embedding, no reduction) — the only difference from
BASELINE_768.py is that each model's hyperparameters are tuned per fold
instead of fixed, using the identical protocol as analysis/GRIDSEARCH.py.

                    fixed hyperparams          grid search (per fold)
  768-dim (raw)     BASELINE_768.py            GRIDSEARCH_768.py  <- this file
  50-dim (PCA)      BASELINE_PCA50.py          GRIDSEARCH_PCA50.py

GRID SEARCH PROTOCOL (identical to analysis/GRIDSEARCH.py): within each
fold, the last 20% of the fold's (chronologically ordered) training rows
are held out as a temporal validation split. Imputer/scaler for BOTH the
structured and FinBERT blocks are fit on the remaining 80% ("sub-train")
only; every grid combination is scored on validation AUC; the winner is
refit on the FULL fold training set (fresh imputer/scaler, matching
BASELINE_768.py) before scoring the actual test fold.

GRIDS: identical to analysis/GRIDSEARCH.py for XGBoost and RandomForest.
ONE deviation: Logistic's grid is extended one step lower, C in
{0.01, 0.03, 0.1, 0.3, 1} instead of {0.03, 0.1, 0.3, 1} — the structured-
only grid search pinned C=0.03 (the low edge of that grid) in most folds,
and this feature matrix is ~7x wider (895 vs 126 columns), which if
anything wants MORE regularization, not less. Everything else matches
analysis/GRIDSEARCH.py's grids exactly.

RUNTIME WARNING: this is the most expensive of the four scripts in this
folder. The feature matrix is ~895 columns (vs 126 in analysis/GRIDSEARCH.py)
and Logistic's `saga` solver scales roughly linearly in feature count, on
top of the grid search's ~4-5x fold-time multiplier already seen in
analysis/GRIDSEARCH.py. Expect a long run; elapsed-time is printed per fold.

OUTPUT (written under FinBERT/ only):
  results/gridsearch_768/wf_predictive_folds.csv (includes winning params)
  results/gridsearch_768/wf_best_params.csv
  results/gridsearch_768/wf_predictive_summary.csv
  figures/gridsearch_768/fig1_auc_stability.png
"""

import os
import sys
import time
import gc
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS — edit these before running
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

THRESHOLDS = [0.5, 0.6, 0.7]   # unused — kept for compatibility

N_BOOTSTRAP = 500
BOOT_SEED   = 42

RANDOM_STATE = 242

VAL_FRAC = 0.2   # temporal validation split used ONLY for hyperparameter search

SEM_COLS = [f"sem_{i:03d}" for i in range(768)]

LR_FIXED = dict(
    penalty      = "l2",
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)
GRID_LR = {"C": [0.01, 0.03, 0.1, 0.3, 1]}   # extended one step lower — see docstring

XGB_FIXED = dict(
    n_estimators      = 300,
    subsample         = 0.8,
    eval_metric       = "auc",
    use_label_encoder = False,
    verbosity         = 0,
    n_jobs            = 4,
    random_state      = RANDOM_STATE,
)
GRID_XGB = {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}

RF_FIXED = dict(
    n_estimators     = 500,
    class_weight     = "balanced",
    n_jobs           = 4,
    random_state     = RANDOM_STATE,
)
GRID_RF = {"min_samples_leaf": [10, 20, 50]}

# ============================================================
# 1. LOAD STRUCTURED DATA
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

# ============================================================
# 1b. RESTORE ORDINAL RISK GRADES
# ============================================================
GRADE_ORDER    = list("ABCDEFG")
SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]

if "grade" in df.columns:
    grade_map = {g: i + 1 for i, g in enumerate(GRADE_ORDER)}
    df["grade_ord"] = df["grade"].astype(str).str.strip().map(grade_map).astype("float32")
if "sub_grade" in df.columns:
    subgrade_map = {s: i + 1 for i, s in enumerate(SUBGRADE_ORDER)}
    df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map(subgrade_map).astype("float32")
print("  Ordinal risk grades restored: grade_ord (1-7), sub_grade_ord (1-35)")

# ============================================================
# 2. STRUCTURED FEATURE LIST  (computed BEFORE the FinBERT merge)
# ============================================================
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

# ============================================================
# 3. MERGE FINBERT EMBEDDINGS  (by id — never row position)
# ============================================================
print("Loading FinBERT embeddings...")
df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
for c in SEM_COLS:
    df_fb[c] = df_fb[c].astype(np.float32)

n_before = len(df)
df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == n_before, "Merge changed row count — id is not a clean 1:1 key"
assert df["has_desc"].isna().sum() == 0, "Unmatched rows after FinBERT merge — check id coverage"
df["has_desc"] = df["has_desc"].astype(np.float32)
print(f"  Merged embeddings for {len(df):,} rows ({df['has_desc'].mean():.1%} have a non-empty description)")
print(f"  FinBERT block: has_desc + {len(SEM_COLS)} sem_* dims = {1 + len(SEM_COLS)} columns (no reduction)")

# ============================================================
# 4. BOOTSTRAP CI
# ============================================================
def bootstrap_ci(y_true, p, metric_fn, n=N_BOOTSTRAP, seed=BOOT_SEED, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    boot_stats = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        y_b, p_b = y_true[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_stats.append(metric_fn(y_b, p_b))
        except Exception:
            pass
    if len(boot_stats) < 10:
        return np.nan, np.nan
    return (float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))))

# ============================================================
# 5. WALK-FORWARD FOLDS
# ============================================================
def build_folds(df, date_col, min_train_months, step_months, min_test_defaults):
    months      = df[date_col].dt.to_period("M")
    all_periods = sorted(months.unique())
    folds = []
    fold_start_idx = min_train_months
    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]
        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, TARGET_COL].sum() >= min_test_defaults:
                break
            test_end_idx += 1
        if test_end_idx >= len(all_periods):
            break
        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)
        folds.append({
            "fold":            len(folds) + 1,
            "train_cutoff":    str(train_cutoff),
            "test_end":        str(test_period),
            "n_train":         int(tr_mask.sum()),
            "n_test":          int(te_mask.sum()),
            "n_test_defaults": int(df.loc[te_mask, TARGET_COL].sum()),
            "tr_idx":          df.index[tr_mask].tolist(),
            "te_idx":          df.index[te_mask].tolist(),
        })
        fold_start_idx += step_months
    return folds

print("\nBuilding Walk-Forward folds...")
folds = build_folds(df, DATE_COL, MIN_TRAIN_MONTHS, STEP_MONTHS, MIN_TEST_DEFAULTS)
assert len(folds) == 14, f"Expected exactly 14 walk-forward folds, got {len(folds)}."
print(f"  Total folds: {len(folds)}")
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  "
          f"test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  "
          f"defaults_in_test={f['n_test_defaults']}")

# ============================================================
# 6. FEATURE MATRIX  (structured block + FinBERT block, each fit on the given TRAIN rows only)
# ============================================================
def build_matrix(df_fit, df_apply):
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_fit_s   = imputer.fit_transform(df_fit[STRUCT_COLS].values.astype(np.float32))
    X_apply_s = imputer.transform(df_apply[STRUCT_COLS].values.astype(np.float32))
    scaler_s = StandardScaler()
    X_fit_s   = scaler_s.fit_transform(X_fit_s)
    X_apply_s = scaler_s.transform(X_apply_s)

    fb_fit_raw   = np.hstack([df_fit[["has_desc"]].values.astype(np.float32),
                               df_fit[SEM_COLS].values.astype(np.float32)])
    fb_apply_raw = np.hstack([df_apply[["has_desc"]].values.astype(np.float32),
                               df_apply[SEM_COLS].values.astype(np.float32)])
    scaler_fb = StandardScaler()
    X_fit_fb   = scaler_fb.fit_transform(fb_fit_raw)
    X_apply_fb = scaler_fb.transform(fb_apply_raw)

    X_fit   = np.hstack([X_fit_s, X_fit_fb])
    X_apply = np.hstack([X_apply_s, X_apply_fb])
    return X_fit, X_apply

FEATURE_NAMES = STRUCT_COLS + ["fb:has_desc"] + [f"fb:sem_{i:03d}" for i in range(768)]

# ============================================================
# 7. GRID SEARCH  (temporal sub-train/validation split within each fold)
# ============================================================
def search_logistic(X_sub, y_sub, X_val, y_val):
    best = {"C": GRID_LR["C"][0]}
    best_auc = -np.inf
    for c in GRID_LR["C"]:
        m = LogisticRegression(C=c, **LR_FIXED)
        m.fit(X_sub, y_sub)
        p = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        if auc > best_auc:
            best_auc, best = auc, {"C": c}
    return best, best_auc

def search_xgb(X_sub, y_sub, X_val, y_val):
    best = {"max_depth": GRID_XGB["max_depth"][0], "learning_rate": GRID_XGB["learning_rate"][0]}
    best_auc = -np.inf
    spw = (y_sub == 0).sum() / (y_sub == 1).sum()
    for md, lr in product(GRID_XGB["max_depth"], GRID_XGB["learning_rate"]):
        m = xgb.XGBClassifier(max_depth=md, learning_rate=lr, scale_pos_weight=spw, **XGB_FIXED)
        m.fit(X_sub, y_sub)
        p = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        if auc > best_auc:
            best_auc, best = auc, {"max_depth": md, "learning_rate": lr}
    return best, best_auc

def search_rf(X_sub, y_sub, X_val, y_val):
    best = {"min_samples_leaf": GRID_RF["min_samples_leaf"][0]}
    best_auc = -np.inf
    for msl in GRID_RF["min_samples_leaf"]:
        m = RandomForestClassifier(min_samples_leaf=msl, **RF_FIXED)
        m.fit(X_sub, y_sub)
        p = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        if auc > best_auc:
            best_auc, best = auc, {"min_samples_leaf": msl}
    return best, best_auc

SEARCHERS = {"Logistic": search_logistic, "XGBoost": search_xgb, "RandomForest": search_rf}

# ============================================================
# 8. MAIN LOOP
# ============================================================
pred_rows, best_params_rows = [], []
run_start = time.perf_counter()

for f in folds:
    t0 = time.perf_counter()
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr  = df.loc[tr]
    df_te  = df.loc[te]

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    n_sub = int(len(df_tr) * (1 - VAL_FRAC))
    df_sub, df_val = df_tr.iloc[:n_sub], df_tr.iloc[n_sub:]
    y_sub = df_sub[TARGET_COL].values.astype(int)
    y_val = df_val[TARGET_COL].values.astype(int)

    can_search = len(np.unique(y_sub)) == 2 and len(np.unique(y_val)) == 2
    if can_search:
        X_sub, X_val = build_matrix(df_sub, df_val)

    X_tr, X_te = build_matrix(df_tr, df_te)

    for mname, searcher in SEARCHERS.items():
        if can_search:
            best_params, best_val_auc = searcher(X_sub, y_sub, X_val, y_val)
        else:
            best_params = {"Logistic": {"C": GRID_LR["C"][0]},
                            "XGBoost":  {"max_depth": GRID_XGB["max_depth"][0],
                                         "learning_rate": GRID_XGB["learning_rate"][0]},
                            "RandomForest": {"min_samples_leaf": GRID_RF["min_samples_leaf"][0]}}[mname]
            best_val_auc = np.nan

        best_params_rows.append({
            "fold": f["fold"], "Model": mname,
            "n_sub": len(df_sub), "n_val": len(df_val),
            "val_AUC": round(best_val_auc, 4) if not np.isnan(best_val_auc) else np.nan,
            **best_params,
        })

        if mname == "Logistic":
            model = LogisticRegression(C=best_params["C"], **LR_FIXED)
        elif mname == "XGBoost":
            spw = (y_tr == 0).sum() / (y_tr == 1).sum()
            model = xgb.XGBClassifier(max_depth=best_params["max_depth"],
                                       learning_rate=best_params["learning_rate"],
                                       scale_pos_weight=spw, **XGB_FIXED)
        else:
            model = RandomForestClassifier(min_samples_leaf=best_params["min_samples_leaf"], **RF_FIXED)

        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]

        auc   = roc_auc_score(y_te, p)
        prauc = average_precision_score(y_te, p)
        brier = brier_score_loss(y_te, p)

        auc_lo,   auc_hi   = bootstrap_ci(y_te, p, roc_auc_score)
        prauc_lo, prauc_hi = bootstrap_ci(y_te, p, average_precision_score)
        brier_lo, brier_hi = bootstrap_ci(y_te, p, brier_score_loss)

        pred_rows.append({
            "fold":         f["fold"],
            "train_cutoff": f["train_cutoff"],
            "test_end":     f["test_end"],
            "n_train":      f["n_train"],
            "n_test":       f["n_test"],
            "n_defaults":   f["n_test_defaults"],
            "Model":        mname,
            "best_params":  str(best_params),
            "AUC":          round(auc,   4),
            "AUC_CI_lo":    round(auc_lo,   4) if not np.isnan(auc_lo)   else np.nan,
            "AUC_CI_hi":    round(auc_hi,   4) if not np.isnan(auc_hi)   else np.nan,
            "PR_AUC":       round(prauc, 4),
            "PR_AUC_CI_lo": round(prauc_lo, 4) if not np.isnan(prauc_lo) else np.nan,
            "PR_AUC_CI_hi": round(prauc_hi, 4) if not np.isnan(prauc_hi) else np.nan,
            "Brier":        round(brier, 4),
            "Brier_CI_lo":  round(brier_lo, 4) if not np.isnan(brier_lo) else np.nan,
            "Brier_CI_hi":  round(brier_hi, 4) if not np.isnan(brier_hi) else np.nan,
            "GINI":         round(2*auc - 1, 4),
        })

    del X_tr, X_te, df_tr, df_te, df_sub, df_val
    if can_search:
        del X_sub, X_val
    gc.collect()

    dt = time.perf_counter() - t0
    total = time.perf_counter() - run_start
    print(f"  Fold {f['fold']} done. ({dt:.1f}s this fold, {total/60:.1f}min elapsed total)")

# ============================================================
# 9. AGGREGATE RESULTS
# ============================================================
df_pred        = pd.DataFrame(pred_rows)
df_best_params = pd.DataFrame(best_params_rows)
PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]

pred_summary = (
    df_pred.groupby("Model")[PRED_METRICS]
    .agg(["mean", "std"])
    .round(4)
)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean ± std across folds) — Structured + FinBERT-768, Grid Search")
print("="*70)
print(pred_summary.to_string())

print("\n" + "="*70)
print("B. BEST HYPERPARAMETERS PER FOLD")
print("="*70)
print(df_best_params.to_string(index=False))

print("\n  Most frequently selected params per model:")
for m in df_best_params["Model"].unique():
    sub = df_best_params[df_best_params["Model"] == m].drop(columns=["fold", "n_sub", "n_val", "val_AUC", "Model"])
    mode_row = sub.mode().iloc[0].to_dict()
    print(f"    [{m}]  {mode_row}")

# ============================================================
# 10. SAVE
# ============================================================
out_dir = SCRIPT_DIR / "results" / "gridsearch_768"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(os.path.join(out_dir, "wf_predictive_folds.csv"), index=False)
df_best_params.to_csv(os.path.join(out_dir, "wf_best_params.csv"), index=False)
pred_summary.to_csv(os.path.join(out_dir, "wf_predictive_summary.csv"))

print(f"\nSaved results to: {out_dir}")

# ============================================================
# 11. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

figures_dir = SCRIPT_DIR / "figures" / "gridsearch_768"
os.makedirs(figures_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]
_CLR  = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5 * len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    color = _CLR[model]
    sub = df_pred[df_pred["Model"] == model].sort_values("fold")
    ax.plot(sub["fold"], sub["AUC"], linestyle="-", marker="o", color=color, linewidth=2, markersize=5)
    if not sub["AUC_CI_lo"].isna().all():
        ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"], alpha=0.15, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("AUC")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig.suptitle("Walk-Forward AUC Stability — Structured + FinBERT-768, Grid-Searched\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to: {figures_dir / 'fig1_auc_stability.png'}")

total_min = (time.perf_counter() - run_start) / 60
print(f"\nTotal run time: {total_min:.1f} minutes")
