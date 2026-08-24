"""
Walk-Forward Cross-Validation for Default Prediction — with Grid Search
=========================================================================
Same protocol, same data, same features as BASELINE.py — the only
difference is that each model's hyperparameters are tuned per fold instead
of fixed. This is a companion/variant of BASELINE.py, not a replacement:
BASELINE.py's numbers are the reference point this file is compared
against, so results are written to separate output folders
(results/walk_forward_gridsearch, results/figures_gridsearch) rather than
overwriting results/walk_forward_v2.

GRID SEARCH PROTOCOL (per fold, no leakage):
  Each fold's training rows are already chronologically sorted (the full
  dataset is sorted by DATE_COL before folds are built, and a fold's train
  set is always a contiguous prefix). Within that prefix, the last 20% of
  rows (VAL_FRAC) are held out as a temporal validation split — i.e. the
  most recent training-period loans, never the test-period ones. Imputer/
  scaler are fit on the remaining 80% ("sub-train") only, applied to the
  validation slice, and every grid combination is scored on validation AUC.
  The winning hyperparameters are then used to refit the model on the FULL
  fold training set (fresh imputer/scaler fit on all of it, exactly like
  BASELINE.py) before scoring the actual test fold. So no combination is
  ever chosen using information from the test period, and the final model
  is always trained on all available history — same guarantee BASELINE.py
  makes, just with an extra tuning step that only ever looks backward.

BASIC GRIDS (kept small — this is meant to be a light first pass, not an
exhaustive search; RandomForest's n_estimators is left fixed at 500 since
sweeping tree count is the most expensive axis and depth is already
controlled via min_samples_leaf):
  Logistic:      C in {0.03, 0.1, 0.3, 1}                              (4 combos)
  XGBoost:       max_depth in {3, 4, 6} x learning_rate in {0.05, 0.1}  (6 combos)
  RandomForest:  min_samples_leaf in {10, 20, 50}                      (3 combos)

Everything else (feature list, EXCLUDE_COLS, walk-forward params, fairness
and text pipeline out of scope, RANDOM_STATE, bootstrap CI) is identical to
BASELINE.py — see that file's docstring for the full v2 changelog.
"""

import os
import sys
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
BASE_DIR        = Path(__file__).resolve().parent.parent
PATH_CSV        = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
TARGET_COL      = "is_default"
DATE_COL        = "issue_month_start"
ZIP_COL         = "zip3"

# NOTE ON CENSORING:
# The CSV contains only loans that have reached their final status.
# is_default is fully observed for every row — no censoring bias.

# Walk-Forward parameters
MIN_TRAIN_MONTHS  = 8    # minimum months before first test fold
MIN_TEST_DEFAULTS = 100  # minimum default events per test window (dynamic)
STEP_MONTHS       = 3    # quarterly step (~10–13 folds expected)

# Decision thresholds — currently unused (kept for compatibility; this
# script has no threshold-dependent metrics since fairness was removed)
THRESHOLDS = [0.5, 0.6, 0.7]

# Bootstrap CI (set N_BOOTSTRAP=0 to skip — faster runs)
N_BOOTSTRAP = 500
BOOT_SEED   = 42

# Fixed random state — shared by all three models
RANDOM_STATE = 242

# Temporal validation split used ONLY for hyperparameter search (see docstring)
VAL_FRAC = 0.2

# Logistic Regression — fixed spec except C, which is grid-searched
LR_FIXED = dict(
    penalty      = "l2",
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)
GRID_LR = {"C": [0.03, 0.1, 0.3, 1]}

# XGBoost — fixed spec except max_depth / learning_rate
XGB_FIXED = dict(
    n_estimators      = 300,
    subsample         = 0.8,
    eval_metric       = "auc",
    use_label_encoder = False,
    verbosity         = 0,
    n_jobs            = -1,
    random_state      = RANDOM_STATE,
)
GRID_XGB = {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}

# Random Forest — fixed spec except min_samples_leaf
RF_FIXED = dict(
    n_estimators     = 500,   # kept fixed — the expensive axis; not swept
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)
GRID_RF = {"min_samples_leaf": [10, 20, 50]}

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)

df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

if ZIP_COL in df.columns:
    df[ZIP_COL] = df[ZIP_COL].astype(object).astype(str).replace("nan", np.nan)
    n_zip  = df[ZIP_COL].nunique(dropna=True)
    n_null = df[ZIP_COL].isna().sum()
    print(f"  ZIP column '{ZIP_COL}': {n_zip} unique groups, {n_null:,} nulls")
else:
    print(f"  WARNING: ZIP column '{ZIP_COL}' not found in CSV")
    zip_cols = [c for c in df.columns if 'zip' in c.lower()]
    print(f"  Columns with zip: {zip_cols}")

# ============================================================
# 1b. RESTORE ORDINAL RISK GRADES  (v2 fix #2 — see BASELINE.py changelog)
# ============================================================
GRADE_ORDER    = list("ABCDEFG")
SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]

if "grade" in df.columns:
    grade_map = {g: i + 1 for i, g in enumerate(GRADE_ORDER)}
    df["grade_ord"] = df["grade"].astype(str).str.strip().map(grade_map).astype("float32")
    n_unmapped = int(df["grade_ord"].isna().sum())
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} rows have unrecognized 'grade' values (left as NaN)")
else:
    print("  WARNING: 'grade' column not found — grade_ord not created")

if "sub_grade" in df.columns:
    subgrade_map = {s: i + 1 for i, s in enumerate(SUBGRADE_ORDER)}
    df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map(subgrade_map).astype("float32")
    n_unmapped = int(df["sub_grade_ord"].isna().sum())
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} rows have unrecognized 'sub_grade' values (left as NaN)")
else:
    print("  WARNING: 'sub_grade' column not found — sub_grade_ord not created")

print("  Ordinal risk grades restored: grade_ord (1-7), sub_grade_ord (1-35) — "
      "both kept; sub_grade_ord refines grade_ord, regularization absorbs the collinearity")

EXCLUDE_COLS = {TARGET_COL, DATE_COL, ZIP_COL,
                "id",   # primary key -- identifier only, never a feature
                "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "emp_title", "title", "desc", "funded_ratio",
                "grade", "sub_grade",   # raw string cols — numeric encodings are grade_ord / sub_grade_ord
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage (see BASELINE.py changelog #3)
STRUCT_COLS = [c for c in df.columns
               if c not in EXCLUDE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                   np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features: {len(STRUCT_COLS)}")

# ============================================================
# 2. BOOTSTRAP CI
# ============================================================
def bootstrap_ci(y_true, p, metric_fn, n=N_BOOTSTRAP, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap CI for a scalar metric. Returns (lo, hi)."""
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
# 3. WALK-FORWARD FOLDS
# ============================================================
def build_folds(df, date_col, min_train_months, step_months, min_test_defaults):
    """
    Expanding-window walk-forward folds.
    Each fold trains on all history up to cutoff.
    Test window expands dynamically until min_test_defaults events are reached.
    """
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

# ============================================================
# 4. GRID SEARCH  (temporal sub-train/validation split within each fold)
# ============================================================
def fit_matrix(df_fit, df_apply, cols):
    """Fit imputer+scaler on df_fit, apply to both df_fit and df_apply."""
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_fit   = imputer.fit_transform(df_fit[cols].values.astype(np.float32))
    X_apply = imputer.transform(df_apply[cols].values.astype(np.float32))
    scaler  = StandardScaler()
    X_fit   = scaler.fit_transform(X_fit)
    X_apply = scaler.transform(X_apply)
    return X_fit, X_apply

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
        m = xgb.XGBClassifier(max_depth=md, learning_rate=lr,
                               scale_pos_weight=spw, **XGB_FIXED)
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

# ============================================================
# 5. MAIN LOOP
# ============================================================
print("\nBuilding Walk-Forward folds...")
folds = build_folds(
    df,
    date_col          = DATE_COL,
    min_train_months  = MIN_TRAIN_MONTHS,
    step_months       = STEP_MONTHS,
    min_test_defaults = MIN_TEST_DEFAULTS,
)
print(f"  Total folds: {len(folds)}")
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  "
          f"test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  "
          f"defaults_in_test={f['n_test_defaults']}")

# Storage
pred_rows        = []   # predictive metrics per fold/model (final, test-fold)
best_params_rows = []   # winning hyperparameters per fold/model (from validation search)

SEARCHERS = {"Logistic": search_logistic, "XGBoost": search_xgb, "RandomForest": search_rf}

for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr  = df.loc[tr]
    df_te  = df.loc[te]

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    # --- Temporal sub-train/validation split for hyperparameter search ---
    # df_tr rows are already chronologically ordered (contiguous prefix of
    # the globally sorted df), so a positional split IS a temporal split.
    n_sub = int(len(df_tr) * (1 - VAL_FRAC))
    df_sub, df_val = df_tr.iloc[:n_sub], df_tr.iloc[n_sub:]
    y_sub = df_sub[TARGET_COL].values.astype(int)
    y_val = df_val[TARGET_COL].values.astype(int)

    can_search = len(np.unique(y_sub)) == 2 and len(np.unique(y_val)) == 2
    if can_search:
        X_sub, X_val = fit_matrix(df_sub, df_val, STRUCT_COLS)

    # --- Final matrix: fit imputer/scaler on the FULL fold train set ---
    X_tr, X_te = fit_matrix(df_tr, df_te, STRUCT_COLS)

    for mname, searcher in SEARCHERS.items():
        if can_search:
            best_params, best_val_auc = searcher(X_sub, y_sub, X_val, y_val)
        else:
            # Fold too small / degenerate validation split — fall back to
            # the first grid entry and skip the search for this fold only.
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

        # --- Refit on the FULL fold training set with the winning params ---
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

        # ── Predictive metrics (threshold-independent) ──────────────────
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

    print(f"  Fold {f['fold']} done.")

# ============================================================
# 6. AGGREGATE RESULTS
# ============================================================
df_pred        = pd.DataFrame(pred_rows)
df_best_params = pd.DataFrame(best_params_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]

# ── A. Predictive performance summary ───────────────────────────────────────
pred_summary = (
    df_pred.groupby("Model")[PRED_METRICS]
    .agg(["mean", "std"])
    .round(4)
)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean ± std across folds)")
print("="*70)
print(pred_summary.to_string())

# ── B. Winning hyperparameters per fold ─────────────────────────────────────
print("\n" + "="*70)
print("B. BEST HYPERPARAMETERS PER FOLD (selected on temporal validation AUC)")
print("="*70)
print(df_best_params.to_string(index=False))

print("\n  Most frequently selected params per model:")
for m in df_best_params["Model"].unique():
    sub = df_best_params[df_best_params["Model"] == m].drop(columns=["fold", "n_sub", "n_val", "val_AUC", "Model"])
    mode_row = sub.mode().iloc[0].to_dict()
    print(f"    [{m}]  {mode_row}")

# ============================================================
# 7. SAVE
# ============================================================
# Separate folder from BASELINE.py's results/walk_forward_v2 so the fixed-
# hyperparameter baseline numbers are preserved as the comparison point.
out_dir = BASE_DIR / "results" / "walk_forward_gridsearch"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(os.path.join(out_dir, "wf_predictive_folds.csv"), index=False)
df_best_params.to_csv(os.path.join(out_dir, "wf_best_params.csv"), index=False)
pred_summary.to_csv(os.path.join(out_dir, "wf_predictive_summary.csv"))

print(f"\nSaved results to: {out_dir}")
print("\nOutput files:")
print("  wf_predictive_folds.csv           — per-fold predictive metrics with bootstrap 95% CIs + winning params")
print("  wf_best_params.csv                — winning hyperparameters per fold/model with validation AUC")
print("  wf_predictive_summary.csv         — predictive summary (mean ± std)")

# ============================================================
# 8. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

figures_dir = BASE_DIR / "results" / "figures_gridsearch"
os.makedirs(figures_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]
_CLR  = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}

print("\nGenerating figures...")

# ── Figure 1: Walk-Forward AUC Stability ────────────────────────────────────
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5 * len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    color = _CLR[model]
    sub = df_pred[df_pred["Model"] == model].sort_values("fold")
    ax.plot(sub["fold"], sub["AUC"],
            linestyle="-", marker="o", color=color,
            linewidth=2, markersize=5)
    if not sub["AUC_CI_lo"].isna().all():
        ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"],
                        alpha=0.15, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("AUC")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig.suptitle("Walk-Forward AUC Stability Across Folds — Grid-Searched\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig1_auc_stability.png")

print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 9. FEATURE IMPORTANCE — LAST FOLD  (uses that fold's winning hyperparameters)
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE — Last Fold")
print("="*70)

last_f = folds[-1]
print(f"  Last fold: train <= {last_f['train_cutoff']}  |  test end: {last_f['test_end']}")
print(f"  Train n={last_f['n_train']:,}  |  Test n={last_f['n_test']:,}")

_tr = last_f["tr_idx"]
_te = last_f["te_idx"]
_df_tr = df.loc[_tr]
_df_te = df.loc[_te]
_y_tr  = _df_tr[TARGET_COL].values.astype(int)
_y_te  = _df_te[TARGET_COL].values.astype(int)

_X_tr, _X_te = fit_matrix(_df_tr, _df_te, STRUCT_COLS)

_last_fold_params = {
    row["Model"]: row for row in best_params_rows if row["fold"] == last_f["fold"]
}

_lr = LogisticRegression(C=_last_fold_params["Logistic"]["C"], **LR_FIXED)
_lr.fit(_X_tr, _y_tr)

_xgb_spw = (_y_tr == 0).sum() / (_y_tr == 1).sum()
_xgb = xgb.XGBClassifier(max_depth=_last_fold_params["XGBoost"]["max_depth"],
                          learning_rate=_last_fold_params["XGBoost"]["learning_rate"],
                          scale_pos_weight=_xgb_spw, **XGB_FIXED)
_xgb.fit(_X_tr, _y_tr)

_rf = RandomForestClassifier(min_samples_leaf=_last_fold_params["RandomForest"]["min_samples_leaf"], **RF_FIXED)
_rf.fit(_X_tr, _y_tr)

# ── LR importance: absolute standardized coefficients ────────────────────────
_lr_imp = np.abs(_lr.coef_[0])
_df_lr_imp = (
    pd.DataFrame({"Feature": STRUCT_COLS, "Importance": _lr_imp})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
_df_lr_imp.index += 1

# ── XGBoost importance: Gain ──────────────────────────────────────────────────
_xgb_gain = _xgb.get_booster().get_score(importance_type="gain")
_df_xgb_imp = pd.DataFrame([
    {"Feature": STRUCT_COLS[int(k.replace("f", ""))], "Gain": round(v, 2)}
    for k, v in _xgb_gain.items()
]).sort_values("Gain", ascending=False).reset_index(drop=True)
_df_xgb_imp.index += 1

# ── RandomForest importance: mean decrease in impurity ───────────────────────
_df_rf_imp = (
    pd.DataFrame({"Feature": STRUCT_COLS,
                  "Importance": np.round(_rf.feature_importances_, 5)})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
_df_rf_imp.index += 1

TOP_N = 20

print("\n── Table 1: XGBoost — Top 20 Features ──")
print(_df_xgb_imp.head(TOP_N).to_string())

print("\n── Table 2: Logistic Regression — Top 20 Features ──")
print(_df_lr_imp.head(TOP_N).to_string())

print("\n── Table 3: RandomForest — Top 20 Features ──")
print(_df_rf_imp.head(TOP_N).to_string())
