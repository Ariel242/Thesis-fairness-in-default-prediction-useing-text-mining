"""
Walk-Forward Cross-Validation for Default Prediction — Structured + FinBERT (PCA-50)
=============================================================================================
Companion to this folder's BASELINE_768.py — identical everything, except the
768-dim sem_* FinBERT embedding is compressed to 50 components via PCA before
being added to the feature matrix.

                    fixed hyperparams          grid search (per fold)
  768-dim (raw)     BASELINE_768.py            GRIDSEARCH_768.py
  50-dim (PCA)      BASELINE_PCA50.py  <- this  GRIDSEARCH_PCA50.py

PCA, NO LEAKAGE: PCA(n_components=50, svd_solver='randomized', random_state=242)
is fit FRESH inside every fold, on that fold's TRAINING rows' sem_* values
only, then only .transform()'d on that fold's test rows — exactly the same
fold-local discipline as the imputer/scaler everywhere else in this project
(mirrors FinBERT/finbert_structured_ablation.py's existing PCA block). No PCA
is ever fit on test rows or on the full dataset before folds are built.
`svd_solver='randomized'` is a deliberate choice, not sklearn's silent
default: with n_components=50 << 768 dims, randomized SVD (Halko et al.) is
far cheaper than exact ('full') SVD with negligible accuracy loss — smoke-
tested here at ~0.5s (fold 1, n_train=6,694) to ~6s (fold 14, n_train=202,442)
per fold, explained variance ratio ~88-90%.

FEATURE MATRIX:
  Structured block : identical STRUCT_COLS to BASELINE_768.py.
  FinBERT block    : has_desc (0/1) + 50 PCA components of sem_*, scaled
                      separately, fit on the fold's TRAIN rows only.
  Final matrix     : hstack([structured_block, finbert_block]).

OUTPUT (written under FinBERT/ only):
  results/baseline_pca50/wf_predictive_folds.csv
  results/baseline_pca50/wf_predictive_summary.csv
  results/baseline_pca50/wf_pca_variance.csv   — explained variance ratio per fold
  figures/baseline_pca50/fig1_auc_stability.png
"""

import os
import sys
import time
import gc
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
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

SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
N_PCA_COMPONENTS = 50

LR_PARAMS = dict(
    penalty      = "l2",
    C            = 0.3,
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)

RF_PARAMS = dict(
    n_estimators     = 500,
    min_samples_leaf = 20,
    class_weight     = "balanced",
    n_jobs           = 4,
    random_state     = RANDOM_STATE,
)

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
print(f"  FinBERT block: has_desc + {N_PCA_COMPONENTS} PCA components (from {len(SEM_COLS)} sem_* dims) = {1 + N_PCA_COMPONENTS} columns")

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
# 6. FEATURE MATRIX  (structured block + PCA-compressed FinBERT block, each fit on TRAIN only)
# ============================================================
def build_matrix(df_tr, df_te):
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_te_s = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))
    scaler_s = StandardScaler()
    X_tr_s = scaler_s.fit_transform(X_tr_s)
    X_te_s = scaler_s.transform(X_te_s)

    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver='randomized', random_state=RANDOM_STATE)
    pca_tr = pca.fit_transform(df_tr[SEM_COLS].values.astype(np.float32))
    pca_te = pca.transform(df_te[SEM_COLS].values.astype(np.float32))
    evr = float(pca.explained_variance_ratio_.sum())

    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
    scaler_fb = StandardScaler()
    X_tr_fb = scaler_fb.fit_transform(fb_tr_raw)
    X_te_fb = scaler_fb.transform(fb_te_raw)

    X_tr = np.hstack([X_tr_s, X_tr_fb])
    X_te = np.hstack([X_te_s, X_te_fb])
    return X_tr, X_te, evr

FEATURE_NAMES = STRUCT_COLS + ["fb:has_desc"] + [f"fb:pca_{i:03d}" for i in range(N_PCA_COMPONENTS)]

# ============================================================
# 7. MAIN LOOP
# ============================================================
pred_rows, pca_rows = [], []
run_start = time.perf_counter()

for f in folds:
    t0 = time.perf_counter()
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr  = df.loc[tr]
    df_te  = df.loc[te]

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    X_tr, X_te, evr = build_matrix(df_tr, df_te)
    pca_rows.append({"fold": f["fold"], "n_components": N_PCA_COMPONENTS, "explained_variance_ratio": round(evr, 4)})

    models = {
        "Logistic": LogisticRegression(**LR_PARAMS),
        "XGBoost":  xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            scale_pos_weight = (y_tr==0).sum() / (y_tr==1).sum(),
            eval_metric      = "auc",
            use_label_encoder= False,
            verbosity        = 0,
            n_jobs           = 4,
            random_state     = RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }

    for mname, model in models.items():
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

    del X_tr, X_te, models, df_tr, df_te
    gc.collect()

    dt = time.perf_counter() - t0
    total = time.perf_counter() - run_start
    print(f"  Fold {f['fold']} done. (EVR={evr:.4f}, {dt:.1f}s this fold, {total/60:.1f}min elapsed total)")

# ============================================================
# 8. AGGREGATE RESULTS
# ============================================================
df_pred = pd.DataFrame(pred_rows)
df_pca  = pd.DataFrame(pca_rows)
PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]

pred_summary = (
    df_pred.groupby("Model")[PRED_METRICS]
    .agg(["mean", "std"])
    .round(4)
)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean ± std across folds) — Structured + FinBERT-PCA50")
print("="*70)
print(pred_summary.to_string())

print("\n" + "="*70)
print("B. PCA EXPLAINED VARIANCE PER FOLD")
print("="*70)
print(df_pca.to_string(index=False))

# ============================================================
# 9. SAVE
# ============================================================
out_dir = SCRIPT_DIR / "results" / "baseline_pca50"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(os.path.join(out_dir, "wf_predictive_folds.csv"), index=False)
pred_summary.to_csv(os.path.join(out_dir, "wf_predictive_summary.csv"))
df_pca.to_csv(os.path.join(out_dir, "wf_pca_variance.csv"), index=False)

print(f"\nSaved results to: {out_dir}")

# ============================================================
# 10. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

figures_dir = SCRIPT_DIR / "figures" / "baseline_pca50"
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

fig.suptitle("Walk-Forward AUC Stability — Structured + FinBERT-PCA50 (fixed hyperparameters)\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to: {figures_dir / 'fig1_auc_stability.png'}")

# ============================================================
# 11. FEATURE IMPORTANCE — LAST FOLD
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE — Last Fold")
print("="*70)

last_f = folds[-1]
_df_tr = df.loc[last_f["tr_idx"]]
_df_te = df.loc[last_f["te_idx"]]
_y_tr  = _df_tr[TARGET_COL].values.astype(int)

_X_tr, _X_te, _evr = build_matrix(_df_tr, _df_te)

_lr = LogisticRegression(**LR_PARAMS)
_lr.fit(_X_tr, _y_tr)

_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    scale_pos_weight=(_y_tr==0).sum() / (_y_tr==1).sum(),
    eval_metric="auc", use_label_encoder=False, verbosity=0, n_jobs=4,
    random_state=RANDOM_STATE,
)
_xgb.fit(_X_tr, _y_tr)

_rf = RandomForestClassifier(**RF_PARAMS)
_rf.fit(_X_tr, _y_tr)

_lr_imp = np.abs(_lr.coef_[0])
_df_lr_imp = (pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": _lr_imp})
              .sort_values("Importance", ascending=False).reset_index(drop=True))
_df_lr_imp.index += 1

_xgb_gain = _xgb.get_booster().get_score(importance_type="gain")
_df_xgb_imp = pd.DataFrame([
    {"Feature": FEATURE_NAMES[int(k.replace("f", ""))], "Gain": round(v, 2)}
    for k, v in _xgb_gain.items()
]).sort_values("Gain", ascending=False).reset_index(drop=True)
_df_xgb_imp.index += 1

_df_rf_imp = (pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": np.round(_rf.feature_importances_, 5)})
              .sort_values("Importance", ascending=False).reset_index(drop=True))
_df_rf_imp.index += 1

TOP_N = 20
print("\n── Table 1: XGBoost — Top 20 Features ──")
print(_df_xgb_imp.head(TOP_N).to_string())
print("\n── Table 2: Logistic Regression — Top 20 Features ──")
print(_df_lr_imp.head(TOP_N).to_string())
print("\n── Table 3: RandomForest — Top 20 Features ──")
print(_df_rf_imp.head(TOP_N).to_string())

total_min = (time.perf_counter() - run_start) / 60
print(f"\nTotal run time: {total_min:.1f} minutes")
