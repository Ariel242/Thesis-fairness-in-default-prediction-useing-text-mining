"""
Walk-Forward Cross-Validation for Default Prediction
=====================================================
Chronological expanding-window protocol.
Scaling and imputation are fit ONLY on the training fold — no leakage.

PARAMETERS (edit below):

-----------------------------------------------------------------------------
v2 CHANGELOG (this file supersedes preliminary_results_cloude.py — the v1
script is kept unmodified in analysis/archive/ as a historical record)
-----------------------------------------------------------------------------
Two bugs in v1 were found during a code review and are fixed here:

1. BASELINE CONTAMINATION ("Structured" arm was not text-free).
   v1 computed numeric text statistics (character length, word count, unique
   words, average word length, type-token ratio for desc/title/emp_title) and
   then folded them into `struct_cols` — the SAME feature list used for the
   "Structured" (no-text) variant. Since description length alone is a known
   predictor of default, the "Structured" baseline already carried a text
   signal, which understates how much TF-IDF adds and confounds the
   Structured vs. Structured+Text comparison the whole thesis is built on.
   (As of v2.2 this is moot here — see below — but the fix is why this
   script's feature list has never included any text-derived column.)

2. SILENT LOSS OF LENDINGCLUB'S RISK GRADE (grade / sub_grade).
   In 03_advanced_prep.ipynb, `grade` and `sub_grade` were converted to an
   ordered pandas Categorical, but pandas serializes Categorical columns to
   CSV as their string labels ("B", "B3"), not the intended numeric codes.
   The modeling script's feature selector only keeps numeric dtypes, so both
   columns were silently dropped from every model — the platform's own risk
   rating never entered the pipeline.
   Fix: `grade` and `sub_grade` are re-encoded here as explicit ordinal
   features (`grade_ord` 1–7, `sub_grade_ord` 1–35) right after load, without
   touching the upstream notebook or CSV. Both are kept (sub_grade_ord is a
   finer-grained refinement of grade_ord); L2/regularized models handle the
   resulting collinearity without issue, and this matches the original
   analysis design, which explicitly listed both as mandatory "classical risk
   variables" in 02_data_prep.ipynb.

3. TEMPORAL LEAKAGE VIA last_fico_range_high / last_fico_range_low.
   These record the borrower's FICO range as of the platform's LAST credit
   pull, not at loan origination — the same post-origination event already
   flagged as leakage via `last_credit_pull_d` in 02_data_prep.ipynb, but the
   FICO values themselves were never added to that leakage list, so they rode
   along into every model. This explains why `last_fico_range_high` dominated
   the feature-importance tables (Gain ~10x the next feature): it is a proxy
   for the outcome, not a predictor available at underwriting time.
   Fix (immediate, no notebook re-run needed): both columns are added to
   EXCLUDE_COLS here, so this script stops using them against the existing
   CSV. The proper long-term fix is upstream: 02_data_prep.ipynb's
   `leakage_columns` now also drops both columns, and `fico_range_low` /
   `fico_range_high` (the legitimate origination-time scores) were added to
   `mandatory_columns` so they survive the multicollinearity filtering
   the way grade/sub_grade already do. (Boruta has since been removed from
   that notebook entirely: it was fit on the pooled 2010-2013 data, so its
   feature selection itself leaked future information across time; feature
   filtering there is now leakage-list + missing-threshold + multicollinearity
   only.) That notebook has not been re-run yet, so `fico_range_high` is not
   yet available in the current CSV — only the EXCLUDE_COLS mitigation is
   active until it is.

v2.1 UPDATE (2026-07-09): RandomForest added as a third model, evaluated
under the identical walk-forward protocol, metrics, figures and importance
tables as Logistic/XGBoost. Initial spec: 500 trees; min_samples_leaf=20
bounds tree size so 500 full-depth trees on ~200K-row folds fit in memory.
A single fixed RANDOM_STATE=242 is now used for all three models.

v2.2 UPDATE: this script is now the TRADITIONAL (fully structured, no text)
model only — one variant, not a Structured vs. Structured+Text comparison.
Two things moved out:
  - Fairness metrics (per-ZIP FNR/FPR/Brier gaps, group coverage) were
    dropped entirely — no longer in scope for this script.
  - The whole text pipeline (raw pull, cleaning/lemmatization, numeric
    text-derived stats, TF-IDF) moved to TF-IDF/tfidf_pipeline.py, run
    independently against the same walk-forward folds.
Consequently the Structured-vs-Structured+Text delta table and the DeLong
test (which existed only to compare the two variants) are also gone — with
a single variant there is nothing left to delta or DeLong against.
-----------------------------------------------------------------------------
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
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

# Logistic Regression — explicit spec for documentation
LR_PARAMS = dict(
    penalty      = "l2",      # L2 regularization
    C            = 0.3,       # inverse strength: smaller = stronger regularization
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)

# Random Forest — explicit spec for documentation
RF_PARAMS = dict(
    n_estimators     = 500,   # initial setting
    min_samples_leaf = 20,    # bounds tree size: 500 unconstrained trees on ~200K-row folds exceed available RAM
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)

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
# 1b. RESTORE ORDINAL RISK GRADES  (v2 fix #2 — see changelog above)
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
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage: FICO as of the LAST credit pull, not at origination (see v2 changelog #3)
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

        # Dynamic test window
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
# 4. MAIN LOOP
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
pred_rows = []   # predictive metrics per fold/model

for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr  = df.loc[tr]
    df_te  = df.loc[te]

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    # --- Structured matrix (fit imputer/scaler on train only) ---
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_te = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # --- Models ---
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
            n_jobs           = -1,
            random_state     = RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }

    for mname, model in models.items():
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
# 5. AGGREGATE RESULTS
# ============================================================
df_pred = pd.DataFrame(pred_rows)

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

# ============================================================
# 6. SAVE
# ============================================================
# v2 fix: write to a separate folder so v1's (pre-fix) results are preserved
# as a historical baseline rather than silently overwritten.
out_dir = BASE_DIR / "results" / "walk_forward_v2"
os.makedirs(out_dir, exist_ok=True)

# Raw per-fold data
df_pred.to_csv(os.path.join(out_dir, "wf_predictive_folds.csv"),  index=False)

# Summary tables
pred_summary.to_csv(os.path.join(out_dir, "wf_predictive_summary.csv"))

print(f"\nSaved results to: {out_dir}")
print("\nOutput files:")
print("  wf_predictive_folds.csv           — per-fold predictive metrics with bootstrap 95% CIs")
print("  wf_predictive_summary.csv         — predictive summary (mean ± std)")

# ============================================================
# 7. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

figures_dir = BASE_DIR / "results" / "figures_v2"
os.makedirs(figures_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

# Consistent palette
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

fig.suptitle("Walk-Forward AUC Stability Across Folds\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig1_auc_stability.png")

print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 8. FEATURE IMPORTANCE — LAST FOLD
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

# Same pipeline as main loop
_imputer = SimpleImputer(strategy="constant", fill_value=0)
_X_tr  = _imputer.fit_transform(_df_tr[STRUCT_COLS].values.astype(np.float32))
_X_te  = _imputer.transform(_df_te[STRUCT_COLS].values.astype(np.float32))
_scaler  = StandardScaler()
_X_tr  = _scaler.fit_transform(_X_tr)
_X_te  = _scaler.transform(_X_te)

# Train models — same params as main loop
_lr = LogisticRegression(**LR_PARAMS)
_lr.fit(_X_tr, _y_tr)

_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    scale_pos_weight=(_y_tr==0).sum() / (_y_tr==1).sum(),
    eval_metric="auc", use_label_encoder=False, verbosity=0, n_jobs=-1,
    random_state=RANDOM_STATE,
)
_xgb.fit(_X_tr, _y_tr)

_rf = RandomForestClassifier(**RF_PARAMS)
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
