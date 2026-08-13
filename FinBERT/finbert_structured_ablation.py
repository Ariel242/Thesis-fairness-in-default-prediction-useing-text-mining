"""
Structured vs Structured+FinBERT — Walk-Forward Ablation
(v2: feature selection + per-algorithm regularization tuning, no fairness)
==========================================================================
Companion to analysis/preliminary_results_v2.py ("v2") — v2 is NOT imported,
edited, or executed by this script. Everything needed (walk-forward folds,
feature selection, regularization tuning, DeLong test, bootstrap CI) is
self-contained inside FinBERT/.

-----------------------------------------------------------------------------
WHAT CHANGED FROM THE 2026-08-03 RUN, AND WHY
-----------------------------------------------------------------------------
The 2026-08-03 run (see FinBERT/run_log.txt) used the FULL raw feature pool
(all ~120+ structured columns, or that plus all 51 FinBERT dims: has_desc +
50 PCA components) and hyperparameters copied verbatim from v2. Result:
FinBERT barely helped Logistic/XGBoost and actively hurt RandomForest
(mean ΔAUC: LR +0.0020, XGB +0.0011, RF -0.0026 — see fb_delta_summary.csv
from that run). That result is confounded by two things this version
isolates:
  1. Every model saw every raw column, including weak/collinear ones —
     no pressure to find the genuinely strongest predictors.
  2. Regularization was fixed, not tuned for this specific feature set —
     a model tuned for the OLD (larger, TF-IDF-free) feature space isn't
     necessarily well-regularized for THIS one.
This version adds (1) mutual-information feature selection from the full
combined pool and (2) per-algorithm regularization tuning, on BOTH arms of
the ablation (Structured and Structured+FinBERT get the identical
selection+tuning treatment), so any remaining AUC gap is attributable to
FinBERT's signal itself — not to leftover noise columns or stale
hyperparameters. Fairness metrics are dropped entirely per explicit request
(out of scope for this predictive-focused run); this script answers "does
FinBERT's deep representation improve default-risk PREDICTION" only.

-----------------------------------------------------------------------------
1. THE TWO VARIANTS  (unchanged from 2026-08-03)
-----------------------------------------------------------------------------
"Structured"          — every numeric, non-leaky, non-identifier column
                         from the post-03_advanced_prep CSV (identical
                         definition to v2's STRUCT_COLS_BASE).
"Structured+FinBERT"  — "Structured" PLUS has_desc + 50-dim PCA-compressed
                         FinBERT sem_* (mean-pooled) embedding of `desc`.
Both arms now go through the SAME two extra steps before modeling:
feature selection (section 2) and regularization tuning (section 3).

-----------------------------------------------------------------------------
2. FEATURE SELECTION: TOP-K BY MUTUAL INFORMATION, PER FOLD, PER ARM
-----------------------------------------------------------------------------
For each arm, each fold's raw pool (Structured: ~120+ columns; Structured+
FinBERT: that plus 51 FinBERT dims) is ranked by mutual_info_classif against
the target, and only the TOP_K_FEATURES=50 strongest are kept for modeling.
Fixed K, chosen in advance (not data-driven per fold) — kept simple, per
explicit request. 50 was chosen so genuine selection pressure exists on
both arms: ~40% of the Structured pool survives, ~28% of the combined pool
survives, so FinBERT's 51 dimensions must out-compete structured columns on
their own merits to make the cut (see fb_selected_features_composition.csv
/ fig3 for how many actually do, per fold).
Leakage discipline: ranking is fit on the fold's TRAINING rows only, then
the same K column indices are applied to that fold's test rows — exactly
mirroring the existing PCA/scaler fold-local discipline. The MI ranking
itself is estimated on a random subsample of up to MI_SAMPLE_CAP=30,000
training rows (kNN-based MI estimation scales poorly past that); the actual
model fit below still uses the FULL training set on the K selected columns
— only the ranking step is subsampled, purely for runtime.
Selection is computed ONCE per (fold, arm) and shared across all three
models — i.e. LR/RF/XGBoost see the same top-K columns within a given
fold/arm. Simpler than per-model selection and defensible: the question is
"what are the strongest predictors," not "what does each model happen to
lean on."

-----------------------------------------------------------------------------
3. REGULARIZATION TUNING: PER ARM, PER ALGORITHM
-----------------------------------------------------------------------------
Per the explicit brief: NOT a broad grid search — a small set of sensible,
hand-picked candidates per model, shaped by what each algorithm's
regularization actually looks like, evaluated on TUNING_FOLD_NUMBERS=[4, 8,
12] (3 of the 14 folds, spread early/mid/late so the chosen setting isn't
overfit to one data volume), and picked by mean AUC on an INNER validation
split — the last INNER_VAL_FRAC=20% of that fold's TRAINING rows,
chronologically. The fold's actual TEST rows are never touched by tuning,
for any of the 14 folds, at any point.
  - Logistic     : single knob (C, inverse L2 strength) — a 1-D sweep
                   around the old baseline (C=0.3): [0.05, 0.1, 0.3, 0.6, 1.0].
  - RandomForest : single knob (min_samples_leaf, the primary
                   regularization AND RAM-safety control — this machine has
                   13.7GB RAM and unconstrained trees OOM) — a 1-D sweep
                   never going below 10: [10, 20, 30, 50].
  - XGBoost      : regularization here is multi-dimensional (depth,
                   L1/L2, learning rate) and these interact, so instead of
                   a cartesian grid it's 5 named, holistic candidates
                   spanning "less regularized" to "more regularized"
                   (see XGB_CANDIDATES below).
Every candidate's per-tuning-fold AUC, mean AUC, and which one won is
written to fb_tuning_results.csv — full documentation of every option
tried, per the explicit request, not just the final choice.
The winning candidate per (arm, model) is locked in and reused, unchanged,
across all 14 folds of the main evaluation below — tuning happens once,
not per fold, keeping this "a few sensible tries" rather than a nested
walk-forward grid search (which would multiply runtime by ~14x for
negligible expected benefit at this candidate-set size).
All other hyperparameters (n_estimators, subsample, class_weight, etc.) are
held at their original v2/2026-08-03 values — only the parameters that
actually control regularization are tuned.
LIMITATION: tuning folds 4/8/12 were chosen for spread, not for
chronological separation from the EARLY main-loop folds' test windows.
Folds 4-8's test periods fall inside, or just before, the inner-validation
window used to tune on fold 8 (and similarly for other overlaps) — so the
chosen hyperparameters were selected using data from at-or-after some early
folds' own test periods. No fold's TEST rows are ever scored by a
candidate (the literal "don't touch the test" instruction is honored), and
since both arms go through the identical tuning procedure this does not
bias the Structured-vs-Structured+FinBERT comparison (ΔAUC, DeLong) — but
it means the ABSOLUTE AUC levels reported for the earliest folds are mildly
optimistic relative to a strict walk-forward-only tuning scheme. Worth a
caveat if these absolute numbers are quoted in the thesis.

-----------------------------------------------------------------------------
4. WALK-FORWARD PROTOCOL — MUST YIELD EXACTLY 14 FOLDS
-----------------------------------------------------------------------------
Expanding-window, chronological, no shuffling — identical parameters to v2
and to the 2026-08-03 run (MIN_TRAIN_MONTHS=8, STEP_MONTHS=3,
MIN_TEST_DEFAULTS=100), which is verified (fb_pca_variance.csv from the
2026-08-03 run) to already produce exactly 14 folds on this dataset. The
script asserts len(folds)==14 immediately after building them and stops
hard if that's ever not true, per the explicit "run on 14 folds" requirement.

-----------------------------------------------------------------------------
5. METRICS
-----------------------------------------------------------------------------
  - AUC, PR-AUC, Brier, GINI per fold per arm per model, each with a
    500-resample percentile bootstrap 95% CI (BOOT_SEED=42).
  - DeLong test (DeLong et al. 1988), per fold per model: Structured vs
    Structured+FinBERT AUC, paired on the same test set.
  - Feature importance + importance mass (FinBERT share vs Structured
    share), computed directly from the already-fitted last-fold
    Structured+FinBERT models (captured during the main loop, not
    refit separately — avoids the duplicate-fitting the 2026-08-03
    version did for this section).
  - NEW: selected-feature composition per fold — how many of the winning
    top-50 are structured vs FinBERT (fb_selected_features_composition.csv,
    fig3) — the most direct answer to "are FinBERT dimensions actually
    among the strongest features, or mostly filtered out."
  - NEW: per-dimension contribution across ALL 14 folds, both arms
    (fb_feature_importance_by_fold.csv — every feature's importance/rank in
    every fold it was selected in) and a rolled-up consistency summary
    (fb_feature_consistency_summary.csv, printed as section G) — for each
    (arm, model): how many distinct features were EVER selected across the
    14 folds, how many were selected in all 14 / >=12 / >=8, and the top-15
    most consistent contributors with their mean rank and mean importance
    share when selected. Answers "how many dimensions contribute
    consistently" directly, rather than inferring it from one fold's
    snapshot.

-----------------------------------------------------------------------------
6. WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------------------------------------------
  - No fairness metrics of any kind (ZIP3 groups, FNR/FPR/Brier gaps,
    thresholds) — out of scope for this run per explicit instruction.
  - Does not modify, import, or execute analysis/preliminary_results_v2.py.
  - Does not combine FinBERT with TF-IDF.
  - Writes nothing outside this FinBERT/ folder.

-----------------------------------------------------------------------------
7. OUTPUTS (written under FinBERT/) — overwrites the 2026-08-03 run's files
   with the same names; fairness-only files from that run (fb_fairness_*,
   fb_group_coverage.csv, fb_zip_summary.csv) are NOT recreated and will be
   stale leftovers until manually removed.
-----------------------------------------------------------------------------
results/
  fb_predictive_folds.csv               — per-fold predictive metrics + CIs
  fb_predictive_summary.csv             — predictive summary (mean +/- std)
  fb_delta_summary.csv                  — FinBERT vs Structured deltas
  fb_delong.csv                         — DeLong test per fold
  fb_pca_variance.csv                   — PCA explained variance, per fold
  fb_tuning_results.csv                 — NEW: every regularization candidate
                                           tried, per tuning fold, + winner flag
  fb_selected_features_composition.csv  — NEW: structured vs FinBERT count
                                           in the selected top-K, per fold/arm
  fb_importance_mass.csv                — % importance FinBERT vs structured
                                           (last fold, selected features only)
  fb_feature_importance_by_fold.csv     — NEW: every feature's importance/
                                           rank in every fold it was selected
                                           in (both arms, all 3 models)
  fb_feature_consistency_summary.csv    — NEW: per (arm, model), per feature —
                                           how many of 14 folds it was
                                           selected in, mean rank, mean
                                           importance share
figures/
  fig1_auc_stability.png
  fig2_delong_delta_auc.png
  fig3_feature_selection_composition.png  — NEW
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS — edit these before running
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent            # .../F-TM-CR/FinBERT
BASE_DIR   = SCRIPT_DIR.parent                           # .../F-TM-CR  (data lives here; never written to)

PATH_CSV     = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
PATH_FINBERT = SCRIPT_DIR / "data" / "finbert_desc_embeddings.parquet"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"

# Walk-Forward parameters (identical to v2 / 2026-08-03 run — must yield 14 folds)
MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

# Bootstrap CI
N_BOOTSTRAP = 500
BOOT_SEED   = 42

# FinBERT block
N_PCA_COMPONENTS = 50   # 768 sem_* dims compressed to this many

# Feature selection (section 2)
TOP_K_FEATURES = 50     # fixed, chosen in advance — see docstring section 2
MI_SAMPLE_CAP  = 30000  # cap rows used to *rank* features (model fit still uses full train set)

# Regularization tuning (section 3)
TUNING_FOLD_NUMBERS = [4, 8, 12]   # 3 of 14 folds, spread early/mid/late
INNER_VAL_FRAC      = 0.20         # last 20% of each tuning fold's TRAIN rows, chronologically

RANDOM_STATE = 242   # shared across models, PCA, MI subsampling, and tuning

# --- Fixed (non-tuned) parts of each model's spec ---
LR_BASE_PARAMS = dict(
    penalty      = "l2",
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)
RF_BASE_PARAMS = dict(
    n_estimators     = 500,
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)

# --- Regularization candidates (section 3) ---
LR_CANDIDATES = [
    {"label": "C=0.05",          "C": 0.05},
    {"label": "C=0.1",           "C": 0.1},
    {"label": "C=0.3 (old baseline)", "C": 0.3},
    {"label": "C=0.6",           "C": 0.6},
    {"label": "C=1.0",           "C": 1.0},
]
RF_CANDIDATES = [
    {"label": "min_leaf=10",              "min_samples_leaf": 10},
    {"label": "min_leaf=20 (old baseline)", "min_samples_leaf": 20},
    {"label": "min_leaf=30",              "min_samples_leaf": 30},
    {"label": "min_leaf=50",              "min_samples_leaf": 50},
]
XGB_CANDIDATES = [
    {"label": "baseline (depth4, lr.05, a0, L1)",        "max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"label": "shallow+L2 (depth3, lr.05, a0, L5)",       "max_depth": 3, "learning_rate": 0.05, "reg_alpha": 0.0, "reg_lambda": 5.0},
    {"label": "shallow+L1L2 (depth3, lr.05, a.5, L5)",    "max_depth": 3, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0},
    {"label": "deeper light-reg (depth5, lr.05, a0, L1)", "max_depth": 5, "learning_rate": 0.05, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"label": "strong reg (depth3, lr.03, a1, L10)",      "max_depth": 3, "learning_rate": 0.03, "reg_alpha": 1.0, "reg_lambda": 10.0},
]
CANDIDATES  = {"Logistic": LR_CANDIDATES, "XGBoost": XGB_CANDIDATES, "RandomForest": RF_CANDIDATES}
MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]
ARMS        = ["Structured", "Structured+FinBERT"]


def make_model(model_name, params, y_tr):
    if model_name == "Logistic":
        p = dict(LR_BASE_PARAMS); p["C"] = params["C"]
        return LogisticRegression(**p)
    elif model_name == "RandomForest":
        p = dict(RF_BASE_PARAMS); p["min_samples_leaf"] = params["min_samples_leaf"]
        return RandomForestClassifier(**p)
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = params["max_depth"],
            learning_rate    = params["learning_rate"],
            reg_alpha        = params["reg_alpha"],
            reg_lambda       = params["reg_lambda"],
            subsample        = 0.8,
            scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum(),
            eval_metric      = "auc",
            use_label_encoder= False,
            verbosity        = 0,
            n_jobs           = -1,
            random_state     = RANDOM_STATE,
        )
    raise ValueError(model_name)

# ============================================================
# 1. LOAD STRUCTURED DATA
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

# ============================================================
# 1b. RESTORE ORDINAL RISK GRADES (identical fix to v2)
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
# 2. STRUCTURED FEATURE LIST (identical definition to v2's STRUCT_COLS_BASE)
# ============================================================
EXCLUDE_COLS = {TARGET_COL, DATE_COL,
                "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "zip3", "emp_title", "title", "desc", "funded_ratio",
                "text_all_clean", "desc_clean", "title_clean", "emp_title_clean",
                "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage — see v2 changelog
STRUCT_COLS_BASE = [c for c in df.columns
                    if c not in EXCLUDE_COLS
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                        np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features (raw pool, pre-selection): {len(STRUCT_COLS_BASE)}")

# ============================================================
# 3. MERGE FINBERT EMBEDDINGS (by id — never row position)
# ============================================================
print("Loading FinBERT embeddings...")
SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
for c in SEM_COLS:
    df_fb[c] = df_fb[c].astype(np.float32)

n_before = len(df)
df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == n_before, "Merge changed row count — id is not a clean 1:1 key"
assert df["has_desc"].isna().sum() == 0, "Unmatched rows after FinBERT merge — check id coverage"
print(f"  Merged embeddings for {len(df):,} rows ({df['has_desc'].mean():.1%} have a non-empty description)")

# ============================================================
# 4. DELONG TEST
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
    auc2, V10_2, V01_2, _,  _  = auc_and_kernel(y, np.asarray(p2))

    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)
    var_diff = (S10[0,0]/n1 + S01[0,0]/n0) + (S10[1,1]/n1 + S01[1,1]/n0) \
             - 2*(S10[0,1]/n1 + S01[0,1]/n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan, np.nan
    diff = auc1 - auc2
    se   = np.sqrt(var_diff)
    z    = diff / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval, diff - 1.96*se, diff + 1.96*se

# ============================================================
# 5. BOOTSTRAP CI
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
# 6. WALK-FORWARD FOLDS (identical logic to v2)
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
            "fold": len(folds) + 1,
            "train_cutoff": str(train_cutoff),
            "test_end": str(test_period),
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "n_test_defaults": int(df.loc[te_mask, TARGET_COL].sum()),
            "tr_idx": df.index[tr_mask].tolist(),   # chronologically ordered — df is globally date-sorted
            "te_idx": df.index[te_mask].tolist(),
        })
        fold_start_idx += step_months
    return folds

print("\nBuilding Walk-Forward folds...")
folds = build_folds(df, DATE_COL, MIN_TRAIN_MONTHS, STEP_MONTHS, MIN_TEST_DEFAULTS)
assert len(folds) == 14, (
    f"Expected exactly 14 walk-forward folds, got {len(folds)}. "
    f"Check MIN_TRAIN_MONTHS/STEP_MONTHS/MIN_TEST_DEFAULTS or whether PATH_CSV changed."
)
print(f"  Total folds: {len(folds)}")
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  defaults_in_test={f['n_test_defaults']}")

# ============================================================
# 7. FEATURE MATRIX + SELECTION (shared by tuning and main loop)
# ============================================================
def build_arm_matrix(df, tr_idx, te_idx, arm):
    """Impute/scale the structured block; add the fold-local-PCA FinBERT block if arm requires it.
    No selection yet — returns the FULL raw pool for this arm."""
    df_tr, df_te = df.loc[tr_idx], df.loc[te_idx]
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer.fit_transform(df_tr[STRUCT_COLS_BASE].values.astype(np.float32))
    X_te_s = imputer.transform(df_te[STRUCT_COLS_BASE].values.astype(np.float32))
    scaler_s = StandardScaler()
    X_tr_s = scaler_s.fit_transform(X_tr_s)
    X_te_s = scaler_s.transform(X_te_s)
    names = list(STRUCT_COLS_BASE)
    kinds = ["structured"] * len(names)

    evr = None
    if arm == "Structured+FinBERT":
        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
        pca_tr = pca.fit_transform(df_tr[SEM_COLS].values)
        pca_te = pca.transform(df_te[SEM_COLS].values)
        evr = float(pca.explained_variance_ratio_.sum())
        fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
        fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
        scaler_fb = StandardScaler()
        fb_tr = scaler_fb.fit_transform(fb_tr_raw)
        fb_te = scaler_fb.transform(fb_te_raw)
        X_tr_pool = np.hstack([X_tr_s, fb_tr])
        X_te_pool = np.hstack([X_te_s, fb_te])
        names = names + ["fb:has_desc"] + [f"fb:pca_{i:03d}" for i in range(N_PCA_COMPONENTS)]
        kinds = kinds + ["FinBERT"] * (1 + N_PCA_COMPONENTS)
    else:
        X_tr_pool, X_te_pool = X_tr_s, X_te_s

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)
    return X_tr_pool, X_te_pool, y_tr, y_te, names, kinds, evr


def select_features(X_tr_pool, y_tr, names, kinds, k=TOP_K_FEATURES, sample_cap=MI_SAMPLE_CAP, seed=RANDOM_STATE):
    """Rank the pool by mutual information (fit on TRAIN only, per docstring section 2) and keep the top-k."""
    n = X_tr_pool.shape[0]
    if n > sample_cap:
        rng = np.random.default_rng(seed)
        sub = rng.choice(n, size=sample_cap, replace=False)
        X_mi, y_mi = X_tr_pool[sub], y_tr[sub]
    else:
        X_mi, y_mi = X_tr_pool, y_tr
    k_eff = min(k, X_tr_pool.shape[1])
    mi = mutual_info_classif(X_mi, y_mi, random_state=seed)
    top_idx = np.sort(np.argsort(mi)[::-1][:k_eff])
    sel_names = [names[i] for i in top_idx]
    sel_kinds = [kinds[i] for i in top_idx]
    return top_idx, sel_names, sel_kinds, mi[top_idx]


def extract_importance(model_name, model, n_features):
    """Raw per-feature importance, aligned to the column order the model was fit on."""
    if model_name == "Logistic":
        return np.abs(model.coef_[0])
    elif model_name == "XGBoost":
        gain = model.get_booster().get_score(importance_type="gain")
        return np.array([gain.get(f"f{i}", 0.0) for i in range(n_features)])
    else:
        return model.feature_importances_

# ============================================================
# 8. REGULARIZATION TUNING (section 3)
# ============================================================
print("\n" + "="*70)
print("REGULARIZATION TUNING (per arm, per model)")
print("="*70)
print(f"  Tuning folds: {TUNING_FOLD_NUMBERS}  |  Inner validation = last {INNER_VAL_FRAC:.0%} "
      f"of each fold's TRAIN rows, chronologically. Test sets are never touched.\n")

tuning_folds = [f for f in folds if f["fold"] in TUNING_FOLD_NUMBERS]
assert len(tuning_folds) == len(TUNING_FOLD_NUMBERS), \
    f"Expected tuning folds {TUNING_FOLD_NUMBERS} not all found among the {len(folds)} folds."

tuning_rows = []
BEST_PARAMS = {arm: {} for arm in ARMS}

for arm in ARMS:
    print(f"\n-- Arm: {arm} --")
    # Selection + inner split computed once per tuning fold, shared across all model candidates
    fold_data = {}
    for tf in tuning_folds:
        n_val = max(1, int(round(len(tf["tr_idx"]) * INNER_VAL_FRAC)))
        inner_tr, inner_val = tf["tr_idx"][:-n_val], tf["tr_idx"][-n_val:]
        X_tr_pool, X_val_pool, y_tr_i, y_val_i, names, kinds, _evr = build_arm_matrix(df, inner_tr, inner_val, arm)
        idx, sel_names, sel_kinds, _ = select_features(X_tr_pool, y_tr_i, names, kinds)
        fold_data[tf["fold"]] = (X_tr_pool[:, idx], X_val_pool[:, idx], y_tr_i, y_val_i)

    for model_name in MODEL_NAMES:
        candidates = CANDIDATES[model_name]
        cand_results = []
        for cand in candidates:
            aucs = []
            for tf in tuning_folds:
                X_tr_sel, X_val_sel, y_tr_i, y_val_i = fold_data[tf["fold"]]
                model = make_model(model_name, cand, y_tr_i)
                model.fit(X_tr_sel, y_tr_i)
                p = model.predict_proba(X_val_sel)[:, 1]
                aucs.append(roc_auc_score(y_val_i, p))
            cand_results.append((cand, aucs, float(np.mean(aucs))))

        best_cand, best_aucs, best_mean = max(cand_results, key=lambda t: t[2])
        BEST_PARAMS[arm][model_name] = best_cand

        for cand, aucs, mean_auc in cand_results:
            is_winner = (cand["label"] == best_cand["label"])
            for tf, auc in zip(tuning_folds, aucs):
                tuning_rows.append({
                    "Arm": arm, "Model": model_name, "Candidate": cand["label"],
                    "params": json.dumps({k: v for k, v in cand.items() if k != "label"}),
                    "tuning_fold": tf["fold"], "inner_val_AUC": round(auc, 4),
                    "mean_AUC_over_tuning_folds": round(mean_auc, 4), "is_winner": is_winner,
                })

        print(f"  [{model_name}] winner: {best_cand['label']}  (mean inner-val AUC = {best_mean:.4f})")
        for cand, aucs, mean_auc in sorted(cand_results, key=lambda t: -t[2]):
            flag = " <-- chosen" if cand["label"] == best_cand["label"] else ""
            print(f"      {cand['label']:<40s} mean AUC={mean_auc:.4f}{flag}")

df_tuning = pd.DataFrame(tuning_rows)

# ============================================================
# 9. MAIN WALK-FORWARD EVALUATION — all 14 folds, tuned + feature-selected
# ============================================================
print("\n" + "="*70)
print("MAIN WALK-FORWARD EVALUATION (14 folds)")
print("="*70)

pred_rows, delong_rows, pca_rows, composition_rows, importance_rows = [], [], [], [], []
last_fold_num = folds[-1]["fold"]
last_fold_fitted = {}   # model_name -> {"model":..., "names":..., "kinds":...} for Structured+FinBERT, last fold

for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    fold_preds = {m: {} for m in MODEL_NAMES}

    for arm in ARMS:
        X_tr_pool, X_te_pool, y_tr, y_te, names, kinds, evr = build_arm_matrix(df, tr, te, arm)
        idx, sel_names, sel_kinds, _ = select_features(X_tr_pool, y_tr, names, kinds)
        X_tr_sel, X_te_sel = X_tr_pool[:, idx], X_te_pool[:, idx]

        n_fb_selected = sum(1 for k_ in sel_kinds if k_ == "FinBERT")
        composition_rows.append({
            "fold": f["fold"], "Arm": arm,
            "n_structured_selected": len(sel_kinds) - n_fb_selected,
            "n_finbert_selected": n_fb_selected,
        })

        if arm == "Structured+FinBERT":
            pca_rows.append({"fold": f["fold"], "n_components": N_PCA_COMPONENTS,
                              "explained_variance_ratio": round(evr, 4)})

        for model_name in MODEL_NAMES:
            params = BEST_PARAMS[arm][model_name]
            model = make_model(model_name, params, y_tr)
            model.fit(X_tr_sel, y_tr)
            p = model.predict_proba(X_te_sel)[:, 1]
            fold_preds[model_name][arm] = p

            if f["fold"] == last_fold_num and arm == "Structured+FinBERT":
                last_fold_fitted[model_name] = {"model": model, "names": sel_names, "kinds": sel_kinds}

            # Per-fold feature contribution — every fold, both arms, no extra fitting (reuses the model above)
            imp_raw = extract_importance(model_name, model, len(sel_names))
            imp_sum = imp_raw.sum()
            imp_share = imp_raw / imp_sum if imp_sum > 0 else imp_raw
            rank_of = np.empty(len(sel_names), dtype=int)
            rank_of[np.argsort(imp_raw)[::-1]] = np.arange(1, len(sel_names) + 1)
            for feat_i, feat_name in enumerate(sel_names):
                importance_rows.append({
                    "fold": f["fold"], "Arm": arm, "Model": model_name,
                    "Feature": feat_name, "Kind": sel_kinds[feat_i],
                    "Importance_raw": round(float(imp_raw[feat_i]), 6),
                    "Importance_share": round(float(imp_share[feat_i]), 6),
                    "Rank": int(rank_of[feat_i]),
                })

            auc   = roc_auc_score(y_te, p)
            prauc = average_precision_score(y_te, p)
            brier = brier_score_loss(y_te, p)
            auc_lo,   auc_hi   = bootstrap_ci(y_te, p, roc_auc_score)
            prauc_lo, prauc_hi = bootstrap_ci(y_te, p, average_precision_score)
            brier_lo, brier_hi = bootstrap_ci(y_te, p, brier_score_loss)

            pred_rows.append({
                "fold": f["fold"], "train_cutoff": f["train_cutoff"], "test_end": f["test_end"],
                "n_train": f["n_train"], "n_test": f["n_test"], "n_defaults": f["n_test_defaults"],
                "Model": model_name, "Variant": arm, "n_features_selected": len(sel_names),
                "AUC": round(auc,4), "AUC_CI_lo": round(auc_lo,4) if not np.isnan(auc_lo) else np.nan,
                "AUC_CI_hi": round(auc_hi,4) if not np.isnan(auc_hi) else np.nan,
                "PR_AUC": round(prauc,4), "PR_AUC_CI_lo": round(prauc_lo,4) if not np.isnan(prauc_lo) else np.nan,
                "PR_AUC_CI_hi": round(prauc_hi,4) if not np.isnan(prauc_hi) else np.nan,
                "Brier": round(brier,4), "Brier_CI_lo": round(brier_lo,4) if not np.isnan(brier_lo) else np.nan,
                "Brier_CI_hi": round(brier_hi,4) if not np.isnan(brier_hi) else np.nan,
                "GINI": round(2*auc-1, 4),
            })

    y_te_full = df.loc[te, TARGET_COL].values.astype(int)
    for model_name in MODEL_NAMES:
        p_s, p_f = fold_preds[model_name]["Structured"], fold_preds[model_name]["Structured+FinBERT"]
        auc_s, auc_f, z, pval, ci_lo, ci_hi = delong_auc_test(y_te_full, p_s, p_f)
        delong_rows.append({
            "fold": f["fold"], "Model": model_name,
            "AUC_Structured": round(auc_s,4), "AUC_Structured+FinBERT": round(auc_f,4),
            "Delta_AUC": round(auc_f-auc_s,4),
            "Z_stat": round(z,3) if not np.isnan(z) else np.nan,
            "p_value": round(pval,4) if not np.isnan(pval) else np.nan,
            "CI_95_lo": round(ci_lo,4) if not np.isnan(ci_lo) else np.nan,
            "CI_95_hi": round(ci_hi,4) if not np.isnan(ci_hi) else np.nan,
        })

    print(f"  Fold {f['fold']} done.")

# ============================================================
# 10. AGGREGATE RESULTS
# ============================================================
df_pred        = pd.DataFrame(pred_rows)
df_delong      = pd.DataFrame(delong_rows)
df_pca         = pd.DataFrame(pca_rows)
df_composition = pd.DataFrame(composition_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]
pred_summary = df_pred.groupby(["Model","Variant"])[PRED_METRICS].agg(["mean","std"]).round(4)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean +/- std across 14 folds)")
print("="*70)
print(pred_summary.to_string())

delta_rows = []
for m in df_pred["Model"].unique():
    for fold in df_pred["fold"].unique():
        base = df_pred[(df_pred["Model"]==m)&(df_pred["Variant"]=="Structured")&(df_pred["fold"]==fold)]
        full = df_pred[(df_pred["Model"]==m)&(df_pred["Variant"]=="Structured+FinBERT")&(df_pred["fold"]==fold)]
        if base.empty or full.empty:
            continue
        row = {"fold": fold, "Model": m}
        for c in PRED_METRICS:
            row[f"Δ_{c}"] = round(float(full.iloc[0][c]) - float(base.iloc[0][c]), 4)
        delta_rows.append(row)
df_delta = pd.DataFrame(delta_rows)
delta_metric_cols = [f"Δ_{c}" for c in PRED_METRICS]
df_delta_summary = df_delta.groupby("Model")[delta_metric_cols].agg(["mean","std"]).round(4)

print("\n" + "="*70)
print("B. DELTA: Structured+FinBERT - Structured (mean +/- std across 14 folds)")
print("="*70)
print(df_delta_summary.to_string())

print("\n  Consistency check (fraction of folds where FinBERT improved AUC):")
for m in df_delta["Model"].unique():
    sub = df_delta[df_delta["Model"]==m]
    frac = (sub["Δ_AUC"]>0).mean(); mean_d = sub["Δ_AUC"].mean()
    verdict = "consistent" if frac>=0.7 else ("mixed" if frac>=0.4 else "mostly worse")
    print(f"    [{m}]  improved in {frac:.0%} of folds  mean ΔAUC={mean_d:+.4f}  -> {verdict}")

print("\n" + "="*70)
print("C. DELONG TEST: Structured vs Structured+FinBERT (per fold)")
print("="*70)
print(df_delong.to_string(index=False))
delong_summary = df_delong.groupby("Model")[["Delta_AUC","Z_stat","p_value"]].mean().round(4)
print("\n  DeLong — mean across folds:")
print(delong_summary.to_string())

print("\n" + "="*70)
print("D. SELECTED-FEATURE COMPOSITION (structured vs FinBERT, per fold)")
print("="*70)
print(df_composition.pivot(index="fold", columns="Arm",
                            values=["n_structured_selected","n_finbert_selected"]).to_string())

print("\n" + "="*70)
print("E. REGULARIZATION TUNING — chosen hyperparameters")
print("="*70)
for arm in ARMS:
    for model_name in MODEL_NAMES:
        print(f"  [{arm}] {model_name}: {BEST_PARAMS[arm][model_name]['label']}")

print("\n" + "="*70)
print("F. PCA EXPLAINED VARIANCE PER FOLD")
print("="*70)
print(df_pca.to_string(index=False))

# ============================================================
# 10b. FEATURE CONTRIBUTION CONSISTENCY — across all 14 folds, both arms
#      (per docstring section 5: which dimensions contribute, and how
#      consistently, not just a last-fold snapshot)
# ============================================================
df_importance = pd.DataFrame(importance_rows)

consistency_parts = []
for arm in ARMS:
    for model_name in MODEL_NAMES:
        sub = df_importance[(df_importance["Arm"]==arm) & (df_importance["Model"]==model_name)]
        grp = sub.groupby(["Feature","Kind"]).agg(
            n_folds_selected=("fold","nunique"),
            mean_rank_when_selected=("Rank","mean"),
            mean_importance_share=("Importance_share","mean"),
        ).reset_index()
        grp["Arm"] = arm
        grp["Model"] = model_name
        grp["pct_folds_selected"] = (grp["n_folds_selected"] / len(folds) * 100).round(1)
        grp["mean_rank_when_selected"] = grp["mean_rank_when_selected"].round(2)
        grp["mean_importance_share"] = grp["mean_importance_share"].round(4)
        consistency_parts.append(grp)
df_consistency = pd.concat(consistency_parts, ignore_index=True)
df_consistency = df_consistency.sort_values(
    ["Arm","Model","n_folds_selected","mean_rank_when_selected"], ascending=[True,True,False,True]
).reset_index(drop=True)

print("\n" + "="*70)
print("G. FEATURE CONTRIBUTION CONSISTENCY (how many dimensions contribute, and how often)")
print("="*70)
for arm in ARMS:
    for model_name in MODEL_NAMES:
        sub = df_consistency[(df_consistency["Arm"]==arm) & (df_consistency["Model"]==model_name)]
        n_ever = len(sub)
        n_all14 = int((sub["n_folds_selected"] == len(folds)).sum())
        n_12plus = int((sub["n_folds_selected"] >= 12).sum())
        n_8plus = int((sub["n_folds_selected"] >= 8).sum())
        print(f"\n  [{arm} / {model_name}]  {n_ever} distinct features were EVER among the top-{TOP_K_FEATURES} "
              f"across the {len(folds)} folds")
        print(f"    selected in all {len(folds)} folds: {n_all14}   |   in >=12/{len(folds)}: {n_12plus}   |   "
              f"in >=8/{len(folds)}: {n_8plus}")
        top15 = sub.head(15)[["Feature","Kind","n_folds_selected","pct_folds_selected",
                               "mean_rank_when_selected","mean_importance_share"]]
        print(f"    Top 15 most consistent contributors:")
        print(top15.to_string(index=False))

# ============================================================
# 11. SAVE
# ============================================================
out_dir = SCRIPT_DIR / "results"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(out_dir / "fb_predictive_folds.csv", index=False)
df_delong.to_csv(out_dir / "fb_delong.csv", index=False)
pred_summary.to_csv(out_dir / "fb_predictive_summary.csv")
df_delta_summary.to_csv(out_dir / "fb_delta_summary.csv")
df_pca.to_csv(out_dir / "fb_pca_variance.csv", index=False)
df_tuning.to_csv(out_dir / "fb_tuning_results.csv", index=False)
df_composition.to_csv(out_dir / "fb_selected_features_composition.csv", index=False)
df_importance.to_csv(out_dir / "fb_feature_importance_by_fold.csv", index=False)
df_consistency.to_csv(out_dir / "fb_feature_consistency_summary.csv", index=False)
print(f"\nSaved results to: {out_dir}")

# ============================================================
# 12. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

figures_dir = SCRIPT_DIR / "figures"
os.makedirs(figures_dir, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

_CLR = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}

print("\nGenerating figures...")

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    color = _CLR[model]
    for variant, ls, marker, alpha in [("Structured","-","o",1.00), ("Structured+FinBERT","--","s",0.70)]:
        sub = df_pred[(df_pred["Model"]==model)&(df_pred["Variant"]==variant)].sort_values("fold")
        ax.plot(sub["fold"], sub["AUC"], linestyle=ls, marker=marker, color=color, alpha=alpha,
                linewidth=2, markersize=5, label=variant)
        if not sub["AUC_CI_lo"].isna().all():
            ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"], alpha=0.12, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]: ax.set_ylabel("AUC")
    ax.legend(title="Variant", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.suptitle("Walk-Forward AUC Stability Across Folds — Tuned + Feature-Selected\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig1_auc_stability.png")

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    sub = df_delong[df_delong["Model"]==model].sort_values("fold")
    pvals = sub["p_value"].fillna(1.0).values
    bar_colors = ["#d73027" if p<0.05 else "#bababa" for p in pvals]
    x = sub["fold"].values; y = sub["Delta_AUC"].values
    ax.bar(x, y, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ci_lo, ci_hi = sub["CI_95_lo"].values, sub["CI_95_hi"].values
    valid = ~(np.isnan(ci_lo)|np.isnan(ci_hi))
    if valid.any():
        half_width = (ci_hi-ci_lo)/2
        ax.errorbar(x[valid], y[valid], yerr=half_width[valid], fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
    ax.axhline(0, color="black", linewidth=1.2, zorder=5)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]: ax.set_ylabel("ΔAUC  (Structured+FinBERT − Structured)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    sig_patch = mpatches.Patch(color="#d73027", alpha=0.85, label="p < 0.05")
    ns_patch  = mpatches.Patch(color="#bababa", alpha=0.85, label="p >= 0.05")
    ax.legend(handles=[sig_patch, ns_patch], title="DeLong test", fontsize=9)
fig.suptitle("DeLong Test: ΔAUC per Fold  (Structured+FinBERT − Structured)\n"
             "(bars above zero = FinBERT improves AUC; error bars = 95% CI)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig2_delong_delta_auc.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig2_delong_delta_auc.png")

fig, ax = plt.subplots(figsize=(10,5))
comp_fb = df_composition[df_composition["Arm"]=="Structured+FinBERT"].sort_values("fold")
ax.bar(comp_fb["fold"], comp_fb["n_structured_selected"], label="Structured", color="#4393c3")
ax.bar(comp_fb["fold"], comp_fb["n_finbert_selected"], bottom=comp_fb["n_structured_selected"],
       label="FinBERT", color="#d6604d")
ax.set_xlabel("Fold (chronological)")
ax.set_ylabel(f"# features selected (of top {TOP_K_FEATURES})")
ax.set_title(f"Composition of the Top-{TOP_K_FEATURES} Selected Features per Fold\n"
             "(Structured+FinBERT arm — how many FinBERT dimensions make the cut)",
             fontsize=12, fontweight="bold")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "fig3_feature_selection_composition.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig3_feature_selection_composition.png")
print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 13. FEATURE IMPORTANCE — LAST FOLD, STRUCTURED+FINBERT
#     (reuses the models already fit in the main loop — no refitting)
# ============================================================
print("\n" + "="*70)
print(f"FEATURE IMPORTANCE — Fold {last_fold_num}, Structured+FinBERT (top-{TOP_K_FEATURES} selected features)")
print("="*70)

def importance_table(model_name):
    info = last_fold_fitted[model_name]
    model, names, kinds = info["model"], info["names"], info["kinds"]
    imp = extract_importance(model_name, model, len(names))
    col = "Gain" if model_name == "XGBoost" else "Importance"
    dfi = (pd.DataFrame({"Feature": names, col: np.round(imp, 5), "Kind": kinds})
           .sort_values(col, ascending=False).reset_index(drop=True))
    dfi.index += 1
    return dfi, col

TOP_N = 20
imp_tables = {}
for model_name in MODEL_NAMES:
    dfi, col = importance_table(model_name)
    imp_tables[model_name] = (dfi, col)
    print(f"\n-- {model_name} — Top {min(TOP_N, len(dfi))} of {len(dfi)} selected features --")
    print(dfi.head(TOP_N).to_string())

mass_rows = []
for model_name, (dfi, col) in imp_tables.items():
    total = dfi[col].sum()
    fb_mass = dfi[dfi["Kind"]=="FinBERT"][col].sum()
    mass_rows.append({
        "Model": model_name,
        "FinBERT_share": round(fb_mass/total, 4) if total > 0 else np.nan,
        "Structured_share": round(1 - fb_mass/total, 4) if total > 0 else np.nan,
        "n_finbert_features_in_selected_set": int((dfi["Kind"]=="FinBERT").sum()),
        "n_structured_features_in_selected_set": int((dfi["Kind"]=="structured").sum()),
    })
df_mass = pd.DataFrame(mass_rows)
df_mass.to_csv(out_dir / "fb_importance_mass.csv", index=False)
print(f"\n-- Importance mass: FinBERT vs Structured (within selected top-{TOP_K_FEATURES}, fold {last_fold_num}) --")
print(df_mass.to_string(index=False))

print(f"\nDone. Results in {out_dir}, figures in {figures_dir}")
