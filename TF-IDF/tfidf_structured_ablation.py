"""
Structured vs Structured+TF-IDF(full) vs Structured+TF-IDF(chi2 top-K) — Walk-Forward Ablation
================================================================================================
Companion to analysis/BASELINE.py (structured-only baseline) and TF-IDF/tfidf_pipeline.py
(which already fit + saved per-fold TF-IDF matrices, uncapped vocab: 6,227 -> 55,028 columns
across the 14 folds). Neither of those scripts is imported/edited/executed here — per this
repo's established convention (no shared module anywhere), build_folds(), the DeLong test,
bootstrap CI, and the fixed model hyperparameters are all copied in verbatim, same as
FinBERT/finbert_structured_ablation.py did for its own (Structured vs Structured+FinBERT)
ablation. This script does not touch FinBERT or any file outside TF-IDF/.

--------------------------------------------------------------------------------------------
1. THE THREE ARMS
--------------------------------------------------------------------------------------------
"Structured"                — every numeric, non-leaky, non-identifier column from the
                               post-03_advanced_prep CSV (identical definition to
                               BASELINE.py's STRUCT_COLS).
"Structured+TFIDF_full"     — Structured PLUS the fold's full (uncapped) TF-IDF matrix,
                               loaded directly from TF-IDF/output/fold{NN}/*.npz — NOT
                               recomputed here.
"Structured+TFIDF_chi2"     — Structured PLUS only the top CHI2_TOP_K=500 TF-IDF columns
                               for that fold, selected by chi2 (SelectKBest), fit on TRAIN
                               only. Selection runs on the TF-IDF block alone — structured
                               columns are never candidates for removal, only the TF-IDF
                               side is being reduced (chi2 was chosen over mutual-info
                               because it is fast/native on large sparse non-negative
                               matrices, and over SVD/PCA because it keeps literal,
                               nameable words/bigrams — interpretability was the explicit
                               reason for using TF-IDF at all).

--------------------------------------------------------------------------------------------
2. HYPERPARAMETERS — FIXED, NOT TUNED PER ARM
--------------------------------------------------------------------------------------------
Unlike FinBERT/finbert_structured_ablation.py (which re-tunes regularization per arm),
this script reuses BASELINE.py's fixed hyperparameters (LR C=0.3, RF n_estimators=500 /
min_samples_leaf=20, XGBoost depth=4/lr=0.05/n=300) unchanged across all three arms.
Decided explicitly: a full per-arm tuning sweep would multiply an already RAM/time-risky
RandomForest fit (500 trees on up to ~55K sparse columns at fold 14) by 4-5 extra
candidates x 3 tuning folds, for uncertain benefit. The only thing that varies by arm is
n_jobs (RandomForest/XGBoost use n_jobs=4 on the two TF-IDF arms instead of -1, to bound
peak RAM on this 13.7GB machine during wide sparse fits).

--------------------------------------------------------------------------------------------
3. KNOWN RISK
--------------------------------------------------------------------------------------------
Structured+TFIDF_full at fold 14 is ~202,442 rows x ~55,154 columns (sparse). RandomForest
(500 trees) on that is the main runtime/RAM risk. DEBUG_MAX_FOLDS lets you smoke-test on
the first N (cheap) folds before committing to the full 14-fold run — fold 1 is only
6,227 TF-IDF columns, fold 14 is 55,028, so early folds give a fast, honest timing signal.

--------------------------------------------------------------------------------------------
4. OUTPUTS (written under TF-IDF/)
--------------------------------------------------------------------------------------------
results/
  ablation_folds.csv               — per fold x arm x model: AUC/PR-AUC/Brier/GINI + 95% CI
  ablation_summary.csv             — mean +/- std across folds, grouped by (arm, model)
  ablation_delong.csv              — pairwise DeLong per fold x model x arm-pair (3 pairs)
  ablation_feature_importance.csv  — per fold x arm x model x feature: importance/rank
                                      (Structured + chi2 arms: ALL features; full arm: top 50
                                      only, per fold/model — storing all ~55K would be noise)
  ablation_feature_consistency.csv — cross-fold stability of the chi2 arm's selected words —
                                      the "which columns contributed, candidates for the final
                                      model" deliverable
  ablation_report.md               — written methodology/results/conclusions document
figures/
  fig1_auc_by_fold.png             — AUC per fold, per model, 3 arms overlaid, with CI bands
  fig2_delong_deltas.png           — ΔAUC bar chart per model x arm-pair, colored by significance
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS — edit these before running
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent            # .../F-TM-CR/TF-IDF
BASE_DIR   = SCRIPT_DIR.parent                           # .../F-TM-CR

PATH_CSV = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
TFIDF_DIR = SCRIPT_DIR / "output"                        # per-fold *.npz/vocab.txt written by tfidf_pipeline.py

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"

# Walk-Forward parameters (identical to BASELINE.py / tfidf_pipeline.py — must yield 14 folds)
MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

# Bootstrap CI
N_BOOTSTRAP = 500
BOOT_SEED   = 42

RANDOM_STATE = 242   # shared across all models — matches BASELINE.py / FinBERT ablation

# chi2 feature selection (TF-IDF columns only — see docstring section 1)
CHI2_TOP_K = 500

ARMS        = ["Structured", "Structured+TFIDF_full", "Structured+TFIDF_chi2"]
MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]

N_JOBS_SMALL = -1   # Structured arm — small/dense, matches BASELINE.py exactly
N_JOBS_WIDE  = 4    # the two TF-IDF arms — RAM safety on wide sparse fits

# Smoke-test knob: set to e.g. 3 to only run the first N (cheap) folds before
# committing to the full 14-fold run. None = run all folds.
DEBUG_MAX_FOLDS = None

# Fixed (non-tuned) hyperparameters — copied verbatim from analysis/BASELINE.py
LR_PARAMS = dict(
    penalty      = "l2",
    C            = 0.3,
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)


def build_model(model_name, y_tr, n_jobs):
    """Fixed hyperparameters (BASELINE.py values) for every arm — only n_jobs varies by arm."""
    if model_name == "Logistic":
        return LogisticRegression(**LR_PARAMS)
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, class_weight="balanced",
            n_jobs=n_jobs, random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
            eval_metric="auc", use_label_encoder=False, verbosity=0,
            n_jobs=n_jobs, random_state=RANDOM_STATE,
        )
    raise ValueError(model_name)


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
# 1. LOAD STRUCTURED DATA
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

# ============================================================
# 1b. RESTORE ORDINAL RISK GRADES (identical fix to BASELINE.py)
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
# 2. STRUCTURED FEATURE LIST (identical definition to BASELINE.py's STRUCT_COLS)
# ============================================================
EXCLUDE_COLS = {TARGET_COL, DATE_COL,
                "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "zip3", "emp_title", "title", "desc", "funded_ratio",
                "text_all_clean", "desc_clean", "title_clean", "emp_title_clean",
                "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage — see BASELINE.py changelog
STRUCT_COLS = [c for c in df.columns
               if c not in EXCLUDE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                   np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features: {len(STRUCT_COLS)}")

# ============================================================
# 3. DELONG TEST (copied verbatim from FinBERT/finbert_structured_ablation.py)
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
# 4. BOOTSTRAP CI (copied verbatim from BASELINE.py)
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
# 5. WALK-FORWARD FOLDS (identical logic to BASELINE.py / tfidf_pipeline.py)
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
            "tr_idx": df.index[tr_mask].tolist(),
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

folds_to_run = folds[:DEBUG_MAX_FOLDS] if DEBUG_MAX_FOLDS else folds
if DEBUG_MAX_FOLDS:
    print(f"\n  DEBUG_MAX_FOLDS={DEBUG_MAX_FOLDS} — only running the first {len(folds_to_run)} fold(s) this run.")

# ============================================================
# 6. MAIN WALK-FORWARD EVALUATION
# ============================================================
print("\n" + "="*70)
print("MAIN WALK-FORWARD EVALUATION")
print("="*70)

ARM_PAIRS = [("Structured", "Structured+TFIDF_full"),
             ("Structured", "Structured+TFIDF_chi2"),
             ("Structured+TFIDF_full", "Structured+TFIDF_chi2")]

pred_rows, delong_rows, importance_rows = [], [], []

for f in folds_to_run:
    t_fold0 = time.time()
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr, df_te = df.loc[tr], df.loc[te]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    # --- Load this fold's pre-computed TF-IDF matrices (from tfidf_pipeline.py) ---
    fold_dir = TFIDF_DIR / f"fold{f['fold']:02d}"
    X_tf_tr = sp.load_npz(fold_dir / "train_tfidf.npz").tocsr()
    X_tf_te = sp.load_npz(fold_dir / "test_tfidf.npz").tocsr()
    vocab = pd.read_csv(fold_dir / "vocab.txt", header=None)[0].astype(str).tolist()

    train_ids = pd.read_csv(fold_dir / "train_ids.csv")[ID_COL].values
    test_ids  = pd.read_csv(fold_dir / "test_ids.csv")[ID_COL].values
    assert np.array_equal(df_tr[ID_COL].values, train_ids), \
        f"Fold {f['fold']}: train id order mismatch between BASELINE-style folds and TF-IDF/output — rebuild TF-IDF output or check PATH_CSV."
    assert np.array_equal(df_te[ID_COL].values, test_ids), \
        f"Fold {f['fold']}: test id order mismatch between BASELINE-style folds and TF-IDF/output — rebuild TF-IDF output or check PATH_CSV."
    assert X_tf_tr.shape[1] == len(vocab) == X_tf_te.shape[1], f"Fold {f['fold']}: vocab/matrix width mismatch."

    # --- Structured block (fit imputer/scaler on train only) ---
    # keep_empty_features=True: some structured columns are entirely NaN within an early
    # fold's (small) training slice — without this, SimpleImputer silently drops them,
    # desyncing the output column count from len(STRUCT_COLS)/`names` used for importance.
    imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
    X_s_tr = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_s_te = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))
    scaler = StandardScaler()
    X_s_tr = scaler.fit_transform(X_s_tr)
    X_s_te = scaler.transform(X_s_te)
    X_s_tr_sp = sp.csr_matrix(X_s_tr)
    X_s_te_sp = sp.csr_matrix(X_s_te)

    # --- chi2 selection on the TF-IDF block only (train fit, applied to test) ---
    k_eff = min(CHI2_TOP_K, X_tf_tr.shape[1])
    selector = SelectKBest(chi2, k=k_eff)
    selector.fit(X_tf_tr, y_tr)
    chi2_idx = selector.get_support(indices=True)
    selected_vocab = [vocab[i] for i in chi2_idx]
    chi2_scores = dict(zip(selected_vocab, selector.scores_[chi2_idx]))
    X_tf_tr_chi2 = selector.transform(X_tf_tr)
    X_tf_te_chi2 = selector.transform(X_tf_te)
    print(f"  Fold {f['fold']}: TF-IDF vocab={len(vocab):,}  chi2-selected={k_eff:,}")

    # "txt:" prefix avoids the name collision bug found 2026-07-16 (a structured column
    # named e.g. "term" previously collided with a TF-IDF token also spelled "term").
    tfidf_full_names = [f"txt:{w}" for w in vocab]
    tfidf_chi2_names = [f"txt:{w}" for w in selected_vocab]

    arm_matrices = {
        "Structured": (X_s_tr, X_s_te, STRUCT_COLS, ["structured"] * len(STRUCT_COLS)),
        "Structured+TFIDF_full": (
            sp.hstack([X_s_tr_sp, X_tf_tr], format="csr"),
            sp.hstack([X_s_te_sp, X_tf_te], format="csr"),
            STRUCT_COLS + tfidf_full_names,
            ["structured"] * len(STRUCT_COLS) + ["tfidf"] * len(tfidf_full_names),
        ),
        "Structured+TFIDF_chi2": (
            sp.hstack([X_s_tr_sp, X_tf_tr_chi2], format="csr"),
            sp.hstack([X_s_te_sp, X_tf_te_chi2], format="csr"),
            STRUCT_COLS + tfidf_chi2_names,
            ["structured"] * len(STRUCT_COLS) + ["tfidf"] * len(tfidf_chi2_names),
        ),
    }

    fold_preds = {m: {} for m in MODEL_NAMES}

    for arm in ARMS:
        X_tr_arm, X_te_arm, names, kinds = arm_matrices[arm]
        n_jobs = N_JOBS_SMALL if arm == "Structured" else N_JOBS_WIDE

        for model_name in MODEL_NAMES:
            t0 = time.time()
            model = build_model(model_name, y_tr, n_jobs)
            model.fit(X_tr_arm, y_tr)
            p = model.predict_proba(X_te_arm)[:, 1]
            elapsed = time.time() - t0
            print(f"    [{model_name:<12s}] arm={arm:<24s} fold={f['fold']:>2}  "
                  f"n_features={len(names):>6,}  fit+predict={elapsed:6.1f}s")

            fold_preds[model_name][arm] = p

            # --- Feature importance ---
            imp = extract_importance(model_name, model, len(names))
            imp_sum = imp.sum()
            imp_share = imp / imp_sum if imp_sum > 0 else imp
            rank_of = np.empty(len(names), dtype=int)
            rank_of[np.argsort(imp)[::-1]] = np.arange(1, len(names) + 1)

            if arm == "Structured+TFIDF_full":
                keep_positions = np.argsort(imp)[::-1][:50]   # top 50 only — see docstring
            else:
                keep_positions = np.arange(len(names))         # Structured / chi2: keep all

            for pos in keep_positions:
                feat_name = names[pos]
                bare = feat_name[4:] if feat_name.startswith("txt:") else feat_name
                importance_rows.append({
                    "fold": f["fold"], "Arm": arm, "Model": model_name,
                    "Feature": feat_name, "Kind": kinds[pos],
                    "Importance_raw": round(float(imp[pos]), 6),
                    "Importance_share": round(float(imp_share[pos]), 6),
                    "Rank": int(rank_of[pos]),
                    "Chi2_Score": round(float(chi2_scores[bare]), 3)
                                  if (arm == "Structured+TFIDF_chi2" and kinds[pos] == "tfidf") else np.nan,
                })

            # --- Predictive metrics ---
            auc   = roc_auc_score(y_te, p)
            prauc = average_precision_score(y_te, p)
            brier = brier_score_loss(y_te, p)
            auc_lo,   auc_hi   = bootstrap_ci(y_te, p, roc_auc_score)
            prauc_lo, prauc_hi = bootstrap_ci(y_te, p, average_precision_score)
            brier_lo, brier_hi = bootstrap_ci(y_te, p, brier_score_loss)

            pred_rows.append({
                "fold": f["fold"], "train_cutoff": f["train_cutoff"], "test_end": f["test_end"],
                "n_train": f["n_train"], "n_test": f["n_test"], "n_defaults": f["n_test_defaults"],
                "Model": model_name, "Arm": arm, "n_features": len(names),
                "AUC": round(auc, 4), "AUC_CI_lo": round(auc_lo, 4) if not np.isnan(auc_lo) else np.nan,
                "AUC_CI_hi": round(auc_hi, 4) if not np.isnan(auc_hi) else np.nan,
                "PR_AUC": round(prauc, 4), "PR_AUC_CI_lo": round(prauc_lo, 4) if not np.isnan(prauc_lo) else np.nan,
                "PR_AUC_CI_hi": round(prauc_hi, 4) if not np.isnan(prauc_hi) else np.nan,
                "Brier": round(brier, 4), "Brier_CI_lo": round(brier_lo, 4) if not np.isnan(brier_lo) else np.nan,
                "Brier_CI_hi": round(brier_hi, 4) if not np.isnan(brier_hi) else np.nan,
                "GINI": round(2*auc - 1, 4),
            })

            del model
        del X_tr_arm, X_te_arm
        gc.collect()

    # --- DeLong: all 3 pairwise arm comparisons, per model ---
    y_te_full = df.loc[te, TARGET_COL].values.astype(int)
    for model_name in MODEL_NAMES:
        for arm_a, arm_b in ARM_PAIRS:
            auc_a, auc_b, z, pval, ci_lo, ci_hi = delong_auc_test(
                y_te_full, fold_preds[model_name][arm_a], fold_preds[model_name][arm_b])
            delong_rows.append({
                "fold": f["fold"], "Model": model_name,
                "Arm_A": arm_a, "Arm_B": arm_b,
                "AUC_A": round(auc_a, 4), "AUC_B": round(auc_b, 4),
                "Delta_AUC": round(auc_b - auc_a, 4),
                "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
                "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
                "CI_95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                "CI_95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
            })

    del X_tf_tr, X_tf_te, X_tf_tr_chi2, X_tf_te_chi2, arm_matrices, fold_preds
    gc.collect()
    print(f"  Fold {f['fold']} done in {time.time()-t_fold0:.1f}s.")

# ============================================================
# 7. AGGREGATE RESULTS
# ============================================================
df_pred   = pd.DataFrame(pred_rows)
df_delong = pd.DataFrame(delong_rows)
df_imp    = pd.DataFrame(importance_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]
pred_summary = df_pred.groupby(["Arm", "Model"])[PRED_METRICS].agg(["mean", "std"]).round(4)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean +/- std across folds)")
print("="*70)
print(pred_summary.to_string())

print("\n" + "="*70)
print("B. DELONG PAIRWISE TESTS — mean across folds")
print("="*70)
delong_mean = df_delong.groupby(["Model", "Arm_A", "Arm_B"])[["Delta_AUC", "p_value"]].mean().round(4)
print(delong_mean.to_string())

# --- Cross-fold feature consistency for the chi2 arm (the interpretability deliverable) ---
chi2_imp = df_imp[(df_imp["Arm"] == "Structured+TFIDF_chi2") & (df_imp["Kind"] == "tfidf")]
consistency_parts = []
for model_name in MODEL_NAMES:
    sub = chi2_imp[chi2_imp["Model"] == model_name]
    grp = sub.groupby("Feature").agg(
        n_folds_selected=("fold", "nunique"),
        mean_rank_when_selected=("Rank", "mean"),
        mean_importance_share=("Importance_share", "mean"),
        mean_chi2_score=("Chi2_Score", "mean"),
    ).reset_index()
    grp["Model"] = model_name
    grp["pct_folds_selected"] = (grp["n_folds_selected"] / len(folds_to_run) * 100).round(1)
    grp["mean_rank_when_selected"] = grp["mean_rank_when_selected"].round(2)
    grp["mean_importance_share"] = grp["mean_importance_share"].round(4)
    grp["mean_chi2_score"] = grp["mean_chi2_score"].round(2)
    consistency_parts.append(grp)
df_consistency = pd.concat(consistency_parts, ignore_index=True)
df_consistency = df_consistency.sort_values(
    ["Model", "n_folds_selected", "mean_rank_when_selected"], ascending=[True, False, True]
).reset_index(drop=True)

print("\n" + "="*70)
print("C. TF-IDF WORD CONSISTENCY (chi2 arm) — top 15 per model")
print("="*70)
for model_name in MODEL_NAMES:
    sub = df_consistency[df_consistency["Model"] == model_name]
    print(f"\n  [{model_name}]  {len(sub)} distinct words ever selected across {len(folds_to_run)} folds")
    print(sub.head(15).to_string(index=False))

# ============================================================
# 8. SAVE CSVs
# ============================================================
out_dir = SCRIPT_DIR / "results"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(out_dir / "ablation_folds.csv", index=False)
pred_summary.to_csv(out_dir / "ablation_summary.csv")
df_delong.to_csv(out_dir / "ablation_delong.csv", index=False)
df_imp.to_csv(out_dir / "ablation_feature_importance.csv", index=False)
df_consistency.to_csv(out_dir / "ablation_feature_consistency.csv", index=False)
print(f"\nSaved CSVs to: {out_dir}")

# ============================================================
# 9. FIGURES
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
_ARM_STYLE = {
    "Structured": dict(linestyle="-", marker="o", alpha=1.00),
    "Structured+TFIDF_full": dict(linestyle="--", marker="s", alpha=0.75),
    "Structured+TFIDF_chi2": dict(linestyle=":", marker="^", alpha=0.75),
}

print("\nGenerating figures...")

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    color = _CLR[model]
    for arm in ARMS:
        style = _ARM_STYLE[arm]
        sub = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == arm)].sort_values("fold")
        ax.plot(sub["fold"], sub["AUC"], color=color, linewidth=2, markersize=5, label=arm, **style)
        if not sub["AUC_CI_lo"].isna().all():
            ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"], alpha=0.10, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("AUC")
    ax.legend(title="Arm", fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.suptitle("Walk-Forward AUC Stability — Structured vs Structured+TF-IDF (full / chi2 top-%d)\n"
             "(shaded band = 95%% bootstrap CI)" % CHI2_TOP_K, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_by_fold.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig1_auc_by_fold.png")

fig, axes = plt.subplots(len(MODEL_NAMES), len(ARM_PAIRS), figsize=(6*len(ARM_PAIRS), 4.2*len(MODEL_NAMES)), sharey=True)
for i, model in enumerate(MODEL_NAMES):
    for j, (arm_a, arm_b) in enumerate(ARM_PAIRS):
        ax = axes[i, j] if len(MODEL_NAMES) > 1 else axes[j]
        sub = df_delong[(df_delong["Model"] == model) & (df_delong["Arm_A"] == arm_a) & (df_delong["Arm_B"] == arm_b)].sort_values("fold")
        pvals = sub["p_value"].fillna(1.0).values
        bar_colors = ["#d73027" if p < 0.05 else "#bababa" for p in pvals]
        x = sub["fold"].values; y = sub["Delta_AUC"].values
        ax.bar(x, y, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
        ci_lo, ci_hi = sub["CI_95_lo"].values, sub["CI_95_hi"].values
        valid = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
        if valid.any():
            half_width = (ci_hi - ci_lo) / 2
            ax.errorbar(x[valid], y[valid], yerr=half_width[valid], fmt="none", color="black", capsize=2, linewidth=1, zorder=4)
        ax.axhline(0, color="black", linewidth=1.0, zorder=5)
        ax.set_title(f"{model}\n{arm_b} − {arm_a}", fontsize=9)
        if i == len(MODEL_NAMES) - 1:
            ax.set_xlabel("Fold")
        if j == 0:
            ax.set_ylabel("ΔAUC")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
sig_patch = mpatches.Patch(color="#d73027", alpha=0.85, label="p < 0.05")
ns_patch  = mpatches.Patch(color="#bababa", alpha=0.85, label="p >= 0.05")
fig.legend(handles=[sig_patch, ns_patch], title="DeLong test", loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("DeLong Test: ΔAUC per Fold, per Arm-Pair", fontsize=13, fontweight="bold", y=1.05)
fig.tight_layout()
fig.savefig(figures_dir / "fig2_delong_deltas.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig2_delong_deltas.png")
print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 10. WRITTEN REPORT (methodology + results + auto-derived conclusions)
# ============================================================
def _best_arm_per_model(df_pred):
    lines = []
    for model in MODEL_NAMES:
        sub = df_pred[df_pred["Model"] == model].groupby("Arm")["AUC"].mean().sort_values(ascending=False)
        best_arm, best_auc = sub.index[0], sub.iloc[0]
        lines.append(f"- **{model}**: best mean AUC = {best_auc:.4f} ({best_arm})")
    return "\n".join(lines)

def _sig_summary(df_delong):
    lines = []
    for model in MODEL_NAMES:
        for arm_a, arm_b in ARM_PAIRS:
            sub = df_delong[(df_delong["Model"] == model) & (df_delong["Arm_A"] == arm_a) & (df_delong["Arm_B"] == arm_b)]
            n_sig = int((sub["p_value"] < 0.05).sum())
            mean_delta = sub["Delta_AUC"].mean()
            lines.append(f"- **{model}**, {arm_b} vs {arm_a}: mean ΔAUC = {mean_delta:+.4f}, "
                          f"significant (DeLong p<0.05) in {n_sig}/{len(sub)} folds")
    return "\n".join(lines)

def _top_words(df_consistency, n=20):
    lines = []
    for model in MODEL_NAMES:
        sub = df_consistency[df_consistency["Model"] == model].head(n)
        words = ", ".join(f"`{w}`" for w in sub["Feature"].str.replace("txt:", "", regex=False))
        lines.append(f"**{model}**: {words}")
    return "\n\n".join(lines)

n_folds_run = len(folds_to_run)
full_run_note = "" if n_folds_run == 14 else (
    f"\n> **Note:** this run used `DEBUG_MAX_FOLDS={DEBUG_MAX_FOLDS}` — only the first "
    f"{n_folds_run} of 14 folds were evaluated. Re-run with `DEBUG_MAX_FOLDS=None` for the full result.\n")

report = f"""# TF-IDF Structured Ablation — Results Report

Generated by `tfidf_structured_ablation.py`. Compares three arms across {n_folds_run} walk-forward
fold(s), {len(MODEL_NAMES)} algorithms: **Structured** (baseline, {len(STRUCT_COLS)} features) vs
**Structured+TFIDF_full** (baseline + the fold's complete TF-IDF vocabulary, uncapped) vs
**Structured+TFIDF_chi2** (baseline + top-{CHI2_TOP_K} TF-IDF columns by chi2 score, fit on train only).
{full_run_note}
## Methodology

- Walk-forward, expanding window: `MIN_TRAIN_MONTHS={MIN_TRAIN_MONTHS}`, `STEP_MONTHS={STEP_MONTHS}`,
  `MIN_TEST_DEFAULTS={MIN_TEST_DEFAULTS}` — identical protocol to `analysis/BASELINE.py` and
  `TF-IDF/tfidf_pipeline.py`, verified to yield exactly 14 folds.
- TF-IDF matrices are **not** recomputed here — loaded directly from `TF-IDF/output/fold{{NN}}/`
  (written by `tfidf_pipeline.py`: `ngram_range=(1,2)`, `min_df=5`, `sublinear_tf=True`, uncapped).
- Reduction method: **chi²** (`SelectKBest`), applied to the TF-IDF block only (structured columns
  are never dropped). Chosen over mutual-info (too slow at this scale — mutual_info_classif's
  kNN-based estimator does not scale to tens of thousands of sparse columns) and over SVD/PCA
  (would replace literal words/bigrams with opaque linear combinations, defeating the interpretability
  goal of using TF-IDF in the first place).
- **Hyperparameters are fixed, not tuned per arm** — reused verbatim from `analysis/BASELINE.py`:
  `Logistic(C=0.3, penalty=l2, solver=saga, class_weight=balanced)`,
  `RandomForest(n_estimators=500, min_samples_leaf=20, class_weight=balanced)`,
  `XGBoost(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, scale_pos_weight=neg/pos per fold)`.
  All three share `random_state={RANDOM_STATE}`. Decided explicitly to avoid multiplying an
  already RAM/time-risky RandomForest fit on wide sparse data by a full regularization sweep.
- Bootstrap 95% CIs: {N_BOOTSTRAP} resamples, seed={BOOT_SEED}. DeLong (1988) pairwise test for
  correlated AUCs on the same test set, run for all 3 arm-pairs, per model, per fold.
- `Structured` arm's `n_jobs=-1`; the two TF-IDF arms use `n_jobs={N_JOBS_WIDE}` (RAM safety on
  wide sparse RandomForest/XGBoost fits on this machine).

## A. Predictive performance — best arm per model (mean AUC across folds)

{_best_arm_per_model(df_pred)}

Full per-arm/per-model mean±std table: `results/ablation_summary.csv`.

## B. DeLong pairwise significance (mean ΔAUC, fraction of folds significant)

{_sig_summary(df_delong)}

Full per-fold DeLong results: `results/ablation_delong.csv`.

## C. Top contributing TF-IDF words (chi2 arm, most consistent across folds)

{_top_words(df_consistency)}

Full cross-fold consistency table (word, # folds selected, mean rank, mean importance share,
mean chi2 score): `results/ablation_feature_consistency.csv`. These are the words most defensible
as candidates to carry into a final, non-walk-forward model — selected repeatedly across
independently-trained folds, not a one-off snapshot.

## Conclusions

This section is generated from the run's own numbers above (section A/B) — re-run the script to
refresh it if the underlying data or parameters change.

"""

for model in MODEL_NAMES:
    best_row = df_pred[df_pred["Model"] == model].groupby("Arm")["AUC"].mean().sort_values(ascending=False)
    best_arm = best_row.index[0]
    struct_auc = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == "Structured")]["AUC"].mean()
    full_auc   = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == "Structured+TFIDF_full")]["AUC"].mean()
    chi2_auc   = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == "Structured+TFIDF_chi2")]["AUC"].mean()

    sub_sf = df_delong[(df_delong["Model"] == model) & (df_delong["Arm_A"] == "Structured") & (df_delong["Arm_B"] == "Structured+TFIDF_full")]
    sub_sc = df_delong[(df_delong["Model"] == model) & (df_delong["Arm_A"] == "Structured") & (df_delong["Arm_B"] == "Structured+TFIDF_chi2")]
    sub_fc = df_delong[(df_delong["Model"] == model) & (df_delong["Arm_A"] == "Structured+TFIDF_full") & (df_delong["Arm_B"] == "Structured+TFIDF_chi2")]
    n_sig_sf = int((sub_sf["p_value"] < 0.05).sum())
    n_sig_sc = int((sub_sc["p_value"] < 0.05).sum())
    n_sig_fc = int((sub_fc["p_value"] < 0.05).sum())

    tfidf_helps = "improves" if full_auc > struct_auc else "does not improve"
    chi2_vs_full = ("recovers essentially all" if abs(chi2_auc - full_auc) < 0.001
                     else ("recovers most of" if chi2_auc >= full_auc - 0.003 else "loses some of"))

    report += (
        f"- **{model}**: TF-IDF (full) {tfidf_helps} on Structured (ΔAUC={full_auc-struct_auc:+.4f}, "
        f"significant in {n_sig_sf}/{n_folds_run} folds). Reducing to the top-{CHI2_TOP_K} chi2-selected "
        f"words {chi2_vs_full} the full arm's benefit (chi2 vs Structured ΔAUC={chi2_auc-struct_auc:+.4f}, "
        f"significant in {n_sig_sc}/{n_folds_run} folds; full vs chi2 ΔAUC={full_auc-chi2_auc:+.4f}, "
        f"significant in {n_sig_fc}/{n_folds_run} folds) while using {CHI2_TOP_K} interpretable word/bigram "
        f"features instead of the full uncapped vocabulary. Best-performing arm overall: {best_arm}.\n"
    )

report += (
    f"\n**Candidate features for a final model**: the words listed in section C above — those selected "
    f"by chi2 in most/all evaluated folds — are the most defensible TF-IDF features to carry forward, "
    f"since they were re-derived independently on each fold's training data rather than chosen once on "
    f"the full dataset.\n"
)

report_path = out_dir / "ablation_report.md"
report_path.write_text(report, encoding="utf-8")
print(f"\nWritten report: {report_path}")

print(f"\nDone. Results in {out_dir}, figures in {figures_dir}")
