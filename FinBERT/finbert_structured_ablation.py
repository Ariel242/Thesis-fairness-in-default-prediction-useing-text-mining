"""
Structured vs Structured+FinBERT — Walk-Forward Ablation
==========================================================
Isolates the marginal contribution of FinBERT's deep text representations to
default prediction, against a pure no-text baseline. This is a companion to
analysis/preliminary_results_v2.py ("v2") — v2 is NOT imported, edited, or
executed by this script. Everything needed (walk-forward folds, DeLong test,
bootstrap CI, fairness metrics) is reimplemented here so this script is fully
self-contained inside FinBERT/.

WHY A SEPARATE SCRIPT (NOT A CHANGE TO v2)
    v2 already answers "does TF-IDF text help?" (Structured vs
    Structured+Text). This script answers a different question first: "do
    FinBERT's deep embeddings help, on their own, over the same no-text
    baseline?" Keeping it separate means v2's existing (already-reviewed)
    results stay untouched, and the FinBERT effect can be read in isolation
    before any decision is made about combining it with TF-IDF.

-----------------------------------------------------------------------------
1. THE TWO VARIANTS
-----------------------------------------------------------------------------
"Structured"
    Identical in definition to v2's "Structured" baseline: every numeric,
    non-leaky, non-identifier column from the post-03_advanced_prep CSV
    (same EXCLUDE_COLS as v2, including the grade/sub_grade ordinal fix and
    the last_fico_range_high/low temporal-leakage exclusion — see v2's
    changelog for why). No text signal of any kind. Reusing v2's exact
    column definition means a difference in AUC between this script's
    "Structured" arm and its "Structured+FinBERT" arm is attributable only
    to the embeddings, not to a differently-defined baseline.

"Structured+FinBERT"
    "Structured" (identical block, identically scaled) PLUS a FinBERT block:
      - has_desc         : 1 if the borrower wrote a (non-boilerplate)
                            description, else 0.
      - pca_000..pca_0{K} : the description's FinBERT "semantic" (mean-
                            pooled) embedding, compressed from 768 to
                            N_PCA_COMPONENTS dimensions by PCA (see below).
    No TF-IDF, no engineered text-length statistics — the FinBERT block is
    the only text signal added, so this is a clean embeddings-only ablation.

-----------------------------------------------------------------------------
2. HOW THE DEEP REPRESENTATION IS BUILT AND INTEGRATED
-----------------------------------------------------------------------------
Source: data/04_finbert/finbert_desc_embeddings.parquet, produced by
analysis/finbert_cls_embeddings.py (FinBERT: yiyanghkust/finbert-pretrain).
That file has ONE row per loan application (keyed by `id`) with two 768-dim
representations per description:
  - cls_*  : final-layer hidden state of the [CLS] token.
  - sem_*  : attention-mask-weighted mean of all final-layer token states
             ("semantic"/mean-pooled — the standard sentence-embedding
             pooling, and generally the stronger general-purpose
             representation; see that script's own docstring).

This script uses ONLY sem_* to start. Doubling to sem_+cls_ (1536 dims)
before even knowing whether 768 dims of signal are useful adds cost with no
established benefit here; cls_* is already sitting in the same parquet and
is a one-line swap for a follow-up ablation if sem_* underperforms.

Merge discipline: sem_*/has_desc are merged onto the modeling dataframe by
`id` (never row position — the parquet is not sorted the way the modeling
frame is), and this merge happens AFTER STRUCT_COLS_BASE is computed from
the CSV alone, so the embedding/has_desc columns can never accidentally leak
into the "Structured" feature list.

Dimensionality reduction (PCA), and why it's needed here specifically:
768 continuous, highly collinear dimensions are a poor match for this
project's tree models: XGBoost is capped at max_depth=4 and RandomForest at
min_samples_leaf=20 (both deliberately shallow — see v2 for why), so neither
can spend many splits carving up a diffuse 768-dim continuous space the way
they can exploit a handful of informative structured columns or sparse
TF-IDF tokens. PCA compresses the embedding into N_PCA_COMPONENTS orthogonal
directions that concentrate the variance, giving all three models (not just
Logistic) a feature block they can actually use within their existing
hyperparameters.

LEAKAGE DISCIPLINE (the most important methodological point in this script):
PCA is fit on the TRAINING portion of each walk-forward fold ONLY, then
applied unchanged to that fold's test portion — exactly mirroring how v2
refits TfidfVectorizer inside every fold. A PCA fit on the full dataset
(train+test pooled) would let the principal directions of FUTURE loans'
embeddings shape the features used to predict on the past-relative-to-them
training data, which is precisely the kind of leakage the walk-forward
design exists to prevent elsewhere in this pipeline. The same fold-local
StandardScaler discipline v2 uses for its numeric blocks is applied here too
(fit on train, applied to test) for [has_desc, pca_000..pca_0{K}] as one
block, after PCA.

has_desc as an explicit feature: ~54% of loan applications have no
description at all, and their embedding is the exact zero vector (see
finbert_cls_embeddings.py). Without has_desc, the model has to infer "wrote
nothing" from "embedding happens to be near the origin" — which is not
guaranteed to be a natural signal in PCA-compressed space. has_desc makes
that a first-class binary feature instead of an inference.

-----------------------------------------------------------------------------
3. REGULARIZATION / MODEL HYPERPARAMETERS
-----------------------------------------------------------------------------
All three models' hyperparameters are IDENTICAL to v2, deliberately. This
holds model capacity fixed so any AUC difference is attributable to the
added representation, not to re-tuning:
  - Logistic  : L2, C=0.3 (fairly strong shrinkage — appropriate given the
                FinBERT block adds up to N_PCA_COMPONENTS+1 correlated new
                dimensions), solver="saga", class_weight="balanced".
  - XGBoost   : 300 trees, max_depth=4 (shallow — limits how much any one
                tree can overfit to individual PCA directions), learning
                rate 0.05, subsample 0.8, scale_pos_weight for imbalance.
  - RandomForest: 500 trees, min_samples_leaf=20 (bounds tree size/RAM),
                class_weight="balanced".
PCA itself is an additional, deliberate form of regularization on top of
each model's native one: compressing 768 dims to N_PCA_COMPONENTS removes
low-variance directions that are mostly noise relative to this sample size,
before any model-specific penalty is even applied.
RANDOM_STATE=242 throughout (models AND PCA's internal solver), matching the
single shared seed convention used across this whole project.

-----------------------------------------------------------------------------
4. WALK-FORWARD PROTOCOL (identical parameters to v2)
-----------------------------------------------------------------------------
Expanding-window, chronological, no shuffling. MIN_TRAIN_MONTHS=8 months
before the first test fold, STEP_MONTHS=3 (quarterly), each test window
dynamically expanded until it contains at least MIN_TEST_DEFAULTS=100
default events. All fold-local fitting (imputer, scaler, PCA) happens
strictly inside the fold loop, fit on that fold's training rows only.

-----------------------------------------------------------------------------
5. METRICS (identical definitions to v2)
-----------------------------------------------------------------------------
  - AUC, PR-AUC (average precision), Brier score, GINI = 2*AUC-1 — each with
    a 500-resample percentile bootstrap 95% CI (BOOT_SEED=42).
  - DeLong test (DeLong et al. 1988) per fold per model: "Structured" vs
    "Structured+FinBERT" AUC, paired on the same test set — tests whether
    the FinBERT block's AUC gain is distinguishable from noise, fold by
    fold (mirrors v2's Structured vs Structured+Text DeLong table).
  - Fairness: FNR gap / FPR gap / Brier gap across ZIP3 groups, at decision
    thresholds {0.5, 0.6, 0.7}, with the same group-inclusion filters as v2
    (MIN_GROUP_N=80 rows, MIN_GROUP_POS=10 positive events per group) — kept
    because fairness across groups is the thesis's core theme, not an
    afterthought specific to the text-vs-no-text comparison.
  - Feature importance (last fold, Structured+FinBERT): per-model top-20
    tables (LR: |standardized coefficient|; XGBoost: Gain; RandomForest:
    mean decrease in impurity), a FinBERT-only top-20 (which PCA components/
    has_desc matter most — not human-readable like TF-IDF tokens, but useful
    as a diagnostic), and an aggregate "importance mass" table showing what
    share of each model's total importance sits in the FinBERT block vs the
    structured block — the single clearest answer to "how much did the deep
    representation matter overall."

-----------------------------------------------------------------------------
6. WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------------------------------------------
  - Does not modify, import, or execute analysis/preliminary_results_v2.py.
  - Does not combine FinBERT with TF-IDF (no "Structured+Text+FinBERT"
    variant) — that combination is a deliberate next step, deferred until
    the results here are reviewed.
  - Writes nothing outside this FinBERT/ folder (results/, figures/ here).

-----------------------------------------------------------------------------
7. OUTPUTS (written under FinBERT/)
-----------------------------------------------------------------------------
results/
  fb_predictive_folds.csv        — per-fold predictive metrics + bootstrap CIs
  fb_fairness_folds.csv          — per-fold fairness metrics by threshold
  fb_delong.csv                  — DeLong test per fold
  fb_predictive_summary.csv      — predictive summary (mean +/- std)
  fb_delta_summary.csv           — FinBERT vs Structured deltas
  fb_pca_variance.csv            — explained variance captured by PCA, per fold
  fb_fairness_summary_thr{05,06,07}.csv
  fb_group_coverage.csv
  fb_zip_summary.csv
  fb_importance_mass.csv         — % importance in FinBERT block vs structured, per model
figures/
  fig1_auc_stability.png
  fig2_delong_delta_auc.png
  fig3_fairness_tradeoff.png
  fig4_threshold_sensitivity.png
"""

import os
import sys
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
# PARAMETERS — edit these before running
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent            # .../F-TM-CR/FinBERT
BASE_DIR   = SCRIPT_DIR.parent                           # .../F-TM-CR  (data lives here; never written to)

# Must match v2's PATH_CSV exactly — same dataset, so the "Structured"
# baseline computed here is directly comparable to v2's.
PATH_CSV      = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
PATH_FINBERT  = BASE_DIR / "data" / "04_finbert" / "finbert_desc_embeddings.parquet"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ZIP_COL    = "zip3"
ID_COL     = "id"

# Walk-Forward parameters (identical to v2)
MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

# Fairness filter (identical to v2)
MIN_GROUP_N   = 80
MIN_GROUP_POS = 10
THRESHOLDS    = [0.5, 0.6, 0.7]

# Bootstrap CI (identical to v2)
N_BOOTSTRAP = 500
BOOT_SEED   = 42

# FinBERT block
N_PCA_COMPONENTS = 50   # 768 sem_* dims compressed to this many; see docstring section 2

# Fixed random state — shared across models AND PCA's internal solver
RANDOM_STATE = 242

# Logistic Regression — identical spec to v2
LR_PARAMS = dict(
    penalty      = "l2",
    C            = 0.3,
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)

# Random Forest — identical spec to v2
RF_PARAMS = dict(
    n_estimators     = 500,
    min_samples_leaf = 20,
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)

def make_xgb(y_tr):
    # Identical spec to v2 (scale_pos_weight recomputed per fold, as in v2)
    return xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum(),
        eval_metric      = "auc",
        use_label_encoder= False,
        verbosity        = 0,
        n_jobs           = -1,
        random_state     = RANDOM_STATE,
    )

# ============================================================
# 1. LOAD STRUCTURED DATA  (CSV only — FinBERT merged in step 3)
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

if ZIP_COL in df.columns:
    df[ZIP_COL] = df[ZIP_COL].astype(object).astype(str).replace("nan", np.nan)
else:
    print(f"  WARNING: ZIP column '{ZIP_COL}' not found in CSV")

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
# 2. STRUCTURED FEATURE LIST  (identical definition to v2's STRUCT_COLS_BASE)
#    Computed BEFORE the FinBERT merge so has_desc/sem_* can never leak in.
# ============================================================
EXCLUDE_COLS = {TARGET_COL, DATE_COL, ZIP_COL,
                "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "emp_title", "title", "desc", "funded_ratio",
                "text_all_clean", "desc_clean", "title_clean", "emp_title_clean",
                "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage — see v2 changelog #3
STRUCT_COLS_BASE = [c for c in df.columns
                    if c not in EXCLUDE_COLS
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                        np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features (Structured baseline, identical to v2): {len(STRUCT_COLS_BASE)}")

# ============================================================
# 3. MERGE FINBERT EMBEDDINGS  (by id — never row position)
# ============================================================
print("Loading FinBERT embeddings...")
SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
for c in SEM_COLS:
    df_fb[c] = df_fb[c].astype(np.float32)   # parquet stores float16 — upcast for numerical stability

n_before = len(df)
df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == n_before, "Merge changed row count — id is not a clean 1:1 key"
assert df["has_desc"].isna().sum() == 0, "Unmatched rows after FinBERT merge — check id coverage"
print(f"  Merged embeddings for {len(df):,} rows "
      f"({df['has_desc'].mean():.1%} have a non-empty description)")

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
# 6. FAIRNESS METRICS
# ============================================================
def fairness_gaps(y_true, p, groups, threshold=0.5, min_n=MIN_GROUP_N, min_pos=MIN_GROUP_POS):
    y_true = np.asarray(y_true).astype(int)
    p_arr  = np.asarray(p)
    yhat   = (p_arr >= threshold).astype(int)
    dfg = pd.DataFrame({"y": y_true, "yhat": yhat, "p": p_arr, "g": np.asarray(groups)}).reset_index(drop=True)

    grp  = dfg.groupby("g").agg(n=("y","size"), pos=("y","sum"))
    coverage = {
        "n_candidate_groups":   int(grp.shape[0]),
        "n_pass_n_threshold":   int((grp["n"] >= min_n).sum()),
        "n_pass_pos_threshold": int((grp["pos"] >= min_pos).sum()),
    }
    keep = grp[(grp["n"] >= min_n) & (grp["pos"] >= min_pos)].index
    coverage["n_included"] = int(len(keep))

    dfg = dfg[dfg["g"].isin(keep)]
    if dfg.empty:
        return {"FNR_gap": np.nan, "FPR_gap": np.nan, "n_groups": 0, "coverage": coverage}

    def rates(sub):
        y, yh = sub["y"].values, sub["yhat"].values
        tp = ((yh==1)&(y==1)).sum(); fn = ((yh==0)&(y==1)).sum()
        fp = ((yh==1)&(y==0)).sum(); tn = ((yh==0)&(y==0)).sum()
        fnr = fn/(fn+tp) if (fn+tp)>0 else np.nan
        fpr = fp/(fp+tn) if (fp+tn)>0 else np.nan
        return pd.Series({"FNR": fnr, "FPR": fpr})

    per = dfg.groupby("g").apply(rates, include_groups=False).dropna()
    if per.empty or len(per) < 2:
        return {"FNR_gap": np.nan, "FPR_gap": np.nan, "n_groups": int(len(per)), "coverage": coverage}
    return {
        "FNR_gap":  float(per["FNR"].max() - per["FNR"].min()),
        "FPR_gap":  float(per["FPR"].max() - per["FPR"].min()),
        "n_groups": int(len(per)),
        "coverage": coverage,
    }

def brier_gap(y_true, p, groups, min_n=MIN_GROUP_N, min_pos=MIN_GROUP_POS):
    dfg = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p), "g": np.asarray(groups)}).reset_index(drop=True)
    dfg["sq"] = (dfg["p"] - dfg["y"])**2
    grp  = dfg.groupby("g").agg(n=("y","size"), pos=("y","sum"))
    keep = grp[(grp["n"] >= min_n) & (grp["pos"] >= min_pos)].index
    dfg  = dfg[dfg["g"].isin(keep)]
    if dfg.empty or dfg["g"].nunique() < 2:
        return np.nan
    per = dfg.groupby("g")["sq"].mean()
    return float(per.max() - per.min())

# ============================================================
# 7. WALK-FORWARD FOLDS (identical logic to v2)
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
print(f"  Total folds: {len(folds)}")
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  defaults_in_test={f['n_test_defaults']}")

# ============================================================
# 8. MAIN LOOP
# ============================================================
pred_rows, fair_rows, coverage_rows, delong_rows, zip_rows, pca_rows = [], [], [], [], [], []

for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr, df_te = df.loc[tr], df.loc[te]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)
    zip_te = df_te[ZIP_COL].values if ZIP_COL in df.columns else None

    # --- Structured block: shared, identical, by both variants ---
    imputer_struct = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer_struct.fit_transform(df_tr[STRUCT_COLS_BASE].values.astype(np.float32))
    X_te_s = imputer_struct.transform(df_te[STRUCT_COLS_BASE].values.astype(np.float32))
    scaler_struct = StandardScaler()
    X_tr_s = scaler_struct.fit_transform(X_tr_s)
    X_te_s = scaler_struct.transform(X_te_s)

    # --- FinBERT block: PCA fit on TRAIN fold only, then has_desc + PCA scaled together ---
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    pca_tr = pca.fit_transform(df_tr[SEM_COLS].values)
    pca_te = pca.transform(df_te[SEM_COLS].values)
    evr = float(pca.explained_variance_ratio_.sum())
    pca_rows.append({"fold": f["fold"], "n_components": N_PCA_COMPONENTS, "explained_variance_ratio": round(evr, 4)})

    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
    scaler_fb = StandardScaler()
    fb_tr = scaler_fb.fit_transform(fb_tr_raw)
    fb_te = scaler_fb.transform(fb_te_raw)

    X_tr_full = np.hstack([X_tr_s, fb_tr])
    X_te_full = np.hstack([X_te_s, fb_te])

    models = {
        "Logistic":     LogisticRegression(**LR_PARAMS),
        "XGBoost":      make_xgb(y_tr),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }

    fold_preds = {}
    for mname, model in models.items():
        m_s = model.__class__(**model.get_params()) if mname != "XGBoost" else make_xgb(y_tr)
        m_s.fit(X_tr_s, y_tr)
        p_s = m_s.predict_proba(X_te_s)[:, 1]

        m_f = model.__class__(**model.get_params()) if mname != "XGBoost" else make_xgb(y_tr)
        m_f.fit(X_tr_full, y_tr)
        p_f = m_f.predict_proba(X_te_full)[:, 1]

        fold_preds[mname] = {"struct": p_s, "full": p_f}

        for variant, p in [("Structured", p_s), ("Structured+FinBERT", p_f)]:
            auc   = roc_auc_score(y_te, p)
            prauc = average_precision_score(y_te, p)
            brier = brier_score_loss(y_te, p)
            auc_lo,   auc_hi   = bootstrap_ci(y_te, p, roc_auc_score)
            prauc_lo, prauc_hi = bootstrap_ci(y_te, p, average_precision_score)
            brier_lo, brier_hi = bootstrap_ci(y_te, p, brier_score_loss)

            pred_rows.append({
                "fold": f["fold"], "train_cutoff": f["train_cutoff"], "test_end": f["test_end"],
                "n_train": f["n_train"], "n_test": f["n_test"], "n_defaults": f["n_test_defaults"],
                "Model": mname, "Variant": variant,
                "AUC": round(auc,4), "AUC_CI_lo": round(auc_lo,4) if not np.isnan(auc_lo) else np.nan,
                "AUC_CI_hi": round(auc_hi,4) if not np.isnan(auc_hi) else np.nan,
                "PR_AUC": round(prauc,4), "PR_AUC_CI_lo": round(prauc_lo,4) if not np.isnan(prauc_lo) else np.nan,
                "PR_AUC_CI_hi": round(prauc_hi,4) if not np.isnan(prauc_hi) else np.nan,
                "Brier": round(brier,4), "Brier_CI_lo": round(brier_lo,4) if not np.isnan(brier_lo) else np.nan,
                "Brier_CI_hi": round(brier_hi,4) if not np.isnan(brier_hi) else np.nan,
                "GINI": round(2*auc-1, 4),
            })

            for thr in THRESHOLDS:
                fair_row = {"fold": f["fold"], "train_cutoff": f["train_cutoff"], "test_end": f["test_end"],
                            "Model": mname, "Variant": variant, "threshold": thr}
                if zip_te is not None:
                    fg = fairness_gaps(y_te, p, zip_te, threshold=thr)
                    bg = brier_gap(y_te, p, zip_te)
                    fair_row.update({
                        "FNR_gap": round(fg["FNR_gap"],4) if not np.isnan(fg["FNR_gap"]) else np.nan,
                        "FPR_gap": round(fg["FPR_gap"],4) if not np.isnan(fg["FPR_gap"]) else np.nan,
                        "Brier_gap": round(bg,4) if not np.isnan(bg) else np.nan,
                        "n_groups": fg["n_groups"],
                    })
                    cov = fg["coverage"]
                    coverage_rows.append({"fold": f["fold"], "Model": mname, "Variant": variant, "threshold": thr, **cov})
                fair_rows.append(fair_row)

                if zip_te is not None and thr == 0.5:
                    p_arr = np.asarray(p); y_arr = np.asarray(y_te).astype(int); g_arr = np.asarray(zip_te)
                    dfz = pd.DataFrame({"y": y_arr, "p": p_arr, "yhat": (p_arr>=thr).astype(int), "g": g_arr}).reset_index(drop=True)
                    for grp_name, grp_df in dfz.groupby("g"):
                        n = len(grp_df); pos = int(grp_df["y"].sum())
                        if n < MIN_GROUP_N or pos < MIN_GROUP_POS:
                            continue
                        y_g, yh_g, p_g = grp_df["y"].values, grp_df["yhat"].values, grp_df["p"].values
                        tp=((yh_g==1)&(y_g==1)).sum(); fn=((yh_g==0)&(y_g==1)).sum()
                        fp=((yh_g==1)&(y_g==0)).sum(); tn=((yh_g==0)&(y_g==0)).sum()
                        try:
                            auc_g = roc_auc_score(y_g, p_g) if 0 < pos < n else np.nan
                        except Exception:
                            auc_g = np.nan
                        zip_rows.append({
                            "zip3": grp_name, "fold": f["fold"], "Model": mname, "Variant": variant,
                            "n": n, "n_default": pos, "default_rate": round(pos/n,4),
                            "AUC": round(auc_g,4) if not np.isnan(auc_g) else np.nan,
                            "FNR": round(fn/(fn+tp),4) if (fn+tp)>0 else np.nan,
                            "FPR": round(fp/(fp+tn),4) if (fp+tn)>0 else np.nan,
                            "Brier": round(float(np.mean((p_g-y_g)**2)),4),
                        })

        auc_s, auc_f, z, pval, ci_lo, ci_hi = delong_auc_test(y_te, fold_preds[mname]["struct"], fold_preds[mname]["full"])
        delong_rows.append({
            "fold": f["fold"], "Model": mname,
            "AUC_Structured": round(auc_s,4), "AUC_Structured+FinBERT": round(auc_f,4),
            "Delta_AUC": round(auc_f-auc_s,4),
            "Z_stat": round(z,3) if not np.isnan(z) else np.nan,
            "p_value": round(pval,4) if not np.isnan(pval) else np.nan,
            "CI_95_lo": round(ci_lo,4) if not np.isnan(ci_lo) else np.nan,
            "CI_95_hi": round(ci_hi,4) if not np.isnan(ci_hi) else np.nan,
        })

    print(f"  Fold {f['fold']} done. (PCA explained variance: {evr:.1%})")

# ============================================================
# 9. AGGREGATE RESULTS
# ============================================================
df_pred     = pd.DataFrame(pred_rows)
df_fair     = pd.DataFrame(fair_rows)
df_coverage = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
df_delong   = pd.DataFrame(delong_rows)
df_pca      = pd.DataFrame(pca_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]
FAIR_METRICS = [c for c in ["FNR_gap", "FPR_gap", "Brier_gap"] if c in df_fair.columns]

pred_summary = df_pred.groupby(["Model","Variant"])[PRED_METRICS].agg(["mean","std"]).round(4)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean +/- std across folds)")
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
print("B. DELTA: Structured+FinBERT - Structured  (mean +/- std across folds)")
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

if FAIR_METRICS:
    print("\n" + "="*70)
    print("D. FAIRNESS SUMMARY BY THRESHOLD (mean +/- std across folds)")
    print("="*70)
    for thr in THRESHOLDS:
        sub = df_fair[df_fair["threshold"]==thr]
        if sub.empty: continue
        summary = sub.groupby(["Model","Variant"])[FAIR_METRICS].agg(["mean","std"]).round(4)
        print(f"\n  Threshold = {thr}")
        print(summary.to_string())

    print("\n" + "="*70)
    print("E. FAIRNESS DIRECTION: Adding FinBERT vs Structured-only")
    print("   (for gap metrics: negative delta = improved; positive = worsened)")
    print("="*70)
    for thr in THRESHOLDS:
        print(f"\n  Threshold = {thr}")
        for m in df_fair["Model"].unique():
            for c in FAIR_METRICS:
                base_vals = df_fair[(df_fair["Model"]==m)&(df_fair["Variant"]=="Structured")&(df_fair["threshold"]==thr)][c].dropna()
                full_vals = df_fair[(df_fair["Model"]==m)&(df_fair["Variant"]=="Structured+FinBERT")&(df_fair["threshold"]==thr)][c].dropna()
                if base_vals.empty or full_vals.empty: continue
                delta = full_vals.mean() - base_vals.mean()
                direction = "improved" if delta<-1e-5 else ("worsened" if delta>1e-5 else "no change")
                print(f"    [{m}] {c}: mean delta={delta:+.4f}  -> {direction}")

if not df_coverage.empty:
    cov_summary = (df_coverage.groupby(["Model","Variant","threshold"])
                   .agg(n_candidate_groups=("n_candidate_groups","mean"),
                        n_pass_n_threshold=("n_pass_n_threshold","mean"),
                        n_pass_pos_threshold=("n_pass_pos_threshold","mean"),
                        n_included=("n_included","mean")).round(1).reset_index())
    print("\n" + "="*70)
    print("F. GROUP COVERAGE SUMMARY (mean across folds)")
    print("="*70)
    print(cov_summary.to_string(index=False))

if zip_rows:
    df_zip = pd.DataFrame(zip_rows)
    df_zip_summary = (df_zip.groupby(["zip3","Model","Variant"])
                      .agg(n_folds=("fold","count"), n_total=("n","sum"), n_default=("n_default","sum"),
                           AUC_mean=("AUC","mean"), FNR_mean=("FNR","mean"), FPR_mean=("FPR","mean"),
                           Brier_mean=("Brier","mean")).reset_index().round(4))
    df_zip_summary["default_rate"] = (df_zip_summary["n_default"]/df_zip_summary["n_total"]).round(4)
    print("\n" + "="*70)
    print("G. ZIP3 SUMMARY (mean across folds, threshold=0.5, top 20 by n_total)")
    print("="*70)
    print(df_zip_summary.sort_values("n_total", ascending=False).head(20).to_string(index=False))
else:
    df_zip_summary = pd.DataFrame()

print("\n" + "="*70)
print("H. PCA EXPLAINED VARIANCE PER FOLD")
print("="*70)
print(df_pca.to_string(index=False))

# ============================================================
# 10. SAVE
# ============================================================
out_dir = SCRIPT_DIR / "results"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(out_dir / "fb_predictive_folds.csv", index=False)
df_fair.to_csv(out_dir / "fb_fairness_folds.csv", index=False)
df_delong.to_csv(out_dir / "fb_delong.csv", index=False)
pred_summary.to_csv(out_dir / "fb_predictive_summary.csv")
df_delta_summary.to_csv(out_dir / "fb_delta_summary.csv")
df_pca.to_csv(out_dir / "fb_pca_variance.csv", index=False)

if FAIR_METRICS:
    for thr in THRESHOLDS:
        sub = df_fair[df_fair["threshold"]==thr]
        if sub.empty: continue
        out_thr = sub.groupby(["Model","Variant"])[FAIR_METRICS].agg(["mean","std"]).round(4)
        thr_tag = str(thr).replace(".","")
        out_thr.to_csv(out_dir / f"fb_fairness_summary_thr{thr_tag}.csv")

if not df_coverage.empty:
    cov_summary.to_csv(out_dir / "fb_group_coverage.csv", index=False)
if not df_zip_summary.empty:
    df_zip_summary.to_csv(out_dir / "fb_zip_summary.csv", index=False)

print(f"\nSaved results to: {out_dir}")

# ============================================================
# 11. FIGURES
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

MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]
_CLR   = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}
_SHORT = {"Logistic": "LR", "XGBoost": "XGB", "RandomForest": "RF"}
_COMBO_COLORS = ["#2166ac", "#74add1", "#d6604d", "#f4a582", "#1a9850", "#a6d96a"]
_COMBO_LABELS = ["LR – Structured", "LR – Structured+FinBERT",
                  "XGB – Structured", "XGB – Structured+FinBERT",
                  "RF – Structured", "RF – Structured+FinBERT"]
_COMBOS = [("Logistic","Structured"), ("Logistic","Structured+FinBERT"),
           ("XGBoost","Structured"), ("XGBoost","Structured+FinBERT"),
           ("RandomForest","Structured"), ("RandomForest","Structured+FinBERT")]

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
fig.suptitle("Walk-Forward AUC Stability Across Folds\n(shaded band = 95% bootstrap CI)", fontsize=13, fontweight="bold")
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

fair_05 = df_fair[df_fair["threshold"]==0.5].copy()
scatter_rows = []
for model, variant in _COMBOS:
    p_sub = df_pred[(df_pred["Model"]==model)&(df_pred["Variant"]==variant)]
    f_sub = fair_05[(fair_05["Model"]==model)&(fair_05["Variant"]==variant)]
    if p_sub.empty or f_sub.empty: continue
    scatter_rows.append({"Model": model, "Variant": variant,
                          "AUC_mean": p_sub["AUC"].mean(), "AUC_std": p_sub["AUC"].std(),
                          "FNR_mean": f_sub["FNR_gap"].mean(), "FNR_std": f_sub["FNR_gap"].std()})
df_sc = pd.DataFrame(scatter_rows)
fig, ax = plt.subplots(figsize=(8,6))
for color, (_, row) in zip(_COMBO_COLORS, df_sc.iterrows()):
    marker = "o" if row["Variant"]=="Structured" else "s"
    ax.errorbar(row["AUC_mean"], row["FNR_mean"], xerr=row["AUC_std"], yerr=row["FNR_std"],
                fmt=marker, color=color, markersize=11, capsize=2, elinewidth=0.8, ecolor="#aaaaaa",
                linewidth=1.4, label=f"{row['Model']} – {row['Variant']}")
    short = _SHORT[row["Model"]] + ("-S" if row["Variant"]=="Structured" else "-S+FB")
    ax.annotate(short, (row["AUC_mean"], row["FNR_mean"]), textcoords="offset points", xytext=(7,4), fontsize=9)
ax.set_xlabel("Mean AUC (+/-1 SD across folds)", fontsize=11)
ax.set_ylabel("Mean FNR Gap (+/-1 SD across folds)", fontsize=11)
ax.set_title("Predictive Accuracy vs. Fairness Trade-off\n(threshold = 0.5; lower FNR Gap = more equitable)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(figures_dir / "fig3_fairness_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig3_fairness_tradeoff.png")

fair_agg = (df_fair.groupby(["Model","Variant","threshold"])[["FNR_gap","FPR_gap","Brier_gap"]]
            .agg(FNR_mean=("FNR_gap","mean"), FNR_std=("FNR_gap","std"),
                 FPR_mean=("FPR_gap","mean"), FPR_std=("FPR_gap","std"),
                 Brier_mean=("Brier_gap","mean"), Brier_std=("Brier_gap","std")).reset_index())
thresholds_ = sorted(fair_agg["threshold"].unique())
x = np.arange(len(thresholds_))
bar_width = 0.8/len(_COMBOS)
fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=False)
for ax, (metric_mean, metric_std, ylabel, title) in zip(
    axes, [("FNR_mean","FNR_std","FNR Gap","False Negative Rate Gap"),
           ("FPR_mean","FPR_std","FPR Gap","False Positive Rate Gap"),
           ("Brier_mean","Brier_std","Brier Gap","Brier Score Gap")]):
    for i, ((model, variant), color, label) in enumerate(zip(_COMBOS, _COMBO_COLORS, _COMBO_LABELS)):
        sub = fair_agg[(fair_agg["Model"]==model)&(fair_agg["Variant"]==variant)].sort_values("threshold")
        offset = (i-(len(_COMBOS)-1)/2)*bar_width
        ax.bar(x+offset, sub[metric_mean], bar_width, yerr=sub[metric_std],
               error_kw={"capsize":2,"elinewidth":0.8,"ecolor":"#aaaaaa"}, color=color, alpha=0.85, label=label)
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in thresholds_])
    ax.set_xlabel("Decision Threshold", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold"); ax.legend(fontsize=8)
fig.suptitle("Fairness Gaps by Decision Threshold (mean +/- SD across folds)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig4_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig4_threshold_sensitivity.png")
print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 12. FEATURE IMPORTANCE — LAST FOLD, STRUCTURED+FINBERT
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE — Last Fold, Structured+FinBERT")
print("="*70)

last_f = folds[-1]
_tr, _te = last_f["tr_idx"], last_f["te_idx"]
_df_tr, _df_te = df.loc[_tr], df.loc[_te]
_y_tr = _df_tr[TARGET_COL].values.astype(int)

_imputer = SimpleImputer(strategy="constant", fill_value=0)
_X_tr_s = _imputer.fit_transform(_df_tr[STRUCT_COLS_BASE].values.astype(np.float32))
_scaler = StandardScaler()
_X_tr_s = _scaler.fit_transform(_X_tr_s)

_pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
_pca_tr = _pca.fit_transform(_df_tr[SEM_COLS].values)
_fb_tr_raw = np.hstack([_df_tr[["has_desc"]].values.astype(np.float32), _pca_tr])
_scaler_fb = StandardScaler()
_fb_tr = _scaler_fb.fit_transform(_fb_tr_raw)

_X_tr_full = np.hstack([_X_tr_s, _fb_tr])

_all_names = STRUCT_COLS_BASE + ["fb:has_desc"] + [f"fb:pca_{i:03d}" for i in range(N_PCA_COMPONENTS)]
_all_kinds = ["structured"]*len(STRUCT_COLS_BASE) + ["FinBERT"]*(1+N_PCA_COMPONENTS)

_lr = LogisticRegression(**LR_PARAMS); _lr.fit(_X_tr_full, _y_tr)
_xgbm = make_xgb(_y_tr); _xgbm.fit(_X_tr_full, _y_tr)
_rf = RandomForestClassifier(**RF_PARAMS); _rf.fit(_X_tr_full, _y_tr)

_lr_imp = np.abs(_lr.coef_[0])
_df_lr_imp = (pd.DataFrame({"Feature": _all_names, "Importance": _lr_imp, "Kind": _all_kinds})
              .sort_values("Importance", ascending=False).reset_index(drop=True))
_df_lr_imp.index += 1

_xgb_gain = _xgbm.get_booster().get_score(importance_type="gain")
_df_xgb_imp = pd.DataFrame([
    {"Feature": _all_names[int(k.replace("f",""))], "Gain": round(v,2), "Kind": _all_kinds[int(k.replace("f",""))]}
    for k, v in _xgb_gain.items()
]).sort_values("Gain", ascending=False).reset_index(drop=True)
_df_xgb_imp.index += 1

_df_rf_imp = (pd.DataFrame({"Feature": _all_names, "Importance": np.round(_rf.feature_importances_,5), "Kind": _all_kinds})
              .sort_values("Importance", ascending=False).reset_index(drop=True))
_df_rf_imp.index += 1

TOP_N = 20
print("\n── Table 1: XGBoost — Top 20 Features (Structured+FinBERT) ──")
print(_df_xgb_imp.head(TOP_N).to_string())
print("\n── Table 2: Logistic Regression — Top 20 Features (Structured+FinBERT) ──")
print(_df_lr_imp.head(TOP_N).to_string())
print("\n── Table 3: RandomForest — Top 20 Features (Structured+FinBERT) ──")
print(_df_rf_imp.head(TOP_N).to_string())

print("\n── Table 4: XGBoost — Top 20 FinBERT-only Features ──")
print(_df_xgb_imp[_df_xgb_imp["Kind"]=="FinBERT"].drop(columns=["Kind"]).head(TOP_N).to_string())
print("\n── Table 5: RandomForest — Top 20 FinBERT-only Features ──")
print(_df_rf_imp[_df_rf_imp["Kind"]=="FinBERT"].drop(columns=["Kind"]).head(TOP_N).to_string())

# ── Importance mass: what share of each model's total importance is FinBERT? ──
mass_rows = []
for mname, dfi, col in [("Logistic", _df_lr_imp, "Importance"), ("XGBoost", _df_xgb_imp, "Gain"), ("RandomForest", _df_rf_imp, "Importance")]:
    total = dfi[col].sum()
    fb_mass = dfi[dfi["Kind"]=="FinBERT"][col].sum()
    mass_rows.append({"Model": mname, "FinBERT_share": round(fb_mass/total, 4) if total>0 else np.nan,
                       "Structured_share": round(1 - fb_mass/total, 4) if total>0 else np.nan})
df_mass = pd.DataFrame(mass_rows)
df_mass.to_csv(out_dir / "fb_importance_mass.csv", index=False)
print("\n── Table 6: Importance mass — FinBERT block vs Structured block (last fold) ──")
print(df_mass.to_string(index=False))

print(f"\nDone. Results in {out_dir}, figures in {figures_dir}")
