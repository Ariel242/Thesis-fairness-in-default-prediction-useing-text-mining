"""
Walk-Forward Cross-Validation for Default Prediction
=====================================================
Chronological expanding-window protocol.
TF-IDF, scaling, and imputation are fit ONLY on the training fold — no leakage.

PARAMETERS (edit below):

-----------------------------------------------------------------------------
v2 CHANGELOG (this file supersedes preliminary_results_cloude.py — the v1
script is kept unmodified in this folder as a historical record)
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
   Fix: these engineered stats are now kept in a separate `TEXT_STAT_COLS`
   list, excluded from the "Structured" variant (`STRUCT_COLS_BASE`), and
   included only in the "Structured+Text" variant (`FULL_STRUCT_COLS`,
   alongside TF-IDF) — so "Structured" now means what it says.

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

No other behavior was changed: same folds, same models, same fairness
metrics, same output file names (still written to results/walk_forward/).
-----------------------------------------------------------------------------
"""

import re
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS — edit these before running
# ============================================================
BASE_DIR        = Path(__file__).resolve().parent.parent
PATH_CSV        = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260108_1516.csv"
TARGET_COL      = "is_default"
TEXT_COL        = "text_all_clean"
RAW_TEXT_COLS   = ["desc", "title", "emp_title"]
DATE_COL        = "issue_month_start"
ZIP_COL         = "zip3"

# NOTE ON CENSORING:
# The CSV contains only loans that have reached their final status.
# is_default is fully observed for every row — no censoring bias.

# Walk-Forward parameters
MIN_TRAIN_MONTHS  = 8    # minimum months before first test fold
MIN_TEST_DEFAULTS = 100  # minimum default events per test window (dynamic)
STEP_MONTHS       = 3    # quarterly step (~10–13 folds expected)

# Fairness filter
MIN_GROUP_N   = 80
MIN_GROUP_POS = 10

# Decision thresholds for fairness evaluation
THRESHOLDS = [0.5, 0.6, 0.7]

# TF-IDF
TFIDF_MAX_FEATURES = 600
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams

# Bootstrap CI (set N_BOOTSTRAP=0 to skip — faster runs)
N_BOOTSTRAP = 500
BOOT_SEED   = 42

# Logistic Regression — explicit spec for documentation
LR_PARAMS = dict(
    penalty      = "l2",      # L2 regularization
    C            = 0.3,       # inverse strength: smaller = stronger regularization
    solver       = "saga",
    max_iter     = 1000,
    class_weight = "balanced",
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
                "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "emp_title", "title", "desc", "funded_ratio",
                "text_all_clean", "desc_clean", "title_clean", "emp_title_clean",
                "grade", "sub_grade",   # raw string cols — numeric encodings are grade_ord / sub_grade_ord
                "last_fico_range_high", "last_fico_range_low"}  # temporal leakage: FICO as of the LAST credit pull, not at origination (see v2 changelog #3)
STRUCT_COLS_BASE = [c for c in df.columns
                    if c not in EXCLUDE_COLS
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                        np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features (pre text-cleaning step): {len(STRUCT_COLS_BASE)}")

# ============================================================
# 2. TEXT CLEANING  (deterministic — done once, no leakage)
# ============================================================
custom_stop = {
    "added","borrower","loan","lending","club",
    "thanks","thank","quot","consideration","time",
    "just","like","want","make","need","br"
}
STOPWORDS = set(ENGLISH_STOP_WORDS).union(custom_stop)
PREFIX_RE = re.compile(r"^\s*borrower\s+added\s+on\s+\d{2}/\d{2}/\d{2}\s*>\s*", re.IGNORECASE)

_lemmatize_mode = "none"
_nlp = None
_wnl = None
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    _lemmatize_mode = "spacy"
    print("  Lemmatizer: spaCy")
except Exception:
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        _wnl = WordNetLemmatizer()
        _lemmatize_mode = "nltk"
        print("  Lemmatizer: NLTK")
    except Exception:
        print("  Lemmatizer: none (no spaCy or NLTK found)")

def _clean_desc(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = PREFIX_RE.sub("", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return ""
    tokens = s.split()
    if _lemmatize_mode == "spacy":
        tokens = [t.lemma_ for t in _nlp(" ".join(tokens))]
    elif _lemmatize_mode == "nltk":
        tokens = [_wnl.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    return " ".join(tokens)

def _clean_short(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return ""
    tokens = s.split()
    if _lemmatize_mode == "spacy":
        tokens = [t.lemma_ for t in _nlp(" ".join(tokens))]
    elif _lemmatize_mode == "nltk":
        tokens = [_wnl.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    return " ".join(tokens)

print("  Cleaning text columns...")
df["desc_clean"]      = df["desc"].apply(_clean_desc)
df["title_clean"]     = df["title"].apply(_clean_short) if "title" in df.columns else ""
df["emp_title_clean"] = df["emp_title"].apply(_clean_short) if "emp_title" in df.columns else ""

df["text_all_clean"] = (
    df["desc_clean"].fillna("") + " " +
    df["title_clean"].fillna("") + " " +
    df["emp_title_clean"].fillna("")
).str.strip()

CLEAN_COLS = ["desc_clean", "title_clean", "emp_title_clean"]
RAW_COLS   = ["desc", "title", "emp_title"]

def _wc(s):
    return s.str.split().str.len().fillna(0)

for raw_c, clean_c in zip(RAW_COLS, CLEAN_COLS):
    if raw_c not in df.columns:
        continue
    raw   = df[raw_c].fillna("").astype(str).str.strip()
    clean = df[clean_c].fillna("").astype(str).str.strip()
    raw_wc   = _wc(raw.str.lower())
    clean_wc = _wc(clean)
    df[f"{clean_c}_char_len"]      = clean.str.len()
    df[f"{clean_c}_uniq_words"]    = clean.apply(lambda x: len(set(x.split())) if x else 0)
    df[f"{clean_c}_avg_word_len"]  = clean.apply(
        lambda x: float(np.mean([len(w) for w in x.split()])) if x else 0.0)
    df[f"{clean_c}_removed_words"] = (raw_wc - clean_wc).clip(lower=0)
    df[f"{clean_c}_ttr"]           = np.where(
        clean_wc > 0, df[f"{clean_c}_uniq_words"] / clean_wc, 0.0)

num_feat_cols = [c for c in df.columns
                 if c.endswith(("_char_len","_uniq_words","_avg_word_len",
                                "_removed_words","_ttr"))]
print(f"  Numeric text features created: {len(num_feat_cols)}")

# ------------------------------------------------------------------
# v2 fix #1 — keep text-derived numeric stats OUT of the "Structured"
# baseline. In v1 these were appended straight into struct_cols, so the
# "Structured" (no-text) variant already carried a text signal (e.g.
# description length is a known predictor of default). That understated
# the marginal value of TF-IDF and biased the core Structured vs.
# Structured+Text comparison. Here they form their own list and are only
# added — together with TF-IDF — to the "Structured+Text" variant.
# ------------------------------------------------------------------
TEXT_STAT_COLS = [c for c in num_feat_cols if c not in STRUCT_COLS_BASE]
FULL_STRUCT_COLS = STRUCT_COLS_BASE + TEXT_STAT_COLS

print(f"  Text-derived numeric stats (kept OUT of 'Structured' baseline): {len(TEXT_STAT_COLS)}")
print(f"  'Structured' variant feature count:            {len(STRUCT_COLS_BASE)}")
print(f"  'Structured+Text' numeric feature count (pre-TF-IDF): {len(FULL_STRUCT_COLS)}")

# ============================================================
# 3. TF-IDF HELPER  (fit on train only — no leakage)
# ============================================================
def make_text_features(train_text, test_text):
    train_text = train_text.fillna("").astype(str)
    test_text  = test_text.fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = TFIDF_NGRAM_RANGE,
        sublinear_tf = True,
        min_df       = 5,
        strip_accents= "unicode",
        analyzer     = "word",
        token_pattern= r"[a-zA-Z]{3,}",
        stop_words   = "english",
        norm         = "l2",
    )
    X_tr = tfidf.fit_transform(train_text)
    X_te = tfidf.transform(test_text)
    return X_tr, X_te

# ============================================================
# 4. BOOTSTRAP CI
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
# 5. FAIRNESS METRICS
# ============================================================
def fairness_gaps(y_true, p, groups, threshold=0.5,
                  min_n=MIN_GROUP_N, min_pos=MIN_GROUP_POS):
    """Returns fairness gaps + group coverage stats."""
    y_true = np.asarray(y_true).astype(int)
    p_arr  = np.asarray(p)
    yhat   = (p_arr >= threshold).astype(int)
    dfg = pd.DataFrame({
        "y": y_true, "yhat": yhat, "p": p_arr,
        "g": np.asarray(groups),
    }).reset_index(drop=True)

    grp  = dfg.groupby("g").agg(n=("y","size"), pos=("y","sum"))
    n_candidates = int(grp.shape[0])
    n_pass_n     = int((grp["n"] >= min_n).sum())
    n_pass_pos   = int((grp["pos"] >= min_pos).sum())
    keep         = grp[(grp["n"] >= min_n) & (grp["pos"] >= min_pos)].index
    n_included   = int(len(keep))

    coverage = {
        "n_candidate_groups":  n_candidates,
        "n_pass_n_threshold":  n_pass_n,
        "n_pass_pos_threshold": n_pass_pos,
        "n_included":          n_included,
    }

    dfg = dfg[dfg["g"].isin(keep)]
    if dfg.empty:
        return {"FNR_gap": np.nan, "FPR_gap": np.nan, "n_groups": 0,
                "coverage": coverage}

    def rates(sub):
        y, yh = sub["y"].values, sub["yhat"].values
        tp = ((yh==1)&(y==1)).sum(); fn = ((yh==0)&(y==1)).sum()
        fp = ((yh==1)&(y==0)).sum(); tn = ((yh==0)&(y==0)).sum()
        fnr = fn/(fn+tp) if (fn+tp)>0 else np.nan
        fpr = fp/(fp+tn) if (fp+tn)>0 else np.nan
        return pd.Series({"FNR": fnr, "FPR": fpr})

    per = dfg.groupby("g").apply(rates, include_groups=False).dropna()
    if per.empty or len(per) < 2:
        return {"FNR_gap": np.nan, "FPR_gap": np.nan,
                "n_groups": int(len(per)), "coverage": coverage}
    return {
        "FNR_gap":  float(per["FNR"].max() - per["FNR"].min()),
        "FPR_gap":  float(per["FPR"].max() - per["FPR"].min()),
        "n_groups": int(len(per)),
        "coverage": coverage,
    }

def brier_gap(y_true, p, groups, min_n=MIN_GROUP_N, min_pos=MIN_GROUP_POS):
    dfg = pd.DataFrame({
        "y": np.asarray(y_true).astype(int),
        "p": np.asarray(p),
        "g": np.asarray(groups),
    }).reset_index(drop=True)
    dfg["sq"] = (dfg["p"] - dfg["y"])**2
    grp  = dfg.groupby("g").agg(n=("y","size"), pos=("y","sum"))
    keep = grp[(grp["n"] >= min_n) & (grp["pos"] >= min_pos)].index
    dfg  = dfg[dfg["g"].isin(keep)]
    if dfg.empty or dfg["g"].nunique() < 2:
        return np.nan
    per = dfg.groupby("g")["sq"].mean()
    return float(per.max() - per.min())

# ============================================================
# 6. DELONG TEST
# ============================================================
def delong_auc_test(y_true, p1, p2):
    """
    DeLong test for comparing two correlated AUCs on the same test set.
    Returns (auc1, auc2, z_stat, p_value, ci_lo, ci_hi).
    DeLong et al. (1988) Biometrics.
    """
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

    var_diff = (S10[0,0]/n1 + S01[0,0]/n0) + \
               (S10[1,1]/n1 + S01[1,1]/n0) - \
               2*(S10[0,1]/n1 + S01[0,1]/n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan, np.nan

    diff  = auc1 - auc2
    se    = np.sqrt(var_diff)
    z     = diff / se
    pval  = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval, diff - 1.96*se, diff + 1.96*se

# ============================================================
# 7. WALK-FORWARD FOLDS
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
# 8. MAIN LOOP
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

# Storage (separated: predictive vs fairness)
pred_rows     = []   # predictive metrics per fold/model/variant (threshold-independent)
fair_rows     = []   # fairness metrics per fold/model/variant/threshold
coverage_rows = []   # group coverage per fold/model/variant/threshold
delong_rows   = []   # DeLong per fold/model
zip_rows      = []   # per-ZIP per-fold stats (threshold=0.5)

for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr  = df.loc[tr]
    df_te  = df.loc[te]

    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    zip_te = df_te[ZIP_COL].values if ZIP_COL in df.columns else None

    # --- Structured-only matrix: no text signal at all (v2 fix #1) ---
    imputer_struct = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer_struct.fit_transform(df_tr[STRUCT_COLS_BASE].values.astype(np.float32))
    X_te_s = imputer_struct.transform(df_te[STRUCT_COLS_BASE].values.astype(np.float32))

    scaler_struct = StandardScaler()
    X_tr_s = scaler_struct.fit_transform(X_tr_s)
    X_te_s = scaler_struct.transform(X_te_s)

    # --- Full numeric matrix: structured + text-derived stats (fit on train only) ---
    imputer_full = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_f_num = imputer_full.fit_transform(df_tr[FULL_STRUCT_COLS].values.astype(np.float32))
    X_te_f_num = imputer_full.transform(df_te[FULL_STRUCT_COLS].values.astype(np.float32))

    scaler_full = StandardScaler()
    X_tr_f_num = scaler_full.fit_transform(X_tr_f_num)
    X_te_f_num = scaler_full.transform(X_te_f_num)

    # --- Text features: fit TF-IDF on train only ---
    X_tr_txt, X_te_txt = make_text_features(df_tr[TEXT_COL], df_te[TEXT_COL])

    # Combined matrix: "Structured+Text" = structured + text-stats + TF-IDF
    X_tr_full = hstack([csr_matrix(X_tr_f_num), X_tr_txt])
    X_te_full = hstack([csr_matrix(X_te_f_num), X_te_txt])

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
        ),
    }

    fold_preds = {}

    for mname, model in models.items():
        m_s = model.__class__(**model.get_params())
        m_s.fit(X_tr_s, y_tr)
        p_s = m_s.predict_proba(X_te_s)[:, 1]

        m_f = model.__class__(**model.get_params())
        m_f.fit(X_tr_full, y_tr)
        p_f = m_f.predict_proba(X_te_full)[:, 1]

        fold_preds[mname] = {"struct": p_s, "full": p_f}

        # ── Predictive metrics (threshold-independent) ──────────────────
        for variant, p in [("Structured", p_s), ("Structured+Text", p_f)]:
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
                "Variant":      variant,
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

            # ── Fairness per threshold ───────────────────────────────────
            for thr in THRESHOLDS:
                fair_row = {
                    "fold":         f["fold"],
                    "train_cutoff": f["train_cutoff"],
                    "test_end":     f["test_end"],
                    "Model":        mname,
                    "Variant":      variant,
                    "threshold":    thr,
                }
                if zip_te is not None:
                    fg = fairness_gaps(y_te, p, zip_te, threshold=thr)
                    bg = brier_gap(y_te, p, zip_te)
                    fair_row.update({
                        "FNR_gap":   round(fg["FNR_gap"],   4) if not np.isnan(fg["FNR_gap"])   else np.nan,
                        "FPR_gap":   round(fg["FPR_gap"],   4) if not np.isnan(fg["FPR_gap"])   else np.nan,
                        "Brier_gap": round(bg, 4)              if not np.isnan(bg)               else np.nan,
                        "n_groups":  fg["n_groups"],
                    })
                    cov = fg["coverage"]
                    coverage_rows.append({
                        "fold":                  f["fold"],
                        "Model":                 mname,
                        "Variant":               variant,
                        "threshold":             thr,
                        "n_candidate_groups":    cov["n_candidate_groups"],
                        "n_pass_n_threshold":    cov["n_pass_n_threshold"],
                        "n_pass_pos_threshold":  cov["n_pass_pos_threshold"],
                        "n_included":            cov["n_included"],
                    })
                fair_rows.append(fair_row)

                # Per-ZIP stats (threshold=0.5 only, avoid duplicate rows)
                if zip_te is not None and thr == 0.5:
                    p_arr = np.asarray(p)
                    y_arr = np.asarray(y_te).astype(int)
                    g_arr = np.asarray(zip_te)
                    dfz   = pd.DataFrame({
                        "y": y_arr, "p": p_arr,
                        "yhat": (p_arr >= thr).astype(int),
                        "g": g_arr,
                    }).reset_index(drop=True)
                    for grp_name, grp_df in dfz.groupby("g"):
                        n   = len(grp_df)
                        pos = int(grp_df["y"].sum())
                        if n < MIN_GROUP_N or pos < MIN_GROUP_POS:
                            continue
                        y_g  = grp_df["y"].values
                        yh_g = grp_df["yhat"].values
                        p_g  = grp_df["p"].values
                        tp = ((yh_g==1)&(y_g==1)).sum()
                        fn = ((yh_g==0)&(y_g==1)).sum()
                        fp = ((yh_g==1)&(y_g==0)).sum()
                        tn = ((yh_g==0)&(y_g==0)).sum()
                        try:
                            auc_g = roc_auc_score(y_g, p_g) if 0 < pos < n else np.nan
                        except Exception:
                            auc_g = np.nan
                        zip_rows.append({
                            "zip3":         grp_name,
                            "fold":         f["fold"],
                            "Model":        mname,
                            "Variant":      variant,
                            "n":            n,
                            "n_default":    pos,
                            "default_rate": round(pos / n, 4),
                            "AUC":          round(auc_g, 4) if not np.isnan(auc_g) else np.nan,
                            "FNR":          round(fn/(fn+tp), 4) if (fn+tp)>0 else np.nan,
                            "FPR":          round(fp/(fp+tn), 4) if (fp+tn)>0 else np.nan,
                            "Brier":        round(float(np.mean((p_g - y_g)**2)), 4),
                        })

        # ── DeLong: Structured vs Structured+Text ───────────────────────
        auc_s, auc_f, z, pval, ci_lo, ci_hi = delong_auc_test(
            y_te, fold_preds[mname]["struct"], fold_preds[mname]["full"]
        )
        delong_rows.append({
            "fold":                f["fold"],
            "Model":               mname,
            "AUC_Structured":      round(auc_s, 4),
            "AUC_Structured+Text": round(auc_f, 4),
            "Delta_AUC":           round(auc_f - auc_s, 4),
            "Z_stat":              round(z,    3) if not np.isnan(z)    else np.nan,
            "p_value":             round(pval, 4) if not np.isnan(pval) else np.nan,
            "CI_95_lo":            round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "CI_95_hi":            round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
        })

    print(f"  Fold {f['fold']} done.")

# ============================================================
# 9. AGGREGATE RESULTS
# ============================================================
df_pred     = pd.DataFrame(pred_rows)
df_fair     = pd.DataFrame(fair_rows)
df_coverage = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
df_delong   = pd.DataFrame(delong_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]
FAIR_METRICS = [c for c in ["FNR_gap", "FPR_gap", "Brier_gap"] if c in df_fair.columns]

# ── A. Predictive performance summary ───────────────────────────────────────
pred_summary = (
    df_pred.groupby(["Model", "Variant"])[PRED_METRICS]
    .agg(["mean", "std"])
    .round(4)
)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean ± std across folds)")
print("="*70)
print(pred_summary.to_string())

# ── B. Delta: Structured+Text − Structured ──────────────────────────────────
delta_rows = []
for m in df_pred["Model"].unique():
    for fold in df_pred["fold"].unique():
        base = df_pred[(df_pred["Model"]==m) & (df_pred["Variant"]=="Structured")      & (df_pred["fold"]==fold)]
        full = df_pred[(df_pred["Model"]==m) & (df_pred["Variant"]=="Structured+Text") & (df_pred["fold"]==fold)]
        if base.empty or full.empty:
            continue
        row = {"fold": fold, "Model": m}
        for c in PRED_METRICS:
            row[f"Δ_{c}"] = round(float(full.iloc[0][c]) - float(base.iloc[0][c]), 4)
        delta_rows.append(row)

df_delta = pd.DataFrame(delta_rows)
delta_metric_cols = [f"Δ_{c}" for c in PRED_METRICS]
df_delta_summary = (
    df_delta.groupby("Model")[delta_metric_cols]
    .agg(["mean", "std"])
    .round(4)
)

print("\n" + "="*70)
print("B. DELTA: Structured+Text − Structured  (mean ± std across folds)")
print("="*70)
print(df_delta_summary.to_string())

print("\n  Consistency check (fraction of folds where text improved AUC):")
for m in df_delta["Model"].unique():
    sub      = df_delta[df_delta["Model"]==m]
    frac     = (sub["Δ_AUC"] > 0).mean()
    mean_d   = sub["Δ_AUC"].mean()
    verdict  = "consistent" if frac >= 0.7 else ("mixed" if frac >= 0.4 else "mostly worse")
    print(f"    [{m}]  improved in {frac:.0%} of folds  "
          f"mean ΔAUC={mean_d:+.4f}  → {verdict}")

# ── C. DeLong summary ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("C. DELONG TEST: Structured vs Structured+Text (per fold)")
print("="*70)
print(df_delong.to_string(index=False))

delong_summary = (
    df_delong.groupby("Model")[["Delta_AUC", "Z_stat", "p_value"]]
    .mean()
    .round(4)
)
print("\n  DeLong — mean across folds:")
print(delong_summary.to_string())

# ── D. Fairness summary by threshold ────────────────────────────────────────
if FAIR_METRICS:
    print("\n" + "="*70)
    print("D. FAIRNESS SUMMARY BY THRESHOLD (mean ± std across folds)")
    print("="*70)
    for thr in THRESHOLDS:
        sub = df_fair[df_fair["threshold"] == thr]
        if sub.empty:
            continue
        summary = (
            sub.groupby(["Model", "Variant"])[FAIR_METRICS]
            .agg(["mean", "std"])
            .round(4)
        )
        print(f"\n  Threshold = {thr}")
        print(summary.to_string())

    # ── E. Fairness direction: did adding text improve or worsen fairness? ───
    print("\n" + "="*70)
    print("E. FAIRNESS DIRECTION: Adding text vs Structured-only")
    print("   (for gap metrics: negative delta = improved; positive = worsened)")
    print("="*70)
    for thr in THRESHOLDS:
        print(f"\n  Threshold = {thr}")
        for m in df_fair["Model"].unique():
            for c in FAIR_METRICS:
                base_vals = df_fair[
                    (df_fair["Model"]==m) &
                    (df_fair["Variant"]=="Structured") &
                    (df_fair["threshold"]==thr)
                ][c].dropna()
                full_vals = df_fair[
                    (df_fair["Model"]==m) &
                    (df_fair["Variant"]=="Structured+Text") &
                    (df_fair["threshold"]==thr)
                ][c].dropna()
                if base_vals.empty or full_vals.empty:
                    continue
                delta     = full_vals.mean() - base_vals.mean()
                direction = "improved" if delta < -1e-5 else ("worsened" if delta > 1e-5 else "no change")
                print(f"    [{m}] {c}: mean delta={delta:+.4f}  → {direction}")

# ── F. Group coverage summary ────────────────────────────────────────────────
if not df_coverage.empty:
    cov_summary = (
        df_coverage.groupby(["Model", "Variant", "threshold"])
        .agg(
            n_candidate_groups   =("n_candidate_groups",   "mean"),
            n_pass_n_threshold   =("n_pass_n_threshold",   "mean"),
            n_pass_pos_threshold =("n_pass_pos_threshold", "mean"),
            n_included           =("n_included",           "mean"),
        )
        .round(1)
        .reset_index()
    )
    print("\n" + "="*70)
    print("F. GROUP COVERAGE SUMMARY (mean across folds)")
    print("   Columns: candidate → pass-N filter → pass-pos filter → included in fairness calc")
    print("="*70)
    print(cov_summary.to_string(index=False))

# ── G. ZIP-level summary ─────────────────────────────────────────────────────
if zip_rows:
    df_zip = pd.DataFrame(zip_rows)
    df_zip_summary = (
        df_zip.groupby(["zip3", "Model", "Variant"])
        .agg(
            n_folds    =("fold",      "count"),
            n_total    =("n",         "sum"),
            n_default  =("n_default", "sum"),
            AUC_mean   =("AUC",       "mean"),
            FNR_mean   =("FNR",       "mean"),
            FPR_mean   =("FPR",       "mean"),
            Brier_mean =("Brier",     "mean"),
        )
        .reset_index()
        .round(4)
    )
    df_zip_summary["default_rate"] = (
        df_zip_summary["n_default"] / df_zip_summary["n_total"]
    ).round(4)
    print("\n" + "="*70)
    print("G. ZIP3 SUMMARY (mean across folds, threshold=0.5, top 20 by n_total)")
    print("="*70)
    print(df_zip_summary.sort_values("n_total", ascending=False).head(20).to_string(index=False))
else:
    df_zip_summary = pd.DataFrame()
    print("\nNo ZIP-level data collected (check MIN_GROUP_N / MIN_GROUP_POS).")

# ============================================================
# 10. SAVE
# ============================================================
# v2 fix: write to a separate folder so v1's (pre-fix) results are preserved
# as a historical baseline rather than silently overwritten.
out_dir = BASE_DIR / "results" / "walk_forward_v2"
os.makedirs(out_dir, exist_ok=True)

# Raw per-fold data
df_pred.to_csv(os.path.join(out_dir, "wf_predictive_folds.csv"),  index=False)
df_fair.to_csv(os.path.join(out_dir, "wf_fairness_folds.csv"),    index=False)
df_delong.to_csv(os.path.join(out_dir, "wf_delong.csv"),          index=False)

# Summary tables
pred_summary.to_csv(os.path.join(out_dir, "wf_predictive_summary.csv"))
df_delta_summary.to_csv(os.path.join(out_dir, "wf_delta_summary.csv"))

# Fairness summaries — one file per threshold
if FAIR_METRICS:
    for thr in THRESHOLDS:
        sub = df_fair[df_fair["threshold"] == thr]
        if sub.empty:
            continue
        out_thr = (
            sub.groupby(["Model", "Variant"])[FAIR_METRICS]
            .agg(["mean", "std"])
            .round(4)
        )
        thr_tag = str(thr).replace(".", "")
        out_thr.to_csv(os.path.join(out_dir, f"wf_fairness_summary_thr{thr_tag}.csv"))

if not df_coverage.empty:
    cov_summary.to_csv(os.path.join(out_dir, "wf_group_coverage.csv"), index=False)
if not df_zip_summary.empty:
    df_zip_summary.to_csv(os.path.join(out_dir, "wf_zip_summary.csv"), index=False)

print(f"\nSaved results to: {out_dir}")
print("\nOutput files:")
print("  wf_predictive_folds.csv           — per-fold predictive metrics with bootstrap 95% CIs")
print("  wf_fairness_folds.csv             — per-fold fairness metrics by threshold")
print("  wf_delong.csv                     — DeLong test per fold")
print("  wf_predictive_summary.csv         — predictive summary (mean ± std)")
print("  wf_delta_summary.csv              — text vs struct deltas")
print("  wf_fairness_summary_thr{05,06,07}.csv — fairness summary per threshold")
print("  wf_group_coverage.csv             — group coverage statistics")
print("  wf_zip_summary.csv                — ZIP3-level summary (threshold=0.5)")

# ============================================================
# 11. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

figures_dir = BASE_DIR / "results" / "figures_v2"
os.makedirs(figures_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

# Consistent palette
_CLR  = {"Logistic": "#2166ac", "XGBoost": "#d6604d"}
_COMBO_COLORS  = ["#2166ac", "#74add1", "#d6604d", "#f4a582"]
_COMBO_LABELS  = ["LR – Structured", "LR – Structured+Text",
                  "XGB – Structured", "XGB – Structured+Text"]
_COMBOS        = [("Logistic","Structured"), ("Logistic","Structured+Text"),
                  ("XGBoost","Structured"),  ("XGBoost","Structured+Text")]

print("\nGenerating figures...")

# ── Figure 1: Walk-Forward AUC Stability ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, model in zip(axes, ["Logistic", "XGBoost"]):
    color = _CLR[model]
    for variant, ls, marker, alpha in [
        ("Structured",       "-",  "o", 1.00),
        ("Structured+Text",  "--", "s", 0.70),
    ]:
        sub = (df_pred[(df_pred["Model"] == model) & (df_pred["Variant"] == variant)]
               .sort_values("fold"))
        ax.plot(sub["fold"], sub["AUC"],
                linestyle=ls, marker=marker, color=color, alpha=alpha,
                linewidth=2, markersize=5, label=variant)
        if not sub["AUC_CI_lo"].isna().all():
            ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"],
                            alpha=0.12, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("AUC")
    ax.legend(title="Variant", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig.suptitle("Walk-Forward AUC Stability Across Folds\n(shaded band = 95% bootstrap CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_auc_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig1_auc_stability.png")

# ── Figure 2: DeLong ΔAUC per fold ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, model in zip(axes, ["Logistic", "XGBoost"]):
    sub = df_delong[df_delong["Model"] == model].sort_values("fold")
    pvals = sub["p_value"].fillna(1.0).values
    bar_colors = ["#d73027" if p < 0.05 else "#bababa" for p in pvals]

    x = sub["fold"].values
    y = sub["Delta_AUC"].values
    ax.bar(x, y, color=bar_colors, alpha=0.85, width=0.6, zorder=3)

    # 95% CI error bars (asymmetric: lo/hi are bounds of delta, not half-widths)
    ci_lo = sub["CI_95_lo"].values
    ci_hi = sub["CI_95_hi"].values
    valid = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
    if valid.any():
        # CI_95 is for (auc_s - auc_f); Delta_AUC = auc_f - auc_s
        # SE is symmetric, so half-width = (ci_hi - ci_lo) / 2
        half_width = (ci_hi - ci_lo) / 2
        ax.errorbar(x[valid], y[valid],
                    yerr=half_width[valid],
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)

    ax.axhline(0, color="black", linewidth=1.2, zorder=5)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("ΔAUC  (Structured+Text − Structured)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    sig_patch = mpatches.Patch(color="#d73027", alpha=0.85, label="p < 0.05")
    ns_patch  = mpatches.Patch(color="#bababa", alpha=0.85, label="p ≥ 0.05")
    ax.legend(handles=[sig_patch, ns_patch], title="DeLong test", fontsize=9)

fig.suptitle("DeLong Test: ΔAUC per Fold  (Structured+Text − Structured)\n"
             "(bars above zero = text improves AUC; error bars = 95% CI)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig2_delong_delta_auc.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig2_delong_delta_auc.png")

# ── Figure 3: Fairness Trade-off Scatter ────────────────────────────────────
fair_05 = df_fair[df_fair["threshold"] == 0.5].copy()
scatter_rows = []
for model, variant in _COMBOS:
    p_sub = df_pred[(df_pred["Model"] == model) & (df_pred["Variant"] == variant)]
    f_sub = fair_05[(fair_05["Model"] == model) & (fair_05["Variant"] == variant)]
    if p_sub.empty or f_sub.empty:
        continue
    scatter_rows.append({
        "Model":        model,
        "Variant":      variant,
        "AUC_mean":     p_sub["AUC"].mean(),
        "AUC_std":      p_sub["AUC"].std(),
        "FNR_mean":     f_sub["FNR_gap"].mean(),
        "FNR_std":      f_sub["FNR_gap"].std(),
    })
df_sc = pd.DataFrame(scatter_rows)

fig, ax = plt.subplots(figsize=(8, 6))
for color, (_, row) in zip(_COMBO_COLORS, df_sc.iterrows()):
    marker = "o" if row["Variant"] == "Structured" else "s"
    ax.errorbar(row["AUC_mean"], row["FNR_mean"],
                xerr=row["AUC_std"], yerr=row["FNR_std"],
                fmt=marker, color=color, markersize=11,
                capsize=2, elinewidth=0.8, ecolor="#aaaaaa",
                linewidth=1.4, label=f"{row['Model']} – {row['Variant']}")
    short = ("LR" if row["Model"] == "Logistic" else "XGB") + \
            ("-S" if row["Variant"] == "Structured" else "-S+T")
    ax.annotate(short, (row["AUC_mean"], row["FNR_mean"]),
                textcoords="offset points", xytext=(7, 4), fontsize=9)

ax.set_xlabel("Mean AUC  (±1 SD across folds)", fontsize=11)
ax.set_ylabel("Mean FNR Gap  (±1 SD across folds)", fontsize=11)
ax.set_title("Predictive Accuracy vs. Fairness Trade-off\n"
             "(threshold = 0.5; lower FNR Gap = more equitable)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(figures_dir / "fig3_fairness_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig3_fairness_tradeoff.png")

# ── Figure 4: Fairness Gaps by Decision Threshold ───────────────────────────
fair_agg = (
    df_fair
    .groupby(["Model", "Variant", "threshold"])[["FNR_gap", "FPR_gap", "Brier_gap"]]
    .agg(FNR_mean=("FNR_gap",   "mean"), FNR_std=("FNR_gap",   "std"),
         FPR_mean=("FPR_gap",   "mean"), FPR_std=("FPR_gap",   "std"),
         Brier_mean=("Brier_gap","mean"), Brier_std=("Brier_gap","std"))
    .reset_index()
)

thresholds = sorted(fair_agg["threshold"].unique())
x          = np.arange(len(thresholds))
bar_width  = 0.18

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax, (metric_mean, metric_std, ylabel, title) in zip(
    axes,
    [("FNR_mean",   "FNR_std",   "FNR Gap",   "False Negative Rate Gap"),
     ("FPR_mean",   "FPR_std",   "FPR Gap",   "False Positive Rate Gap"),
     ("Brier_mean", "Brier_std", "Brier Gap", "Brier Score Gap")],
):
    for i, ((model, variant), color, label) in enumerate(
        zip(_COMBOS, _COMBO_COLORS, _COMBO_LABELS)
    ):
        sub = (fair_agg[(fair_agg["Model"] == model) & (fair_agg["Variant"] == variant)]
               .sort_values("threshold"))
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, sub[metric_mean], bar_width,
               yerr=sub[metric_std],
               error_kw={"capsize": 2, "elinewidth": 0.8, "ecolor": "#aaaaaa"},
               color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

fig.suptitle("Fairness Gaps by Decision Threshold  (mean ± SD across folds)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "fig4_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  fig4_threshold_sensitivity.png")

print(f"\nAll figures saved to: {figures_dir}")

# ── Summary tables per threshold ─────────────────────────────────────────────
pred_means = (
    df_pred
    .groupby(["Model", "Variant"])[["AUC", "PR_AUC", "Brier"]]
    .mean()
    .round(4)
    .reset_index()
)

print("\n" + "="*70)
print("SUMMARY TABLES BY THRESHOLD")
print("="*70)

for thr in THRESHOLDS:
    fair_means = (
        df_fair[df_fair["threshold"] == thr]
        .groupby(["Model", "Variant"])[["Brier_gap", "FNR_gap", "FPR_gap"]]
        .mean()
        .round(4)
        .reset_index()
    )
    tbl = pred_means.merge(fair_means, on=["Model", "Variant"])
    tbl = tbl[["Model", "Variant", "AUC", "PR_AUC", "Brier",
               "Brier_gap", "FNR_gap", "FPR_gap"]]

    thr_tag = str(thr).replace(".", "")
    tbl.to_csv(out_dir / f"wf_summary_thr{thr_tag}.csv", index=False)

    print(f"\n  Threshold = {thr}")
    print(tbl.to_string(index=False))

print(f"\nSummary tables saved to: {out_dir}")

# ============================================================
# 12. FEATURE IMPORTANCE — LAST FOLD, STRUCTURED+TEXT
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE — Last Fold, Structured+Text")
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

# Structured+Text numeric part — same pipeline as main loop (v2 fix #1: FULL_STRUCT_COLS)
_imputer = SimpleImputer(strategy="constant", fill_value=0)
_X_tr_s  = _imputer.fit_transform(_df_tr[FULL_STRUCT_COLS].values.astype(np.float32))
_X_te_s  = _imputer.transform(_df_te[FULL_STRUCT_COLS].values.astype(np.float32))
_scaler  = StandardScaler()
_X_tr_s  = _scaler.fit_transform(_X_tr_s)
_X_te_s  = _scaler.transform(_X_te_s)

# Text features — same TF-IDF as main loop
_X_tr_txt, _X_te_txt = make_text_features(_df_tr[TEXT_COL], _df_te[TEXT_COL])

# Reconstruct TF-IDF vocabulary for feature names
_tfidf_fi = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=True, min_df=5,
    strip_accents="unicode", analyzer="word",
    token_pattern=r"[a-zA-Z]{3,}",
    stop_words="english", norm="l2",
)
_tfidf_fi.fit(_df_tr[TEXT_COL].fillna("").astype(str))
_tfidf_names = _tfidf_fi.get_feature_names_out().tolist()

_all_names = FULL_STRUCT_COLS + _tfidf_names

# Full matrix
_X_tr_full = hstack([csr_matrix(_X_tr_s), _X_tr_txt])
_X_te_full = hstack([csr_matrix(_X_te_s), _X_te_txt])

# Train models — same params as main loop
_lr = LogisticRegression(**LR_PARAMS)
_lr.fit(_X_tr_full, _y_tr)

_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    scale_pos_weight=(_y_tr==0).sum() / (_y_tr==1).sum(),
    eval_metric="auc", use_label_encoder=False, verbosity=0, n_jobs=-1,
)
_xgb.fit(_X_tr_full, _y_tr)

def _feature_kind(name):
    """v2 addition: distinguish TF-IDF tokens from engineered text-stat features,
    so the contamination fix (v2 changelog #1) is visible in the importance tables."""
    if name in _tfidf_names:
        return "TF-IDF"
    if name in TEXT_STAT_COLS:
        return "text-stat"
    return ""

# ── LR importance: absolute standardized coefficients ────────────────────────
_lr_imp = np.abs(_lr.coef_[0])
_df_lr_imp = (
    pd.DataFrame({"Feature": _all_names, "Importance": _lr_imp})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
_df_lr_imp.index += 1
_df_lr_imp["Text?"] = _df_lr_imp["Feature"].map(_feature_kind)

# ── XGBoost importance: Gain ──────────────────────────────────────────────────
_xgb_gain = _xgb.get_booster().get_score(importance_type="gain")
_df_xgb_imp = pd.DataFrame([
    {"Feature": _all_names[int(k.replace("f", ""))], "Gain": round(v, 2)}
    for k, v in _xgb_gain.items()
]).sort_values("Gain", ascending=False).reset_index(drop=True)
_df_xgb_imp.index += 1
_df_xgb_imp["Text?"] = _df_xgb_imp["Feature"].map(_feature_kind)

# ── XGBoost text-only importance ─────────────────────────────────────────────
_df_xgb_text = (
    _df_xgb_imp[_df_xgb_imp["Text?"] == "TF-IDF"]
    .drop(columns=["Text?"])
    .reset_index(drop=True)
)
_df_xgb_text.index += 1

TOP_N = 20

print("\n── Table 1: XGBoost — Top 20 Features (Structured+Text) ──")
print(_df_xgb_imp.head(TOP_N).to_string())

print("\n── Table 2: Logistic Regression — Top 20 Features (Structured+Text) ──")
print(_df_lr_imp.head(TOP_N).to_string())

print("\n── Table 3: XGBoost — Top 20 Text (TF-IDF) Features ──")
print(_df_xgb_text.head(TOP_N).to_string())
