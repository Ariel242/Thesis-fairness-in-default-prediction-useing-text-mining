"""
TF-IDF Text Pipeline for Default Prediction
============================================
Standalone text pipeline, split out of analysis/preliminary_results_v2.py
in v2.2: that script now trains the traditional (fully structured, no text)
model only. This script owns the text side end-to-end — raw pull, cleaning /
lemmatization, numeric text-derived stats, and TF-IDF vectorization — so the
two can be developed and rerun independently.

TF-IDF is still fit walk-forward (per fold, on the training rows only, then
applied to test) — same folds, same protocol as before, so there is no
leakage and the fold boundaries line up exactly with
preliminary_results_v2.py's own build_folds() (same params, same input CSV).

The vocabulary is UNCAPPED here (previously TFIDF_MAX_FEATURES=1000 in the
old combined script): every token that passes min_df=5 is kept, so "the full
matrix" per fold means the complete vocabulary for that fold's training
text, not a top-N cut.

Output (written to TF-IDF/output/):
  text_all_clean.csv    — id + cleaned/lemmatized text_all_clean (whole corpus, computed once)
  text_stats.csv        — id + numeric text-derived stats (char_len, uniq_words,
                           avg_word_len, removed_words, ttr) per raw column (whole corpus)
  folds_manifest.csv    — fold boundaries (train_cutoff / test_end / n_train / n_test / n_test_defaults)
  fold{NN}/train_ids.csv, test_ids.csv       — row ids per fold (join key back to the main CSV)
  fold{NN}/train_tfidf.npz, test_tfidf.npz   — sparse TF-IDF matrices, fit on train only
  fold{NN}/vocab.txt                         — TF-IDF feature (n-gram) names for that fold, in column order
"""

import re
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.sparse import save_npz
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent          # F-TM-CR
PATH_CSV = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
OUT_DIR  = Path(__file__).resolve().parent / "output"       # TF-IDF/output

TARGET_COL    = "is_default"
DATE_COL      = "issue_month_start"
ID_COL        = "id"
TEXT_COL      = "text_all_clean"
RAW_TEXT_COLS = ["desc", "title", "emp_title"]

# Walk-Forward parameters — MUST match preliminary_results_v2.py exactly,
# so fold boundaries here line up with the modeling script's folds.
MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

# TF-IDF — uncapped: every token passing min_df=5 is kept (no top-N cut)
TFIDF_MAX_FEATURES = None
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)

df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

if ID_COL not in df.columns:
    raise ValueError(
        f"Expected id column '{ID_COL}' not found in CSV — needed to join "
        f"TF-IDF output back to the main dataset in preliminary_results_v2.py."
    )

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

df[TEXT_COL] = (
    df["desc_clean"].fillna("") + " " +
    df["title_clean"].fillna("") + " " +
    df["emp_title_clean"].fillna("")
).str.strip()

df[[ID_COL, TEXT_COL]].to_csv(OUT_DIR / "text_all_clean.csv", index=False)
print(f"  Saved cleaned text: {OUT_DIR / 'text_all_clean.csv'}")

# ============================================================
# 3. NUMERIC TEXT-DERIVED STATS  (char length, word count, TTR, ...)
# ============================================================
CLEAN_COLS = ["desc_clean", "title_clean", "emp_title_clean"]
RAW_COLS   = ["desc", "title", "emp_title"]

def _wc(s):
    return s.str.split().str.len().fillna(0)

stats_df = df[[ID_COL]].copy()
for raw_c, clean_c in zip(RAW_COLS, CLEAN_COLS):
    if raw_c not in df.columns:
        continue
    raw   = df[raw_c].fillna("").astype(str).str.strip()
    clean = df[clean_c].fillna("").astype(str).str.strip()
    raw_wc   = _wc(raw.str.lower())
    clean_wc = _wc(clean)
    stats_df[f"{clean_c}_char_len"]      = clean.str.len()
    stats_df[f"{clean_c}_uniq_words"]    = clean.apply(lambda x: len(set(x.split())) if x else 0)
    stats_df[f"{clean_c}_avg_word_len"]  = clean.apply(
        lambda x: float(np.mean([len(w) for w in x.split()])) if x else 0.0)
    stats_df[f"{clean_c}_removed_words"] = (raw_wc - clean_wc).clip(lower=0)
    stats_df[f"{clean_c}_ttr"]           = np.where(
        clean_wc > 0, stats_df[f"{clean_c}_uniq_words"] / clean_wc, 0.0)

stats_df.to_csv(OUT_DIR / "text_stats.csv", index=False)
print(f"  Saved text-derived numeric stats: {OUT_DIR / 'text_stats.csv'}  ({stats_df.shape[1]-1} columns)")

# ============================================================
# 4. WALK-FORWARD FOLDS  (identical logic/params to preliminary_results_v2.py)
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

print("\nBuilding Walk-Forward folds...")
folds = build_folds(
    df,
    date_col          = DATE_COL,
    min_train_months  = MIN_TRAIN_MONTHS,
    step_months       = STEP_MONTHS,
    min_test_defaults = MIN_TEST_DEFAULTS,
)
print(f"  Total folds: {len(folds)}")

manifest_rows = []
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  "
          f"test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  "
          f"defaults_in_test={f['n_test_defaults']}")
    manifest_rows.append({k: v for k, v in f.items() if k not in ("tr_idx", "te_idx")})

pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "folds_manifest.csv", index=False)
print(f"  Saved fold boundaries: {OUT_DIR / 'folds_manifest.csv'}")

# ============================================================
# 5. TF-IDF PER FOLD  (fit on train only — no leakage; uncapped vocabulary)
# ============================================================
print("\nFitting TF-IDF per fold (train-only, no max_features cap)...")
for f in folds:
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr, df_te = df.loc[tr], df.loc[te]

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
    X_tr = tfidf.fit_transform(df_tr[TEXT_COL].fillna("").astype(str))
    X_te = tfidf.transform(df_te[TEXT_COL].fillna("").astype(str))
    vocab = tfidf.get_feature_names_out()

    fold_dir = OUT_DIR / f"fold{f['fold']:02d}"
    os.makedirs(fold_dir, exist_ok=True)

    save_npz(fold_dir / "train_tfidf.npz", X_tr)
    save_npz(fold_dir / "test_tfidf.npz",  X_te)
    pd.Series(vocab).to_csv(fold_dir / "vocab.txt", index=False, header=False)
    df_tr[[ID_COL]].to_csv(fold_dir / "train_ids.csv", index=False)
    df_te[[ID_COL]].to_csv(fold_dir / "test_ids.csv",  index=False)

    print(f"  Fold {f['fold']:>2}: vocab={len(vocab):,}  "
          f"train={X_tr.shape[0]:,}x{X_tr.shape[1]:,}  test={X_te.shape[0]:,}x{X_te.shape[1]:,}")

print(f"\nDone. Output written to: {OUT_DIR}")
