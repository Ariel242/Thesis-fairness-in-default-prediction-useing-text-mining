# -*- coding: utf-8 -*-
"""
01_extract_frozen_vocabulary.py
================================
Step 1 of the Distress Lexicon protocol (see Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf,
section 2, "Strict Vocabulary Extraction & Freezing").

Extracts a unigram/bigram vocabulary from the DESC column, restricted to ONLY
the initial baseline training window (the first MIN_TRAIN_MONTHS calendar
months of issue_month_start), before any walk-forward test fold exists. This
mirrors the exact fold-boundary parameters already used by TF-IDF/tfidf_pipeline.py
and analysis/preliminary_results_v2.py (MIN_TRAIN_MONTHS=8, DATE_COL=issue_month_start,
same input CSV) -- values copied inline here, not imported, so this script has
no runtime dependency outside Dictionarys/.

This supersedes the old extract_vocabulary.py, which read the full raw
un-windowed dataset (data/accepted_2007_to_2018Q4.csv) -- exactly the kind of
data leakage this step exists to prevent.

Output: Dictionarys/dictionaries/frozen_vocabulary_initial_window.csv
        (Term, Type, Document_Frequency)
        Dictionarys/results/frozen_window_manifest.md (audit log)
"""

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ============================================================
# PARAMETERS -- copied inline from TF-IDF/tfidf_pipeline.py (read-only
# reference; not imported) so fold boundaries line up exactly.
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../Dictionarys/scripts
DICT_DIR   = SCRIPT_DIR.parent                                # .../Dictionarys
REPO_ROOT  = DICT_DIR.parent                                  # .../F-TM-CR

INPUT_CSV = REPO_ROOT / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"

OUT_VOCAB_CSV = DICT_DIR / "dictionaries" / "frozen_vocabulary_initial_window.csv"
OUT_MANIFEST  = DICT_DIR / "results" / "frozen_window_manifest.md"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"
DESC_COL   = "desc"

MIN_TRAIN_MONTHS = 8  # must match TF-IDF/tfidf_pipeline.py's build_folds()

# freq > 15 per the PDF's "Strict Vocabulary Extraction & Freezing" step
MIN_DOC_FREQ = 16
NGRAM_RANGE  = (1, 2)

# Same custom stopwords as TF-IDF/tfidf_pipeline.py's cleaning step, applied
# consistently so the frozen vocabulary matches how features will later be
# extracted from cleaned text (script 04).
CUSTOM_STOP_WORDS = {
    "added", "borrower", "loan", "lending", "club",
    "thanks", "thank", "quot", "consideration", "time",
    "just", "like", "want", "make", "need", "br",
}
STOPWORDS = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))

PREFIX_RE = re.compile(r"^\s*borrower\s+added\s+on\s+\d{2}/\d{2}/\d{2}\s*>\s*", re.IGNORECASE)

_WNL = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    return _WNL.lemmatize(_WNL.lemmatize(token, "v"), "n")


def clean_desc(text) -> str:
    """Same cleaning approach as TF-IDF/tfidf_pipeline.py's _clean_desc:
    strip the 'Borrower added on...' boilerplate prefix, lowercase, drop
    digits/punctuation, lemmatize (NLTK -- no spaCy installed in this env)."""
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = PREFIX_RE.sub("", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    tokens = [_lemma(t) for t in s.split()]
    return " ".join(tokens)


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV, usecols=[ID_COL, DATE_COL, TARGET_COL, DESC_COL],
                      parse_dates=[DATE_COL], low_memory=False)
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"  Rows total: {len(df):,}")

    # ------------------------------------------------------------
    # Initial baseline training window boundary (fold-1 train_cutoff,
    # identical logic to build_folds() in TF-IDF/tfidf_pipeline.py)
    # ------------------------------------------------------------
    months = df[DATE_COL].dt.to_period("M")
    all_periods = sorted(months.unique())
    if len(all_periods) < MIN_TRAIN_MONTHS:
        raise ValueError(
            f"Only {len(all_periods)} distinct months in the data; "
            f"need at least MIN_TRAIN_MONTHS={MIN_TRAIN_MONTHS}."
        )
    train_cutoff = all_periods[MIN_TRAIN_MONTHS - 1]
    window_mask = months <= train_cutoff
    df_window = df.loc[window_mask].copy()

    print(f"  Initial window: {all_periods[0]} .. {train_cutoff} "
          f"({MIN_TRAIN_MONTHS} months)")
    print(f"  Rows in window: {len(df_window):,} / {len(df):,}")

    # ------------------------------------------------------------
    # Clean + lemmatize DESC, window rows only
    # ------------------------------------------------------------
    print("  Cleaning + lemmatizing desc (window rows only)...")
    texts = df_window[DESC_COL].apply(clean_desc)
    texts = texts[texts != ""]
    if texts.empty:
        raise ValueError("No non-empty desc text remained in the initial window.")
    print(f"  Non-empty desc rows in window: {len(texts):,}")

    # ------------------------------------------------------------
    # Extract vocabulary: document frequency > 15, no top-N cap
    # ------------------------------------------------------------
    vectorizer = CountVectorizer(
        ngram_range=NGRAM_RANGE,
        stop_words=STOPWORDS,
        min_df=MIN_DOC_FREQ,
        binary=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    try:
        doc_term = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise ValueError(
            f"No terms remained after applying min_df={MIN_DOC_FREQ} "
            f"on the initial-window text."
        ) from exc

    terms = vectorizer.get_feature_names_out()
    doc_freq = doc_term.sum(axis=0).A1.astype(int)

    vocab = pd.DataFrame({
        "Term": terms,
        "Type": ["Bigram" if " " in t else "Unigram" for t in terms],
        "Document_Frequency": doc_freq,
    }).sort_values(by=["Document_Frequency", "Term"], ascending=[False, True]).reset_index(drop=True)

    OUT_VOCAB_CSV.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(OUT_VOCAB_CSV, index=False, encoding="utf-8-sig")

    n_uni = int((vocab["Type"] == "Unigram").sum())
    n_bi = int((vocab["Type"] == "Bigram").sum())
    print(f"  Unigrams: {n_uni:,}  Bigrams: {n_bi:,}  Total: {len(vocab):,}")
    print(f"  Output: {OUT_VOCAB_CSV}")

    # ------------------------------------------------------------
    # Audit manifest
    # ------------------------------------------------------------
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        f.write("# Frozen Vocabulary Extraction -- Audit Manifest\n\n")
        f.write(f"- Executed: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Input CSV: `{INPUT_CSV.relative_to(REPO_ROOT)}`\n")
        f.write(f"- Date column: `{DATE_COL}` | Target column: `{TARGET_COL}`\n")
        f.write(f"- MIN_TRAIN_MONTHS: {MIN_TRAIN_MONTHS} "
                f"(must match TF-IDF/tfidf_pipeline.py's build_folds())\n")
        f.write(f"- Initial window: {all_periods[0]} .. {train_cutoff}\n")
        f.write(f"- Rows in window (is_default in [0,1]): {len(df_window):,} "
                f"/ {len(df):,} total rows\n")
        f.write(f"- Non-empty desc rows in window: {len(texts):,}\n")
        f.write(f"- min_df (document frequency pruning, PDF's freq>15): {MIN_DOC_FREQ}\n")
        f.write(f"- ngram_range: {NGRAM_RANGE} | top-N cap: none (uncapped)\n")
        f.write(f"- Vocabulary size: {len(vocab):,} "
                f"(Unigrams: {n_uni:,}, Bigrams: {n_bi:,})\n")
        f.write(f"- Output: `{OUT_VOCAB_CSV.relative_to(REPO_ROOT)}`\n")
    print(f"  Manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
