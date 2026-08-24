# -*- coding: utf-8 -*-
"""
04_build_lexicon_features.py
==============================
Step 4 of the Distress Lexicon protocol (see
Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf, section 2,
"Primary & Secondary Feature Engineering on Loan Text").

Applies the FROZEN lexicon (Dictionarys/dictionaries/LC_Distress_Lexicon_v1.csv,
produced by 02_llm_zero_shot_annotate.py) UNCHANGED to the full corpus (all
rows, all walk-forward folds -- freezing applies to vocabulary *extraction*
only, not to feature application, per the PDF).

Uses the same desc-cleaning approach as 01_extract_frozen_vocabulary.py so
token matching is consistent between how the lexicon was built and how it's
applied here.

Primary features (PDF section 2):
  risk_word_count    -- total distress terms matched in description
  risk_word_density  -- risk_word_count / total word count
  risk_word_flag     -- 1 if risk_word_count >= 1 else 0

Sub-category features (PDF section 2, category -> feature name mapping):
  DEBT_PRESSURE          -> distress_debt_pressure_count
  LIQUIDITY_SHORTAGE     -> distress_liquidity_shortage_count
  SHOCK_EMERGENCY        -> distress_medical_life_shock_count
  HIGH_RISK_REFINANCING  -> distress_refinancing_burden_count

Output: Dictionarys/output/lexicon_features.csv (id + these 7 columns),
shaped like TF-IDF/output/text_stats.csv for downstream joining by id.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer

SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent
REPO_ROOT  = DICT_DIR.parent

INPUT_CSV = REPO_ROOT / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
LEXICON_CSV = DICT_DIR / "dictionaries" / "LC_Distress_Lexicon_v1.csv"
OUT_FEATURES_CSV = DICT_DIR / "output" / "lexicon_features.csv"

ID_COL   = "id"
DESC_COL = "desc"

CATEGORY_COUNT_COL = {
    "DEBT_PRESSURE": "distress_debt_pressure_count",
    "LIQUIDITY_SHORTAGE": "distress_liquidity_shortage_count",
    "SHOCK_EMERGENCY": "distress_medical_life_shock_count",
    "HIGH_RISK_REFINANCING": "distress_refinancing_burden_count",
}

PREFIX_RE = re.compile(r"^\s*borrower\s+added\s+on\s+\d{2}/\d{2}/\d{2}\s*>\s*", re.IGNORECASE)
_WNL = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    return _WNL.lemmatize(_WNL.lemmatize(token, "v"), "n")


def clean_desc(text) -> str:
    """Same cleaning approach as 01_extract_frozen_vocabulary.py's clean_desc."""
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


def count_matches(tokens: list[str], term_set: set[str], bigram_set: set[str]) -> int:
    n = sum(1 for t in tokens if t in term_set)
    if bigram_set:
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        n += sum(1 for b in bigrams if b in bigram_set)
    return n


def main() -> None:
    print("Loading lexicon...")
    lexicon = pd.read_csv(LEXICON_CSV)
    lexicon = lexicon[lexicon["is_distress"] == 1].copy()
    print(f"  Distress terms in lexicon: {len(lexicon):,} "
          f"(of {len(pd.read_csv(LEXICON_CSV)):,} total, incl. NEUTRAL_BENIGN)")

    # Per-category term sets, split into unigram/bigram for matching
    category_terms = {}
    for cat in CATEGORY_COUNT_COL:
        terms = set(lexicon.loc[lexicon["category"] == cat, "token"].astype(str))
        uni = {t for t in terms if " " not in t}
        bi = {t for t in terms if " " in t}
        category_terms[cat] = (uni, bi)
        print(f"  {cat}: {len(terms)} terms ({len(uni)} unigram, {len(bi)} bigram)")

    all_terms = set(lexicon["token"].astype(str))
    all_uni = {t for t in all_terms if " " not in t}
    all_bi = {t for t in all_terms if " " in t}

    print("\nLoading loan corpus...")
    df = pd.read_csv(INPUT_CSV, usecols=[ID_COL, DESC_COL], low_memory=False)
    print(f"  Rows: {len(df):,}")

    print("Cleaning + lemmatizing desc (full corpus)...")
    df["_clean"] = df[DESC_COL].apply(clean_desc)
    df["_tokens"] = df["_clean"].apply(lambda s: s.split() if s else [])
    df["_word_count"] = df["_tokens"].apply(len)

    print("Matching lexicon terms per loan...")
    df["risk_word_count"] = df["_tokens"].apply(lambda toks: count_matches(toks, all_uni, all_bi))
    safe_word_count = df["_word_count"].where(df["_word_count"] > 0, other=1)
    df["risk_word_density"] = (df["risk_word_count"] / safe_word_count).where(df["_word_count"] > 0, other=0.0)
    df["risk_word_flag"] = (df["risk_word_count"] >= 1).astype(int)

    for cat, col in CATEGORY_COUNT_COL.items():
        uni, bi = category_terms[cat]
        df[col] = df["_tokens"].apply(lambda toks: count_matches(toks, uni, bi))

    feature_cols = [ID_COL, "risk_word_count", "risk_word_density", "risk_word_flag"] + \
        list(CATEGORY_COUNT_COL.values())
    out = df[feature_cols].copy()

    OUT_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FEATURES_CSV, index=False)

    print(f"\nLoans with >=1 distress term: {out['risk_word_flag'].sum():,} / {len(out):,} "
          f"({out['risk_word_flag'].mean():.2%})")
    print(f"Output: {OUT_FEATURES_CSV}")


if __name__ == "__main__":
    main()
