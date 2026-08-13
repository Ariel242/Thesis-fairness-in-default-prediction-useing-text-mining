# -*- coding: utf-8 -*-
"""
extract_vocabulary.py
=====================
Extracts a filtered unigram/bigram vocabulary from the DESC column of the
RAW LendingClub dataset (the full original file, before any filtering).

Output: vocabulary_frequencies.csv with columns Term / Type / Document_Frequency,
where Document_Frequency is the number of DISTINCT loan descriptions containing
the term (document-level occurrence, not total word counts — enforced via
CountVectorizer(binary=True)).
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Domain stopwords on top of the standard English list. Note: a stopword is
# removed at the token level, so every bigram containing it drops too
# (e.g. "br" + "borrower" kills the "br borrower" boilerplate bigram).
# Texts are lemmatized BEFORE vectorization, so "pay" here also covers
# paying/pays/paid and "card" covers cards.
CUSTOM_STOP_WORDS = {"loan", "pay", "debt", "br", "borrower"}

_WNL = WordNetLemmatizer()
_TOKEN_RE = re.compile(r"[a-z]+")


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    # verb pass then noun pass unifies both inflection families
    # (paying/paid -> pay, cards -> card, payments -> payment)
    return _WNL.lemmatize(_WNL.lemmatize(token, "v"), "n")


def lemmatize_text(text: str) -> str:
    return " ".join(_lemma(t) for t in _TOKEN_RE.findall(text))

SCRIPT_DIR = Path(__file__).resolve().parent             # .../F-TM-CR/Dictionarys/scripts
REPO_ROOT  = SCRIPT_DIR.parent.parent                     # .../F-TM-CR
INPUT_CSV  = REPO_ROOT / "data" / "accepted_2007_to_2018Q4.csv"

OUTPUT_DIR = SCRIPT_DIR.parent / "dictionaries"           # .../F-TM-CR/Dictionarys/dictionaries
OUTPUT_CSV = OUTPUT_DIR / "vocabulary_frequencies_lending_club.csv"

TARGET_COLUMN = "DESC"  # matched case-insensitively (the raw file uses 'desc')


def resolve_desc_column(csv_path: Path) -> str:
    """Return the actual DESC column name in the file, or raise a clear error."""
    header = pd.read_csv(csv_path, nrows=0)
    if TARGET_COLUMN in header.columns:
        return TARGET_COLUMN
    # case-insensitive fallback (the raw LendingClub file names it 'desc')
    matches = [c for c in header.columns if c.strip().lower() == TARGET_COLUMN.lower()]
    if matches:
        return matches[0]
    raise KeyError(
        f"Column '{TARGET_COLUMN}' not found in {csv_path.name}. "
        f"Available columns include: {list(header.columns)[:10]} ..."
    )


def load_descriptions(csv_path: Path) -> pd.Series:
    """Load only the DESC column (the raw file is ~1.6GB) and clean it."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    desc_col = resolve_desc_column(csv_path)
    df = pd.read_csv(csv_path, usecols=[desc_col], low_memory=False)

    texts = (
        df[desc_col]
        .dropna()             # keep only rows where DESC is not missing
        .astype(str)          # ensure string type
        .str.lower()          # lowercase everything
        .str.strip()
    )
    texts = texts[texts != ""]  # drop empty / whitespace-only descriptions

    if texts.empty:
        raise ValueError("No valid (non-empty) DESC texts remained after filtering.")
    return texts


def extract_vocabulary(texts: pd.Series) -> pd.DataFrame:
    """Fit CountVectorizer and return a Term/Type/Document_Frequency DataFrame."""
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words=list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)),  # standard English + domain stopwords
        min_df=50,               # term must appear in >= 50 distinct descriptions
        max_df=0.70,             # drop terms present in > 70% of descriptions (raised from 0.50: lemmatization merges inflections, pushing core domain concepts like card/credit card past 50%)
        binary=True,             # document-level occurrence, not term counts
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # letters-only tokens of 3+ letters: numbers, mixed tokens (24k) and 1-2 letter words never enter
    )
    try:
        doc_term = vectorizer.fit_transform(texts)
    except ValueError as exc:
        # raised by sklearn when min_df/max_df leave an empty vocabulary
        raise ValueError(
            "No terms remained after applying min_df=50 / max_df=0.50."
        ) from exc

    terms = vectorizer.get_feature_names_out()
    # column sums of a binary doc-term matrix = number of documents per term
    doc_freq = doc_term.sum(axis=0).A1.astype(int)

    vocab = pd.DataFrame({
        "Term": terms,
        "Type": ["Bigram" if " " in t else "Unigram" for t in terms],
        "Document_Frequency": doc_freq,
    })
    return vocab.sort_values(
        by=["Document_Frequency", "Term"],
        ascending=[False, True],
    ).reset_index(drop=True)


def main() -> None:
    try:
        texts = load_descriptions(INPUT_CSV)
        print("Lemmatizing (NLTK WordNet, verb+noun passes)...")
        texts = texts.map(lemmatize_text)
        vocab = extract_vocabulary(texts)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    n_uni = int((vocab["Type"] == "Unigram").sum())
    n_bi = int((vocab["Type"] == "Bigram").sum())
    print(f"Valid loan descriptions processed: {len(texts):,}")
    print(f"Unigrams extracted:                {n_uni:,}")
    print(f"Bigrams extracted:                 {n_bi:,}")
    print(f"Total vocabulary size:             {len(vocab):,}")
    print(f"Output file:                       {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
