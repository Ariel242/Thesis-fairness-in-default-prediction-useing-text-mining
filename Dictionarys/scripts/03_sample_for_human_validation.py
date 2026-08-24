# -*- coding: utf-8 -*-
"""
03_sample_for_human_validation.py
===================================
Step 3 (sampling half) of the Distress Lexicon protocol (see
Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf, section 2,
"Human-in-the-Loop Validation").

Draws a random stratified sample of 150-200 terms across the LLM-annotated
categories from Dictionarys/dictionaries/LC_Distress_Lexicon_v1.csv (produced
by 02_llm_zero_shot_annotate.py), for blind human expert annotation.

Two outputs are kept deliberately separate so Ariel's manual labeling stays
blind to the LLM's own labels:
  - results/human_validation_blind.csv  -- token + empty human_label column,
    for Ariel to fill in by hand.
  - results/human_validation_key.csv    -- token + llm_category, NOT to be
    looked at while labeling; used later by 05_compute_interrater_reliability.py
    (not yet built) to compute Cohen's kappa / confusion matrix / precision.

Usage:
  python 03_sample_for_human_validation.py [--n 175] [--seed 42]
"""

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent

LEXICON_CSV = DICT_DIR / "dictionaries" / "LC_Distress_Lexicon_v1.csv"
OUT_BLIND_CSV = DICT_DIR / "results" / "human_validation_blind.csv"
OUT_KEY_CSV   = DICT_DIR / "results" / "human_validation_key.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=175,
                         help="Total sample size across all categories (150-200 per the PDF).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (150 <= args.n <= 200):
        print(f"WARNING: --n={args.n} is outside the PDF's specified 150-200 range.")

    lexicon = pd.read_csv(LEXICON_CSV)
    n_categories = lexicon["category"].nunique()
    per_category = max(1, args.n // n_categories)

    print(f"Lexicon size: {len(lexicon):,} terms across {n_categories} categories")
    print(f"Target sample: {args.n} total (~{per_category}/category)")

    sampled_parts = []
    for cat, group in lexicon.groupby("category"):
        take = min(per_category, len(group))
        sampled_parts.append(group.sample(n=take, random_state=args.seed))
        print(f"  {cat}: sampled {take} / {len(group)} available")

    sample = pd.concat(sampled_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    blind = sample[["token"]].copy()
    blind["human_label"] = ""
    OUT_BLIND_CSV.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(OUT_BLIND_CSV, index=False, encoding="utf-8-sig")

    key = sample[["token", "category"]].rename(columns={"category": "llm_category"})
    key.to_csv(OUT_KEY_CSV, index=False, encoding="utf-8-sig")

    print(f"\nTotal sampled: {len(sample)}")
    print(f"Blind file (for Ariel to fill in by hand): {OUT_BLIND_CSV}")
    print(f"Key file (do not view before labeling):     {OUT_KEY_CSV}")


if __name__ == "__main__":
    main()
