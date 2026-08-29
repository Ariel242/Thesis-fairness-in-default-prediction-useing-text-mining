# -*- coding: utf-8 -*-
"""
06_join_zip3_labels_to_loans.py
====================================
STAGE 6 (final stage) of the ZIP3 grouping pipeline (see
../../הסבר והצדקה מתודולוגית לאשכולות.docx, section 9, "היחידה הגאוגרפית והמשקולות").

WHAT THIS STAGE DOES
---------------------
Every earlier stage worked at the ZIP3 level (829 rows, one per area) -- groups were
NEVER computed on the loan-level data directly. Doc section 9 explains why this order
matters: if the tertile cutpoints were computed on all loans at once, a ZIP3 with 20,000
loans would dominate the cutpoints far more than a ZIP3 with 500 loans, which would make
the "ZIP3 grouping" implicitly a "loan-volume-weighted grouping" instead. Instead, each
ZIP3 is one observation when the groups are built (Stages 1-5); THIS stage is the only
place the ZIP3-level labels get attached to individual loans, via a simple `zip3` join.

This stage does not touch or resave the (large) loan-level structured dataset -- it
outputs a small, standalone label file keyed by loan `id` + `zip3`, meant to be merged
into whichever modeling/fairness script needs it later (`df.merge(labels, on="id")`),
matching the id-based-merge convention already used elsewhere in this repo (e.g.
FinBERT's embeddings are merged by `id`, never by row position).

Also reports JOIN COVERAGE: what fraction of loans matched a ZIP3 in the census file,
and which zip3 codes (if any) appear in the loan data but not in the census file (they
would end up with missing group labels and must be excluded or flagged downstream,
never silently dropped).

INPUT:  census/results/02_ses_disadvantage_index.csv   (continuous SESDisadvantage)
        census/results/03_ses_groups.csv                (SES_group)
        census/results/04_demographic_concentration_groups.csv (raw shares + *Concentration groups)
        data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_20260715_2119.csv
            (canonical loan-level file used throughout analysis/ FinBERT/ TF-IDF/ Dictionarys/ --
            only its `id` and `zip3` columns are read here)
OUTPUT: census/results/06_loan_level_zip3_group_labels.csv
        (id, zip3, SESDisadvantage, SES_group, each raw demographic share,
         each *Concentration group)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
BASE_DIR = CENSUS_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

SES_INDEX_CSV = RESULTS_DIR / "02_ses_disadvantage_index.csv"
SES_GROUPS_CSV = RESULTS_DIR / "03_ses_groups.csv"
DEMO_GROUPS_CSV = RESULTS_DIR / "04_demographic_concentration_groups.csv"
LOAN_CSV = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"

OUTPUT_CSV = RESULTS_DIR / "06_loan_level_zip3_group_labels.csv"


def build_zip3_label_table() -> pd.DataFrame:
    """One row per ZIP3: continuous SES score + SES group + every demographic group,
    ready to be joined onto loan-level data."""
    ses_index = pd.read_csv(SES_INDEX_CSV, dtype={"zip3": str})[["zip3", "SESDisadvantage"]]
    ses_groups = pd.read_csv(SES_GROUPS_CSV, dtype={"zip3": str})[["zip3", "SES_group"]]
    demo_groups = pd.read_csv(DEMO_GROUPS_CSV, dtype={"zip3": str})

    zip3_labels = ses_index.merge(ses_groups, on="zip3", validate="one_to_one")
    zip3_labels = zip3_labels.merge(demo_groups, on="zip3", validate="one_to_one")
    return zip3_labels


def main() -> None:
    zip3_labels = build_zip3_label_table()
    print(f"ZIP3-level label table: {len(zip3_labels)} ZIP3s, {len(zip3_labels.columns)} columns.")

    loans = pd.read_csv(LOAN_CSV, usecols=["id", "zip3"], dtype={"id": str, "zip3": str})
    print(f"Loan-level file: {len(loans)} loans.")

    merged = loans.merge(zip3_labels, on="zip3", how="left", validate="many_to_one")

    n_unmatched = merged["SES_group"].isna().sum()
    coverage = 1 - n_unmatched / len(merged)
    print(f"Join coverage: {coverage:.2%} of loans matched a ZIP3 group "
          f"({n_unmatched} unmatched loans).")
    if n_unmatched > 0:
        unmatched_zip3 = sorted(loans.loc[~loans["zip3"].isin(zip3_labels["zip3"]), "zip3"].unique())
        print(f"  Unmatched zip3 codes ({len(unmatched_zip3)}): {unmatched_zip3[:20]}"
              f"{' ...' if len(unmatched_zip3) > 20 else ''}")

    RESULTS_DIR.mkdir(exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
