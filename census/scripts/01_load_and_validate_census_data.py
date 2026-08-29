# -*- coding: utf-8 -*-
"""
01_load_and_validate_census_data.py
====================================
STAGE 1 of the ZIP3 grouping pipeline (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 9, "היחידה הגאוגרפית והמשקולות" -- the ZIP3, not the loan, is the unit of
analysis for group construction).

WHAT THIS STAGE DOES
---------------------
Loads the raw ACS ZIP3 census file, checks it is safe to build group definitions on
top of, and re-saves it with one fix applied: `zip3` forced to a zero-padded 3-character
string (a plain `pd.read_csv` silently reads "007" as the integer 7, which would break
every later join against the loan-level data's own zero-padded `zip3` column).

Nothing here computes any group or index yet -- this stage is purely "is the input
trustworthy", so every later stage can assume a clean starting point without repeating
these checks itself.

INPUT:  census/zip3_census_features_acs5_2012.csv
OUTPUT: census/results/01_census_data_validated.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent      # .../census/scripts
CENSUS_DIR = SCRIPT_DIR.parent                     # .../census
RESULTS_DIR = CENSUS_DIR / "results"

INPUT_CSV = CENSUS_DIR / "zip3_census_features_acs5_2012.csv"
OUTPUT_CSV = RESULTS_DIR / "01_census_data_validated.csv"

EXPECTED_N_ZIP3 = 829

# Every column that must be a proportion in [0, 1]. median_income_approx,
# median_home_value_approx (dollars) and total_pop / total_households (counts)
# are intentionally excluded from this check.
RATE_COLUMNS = [
    "share_white", "share_black", "share_asian",
    "share_white_nh", "share_black_nh", "share_asian_nh", "share_hispanic",
    "share_foreign_born", "share_non_english_hh", "share_ltd_english_hh",
    "poverty_rate", "unemployment_rate", "share_bachelor_plus",
    "homeownership_rate", "uninsured_rate",
]


def load_and_validate() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, dtype={"zip3": str})

    # -- structural checks -------------------------------------------------
    assert df["zip3"].str.len().eq(3).all(), \
        "Found zip3 codes that are not exactly 3 characters -- check zero-padding."
    assert not df["zip3"].duplicated().any(), "zip3 is expected to be a unique key."
    assert len(df) == EXPECTED_N_ZIP3, \
        f"Expected {EXPECTED_N_ZIP3} ZIP3 rows, got {len(df)}. Did the source file change?"

    # -- missing-value check -------------------------------------------------
    n_missing = df.isna().sum().sum()
    assert n_missing == 0, f"Found {n_missing} missing values -- expected a fully imputed file."

    # -- range check on rate/share columns ------------------------------------
    for col in RATE_COLUMNS:
        out_of_range = ~df[col].between(0, 1)
        assert not out_of_range.any(), f"Column '{col}' has values outside [0, 1]."

    return df


def main() -> None:
    print(f"Loading: {INPUT_CSV}")
    df = load_and_validate()
    print(f"  Validated {len(df)} ZIP3 rows, {len(df.columns)} columns, 0 missing values.")

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
