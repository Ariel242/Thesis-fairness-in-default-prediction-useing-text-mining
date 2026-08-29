# -*- coding: utf-8 -*-
"""
02_build_ses_disadvantage_index.py
====================================
STAGE 2 of the ZIP3 grouping pipeline (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 2, "החלוקה הסוציו־אקונומית").

WHAT THIS STAGE DOES
---------------------
Builds the continuous Socioeconomic Disadvantage Index, SESDisadvantage_i, for every
ZIP3 (i). This is Stage 2 only -- turning the index into Low/Medium/High GROUPS is a
separate stage (03), kept apart so the continuous score itself is always inspectable
on its own, e.g. for the section-11.2 robustness check ("SES score חלופי").

METHOD (doc section 2.2-2.3)
-----------------------------
The 7 SES indicators are on incompatible scales (proportions vs. dollars) and some are
skewed, so raw values cannot simply be averaged. Instead, each indicator is converted to
a percentile rank P_ij in [0, 1] (pandas' `rank(pct=True)`: the value 1.0 means "the
single highest raw value among all 829 ZIP3s"). Each indicator is then re-oriented so
that a HIGHER value always means MORE disadvantage:

    D_ij = P_ij          for indicators where high = bad  (poverty, unemployment, uninsured)
    D_ij = 1 - P_ij       for indicators where high = good (education, homeownership,
                                                             income, home value)

The final index is the unweighted mean of the 7 D_ij values (doc section 2.4 explains
why equal weights, not a data-driven weighting, were chosen: no defensible theoretical
basis for unequal weights, and equal weights keep the index simple and reproducible --
the same reasoning the CDC/ATSDR Social Vulnerability Index uses).

INPUT:  census/results/01_census_data_validated.csv
OUTPUT: census/results/02_ses_disadvantage_index.csv
        (zip3, the 7 raw values, the 7 D_ij disadvantage scores, SESDisadvantage_i)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

INPUT_CSV = RESULTS_DIR / "01_census_data_validated.csv"
OUTPUT_CSV = RESULTS_DIR / "02_ses_disadvantage_index.csv"

# The K=7 SES indicators (doc, Table 0) and which direction counts as "disadvantage".
# "higher_is_worse"  -> D_ij = P_ij
# "higher_is_better" -> D_ij = 1 - P_ij
SES_INDICATORS = {
    "poverty_rate": "higher_is_worse",
    "unemployment_rate": "higher_is_worse",
    "share_bachelor_plus": "higher_is_better",
    "homeownership_rate": "higher_is_better",
    "uninsured_rate": "higher_is_worse",
    "median_income_approx": "higher_is_better",
    "median_home_value_approx": "higher_is_better",
}


def build_ses_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["zip3"] + list(SES_INDICATORS)].copy()

    disadvantage_cols = []
    for var, direction in SES_INDICATORS.items():
        percentile_rank = df[var].rank(pct=True)  # P_ij in [0, 1]
        d_col = f"D_{var}"
        out[d_col] = percentile_rank if direction == "higher_is_worse" else 1 - percentile_rank
        disadvantage_cols.append(d_col)

    out["SESDisadvantage"] = out[disadvantage_cols].mean(axis=1)
    return out


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"zip3": str})
    out = build_ses_index(df)

    print(f"Built SESDisadvantage for {len(out)} ZIP3s from {len(SES_INDICATORS)} indicators.")
    print(out["SESDisadvantage"].describe().round(4).to_string())

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
