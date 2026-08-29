# -*- coding: utf-8 -*-
"""
05_balance_diagnostics.py
====================================
STAGE 5 of the ZIP3 grouping pipeline (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 6, "בדיקת balance בין הקבוצות").

WHAT THIS STAGE DOES
---------------------
Demographic composition and SES are correlated in reality (doc section 5: residential
segregation means "High Black concentration" ZIP3s are, empirically, often also
lower-SES). Stages 3 and 4 deliberately defined every group using ONLY its own
dimension ("Define first, balance second", doc section 5) -- this stage is the
"balance second" half: it measures, but does NOT correct, how different the groups
turn out to be on the OTHER dimension.

For every demographic grouping (Black/Hispanic/Asian/White/ForeignBorn/LimitedEnglish
Concentration), this reports how its Low/Medium/High groups differ on the 7 SES
covariates. Symmetrically, for the SES grouping, it reports how Low/Medium/High
disadvantage groups differ on the demographic shares. This is purely descriptive
diagnostics -- any later "does the gap survive after balancing" analysis (doc section 7,
matching/reweighting) is out of scope for this stage and is a downstream sensitivity
check, not something this pipeline runs automatically.

METRIC: Standardized Mean Difference (doc section 6)
--------------------------------------------------------
    SMD(A, B) = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)

Unlike a p-value, SMD is not affected by sample size -- it expresses the group
difference in standard-deviation units, which is why it is the standard covariate-balance
metric in the matching/causal-inference literature (Austin 2009, cited in the doc's
methodological sources). This stage computes it for all 3 pairwise comparisons
(Low-Medium, Low-High, Medium-High) per covariate per grouping variable.

INPUT:  census/results/01_census_data_validated.csv
        census/results/03_ses_groups.csv
        census/results/04_demographic_concentration_groups.csv
OUTPUT: census/results/05_balance_diagnostics_detail.csv   (one row per grouping x comparison x covariate)
        census/results/05_balance_diagnostics_summary.csv  (one row per grouping x comparison: mean/max |SMD|)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from itertools import combinations
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

VALIDATED_CSV = RESULTS_DIR / "01_census_data_validated.csv"
SES_GROUPS_CSV = RESULTS_DIR / "03_ses_groups.csv"
DEMO_GROUPS_CSV = RESULTS_DIR / "04_demographic_concentration_groups.csv"

DETAIL_CSV = RESULTS_DIR / "05_balance_diagnostics_detail.csv"
SUMMARY_CSV = RESULTS_DIR / "05_balance_diagnostics_summary.csv"

GROUP_LEVELS = ["Low", "Medium", "High"]

# The 7 SES covariates checked when balancing a DEMOGRAPHIC grouping (doc section 6).
SES_COVARIATES = [
    "poverty_rate", "unemployment_rate", "share_bachelor_plus",
    "homeownership_rate", "uninsured_rate",
    "median_income_approx", "median_home_value_approx",
]

# The demographic covariates checked when balancing the SES grouping (symmetric check).
DEMOGRAPHIC_COVARIATES = [
    "share_black_nh", "share_hispanic", "share_asian_nh",
    "share_white_nh", "share_foreign_born", "share_ltd_english_hh",
]

DEMOGRAPHIC_GROUP_COLUMNS = [
    "BlackConcentration", "HispanicConcentration", "AsianConcentration",
    "WhiteConcentration", "ForeignBornConcentration", "LimitedEnglishConcentration",
]


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized Mean Difference between two groups on one covariate."""
    pooled_sd = ((a.var() + b.var()) / 2) ** 0.5
    if pooled_sd == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def balance_table(df: pd.DataFrame, group_col: str, covariates: list) -> pd.DataFrame:
    """All pairwise (Low-Medium, Low-High, Medium-High) SMDs for one grouping variable,
    across every covariate in `covariates`."""
    rows = []
    for level_a, level_b in combinations(GROUP_LEVELS, 2):
        sub_a = df[df[group_col] == level_a]
        sub_b = df[df[group_col] == level_b]
        for cov in covariates:
            rows.append({
                "grouping_variable": group_col,
                "comparison": f"{level_a}-{level_b}",
                "covariate": cov,
                "mean_A": round(sub_a[cov].mean(), 4),
                "mean_B": round(sub_b[cov].mean(), 4),
                "SMD": round(smd(sub_a[cov], sub_b[cov]), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    validated = pd.read_csv(VALIDATED_CSV, dtype={"zip3": str})
    ses_groups = pd.read_csv(SES_GROUPS_CSV, dtype={"zip3": str})
    demo_groups = pd.read_csv(DEMO_GROUPS_CSV, dtype={"zip3": str})

    df = validated.merge(ses_groups[["zip3", "SES_group"]], on="zip3", validate="one_to_one")
    df = df.merge(demo_groups[["zip3"] + DEMOGRAPHIC_GROUP_COLUMNS], on="zip3", validate="one_to_one")

    detail_tables = []

    # -- how balanced are the demographic groupings on SES? -----------------------
    for group_col in DEMOGRAPHIC_GROUP_COLUMNS:
        detail_tables.append(balance_table(df, group_col, SES_COVARIATES))

    # -- how balanced is the SES grouping on demographics? (symmetric check) -------
    detail_tables.append(balance_table(df, "SES_group", DEMOGRAPHIC_COVARIATES))

    detail = pd.concat(detail_tables, ignore_index=True)

    summary = (
        detail.groupby(["grouping_variable", "comparison"])["SMD"]
        .agg(mean_abs_SMD=lambda s: s.abs().mean(), max_abs_SMD=lambda s: s.abs().max())
        .round(4)
        .reset_index()
    )

    print("Balance summary (mean / max |SMD| per grouping x comparison; "
          "as a rule of thumb, |SMD| > 0.1 is commonly flagged as meaningful imbalance):")
    print(summary.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {DETAIL_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
