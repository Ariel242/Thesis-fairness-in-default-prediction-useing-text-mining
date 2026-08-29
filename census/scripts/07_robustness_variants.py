# -*- coding: utf-8 -*-
"""
07_robustness_variants.py
====================================
OPTIONAL sensitivity-check helpers (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 11, "Robustness checks").

NOT part of the main pipeline. run_pipeline.py does NOT call this script -- Stages
01-06 already produce the primary group definitions the fairness analysis should use.
This script exists so that, later, someone can check the main conclusions are not an
artifact of one specific modeling choice, per the doc's own explicit warning (section 8,
"מניעת בחירה תלוית תוצאה"): these alternate specifications must be defined ahead of
time and never chosen after looking at fairness results.

Each function below is independent and mirrors exactly one doc subsection:

  ses_quartiles(...)              -- doc 11.1: same SES index, 4 groups instead of 3
  ses_score_zscore(...)           -- doc 11.2: alternate SES index via standardized Z-scores
  ses_score_pca(...)              -- doc 11.2: alternate SES index via first PCA component
  high_vs_low_only(...)           -- doc 11.4: drop the Medium group for a sharper 2-group contrast

(Doc 11.3, "continuous concentration", needs no new code: the raw share_* columns are
already carried through Stage 4's output for exactly this purpose.)

Run this file directly for a demonstration that also cross-checks the two alternate SES
scores against the primary percentile-based one (Spearman correlation) -- but, as with
every other script in this pipeline, nothing runs automatically on import.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

VALIDATED_CSV = RESULTS_DIR / "01_census_data_validated.csv"
SES_INDEX_CSV = RESULTS_DIR / "02_ses_disadvantage_index.csv"

# Same 7 indicators and directions as 02_build_ses_disadvantage_index.py, duplicated
# here deliberately (see 04_build_demographic_concentration.py's docstring note on
# this repo's "each script owns its own copy" convention).
SES_INDICATORS = {
    "poverty_rate": "higher_is_worse",
    "unemployment_rate": "higher_is_worse",
    "share_bachelor_plus": "higher_is_better",
    "homeownership_rate": "higher_is_better",
    "uninsured_rate": "higher_is_worse",
    "median_income_approx": "higher_is_better",
    "median_home_value_approx": "higher_is_better",
}


# ------------------------------------------------------------------
# 11.1 -- SES quartiles instead of tertiles
# ------------------------------------------------------------------
def ses_quartiles(score: pd.Series) -> pd.Series:
    """Low / Lower-Middle / Upper-Middle / High, tie-safe (same method as the primary
    tertile split in 03_build_ses_tertiles.py, generalized to 4 groups)."""
    q1, q2, q3 = np.percentile(score, [25, 50, 75])
    return pd.Series(
        np.select(
            [score <= q1, score <= q2, score <= q3],
            ["Low", "Lower-Middle", "Upper-Middle"],
            default="High",
        ),
        index=score.index,
    )


# ------------------------------------------------------------------
# 11.2 -- alternate SES index: standardized Z-scores
# ------------------------------------------------------------------
def ses_score_zscore(df: pd.DataFrame) -> pd.Series:
    """Same 7 indicators and same disadvantage direction as the primary index, but
    combined via Z-scores (mean of standardized, direction-aligned values) instead of
    percentile ranks. Sensitive to outliers/skew in a way the primary index is not --
    that difference is the point of comparing the two."""
    z = pd.DataFrame(index=df.index)
    for var, direction in SES_INDICATORS.items():
        z_col = StandardScaler().fit_transform(df[[var]]).ravel()
        z[var] = -z_col if direction == "higher_is_better" else z_col
    return z.mean(axis=1)


# ------------------------------------------------------------------
# 11.2 -- alternate SES index: first principal component
# ------------------------------------------------------------------
def ses_score_pca(df: pd.DataFrame) -> pd.Series:
    """Same 7 direction-aligned Z-scored indicators, combined via PCA's first
    component instead of a simple mean -- captures whatever single axis explains the
    most shared variance across the 7 indicators, rather than weighting them equally."""
    z = pd.DataFrame(index=df.index)
    for var, direction in SES_INDICATORS.items():
        z_col = StandardScaler().fit_transform(df[[var]]).ravel()
        z[var] = -z_col if direction == "higher_is_better" else z_col

    pc1 = PCA(n_components=1, random_state=242).fit_transform(z).ravel()
    # PCA's sign is arbitrary -- flip it if needed so higher = more disadvantage,
    # using the primary percentile-based index as the reference direction.
    if np.corrcoef(pc1, z.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pd.Series(pc1, index=df.index)


# ------------------------------------------------------------------
# 11.4 -- High-vs-Low only (drop Medium for a sharper two-group contrast)
# ------------------------------------------------------------------
def high_vs_low_only(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Filters to just the Low and High rows of any Low/Medium/High grouping column
    (SES_group or any *Concentration column from Stage 4)."""
    return df[df[group_col].isin(["Low", "High"])].copy()


def main() -> None:
    validated = pd.read_csv(VALIDATED_CSV, dtype={"zip3": str})
    primary = pd.read_csv(SES_INDEX_CSV, dtype={"zip3": str})

    print("11.1 -- SES quartiles (group sizes):")
    quartiles = ses_quartiles(primary["SESDisadvantage"])
    print(quartiles.value_counts().reindex(["Low", "Lower-Middle", "Upper-Middle", "High"]).to_string())

    z_score = ses_score_zscore(validated)
    pca_score = ses_score_pca(validated)

    print("\n11.2 -- Spearman correlation of alternate SES scores vs. the primary "
          "percentile-based SESDisadvantage (should be high if the index is not an "
          "artifact of one specific scoring method):")
    print(f"  percentile-based vs. z-score : "
          f"{primary['SESDisadvantage'].corr(z_score, method='spearman'):.4f}")
    print(f"  percentile-based vs. PCA     : "
          f"{primary['SESDisadvantage'].corr(pca_score, method='spearman'):.4f}")


if __name__ == "__main__":
    main()
