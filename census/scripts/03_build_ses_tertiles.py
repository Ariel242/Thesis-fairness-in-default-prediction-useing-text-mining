# -*- coding: utf-8 -*-
"""
03_build_ses_tertiles.py
====================================
STAGE 3 of the ZIP3 grouping pipeline (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 2.5, "חלוקת מדד ה־SES ל־tertiles").

WHAT THIS STAGE DOES
---------------------
Converts the continuous SESDisadvantage score (Stage 2) into three groups -- Low /
Medium / High disadvantage -- using empirical tertile cutpoints (33.3rd and 66.7th
percentiles of SESDisadvantage across all 829 ZIP3s). Tertiles were chosen over an
arbitrary fixed threshold because the underlying concept ("disadvantaged area") has no
natural cutoff -- see doc section 2.5 for the full justification (no external threshold
needed, groups stay reasonably balanced in size, and it matches precedents in the
neighborhood-SES literature).

TIE HANDLING (doc section 10, "טיפול ב־ties בגבולות tertile")
-----------------------------------------------------------------
Because SESDisadvantage is built from proportions, it is possible for several ZIP3s to
land on the exact cutpoint value. The rule is: NEVER split ZIP3s with an identical score
across two groups just to force equal-sized thirds. This is implemented with simple
`<=` / `>` comparisons against the two cutpoints (not `pd.qcut`, which tries to force
equal bin sizes and errors or splits ties arbitrarily when there are many duplicate
values): every ZIP3 at exactly the low cutpoint ends up in "Low", every ZIP3 at exactly
the high cutpoint ends up in "Medium". Group sizes may therefore differ slightly from an
exact one-third split -- this is expected and reported below, not a bug.

INPUT:  census/results/02_ses_disadvantage_index.csv
OUTPUT: census/results/03_ses_groups.csv (zip3, SESDisadvantage, SES_group)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

INPUT_CSV = RESULTS_DIR / "02_ses_disadvantage_index.csv"
OUTPUT_CSV = RESULTS_DIR / "03_ses_groups.csv"


def tie_safe_tertiles(score: pd.Series) -> pd.Series:
    """Low / Medium / High from empirical 33.3rd / 66.7th percentiles, keeping every
    tied value in the same group (see module docstring, "TIE HANDLING")."""
    q1, q2 = np.percentile(score, [100 / 3, 200 / 3])
    return pd.Series(
        np.select([score <= q1, score <= q2], ["Low", "Medium"], default="High"),
        index=score.index,
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"zip3": str})

    df["SES_group"] = tie_safe_tertiles(df["SESDisadvantage"])

    print("SES group sizes (expected close to, but not necessarily exactly, 1/3 each):")
    print(df["SES_group"].value_counts().reindex(["Low", "Medium", "High"]).to_string())

    out = df[["zip3", "SESDisadvantage", "SES_group"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
