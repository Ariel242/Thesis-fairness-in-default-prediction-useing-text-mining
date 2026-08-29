# -*- coding: utf-8 -*-
"""
04_build_demographic_concentration.py
====================================
STAGE 4 of the ZIP3 grouping pipeline (see ../../הסבר והצדקה מתודולוגית לאשכולות.docx,
section 3, "החלוקה הדמוגרפית").

WHAT THIS STAGE DOES
---------------------
Unlike SES (Stage 2-3), the demographic dimension is deliberately NOT collapsed into one
combined index. Doc section 3.1 explains why: there is no natural "advantage <->
disadvantage" axis between racial/ethnic groups the way there is for SES, so treating,
say, "30% Black + 20% Hispanic" as equivalent to "10% Black + 40% Hispanic" would be an
arbitrary, unjustified scoring choice.

Instead, each demographic share gets its OWN independent Low/Medium/High tertile split
(same tie-safe method as Stage 3, copied here verbatim rather than imported -- see the
note at the bottom of this docstring). A single ZIP3 therefore ends up with several
labels at once, e.g. "High Black concentration" + "Low Hispanic concentration" + "Low
Asian concentration" -- the point (doc section 3.3) is to preserve this multi-dimensional
picture rather than compress it into one cluster label.

VARIABLE CHOICES (doc sections 3.2, 3.6)
-------------------------------------------
- Race/ethnicity uses the non-Hispanic ("_nh") variants plus share_hispanic, NOT the
  "any ethnicity" share_black/share_asian/share_white -- this avoids the conceptual
  overlap of counting someone as both e.g. "Hispanic" and "Black" under two different
  variables (doc section 3.2). share_white_nh is included as a supplementary/secondary
  analysis, per doc Table 1.
- share_foreign_born and share_ltd_english_hh are treated as their own demographic
  contextual dimensions, NOT folded into the SES index (doc section 3.6: being
  foreign-born or speaking limited English is not the same construct as economic
  disadvantage) and NOT combined arithmetically with the race/ethnicity shares either.

WHY NOT K-MEANS: see doc section 4 in full -- in short, the research question here is
"does the model behave differently along a pre-specified, theoretically defined
dimension", not "what clusters exist empirically". Rule-based tertiles keep every
comparison directly interpretable ("high vs. low Black-concentration ZIP3s"), whereas a
K-Means cluster label mixes dimensions in a way that would be much harder to interpret
and is sensitive to scaling/initialization choices that have nothing to do with the
research question.

INPUT:  census/results/01_census_data_validated.csv
OUTPUT: census/results/04_demographic_concentration_groups.csv
        (zip3, each raw share, each *Concentration Low/Medium/High group)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CENSUS_DIR / "results"

INPUT_CSV = RESULTS_DIR / "01_census_data_validated.csv"
OUTPUT_CSV = RESULTS_DIR / "04_demographic_concentration_groups.csv"

# group label -> source share column (doc section 3.2 / Table 1)
DEMOGRAPHIC_DIMENSIONS = {
    "BlackConcentration": "share_black_nh",
    "HispanicConcentration": "share_hispanic",
    "AsianConcentration": "share_asian_nh",
    "WhiteConcentration": "share_white_nh",          # supplementary, per doc Table 1
    "ForeignBornConcentration": "share_foreign_born",
    "LimitedEnglishConcentration": "share_ltd_english_hh",
}


def tie_safe_tertiles(score: pd.Series) -> pd.Series:
    """Identical logic to 03_build_ses_tertiles.py's own copy -- duplicated
    deliberately, matching this repo's established convention (see e.g.
    Dictionarys/scripts/07_lexicon_structured_ablation.py's DeLong test) of each
    pipeline script owning its own copy of small shared logic rather than importing
    across scripts."""
    q1, q2 = np.percentile(score, [100 / 3, 200 / 3])
    return pd.Series(
        np.select([score <= q1, score <= q2], ["Low", "Medium"], default="High"),
        index=score.index,
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"zip3": str})

    out = df[["zip3"] + list(DEMOGRAPHIC_DIMENSIONS.values())].copy()

    print("Demographic concentration group sizes:")
    for group_label, share_col in DEMOGRAPHIC_DIMENSIONS.items():
        out[group_label] = tie_safe_tertiles(df[share_col])
        counts = out[group_label].value_counts().reindex(["Low", "Medium", "High"])
        print(f"  {group_label} (from {share_col}): " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
