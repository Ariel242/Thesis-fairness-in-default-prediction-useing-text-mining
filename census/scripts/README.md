# ZIP3 grouping pipeline

Implements the methodology in `../הסבר והצדקה מתודולוגית לאשכולות.docx`: splits the
829 ZIP3 areas into two **separate, non-mixed** group systems --

- a **socioeconomic (SES)** dimension: one combined Low / Medium / High disadvantage score, and
- a **demographic** dimension: six independent Low / Medium / High concentration groups
  (Black, Hispanic, Asian, White, foreign-born, limited-English) --

so that fairness metrics can later be compared across areas without conflating "poorer
area" with "area with more of group X" (see doc section 1 for why that separation
matters, and section 4 for why this uses rule-based tertiles instead of K-Means).

**Not run yet.** These scripts were prepared but never executed against the real data --
run `python run_pipeline.py` (from this folder, or `python census/scripts/run_pipeline.py`
from the repo root) when ready.

## Stages

| # | Script | Reads | Writes | Doc section |
|---|--------|-------|--------|-------------|
| 1 | `01_load_and_validate_census_data.py` | `census/zip3_census_features_acs5_2012.csv` | `results/01_census_data_validated.csv` | 9 |
| 2 | `02_build_ses_disadvantage_index.py` | stage 1 | `results/02_ses_disadvantage_index.csv` | 2.2-2.3 |
| 3 | `03_build_ses_tertiles.py` | stage 2 | `results/03_ses_groups.csv` | 2.5, 10 |
| 4 | `04_build_demographic_concentration.py` | stage 1 | `results/04_demographic_concentration_groups.csv` | 3 |
| 5 | `05_balance_diagnostics.py` | stages 1, 3, 4 | `results/05_balance_diagnostics_{detail,summary}.csv` | 6 |
| 6 | `06_join_zip3_labels_to_loans.py` | stages 2, 3, 4 + the canonical loan CSV | `results/06_loan_level_zip3_group_labels.csv` | 9 |

`run_pipeline.py` runs stages 1-6 in order with one command.

`07_robustness_variants.py` is **not** part of the default run. It holds optional,
pre-registered sensitivity checks (SES quartiles instead of tertiles, an alternate
Z-score/PCA-based SES index, a High-vs-Low-only subset) for later, manual use -- per
doc section 8, these must be decided in advance and never chosen after seeing fairness
results.

## Reading the outputs in order

Each stage writes a small, human-inspectable CSV before the next stage reads it, so you
can open any intermediate file to see exactly what one step changed:

1. `01_census_data_validated.csv` -- same 829 rows as the source file, just with `zip3`
   guaranteed to be a clean 3-character string.
2. `02_ses_disadvantage_index.csv` -- adds the 7 percentile-based `D_*` columns and the
   final continuous `SESDisadvantage` score (0-1, higher = more disadvantaged).
3. `03_ses_groups.csv` -- adds `SES_group` (Low/Medium/High).
4. `04_demographic_concentration_groups.csv` -- adds 6 independent `*Concentration`
   columns (Low/Medium/High each).
5. `05_balance_diagnostics_summary.csv` -- one row per grouping x pairwise comparison,
   with mean/max |SMD| across the covariates on the "other" dimension. `|SMD| > 0.1` is
   a common rule-of-thumb flag for meaningful imbalance.
6. `06_loan_level_zip3_group_labels.csv` -- the final artifact: `id`, `zip3`, and every
   group/score column above, one row per loan, ready to `merge(..., on="id")` into any
   fairness-evaluation script.

## Key design choices carried over from the doc (so the code isn't a black box)

- **Tertile cutpoints are computed on the 829 ZIP3s, never on the loans directly** --
  otherwise a high-loan-volume ZIP3 would dominate the cutpoints (doc section 9).
- **Ties at a cutpoint are never split apart** just to force equal-sized thirds (doc
  section 10) -- implemented with `<=`/`>` comparisons against the empirical
  percentiles, not `pd.qcut`.
- **SES is one combined index; demographics are not.** There's a natural
  advantage&harr;disadvantage axis for SES but no equivalent single axis across racial/
  ethnic groups, so collapsing them into one score would be an arbitrary, unjustified
  choice (doc section 3.1).
- **Group construction never looks at loan outcomes, predictions, or fairness metrics**
  (doc section 8) -- every cutpoint here is a function of the census file alone.
