# Code reference — what each script actually does

This is a precise, code-level companion to `README.md` (which maps stages to the
methodology doc). Where `README.md` explains *why* each design choice was made, this
file documents *exactly* what each script computes: every input/output column, every
formula as implemented, every constant and function signature. If the code and this
file ever disagree, the code is correct and this file is stale — update it.

All file paths below are relative to `census/`. All `zip3` columns, everywhere, are
3-character strings (`dtype={"zip3": str}` on every read).

---

## `01_load_and_validate_census_data.py`

**Reads:** `zip3_census_features_acs5_2012.csv` (829 rows, 20 columns, as delivered by
`ZIP3_code_API.ipynb`).

**Function `load_and_validate() -> pd.DataFrame`** — loads the CSV with `zip3` forced to
`str`, then runs four `assert` checks (any failure stops the script, nothing is written):

1. `df["zip3"].str.len().eq(3).all()` — every zip3 is exactly 3 characters.
2. `not df["zip3"].duplicated().any()` — zip3 is a unique key.
3. `len(df) == EXPECTED_N_ZIP3` where `EXPECTED_N_ZIP3 = 829`.
4. `df.isna().sum().sum() == 0` — no missing values anywhere.
5. For every column in `RATE_COLUMNS` (the 15 proportion columns — every `share_*`
   column plus `poverty_rate`, `unemployment_rate`, `share_bachelor_plus`,
   `homeownership_rate`, `uninsured_rate`): `df[col].between(0, 1)` must hold for all
   rows. `median_income_approx`, `median_home_value_approx` (dollars) and `total_pop`,
   `total_households` (counts) are deliberately excluded from this check.

**Writes:** `results/01_census_data_validated.csv` — same 829 rows, same 20 columns as
the input, unchanged except `zip3`'s dtype is now guaranteed to be a zero-padded string
in the saved CSV too.

---

## `02_build_ses_disadvantage_index.py`

**Reads:** `results/01_census_data_validated.csv`.

**Constant `SES_INDICATORS`** (dict, 7 entries) — variable name -> direction:

| Variable | Direction |
|---|---|
| `poverty_rate` | `higher_is_worse` |
| `unemployment_rate` | `higher_is_worse` |
| `share_bachelor_plus` | `higher_is_better` |
| `homeownership_rate` | `higher_is_better` |
| `uninsured_rate` | `higher_is_worse` |
| `median_income_approx` | `higher_is_better` |
| `median_home_value_approx` | `higher_is_better` |

**Function `build_ses_index(df) -> pd.DataFrame`:**

For each variable `var` above:

```python
percentile_rank = df[var].rank(pct=True)   # pandas default: average rank method, ascending
D_col = percentile_rank                     if direction == "higher_is_worse"
D_col = 1 - percentile_rank                 if direction == "higher_is_better"
```

`rank(pct=True)` returns each row's fractional rank in `(0, 1]` (the single highest raw
value gets exactly `1.0`; ties get pandas' default *average* rank, so tied raw values
get an identical, averaged percentile — this is a different tie rule from the group-level
tie handling in scripts 03/04, which is deliberate: percentile ranks are a continuous
score, ties only need to matter once the score is cut into discrete groups).

Each `D_col` is stored as its own output column named `D_<var>`, e.g. `D_poverty_rate`.

```python
SESDisadvantage = mean(D_poverty_rate, D_unemployment_rate, D_share_bachelor_plus,
                        D_homeownership_rate, D_uninsured_rate,
                        D_median_income_approx, D_median_home_value_approx)
```
i.e. an unweighted `row-wise .mean()` across exactly the 7 `D_*` columns (`K = 7`,
matching the doc's `SESDisadvantage_i = (1/K) * sum(D_ij)`).

**Writes:** `results/02_ses_disadvantage_index.csv` with columns:
`zip3`, the 7 raw SES variables (unchanged), the 7 `D_*` columns, `SESDisadvantage`
(float in `[0, 1]`, higher = more disadvantaged).

---

## `03_build_ses_tertiles.py`

**Reads:** `results/02_ses_disadvantage_index.csv`.

**Function `tie_safe_tertiles(score: pd.Series) -> pd.Series`:**

```python
q1, q2 = np.percentile(score, [100/3, 200/3])     # empirical 33.33rd / 66.67th percentile
group = np.select(
    [score <= q1, score <= q2],
    ["Low", "Medium"],
    default="High",
)
```

Applied to the `SESDisadvantage` column. Note the comparison is `<=` on both cutpoints:
any ZIP3 with a score exactly equal to `q1` is placed in `Low` (not split into `Low`
and `Medium`); any ZIP3 exactly equal to `q2` is placed in `Medium`. This is the one
piece of logic duplicated verbatim in `04_build_demographic_concentration.py`.

**Writes:** `results/03_ses_groups.csv` with columns: `zip3`, `SESDisadvantage`,
`SES_group` (one of `"Low"`, `"Medium"`, `"High"`).

Console output also prints `df["SES_group"].value_counts()` — on the real data (verified
by direct inspection, not by running this script) this comes out to **277 / 276 / 276**
(Low/Medium/High).

---

## `04_build_demographic_concentration.py`

**Reads:** `results/01_census_data_validated.csv` (raw shares, not the SES-index file —
this stage doesn't need `SESDisadvantage` at all).

**Constant `DEMOGRAPHIC_DIMENSIONS`** (dict, 6 entries) — output group-column name ->
source share column:

| Output column | Source column |
|---|---|
| `BlackConcentration` | `share_black_nh` |
| `HispanicConcentration` | `share_hispanic` |
| `AsianConcentration` | `share_asian_nh` |
| `WhiteConcentration` | `share_white_nh` |
| `ForeignBornConcentration` | `share_foreign_born` |
| `LimitedEnglishConcentration` | `share_ltd_english_hh` |

Note: `share_black`, `share_asian`, `share_white` (the "any ethnicity", non-`_nh`
variants) and `share_non_english_hh` are read into the output (carried through from
stage 1) but are **not** turned into a group column — only the 6 columns above get a
`*Concentration` label. `total_pop` and `total_households` are not read by this script
at all.

**Function `tie_safe_tertiles(score)`** — byte-for-byte identical logic to script 03's
version (independently defined here, not imported — see the docstring's note on why).

For each of the 6 dimensions, independently: `out[group_label] = tie_safe_tertiles(df[share_col])`.
Each of the 6 tertile splits uses its own `q1`/`q2` cutpoints computed only from that one
column — there is no interaction between, e.g., the Black-concentration cutpoints and
the Hispanic-concentration cutpoints.

**Writes:** `results/04_demographic_concentration_groups.csv` with columns: `zip3`, the
6 raw source share columns, and the 6 `*Concentration` group columns.

Verified group sizes on the real data (direct inspection, not a pipeline run):

| Column | Low | Medium | High |
|---|---|---|---|
| `BlackConcentration` | 277 | 276 | 276 |
| `HispanicConcentration` | 277 | 276 | 276 |
| `AsianConcentration` | 280 | 273 | 276 |
| `WhiteConcentration` | 277 | 276 | 276 |
| `ForeignBornConcentration` | 277 | 276 | 276 |
| `LimitedEnglishConcentration` | 280 | 273 | 276 |

(`AsianConcentration` and `LimitedEnglishConcentration` have 4 tied ZIP3s sitting exactly
on the low cutpoint, which is why their `Low` group has 3-4 more members than the rest —
this is the tie rule in action, not an error.)

---

## `05_balance_diagnostics.py`

**Reads:** `results/01_census_data_validated.csv` (raw covariate values), `results/03_ses_groups.csv`
(`SES_group`), `results/04_demographic_concentration_groups.csv` (the 6 `*Concentration` columns).
Merged on `zip3` with `validate="one_to_one"` for both merges.

**Constants:**
- `GROUP_LEVELS = ["Low", "Medium", "High"]`
- `SES_COVARIATES` — the same 7 variable names as script 02's `SES_INDICATORS` keys
  (raw values this time, not the `D_*` disadvantage-direction versions).
- `DEMOGRAPHIC_COVARIATES = ["share_black_nh", "share_hispanic", "share_asian_nh", "share_white_nh", "share_foreign_born", "share_ltd_english_hh"]`
  (raw shares, same 6 as script 04's sources).
- `DEMOGRAPHIC_GROUP_COLUMNS` — the 6 `*Concentration` column names from script 04.

**Function `smd(a: pd.Series, b: pd.Series) -> float`:**

```python
pooled_sd = sqrt((a.var() + b.var()) / 2)
return 0.0 if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd
```

`a.var()` / `b.var()` use pandas' default (`ddof=1`, sample variance). The zero-guard
returns `0.0` rather than raising or dividing to produce `inf`/`nan`.

**Function `balance_table(df, group_col, covariates) -> pd.DataFrame`:**

For every one of the 3 pairs in `itertools.combinations(["Low","Medium","High"], 2)`
(`Low`-`Medium`, `Low`-`High`, `Medium`-`High`) and every covariate in the given list,
appends one row: `{grouping_variable, comparison, covariate, mean_A, mean_B, SMD}`
(means and SMD rounded to 4 decimals).

**`main()` builds two kinds of tables and concatenates them:**

1. For each of the 6 `DEMOGRAPHIC_GROUP_COLUMNS`: `balance_table(df, group_col, SES_COVARIATES)`
   — "are the demographic groups balanced on SES?"
2. One more call: `balance_table(df, "SES_group", DEMOGRAPHIC_COVARIATES)` — "is the SES
   grouping balanced on demographics?" (the symmetric direction).

So the detail table has `(6 demographic groupings + 1 SES grouping) x 3 comparisons x
(7 or 6 covariates)` rows = `6*3*7 + 1*3*6 = 126 + 18 = 144` rows total.

**Summary:** `detail.groupby(["grouping_variable","comparison"])["SMD"].agg(mean_abs_SMD=lambda s: s.abs().mean(), max_abs_SMD=lambda s: s.abs().max())`,
rounded to 4 decimals — one row per `grouping_variable x comparison` (21 rows: 7
groupings x 3 comparisons).

**Writes:** `results/05_balance_diagnostics_detail.csv` (144 rows) and
`results/05_balance_diagnostics_summary.csv` (21 rows). Console output prints the
summary table with a note that `|SMD| > 0.1` is a common imbalance-flag threshold (this
threshold is only mentioned in the printed message — it does not filter or alter any
saved row).

---

## `06_join_zip3_labels_to_loans.py`

**Reads:**
- `results/02_ses_disadvantage_index.csv` -> keeps only `zip3`, `SESDisadvantage`.
- `results/03_ses_groups.csv` -> keeps only `zip3`, `SES_group`.
- `results/04_demographic_concentration_groups.csv` -> all columns (6 raw shares + 6
  `*Concentration` columns + `zip3`).
- `../data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_20260715_2119.csv`
  (the canonical loan-level file used throughout `analysis/`, `FinBERT/`, `TF-IDF/`,
  `Dictionarys/`) -> reads **only** the `id` and `zip3` columns (`usecols=["id","zip3"]`),
  both forced to `str`.

**Function `build_zip3_label_table() -> pd.DataFrame`:** two `merge(..., on="zip3", validate="one_to_one")`
calls chaining the three ZIP3-level files above into one 829-row table.

**`main()`:**
```python
merged = loans.merge(zip3_labels, on="zip3", how="left", validate="many_to_one")
```
A `LEFT` join — every loan is kept even if its `zip3` has no match (those rows get `NaN`
in every group/score column). Coverage is computed as
`1 - merged["SES_group"].isna().sum() / len(merged)` and printed, along with up to the
first 20 distinct unmatched `zip3` codes (sorted) if any exist.

**Writes:** `results/06_loan_level_zip3_group_labels.csv` — one row per loan, columns:
`id`, `zip3`, `SESDisadvantage`, `SES_group`, the 6 raw demographic shares, the 6
`*Concentration` columns. This is the file meant to be merged (`on="id"`) into any later
fairness-evaluation script.

---

## `07_robustness_variants.py` (not called by `run_pipeline.py`)

**Reads (only when run directly, in its own `main()`):** `results/01_census_data_validated.csv`,
`results/02_ses_disadvantage_index.csv`.

**Function `ses_quartiles(score) -> pd.Series`:** same tie-safe pattern as script 03's
`tie_safe_tertiles`, generalized to 4 groups via `np.percentile(score, [25, 50, 75])` and
`np.select` over 3 conditions, labels `"Low"`, `"Lower-Middle"`, `"Upper-Middle"`,
default `"High"`.

**Function `ses_score_zscore(df) -> pd.Series`:** for each of the same 7
`SES_INDICATORS` (dict duplicated from script 02), fits `sklearn.preprocessing.StandardScaler()`
on that single column, negates it if `direction == "higher_is_better"` (so the sign
convention matches the primary index: higher output = more disadvantage), then returns
the row-wise mean across the 7 standardized-and-aligned columns.

**Function `ses_score_pca(df) -> pd.Series`:** builds the same 7 direction-aligned
Z-score columns as `ses_score_zscore`, then `sklearn.decomposition.PCA(n_components=1, random_state=242).fit_transform(...)`
on that 829x7 matrix. Sign is arbitrary in PCA, so the result is flipped
(`pc1 = -pc1`) whenever `np.corrcoef(pc1, z.mean(axis=1))[0,1] < 0`, i.e. whenever the
raw first component would be negatively correlated with the simple Z-score mean —
guaranteeing higher output always means more disadvantage, consistent with every other
score in this pipeline.

**Function `high_vs_low_only(df, group_col) -> pd.DataFrame`:** `df[df[group_col].isin(["Low","High"])].copy()`
— works on `SES_group` or any of the 6 `*Concentration` columns.

**`main()`** (only runs if this file is executed directly) demonstrates all of the above
and prints the Spearman correlation of `SESDisadvantage` (primary) against both
`ses_score_zscore` and `ses_score_pca`, as the doc's section 11.2 robustness check.

---

## `run_pipeline.py`

**Constant `STAGES`** — the 6 stage-script names, in order, as strings (`07` is not in
this list).

**Function `load_stage_module(stage_name) -> module`:** loads e.g.
`03_build_ses_tertiles.py` via `importlib.util.spec_from_file_location` +
`spec.loader.exec_module(...)`. This is necessary (rather than a normal `import`)
because Python module names can't start with a digit — a literal `import 03_foo` is a
`SyntaxError`. Loading by file path sidesteps that, and since `exec_module` runs the
file with `__name__` set to `stage_name` (not `"__main__"`), each stage script's own
`if __name__ == "__main__": main()` guard does NOT double-fire — `run_pipeline.py` calls
`module.main()` explicitly, once, itself.

**`main()`:** loops over `STAGES` in order, loads each module, calls its `main()`,
prints a `====` header and elapsed time per stage. No error handling beyond letting an
`AssertionError` (from stage 1's validation) or any other exception propagate and stop
the whole run — there is no partial-checkpoint/resume logic in this pipeline (unlike
e.g. `strict_temporal_v2/run_full.py`), since every stage here runs in well under a
second on 829 rows.
