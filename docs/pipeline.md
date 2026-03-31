# Pipeline Documentation

This project runs in five sequential steps. Each step's output serves as the next step's input. Steps 1–4 are run once to prepare the data; Step 5 is the main modeling pipeline.

---

## Step 1 — Exploratory Data Analysis
**File:** `analysis/01_EDA_accepted.ipynb`

**Input:** `data/accepted_2007_to_2018Q4.csv` (raw LendingClub data, ~1.6GB)

**What it does:**
- Inspects the raw dataset structure (151 columns, all loan vintages 2007–2018)
- Examines text field coverage (`desc`, `title`, `emp_title`) over time
- Filters loans to the 2010–2013 period and saves a year-filtered intermediate file
- Selects an initial set of ~50 columns based on relevance and missingness

**Outputs:**
- `data/filtered_by_years_only.csv` — loans from 2010–2013, all columns
- `data/filtered_50_cols.csv` — same loans, pre-selected columns

---

## Step 2 — Data Preparation & Feature Selection
**File:** `analysis/02_data_prep.ipynb`

**Input:** `data/accepted_2007_to_2018Q4.csv`

**What it does:**
- Constructs the binary target: `is_default = 1` for Charged Off / Default loans, `0` for Fully Paid
- Removes **leakage columns** — 18 post-origination fields that would not be available at loan issuance (e.g., `total_pymnt`, `out_prncp`, `last_pymnt_d`, settlement fields)
- Drops features with >50% missing values (except `desc`)
- Runs **Boruta feature selection** (Random Forest-based) on the remaining numeric features — confirms 28 features, rejects 26
- Drops multicollinear pairs (threshold r > 0.95): e.g., `fico_range_low/high` (r=1.00), `funded_amnt/installment` (r=0.955)
- Final structured feature set: **40 features**, 221,428 observations, 15.56% default rate

**Outputs:**
- `data/lc_after_02_data_prep/lc_basic_database.csv` — 40 structured features, no text
- `data/lc_after_02_data_prep/lc_basic_database_+_desc.csv` — same + `desc` column

---

## Step 3 — Advanced Feature Engineering
**File:** `analysis/03_advanced_prep.ipynb`

**Input:** `data/lc_after_02_data_prep/lc_basic_database_+_desc.csv`

**What it does:**
- Parses `issue_d` into time variables: `issue_month_start`, `issue_ym`, `month_idx`
- Engineers `funded_ratio = funded_amnt / loan_amnt` and drops `funded_amnt`
- Applies **log1p transformation** (with 99.5th-percentile cap) to 8 skewed amount features: `annual_inc`, `loan_amnt`, `revol_bal`, `tot_hi_cred_lim`, `total_bc_limit`, `total_rev_hi_lim`, `avg_cur_bal`, `bc_open_to_buy`
- Converts `term` (string "36 months") to numeric
- Ordinally encodes `grade` (A–G), `sub_grade` (A1–G5), `emp_length` (0–10)
- One-hot encodes (drop_first=True): `addr_state` (49 dummies), `purpose` (13), `home_ownership` (4), `verification_status` (2)
- Extracts `zip3` (first 3 digits of zip code)

**Output:** `data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_<timestamp>.csv`
~108 columns: 40 structured features + one-hot dummies + text columns (`desc`, `title`, `emp_title`) + time variables

---

## Step 4 — Census Data Collection
**File:** `data/ZIP3_code_API.ipynb`

**Input:** `data/zip3_counts_from_lc.csv` (list of ZIP3s present in LendingClub data)

**What it does:**
- Queries the **US Census Bureau ACS 5-Year API** for years 2011–2014
- Fetches 7 variables per ZIP Code Tabulation Area (ZCTA): total population, median income, white/Black/Hispanic counts, poverty count, total households
- Aggregates from ZCTA level to ZIP3 level (weighted median income, summed counts)
- Computes derived rates: `share_black`, `share_hispanic`, `poverty_rate`
- Matches census ZIP3s to LendingClub ZIP3s — coverage: 890/956 (93.1%)
- Creates decile-based group labels (`income_decile`, `poverty_decile`, `high_black`)

**Note:** Requires a Census API key set in the environment variable `CENSUS_API_KEY`.

**Outputs:**
- `data/census_final_data/zip3_census_panel_2011_2014.csv` — yearly panel (3,576 rows: 894 ZIP3s × 4 years)
- `data/census_final_data/zip3_census_profile_mean.csv` — time-averaged profile per ZIP3
- `data/census_final_data/zip3_groups.csv` — ZIP3s with decile labels
- `data/census_final_data/coverage_report.txt` — match rate summary
- `data/census_final_data/missing_zip3_in_panel.csv` — ZIP3s with no census match

---

## Step 5 — Walk-Forward Modeling Pipeline
**File:** `analysis/preliminary_results_cloude.py`

**Input:** `data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_<timestamp>.csv`

**What it does:**

**Text preprocessing:**
- Cleans `desc`, `title`, `emp_title`: lowercases, removes HTML tags and LendingClub boilerplate prefix ("Borrower added on..."), strips numbers and punctuation, lemmatizes (spaCy if available, NLTK fallback), removes stopwords (English + custom set)
- Concatenates cleaned fields into a single `text_all_clean` column
- Computes numeric text features per field: character length, word count, unique words, average word length, type-token ratio (TTR)

**Walk-forward cross-validation:**
- Expanding training window, quarterly step (`STEP_MONTHS = 3`)
- Each test fold requires at least `MIN_TEST_DEFAULTS = 100` default events
- Minimum training history: `MIN_TRAIN_MONTHS = 8`
- Result: ~10–13 folds

**Two feature variants per fold:**
- **Structured** — numeric + one-hot features only
- **Structured + Text** — above + TF-IDF on `text_all_clean` (top `TFIDF_MAX_FEATURES` unigrams/bigrams, `min_df=5`, sublinear TF, L2 norm)

**Two models per variant:**
- **Logistic Regression** — L2 penalty, C=0.3, solver=saga, balanced class weights
- **XGBoost** — 300 trees, depth=4, learning_rate=0.05, subsample=0.8, `scale_pos_weight` computed per fold from class ratio

**Evaluation per fold:**
- Predictive: ROC-AUC, PR-AUC, Brier Score, GINI — with 95% bootstrap CIs (`N_BOOTSTRAP = 500`)
- Fairness (at thresholds 0.5, 0.6, 0.7): FNR gap, FPR gap, Brier gap across ZIP3 groups (min group size: `MIN_GROUP_N = 100`, `MIN_GROUP_POS = 10`)
- DeLong test for statistical comparison of AUCs between Structured vs Structured+Text

**Outputs:** see [`docs/outputs.md`](outputs.md)

---

## Parameters Reference (`preliminary_results_cloude.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_TRAIN_MONTHS` | 8 | Minimum months of history before first test fold |
| `MIN_TEST_DEFAULTS` | 100 | Minimum default events required per test window |
| `STEP_MONTHS` | 3 | Months between consecutive fold steps |
| `MIN_GROUP_N` | 100 | Minimum observations per ZIP3 group for fairness evaluation |
| `MIN_GROUP_POS` | 10 | Minimum defaults per ZIP3 group for fairness evaluation |
| `THRESHOLDS` | [0.5, 0.6, 0.7] | Decision thresholds for FNR/FPR fairness metrics |
| `TFIDF_MAX_FEATURES` | 600 | Maximum TF-IDF vocabulary size |
| `TFIDF_NGRAM_RANGE` | (1, 2) | Unigrams and bigrams |
| `N_BOOTSTRAP` | 500 | Bootstrap iterations for confidence intervals (0 = skip) |
| `LR_PARAMS.C` | 0.3 | Logistic Regression regularization strength (lower = stronger) |
