# Feature Dictionary — post-`02_data_prep` dataset

All 71 columns of `data/lc_after_02_data_prep/lc_basic_database.csv` (the `+_desc` file adds one more column, `desc`). Descriptions follow the official LendingClub data dictionary. Generated 2026-07-13; regenerate after changing 02's feature filtering.

**enters_model** legend: `yes` = model feature (possibly transformed in 03); `via encoding` = enters as a derived encoding; `via text pipeline` = enters as text stats + TF-IDF (Structured+Text arm only); `no` = identifier / target / time axis / excluded.

| # | Feature | Description | Variable type | Dtype | Missing % | Enters model | Notes |
|---|---------|-------------|---------------|-------|-----------|--------------|-------|
| 1 | `acc_now_delinq` | Number of accounts on which the borrower is currently delinquent | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 2 | `acc_open_past_24mths` | Number of trades opened in the past 24 months | count | float64 | 17.95 | yes | Numeric feature, passed through as-is |
| 3 | `addr_state` | US state provided by the borrower | categorical (nominal, 49 levels) | object | 0.0 | yes | One-hot encoded in 03 |
| 4 | `annual_inc` | Self-reported annual income at registration | continuous (amount) | float64 | 0.0 | yes | log1p-transformed (capped at q0.995) in 03 |
| 5 | `avg_cur_bal` | Average current balance across all accounts | continuous (amount) | float64 | 27.0 | yes | log1p-transformed (capped at q0.995) in 03 |
| 6 | `bc_open_to_buy` | Total open-to-buy (unused limit) on revolving bankcards | continuous (amount) | float64 | 18.63 | yes | log1p-transformed (capped at q0.995) in 03 |
| 7 | `bc_util` | Bankcard utilization: total balance / credit limit across bankcards (%) | continuous (%) | float64 | 18.67 | yes | Numeric feature, passed through as-is |
| 8 | `chargeoff_within_12_mths` | Number of charge-offs within the last 12 months | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 9 | `collections_12_mths_ex_med` | Collections in the last 12 months, excluding medical | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 10 | `delinq_2yrs` | Number of 30+ days past-due delinquency incidences in the past 2 years | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 11 | `delinq_amnt` | Past-due amount owed on accounts currently delinquent | continuous (amount) | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 12 | `dti` | Debt-to-income ratio: monthly debt payments (excl. mortgage) / monthly income (%) | continuous (%) | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 13 | `earliest_cr_line` | Month the borrower's earliest reported credit line was opened | date | object | 0.0 | via encoding | Converted in 03 (2026-07-16) to `credit_history_months` = months from earliest_cr_line to issue_d (credit-history length at origination); raw date dropped, negatives set to NA |
| 14 | `emp_length` | Employment length in years: 0 (<1) to 10 (10+) | ordinal (0-10) | object | 4.08 | yes | Converted to numeric years (0-10) in 03 |
| 15 | `emp_title` | Job title supplied by the borrower | free text | object | 5.96 | via text pipeline | Raw column excluded in v2; enters via cleaned-text stats + TF-IDF (Structured+Text arm only) |
| 16 | `funded_amnt` | Total amount committed to the loan at funding | continuous (amount) | float64 | 0.0 | no (currently) | Kept by the 2026-07-13 multicollinearity rule; 03 replaces it with funded_ratio and drops it, and funded_ratio is in v2 EXCLUDE_COLS |
| 17 | `grade` | LendingClub assigned risk grade, A (best) to G (worst) | ordinal (7 levels) | object | 0.0 | via encoding | Raw string excluded in v2; enters as ordinal grade_ord / sub_grade_ord |
| 18 | `home_ownership` | Home ownership status: RENT / OWN / MORTGAGE / OTHER / NONE | categorical (nominal) | object | 0.0 | yes | One-hot encoded in 03 |
| 19 | `id` | Unique LendingClub listing ID for the loan | identifier | int64 | 0.0 | no | Primary key; EXCLUDE_COLS in v2 — identifier only |
| 20 | `initial_list_status` | Initial listing status of the loan: w (whole) / f (fractional) | binary categorical | object | 0.0 | via encoding | Converted in 03 (2026-07-16) to binary flag `initial_list_status_w` (1 = w, 0 = f); raw column dropped |
| 21 | `inq_last_6mths` | Credit inquiries in the last 6 months (excl. auto and mortgage) | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 22 | `int_rate` | Interest rate on the loan (%) | continuous (%) | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 23 | `is_default` | Target: 1 = Charged Off / Default, 0 = Fully Paid | binary (target) | int64 | 0.0 | no | Target variable, not a feature |
| 24 | `issue_d` | Month the loan was funded | date | object | 0.0 | no | Defines the walk-forward folds; never a feature |
| 25 | `loan_amnt` | Loan amount applied for by the borrower | continuous (amount) | float64 | 0.0 | yes | log1p-transformed (capped at q0.995) in 03 |
| 26 | `mo_sin_old_il_acct` | Months since the oldest installment account was opened | count (months) | float64 | 29.78 | yes | Numeric feature, passed through as-is |
| 27 | `mo_sin_old_rev_tl_op` | Months since the oldest revolving account was opened | count (months) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 28 | `mo_sin_rcnt_rev_tl_op` | Months since the most recent revolving account was opened | count (months) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 29 | `mo_sin_rcnt_tl` | Months since the most recent account of any type was opened | count (months) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 30 | `mort_acc` | Number of mortgage accounts | count | float64 | 17.95 | yes | Numeric feature, passed through as-is |
| 31 | `mths_since_last_delinq` | Months since the borrower's last delinquency; 999 = never delinquent | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 32 | `mths_since_last_major_derog` | Months since most recent 90-day-or-worse rating; 999 = never | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 33 | `mths_since_last_record` | Months since the last public record; 999 = no public record | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 34 | `mths_since_recent_bc` | Months since most recent bankcard account was opened; 999 = never | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 35 | `mths_since_recent_bc_dlq` | Months since most recent bankcard delinquency; 999 = never | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 36 | `mths_since_recent_inq` | Months since most recent credit inquiry; 999 = never | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 37 | `mths_since_recent_revol_delinq` | Months since most recent revolving delinquency; 999 = never | count (months, 999=never) | float64 | 0.0 | yes | NaN filled with 999 in 02 |
| 38 | `num_accts_ever_120_pd` | Number of accounts ever 120+ days past due | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 39 | `num_actv_bc_tl` | Number of currently active bankcard accounts | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 40 | `num_actv_rev_tl` | Number of currently active revolving trades | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 41 | `num_bc_sats` | Number of satisfactory (in good standing) bankcard accounts | count | float64 | 21.75 | yes | Numeric feature, passed through as-is |
| 42 | `num_bc_tl` | Total number of bankcard accounts | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 43 | `num_il_tl` | Total number of installment accounts | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 44 | `num_op_rev_tl` | Number of open revolving accounts | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 45 | `num_rev_accts` | Total number of revolving accounts | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 46 | `num_tl_120dpd_2m` | Accounts currently 120 days past due (updated in past 2 months) | count | float64 | 27.12 | yes | Numeric feature, passed through as-is |
| 47 | `num_tl_30dpd` | Accounts currently 30 days past due (updated in past 2 months) | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 48 | `num_tl_90g_dpd_24m` | Accounts 90+ days past due in the last 24 months | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 49 | `num_tl_op_past_12m` | Accounts opened in the past 12 months | count | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 50 | `open_acc` | Number of open credit lines in the borrower's credit file | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 51 | `pct_tl_nvr_dlq` | Percent of trades never delinquent (%) | continuous (%) | float64 | 27.07 | yes | Numeric feature, passed through as-is |
| 52 | `percent_bc_gt_75` | Percent of bankcard accounts with utilization above 75% (%) | continuous (%) | float64 | 18.63 | yes | Numeric feature, passed through as-is |
| 53 | `pub_rec` | Number of derogatory public records | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 54 | `pub_rec_bankruptcies` | Number of public-record bankruptcies | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 55 | `purpose` | Borrower-selected loan purpose (debt_consolidation, credit_card, ...; small_business excluded from study) | categorical (nominal) | object | 0.0 | yes | One-hot encoded in 03 |
| 56 | `revol_bal` | Total credit revolving balance | continuous (amount) | float64 | 0.0 | yes | log1p-transformed (capped at q0.995) in 03 |
| 57 | `revol_util` | Revolving line utilization: balance relative to available revolving credit (%) | continuous (%) | float64 | 0.07 | yes | Numeric feature, passed through as-is |
| 58 | `sub_grade` | LendingClub assigned risk sub-grade, A1 to G5 | ordinal (35 levels) | object | 0.0 | via encoding | Raw string excluded in v2; enters as ordinal grade_ord / sub_grade_ord |
| 59 | `tax_liens` | Number of tax liens | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 60 | `term` | Loan repayment term: 36 or 60 months | ordinal (2 levels) | object | 0.0 | yes | Converted to numeric (36/60) in 03 |
| 61 | `title` | Loan title supplied by the borrower | free text | object | 0.01 | via text pipeline | Raw column excluded in v2; enters via cleaned-text stats + TF-IDF (Structured+Text arm only) |
| 62 | `tot_coll_amt` | Total collection amounts ever owed | continuous (amount) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 63 | `tot_cur_bal` | Total current balance of all accounts | continuous (amount) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 64 | `total_acc` | Total number of credit lines ever in the credit file | count | float64 | 0.0 | yes | Numeric feature, passed through as-is |
| 65 | `total_bal_ex_mort` | Total credit balance excluding mortgage | continuous (amount) | float64 | 17.95 | yes | Numeric feature, passed through as-is |
| 66 | `total_bc_limit` | Total bankcard high credit / credit limit | continuous (amount) | float64 | 17.95 | yes | log1p-transformed (capped at q0.995) in 03 |
| 67 | `total_il_high_credit_limit` | Total installment high credit / credit limit | continuous (amount) | float64 | 27.0 | yes | Numeric feature, passed through as-is |
| 68 | `total_rev_hi_lim` | Total revolving high credit / credit limit | continuous (amount) | float64 | 27.0 | yes | log1p-transformed (capped at q0.995) in 03 |
| 69 | `verification_status` | Income verification: Verified / Source Verified / Not Verified | categorical (nominal) | object | 0.0 | yes | One-hot encoded in 03 |
| 70 | `zip_code` | First 3 digits of the borrower's ZIP code | categorical (high cardinality) | object | 0.0 | no | EXCLUDE_COLS in v2 (high cardinality); zip3 derived in 03 but not modeled |
| 71 | `fico_midpoint` | Engineered in 02: mean of fico_range_low and fico_range_high — FICO score at origination | continuous (score) | float64 | 0.0 | yes | Engineered in 02 |
