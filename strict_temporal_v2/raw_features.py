# -*- coding: utf-8 -*-
"""
raw_features.py
================
Reimplements the GLOBAL, non-distribution-fit steps of
analysis/02_data_prep.ipynb + analysis/03_advanced_prep.ipynb, reading
directly from data/accepted_2007_to_2018Q4.csv. Every transform here is
either a fixed row/column filter, a fixed lookup table (grade A-G,
sub_grade A1-G5), or row-wise arithmetic (fico_midpoint, funded_ratio,
credit_history_months) -- none of it is *fit* on the population, so
computing it once, before folds exist, is leakage-safe.

Deliberately NOT reimplemented here (moved to structured_preprocessing.py,
per-fold, train-only): the >99%-missing column selection, the r>0.95
correlation filter, the 99.5th-percentile clip, and one-hot encoding. This
module's output is intentionally WIDER than the old
data/lc_after_02_data_prep/lc_basic_database_+_desc.csv (which already had
those distribution-fit steps baked in globally) -- every raw candidate
column survives here so structured_preprocessing.py can decide per fold.

sanity_check() reproduces Ariel's exact assertion: N=217,287, default rate
~15.36%, before anything downstream runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "accepted_2007_to_2018Q4.csv"

TARGET_COL = "is_default"
ID_COL = "id"
DATE_COL = "issue_d"

EXPECTED_N = 217_287
EXPECTED_DEFAULT_RATE = 0.1536

# Verbatim from analysis/02_data_prep.ipynb
LEAKAGE_COLUMNS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "last_credit_pull_d", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_term",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "last_fico_range_high", "last_fico_range_low",
]

IRRELEVANT_COLUMNS = ["member_id", "url", "policy_code"]

MTHS_SINCE_FILL_COLS = [
    "mths_since_last_delinq", "mths_since_recent_revol_delinq",
    "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
    "mths_since_last_record", "mths_since_recent_bc", "mths_since_recent_inq",
]
MTHS_SINCE_FILL_VALUE = 999  # NaN means "the event never happened" -- fixed sentinel, not fit

GRADE_ORDER = list("ABCDEFG")
SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]

# Columns that never become correlation-filter candidates in structured_preprocessing.py
# (mirrors 02_data_prep.ipynb's `mandatory_columns`, with fico_range_low/high replaced
# by fico_midpoint since the consolidation now happens here instead of after the filter --
# the original excluded fico_range_low/high from candidacy the same way).
PROTECTED_COLUMNS = {
    ID_COL, TARGET_COL, DATE_COL, "desc", "has_desc", "addr_state",
    "grade_ord", "sub_grade_ord", "annual_inc", "dti", "fico_midpoint",
    "loan_amnt", "term", "int_rate", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "zip3", "emp_title", "title",
    "credit_history_months", "initial_list_status_w",
    "issue_month_start", "issue_ym", "month_idx",
}

# One-hot candidates (fit per fold in structured_preprocessing.py)
ONE_HOT_COLS = ["addr_state", "purpose", "home_ownership", "verification_status"]

# The 7 variables clipped + log1p'd per fold in structured_preprocessing.py
CLIP_LOG1P_COLS = [
    "annual_inc", "avg_cur_bal", "bc_open_to_buy", "loan_amnt",
    "revol_bal", "total_bc_limit", "total_rev_hi_lim",
]


def load_raw_features() -> pd.DataFrame:
    print("Loading raw CSV...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    # --- Year / status / purpose filter, target definition (02_data_prep.ipynb) ---
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df[(df["issue_d"].dt.year >= 2010) & (df["issue_d"].dt.year <= 2013)]

    target_statuses = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(target_statuses)]
    df[TARGET_COL] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)

    n_before = len(df)
    df = df[df["purpose"] != "small_business"]
    print(f"  Removed {n_before - len(df):,} small_business loans")
    print(f"  Rows after year/status/purpose filter: {len(df):,}")

    # --- Normalize id dtype -- the raw CSV has trailing footer/summary rows
    # (e.g. "Total amount funded in policy code 1: ...") that force the whole
    # `id` column to object dtype even though they're excluded by the filters
    # above (their issue_d is NaN). Re-cast here so every downstream join by
    # id (lexicon, FinBERT parquet) uses a consistent numeric key.
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("int64")
    assert df[ID_COL].isna().sum() == 0, "Unexpected non-numeric id after filtering -- investigate before continuing."
    assert not df[ID_COL].duplicated().any(), "Duplicate id values after filtering -- id is not a valid primary key."

    # --- Leakage-column removal ---
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    # --- mths_since_* fixed-sentinel fill (999 = "event never happened") ---
    filled = [c for c in MTHS_SINCE_FILL_COLS if c in df.columns]
    df[filled] = df[filled].fillna(MTHS_SINCE_FILL_VALUE)

    # --- Drop identifier/constant columns never used as features ---
    df = df.drop(columns=[c for c in IRRELEVANT_COLUMNS if c in df.columns])

    # --- fico_midpoint (row-wise arithmetic -- no fitting) ---
    df["fico_midpoint"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])

    # --- Time features (03_advanced_prep.ipynb) ---
    df["issue_month_start"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()
    df["issue_ym"] = df["issue_d"].dt.to_period("M").astype(str)
    min_m = df["issue_month_start"].min()
    df["month_idx"] = ((df["issue_month_start"].dt.year - min_m.year) * 12
                        + (df["issue_month_start"].dt.month - min_m.month))

    # --- funded_ratio, drop funded_amnt (near-duplicate of loan_amnt) ---
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    df["funded_amnt"] = pd.to_numeric(df["funded_amnt"], errors="coerce")
    df["funded_ratio"] = np.where(
        df["loan_amnt"].notna() & (df["loan_amnt"] != 0),
        df["funded_amnt"] / df["loan_amnt"], np.nan,
    )
    df = df.drop(columns=["funded_amnt"])

    # --- term -> numeric ---
    df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # --- grade / sub_grade -> fixed ordinal lookup (no fitting), raw strings dropped ---
    grade_map = {g: i + 1 for i, g in enumerate(GRADE_ORDER)}
    df["grade_ord"] = df["grade"].astype(str).str.strip().map(grade_map).astype("float32")
    subgrade_map = {s: i + 1 for i, s in enumerate(SUBGRADE_ORDER)}
    df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map(subgrade_map).astype("float32")
    df = df.drop(columns=["grade", "sub_grade"])

    # --- emp_length -> numeric years 0-10 ---
    s = df["emp_length"].astype(str).str.lower().str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10", "n/a": np.nan}).str.extract(r"(\d+)")[0]
    df["emp_length"] = pd.to_numeric(s, errors="coerce")

    # --- zip3 (kept for a future fairness_cluster hook -- never enters X) ---
    df["zip3"] = df["zip_code"].astype(str).str.extract(r"(\d{3})")[0]

    # --- earliest_cr_line -> credit_history_months (row-wise date diff) ---
    ecl = pd.to_datetime(df["earliest_cr_line"].astype(str).str.strip(), format="%b-%Y", errors="coerce")
    df["credit_history_months"] = ((df["issue_d"].dt.year - ecl.dt.year) * 12
                                    + (df["issue_d"].dt.month - ecl.dt.month))
    df.loc[df["credit_history_months"] < 0, "credit_history_months"] = np.nan
    df = df.drop(columns=["earliest_cr_line"])

    # --- initial_list_status -> single binary flag ---
    vals = set(df["initial_list_status"].dropna().astype(str).str.strip().unique())
    if not vals <= {"f", "w"}:
        raise ValueError(f"initial_list_status has unexpected values: {vals - {'f', 'w'}}")
    df["initial_list_status_w"] = (df["initial_list_status"].astype(str).str.strip() == "w").astype("int8")
    df = df.drop(columns=["initial_list_status"])

    # --- has_desc (new -- computed directly from desc, no fitting) ---
    df["has_desc"] = df["desc"].notna() & (df["desc"].astype(str).str.strip() != "")
    df["has_desc"] = df["has_desc"].astype("int8")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"  Final raw_features shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def sanity_check(df: pd.DataFrame) -> None:
    n = len(df)
    rate = df[TARGET_COL].mean()
    print(f"Sanity check: N={n:,} (expected {EXPECTED_N:,}), "
          f"default rate={rate:.4%} (expected ~{EXPECTED_DEFAULT_RATE:.2%})")
    assert n == EXPECTED_N, f"N mismatch: got {n:,}, expected {EXPECTED_N:,}"
    assert abs(rate - EXPECTED_DEFAULT_RATE) < 0.001, (
        f"Default rate mismatch: got {rate:.4%}, expected ~{EXPECTED_DEFAULT_RATE:.2%}"
    )
    print("  Sanity check PASSED.")


if __name__ == "__main__":
    df = load_raw_features()
    sanity_check(df)
