# -*- coding: utf-8 -*-
"""
structured_preprocessing.py
============================
Per-fold, train-only structured preprocessing -- the core leakage fix.
Every distribution-dependent step here is fit on the fold's TRAIN rows
only and applied unchanged to TEST: missing-column selection (>99%,
threshold value unchanged from the original pipeline -- only the fitting
scope changes, confirmed with Ariel), correlation filtering (r>0.95,
deterministic tie-break), 99.5th-percentile clip + log1p (7 vars), and
one-hot encoding (OneHotEncoder, unseen-category-safe).

Never assumes a fixed feature count -- returns the actual per-fold
feature names alongside the matrices, per the spec.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from strict_temporal_v2.raw_features import (
    CLIP_LOG1P_COLS, ONE_HOT_COLS, PROTECTED_COLUMNS, TARGET_COL,
)

MISSING_THRESHOLD = 0.99  # unchanged from analysis/02_data_prep.ipynb -- confirmed with Ariel
CORR_THRESHOLD = 0.95
CLIP_QUANTILE = 0.995

# Columns that are never candidates for anything below (identifiers, target,
# text, date bookkeeping) -- excluded from missing/correlation/clip/one-hot
# entirely, not because they're protected from removal, but because they
# aren't structured model features in the first place.
#
# Matches the verified EXCLUDE_COLS used by every existing ablation script
# (analysis/BASELINE.py, TF-IDF/tfidf_structured_ablation.py, etc.) exactly:
# {TARGET_COL, DATE_COL, "id", "issue_d", "issue_ym", "month_idx",
#  "issue_month_start", "zip_code", "zip3", "emp_title", "title", "desc",
#  "funded_ratio", "grade", "sub_grade", "last_fico_range_high/low"}.
# grade/sub_grade/last_fico_* never reach this module (already dropped or
# converted in raw_features.py). funded_ratio is deliberately excluded --
# not a mistake, it matches the existing pipeline's own STRUCT_COLS, which
# never included it despite being engineered specifically to replace
# funded_amnt; not changing that undocumented decision here.
# has_desc is likewise excluded from the base "structured" candidate list --
# it is attached explicitly, per-representation, by representations.py /
# run_dry.py / run_full.py, never silently included in the baseline.
NON_FEATURE_COLS = {
    TARGET_COL, "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
    "desc", "loan_status", "zip_code", "zip3", "emp_title", "title",
    "funded_ratio", "has_desc",
}


@dataclass
class FoldPreprocessResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    feature_names: list = field(repr=False)
    audit: dict = field(repr=False)


def _numeric_candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, "Int64", "float32", "float64"]]


def _select_missing_cols(df_train: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], list[str]]:
    """Columns with >MISSING_THRESHOLD missing in TRAIN are dropped. desc is
    never a candidate (handled separately, not part of candidate_cols)."""
    null_frac = df_train[candidate_cols].isnull().mean()
    dropped = null_frac[null_frac > MISSING_THRESHOLD].index.tolist()
    kept = [c for c in candidate_cols if c not in dropped]
    return kept, dropped


def _correlation_filter(df_train: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], list[dict]]:
    """r>0.95 pairwise on TRAIN only. Protected columns (PROTECTED_COLUMNS)
    are never candidates for removal, matching 02_data_prep.ipynb's
    `mandatory_columns` exclusion. Deterministic tie-break: drop the column
    with more missing in train; on an exact tie, drop the alphabetically
    later name."""
    testable = [c for c in candidate_cols if c not in PROTECTED_COLUMNS]
    if len(testable) < 2:
        return candidate_cols, []

    corr = df_train[testable].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        hits = upper.index[upper[col] > CORR_THRESHOLD].tolist()
        for h in hits:
            pairs.append((col, h, float(upper.loc[h, col])))

    null_pct = df_train[testable].isnull().mean()
    to_drop: set[str] = set()
    log_rows = []
    for f1, f2, r in pairs:
        if null_pct[f1] > null_pct[f2]:
            drop = f1
        elif null_pct[f2] > null_pct[f1]:
            drop = f2
        else:
            drop = max(f1, f2)
        to_drop.add(drop)
        log_rows.append({"feature_1": f1, "feature_2": f2, "correlation": round(r, 4), "dropped": drop})

    kept = [c for c in candidate_cols if c not in to_drop]
    return kept, log_rows


def _clip_and_log1p(df_train: pd.DataFrame, df_test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_train = df_train.copy()
    df_test = df_test.copy()
    thresholds = {}
    for c in cols:
        if c not in df_train.columns:
            continue
        cap = float(df_train[c].quantile(CLIP_QUANTILE))
        thresholds[c] = cap
        df_train[c] = np.log1p(df_train[c].clip(upper=cap))
        df_test[c] = np.log1p(df_test[c].clip(upper=cap))
    return df_train, df_test, thresholds


def preprocess_fold(df: pd.DataFrame, tr_idx: list, te_idx: list) -> FoldPreprocessResult:
    df_tr_raw = df.loc[tr_idx].copy()
    df_te_raw = df.loc[te_idx].copy()

    audit: dict = {}

    # 1) Missing-column selection (train-only, >99%)
    numeric_candidates = _numeric_candidate_cols(df_tr_raw)
    kept_after_missing, dropped_missing = _select_missing_cols(df_tr_raw, numeric_candidates)
    audit["missing_cols_removed"] = dropped_missing
    audit["n_cols_before_missing_filter"] = len(numeric_candidates)
    audit["n_cols_after_missing_filter"] = len(kept_after_missing)

    # 2) Correlation filter (train-only, r>0.95)
    kept_after_corr, corr_log = _correlation_filter(df_tr_raw, kept_after_missing)
    audit["correlated_pairs"] = corr_log
    audit["n_cols_after_correlation_filter"] = len(kept_after_corr)

    numeric_final = kept_after_corr

    # 3) Clip (99.5th pct, train-only) + log1p, only for the 7 designated vars present
    clip_cols_present = [c for c in CLIP_LOG1P_COLS if c in numeric_final]
    df_tr, df_te, clip_thresholds = _clip_and_log1p(df_tr_raw, df_te_raw, clip_cols_present)
    audit["clip_thresholds"] = clip_thresholds

    X_tr_num = df_tr[numeric_final].astype(np.float32)
    X_te_num = df_te[numeric_final].astype(np.float32)

    # 4) One-hot encode (train-only fit, unseen categories in test -> all-zero row)
    ohe_cols_present = [c for c in ONE_HOT_COLS if c in df.columns]
    for c in ohe_cols_present:
        df_tr[c] = df_tr[c].astype(str).fillna("Missing").replace({"nan": "Missing"})
        df_te[c] = df_te[c].astype(str).fillna("Missing").replace({"nan": "Missing"})

    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    ohe_tr = encoder.fit_transform(df_tr[ohe_cols_present]) if ohe_cols_present else np.empty((len(df_tr), 0))
    ohe_te = encoder.transform(df_te[ohe_cols_present]) if ohe_cols_present else np.empty((len(df_te), 0))
    ohe_names = list(encoder.get_feature_names_out(ohe_cols_present)) if ohe_cols_present else []
    audit["categorical_feature_count"] = len(ohe_names)

    X_tr = pd.concat([
        X_tr_num.reset_index(drop=True),
        pd.DataFrame(ohe_tr, columns=ohe_names),
    ], axis=1)
    X_te = pd.concat([
        X_te_num.reset_index(drop=True),
        pd.DataFrame(ohe_te, columns=ohe_names),
    ], axis=1)
    X_tr.index = df_tr.index
    X_te.index = df_te.index

    feature_names = numeric_final + ohe_names
    audit["final_structured_feature_count"] = len(feature_names)

    # 5) Numeric imputation (unchanged strategy: constant 0), train-fit
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr[numeric_final] = imputer.fit_transform(X_tr[numeric_final])
    X_te[numeric_final] = imputer.transform(X_te[numeric_final])

    return FoldPreprocessResult(X_train=X_tr, X_test=X_te, feature_names=feature_names, audit=audit)


def scale_matrix(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler, fit on train only. Preserves the existing pipeline's
    actual behavior (verified in BASELINE.py / TF-IDF / Lexicon ablations):
    ONE scaled matrix is fed to both Logistic Regression and XGBoost --
    scaling was never gated to Logistic-only in the prior code, so this
    keeps that as-is rather than introducing a new split (per Ariel's
    instruction not to change undocumented research decisions)."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.values.astype(np.float32))
    X_te_s = scaler.transform(X_test.values.astype(np.float32))
    return X_tr_s, X_te_s
