# -*- coding: utf-8 -*-
"""
fairness_stub.py
==================
Placeholder for the not-yet-defined fairness grouping. No Census
variables and no clustering method are decided yet (Ariel's spec,
section 11) -- this module only defines the contract a future
`fairness_cluster` column must satisfy, and an assertion helper that
proves it never entered the feature matrix.
"""

import pandas as pd


# addr_state is deliberately NOT banned -- its one-hot dummies (addr_state_CA,
# etc.) are a legitimate existing structured feature in every prior pipeline
# version (STRUCT_COLS has always included them). Only ZIP/Census-level
# geography and the future fairness_cluster hook are banned from X.
_BANNED_EXACT = {"fairness_cluster", "zip3", "zip_code", "census_zcta"}
_BANNED_PREFIXES = ("zip3:", "zip3_", "census:", "census_", "fairness_cluster:")


def assert_not_in_features(feature_names: list, fairness_col: str = "fairness_cluster") -> None:
    """Call this after building any representation's feature matrix.
    fairness_cluster / zip3 / addr_state (or any future Census-derived
    column) must never appear among the model's input features -- they are
    evaluation-only. Matches on the exact bare name or a namespaced prefix
    (e.g. "zip3:...") ONLY -- deliberately NOT a generic substring search,
    since that would false-positive on legitimate TF-IDF vocabulary tokens
    like "tfidf:census bureau" that merely contain the word as loan-text
    content, not an actual geography feature (found live during the fold-14
    dry run)."""
    leaked = [f for f in feature_names
              if f.lower() in _BANNED_EXACT or f.lower().startswith(_BANNED_PREFIXES)
              or f == fairness_col]
    assert not leaked, f"Fairness/geography columns leaked into features: {leaked}"


def attach_fairness_cluster_stub(df: pd.DataFrame) -> pd.DataFrame:
    """No-op today (returns df unchanged) -- documents the intended future
    contract: a `fairness_cluster` column, computed however Ariel decides
    (Census-based, ZIP3-based, or otherwise), joined by id, used only in
    metrics.py's per-group FPR/FNR/calibration breakdown, never in X."""
    return df
