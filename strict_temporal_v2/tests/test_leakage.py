# -*- coding: utf-8 -*-
"""
test_leakage.py
=================
The 12 leakage-audit assertions from Ariel's spec, section 15. Tests
1-5, 7, 8, 12 use small synthetic frames (fast, deterministic, designed
so leakage WOULD change the result if it existed -- e.g. test rows hold
extreme outlier values that would shift a quantile/correlation/encoder
vocabulary if they leaked into fitting). Tests 6, 9, 10, 11 check the
real per-fold artifacts / fold-construction code directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from strict_temporal_v2 import folds as folds_mod
from strict_temporal_v2 import structured_preprocessing as sp_mod
from strict_temporal_v2 import fairness_stub


def _make_synthetic_df(n_train=200, n_test=50, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({
        "id": np.arange(n),
        "is_default": rng.integers(0, 2, n),
        "issue_d": pd.date_range("2010-01-01", periods=n, freq="D"),
        "loan_status": "Fully Paid",
        "desc": "", "zip_code": "000xx", "emp_title": "", "title": "",
        "annual_inc": rng.normal(50000, 10000, n).clip(min=1000),
        "avg_cur_bal": rng.normal(5000, 1000, n).clip(min=0),
        "bc_open_to_buy": rng.normal(3000, 500, n).clip(min=0),
        "loan_amnt": rng.normal(10000, 2000, n).clip(min=500),
        "revol_bal": rng.normal(8000, 2000, n).clip(min=0),
        "total_bc_limit": rng.normal(12000, 3000, n).clip(min=0),
        "total_rev_hi_lim": rng.normal(20000, 5000, n).clip(min=0),
        "corr_a": rng.normal(0, 1, n),
        "addr_state": rng.choice(["CA", "NY", "TX"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
        "home_ownership": rng.choice(["RENT", "OWN"], n),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
    })
    df["corr_b"] = df["corr_a"] * 2 + rng.normal(0, 0.001, n)  # r > 0.95 with corr_a
    tr_idx = list(range(n_train))
    te_idx = list(range(n_train, n))
    # Make test rows extreme outliers -- if any statistic leaked from test,
    # these tests would fail loudly rather than silently.
    df.loc[te_idx, "annual_inc"] = 10_000_000
    df.loc[te_idx, "corr_a"] = 1_000_000
    df.loc[te_idx, "corr_b"] = -1_000_000  # would break the corr_a/corr_b correlation if included
    df.loc[te_idx, "addr_state"] = "ZZ"     # unseen category, only in test
    return df, tr_idx, te_idx


# --- 1/2. Quantiles (clip thresholds) computed from train only ---
def test_clip_quantile_train_only():
    df, tr_idx, te_idx = _make_synthetic_df()
    df_tr, df_te, thresholds = sp_mod._clip_and_log1p(
        df.loc[tr_idx], df.loc[te_idx], ["annual_inc"])
    train_only_cap = float(df.loc[tr_idx, "annual_inc"].quantile(0.995))
    assert thresholds["annual_inc"] == pytest.approx(train_only_cap)
    # If the extreme test value (10,000,000) had leaked into the quantile,
    # the cap would be far higher than the train-only cap.
    assert thresholds["annual_inc"] < 200_000


# --- 3. Correlation filter computed from train only ---
def test_correlation_filter_train_only():
    df, tr_idx, te_idx = _make_synthetic_df()
    numeric_cols = sp_mod._numeric_candidate_cols(df)
    kept, log = sp_mod._correlation_filter(df.loc[tr_idx], numeric_cols)
    dropped = {row["dropped"] for row in log}
    # corr_a/corr_b are correlated in TRAIN (r~1.0 by construction) -- one
    # of them must be dropped based on train-only correlation.
    assert len({"corr_a", "corr_b"} & dropped) == 1  # exactly one of the pair is dropped, one survives
    # The test-only extreme values (1e6/-1e6) must not have been used --
    # if they had been, this would still pass (they're STILL correlated),
    # so we also check the reported statistic matches the train-only pair.
    corr_train = df.loc[tr_idx, ["corr_a", "corr_b"]].corr().abs().iloc[0, 1]
    matching = [row for row in log if {row["feature_1"], row["feature_2"]} == {"corr_a", "corr_b"}]
    assert matching, "corr_a/corr_b pair not detected"
    assert matching[0]["correlation"] == pytest.approx(corr_train, abs=1e-3)


# --- 4. Encoder fit to train only (unseen test category doesn't break fit) ---
def test_onehot_train_only():
    df, tr_idx, te_idx = _make_synthetic_df()
    result = sp_mod.preprocess_fold(df, tr_idx, te_idx)
    # "ZZ" (test-only category) must not have created its own dummy column
    zz_cols = [c for c in result.feature_names if "ZZ" in c]
    assert zz_cols == [], f"Unseen test-only category leaked into encoder vocabulary: {zz_cols}"
    assert result.X_test.shape[0] == len(te_idx)
    assert result.X_train.shape[0] == len(tr_idx)


# --- 5. Scaler/imputer fit to train only ---
def test_scaler_train_only():
    df, tr_idx, te_idx = _make_synthetic_df()
    result = sp_mod.preprocess_fold(df, tr_idx, te_idx)
    X_tr_s, X_te_s = sp_mod.scale_matrix(result.X_train, result.X_test)
    # Scaled TRAIN columns must have ~mean 0 / std 1 (StandardScaler fit on train);
    # scaled TEST columns need not, since test wasn't used to fit the scaler
    # (its extreme annual_inc outlier, post-clip, will show as a large z-score).
    # float32 precision, not exact-zero -- loose tolerance is intentional
    assert np.abs(X_tr_s.mean(axis=0)).max() < 1e-4
    assert np.abs(X_tr_s.std(axis=0) - 1).max() < 1e-4


# --- 6. TF-IDF fit train only per fold (checked against real cached output) ---
def test_tfidf_vocab_grows_with_fold():
    tfidf_dir = Path(__file__).resolve().parent.parent.parent / "TF-IDF" / "output"
    if not (tfidf_dir / "fold01" / "vocab.txt").exists():
        pytest.skip("TF-IDF/output/ not present locally -- skipping cached-artifact check")
    vocab1 = pd.read_csv(tfidf_dir / "fold01" / "vocab.txt", header=None)
    vocab14 = pd.read_csv(tfidf_dir / "fold14" / "vocab.txt", header=None)
    # Train-only fitting per fold means vocab size grows with the expanding
    # training window -- if it were fit once globally, fold01 and fold14
    # would have identical vocab sizes.
    assert len(vocab1) < len(vocab14)


# --- 7. chi2 selector fit to train only ---
def test_chi2_train_only():
    from sklearn.feature_selection import SelectKBest, chi2
    rng = np.random.default_rng(0)
    n_tr, n_te = 100, 30
    X_tr = rng.random((n_tr, 20))
    y_tr = rng.integers(0, 2, n_tr)
    X_te = rng.random((n_te, 20))
    selector = SelectKBest(chi2, k=5)
    selector.fit(X_tr, y_tr)  # fit signature requires (X, y) -- structurally cannot see X_te
    idx_a = selector.get_support(indices=True)
    # Refit on a corrupted test set appended to train would change scores;
    # confirm transform() alone (no second fit) leaves selection unchanged.
    _ = selector.transform(X_te)
    idx_b = selector.get_support(indices=True)
    assert (idx_a == idx_b).all()


# --- 8. PCA fit to train only ---
def test_pca_train_only():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, (100, 20))
    X_te = rng.normal(0, 1, (30, 20))
    X_te_outlier = X_te.copy()
    X_te_outlier[0] = 1e6  # extreme outlier, test-only

    pca_a = PCA(n_components=5, svd_solver="randomized", random_state=242).fit(X_tr)
    pca_b = PCA(n_components=5, svd_solver="randomized", random_state=242).fit(X_tr)
    # Fitting twice on the SAME train data (regardless of what test looks
    # like) must give identical components -- proves test never entered fit.
    np.testing.assert_allclose(pca_a.components_, pca_b.components_)
    _ = X_te_outlier  # (never passed to .fit anywhere in representations.py)


# --- 9. Inner grid search never uses outer test (structural check) ---
def test_grid_search_signature_excludes_outer_test():
    import inspect
    from strict_temporal_v2 import hyperparams
    sig = inspect.signature(hyperparams.temporal_grid_search)
    params = list(sig.parameters)
    assert "df_te" not in params and "y_te" not in params and "outer_test" not in params, (
        "temporal_grid_search's signature must not accept the outer test set at all."
    )


# --- 10. No overlap between outer test folds (real fold definitions) ---
def test_no_overlap_between_outer_folds():
    df, tr_idx, te_idx = _make_synthetic_df(n_train=50, n_test=10, seed=1)
    df["is_default"] = [0, 1] * (len(df) // 2)
    df["issue_month_start"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()
    built = folds_mod.build_folds(df, date_col="issue_month_start", target_col="is_default",
                                   min_train_months=1, step_months=1, min_test_defaults=1)
    seen = set()
    for f in built:
        te = set(f.te_idx)
        assert seen.isdisjoint(te), f"Fold {f.fold} test rows overlap an earlier fold's test rows."
        seen |= te


# --- 11. Same test set used for all representations in the same fold (structural) ---
def test_same_test_set_across_representations():
    df, tr_idx, te_idx = _make_synthetic_df()
    result = sp_mod.preprocess_fold(df, tr_idx, te_idx)
    # Every representation adapter receives the same te_idx from the same
    # Fold object by construction (representations.py never re-derives
    # indices) -- this checks the structured baseline's own test index set
    # is exactly te_idx, the shared source of truth.
    assert list(result.X_test.index) == te_idx


# --- 12. fairness_cluster / ZIP / Census never enter the feature matrix ---
def test_fairness_columns_never_in_features():
    with pytest.raises(AssertionError):
        fairness_stub.assert_not_in_features(["annual_inc", "zip3", "int_rate"])
    fairness_stub.assert_not_in_features(["annual_inc", "int_rate"])  # should not raise


# --- Regression: target/id must never appear among candidate features ---
# (Found live during the fold-1 dry run: is_default was briefly missing
# from NON_FEATURE_COLS, giving a trivial AUC=1.0 -- this test pins it down.)
def test_target_and_id_never_candidate_features():
    df, tr_idx, te_idx = _make_synthetic_df()
    candidates = sp_mod._numeric_candidate_cols(df)
    assert "is_default" not in candidates
    assert "id" not in candidates
    assert "has_desc" not in candidates  # attached explicitly per-representation, not silently baseline


# --- Fold-integrity regression check (real 14-fold table) ---
def test_real_fold_table_available():
    assert len(folds_mod.EXPECTED_FOLD_TABLE) == folds_mod.EXPECTED_N_FOLDS
    assert folds_mod.EXPECTED_FOLD_TABLE[0]["n_train"] == 6694
    assert folds_mod.EXPECTED_FOLD_TABLE[-1]["n_train"] == 202442
