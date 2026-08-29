# -*- coding: utf-8 -*-
"""
metrics.py
===========
Predictive metrics (AUC/PR-AUC/GINI/Brier), the DeLong paired test (copied
verbatim from every existing ablation script), and bootstrap CIs -- plus
Benjamini-Hochberg FDR correction across the family of per-fold DeLong
tests (Ariel's spec section 12: raw p-values are always kept, q-values
are added alongside, never replacing them).

y_prob throughout is the model's RAW predicted probability -- class_weight
='balanced' / scale_pos_weight mean these are NOT necessarily calibrated
operational PDs. This is deliberate and documented, not a bug: no Platt/
Isotonic calibration is applied in this module (spec section 9).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

N_BOOTSTRAP = 500
BOOT_SEED = 42


def predictive_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    return {
        "AUC": auc,
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "GINI": 2 * auc - 1,
    }


def bootstrap_ci(y_true, p, metric_fn, n=N_BOOTSTRAP, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap CI, per fold. Copied verbatim from the existing
    ablation scripts' own implementation."""
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    boot_stats = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        y_b, p_b = y_true[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_stats.append(metric_fn(y_b, p_b))
        except Exception:
            pass
    if len(boot_stats) < 10:
        return np.nan, np.nan
    return (float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))))


def delong_auc_test(y_true, p1, p2):
    """DeLong et al. (1988) paired test for two correlated AUCs on the same
    test set. Copied verbatim from TF-IDF/tfidf_structured_ablation.py /
    FinBERT/finbert_structured_ablation.py / Dictionarys' own lexicon
    ablation -- same implementation, not re-derived."""
    def auc_and_kernel(y, p):
        pos = p[y == 1]; neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([np.mean(pi > neg) + 0.5 * np.mean(pi == neg) for pi in pos])
        V01 = np.array([np.mean(pj < pos) + 0.5 * np.mean(pj == pos) for pj in neg])
        return V10.mean(), V10, V01, n1, n0

    y = np.asarray(y_true).astype(int)
    auc1, V10_1, V01_1, n1, n0 = auc_and_kernel(y, np.asarray(p1))
    auc2, V10_2, V01_2, _, _ = auc_and_kernel(y, np.asarray(p2))

    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)
    var_diff = (S10[0, 0] / n1 + S01[0, 0] / n0) + (S10[1, 1] / n1 + S01[1, 1] / n0) \
        - 2 * (S10[0, 1] / n1 + S01[0, 1] / n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan
    diff = auc1 - auc2
    se = np.sqrt(var_diff)
    z = diff / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH FDR correction. Returns q-values in the same order as the input
    p-values. NaN p-values pass through as NaN q-values (excluded from the
    correction, not treated as 0)."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return q.tolist()

    p_valid = p[valid]
    m = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order] * m / (np.arange(m) + 1)
    # enforce monotonicity (standard BH step-up)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)

    q_valid = np.empty(m)
    q_valid[order] = ranked
    q[valid] = q_valid
    return q.tolist()


def add_bh_correction(delong_df: pd.DataFrame, group_cols: list[str], p_col: str = "p_value") -> pd.DataFrame:
    """Applies BH correction within each group (e.g. per Model), adds a
    `q_value` column. Raw p-values in `p_col` are never modified or dropped."""
    out = delong_df.copy()
    out["q_value"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        out.loc[idx, "q_value"] = benjamini_hochberg(out.loc[idx, p_col].tolist())
    return out
