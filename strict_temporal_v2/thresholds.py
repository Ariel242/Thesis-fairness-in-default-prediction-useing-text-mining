# -*- coding: utf-8 -*-
"""
thresholds.py
==============
Decision-threshold policy for fairness/operating-point metrics.

The old [0.5, 0.6, 0.7] list is confirmed dead code in every existing
script (BASELINE.py, GRIDSEARCH.py, BASELINE_PCA50.py all mark it
"currently unused -- kept for compatibility"). It's kept here ONLY as an
explicitly-labeled legacy/optional sensitivity grid, never as the default.
"""

import numpy as np
from sklearn.metrics import roc_curve

LEGACY_SENSITIVITY_THRESHOLDS = [0.5, 0.6, 0.7]  # optional sensitivity analysis only -- not a live decision


def youden_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Youden's J statistic (sensitivity + specificity - 1), maximized over
    the ROC curve. MUST be called on a validation split, never on the outer
    test set -- callers are responsible for passing validation-only data."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def resolve_threshold(policy: str, y_val: np.ndarray | None = None,
                       p_val: np.ndarray | None = None, explicit: float | None = None) -> float | list[float]:
    """policy: 'legacy_fixed' (returns the [0.5,0.6,0.7] list, sensitivity-
    analysis only), 'validation_youden' (computed on y_val/p_val, never on
    outer test), or 'explicit' (returns `explicit` unchanged)."""
    if policy == "legacy_fixed":
        return LEGACY_SENSITIVITY_THRESHOLDS
    elif policy == "validation_youden":
        if y_val is None or p_val is None:
            raise ValueError("validation_youden requires y_val and p_val (validation-only, never outer test).")
        return youden_threshold(y_val, p_val)
    elif policy == "explicit":
        if explicit is None:
            raise ValueError("policy='explicit' requires an `explicit` threshold value.")
        return explicit
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")
