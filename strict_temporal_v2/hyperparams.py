# -*- coding: utf-8 -*-
"""
hyperparams.py
===============
Fixed ablation configuration (used for the main Structured vs
Structured+representation comparison) and the temporal grid-search
configuration (used only for the separately-run final-config tuning,
never for the main ablation).

IMPORTANT, verified via file mtimes (analysis/BASELINE.py 2026-08-16
18:01, analysis/GRIDSEARCH.py 2026-08-16 22:48 -- ~4h45m later): the
FIXED values below PREDATE analysis/GRIDSEARCH.py and are NOT its
output. When GRIDSEARCH.py was later run, its own most-frequent winners
across the 14 folds were actually C=0.03 (not 0.3) and max_depth=3 (not
4) -- i.e. the fixed config here is a priori, not grid-search-derived,
and is somewhat less regularized than what tuning prefers. Do not
describe FIXED_LR/FIXED_XGB as "selected by grid search" anywhere.
"""

RANDOM_STATE = 242

FIXED_LR = dict(
    penalty="l2",
    C=0.3,
    solver="saga",
    max_iter=1000,
    tol=1e-4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

FIXED_XGB = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    eval_metric="auc",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    # No early stopping -- all 300 trees always fit, matching the existing
    # pipeline (no eval_set / early_stopping_rounds anywhere in it).
)


def build_xgb_params(y_train) -> dict:
    """scale_pos_weight recomputed from the actual fold's train class ratio."""
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    return {**FIXED_XGB, "scale_pos_weight": spw}


# ------------------------------------------------------------
# Temporal grid search (separate final-config tuning -- NOT part of the
# main ablation run). representation is a config parameter, not hardcoded.
# ------------------------------------------------------------
VAL_FRAC = 0.2  # last 20% chronologically of outer train -> sub-validation

GRID_LR = {"C": [0.01, 0.03, 0.1, 0.3, 1]}
GRID_XGB = {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}


def temporal_grid_search(build_matrix_fn, df_tr, y_tr, representation: str,
                          val_frac: float = VAL_FRAC):
    """Generic per-fold temporal grid search, usable for any representation.

    build_matrix_fn(df_sub, df_val, representation) -> (X_sub, X_val) must
    itself refit all preprocessing (imputer/scaler/TF-IDF/PCA/etc.) on
    df_sub only -- this function does not do that itself, it only performs
    the chronological split and the search loop.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    n_sub = int(len(df_tr) * (1 - val_frac))
    df_sub, df_val = df_tr.iloc[:n_sub], df_tr.iloc[n_sub:]
    y_sub, y_val = y_tr[:n_sub], y_tr[n_sub:]

    if len(np.unique(y_sub)) < 2 or len(np.unique(y_val)) < 2:
        return None  # degenerate split -- caller should fall back to FIXED_*

    X_sub, X_val = build_matrix_fn(df_sub, df_val, representation)

    best_lr, best_lr_auc = {"C": GRID_LR["C"][0]}, -np.inf
    for c in GRID_LR["C"]:
        m = LogisticRegression(**{**FIXED_LR, "C": c})
        m.fit(X_sub, y_sub)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_lr_auc:
            best_lr_auc, best_lr = auc, {"C": c}

    spw_sub = (y_sub == 0).sum() / (y_sub == 1).sum()
    best_xgb, best_xgb_auc = {"max_depth": GRID_XGB["max_depth"][0],
                               "learning_rate": GRID_XGB["learning_rate"][0]}, -np.inf
    from itertools import product
    for md, lr in product(GRID_XGB["max_depth"], GRID_XGB["learning_rate"]):
        params = {**FIXED_XGB, "max_depth": md, "learning_rate": lr, "scale_pos_weight": spw_sub}
        m = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
        m.fit(X_sub, y_sub)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_xgb_auc:
            best_xgb_auc, best_xgb = auc, {"max_depth": md, "learning_rate": lr}

    return {"Logistic": (best_lr, best_lr_auc), "XGBoost": (best_xgb, best_xgb_auc)}
