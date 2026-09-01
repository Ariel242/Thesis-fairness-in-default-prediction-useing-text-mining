# -*- coding: utf-8 -*-
"""
run_thesis_final.py
====================
The tuned, thesis-reportable run. BUILT BUT NOT YET EXECUTED (Ariel, 2026-08-31:
"אל תריץ כרגע... כרגע אני רוצה שתכין את הPIPELINE").

WHY THIS EXISTS
----------------
`run_full.py` produced `results/strict_temporal_v2/predictions.csv` -- the prediction
set every fairness analysis in census/ (stages 08-15) is built on -- using the FIXED
hyperparameters in hyperparams.py. That module's own docstring records the problem:
those values PREDATE analysis/GRIDSEARCH.py and are not its output; when the grid
search was actually run, its most-frequent winners were C=0.03 (not 0.3) and
max_depth=3 (not 4). A thesis cannot report hyperparameters chosen a priori and
described as tuned, so this script re-runs the pipeline with a genuine per-fold grid
search and writes to a separate output directory, leaving run_full.py's results intact
for comparison.

SCOPE -- only what the thesis actually reports (agreed 2026-08-31)
-------------------------------------------------------------------
  3 representations : structured, tfidf_full, finbert_pca50
  3 models          : Logistic, XGBoost, RandomForest
RandomForest is included here for the first time in this package -- run_full.py
excluded it per an earlier instruction, which is why no fairness result so far covers
it. The other four representations (structured_has_desc, lexicon, tfidf_chi2,
finbert_768) are deliberately NOT tuned: there is no point spending hours tuning arms
that will not appear in the results chapter. They remain available, untuned, in
run_full.py's output.

GRID SEARCH PROTOCOL (no leakage, and none into the outer test fold)
---------------------------------------------------------------------
Within each fold's training rows -- which are already chronologically sorted and form a
contiguous prefix -- the last VAL_FRAC (20%) becomes a temporal inner-validation split.
Every candidate config is scored on that split's AUC; the winner is then REFIT on the
full fold training set before touching the outer test fold. The outer test period is
never involved in choosing anything.

Critically, the inner split rebuilds ALL preprocessing on the sub-train only:

  - structured / finbert_pca50: `build_final_matrix(df, sub_idx, val_idx, ...)` already
    refits the imputer, correlation filter, clipper, one-hot encoder, scaler and PCA on
    whatever index it is handed, so passing it the inner split is sufficient.

  - tfidf_full: NOT sufficient, and this is the subtle one. `attach_tfidf_full()` loads
    a per-fold TF-IDF matrix cached under TF-IDF/output/foldNN/, which was fit on the
    fold's FULL training set -- including the rows that become the inner validation
    split. Reusing it during the search would let the vocabulary and IDF weights see
    the validation rows, exactly the class of leakage this whole package exists to
    remove (and the row counts would not align anyway). So for the inner split this
    script refits a fresh TfidfVectorizer on the sub-train text only, using settings
    copied verbatim from TF-IDF/tfidf_pipeline.py (ngram_range=(1,2), min_df=5,
    sublinear_tf=True, stop_words="english"), EXCEPT capped at max_features=20,000 for
    the search only (see TFIDF_INNER_SEARCH_KWARGS -- added 2026-09-01 after 15
    consecutive near-instant kills specifically at fold 14's uncapped inner-search fit,
    the one tfidf_full-specific step whose cost scales with fold size and that did not
    exist at all in run_full.py). The cached fold-level matrices -- uncapped, exactly
    matching TF-IDF/tfidf_pipeline.py -- are still used for the FINAL refit and test
    prediction, where they are both correct and what the thesis actually reports.

RUNTIME AND THE KNOWN RISK
----------------------------
17 fits per fold per representation (5 LR + 6 XGB + 3 RF on the inner split, then 3
final refits), times 3 representations times 14 folds. Estimated 8-10 hours total.

The real risk is RandomForest on tfidf_full, whose fold-14 matrix is ~55,000 sparse
columns on ~202,000 rows, on a 13.7GB machine with documented OOM history (see
FinBERT/BASELINE_768.py's MEMORY NOTE, which was killed at fold 13/14 on only 895
columns). Mitigations built in here: RF and XGBoost default to n_jobs=4 rather than -1
(bounded peak memory; results are unaffected at a fixed random_state), matrices are
explicitly deleted and gc.collect()'d between representations, and the run is
checkpointed per fold so a kill costs one fold rather than the whole run. If fold 13/14
still cannot complete for tfidf_full x RandomForest, `--skip-rf-on tfidf_full` drops
that one combination and records it as a disclosed limitation rather than failing the
run.

USAGE
------
  python -m strict_temporal_v2.run_thesis_final                    # run/resume all 14 folds
  python -m strict_temporal_v2.run_thesis_final --restart          # ignore checkpoint
  python -m strict_temporal_v2.run_thesis_final --folds 1          # smoke-test one fold first
  python -m strict_temporal_v2.run_thesis_final --skip-rf-on tfidf_full
  python -m strict_temporal_v2.run_thesis_final --n-jobs 2         # tighter memory bound

RECOMMENDED FIRST STEP: `--folds 1` finishes in minutes and verifies the whole path
end to end before committing to an overnight run.

OUTPUTS (results/strict_temporal_v2_tuned/) -- predictions.csv deliberately uses the
SAME COLUMN SCHEMA as run_full.py's, so every census/ fairness stage (08-15) can be
pointed at this file by changing one path constant and needs no other modification.
  predictions.csv       -- id, issue_d, fold, model, representation, y_true, y_prob, has_desc
  fold_metrics.csv      -- AUC/PR-AUC/GINI/Brier per fold x model x representation
  best_params.csv       -- the winning config per fold x model x representation, with
                            its inner-validation AUC and every candidate's score
  completed_folds.txt   -- resume marker
"""

import argparse
import gc
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from strict_temporal_v2 import folds as folds_mod
from strict_temporal_v2 import hyperparams as hp
from strict_temporal_v2 import metrics as metrics_mod
from strict_temporal_v2 import representations as reprs
from strict_temporal_v2 import structured_preprocessing as sp_prep
from strict_temporal_v2.raw_features import load_raw_features, sanity_check
from strict_temporal_v2.fairness_stub import assert_not_in_features

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results" / "strict_temporal_v2_tuned"
PRED_CSV = OUT_DIR / "predictions.csv"
METRICS_CSV = OUT_DIR / "fold_metrics.csv"
BEST_PARAMS_CSV = OUT_DIR / "best_params.csv"
MARKER_FILE = OUT_DIR / "completed_folds.txt"

TEXT_CSV = BASE_DIR / "TF-IDF" / "output" / "text_all_clean.csv"
TEXT_COL = "text_all_clean"

REPRESENTATIONS = ["structured", "tfidf_full", "finbert_pca50"]
MODELS = ["Logistic", "XGBoost", "RandomForest"]

VAL_FRAC = 0.2
RANDOM_STATE = 242

# Grids -- identical to analysis/GRIDSEARCH.py and hyperparams.GRID_*, so the tuned
# run is directly comparable to the earlier structured-only grid search.
GRID_LR = {"C": [0.01, 0.03, 0.1, 0.3, 1]}
GRID_XGB = {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}
GRID_RF = {"min_samples_leaf": [10, 20, 50]}

# TF-IDF settings copied verbatim from TF-IDF/tfidf_pipeline.py (see module docstring),
# used for the FINAL model (fit on the cached, uncapped fold-level matrix).
TFIDF_KWARGS = dict(max_features=None, ngram_range=(1, 2), sublinear_tf=True,
                    min_df=5, stop_words="english")

# Inner grid-search ONLY: 2026-09-01, after 15 consecutive near-instant kills specifically
# at fold 14's tfidf_full inner search, regardless of --skip-rf-on -- narrowing the
# search's own TfidfVectorizer.fit() (uncapped, ngram (1,2), on fold 14's ~162K-row
# sub-train) to the likely cause, since it is the one tfidf_full-specific step that
# scales with fold size and did not exist at all in run_full.py. This cap applies ONLY
# to the 5+6+3 candidate fits used to pick a winning config -- the FINAL, THESIS-REPORTED
# model for every fold is still fit on the uncapped cached fold-level matrix via
# TFIDF_KWARGS above, so this is a search-efficiency choice, not a change to what gets
# reported. Disclosed here rather than silently narrowed.
TFIDF_INNER_SEARCH_KWARGS = dict(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True,
                                 min_df=5, stop_words="english")

LR_BASE = dict(penalty="l2", solver="saga", max_iter=1000, tol=1e-4,
               class_weight="balanced", random_state=RANDOM_STATE)
XGB_BASE = dict(n_estimators=300, subsample=0.8, eval_metric="auc",
                random_state=RANDOM_STATE)
RF_BASE = dict(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE)


# ------------------------------------------------------------------
# checkpointing
# ------------------------------------------------------------------
def _append_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _read_completed() -> set:
    if not MARKER_FILE.exists():
        return set()
    return {int(x) for x in MARKER_FILE.read_text().split() if x.strip()}


def _mark_complete(fold_num: int) -> None:
    with open(MARKER_FILE, "a", encoding="utf-8") as f:
        f.write(f"{fold_num}\n")


# ------------------------------------------------------------------
# inner-split matrix construction (leak-free -- see module docstring)
# ------------------------------------------------------------------
_TEXT_CACHE = {}


def _get_text(ids: pd.Series) -> pd.Series:
    """Cleaned text for the given loan ids, in the given order, from the file
    TF-IDF/tfidf_pipeline.py already produced."""
    if "df" not in _TEXT_CACHE:
        t = pd.read_csv(TEXT_CSV)
        _TEXT_CACHE["df"] = t.set_index("id")[TEXT_COL]
    return _TEXT_CACHE["df"].reindex(ids.values).fillna("").astype(str)


def build_inner_matrix(df, sub_idx, val_idx, representation, y_sub, n_jobs):
    """Matrix pair for the grid-search inner split, with EVERY preprocessing step fit
    on sub_idx only. For tfidf_full this refits TF-IDF from scratch rather than reusing
    the fold-level cache, which would leak the validation rows into the vocabulary."""
    if representation != "tfidf_full":
        X_sub, X_val, names, _audit = reprs.build_final_matrix(
            df, sub_idx, val_idx, representation, fold_num=-1, y_tr=y_sub)
        return X_sub, X_val, names

    # tfidf_full: replicate build_final_matrix's tfidf branch, but with a fresh
    # vectorizer fit on the sub-train text only.
    result = sp_prep.preprocess_fold(df, sub_idx, val_idx)
    hd_sub = reprs._has_desc_block(df.loc[sub_idx, "has_desc"])
    hd_val = reprs._has_desc_block(df.loc[val_idx, "has_desc"])
    combo_sub = np.hstack([result.X_train.values.astype(np.float32), hd_sub])
    combo_val = np.hstack([result.X_test.values.astype(np.float32), hd_val])
    struct_sub, struct_val = reprs._scale(combo_sub, combo_val)

    vec = TfidfVectorizer(**TFIDF_INNER_SEARCH_KWARGS)
    tf_sub = vec.fit_transform(_get_text(df.loc[sub_idx, "id"]))
    tf_val = vec.transform(_get_text(df.loc[val_idx, "id"]))

    X_sub = sp.hstack([sp.csr_matrix(struct_sub), tf_sub], format="csr")
    X_val = sp.hstack([sp.csr_matrix(struct_val), tf_val], format="csr")
    names = list(result.feature_names) + ["has_desc"] + [f"tfidf:{w}" for w in vec.get_feature_names_out()]
    return X_sub, X_val, names


# ------------------------------------------------------------------
# model construction + grid search
# ------------------------------------------------------------------
def make_model(model_name: str, params: dict, y_train: np.ndarray, n_jobs: int):
    if model_name == "Logistic":
        return LogisticRegression(**{**LR_BASE, **params})
    if model_name == "XGBoost":
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        return xgb.XGBClassifier(**{**XGB_BASE, **params},
                                 scale_pos_weight=spw, n_jobs=n_jobs,
                                 use_label_encoder=False, verbosity=0)
    return RandomForestClassifier(**{**RF_BASE, **params}, n_jobs=n_jobs)


def candidate_configs(model_name: str) -> list:
    if model_name == "Logistic":
        return [{"C": c} for c in GRID_LR["C"]]
    if model_name == "XGBoost":
        return [{"max_depth": md, "learning_rate": lr}
                for md, lr in product(GRID_XGB["max_depth"], GRID_XGB["learning_rate"])]
    return [{"min_samples_leaf": m} for m in GRID_RF["min_samples_leaf"]]


def grid_search_model(model_name, X_sub, y_sub, X_val, y_val, n_jobs):
    """Scores every candidate on the inner validation AUC; returns the winner plus the
    full scoreboard so the thesis can report what was tried, not just what won."""
    scored = []
    for params in candidate_configs(model_name):
        model = make_model(model_name, params, y_sub, n_jobs)
        model.fit(X_sub, y_sub)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        scored.append({"params": params, "val_AUC": auc})
        del model
        gc.collect()
    best = max(scored, key=lambda r: r["val_AUC"])
    return best, scored


# ------------------------------------------------------------------
# one fold
# ------------------------------------------------------------------
def run_fold(df, fold, args) -> None:
    tr_idx, te_idx = fold.tr_idx, fold.te_idx
    y_tr = df.loc[tr_idx, "is_default"].values.astype(int)
    y_te = df.loc[te_idx, "is_default"].values.astype(int)

    ids_te = df.loc[te_idx, "id"].values
    issue_d_te = df.loc[te_idx, "issue_d"].values
    has_desc_te = df.loc[te_idx, "has_desc"].values

    n_sub = int(len(tr_idx) * (1 - VAL_FRAC))
    sub_idx, val_idx = tr_idx[:n_sub], tr_idx[n_sub:]
    y_sub, y_val = y_tr[:n_sub], y_tr[n_sub:]

    pred_rows, metric_rows, param_rows = [], [], []

    for representation in REPRESENTATIONS:
        models_here = [m for m in MODELS
                       if not (m == "RandomForest" and representation in args.skip_rf_on)]
        if not models_here:
            continue

        # -- inner split: search --------------------------------------------------
        t0 = time.perf_counter()
        X_sub, X_val, _ = build_inner_matrix(df, sub_idx, val_idx, representation, y_sub, args.n_jobs)
        winners = {}
        for model_name in models_here:
            best, scored = grid_search_model(model_name, X_sub, y_sub, X_val, y_val, args.n_jobs)
            winners[model_name] = best["params"]
            param_rows.append({
                "fold": fold.fold, "representation": representation, "model": model_name,
                "n_sub": len(sub_idx), "n_val": len(val_idx),
                "best_params": json.dumps(best["params"]),
                "best_val_AUC": round(best["val_AUC"], 4),
                "all_candidates": json.dumps(
                    [{"params": s["params"], "val_AUC": round(s["val_AUC"], 4)} for s in scored]),
            })
        del X_sub, X_val
        gc.collect()
        search_s = time.perf_counter() - t0

        # -- outer: refit winners on the full fold train, predict the test fold -----
        X_tr, X_te, feature_names, _audit = reprs.build_final_matrix(
            df, tr_idx, te_idx, representation, fold.fold, y_tr)
        assert_not_in_features(feature_names)

        for model_name in models_here:
            model = make_model(model_name, winners[model_name], y_tr, args.n_jobs)
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]

            m = metrics_mod.predictive_metrics(y_te, p)
            metric_rows.append({
                "fold": fold.fold, "representation": representation, "model": model_name,
                "n_train": fold.n_train, "n_test": fold.n_test,
                "n_defaults": fold.n_test_defaults, "n_features": len(feature_names),
                "best_params": json.dumps(winners[model_name]),
                **{k: round(v, 4) for k, v in m.items()},
            })
            for i in range(len(te_idx)):
                pred_rows.append({
                    "id": int(ids_te[i]), "issue_d": str(issue_d_te[i]), "fold": fold.fold,
                    "model": model_name, "representation": representation,
                    "y_true": int(y_te[i]), "y_prob": float(p[i]),
                    "has_desc": int(has_desc_te[i]),
                })
            del model
            gc.collect()

        del X_tr, X_te
        gc.collect()
        print(f"    {representation}: search {search_s/60:.1f}min, "
              f"total {(time.perf_counter()-t0)/60:.1f}min "
              f"[{', '.join(f'{k}={v}' for k, v in winners.items())}]")

    _append_csv(PRED_CSV, pred_rows)
    _append_csv(METRICS_CSV, metric_rows)
    _append_csv(BEST_PARAMS_CSV, param_rows)
    _mark_complete(fold.fold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                        help="Ignore any existing checkpoint and start clean.")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Run only these fold numbers (e.g. --folds 1). Useful as a smoke test.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="n_jobs for XGBoost and RandomForest (default 4, not -1, to bound "
                             "peak memory on this 13.7GB machine -- see module docstring).")
    parser.add_argument("--skip-rf-on", nargs="*", default=[],
                        help="Representations to skip RandomForest for, e.g. --skip-rf-on tfidf_full. "
                             "Use only if that combination cannot complete; it is a disclosed limitation.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.restart:
        for p in [PRED_CSV, METRICS_CSV, BEST_PARAMS_CSV, MARKER_FILE]:
            if p.exists():
                p.unlink()
        print("--restart: cleared checkpoint.")

    if args.skip_rf_on:
        print(f"NOTE: skipping RandomForest for {args.skip_rf_on} (disclosed limitation).")

    completed = _read_completed()
    if completed:
        print(f"Resuming: folds {sorted(completed)} already done -- skipping.")

    print("Loading raw features...")
    df = load_raw_features()
    sanity_check(df)

    built = folds_mod.build_folds(df)
    folds_mod.assert_fold_integrity(built, df)
    print(f"Fold integrity PASSED ({len(built)} folds).")
    print(f"Scope: {len(REPRESENTATIONS)} representations x {len(MODELS)} models, "
          f"grid search per fold (LR {len(GRID_LR['C'])}, "
          f"XGB {len(GRID_XGB['max_depth'])*len(GRID_XGB['learning_rate'])}, "
          f"RF {len(GRID_RF['min_samples_leaf'])} candidates).")

    run_start = time.perf_counter()
    for fold in built:
        if fold.fold in completed:
            continue
        if args.folds and fold.fold not in args.folds:
            continue
        print(f"\nFold {fold.fold} (n_train={fold.n_train:,}, n_test={fold.n_test:,}):")
        t0 = time.perf_counter()
        run_fold(df, fold, args)
        print(f"  Fold {fold.fold} done in {(time.perf_counter()-t0)/60:.1f}min "
              f"({(time.perf_counter()-run_start)/60:.1f}min elapsed). Checkpointed.")

    print(f"\nComplete. Outputs in {OUT_DIR}")
    print("To run the fairness analysis on these tuned predictions, point census/scripts' "
          "PREDICTIONS_CSV constant at results/strict_temporal_v2_tuned/predictions.csv")


if __name__ == "__main__":
    main()
