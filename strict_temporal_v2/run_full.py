# -*- coding: utf-8 -*-
"""
run_full.py
============
The main comparison: all 14 walk-forward folds x 7 representations x 2
models (Logistic, XGBoost -- no RandomForest, per spec). Writes to
results/strict_temporal_v2/ ONLY -- never touches any existing results
folder. Should only be run after run_dry.py has been reviewed and
approved (per the plan's own gate) -- this script does not check that
for you.

CHECKPOINTED: each fold's rows are appended to disk immediately after
that fold finishes (not held in memory across all 14 folds), and a
completed_folds.txt marker tracks progress. If the process is killed
partway (this happened once, likely OOM, with zero output saved by the
earlier non-checkpointed version), simply re-running this same command
skips every already-completed fold and continues from where it stopped.

Usage:
  python -m strict_temporal_v2.run_full            # run/resume all 14 folds
  python -m strict_temporal_v2.run_full --restart   # ignore checkpoint, start clean

Outputs (results/strict_temporal_v2/):
  predictions.csv           -- id, issue_d, fold, model, representation,
                                y_true, y_prob, has_desc
  fold_metrics.csv          -- AUC/PR-AUC/GINI/Brier + bootstrap CI per
                                fold x model x representation
  preprocessing_audit.jsonl -- one JSON object per fold x representation
                                (missing-cols removed, correlated pairs,
                                clip thresholds, feature counts, TF-IDF
                                vocab size, chi2 terms, PCA variance)
  delong_raw.csv            -- per fold x model x representation, raw
                                DeLong p-value (written incrementally)
  delong.csv                -- delong_raw.csv + BH-adjusted q-value,
                                written once at the end (or by
                                finalize_delong() run standalone)
  completed_folds.txt       -- one fold number per line -- the resume marker
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from strict_temporal_v2 import folds as folds_mod
from strict_temporal_v2 import hyperparams as hp
from strict_temporal_v2 import metrics as metrics_mod
from strict_temporal_v2 import representations as reprs
from strict_temporal_v2.raw_features import load_raw_features, sanity_check
from strict_temporal_v2.fairness_stub import assert_not_in_features

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "strict_temporal_v2"
PRED_CSV = OUT_DIR / "predictions.csv"
METRICS_CSV = OUT_DIR / "fold_metrics.csv"
AUDIT_JSONL = OUT_DIR / "preprocessing_audit.jsonl"
DELONG_RAW_CSV = OUT_DIR / "delong_raw.csv"
DELONG_FINAL_CSV = OUT_DIR / "delong.csv"
MARKER_FILE = OUT_DIR / "completed_folds.txt"


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _read_completed_folds() -> set[int]:
    if not MARKER_FILE.exists():
        return set()
    return {int(line.strip()) for line in MARKER_FILE.read_text().splitlines() if line.strip()}


def _mark_fold_complete(fold_num: int) -> None:
    with open(MARKER_FILE, "a", encoding="utf-8") as f:
        f.write(f"{fold_num}\n")


def finalize_delong() -> None:
    """Reads the accumulated delong_raw.csv (all folds' raw p-values) and
    writes delong.csv with BH-adjusted q-values added. Safe to re-run at
    any point -- recomputed from delong_raw.csv, never destructive."""
    if not DELONG_RAW_CSV.exists():
        print("No delong_raw.csv yet -- nothing to finalize.")
        return
    df_delong = pd.read_csv(DELONG_RAW_CSV)
    df_delong = metrics_mod.add_bh_correction(df_delong, group_cols=["Model"], p_col="p_value")
    df_delong.to_csv(DELONG_FINAL_CSV, index=False)
    print(f"Wrote {DELONG_FINAL_CSV} ({len(df_delong)} rows, BH-adjusted within each Model).")


def run_fold(df, fold, y_tr, y_te) -> None:
    tr_idx, te_idx = fold.tr_idx, fold.te_idx
    ids_te = df.loc[te_idx, "id"].values
    issue_d_te = df.loc[te_idx, "issue_d"].values
    has_desc_te = df.loc[te_idx, "has_desc"].values

    pred_rows, metric_rows, audit_rows, delong_rows = [], [], [], []
    fold_probs = {"Logistic": {}, "XGBoost": {}}

    for representation in reprs.REPRESENTATIONS:
        X_tr, X_te, feature_names, audit = reprs.build_final_matrix(
            df, tr_idx, te_idx, representation, fold.fold, y_tr)
        assert_not_in_features(feature_names)
        audit_rows.append({"fold": fold.fold, "representation": representation, **audit})

        for model_name in ["Logistic", "XGBoost"]:
            if model_name == "Logistic":
                model = LogisticRegression(**hp.FIXED_LR)
            else:
                model = xgb.XGBClassifier(**hp.build_xgb_params(y_tr), use_label_encoder=False, verbosity=0)
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]
            fold_probs[model_name][representation] = p

            m = metrics_mod.predictive_metrics(y_te, p)
            auc_lo, auc_hi = metrics_mod.bootstrap_ci(
                y_te, p, lambda yt, pp: metrics_mod.predictive_metrics(yt, pp)["AUC"])
            metric_rows.append({
                "fold": fold.fold, "representation": representation, "Model": model_name,
                "n_train": fold.n_train, "n_test": fold.n_test, "n_defaults": fold.n_test_defaults,
                "n_features": len(feature_names),
                **{k: round(v, 4) for k, v in m.items()},
                "AUC_CI_lo": round(auc_lo, 4) if not np.isnan(auc_lo) else np.nan,
                "AUC_CI_hi": round(auc_hi, 4) if not np.isnan(auc_hi) else np.nan,
            })

            for i in range(len(te_idx)):
                pred_rows.append({
                    "id": int(ids_te[i]), "issue_d": str(issue_d_te[i]), "fold": fold.fold,
                    "model": model_name, "representation": representation,
                    "y_true": int(y_te[i]), "y_prob": float(p[i]), "has_desc": int(has_desc_te[i]),
                })

        del X_tr, X_te  # drop the (possibly wide) matrix before the next representation

    for model_name in ["Logistic", "XGBoost"]:
        p_struct = fold_probs[model_name]["structured"]
        for representation in reprs.REPRESENTATIONS:
            if representation == "structured":
                continue
            auc1, auc2, z, pval = metrics_mod.delong_auc_test(
                y_te, p_struct, fold_probs[model_name][representation])
            delong_rows.append({
                "fold": fold.fold, "Model": model_name, "representation": representation,
                "AUC_structured": round(auc1, 4), "AUC_representation": round(auc2, 4),
                "Delta_AUC": round(auc2 - auc1, 4),
                "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
                "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
            })

    # Flush this fold's rows to disk immediately, then let them go out of scope.
    _append_csv(PRED_CSV, pred_rows)
    _append_csv(METRICS_CSV, metric_rows)
    _append_jsonl(AUDIT_JSONL, audit_rows)
    _append_csv(DELONG_RAW_CSV, delong_rows)
    _mark_fold_complete(fold.fold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="Ignore any existing checkpoint and start clean.")
    parser.add_argument("--xgb-n-jobs", type=int, default=None,
                         help="Override hyperparams.FIXED_XGB['n_jobs'] (spec default: -1) for this "
                              "invocation only -- a disclosed, approved deviation for resource-constrained "
                              "retries (e.g. fold 14's ~55K-column TF-IDF arm hit an apparent OOM kill twice "
                              "at n_jobs=-1; Ariel approved --xgb-n-jobs 4 for the retry). Does not change "
                              "hyperparams.py's own documented default.")
    args = parser.parse_args()

    if args.xgb_n_jobs is not None:
        print(f"NOTE: overriding XGBoost n_jobs to {args.xgb_n_jobs} for this run "
              f"(spec default is -1; disclosed, approved resource-safety deviation).")
        hp.FIXED_XGB["n_jobs"] = args.xgb_n_jobs

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.restart:
        for p in [PRED_CSV, METRICS_CSV, AUDIT_JSONL, DELONG_RAW_CSV, DELONG_FINAL_CSV, MARKER_FILE]:
            if p.exists():
                p.unlink()
        print("--restart: cleared any existing strict_temporal_v2 checkpoint.")

    completed = _read_completed_folds()
    if completed:
        print(f"Resuming: {len(completed)} fold(s) already completed ({sorted(completed)}) -- will skip these.")

    print("Loading raw features...")
    df = load_raw_features()
    sanity_check(df)

    built = folds_mod.build_folds(df)
    folds_mod.assert_fold_integrity(built, df)
    print(f"Fold integrity assertions PASSED ({len(built)} folds).")

    for fold in built:
        if fold.fold in completed:
            print(f"Fold {fold.fold}: already completed -- skipping.")
            continue
        t0 = time.perf_counter()
        y_tr = df.loc[fold.tr_idx, "is_default"].values.astype(int)
        y_te = df.loc[fold.te_idx, "is_default"].values.astype(int)
        run_fold(df, fold, y_tr, y_te)
        gc.collect()  # release fitted models / matrices before the next (possibly larger) fold
        print(f"Fold {fold.fold} done in {time.perf_counter() - t0:.1f}s. "
              f"(checkpointed -- safe to stop and resume from here)")

    finalize_delong()
    print(f"\nAll folds complete. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
