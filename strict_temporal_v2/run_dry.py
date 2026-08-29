# -*- coding: utf-8 -*-
"""
run_dry.py
===========
Smoke test: runs the full Structured vs Structured+representation
comparison on ONLY fold 1 (and fold 14, if --with-fold14 is passed), for
all 7 representations x 2 models (Logistic, XGBoost -- no RandomForest,
per spec). Reports shapes, feature counts, AUC, wall time, and peak
memory. Does NOT write to results/strict_temporal_v2/ -- that only
happens in run_full.py, after the dry run is reviewed.

Usage:
  python -m strict_temporal_v2.run_dry                # fold 1 only
  python -m strict_temporal_v2.run_dry --with-fold14   # fold 1 + fold 14
"""

import argparse
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from strict_temporal_v2 import folds as folds_mod
from strict_temporal_v2 import hyperparams as hp
from strict_temporal_v2 import representations as reprs
from strict_temporal_v2.raw_features import load_raw_features, sanity_check
from strict_temporal_v2.fairness_stub import assert_not_in_features


def run_one(df, fold, representation: str) -> dict:
    t0 = time.perf_counter()
    tr_idx, te_idx = fold.tr_idx, fold.te_idx
    y_tr = df.loc[tr_idx, "is_default"].values.astype(int)
    y_te = df.loc[te_idx, "is_default"].values.astype(int)

    X_tr, X_te, feature_names, audit = reprs.build_final_matrix(df, tr_idx, te_idx, representation, fold.fold, y_tr)
    assert_not_in_features(feature_names)

    row = {"representation": representation, "fold": fold.fold, "n_features": len(feature_names)}
    for model_name in ["Logistic", "XGBoost"]:
        if model_name == "Logistic":
            model = LogisticRegression(**hp.FIXED_LR)  # saga handles sparse (TF-IDF) and dense alike
        else:
            model = xgb.XGBClassifier(**hp.build_xgb_params(y_tr), use_label_encoder=False, verbosity=0)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        row[f"AUC_{model_name}"] = round(roc_auc_score(y_te, p), 4)

    row["wall_time_s"] = round(time.perf_counter() - t0, 1)
    row.update({k: v for k, v in audit.items() if k in ("tfidf_vocab_size", "pca_explained_variance")})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-fold14", action="store_true")
    args = parser.parse_args()

    tracemalloc.start()
    print("Loading raw features...")
    df = load_raw_features()
    sanity_check(df)

    built = folds_mod.build_folds(df)
    folds_mod.assert_fold_integrity(built, df)
    print(f"Fold integrity assertions PASSED ({len(built)} folds).")

    target_folds = [built[0]] + ([built[13]] if args.with_fold14 else [])

    rows = []
    for fold in target_folds:
        print(f"\n=== Fold {fold.fold} (train_cutoff={fold.train_cutoff}, "
              f"n_train={fold.n_train:,}, n_test={fold.n_test:,}) ===")
        for representation in reprs.REPRESENTATIONS:
            print(f"  {representation}...", end=" ", flush=True)
            row = run_one(df, fold, representation)
            rows.append(row)
            print(f"AUC(LR)={row['AUC_Logistic']} AUC(XGB)={row['AUC_XGBoost']} "
                  f"n_feat={row['n_features']} time={row['wall_time_s']}s")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("DRY RUN SUMMARY")
    print("=" * 70)
    print(result_df[["fold", "representation", "n_features", "AUC_Logistic", "AUC_XGBoost", "wall_time_s"]]
          .to_string(index=False))
    print(f"\nPeak traced Python memory: {peak / 1e9:.2f} GB "
          f"(process RSS is typically higher -- this excludes C-level allocations "
          f"like XGBoost's internal buffers)")
    print(f"Total wall time: {result_df['wall_time_s'].sum():.1f}s")


if __name__ == "__main__":
    main()
