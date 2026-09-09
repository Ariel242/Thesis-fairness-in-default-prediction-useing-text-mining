# -*- coding: utf-8 -*-
"""
compute_delong_tuned.py
=========================
run_thesis_final.py (the grid-searched final run, results/strict_temporal_v2_tuned/)
writes predictions.csv / fold_metrics.csv / best_params.csv but never computes the
DeLong significance test -- that logic only ever ran inside run_full.py, against the
older, untuned results/strict_temporal_v2/ predictions. This script closes that gap
post-hoc, from the already-saved tuned predictions, with NO retraining: per fold x
model, it pairs each non-structured representation's predicted probabilities against
that same fold/model's structured predictions (same held-out test set, same ids -- a
prerequisite for DeLong's paired test), and applies the identical delong_auc_test /
add_bh_correction functions run_full.py itself uses (metrics.py), so the tuned run's
significance test is directly comparable to the untuned one's.

INPUT:  results/strict_temporal_v2_tuned/predictions.csv
OUTPUT: results/strict_temporal_v2_tuned/delong_raw.csv  (per fold x model x representation)
        results/strict_temporal_v2_tuned/delong.csv      (+ BH q-value, within each Model)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from strict_temporal_v2 import metrics as metrics_mod

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "strict_temporal_v2_tuned"
PRED_CSV = OUT_DIR / "predictions.csv"
DELONG_RAW_CSV = OUT_DIR / "delong_raw.csv"
DELONG_FINAL_CSV = OUT_DIR / "delong.csv"


def main() -> None:
    preds = pd.read_csv(PRED_CSV)
    models = sorted(preds["model"].unique())
    representations = sorted(preds["representation"].unique())
    folds = sorted(preds["fold"].unique())
    print(f"Loaded {len(preds):,} prediction rows: {len(folds)} folds x "
          f"{len(models)} models {models} x {len(representations)} representations {representations}.")

    if "structured" not in representations:
        raise ValueError("'structured' representation not found in predictions.csv -- "
                          "DeLong needs it as the comparison baseline.")

    delong_rows = []
    for fold in folds:
        fold_df = preds[preds["fold"] == fold]
        for model_name in models:
            model_df = fold_df[fold_df["model"] == model_name]
            struct = model_df[model_df["representation"] == "structured"].set_index("id")

            for representation in representations:
                if representation == "structured":
                    continue
                rep_df = model_df[model_df["representation"] == representation].set_index("id")

                # Align on id -- both arms must share the exact same held-out test set
                # for DeLong's paired-covariance assumption to hold.
                common_ids = struct.index.intersection(rep_df.index)
                if len(common_ids) != len(struct) or len(common_ids) != len(rep_df):
                    raise ValueError(
                        f"fold {fold}, model {model_name}, representation {representation}: "
                        f"id mismatch between structured (n={len(struct)}) and "
                        f"{representation} (n={len(rep_df)}), common={len(common_ids)}. "
                        f"DeLong requires the identical test set for both arms."
                    )
                struct_aligned = struct.loc[common_ids]
                rep_aligned = rep_df.loc[common_ids]
                if not (struct_aligned["y_true"].values == rep_aligned["y_true"].values).all():
                    raise ValueError(
                        f"fold {fold}, model {model_name}, representation {representation}: "
                        f"y_true mismatch after id alignment -- data integrity problem."
                    )

                y_te = struct_aligned["y_true"].values
                p_struct = struct_aligned["y_prob"].values
                p_repr = rep_aligned["y_prob"].values

                auc1, auc2, z, pval = metrics_mod.delong_auc_test(y_te, p_struct, p_repr)
                delong_rows.append({
                    "fold": int(fold), "Model": model_name, "representation": representation,
                    "AUC_structured": round(auc1, 4), "AUC_representation": round(auc2, 4),
                    "Delta_AUC": round(auc2 - auc1, 4),
                    "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
                    "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
                })
        print(f"  fold {fold} done.")

    delong_raw = pd.DataFrame(delong_rows)
    delong_raw.to_csv(DELONG_RAW_CSV, index=False)
    print(f"\nSaved: {DELONG_RAW_CSV} ({len(delong_raw)} rows)")

    delong_final = metrics_mod.add_bh_correction(delong_raw, group_cols=["Model"], p_col="p_value")
    delong_final.to_csv(DELONG_FINAL_CSV, index=False)
    print(f"Saved: {DELONG_FINAL_CSV} (BH-adjusted within each Model)")

    print("\n-- Mean Delta_AUC and significance count, per model x representation --")
    summary = delong_final.groupby(["Model", "representation"]).agg(
        mean_Delta_AUC=("Delta_AUC", "mean"),
        n_sig_raw_p05=("p_value", lambda s: (s < 0.05).sum()),
        n_sig_BH_q05=("q_value", lambda s: (s < 0.05).sum()),
        n_folds=("fold", "count"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
