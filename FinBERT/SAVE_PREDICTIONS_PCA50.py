"""
Save raw per-fold test predictions — Structured + FinBERT-PCA50, fixed hyperparameters
=============================================================================================
Companion to SAVE_PREDICTIONS_768.py — see that file's docstring for why this exists
(the combined DELONG_768_VS_PCA50.py hit OOM twice at fold 11/14 fitting both variants
in one process). Run as a SEPARATE `python` invocation from SAVE_PREDICTIONS_768.py.

OUTPUT: results/predictions_pca50.csv — columns: fold, Model, id, y_true, p_pred
"""

import os
import sys
import time
import gc
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
PATH_CSV     = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
PATH_FINBERT = SCRIPT_DIR / "data" / "finbert_desc_embeddings.parquet"

TARGET_COL, DATE_COL, ID_COL, ZIP_COL = "is_default", "issue_month_start", "id", "zip3"
MIN_TRAIN_MONTHS, MIN_TEST_DEFAULTS, STEP_MONTHS = 8, 100, 3
RANDOM_STATE = 242
SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
N_PCA_COMPONENTS = 50

LR_PARAMS = dict(penalty="l2", C=0.3, solver="saga", max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=20, class_weight="balanced", n_jobs=4, random_state=RANDOM_STATE)

print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()

GRADE_ORDER = list("ABCDEFG"); SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]
df["grade_ord"] = df["grade"].astype(str).str.strip().map({g: i+1 for i, g in enumerate(GRADE_ORDER)}).astype("float32")
df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map({s: i+1 for i, s in enumerate(SUBGRADE_ORDER)}).astype("float32")

EXCLUDE_COLS = {TARGET_COL, DATE_COL, ZIP_COL, "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "emp_title", "title", "desc", "funded_ratio", "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low"}
STRUCT_COLS = [c for c in df.columns if c not in EXCLUDE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features: {len(STRUCT_COLS)}")

print("Loading FinBERT embeddings...")
df_fb = pd.read_parquet(PATH_FINBERT, columns=[ID_COL, "has_desc"] + SEM_COLS)
for c in SEM_COLS:
    df_fb[c] = df_fb[c].astype(np.float32)
n_before = len(df)
df = df.merge(df_fb, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == n_before
assert df["has_desc"].isna().sum() == 0
df["has_desc"] = df["has_desc"].astype(np.float32)
print(f"  Merged embeddings for {len(df):,} rows")

def build_folds(df, date_col, min_train_months, step_months, min_test_defaults):
    months = df[date_col].dt.to_period("M"); all_periods = sorted(months.unique())
    folds = []; fold_start_idx = min_train_months
    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]
        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, TARGET_COL].sum() >= min_test_defaults: break
            test_end_idx += 1
        if test_end_idx >= len(all_periods): break
        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)
        folds.append({"fold": len(folds)+1, "tr_idx": df.index[tr_mask].tolist(), "te_idx": df.index[te_mask].tolist()})
        fold_start_idx += step_months
    return folds

print("\nBuilding Walk-Forward folds...")
folds = build_folds(df, DATE_COL, MIN_TRAIN_MONTHS, STEP_MONTHS, MIN_TEST_DEFAULTS)
assert len(folds) == 14
print(f"  Total folds: {len(folds)}")

def build_matrix(df_tr, df_te):
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_tr_s = imputer.fit_transform(df_tr[STRUCT_COLS].values.astype(np.float32))
    X_te_s = imputer.transform(df_te[STRUCT_COLS].values.astype(np.float32))
    scaler_s = StandardScaler(); X_tr_s = scaler_s.fit_transform(X_tr_s); X_te_s = scaler_s.transform(X_te_s)

    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver='randomized', random_state=RANDOM_STATE)
    pca_tr = pca.fit_transform(df_tr[SEM_COLS].values.astype(np.float32))
    pca_te = pca.transform(df_te[SEM_COLS].values.astype(np.float32))
    fb_tr_raw = np.hstack([df_tr[["has_desc"]].values.astype(np.float32), pca_tr])
    fb_te_raw = np.hstack([df_te[["has_desc"]].values.astype(np.float32), pca_te])
    scaler_fb = StandardScaler(); X_tr_fb = scaler_fb.fit_transform(fb_tr_raw); X_te_fb = scaler_fb.transform(fb_te_raw)
    return np.hstack([X_tr_s, X_tr_fb]), np.hstack([X_te_s, X_te_fb])

out_dir = SCRIPT_DIR / "results"
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir / "predictions_pca50.csv"

# --- Checkpoint/resume (same discipline as SAVE_PREDICTIONS_768.py, which
# needed 5 attempts before completing due to intermittent kills). ---
done_folds = set()
if out_path.exists():
    existing = pd.read_csv(out_path)
    done_folds = set(existing["fold"].unique().tolist())
    print(f"Resuming: {len(done_folds)} fold(s) already saved ({sorted(done_folds)}) — skipping those.")
else:
    pd.DataFrame(columns=["fold", "Model", "id", "y_true", "p_pred"]).to_csv(out_path, index=False)

run_start = time.perf_counter()

for f in folds:
    if f["fold"] in done_folds:
        continue
    t0 = time.perf_counter()
    df_tr, df_te = df.loc[f["tr_idx"]], df.loc[f["te_idx"]]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)
    ids_te = df_te[ID_COL].values

    X_tr, X_te = build_matrix(df_tr, df_te)

    models = {
        "Logistic": LogisticRegression(**LR_PARAMS),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(), eval_metric="auc", use_label_encoder=False,
            verbosity=0, n_jobs=4, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }
    fold_rows = []
    for mname, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        for i in range(len(y_te)):
            fold_rows.append({"fold": f["fold"], "Model": mname, "id": ids_te[i],
                               "y_true": int(y_te[i]), "p_pred": round(float(p[i]), 6)})
        del model

    pd.DataFrame(fold_rows).to_csv(out_path, mode="a", header=False, index=False)

    del X_tr, X_te, models, df_tr, df_te, fold_rows
    gc.collect()
    dt = time.perf_counter() - t0
    total = time.perf_counter() - run_start
    print(f"  Fold {f['fold']} done and saved. ({dt:.1f}s this fold, {total/60:.1f}min elapsed this session)")

final = pd.read_csv(out_path)
n_folds_done = final["fold"].nunique()
print(f"\nSaved: {out_path}  ({len(final):,} rows, {n_folds_done}/14 folds)")
if n_folds_done == 14:
    print("ALL FOLDS COMPLETE.")
else:
    print(f"INCOMPLETE — {14 - n_folds_done} fold(s) remaining. Rerun this script to resume.")
print(f"This session's run time: {(time.perf_counter()-run_start)/60:.1f} minutes")
