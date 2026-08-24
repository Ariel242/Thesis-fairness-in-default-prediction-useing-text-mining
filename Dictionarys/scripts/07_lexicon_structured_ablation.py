"""
Structured vs Structured+Lexicon — Walk-Forward Ablation
===========================================================
Downstream evaluation step of the Distress Lexicon protocol (see
Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf, section 2,
"Downstream Walk-Forward Evaluation"). Companion to
TF-IDF/tfidf_structured_ablation.py and FinBERT/finbert_structured_ablation.py
-- same walk-forward protocol, same fixed hyperparameters (copied verbatim
from analysis/BASELINE.py, per this repo's no-shared-module convention), same
metrics -- applied to the frozen Distress Lexicon's 7 features instead of
TF-IDF or FinBERT embeddings. Neither of those scripts nor BASELINE.py is
imported/edited/executed here.

--------------------------------------------------------------------------------------------
1. THE TWO ARMS
--------------------------------------------------------------------------------------------
"Structured"          -- every numeric, non-leaky, non-identifier column from the
                          post-03_advanced_prep CSV (identical definition to BASELINE.py's
                          STRUCT_COLS / the other two ablations).
"Structured+Lexicon"  -- Structured PLUS the 7 lexicon features from
                          Dictionarys/output/lexicon_features.csv (joined by id).
                          Unlike TF-IDF (refit per fold) or FinBERT (embeddings per fold),
                          the lexicon is FROZEN -- these 7 columns are identical for every
                          fold; only the imputer/scaler is refit on each fold's training
                          rows, matching how the Structured block itself is handled.

--------------------------------------------------------------------------------------------
2. HYPERPARAMETERS -- FIXED, NOT TUNED PER ARM
--------------------------------------------------------------------------------------------
Reuses BASELINE.py's fixed hyperparameters (LR C=0.3, RF n_estimators=500 /
min_samples_leaf=20, XGBoost depth=4/lr=0.05/n=300) unchanged across both arms,
same as the TF-IDF ablation's "Structured" arm and the RandomForest/XGBoost/Logistic
trio used throughout this repo's other ablations.

--------------------------------------------------------------------------------------------
3. OUTPUTS (written under Dictionarys/, per the boundary agreed for this pipeline)
--------------------------------------------------------------------------------------------
results/
  lexicon_ablation_folds.csv               -- per fold x arm x model: AUC/PR-AUC/Brier/GINI + 95% CI
  lexicon_ablation_summary.csv             -- mean +/- std across folds, grouped by (arm, model)
  lexicon_ablation_delong.csv              -- pairwise DeLong per fold x model (1 pair: Structured vs Structured+Lexicon)
  lexicon_ablation_feature_importance.csv  -- per fold x arm x model x feature: importance/rank (all features kept -- only 7 lexicon + ~40 structured)
  lexicon_ablation_report.md               -- written methodology/results/conclusions document
figures/
  lexicon_fig1_auc_by_fold.png             -- AUC per fold, per model, 2 arms overlaid, with CI bands
  lexicon_fig2_delong_deltas.png           -- ΔAUC bar chart per model, colored by significance
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../Dictionarys/scripts
DICT_DIR   = SCRIPT_DIR.parent                          # .../Dictionarys
BASE_DIR   = DICT_DIR.parent                            # .../F-TM-CR

PATH_CSV     = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
LEXICON_CSV  = DICT_DIR / "output" / "lexicon_features.csv"

TARGET_COL = "is_default"
DATE_COL   = "issue_month_start"
ID_COL     = "id"

LEXICON_COLS = [
    "risk_word_count", "risk_word_density", "risk_word_flag",
    "distress_debt_pressure_count", "distress_liquidity_shortage_count",
    "distress_medical_life_shock_count", "distress_refinancing_burden_count",
]

# Walk-Forward parameters (identical to BASELINE.py / TF-IDF / FinBERT ablations -- must yield 14 folds)
MIN_TRAIN_MONTHS  = 8
MIN_TEST_DEFAULTS = 100
STEP_MONTHS       = 3

N_BOOTSTRAP = 500
BOOT_SEED   = 42
RANDOM_STATE = 242

ARMS        = ["Structured", "Structured+Lexicon"]
MODEL_NAMES = ["Logistic", "XGBoost", "RandomForest"]

DEBUG_MAX_FOLDS = None  # set e.g. 3 to smoke-test before the full 14-fold run

LR_PARAMS = dict(
    penalty="l2", C=0.3, solver="saga", max_iter=1000,
    class_weight="balanced", random_state=RANDOM_STATE,
)


def build_model(model_name, y_tr):
    if model_name == "Logistic":
        return LogisticRegression(**LR_PARAMS)
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
            eval_metric="auc", use_label_encoder=False, verbosity=0,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    raise ValueError(model_name)


def extract_importance(model_name, model, n_features):
    if model_name == "Logistic":
        return np.abs(model.coef_[0])
    elif model_name == "XGBoost":
        gain = model.get_booster().get_score(importance_type="gain")
        return np.array([gain.get(f"f{i}", 0.0) for i in range(n_features)])
    else:
        return model.feature_importances_


# ============================================================
# 1. LOAD STRUCTURED DATA + JOIN LEXICON FEATURES
# ============================================================
print("Loading structured data...")
df = pd.read_csv(PATH_CSV, parse_dates=[DATE_COL], low_memory=False)
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
print(f"  Rows after filtering: {len(df):,}  |  Default rate: {df[TARGET_COL].mean():.2%}")

print("Joining frozen lexicon features...")
lex = pd.read_csv(LEXICON_CSV)
df = df.merge(lex, on=ID_COL, how="left", validate="one_to_one")
missing_lex = df[LEXICON_COLS].isna().any(axis=1).sum()
if missing_lex:
    print(f"  WARNING: {missing_lex} rows had no lexicon feature match -- filled with 0.")
    df[LEXICON_COLS] = df[LEXICON_COLS].fillna(0)
print(f"  Lexicon columns joined: {LEXICON_COLS}")

# ============================================================
# 1b. RESTORE ORDINAL RISK GRADES (identical fix to BASELINE.py / other ablations)
# ============================================================
GRADE_ORDER    = list("ABCDEFG")
SUBGRADE_ORDER = [f"{g}{i}" for g in GRADE_ORDER for i in range(1, 6)]

if "grade" in df.columns:
    grade_map = {g: i + 1 for i, g in enumerate(GRADE_ORDER)}
    df["grade_ord"] = df["grade"].astype(str).str.strip().map(grade_map).astype("float32")
if "sub_grade" in df.columns:
    subgrade_map = {s: i + 1 for i, s in enumerate(SUBGRADE_ORDER)}
    df["sub_grade_ord"] = df["sub_grade"].astype(str).str.strip().map(subgrade_map).astype("float32")
print("  Ordinal risk grades restored: grade_ord (1-7), sub_grade_ord (1-35)")

# ============================================================
# 2. STRUCTURED FEATURE LIST (identical definition to BASELINE.py's STRUCT_COLS)
# ============================================================
EXCLUDE_COLS = {TARGET_COL, DATE_COL,
                "id", "issue_d", "issue_ym", "month_idx", "issue_month_start",
                "zip_code", "zip3", "emp_title", "title", "desc", "funded_ratio",
                "text_all_clean", "desc_clean", "title_clean", "emp_title_clean",
                "grade", "sub_grade",
                "last_fico_range_high", "last_fico_range_low",
                } | set(LEXICON_COLS)  # lexicon cols added explicitly per arm, not part of "Structured"
STRUCT_COLS = [c for c in df.columns
               if c not in EXCLUDE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                                   np.int8, "Int64", "float32", "float64"]]
print(f"  Structured features: {len(STRUCT_COLS)}")
print(f"  Lexicon features: {len(LEXICON_COLS)}")

# ============================================================
# 3. DELONG TEST (copied verbatim from FinBERT/TF-IDF ablations)
# ============================================================
def delong_auc_test(y_true, p1, p2):
    def auc_and_kernel(y, p):
        pos = p[y == 1]; neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([np.mean(pi > neg) + 0.5*np.mean(pi == neg) for pi in pos])
        V01 = np.array([np.mean(pj < pos) + 0.5*np.mean(pj == pos) for pj in neg])
        return V10.mean(), V10, V01, n1, n0

    y = np.asarray(y_true).astype(int)
    auc1, V10_1, V01_1, n1, n0 = auc_and_kernel(y, np.asarray(p1))
    auc2, V10_2, V01_2, _,  _  = auc_and_kernel(y, np.asarray(p2))

    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)
    var_diff = (S10[0,0]/n1 + S01[0,0]/n0) + (S10[1,1]/n1 + S01[1,1]/n0) \
             - 2*(S10[0,1]/n1 + S01[0,1]/n0)
    if var_diff <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan, np.nan
    diff = auc1 - auc2
    se   = np.sqrt(var_diff)
    z    = diff / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, pval, diff - 1.96*se, diff + 1.96*se

# ============================================================
# 4. BOOTSTRAP CI (copied verbatim from BASELINE.py / other ablations)
# ============================================================
def bootstrap_ci(y_true, p, metric_fn, n=N_BOOTSTRAP, seed=BOOT_SEED, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
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

# ============================================================
# 5. WALK-FORWARD FOLDS (identical logic to BASELINE.py / TF-IDF / FinBERT ablations)
# ============================================================
def build_folds(df, date_col, min_train_months, step_months, min_test_defaults):
    months      = df[date_col].dt.to_period("M")
    all_periods = sorted(months.unique())
    folds = []
    fold_start_idx = min_train_months
    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]
        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, TARGET_COL].sum() >= min_test_defaults:
                break
            test_end_idx += 1
        if test_end_idx >= len(all_periods):
            break
        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)
        folds.append({
            "fold": len(folds) + 1,
            "train_cutoff": str(train_cutoff),
            "test_end": str(test_period),
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "n_test_defaults": int(df.loc[te_mask, TARGET_COL].sum()),
            "tr_idx": df.index[tr_mask].tolist(),
            "te_idx": df.index[te_mask].tolist(),
        })
        fold_start_idx += step_months
    return folds

print("\nBuilding Walk-Forward folds...")
folds = build_folds(df, DATE_COL, MIN_TRAIN_MONTHS, STEP_MONTHS, MIN_TEST_DEFAULTS)
assert len(folds) == 14, (
    f"Expected exactly 14 walk-forward folds, got {len(folds)}. "
    f"Check MIN_TRAIN_MONTHS/STEP_MONTHS/MIN_TEST_DEFAULTS or whether PATH_CSV changed."
)
print(f"  Total folds: {len(folds)}")
for f in folds:
    print(f"  Fold {f['fold']}: train <= {f['train_cutoff']}  |  test {f['train_cutoff']} - {f['test_end']}  |  "
          f"n_train={f['n_train']:,}  n_test={f['n_test']:,}  defaults_in_test={f['n_test_defaults']}")

folds_to_run = folds[:DEBUG_MAX_FOLDS] if DEBUG_MAX_FOLDS else folds
if DEBUG_MAX_FOLDS:
    print(f"\n  DEBUG_MAX_FOLDS={DEBUG_MAX_FOLDS} — only running the first {len(folds_to_run)} fold(s) this run.")

# ============================================================
# 6. MAIN WALK-FORWARD EVALUATION
# ============================================================
print("\n" + "="*70)
print("MAIN WALK-FORWARD EVALUATION")
print("="*70)

pred_rows, delong_rows, importance_rows = [], [], []

for f in folds_to_run:
    t_fold0 = time.time()
    tr, te = f["tr_idx"], f["te_idx"]
    df_tr, df_te = df.loc[tr], df.loc[te]
    y_tr = df_tr[TARGET_COL].values.astype(int)
    y_te = df_te[TARGET_COL].values.astype(int)

    arm_cols = {
        "Structured": STRUCT_COLS,
        "Structured+Lexicon": STRUCT_COLS + LEXICON_COLS,
    }

    fold_preds = {m: {} for m in MODEL_NAMES}

    for arm in ARMS:
        cols = arm_cols[arm]
        imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
        X_tr = imputer.fit_transform(df_tr[cols].values.astype(np.float32))
        X_te = imputer.transform(df_te[cols].values.astype(np.float32))
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        for model_name in MODEL_NAMES:
            t0 = time.time()
            model = build_model(model_name, y_tr)
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]
            elapsed = time.time() - t0
            print(f"    [{model_name:<12s}] arm={arm:<20s} fold={f['fold']:>2}  "
                  f"n_features={len(cols):>4}  fit+predict={elapsed:6.1f}s")

            fold_preds[model_name][arm] = p

            imp = extract_importance(model_name, model, len(cols))
            imp_sum = imp.sum()
            imp_share = imp / imp_sum if imp_sum > 0 else imp
            rank_of = np.empty(len(cols), dtype=int)
            rank_of[np.argsort(imp)[::-1]] = np.arange(1, len(cols) + 1)

            for pos, feat_name in enumerate(cols):
                importance_rows.append({
                    "fold": f["fold"], "Arm": arm, "Model": model_name,
                    "Feature": feat_name,
                    "Kind": "lexicon" if feat_name in LEXICON_COLS else "structured",
                    "Importance_raw": round(float(imp[pos]), 6),
                    "Importance_share": round(float(imp_share[pos]), 6),
                    "Rank": int(rank_of[pos]),
                })

            auc   = roc_auc_score(y_te, p)
            prauc = average_precision_score(y_te, p)
            brier = brier_score_loss(y_te, p)
            auc_lo,   auc_hi   = bootstrap_ci(y_te, p, roc_auc_score)
            prauc_lo, prauc_hi = bootstrap_ci(y_te, p, average_precision_score)
            brier_lo, brier_hi = bootstrap_ci(y_te, p, brier_score_loss)

            pred_rows.append({
                "fold": f["fold"], "train_cutoff": f["train_cutoff"], "test_end": f["test_end"],
                "n_train": f["n_train"], "n_test": f["n_test"], "n_defaults": f["n_test_defaults"],
                "Model": model_name, "Arm": arm, "n_features": len(cols),
                "AUC": round(auc, 4), "AUC_CI_lo": round(auc_lo, 4) if not np.isnan(auc_lo) else np.nan,
                "AUC_CI_hi": round(auc_hi, 4) if not np.isnan(auc_hi) else np.nan,
                "PR_AUC": round(prauc, 4), "PR_AUC_CI_lo": round(prauc_lo, 4) if not np.isnan(prauc_lo) else np.nan,
                "PR_AUC_CI_hi": round(prauc_hi, 4) if not np.isnan(prauc_hi) else np.nan,
                "Brier": round(brier, 4), "Brier_CI_lo": round(brier_lo, 4) if not np.isnan(brier_lo) else np.nan,
                "Brier_CI_hi": round(brier_hi, 4) if not np.isnan(brier_hi) else np.nan,
                "GINI": round(2*auc - 1, 4),
            })
            del model
        del X_tr, X_te
        gc.collect()

    y_te_full = df.loc[te, TARGET_COL].values.astype(int)
    for model_name in MODEL_NAMES:
        auc_a, auc_b, z, pval, ci_lo, ci_hi = delong_auc_test(
            y_te_full, fold_preds[model_name]["Structured"], fold_preds[model_name]["Structured+Lexicon"])
        delong_rows.append({
            "fold": f["fold"], "Model": model_name,
            "Arm_A": "Structured", "Arm_B": "Structured+Lexicon",
            "AUC_A": round(auc_a, 4), "AUC_B": round(auc_b, 4),
            "Delta_AUC": round(auc_b - auc_a, 4),
            "Z_stat": round(z, 3) if not np.isnan(z) else np.nan,
            "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "CI_95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "CI_95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
        })

    del fold_preds
    gc.collect()
    print(f"  Fold {f['fold']} done in {time.time()-t_fold0:.1f}s.")

# ============================================================
# 7. AGGREGATE RESULTS
# ============================================================
df_pred   = pd.DataFrame(pred_rows)
df_delong = pd.DataFrame(delong_rows)
df_imp    = pd.DataFrame(importance_rows)

PRED_METRICS = ["AUC", "PR_AUC", "Brier", "GINI"]
pred_summary = df_pred.groupby(["Arm", "Model"])[PRED_METRICS].agg(["mean", "std"]).round(4)
print("\n" + "="*70)
print("A. PREDICTIVE PERFORMANCE SUMMARY (mean +/- std across folds)")
print("="*70)
print(pred_summary.to_string())

print("\n" + "="*70)
print("B. DELONG PAIRWISE TEST — mean across folds")
print("="*70)
delong_mean = df_delong.groupby(["Model"])[["Delta_AUC", "p_value"]].mean().round(4)
print(delong_mean.to_string())

# --- Cross-fold consistency of lexicon feature importance ---
lex_imp = df_imp[(df_imp["Arm"] == "Structured+Lexicon") & (df_imp["Kind"] == "lexicon")]
consistency_parts = []
for model_name in MODEL_NAMES:
    sub = lex_imp[lex_imp["Model"] == model_name]
    grp = sub.groupby("Feature").agg(
        mean_rank=("Rank", "mean"),
        mean_importance_share=("Importance_share", "mean"),
    ).reset_index()
    grp["Model"] = model_name
    grp["mean_rank"] = grp["mean_rank"].round(2)
    grp["mean_importance_share"] = grp["mean_importance_share"].round(4)
    consistency_parts.append(grp)
df_consistency = pd.concat(consistency_parts, ignore_index=True)
df_consistency = df_consistency.sort_values(["Model", "mean_rank"]).reset_index(drop=True)

print("\n" + "="*70)
print("C. LEXICON FEATURE IMPORTANCE (mean rank / share across folds)")
print("="*70)
for model_name in MODEL_NAMES:
    sub = df_consistency[df_consistency["Model"] == model_name]
    print(f"\n  [{model_name}]")
    print(sub.to_string(index=False))

# ============================================================
# 8. SAVE CSVs
# ============================================================
out_dir = DICT_DIR / "results"
os.makedirs(out_dir, exist_ok=True)

df_pred.to_csv(out_dir / "lexicon_ablation_folds.csv", index=False)
pred_summary.to_csv(out_dir / "lexicon_ablation_summary.csv")
df_delong.to_csv(out_dir / "lexicon_ablation_delong.csv", index=False)
df_imp.to_csv(out_dir / "lexicon_ablation_feature_importance.csv", index=False)
print(f"\nSaved CSVs to: {out_dir}")

# ============================================================
# 9. FIGURES
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

figures_dir = DICT_DIR / "figures"
os.makedirs(figures_dir, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150})

_CLR = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}
_ARM_STYLE = {
    "Structured": dict(linestyle="-", marker="o", alpha=1.00),
    "Structured+Lexicon": dict(linestyle="--", marker="s", alpha=0.80),
}

print("\nGenerating figures...")

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5), sharey=True)
for ax, model in zip(axes, MODEL_NAMES):
    color = _CLR[model]
    for arm in ARMS:
        style = _ARM_STYLE[arm]
        sub = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == arm)].sort_values("fold")
        ax.plot(sub["fold"], sub["AUC"], color=color, linewidth=2, markersize=5, label=arm, **style)
        if not sub["AUC_CI_lo"].isna().all():
            ax.fill_between(sub["fold"], sub["AUC_CI_lo"], sub["AUC_CI_hi"], alpha=0.10, color=color)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (chronological)")
    if ax is axes[0]:
        ax.set_ylabel("AUC")
    ax.legend(title="Arm", fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.suptitle("Walk-Forward AUC Stability — Structured vs Structured+Lexicon\n"
             "(shaded band = 95% bootstrap CI)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(figures_dir / "lexicon_fig1_auc_by_fold.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  lexicon_fig1_auc_by_fold.png")

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6*len(MODEL_NAMES), 4.2), sharey=True)
for j, model in enumerate(MODEL_NAMES):
    ax = axes[j]
    sub = df_delong[df_delong["Model"] == model].sort_values("fold")
    pvals = sub["p_value"].fillna(1.0).values
    bar_colors = ["#d73027" if p < 0.05 else "#bababa" for p in pvals]
    x = sub["fold"].values; y = sub["Delta_AUC"].values
    ax.bar(x, y, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ci_lo, ci_hi = sub["CI_95_lo"].values, sub["CI_95_hi"].values
    valid = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
    if valid.any():
        half_width = (ci_hi - ci_lo) / 2
        ax.errorbar(x[valid], y[valid], yerr=half_width[valid], fmt="none", color="black", capsize=2, linewidth=1, zorder=4)
    ax.axhline(0, color="black", linewidth=1.0, zorder=5)
    ax.set_title(f"{model}\nStructured+Lexicon − Structured", fontsize=10)
    ax.set_xlabel("Fold")
    if j == 0:
        ax.set_ylabel("ΔAUC")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
sig_patch = mpatches.Patch(color="#d73027", alpha=0.85, label="p < 0.05")
ns_patch  = mpatches.Patch(color="#bababa", alpha=0.85, label="p >= 0.05")
fig.legend(handles=[sig_patch, ns_patch], title="DeLong test", loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.08))
fig.suptitle("DeLong Test: ΔAUC per Fold", fontsize=13, fontweight="bold", y=1.12)
fig.tight_layout()
fig.savefig(figures_dir / "lexicon_fig2_delong_deltas.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  lexicon_fig2_delong_deltas.png")
print(f"\nAll figures saved to: {figures_dir}")

# ============================================================
# 10. WRITTEN REPORT
# ============================================================
def _best_arm_per_model(df_pred):
    lines = []
    for model in MODEL_NAMES:
        sub = df_pred[df_pred["Model"] == model].groupby("Arm")["AUC"].mean().sort_values(ascending=False)
        best_arm, best_auc = sub.index[0], sub.iloc[0]
        lines.append(f"- **{model}**: best mean AUC = {best_auc:.4f} ({best_arm})")
    return "\n".join(lines)

def _sig_summary(df_delong):
    lines = []
    for model in MODEL_NAMES:
        sub = df_delong[df_delong["Model"] == model]
        n_sig = int((sub["p_value"] < 0.05).sum())
        mean_delta = sub["Delta_AUC"].mean()
        lines.append(f"- **{model}**: mean ΔAUC (Structured+Lexicon − Structured) = {mean_delta:+.4f}, "
                      f"significant (DeLong p<0.05) in {n_sig}/{len(sub)} folds")
    return "\n".join(lines)

def _lexicon_feature_ranking(df_consistency, n=7):
    lines = []
    for model in MODEL_NAMES:
        sub = df_consistency[df_consistency["Model"] == model].head(n)
        feats = ", ".join(f"`{r.Feature}` (mean rank {r.mean_rank:.1f})" for r in sub.itertuples())
        lines.append(f"**{model}**: {feats}")
    return "\n\n".join(lines)

n_folds_run = len(folds_to_run)
full_run_note = "" if n_folds_run == 14 else (
    f"\n> **Note:** this run used `DEBUG_MAX_FOLDS={DEBUG_MAX_FOLDS}` — only the first "
    f"{n_folds_run} of 14 folds were evaluated. Re-run with `DEBUG_MAX_FOLDS=None` for the full result.\n")

report = f"""# Lexicon Structured Ablation — Results Report

Generated by `07_lexicon_structured_ablation.py`. Compares two arms across {n_folds_run} walk-forward
fold(s), {len(MODEL_NAMES)} algorithms: **Structured** (baseline, {len(STRUCT_COLS)} features) vs
**Structured+Lexicon** (baseline + the 7 frozen Distress Lexicon features from
`Dictionarys/output/lexicon_features.csv`).
{full_run_note}
## Methodology

- Walk-forward, expanding window: `MIN_TRAIN_MONTHS={MIN_TRAIN_MONTHS}`, `STEP_MONTHS={STEP_MONTHS}`,
  `MIN_TEST_DEFAULTS={MIN_TEST_DEFAULTS}` — identical protocol to `analysis/BASELINE.py`,
  `TF-IDF/tfidf_structured_ablation.py`, and `FinBERT/finbert_structured_ablation.py`, verified to
  yield exactly 14 folds.
- The lexicon is **frozen** (built once from the initial training window, per
  `01_extract_frozen_vocabulary.py` / `02_llm_zero_shot_annotate.py`) — the same 7 feature columns
  are used unchanged in every fold; only the per-fold imputer/scaler is refit on that fold's
  training rows, exactly as for the Structured block.
- **Hyperparameters are fixed, not tuned per arm** — reused verbatim from `analysis/BASELINE.py`:
  `Logistic(C=0.3, penalty=l2, solver=saga, class_weight=balanced)`,
  `RandomForest(n_estimators=500, min_samples_leaf=20, class_weight=balanced)`,
  `XGBoost(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, scale_pos_weight=neg/pos per fold)`.
  All three share `random_state={RANDOM_STATE}`.
- Bootstrap 95% CIs: {N_BOOTSTRAP} resamples, seed={BOOT_SEED}. DeLong (1988) pairwise test for
  correlated AUCs on the same test set, Structured vs Structured+Lexicon, per model, per fold.

## A. Predictive performance — best arm per model (mean AUC across folds)

{_best_arm_per_model(df_pred)}

Full per-arm/per-model mean±std table: `results/lexicon_ablation_summary.csv`.

## B. DeLong pairwise significance (mean ΔAUC, fraction of folds significant)

{_sig_summary(df_delong)}

Full per-fold DeLong results: `results/lexicon_ablation_delong.csv`.

## C. Lexicon feature importance ranking (mean rank among all features, lower = more important)

{_lexicon_feature_ranking(df_consistency)}

Full table: `results/lexicon_ablation_feature_importance.csv`.

## Conclusions

This section is generated from the run's own numbers above (section A/B) — re-run the script to
refresh it if the underlying data or parameters change.

"""

for model in MODEL_NAMES:
    struct_auc = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == "Structured")]["AUC"].mean()
    lex_auc    = df_pred[(df_pred["Model"] == model) & (df_pred["Arm"] == "Structured+Lexicon")]["AUC"].mean()
    sub = df_delong[df_delong["Model"] == model]
    n_sig = int((sub["p_value"] < 0.05).sum())
    lexicon_helps = "improves" if lex_auc > struct_auc else "does not improve"
    report += (
        f"- **{model}**: the Distress Lexicon {lexicon_helps} on Structured "
        f"(ΔAUC={lex_auc-struct_auc:+.4f}, significant in {n_sig}/{n_folds_run} folds).\n"
    )

report_path = out_dir / "lexicon_ablation_report.md"
report_path.write_text(report, encoding="utf-8")
print(f"\nWritten report: {report_path}")

print(f"\nDone. Results in {out_dir}, figures in {figures_dir}")
