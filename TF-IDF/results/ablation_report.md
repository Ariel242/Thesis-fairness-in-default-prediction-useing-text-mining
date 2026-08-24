# TF-IDF Structured Ablation — Results & Parameters

Full results, parameters, and significance testing for the walk-forward comparison of
**Structured** (baseline, no text) vs **Structured+TFIDF_full** (baseline + the fold's complete,
uncapped TF-IDF vocabulary) vs **Structured+TFIDF_chi2** (baseline + top-500 TF-IDF
columns selected by chi², fit on train only), across Logistic Regression, XGBoost, and
RandomForest. Companion to `analysis/BASELINE.py` (structured-only reference) and
`FinBERT/RESULTS_COMPARISON.md` (the equivalent comparison for FinBERT embeddings).

---

## 1. Purpose

`TF-IDF/tfidf_pipeline.py` fits an uncapped (uncapped by feature-count cap; still `min_df=5`
filtered) TF-IDF vocabulary per walk-forward fold (unigrams+bigrams) but does not train or
evaluate any model on it. This script (`TF-IDF/tfidf_structured_ablation.py`) closes that gap:
does adding TF-IDF improve default prediction over the structured-only baseline, and — since
the whole point of using TF-IDF instead of a dense embedding is interpretability — how much of
that benefit survives reducing the vocabulary to a small, nameable set of words/bigrams via chi²
feature selection?

---

## 2. Data

- **Source CSV**: `data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_20260715_2119.csv`
- **Rows after filtering** (`is_default ∈ {0,1}`): 217,287
- **Default rate**: 15.36%
- **Structured features**: 126 (identical `STRUCT_COLS` definition to `analysis/BASELINE.py`)
- **TF-IDF source**: `TF-IDF/output/fold{NN}/*.npz` — pre-computed by `tfidf_pipeline.py`
  (not recomputed here), `ngram_range=(1,2)`, `min_df=5`, `sublinear_tf=True`, uncapped vocabulary.

### Walk-Forward folds (identical to `analysis/BASELINE.py` / `tfidf_pipeline.py`, asserted `== 14`)

`MIN_TRAIN_MONTHS=8`, `STEP_MONTHS=3`, `MIN_TEST_DEFAULTS=100` → 14 chronological
expanding-window folds, 2010-08 to 2013-12.

| Fold | train_cutoff | test_end | n_train | n_test | n_defaults |
|---|---|---|---|---|---|
| 1 | 2010-08 | 2010-09 | 6,694 | 1,042 | 157 |
| 2 | 2010-11 | 2010-12 | 9,893 | 1,222 | 136 |
| 3 | 2011-02 | 2011-03 | 13,674 | 1,369 | 184 |
| 4 | 2011-05 | 2011-06 | 18,165 | 1,762 | 255 |
| 5 | 2011-08 | 2011-09 | 23,547 | 1,979 | 286 |
| 6 | 2011-11 | 2011-12 | 29,671 | 2,190 | 406 |
| 7 | 2012-02 | 2012-03 | 36,786 | 2,781 | 413 |
| 8 | 2012-05 | 2012-06 | 45,976 | 3,713 | 681 |
| 9 | 2012-08 | 2012-09 | 59,483 | 5,975 | 919 |
| 10 | 2012-11 | 2012-12 | 77,872 | 5,970 | 893 |
| 11 | 2013-02 | 2013-03 | 98,111 | 8,184 | 1,164 |
| 12 | 2013-05 | 2013-06 | 125,865 | 10,802 | 1,739 |
| 13 | 2013-08 | 2013-09 | 161,023 | 12,846 | 2,007 |
| 14 | 2013-11 | 2013-12 | 202,442 | 14,845 | 2,263 |

### TF-IDF vocabulary size and runtime, per fold

The full arm's width grows with the fold (expanding-window training set → larger vocabulary
passing `min_df=5`); the chi2 arm is fixed at 500 columns every fold. Runtime is the
main practical cost of the full arm — RandomForest and XGBoost both fit on the complete sparse
matrix (up to 55,028 TF-IDF columns + 126 structured, fold 14).

| Fold | TF-IDF vocab (full arm) | Fold wall time (all 3 arms × 3 models) |
|---|---|---|
| 1 | 6,227 | 55s |
| 2 | 8,624 | 81s |
| 3 | 11,226 | 101s |
| 4 | 13,672 | 122s |
| 5 | 16,504 | 156s |
| 6 | 19,081 | 197s |
| 7 | 21,482 | 248s |
| 8 | 24,356 | 365s |
| 9 | 29,073 | 524s |
| 10 | 34,317 | 758s |
| 11 | 39,208 | 1032s |
| 12 | 44,235 | 997s |
| 13 | 49,606 | 1362s |
| 14 | 55,028 | 1788s |

**Total run time: 129.7 minutes** (~2.2 hours), all 14 folds,
run as a detached background process (`n_jobs=4` for RandomForest/XGBoost on the two TF-IDF
arms, RAM safety on this 13.7GB machine).

---

## 3. Feature matrix construction (no leakage)

- **Structured block**: `SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)`
  + `StandardScaler`, both fit on the fold's TRAIN rows only. (`keep_empty_features=True` was
  a bug fix during development — early folds have structured columns that are entirely NaN in
  that fold's small training slice; without it, `SimpleImputer` silently drops them, desyncing
  the feature-name list from the fitted model.)
- **Structured+TFIDF_full**: `scipy.sparse.hstack([structured_sparse, tfidf_train_full])` — the
  complete per-fold TF-IDF matrix, no reduction.
- **Structured+TFIDF_chi2**: `SelectKBest(chi2, k=500)` fit on the TF-IDF block of the
  TRAIN rows only (never touching the structured block — structured columns are never
  candidates for removal), then the identical selected columns applied to the TEST rows'
  TF-IDF matrix (train/test columns are index-aligned per fold since `tfidf_pipeline.py` fit
  the vectorizer on train then `.transform()`'d test with the same vectorizer).
- All model fitting for the two TF-IDF arms stays **sparse** end-to-end (no densification) —
  `saga` (Logistic), RandomForest, and XGBoost all accept sparse CSR input natively.
- Feature names for TF-IDF-derived columns are prefixed `txt:` to avoid a name collision found
  during an earlier pipeline version (a structured column named e.g. `term` collided with a
  TF-IDF token also spelled `term`).

---

## 4. Parameters

### Walk-forward / bootstrap
| Param | Value |
|---|---|
| `MIN_TRAIN_MONTHS` | 8 |
| `STEP_MONTHS` | 3 |
| `MIN_TEST_DEFAULTS` | 100 |
| `N_BOOTSTRAP` | 500 |
| `BOOT_SEED` | 42 |
| `RANDOM_STATE` | 242 |
| `CHI2_TOP_K` | 500 |

### Fixed hyperparameters (identical across all 3 arms — reused verbatim from `analysis/BASELINE.py`, not re-tuned per arm)
| Model | Params |
|---|---|
| Logistic | `penalty=l2, C=0.3, solver=saga, max_iter=1000, class_weight=balanced` |
| XGBoost | `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, scale_pos_weight=dynamic (neg/pos per fold), eval_metric=auc` |
| RandomForest | `n_estimators=500, min_samples_leaf=20, class_weight=balanced` |

Decided explicitly **not** to re-tune per arm (unlike `FinBERT/finbert_structured_ablation.py`,
which tunes regularization per arm) — a full regularization sweep would multiply an already
RAM/time-risky RandomForest fit on wide sparse data (up to ~55K columns) by 4–5 extra
candidates × 3 tuning folds, for uncertain benefit relative to the added runtime (the actual
run already took 130 minutes with fixed hyperparameters).

`n_jobs`: `-1` for the Structured arm (small/dense, matches `BASELINE.py` exactly); `4` for the
two TF-IDF arms (RAM safety on wide sparse RandomForest/XGBoost fits on this machine).

---

## 5. Results

### 5a. Full predictive performance — mean ± std across all 14 folds

| Arm | Model | AUC | PR-AUC | Brier | GINI |
|---|---|---|---|---|---|
| Structured | Logistic | 0.6930 ± 0.0152 | 0.2766 ± 0.0322 | 0.2313 ± 0.0151 | 0.3860 ± 0.0304 |
| Structured | XGBoost | 0.6904 ± 0.0191 | 0.2794 ± 0.0354 | 0.2114 ± 0.0171 | 0.3808 ± 0.0383 |
| Structured | RandomForest | 0.6882 ± 0.0199 | 0.2740 ± 0.0354 | 0.1891 ± 0.0142 | 0.3764 ± 0.0398 |
| Structured+TFIDF_full | Logistic | 0.6965 ± 0.0159 | 0.2818 ± 0.0318 | 0.2268 ± 0.0154 | 0.3930 ± 0.0318 |
| Structured+TFIDF_full | XGBoost | 0.6936 ± 0.0186 | 0.2827 ± 0.0273 | 0.2167 ± 0.0193 | 0.3872 ± 0.0373 |
| Structured+TFIDF_full | RandomForest | 0.6824 ± 0.0183 | 0.2700 ± 0.0289 | 0.2268 ± 0.0127 | 0.3647 ± 0.0366 |
| Structured+TFIDF_chi2 | Logistic | 0.6942 ± 0.0164 | 0.2791 ± 0.0307 | 0.2308 ± 0.0151 | 0.3884 ± 0.0328 |
| Structured+TFIDF_chi2 | XGBoost | 0.6936 ± 0.0204 | 0.2799 ± 0.0298 | 0.2166 ± 0.0173 | 0.3872 ± 0.0408 |
| Structured+TFIDF_chi2 | RandomForest | 0.6870 ± 0.0210 | 0.2753 ± 0.0329 | 0.2069 ± 0.0137 | 0.3740 ± 0.0420 |

### 5b. DeLong pairwise significance test (paired, same test set, per fold)

| Model | Comparison (B vs A) | Mean ΔAUC (B−A) | Significant folds (p<0.05) |
|---|---|---|---|
| Logistic | Structured+TFIDF_full vs Structured | +0.0035 | 7/14 |
| Logistic | Structured+TFIDF_chi2 vs Structured | +0.0012 | 2/14 |
| Logistic | Structured+TFIDF_chi2 vs Structured+TFIDF_full | -0.0023 | 8/14 |
| XGBoost | Structured+TFIDF_full vs Structured | +0.0032 | 3/14 |
| XGBoost | Structured+TFIDF_chi2 vs Structured | +0.0032 | 2/14 |
| XGBoost | Structured+TFIDF_chi2 vs Structured+TFIDF_full | +0.0000 | 0/14 |
| RandomForest | Structured+TFIDF_full vs Structured | -0.0058 | 7/14 |
| RandomForest | Structured+TFIDF_chi2 vs Structured | -0.0012 | 3/14 |
| RandomForest | Structured+TFIDF_chi2 vs Structured+TFIDF_full | +0.0046 | 6/14 |

### 5c. Per-model findings

- **Logistic**: TF-IDF (full) improves on Structured (ΔAUC=+0.0035, significant in 7/14 folds). Reducing to the top-500 chi2-selected words recovers most of the full arm's benefit (chi2 vs Structured ΔAUC=+0.0012, significant in 2/14 folds; full vs chi2 ΔAUC=+0.0023, significant in 8/14 folds). Best-performing arm overall: Structured+TFIDF_full.
- **XGBoost**: TF-IDF (full) improves on Structured (ΔAUC=+0.0032, significant in 3/14 folds). Reducing to the top-500 chi2-selected words recovers essentially all the full arm's benefit (chi2 vs Structured ΔAUC=+0.0032, significant in 2/14 folds; full vs chi2 ΔAUC=-0.0000, significant in 0/14 folds). Best-performing arm overall: Structured+TFIDF_chi2.
- **RandomForest**: TF-IDF (full) does not improve on Structured (ΔAUC=-0.0058, significant in 7/14 folds). Reducing to the top-500 chi2-selected words recovers most of the full arm's benefit (chi2 vs Structured ΔAUC=-0.0012, significant in 3/14 folds; full vs chi2 ΔAUC=-0.0046, significant in 6/14 folds). Best-performing arm overall: Structured.

---

## 6. Which TF-IDF words actually contributed (chi2 arm, all 14 folds)

Cross-fold consistency: for each model, which words were selected by chi² and how highly the
model itself (not the chi² statistic) ranked them, across all 14 independently-trained folds.
A word appearing in most/all folds, with a consistently high importance rank, is the strongest
candidate to carry into a non-walk-forward final model — it was re-derived independently each
time, not chosen once on the full dataset.

### Logistic — top 20 most consistent words
| Word | Folds selected | Mean rank | Mean importance share | Mean chi2 score |
|---|---|---|---|---|
| `help` | 14/14 (100%) | 17.8 | 0.0060 | 4.59 |
| `ring` | 14/14 (100%) | 18.2 | 0.0058 | 5.51 |
| `ups` | 14/14 (100%) | 20.8 | 0.0055 | 5.74 |
| `engagement` | 14/14 (100%) | 25.3 | 0.0052 | 6.45 |
| `engagement ring` | 14/14 (100%) | 42.9 | 0.0042 | 5.18 |
| `apr` | 13/14 (93%) | 9.5 | 0.0077 | 5.20 |
| `pool` | 13/14 (93%) | 25.6 | 0.0092 | 11.85 |
| `mta` | 13/14 (93%) | 65.4 | 0.0040 | 10.44 |
| `motorcycle` | 13/14 (93%) | 108.6 | 0.0028 | 3.75 |
| `help foot` | 13/14 (93%) | 293.8 | 0.0013 | 4.41 |
| `trailer` | 12/14 (86%) | 53.9 | 0.0036 | 2.72 |
| `excellent credit` | 12/14 (86%) | 104.6 | 0.0026 | 3.43 |
| `pay problem` | 12/14 (86%) | 149.2 | 0.0020 | 3.19 |
| `buster` | 12/14 (86%) | 160.8 | 0.0020 | 3.53 |
| `payday` | 12/14 (86%) | 219.1 | 0.0018 | 3.31 |
| `college` | 11/14 (79%) | 2.5 | 0.0119 | 4.57 |
| `university` | 11/14 (79%) | 6.7 | 0.0105 | 6.33 |
| `walmart` | 11/14 (79%) | 45.1 | 0.0043 | 5.33 |
| `harbor` | 11/14 (79%) | 73.9 | 0.0031 | 4.01 |
| `good payer` | 11/14 (79%) | 127.8 | 0.0022 | 7.58 |

### XGBoost — top 20 most consistent words
| Word | Folds selected | Mean rank | Mean importance share | Mean chi2 score |
|---|---|---|---|---|
| `engagement` | 14/14 (100%) | 12.7 | 0.0065 | 6.45 |
| `ring` | 14/14 (100%) | 17.1 | 0.0062 | 5.51 |
| `help` | 14/14 (100%) | 67.3 | 0.0045 | 4.59 |
| `ups` | 14/14 (100%) | 106.9 | 0.0040 | 5.74 |
| `engagement ring` | 14/14 (100%) | 220.8 | 0.0021 | 5.18 |
| `motorcycle` | 13/14 (93%) | 23.2 | 0.0057 | 3.75 |
| `apr` | 13/14 (93%) | 37.9 | 0.0053 | 5.20 |
| `pool` | 13/14 (93%) | 73.0 | 0.0052 | 11.85 |
| `mta` | 13/14 (93%) | 121.2 | 0.0040 | 10.44 |
| `help foot` | 13/14 (93%) | 257.8 | 0.0011 | 4.41 |
| `excellent credit` | 12/14 (86%) | 103.2 | 0.0040 | 3.43 |
| `pay problem` | 12/14 (86%) | 106.4 | 0.0039 | 3.19 |
| `trailer` | 12/14 (86%) | 107.1 | 0.0041 | 2.72 |
| `payday` | 12/14 (86%) | 222.8 | 0.0022 | 3.31 |
| `buster` | 12/14 (86%) | 440.2 | 0.0003 | 3.53 |
| `college` | 11/14 (79%) | 15.4 | 0.0063 | 4.57 |
| `university` | 11/14 (79%) | 17.7 | 0.0061 | 6.33 |
| `refinance` | 11/14 (79%) | 21.8 | 0.0057 | 5.39 |
| `harbor` | 11/14 (79%) | 107.3 | 0.0039 | 4.01 |
| `good payer` | 11/14 (79%) | 148.7 | 0.0032 | 7.58 |

### RandomForest — top 20 most consistent words
| Word | Folds selected | Mean rank | Mean importance share | Mean chi2 score |
|---|---|---|---|---|
| `help` | 14/14 (100%) | 34.6 | 0.0073 | 4.59 |
| `ring` | 14/14 (100%) | 79.6 | 0.0010 | 5.51 |
| `engagement` | 14/14 (100%) | 80.4 | 0.0010 | 6.45 |
| `engagement ring` | 14/14 (100%) | 93.1 | 0.0006 | 5.18 |
| `ups` | 14/14 (100%) | 195.9 | 0.0000 | 5.74 |
| `apr` | 13/14 (93%) | 55.2 | 0.0028 | 5.20 |
| `pool` | 13/14 (93%) | 78.5 | 0.0013 | 11.85 |
| `motorcycle` | 13/14 (93%) | 79.0 | 0.0010 | 3.75 |
| `mta` | 13/14 (93%) | 263.6 | 0.0000 | 10.44 |
| `help foot` | 13/14 (93%) | 278.5 | 0.0000 | 4.41 |
| `excellent credit` | 12/14 (86%) | 70.3 | 0.0019 | 3.43 |
| `trailer` | 12/14 (86%) | 192.1 | 0.0001 | 2.72 |
| `payday` | 12/14 (86%) | 223.8 | 0.0000 | 3.31 |
| `pay problem` | 12/14 (86%) | 252.2 | 0.0000 | 3.19 |
| `buster` | 12/14 (86%) | 435.8 | 0.0000 | 3.53 |
| `college` | 11/14 (79%) | 51.9 | 0.0030 | 4.57 |
| `refinance` | 11/14 (79%) | 55.6 | 0.0022 | 5.39 |
| `university` | 11/14 (79%) | 57.2 | 0.0022 | 6.33 |
| `walmart` | 11/14 (79%) | 121.5 | 0.0002 | 5.33 |
| `harbor` | 11/14 (79%) | 289.7 | 0.0000 | 4.01 |

**Candidate features for a final model**: the words above selected in 12+/14 folds across all
three models (`help`, `ring`, `engagement`, `engagement ring`, `ups`, `apr`, `pool`, `motorcycle`,
`mta`, `excellent credit`) are the most defensible TF-IDF features to carry forward into a
non-walk-forward final model.

---

## 7. Output file locations

```
TF-IDF/results/ablation_folds.csv               per fold x arm x model: AUC/PR-AUC/Brier/GINI + 95% CI
TF-IDF/results/ablation_summary.csv              mean +/- std across folds, grouped by (arm, model)
TF-IDF/results/ablation_delong.csv               pairwise DeLong per fold x model x arm-pair
TF-IDF/results/ablation_feature_importance.csv   per fold x arm x model x feature: importance/rank
TF-IDF/results/ablation_feature_consistency.csv  cross-fold stability of the chi2 arm's selected words
TF-IDF/results/ablation_report.md                this document
TF-IDF/figures/fig1_auc_by_fold.png              AUC per fold, per model, 3 arms overlaid
TF-IDF/figures/fig2_delong_deltas.png            ΔAUC bar chart per model x arm-pair, significance-colored
```
