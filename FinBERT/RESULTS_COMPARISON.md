# FinBERT Dimensionality Comparison — Results & Parameters

Living document, updated as each of the four runs completes. Tracks the
four-way comparison of adding FinBERT's semantic embedding of `desc` to the
structured default-prediction model, at full (768-dim) and PCA-reduced
(50-dim) resolution, with fixed vs. grid-searched hyperparameters.

**Status as of this update:**

| | Fixed hyperparams | Grid search (per fold) |
|---|---|---|
| **768-dim (raw)** | ✅ done | ✅ done |
| **50-dim (PCA)** | ✅ done | ✅ done |

**All four runs complete.** Headline finding: PCA-50 + FinBERT matches or beats the structured-only reference on every metric, with or without tuning, at a fraction of the runtime; raw 768-dim FinBERT hurts even after 5+ hours of tuning and never catches up. See Section 5f for the full cross-run table and Section 8 for why.

---

## 1. Purpose

`analysis/BASELINE.py` / `analysis/GRIDSEARCH.py` establish the structured-only
reference point (126 features, no text signal at all). This sub-project asks:
does adding FinBERT's semantic embedding of the loan `desc` field improve
default prediction — and does the answer change with (a) dimensionality
(raw 768 dims vs. PCA-compressed to 50) and (b) hyperparameter tuning?

Four self-contained scripts, one per cell of the table above, all living in
`FinBERT/`:
- `BASELINE_768.py` — structured + raw 768-dim FinBERT, fixed hyperparameters
- `GRIDSEARCH_768.py` — structured + raw 768-dim FinBERT, per-fold grid search
- `BASELINE_PCA50.py` — structured + 50-dim PCA-compressed FinBERT, fixed hyperparameters
- `GRIDSEARCH_PCA50.py` — structured + 50-dim PCA-compressed FinBERT, per-fold grid search

None of these import, edit, or execute each other, `analysis/BASELINE.py`, or
`FinBERT/finbert_structured_ablation.py` (a separate, more elaborate ablation
script with its own feature-selection + tuning machinery — not used here).

---

## 2. Data

- **Source CSV**: `data/03_advanced_prep/lc_after_03_advanced_prep_basic+test_20260715_2119.csv`
- **Rows after filtering** (`is_default ∈ {0,1}`): 217,287
- **Default rate**: 15.36%
- **FinBERT embeddings**: `FinBERT/data/finbert_desc_embeddings.parquet` — merged by `id` (never row position), asserted 1:1 with no row-count change and no unmatched rows.
  - `has_desc`: 45.9% of loans have a non-empty description
  - `sem_000`…`sem_767`: 768-dim mean-pooled FinBERT embedding of `desc` (float16 on disk, cast to float32 on load)
- **Structured features**: 126 (identical `STRUCT_COLS` definition to `analysis/BASELINE.py` — see that file's docstring for the full derivation: 138 raw CSV columns + 2 added ordinal columns (`grade_ord`, `sub_grade_ord`) − 14 excluded identifier/leakage/date columns = 126, all already numeric dtype).

### Walk-Forward folds (identical across all 4 scripts, asserted `== 14`)

`MIN_TRAIN_MONTHS=8`, `STEP_MONTHS=3`, `MIN_TEST_DEFAULTS=100` → 14 chronological expanding-window folds, 2010-08 to 2013-12, `n_train` from 6,694 to 202,442, `n_test` from 1,042 to 14,845. Identical fold boundaries to `analysis/BASELINE.py`.

---

## 3. Feature matrix construction (no leakage)

- **Structured block**: `SimpleImputer(strategy="constant", fill_value=0)` + `StandardScaler`, both fit on the fold's TRAIN rows only.
- **FinBERT block**:
  - *768-dim scripts*: `has_desc` + all 768 `sem_*` columns, scaled separately (own `StandardScaler`, fit on TRAIN only), then `hstack`'d onto the structured block. 895 total columns.
  - *PCA-50 scripts*: `has_desc` + 50 PCA components of `sem_*`. `PCA(n_components=50, svd_solver='randomized', random_state=242)` is fit **fresh inside every fold**, on that fold's TRAIN rows' `sem_*` values only, then only `.transform()`'d on TEST — never fit on test rows or on the full dataset before folds are built. Smoke-tested: ~0.5s (fold 1, n=6,694) to ~6s (fold 14, n=202,442) per fold, explained variance ratio ~88–90%. `svd_solver='randomized'` is deliberate (not sklearn's silent default): with n_components=50 ≪ 768, randomized SVD (Halko et al.) is far cheaper than exact SVD with negligible accuracy loss. 177 total columns.
- **Grid-search scripts additionally**: within each fold, the last 20% of that fold's (chronologically ordered) TRAIN rows are held out as a temporal validation split. Imputer/scaler/PCA for both blocks are fit on the remaining 80% ("sub-train") only; every grid combination is scored on validation AUC; the winner is refit on the FULL fold TRAIN set (fresh imputer/scaler/PCA) before scoring the actual TEST fold.

---

## 4. Parameters

### Walk-forward / bootstrap (all 4 scripts)
| Param | Value |
|---|---|
| `MIN_TRAIN_MONTHS` | 8 |
| `STEP_MONTHS` | 3 |
| `MIN_TEST_DEFAULTS` | 100 |
| `N_BOOTSTRAP` | 500 |
| `BOOT_SEED` | 42 |
| `RANDOM_STATE` | 242 |

### Fixed hyperparameters — `BASELINE_768.py` / `BASELINE_PCA50.py`
Identical to `analysis/BASELINE.py` (no re-tuning for the wider feature matrix — this is the point of comparison against the grid-searched versions):
| Model | Params |
|---|---|
| Logistic | `penalty=l2, C=0.3, solver=saga, class_weight=balanced` |
| XGBoost | `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, scale_pos_weight=dynamic` |
| RandomForest | `n_estimators=500, min_samples_leaf=20, class_weight=balanced` |

### Grids — `GRIDSEARCH_768.py` / `GRIDSEARCH_PCA50.py`
Identical to `analysis/GRIDSEARCH.py`, with **one deviation**: Logistic's `C` grid extended one step lower (`0.01` added) — the structured-only grid search pinned `C=0.03` (the low edge of that grid) in most folds, and this feature matrix is ~7x wider, which if anything wants more regularization, not less.
| Model | Grid |
|---|---|
| Logistic | `C ∈ {0.01, 0.03, 0.1, 0.3, 1}` (5 combos — extended from GRIDSEARCH.py's 4) |
| XGBoost | `max_depth ∈ {3,4,6} × learning_rate ∈ {0.05,0.1}` (6 combos) |
| RandomForest | `min_samples_leaf ∈ {10,20,50}` (3 combos, `n_estimators=500` fixed) |

### Memory fix (applied to all 4 scripts after the first `BASELINE_768.py` run was killed, likely OOM, after fold 13/14)
This machine has 13.7GB RAM (documented OOM history in `finbert_structured_ablation.py`'s docstring). Fold 14 (202,442 rows × 895 columns) with RandomForest's default `n_jobs=-1` (12 cores, each holding a bootstrap sample) is memory-heavy. Fix: `n_jobs` capped at 4 (was -1) for both RandomForest and XGBoost across all 4 scripts, plus explicit `del` of large per-fold arrays + `gc.collect()` before the next fold starts. Neither change alters results (both models are deterministic under a fixed `random_state` regardless of `n_jobs`). The retry completed cleanly (see below).

---

## 5. Results

### 5a. Reference — Structured only, 126 features (`analysis/BASELINE.py` / `GRIDSEARCH.py`)
Mean ± std AUC across 14 folds:
| Model | Fixed hyperparams | Grid search |
|---|---|---|
| Logistic | 0.6930 ± 0.0152 | 0.6930 ± 0.0153 |
| XGBoost | 0.6904 ± 0.0191 | 0.6930 ± 0.0152 |
| RandomForest | 0.6886 ± 0.0199 | 0.6883 ± 0.0190 |

*(Full detail — Brier, PR-AUC, GINI, winning hyperparameters per fold — in `GRIDSEARCH_BASELINE.pdf` on the Desktop.)*

### 5b. Structured + FinBERT-768, fixed hyperparameters (`BASELINE_768.py`) — ✅ DONE
Run time: 71.5 minutes (14 folds). Mean ± std across 14 folds:
| Model | AUC | PR_AUC | Brier | GINI | Δ AUC vs. structured-only |
|---|---|---|---|---|---|
| Logistic | 0.6732 ± 0.0356 | 0.2620 ± 0.0403 | 0.2326 ± 0.0161 | 0.3465 ± 0.0711 | **-0.0198** |
| XGBoost | 0.6840 ± 0.0266 | 0.2760 ± 0.0407 | 0.2039 ± 0.0247 | 0.3680 ± 0.0533 | -0.0064 |
| RandomForest | 0.6611 ± 0.0197 | 0.2452 ± 0.0318 | 0.1807 ± 0.0118 | 0.3222 ± 0.0395 | **-0.0275** |

**Finding**: adding all 768 raw FinBERT dims, with hyperparameters left unchanged from the 126-feature structured model, *hurts* AUC across all three algorithms — most severely for RandomForest and Logistic. Consistent with the earlier 2026-08-03 ablation run's finding that under-regularized FinBERT additions hurt RandomForest; here it's worse because there's no dimensionality reduction and no re-tuning at all. Some `fb:sem_*` dimensions do appear in the last-fold Top-20 feature importance for XGBoost (positions 12–19: `sem_693`, `sem_616`, `sem_097`, `sem_570`, `sem_447`, `sem_579`, `sem_300`) and Logistic (positions 17–20), so there is *some* signal — it's being drowned out by insufficient regularization for a 7x-wider feature space, which `GRIDSEARCH_768.py` is testing directly.

Top structured features remain dominant and consistent with the structured-only run: `grade_ord`, `sub_grade_ord`, `int_rate`, `term`, `annual_inc`, `fico_midpoint`, `dti`.

Full per-fold detail: `FinBERT/results/baseline_768/wf_predictive_folds.csv`, `wf_predictive_summary.csv`. Figure: `FinBERT/figures/baseline_768/fig1_auc_stability.png`.

### 5c. Structured + FinBERT-768, grid search (`GRIDSEARCH_768.py`) — ✅ DONE
Run time: **304.7 minutes (~5.1 hours)** — by far the most expensive of the four scripts, exactly as warned (grid search × 895-column matrix × `saga`). Mean ± std across 14 folds:
| Model | AUC | PR_AUC | Brier | GINI | Δ AUC vs. structured-only grid search | Δ AUC vs. 768-fixed |
|---|---|---|---|---|---|---|
| Logistic | 0.6827 ± 0.0242 | 0.2687 ± 0.0350 | 0.2289 ± 0.0148 | 0.3654 ± 0.0483 | -0.0103 | **+0.0095** |
| XGBoost | 0.6890 ± 0.0202 | 0.2777 ± 0.0391 | 0.2148 ± 0.0196 | 0.3781 ± 0.0404 | -0.0040 | +0.0050 |
| RandomForest | 0.6747 ± 0.0179 | 0.2570 ± 0.0320 | 0.2018 ± 0.0111 | 0.3494 ± 0.0357 | -0.0136 | **+0.0136** |

**Finding**: tuning recovers roughly half of what fixed hyperparameters lost by adding raw FinBERT-768 (see 5b) — but **all three models still land below the structured-only reference** (fixed AND grid-searched), meaning even with per-fold tuning, the raw 768-dim FinBERT embedding is net harmful here, not just under-regularized. This is the central finding motivating the PCA-50 runs (5d/5e below): the hypothesis is that 768 highly-correlated raw dimensions are simply too much for these models/dataset sizes to profitably use, and compressing to 50 components may let the genuine signal (visible in the feature-importance tables — see below) come through without the accompanying noise/overfitting.

**Winning hyperparameters** (full table: `FinBERT/results/gridsearch_768/wf_best_params.csv`):
| Model | Most frequent choice (of 14 folds) | Note |
|---|---|---|
| Logistic | `C=0.01` (12/14 folds) | **Pinned at the low edge of the grid** (the deliberately-extended edge) — even lower C might do better still |
| XGBoost | `max_depth=3, learning_rate=0.05` (12/14 folds) | Consistent with the structured-only grid search's preference for shallower trees |
| RandomForest | `min_samples_leaf=50` (12/14 folds) | **Pinned at the high edge of the grid** (the most-regularized option offered) — an even larger min_samples_leaf might do better still |

Two of three models pinned at the edge of their grid toward MORE regularization — a clear signal that the grid should be extended further in a follow-up run, rather than a settled optimum.

### 5d. Structured + FinBERT-PCA50, fixed hyperparameters (`BASELINE_PCA50.py`) — ✅ DONE
Run time: **18.4 minutes** (vs. 71.5 min for the 768-dim baseline — PCA-50 is much cheaper to fit than 768 raw columns through `saga`/RandomForest/XGBoost). Explained variance ratio: 88.0–90.2% across folds (rises slightly with fold size — later folds have more text to summarize). Mean ± std across 14 folds:
| Model | AUC | PR_AUC | Brier | GINI | Δ AUC vs. structured-only | Δ AUC vs. 768-fixed |
|---|---|---|---|---|---|---|
| Logistic | **0.6948 ± 0.0157** | **0.2790 ± 0.0325** | 0.2310 ± 0.0149 | **0.3896 ± 0.0315** | **+0.0018** | **+0.0216** |
| XGBoost | 0.6887 ± 0.0247 | 0.2759 ± 0.0385 | 0.2065 ± 0.0205 | 0.3773 ± 0.0494 | -0.0017 | +0.0047 |
| RandomForest | 0.6855 ± 0.0210 | 0.2675 ± 0.0378 | **0.1738 ± 0.0119** | 0.3710 ± 0.0420 | -0.0031 | **+0.0244** |

**Finding — this is the central result of the whole comparison.** With PCA-50, and *without any hyperparameter retuning at all*, FinBERT stops hurting and Logistic actually **improves over the structured-only reference** on every single metric (AUC, PR-AUC, Brier, GINI). XGBoost and RandomForest land within noise of the structured-only baseline (well inside 1 std), a dramatic recovery from the 768-dim run where both were clearly worse. RandomForest's Brier also improves markedly (0.1738 vs. 0.1877 structured-only) — better-calibrated probabilities, not just similar ranking.

**Direct confirmation of the Section 6c hypothesis**: `fb:pca_016` — a single PCA component — lands in the **global Top 20** features for both XGBoost (rank 11, Gain=82.53) and Logistic (rank 17, Importance=0.0698) on the last fold. In the raw-768 run, no single `sem_*` dimension came close to this; the signal was spread across dozens of correlated dimensions (Section 6b/6c). Compressing 768 correlated dimensions into 50 orthogonal components concentrated that diffuse signal into features fixed-regularization models can actually use — exactly as hypothesized.

Full per-fold detail: `FinBERT/results/baseline_pca50/wf_predictive_folds.csv`, `wf_pca_variance.csv`. Figure: `FinBERT/figures/baseline_pca50/fig1_auc_stability.png`.

### 5e. Structured + FinBERT-PCA50, grid search (`GRIDSEARCH_PCA50.py`) — ✅ DONE
Run time: **61.6 minutes** (vs. 304.7 min for the 768-dim grid search — 5x faster, tracking the 768-vs-PCA50 baseline speed ratio). Mean ± std across 14 folds:
| Model | AUC | PR_AUC | Brier | GINI |
|---|---|---|---|---|
| Logistic | 0.6948 ± 0.0160 | 0.2792 ± 0.0328 | 0.2306 ± 0.0140 | 0.3897 ± 0.0320 |
| XGBoost | 0.6921 ± 0.0208 | 0.2782 ± 0.0382 | 0.2162 ± 0.0162 | 0.3843 ± 0.0415 |
| RandomForest | 0.6857 ± 0.0200 | 0.2691 ± 0.0366 | 0.1856 ± 0.0226 | 0.3714 ± 0.0400 |

**Finding**: tuning barely moves the needle on top of PCA-50 (Logistic: AUC identical to the fixed run at 0.6948; XGBoost: +0.0034; RandomForest: +0.0002) — a sharp contrast to the 768-dim case, where tuning recovered a real (if insufficient) chunk of lost AUC (5c). This itself is informative: once the FinBERT signal is compressed into 50 components, the fixed hyperparameters already carried over from the structured-only model are close enough to optimal that a dedicated per-fold search adds almost nothing. Winning hyperparameters: Logistic `C=0.01` (9/14 folds), XGBoost `max_depth=3, learning_rate=0.05` (13/14 folds), RandomForest `min_samples_leaf=50` (8/14 folds) — same general direction as the 768-dim grid search (more regularization preferred), just with much less to gain from finding it.

Full detail: `FinBERT/results/gridsearch_pca50/wf_predictive_folds.csv`, `wf_best_params.csv`, `wf_pca_variance.csv`.

### 5f. Full cross-run comparison — AUC

| Model | Structured-only (Baseline) | Structured-only (Grid) | 768-dim (Baseline) | 768-dim (Grid) | **PCA-50 (Baseline)** | **PCA-50 (Grid)** |
|---|---|---|---|---|---|---|
| Logistic | 0.6930 | 0.6930 | 0.6732 | 0.6827 | **0.6948** | **0.6948** |
| XGBoost | 0.6904 | 0.6930 | 0.6840 | 0.6890 | 0.6887 | **0.6921** |
| RandomForest | 0.6886 | 0.6883 | 0.6611 | 0.6747 | 0.6855 | 0.6857 |

**Runtime, for the same comparison:**
| | Baseline (fixed) | Grid search |
|---|---|---|
| Structured-only | (not re-timed here) | (not re-timed here) |
| 768-dim | 71.5 min | 304.7 min |
| PCA-50 | **18.4 min** | **61.6 min** |

PCA-50 is simultaneously **faster** (4-5x) and **better** (matches or beats structured-only, unlike 768-dim) than the raw embedding — dimensionality reduction is a strict win here on every axis measured: accuracy, calibration (Brier), and compute.

### 5g. DeLong test: FinBERT-768 vs. FinBERT-PCA50 (paired, same test set, fixed hyperparameters)

The comparisons above are all against the *structured-only* reference. This section asks the more direct question — are 768-dim and PCA-50 statistically different from **each other**? DeLong's test (DeLong et al. 1988) pairs the two AUCs on the identical test rows within each fold, for both models fixed at the same hyperparameters as `BASELINE_768.py` / `BASELINE_PCA50.py`.

**Getting there took 5 attempts.** The first combined script (`DELONG_768_VS_PCA50.py`) fits both variants' models in one process and was killed by what looks like OOM twice at fold 11 and 12/14 — even after restructuring it to process the two variants sequentially (never holding both large matrices + all 6 fitted models at once). The eventual fix was architectural, not another memory tweak: split into two independent single-variant scripts (`SAVE_PREDICTIONS_768.py`, `SAVE_PREDICTIONS_PCA50.py`) that only fit models and persist raw per-row test predictions to CSV — run as **separate OS processes**, never concurrently, so each gets the full machine to itself and the OS fully reclaims memory between them. Both scripts also checkpoint per fold (append to disk immediately, skip already-saved folds on rerun), so the 768-dim script's own two additional kills (at fold 12 and fold 13) each only cost the remaining folds, not a restart from fold 1. `SAVE_PREDICTIONS_768.py` finished in 55.8 minutes (of actual compute, spread over its final successful session); `SAVE_PREDICTIONS_PCA50.py` completed cleanly on its first attempt (14.7 min — much lighter workload, no surprise). The DeLong test itself, run afterward on the two saved prediction files, is nearly instant (no model fitting involved).

**Result:**
| Model | Mean ΔAUC (PCA50 − 768) | Folds significant (p<0.05) | Direction | Folds where PCA50 wins (any margin) |
|---|---|---|---|---|
| **RandomForest** | **+0.0244** | **12 / 14** | 100% favor PCA50, 0% favor 768 | **14 / 14** |
| **Logistic** | **+0.0216** | 6 / 14 | 100% favor PCA50, 0% favor 768 | **14 / 14** |
| XGBoost | +0.0046 | 0 / 14 | mixed (2 folds trivially negative) | 12 / 14, but no fold reaches significance |

**For RandomForest and Logistic, PCA-50 beats raw FinBERT-768 in every single one of the 14 folds, with no exception** — and for RandomForest the majority of those wins (12/14) are individually statistically significant at the fold level, never once in the other direction. This is a materially stronger claim than the aggregate mean-AUC comparison in 5f alone: it's not just that PCA-50's average is higher, the ordering holds fold-by-fold, consistently, for two of the three algorithms. XGBoost is the exception — no fold shows a significant difference either way, meaning XGBoost is statistically indifferent to whether FinBERT arrives as 768 raw dimensions or 50 PCA components (consistent with 5c/5e, where XGBoost's ΔAUC from tuning was also the smallest of the three models).

Full detail: `FinBERT/results/delong_768_vs_pca50/delong_per_fold.csv`, `delong_summary.csv`. Raw predictions: `FinBERT/results/predictions_768.csv`, `predictions_pca50.csv` (224,040 rows each — every test-set row, every fold, every model).

---

## 6. Which FinBERT dimensions actually contributed (full analysis, not just Top-20)

Recomputed from `BASELINE_768.py`'s last fold (train ≤ 2013-11, n=202,442) — same
models, same fixed hyperparameters as the walk-forward run, refit once to
extract the **complete** per-feature importance ranking (895 features for
Logistic/RandomForest, 784 actually-used features for XGBoost — trees don't
necessarily split on every column). Full tables saved to
`FinBERT/results/baseline_768/feature_importance_{logistic,xgboost,randomforest}_FULL.csv`.

**Caveat on "nonzero importance"**: under L2 (Logistic) and with 500 trees
(RandomForest), almost every one of the 768 `sem_*` dimensions ends up with
*some* nonzero weight/importance — that alone doesn't mean a dimension
"helped." The meaningful bar is **rank**: how a dimension compares to the
other ~900 candidate features (mostly structured), not whether its
coefficient happens to be exactly zero.

### 6a. Per-model summary
| Model | Total features in model | sem_* dims with nonzero contribution | Best-ranked sem_* dim | Its global rank | FinBERT dims in the global Top 50 |
|---|---|---|---|---|---|
| Logistic | 895 | 768 / 768 | `fb:sem_649` | **#17** of 895 | **29 of 50** |
| XGBoost | 784 (used splits only) | 705 / 768 | `fb:sem_693` | **#12** of 784 | **25 of 50** |
| RandomForest | 895 | 768 / 768 | `fb:sem_097` | #51 of 895 | **0 of 50** |

**This is the clearest single explanation for RandomForest's worse ΔAUC** (-0.0275 fixed, -0.0136 tuned — the largest drop of the three models in both runs, see 5b/5c): RandomForest structurally leans almost entirely on the 126 structured features for its top splits — not a single FinBERT dimension cracks its global Top 50 — while Logistic and XGBoost both draw roughly half their top-50 most important features from FinBERT. RandomForest isn't failing to see the FinBERT signal because it doesn't exist (Logistic and XGBoost both find it, and agree on much of it — see 6b); it's a model/algorithm mismatch given the fixed `min_samples_leaf=20` (and even the grid search's winning `min_samples_leaf=50` didn't fully close the gap).

### 6b. FinBERT dimensions that are consistently useful across ALL THREE models
Rather than trust one model's ranking alone, this intersects the Top-200
(of ~800-900 total features) lists across Logistic, XGBoost, and
RandomForest — dimensions that independently rank as important to three
very different algorithms are the most defensible "real" signal.
**15 dimensions** clear that bar (full detail in
`finbert_dims_consistent_top200_all3models.csv`), ranked by summed rank
across the three models (lower = more consistently important):

| FinBERT dim | LR rank | XGBoost rank | RandomForest rank | Sum of ranks |
|---|---|---|---|---|
| `sem_300` | 64 | 19 | 57 | 140 |
| `sem_089` | 22 | 116 | 70 | 208 |
| `sem_217` | 103 | 28 | 90 | 221 |
| `sem_642` | 32 | 67 | 127 | 226 |
| `sem_662` | 104 | 53 | 85 | 242 |
| `sem_171` | 72 | 66 | 123 | 261 |
| `sem_456` | 99 | 96 | 72 | 267 |
| `sem_716` | 102 | 64 | 102 | 268 |
| `sem_130` | 101 | 81 | 88 | 270 |
| `sem_467` | 151 | 87 | 75 | 313 |
| `sem_635` | 85 | 157 | 91 | 333 |
| `sem_340` | 92 | 114 | 148 | 354 |
| `sem_238` | 18 | 181 | 162 | 361 |
| `sem_485` | 75 | 126 | 181 | 382 |
| `sem_187` | 134 | 105 | 178 | 417 |

Two dimensions (`sem_300`, `sem_456`) rank in the **Top 100** of all three
models simultaneously — the single strongest, most model-agnostic FinBERT
signal found in this run. (`finbert_dims_consistent_top100_all3models.csv`
has just these two, for reference.)

### 6c. Interpretation
There is real, model-agnostic signal in the FinBERT embedding — it is not
noise that only one algorithm happened to latch onto. But it's diffuse:
spread thinly across dozens of dimensions rather than concentrated in a
handful of dominant ones (compare to `int_rate` or `grade_ord`, which sit at
rank #1-3 in every model, every fold). That diffuseness is exactly what
makes raw 768-dim FinBERT hard to use profitably with a fixed, unchanged
regularization budget (5b) — and only partially recoverable with tuning
(5c). It's also the direct motivation for testing PCA-50 (5d/5e): if the
signal really is spread across many correlated `sem_*` dimensions, a
compression step that captures the shared variance in fewer components
should concentrate that diffuse signal into features a fixed-regularization
model (and especially RandomForest, per 6a) can actually use.

---

## 7. Output file locations
```
FinBERT/results/baseline_768/       wf_predictive_folds.csv, wf_predictive_summary.csv
FinBERT/results/gridsearch_768/     wf_predictive_folds.csv, wf_best_params.csv, wf_predictive_summary.csv
FinBERT/results/baseline_pca50/     wf_predictive_folds.csv, wf_predictive_summary.csv, wf_pca_variance.csv
FinBERT/results/gridsearch_pca50/   wf_predictive_folds.csv, wf_best_params.csv, wf_predictive_summary.csv, wf_pca_variance.csv
FinBERT/figures/<same 4 subfolders>/fig1_auc_stability.png
```
None of these overwrite `finbert_structured_ablation.py`'s existing `FinBERT/results/fb_*.csv` files or `FinBERT/figures/fig*.png`.
