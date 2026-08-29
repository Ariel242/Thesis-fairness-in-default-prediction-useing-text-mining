# strict_temporal_v2 — Changes Report

## 1. Problems found

The existing pipeline (`analysis/02_data_prep.ipynb` + `analysis/03_advanced_prep.ipynb`)
computes all structured-data preprocessing **once, globally, on all 217,287 rows
of 2010–2013**, before any walk-forward fold exists:

- Missing-column selection (>99% threshold)
- Correlation filtering (r>0.95, 5 columns dropped)
- 99.5th-percentile clipping + log1p (7 variables)
- One-hot encoding (`pd.get_dummies`, 4 categorical columns)

Only the model-fitting stage (imputer/scaler/TF-IDF/χ²/PCA inside each
ablation script) was already fit train-only per fold. This is a real,
disclosed data-leakage gap for the structured block, confirmed by direct
inspection of the notebook code (not hypothetical).

Two additional, unrelated facts were confirmed during the same review:

- The pipeline's "fixed" hyperparameters (`analysis/BASELINE.py`: LR C=0.3,
  XGBoost max_depth=4/learning_rate=0.05) **predate**
  `analysis/GRIDSEARCH.py` (file mtimes: 18:01 vs 22:48, same day) — they
  are not derived from it. When the grid search was later run, its actual
  most-frequent winners across the 14 folds were C=0.03 and max_depth=3,
  not the fixed values.
- `THRESHOLDS = [0.5, 0.6, 0.7]` is dead code in every script that defines
  it (`BASELINE.py`, `GRIDSEARCH.py`, `BASELINE_PCA50.py`), explicitly
  marked "currently unused — kept for compatibility."

## 2. Files created

All new, nothing existing modified:

```
strict_temporal_v2/
  __init__.py
  folds.py                    -- shared, tested 14-fold definition + integrity assertions
  raw_features.py             -- global, non-fit steps (reimplements 02/03's row-wise logic)
  structured_preprocessing.py -- per-fold, train-only: missing/correlation/clip/one-hot/impute
  representations.py          -- lexicon / TF-IDF / FinBERT-768 / FinBERT-PCA50 adapters
  hyperparams.py               -- fixed config + temporal grid-search config
  thresholds.py                -- threshold policy (legacy_fixed / validation_youden / explicit)
  metrics.py                   -- AUC/PR-AUC/GINI/Brier, DeLong, bootstrap CI, BH-FDR
  fairness_stub.py             -- fairness_cluster passthrough contract + leak assertion
  run_dry.py                   -- fold 1 (+ fold 14) smoke test, no file outputs
  run_full.py                  -- all 14 folds, writes results/strict_temporal_v2/
  tests/
    __init__.py
    test_leakage.py            -- 13 pytest tests (the 12 from the spec + 1 regression test)

docs/strict_temporal_v2_changes.md   -- this file
```

Nothing in `analysis/`, `TF-IDF/`, `FinBERT/`, `Dictionarys/`, or any
existing `results/` folder was modified.

## 3. Leakage tests — result

`pytest strict_temporal_v2/tests/test_leakage.py -v` — **13/13 passed.**

Covers: clip-quantile train-only, correlation-filter train-only, one-hot
encoder train-only (unseen test category), scaler train-only, TF-IDF
train-only (vocab grows with fold size — regression check against cached
`TF-IDF/output/`), χ² train-only, PCA train-only, grid-search signature
excludes outer test, no overlap between outer test folds, same test set
shared across representations, fairness columns never enter features,
target/id never candidate features (regression test, see §4), and the
real 14-fold table regression check.

## 4. Bugs found and fixed during the smoke test itself

Two real bugs surfaced running the fold-1 dry run (not caught by the unit
tests until fixed and backfilled as regression tests):

1. **Target leaking into features (critical).** `NON_FEATURE_COLS` in
   `structured_preprocessing.py` did not exclude `is_default` (or
   `funded_ratio`, or `has_desc`) — the target column was being fit as an
   ordinary numeric candidate feature, giving a trivial AUC=1.0 on fold 1.
   Fixed by aligning `NON_FEATURE_COLS` exactly with the verified
   `EXCLUDE_COLS` set already used by every existing ablation script
   (`TARGET_COL`, id/date/text columns, `funded_ratio`, `has_desc`).
   Added `test_target_and_id_never_candidate_features` as a permanent
   regression guard.
2. **`id` dtype mismatch breaking the lexicon/FinBERT joins.** The raw
   CSV contains trailing footer/summary rows (e.g. `"Total amount funded
   in policy code 1: ..."`) that force the whole `id` column to `object`
   dtype, even though those rows are excluded by the year/status filters
   (their `issue_d` is NaN). Fixed by explicitly re-casting `id` to
   `int64` (with a duplicate/null assertion) right after filtering in
   `raw_features.py`.
3. **Fairness-leak check too naive (false positive, not a real leak).**
   `assert_not_in_features` used a substring match for `"zip"` /
   `"census"` anywhere in a feature name, which flagged legitimate TF-IDF
   vocabulary tokens like `tfidf:census bureau` (loan-text content, not a
   geography feature) as a leak. Fixed to match only the exact banned
   names or a namespaced prefix (`zip3:`, `census:`, etc.), confirmed via
   the fold-14 dry run.
4. **Structured block left unscaled before sparse-hstacking with TF-IDF
   (critical, only visible at fold-14 scale).** The first working version
   of `representations.py`/`run_dry.py` skipped scaling entirely on the
   sparse (TF-IDF) code path. On fold 1 (small, 89 structured columns)
   this didn't visibly break anything; on fold 14 it collapsed Logistic
   Regression to an identical, badly-degraded AUC=0.5834 for *both*
   `tfidf_full` and `tfidf_chi2` (XGBoost, being scale-invariant per
   feature, was unaffected and stayed ~0.70 — that mismatch was the
   giveaway). Root cause: the structured block's raw, wildly different
   magnitudes (loan amounts in the thousands next to 0/1 dummies) sitting
   unscaled next to TF-IDF's small L2-normalized weights broke saga's
   convergence. Fixed by verifying and replicating each representation's
   *existing* scaling precedent exactly (checked directly against
   `FinBERT/BASELINE_768.py`, `FinBERT/BASELINE_PCA50.py`, and
   `TF-IDF/tfidf_structured_ablation.py`):
   - `structured` / `structured_has_desc` / `lexicon`: one `StandardScaler`
     over the whole combined block (matches the existing lexicon ablation).
   - `finbert_768` / `finbert_pca50`: **two** separate scalers — one for
     structured alone, one for (has_desc + embedding/PCA block) — exactly
     matching the existing FinBERT scripts.
   - `tfidf_full` / `tfidf_chi2`: one scaler for (structured + has_desc)
     only; the TF-IDF block itself is never rescaled, then sparse-hstacked
     — exactly matching the existing TF-IDF ablation script.
   This also prompted a refactor: the matrix-building logic that had been
   duplicated separately in `run_dry.py` and `run_full.py` was
   consolidated into one function, `representations.build_final_matrix()`,
   so there is now exactly one implementation to get right instead of two
   that can silently drift apart (which is how this bug's inconsistency
   was possible in the first place).

After all four fixes, fold-1 and fold-14 AUCs land in the same plausible
range as the old (leaky) pipeline's own fold-1/fold-14 numbers (see §6),
which is the expected outcome — the leakage fix changes *which* columns
survive per fold, not the general scale of predictive performance.

## 5. Full run results and timing

The full 14-fold run (§7/§8 below) is now complete. Folds 1-13 were run in an earlier
session (per-fold timing not captured to a log file); fold 14 — the largest fold and the
one with the ~55K-column TF-IDF arm — was re-run on 2026-08-25 with `--xgb-n-jobs 4`
(per the disclosed, approved deviation documented in `run_full.py`'s own `--xgb-n-jobs`
help text: fold 14 hit an apparent OOM kill twice at the spec default `n_jobs=-1`) and
completed cleanly in 1771.4s (~29.5 min). No further OOM after the n_jobs cap.

**A. Predictive performance — mean AUC ± std across all 14 folds, by representation x model**
(Logistic / XGBoost only, per spec §16 — RandomForest not run in this comparison):

| Representation | Logistic AUC | XGBoost AUC |
|---|---|---|
| structured (baseline) | 0.6937 ± 0.0154 | 0.6916 ± 0.0165 |
| structured_has_desc | 0.6936 ± 0.0155 | 0.6910 ± 0.0178 |
| lexicon | 0.6946 ± 0.0147 | 0.6912 ± 0.0173 |
| tfidf_full | 0.6971 ± 0.0159 | 0.6927 ± 0.0200 |
| tfidf_chi2 | 0.6946 ± 0.0168 | 0.6950 ± 0.0189 |
| finbert_768 | 0.6738 ± 0.0353 | 0.6832 ± 0.0274 |
| finbert_pca50 | 0.6956 ± 0.0159 | 0.6907 ± 0.0195 |

**B. DeLong vs. Structured, mean ΔAUC and significance (14 folds), from `delong.csv`
(includes the BH-FDR q-value `finalize_delong()` adds — raw p<0.05 counts are shown
alongside the BH-adjusted q<0.05 counts, since most of the raw-significant folds do NOT
survive multiple-testing correction):**

| Representation | Model | mean ΔAUC | folds p<0.05 (raw) | folds q<0.05 (BH) |
|---|---|---|---|---|
| tfidf_full | Logistic | +0.0035 | 8/14 | 0/14 |
| tfidf_full | XGBoost | +0.0012 | 3/14 | 1/14 |
| tfidf_chi2 | Logistic | +0.0010 | 0/14 | 0/14 |
| tfidf_chi2 | XGBoost | +0.0034 | 2/14 | 0/14 |
| lexicon | Logistic | +0.0009 | 2/14 | 0/14 |
| lexicon | XGBoost | -0.0004 | 0/14 | 0/14 |
| finbert_pca50 | Logistic | +0.0019 | 3/14 | 0/14 |
| finbert_pca50 | XGBoost | -0.0009 | 0/14 | 0/14 |
| finbert_768 | Logistic | -0.0199 | 5/14 | 2/14 |
| finbert_768 | XGBoost | -0.0084 | 0/14 | 0/14 |
| structured_has_desc | Logistic | -0.0001 | 0/14 | 0/14 |
| structured_has_desc | XGBoost | -0.0006 | 1/14 | 0/14 |

**Takeaway:** after fixing the leakage (per-fold structured preprocessing) and applying
BH-FDR correction across folds, the leak-fixed picture is a *cleaner null* than the old
pipeline's own numbers (§6 below) — TF-IDF (full) is still the best-performing arm and the
only one with any raw significance to speak of, but its significance mostly does not
survive BH correction (Logistic: 8/14 raw-significant folds drops to 0/14 after
correction). FinBERT-768 (raw, unreduced) is the clearest loser, both in mean ΔAUC and in
being the only arm with BH-significant folds working *against* it. No representation shows
a robust, correction-surviving improvement over Structured alone in this leak-fixed
protocol.

Full outputs: `results/strict_temporal_v2/fold_metrics.csv` (per fold x representation x
model), `results/strict_temporal_v2/delong.csv` (per fold x model x representation, incl.
BH q-value), `results/strict_temporal_v2/predictions.csv` (raw per-row probabilities),
`results/strict_temporal_v2/preprocessing_audit.jsonl` (per-fold preprocessing trace).

## 6. Which old results are NOT directly comparable to strict_temporal_v2

Every existing result under `results/`, `TF-IDF/results/`,
`FinBERT/results/`, and `Dictionarys/results/` used the old,
globally-preprocessed structured block. They remain valid as **prior,
disclosed-limitation results** but should not be presented as
apples-to-apples with `results/strict_temporal_v2/` output — the
structured feature set, its exact column count, and (for a handful of
folds) which columns survive the missing/correlation filters can differ
between the two, since those filters are now fit per fold instead of once
globally.

TF-IDF and FinBERT-768 embeddings themselves are unchanged and *are*
directly reused (not recomputed) — only the structured block, and the new
`has_desc` addition, differ.

## 7. Which reruns are needed

- The full 14-fold `strict_temporal_v2` comparison (§8 below) — **complete** (all 14
  folds, results in §5 above).
- No changes needed to `TF-IDF/tfidf_pipeline.py` or
  `FinBERT/finbert_cls_embeddings.py` output — both reused as-is.
- The temporal grid-search final-config tuning (`hyperparams.temporal_grid_search`)
  is built but not yet run for any representation — deferred until a
  final representation is chosen, per the spec.

## 8. Exact commands to run the pipeline

```bash
# 1. Leakage-audit tests (fast, no data) — run first, always
python -m pytest strict_temporal_v2/tests/test_leakage.py -v

# 2. Sanity check on the reimplemented raw/global feature engineering
python -m strict_temporal_v2.raw_features

# 3. Smoke test — fold 1 only
python -m strict_temporal_v2.run_dry

# 4. Smoke test — fold 1 + fold 14
python -m strict_temporal_v2.run_dry --with-fold14

# 5. Full 14-fold run (NOT yet executed — writes results/strict_temporal_v2/)
python -m strict_temporal_v2.run_full
```

## 9. Deviations from the spec (and why)

- **Missing threshold**: spec text said "50%"; verified current code uses
  99%. Ariel confirmed keeping 99% (only the *fitting scope* changes to
  train-only, not the threshold value).
- **Shared module structure**: this repo's existing convention is "no
  shared code between ablation scripts" (each duplicates its own
  `build_folds()`). `strict_temporal_v2/` deliberately breaks that
  convention with one shared, importable package — needed for the 13
  leakage-audit unit tests to certify one implementation instead of
  seven duplicated copies. Flagged in the plan before implementation;
  no objection raised.
- **RandomForest**: not run in this comparison, per explicit instruction
  (§16 of the spec: "אין צורך כרגע להריץ מחדש Random Forest").
- No other methodological decisions were changed beyond what the spec
  explicitly requested.
