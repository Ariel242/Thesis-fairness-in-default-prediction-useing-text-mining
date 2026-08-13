# TF-IDF

TF-IDF is currently generated inline inside `analysis/preliminary_results_v2.py` (fit on `text_all_clean`, top `TFIDF_MAX_FEATURES` unigrams/bigrams — see `docs/pipeline.md`), not as a standalone script.

This folder exists as the landing place for TF-IDF-specific outputs once that logic is split out of `preliminary_results_v2.py`. Until then, `results/` and `figures/` stay empty and TF-IDF results live alongside the rest of the main pipeline's output in `results/`.
