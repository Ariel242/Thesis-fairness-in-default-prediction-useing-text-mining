# Frozen Vocabulary Extraction -- Audit Manifest

- Executed: 2026-08-20T18:39:14
- Input CSV: `data\03_advanced_prep\lc_after_03_advanced_prep_basic+test_20260715_2119.csv`
- Date column: `issue_month_start` | Target column: `is_default`
- MIN_TRAIN_MONTHS: 8 (must match TF-IDF/tfidf_pipeline.py's build_folds())
- Initial window: 2010-01 .. 2010-08
- Rows in window (is_default in [0,1]): 6,694 / 217,287 total rows
- Non-empty desc rows in window: 4,503
- min_df (document frequency pruning, PDF's freq>15): 16
- ngram_range: (1, 2) | top-N cap: none (uncapped)
- Vocabulary size: 1,650 (Unigrams: 1,015, Bigrams: 635)
- Output: `Dictionarys\dictionaries\frozen_vocabulary_initial_window.csv`
