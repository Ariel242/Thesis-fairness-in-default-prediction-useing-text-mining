# Fairness in Default Prediction Using Text Mining

A research project examining whether adding unstructured text data (loan descriptions) to credit risk models affects both **predictive accuracy** and **algorithmic fairness** across geographic groups.

## What This Project Does

Using LendingClub loan data (2010–2013, ~220K loans), the project trains two types of models:

- **Structured only** — traditional credit features (loan amount, grade, DTI, income, etc.)
- **Structured + Text** — same features plus TF-IDF representation of the borrower's loan description

Both model types (Logistic Regression and XGBoost) are evaluated using **walk-forward cross-validation** — a time-series protocol where the model is always trained on past data and tested on future data, preventing data leakage.

## Research Question

> Does adding text features improve or harm fairness across geographic groups (ZIP codes)?

Fairness is measured by the gap in error rates (FNR, FPR) between different ZIP code groups, using census demographic data to provide context.

## Project Structure

```
analysis/       # Data preparation and modeling notebooks + main pipeline script (structured + TF-IDF)
data/           # Processed datasets (raw file, 02/03 pipeline outputs)
census/         # Census data collection notebook + ACS demographic features matched to ZIP3s
FinBERT/        # FinBERT text representation: embedding script, embeddings, ablation script, results/figures
Dictionarys/    # Dictionary-based text representation: build scripts (scripts/), the dictionaries themselves
                # (dictionaries/), and results/ (once a dictionary-based model is run)
TF-IDF/         # Landing place for TF-IDF-specific results once split out of preliminary_results_v2.py
                # (currently TF-IDF is generated inline in the main pipeline; see TF-IDF/README.md)
results/        # Main pipeline outputs: predictive metrics, fairness metrics, fold-level results
docs/           # Pipeline and outputs documentation
הצעה לתזה/       # Archive: all model results generated before this repo reorganization (thesis proposal era)
```

## Key Results

> **Note:** the results below are from `analysis/archive/preliminary_results_cloude.py` (v1). A code review found that v1's "Structured" baseline was not actually text-free (it included text-derived numeric stats) and that it silently dropped LendingClub's `grade`/`sub_grade` risk rating — see `docs/pipeline.md` ("Step 5 (v2) — Changelog") for details. Both are fixed in `analysis/preliminary_results_v2.py`; the numbers below should be treated as provisional until v2's full run replaces them.

- Adding text features had **minimal effect on predictive accuracy** (ΔAUC ≈ 0%)
- Effects on fairness were **model-dependent**: slight degradation in Logistic Regression, slight improvement in XGBoost
- Geographic disparities in error rates persist regardless of feature set

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md) — step-by-step pipeline description: inputs, outputs, and parameters for each notebook and script
- [`docs/outputs.md`](docs/outputs.md) — schema and description of every results file

## Data

Raw LendingClub data is not included in this repository due to file size. It can be downloaded from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
