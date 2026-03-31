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
analysis/          # Data preparation and modeling notebooks + main pipeline script
data/              # Processed datasets and census data
results/           # Model outputs: predictive metrics, fairness metrics, fold-level results
census_final_data/ # ACS demographic panel (2011–2014) matched to ZIP codes
```

## Key Results

- Adding text features had **minimal effect on predictive accuracy** (ΔAUC ≈ 0%)
- Effects on fairness were **model-dependent**: slight degradation in Logistic Regression, slight improvement in XGBoost
- Geographic disparities in error rates persist regardless of feature set

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md) — step-by-step pipeline description: inputs, outputs, and parameters for each notebook and script

## Data

Raw LendingClub data is not included in this repository due to file size. It can be downloaded from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
