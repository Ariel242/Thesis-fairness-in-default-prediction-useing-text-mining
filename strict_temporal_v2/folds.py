# -*- coding: utf-8 -*-
"""
folds.py
========
The single, shared, tested definition of the 14 walk-forward folds used
throughout F-TM-CR. Copied verbatim in logic/parameters from
analysis/BASELINE.py / TF-IDF/tfidf_pipeline.py / FinBERT/BASELINE_768.py
(all of which independently duplicate the same function per this repo's
existing convention) -- this module exists so strict_temporal_v2's leakage
tests have exactly one implementation to certify, instead of seven.

Does NOT change the fold definition in any way: same MIN_TRAIN_MONTHS=8,
STEP_MONTHS=3, MIN_TEST_DEFAULTS=100, same expanding-window logic.
"""

from dataclasses import dataclass, field

import pandas as pd

TARGET_COL = "is_default"
DATE_COL = "issue_month_start"
ID_COL = "id"

MIN_TRAIN_MONTHS = 8
STEP_MONTHS = 3
MIN_TEST_DEFAULTS = 100

EXPECTED_N_FOLDS = 14

# The known-correct 14-fold table, verified against
# TF-IDF/results/ablation_report.md and reproduced by every existing
# ablation script. Used by assert_fold_integrity() as a regression check.
EXPECTED_FOLD_TABLE = [
    {"fold": 1,  "train_cutoff": "2010-08", "test_end": "2010-09", "n_train": 6694,   "n_test": 1042,  "n_test_defaults": 157},
    {"fold": 2,  "train_cutoff": "2010-11", "test_end": "2010-12", "n_train": 9893,   "n_test": 1222,  "n_test_defaults": 136},
    {"fold": 3,  "train_cutoff": "2011-02", "test_end": "2011-03", "n_train": 13674,  "n_test": 1369,  "n_test_defaults": 184},
    {"fold": 4,  "train_cutoff": "2011-05", "test_end": "2011-06", "n_train": 18165,  "n_test": 1762,  "n_test_defaults": 255},
    {"fold": 5,  "train_cutoff": "2011-08", "test_end": "2011-09", "n_train": 23547,  "n_test": 1979,  "n_test_defaults": 286},
    {"fold": 6,  "train_cutoff": "2011-11", "test_end": "2011-12", "n_train": 29671,  "n_test": 2190,  "n_test_defaults": 406},
    {"fold": 7,  "train_cutoff": "2012-02", "test_end": "2012-03", "n_train": 36786,  "n_test": 2781,  "n_test_defaults": 413},
    {"fold": 8,  "train_cutoff": "2012-05", "test_end": "2012-06", "n_train": 45976,  "n_test": 3713,  "n_test_defaults": 681},
    {"fold": 9,  "train_cutoff": "2012-08", "test_end": "2012-09", "n_train": 59483,  "n_test": 5975,  "n_test_defaults": 919},
    {"fold": 10, "train_cutoff": "2012-11", "test_end": "2012-12", "n_train": 77872,  "n_test": 5970,  "n_test_defaults": 893},
    {"fold": 11, "train_cutoff": "2013-02", "test_end": "2013-03", "n_train": 98111,  "n_test": 8184,  "n_test_defaults": 1164},
    {"fold": 12, "train_cutoff": "2013-05", "test_end": "2013-06", "n_train": 125865, "n_test": 10802, "n_test_defaults": 1739},
    {"fold": 13, "train_cutoff": "2013-08", "test_end": "2013-09", "n_train": 161023, "n_test": 12846, "n_test_defaults": 2007},
    {"fold": 14, "train_cutoff": "2013-11", "test_end": "2013-12", "n_train": 202442, "n_test": 14845, "n_test_defaults": 2263},
]


@dataclass
class Fold:
    fold: int
    train_cutoff: str
    test_end: str
    n_train: int
    n_test: int
    n_test_defaults: int
    tr_idx: list = field(repr=False)
    te_idx: list = field(repr=False)


def build_folds(df: pd.DataFrame, date_col: str = DATE_COL, target_col: str = TARGET_COL,
                 min_train_months: int = MIN_TRAIN_MONTHS, step_months: int = STEP_MONTHS,
                 min_test_defaults: int = MIN_TEST_DEFAULTS) -> list[Fold]:
    """Expanding-window walk-forward folds -- identical logic to every
    existing ablation script's own copy of this function."""
    months = df[date_col].dt.to_period("M")
    all_periods = sorted(months.unique())

    folds = []
    fold_start_idx = min_train_months

    while fold_start_idx < len(all_periods):
        train_cutoff = all_periods[fold_start_idx - 1]

        test_end_idx = fold_start_idx
        while test_end_idx < len(all_periods):
            test_period = all_periods[test_end_idx]
            te_mask = (months > train_cutoff) & (months <= test_period)
            if df.loc[te_mask, target_col].sum() >= min_test_defaults:
                break
            test_end_idx += 1

        if test_end_idx >= len(all_periods):
            break

        test_period = all_periods[test_end_idx]
        tr_mask = months <= train_cutoff
        te_mask = (months > train_cutoff) & (months <= test_period)

        folds.append(Fold(
            fold=len(folds) + 1,
            train_cutoff=str(train_cutoff),
            test_end=str(test_period),
            n_train=int(tr_mask.sum()),
            n_test=int(te_mask.sum()),
            n_test_defaults=int(df.loc[te_mask, target_col].sum()),
            tr_idx=df.index[tr_mask].tolist(),
            te_idx=df.index[te_mask].tolist(),
        ))

        fold_start_idx += step_months

    return folds


def assert_fold_integrity(folds: list[Fold], df: pd.DataFrame, date_col: str = DATE_COL) -> None:
    """The leakage-audit assertions from the spec's section 15 (items 10-11,
    plus the fold-table regression check). Raises AssertionError with a
    specific message on any violation."""
    assert len(folds) == EXPECTED_N_FOLDS, (
        f"Expected {EXPECTED_N_FOLDS} folds, got {len(folds)} -- fold definition changed."
    )

    all_test_idx: set[int] = set()
    for f in folds:
        tr_set, te_set = set(f.tr_idx), set(f.te_idx)

        # (a) train/test disjoint within the fold
        assert tr_set.isdisjoint(te_set), f"Fold {f.fold}: train/test overlap within the fold."

        # (b) every test-row issue_d is later than every train-row issue_d in that fold
        max_train_date = df.loc[f.tr_idx, date_col].max()
        min_test_date = df.loc[f.te_idx, date_col].min()
        assert min_test_date > max_train_date, (
            f"Fold {f.fold}: earliest test date {min_test_date} is not after "
            f"latest train date {max_train_date}."
        )

        # (c) no loan appears as test in more than one fold
        overlap = all_test_idx & te_set
        assert not overlap, f"Fold {f.fold}: {len(overlap)} test rows already appeared as test in an earlier fold."
        all_test_idx |= te_set

        # (d) fold sizes match the known-correct table (regression check)
        expected = EXPECTED_FOLD_TABLE[f.fold - 1]
        assert f.train_cutoff == expected["train_cutoff"], (
            f"Fold {f.fold}: train_cutoff {f.train_cutoff} != expected {expected['train_cutoff']}."
        )
        assert f.test_end == expected["test_end"], (
            f"Fold {f.fold}: test_end {f.test_end} != expected {expected['test_end']}."
        )
        assert f.n_train == expected["n_train"], (
            f"Fold {f.fold}: n_train {f.n_train} != expected {expected['n_train']}."
        )
        assert f.n_test == expected["n_test"], (
            f"Fold {f.fold}: n_test {f.n_test} != expected {expected['n_test']}."
        )
        assert f.n_test_defaults == expected["n_test_defaults"], (
            f"Fold {f.fold}: n_test_defaults {f.n_test_defaults} != expected {expected['n_test_defaults']}."
        )
