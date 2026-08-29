# -*- coding: utf-8 -*-
"""
representations.py
====================
build_final_matrix() is the SINGLE, tested path that builds the final
train/test matrix for every representation. It replaces an earlier design
with one loosely-scaled helper per representation, duplicated separately
in run_dry.py and run_full.py -- that duplication is exactly how a real
bug slipped through (see the docstring on build_final_matrix for the
scaling bug found and fixed live during the fold-14 dry run).

Reused, not recomputed:
  - TF-IDF (full + chi2-500): loaded from TF-IDF/output/fold{NN}/*.npz,
    already fit train-only per fold by the existing tfidf_pipeline.py.
  - FinBERT embeddings (768-dim): loaded from
    FinBERT/data/finbert_desc_embeddings.parquet, a frozen pretrained
    feature extractor's output -- not fit on this data at all.
  - Lexicon: loaded from Dictionarys/output/lexicon_features.csv, a
    frozen dictionary's output -- not fit per fold by design.

Newly fit per fold here (train-only, no leakage):
  - FinBERT-PCA50: PCA(n_components=50, svd_solver='randomized',
    random_state=242), fit on the fold's train sem_* embeddings
    (including zero-vectors for missing desc), no pre-scaling -- exactly
    mirroring FinBERT/BASELINE_PCA50.py's own logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from strict_temporal_v2 import structured_preprocessing as sp_prep

BASE_DIR = Path(__file__).resolve().parent.parent
TFIDF_OUTPUT_DIR = BASE_DIR / "TF-IDF" / "output"
FINBERT_PARQUET = BASE_DIR / "FinBERT" / "data" / "finbert_desc_embeddings.parquet"
LEXICON_FEATURES_CSV = BASE_DIR / "Dictionarys" / "output" / "lexicon_features.csv"

ID_COL = "id"
SEM_COLS = [f"sem_{i:03d}" for i in range(768)]
N_PCA_COMPONENTS = 50
PCA_RANDOM_STATE = 242

LEXICON_COLS = ["risk_word_count", "risk_word_density", "risk_word_flag",
                 "distress_debt_pressure_count", "distress_liquidity_shortage_count",
                 "distress_medical_life_shock_count", "distress_refinancing_burden_count"]

REPRESENTATIONS = [
    "structured", "structured_has_desc", "lexicon", "tfidf_full",
    "tfidf_chi2", "finbert_768", "finbert_pca50",
]


def _has_desc_block(has_desc_series: pd.Series) -> np.ndarray:
    return has_desc_series.values.reshape(-1, 1).astype(np.float32)


def _scale(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te)


# Module-level caches -- both files are read by every fold x arm call
# (lexicon: 2 arms, FinBERT: 2 arms x 14 folds = up to 28 re-reads of a
# 666MB parquet without this). Loaded once per process, reused thereafter;
# a full 14-fold run_full.py execution is one process, so this is a real
# saving, not a leak across separate runs.
_LEXICON_CACHE: pd.DataFrame | None = None
_FINBERT_CACHE: pd.DataFrame | None = None


def _get_lexicon_df() -> pd.DataFrame:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is None:
        _LEXICON_CACHE = pd.read_csv(LEXICON_FEATURES_CSV).set_index(ID_COL)
    return _LEXICON_CACHE


def _get_finbert_df() -> pd.DataFrame:
    global _FINBERT_CACHE
    if _FINBERT_CACHE is None:
        _FINBERT_CACHE = pd.read_parquet(FINBERT_PARQUET, columns=[ID_COL] + SEM_COLS).set_index(ID_COL)
    return _FINBERT_CACHE


def _load_lexicon_block(ids_tr: pd.Series, ids_te: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lex = _get_lexicon_df()
    lex_tr = lex.loc[ids_tr.values, LEXICON_COLS].values.astype(np.float32)
    lex_te = lex.loc[ids_te.values, LEXICON_COLS].values.astype(np.float32)
    return lex_tr, lex_te


def _load_finbert_sem_block(ids_tr: pd.Series, ids_te: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    fb = _get_finbert_df()
    sem_tr = fb.loc[ids_tr.values, SEM_COLS].values.astype(np.float32)
    sem_te = fb.loc[ids_te.values, SEM_COLS].values.astype(np.float32)
    return sem_tr, sem_te


def attach_tfidf_full(X_tr, X_te, fold_num: int):
    """X_tr/X_te here must already be the SCALED (structured[+has_desc])
    dense block -- this only sparse-hstacks the fold's already-fit
    (train-only) TF-IDF matrix onto it, matching
    TF-IDF/tfidf_structured_ablation.py exactly (structured block scaled,
    TF-IDF block never rescaled)."""
    fold_dir = TFIDF_OUTPUT_DIR / f"fold{fold_num:02d}"
    X_tf_tr = sp.load_npz(fold_dir / "train_tfidf.npz").tocsr()
    X_tf_te = sp.load_npz(fold_dir / "test_tfidf.npz").tocsr()
    vocab = pd.read_csv(fold_dir / "vocab.txt", header=None)[0].astype(str).tolist()
    names = [f"tfidf:{w}" for w in vocab]
    return sp.hstack([sp.csr_matrix(X_tr), X_tf_tr], format="csr"), \
        sp.hstack([sp.csr_matrix(X_te), X_tf_te], format="csr"), names


def attach_tfidf_chi2(X_tr, X_te, fold_num: int, y_tr: np.ndarray, k: int = 500):
    from sklearn.feature_selection import SelectKBest, chi2
    fold_dir = TFIDF_OUTPUT_DIR / f"fold{fold_num:02d}"
    X_tf_tr = sp.load_npz(fold_dir / "train_tfidf.npz").tocsr()
    X_tf_te = sp.load_npz(fold_dir / "test_tfidf.npz").tocsr()
    vocab = pd.read_csv(fold_dir / "vocab.txt", header=None)[0].astype(str).tolist()

    k_eff = min(k, X_tf_tr.shape[1])
    selector = SelectKBest(chi2, k=k_eff)
    selector.fit(X_tf_tr, y_tr)
    idx = selector.get_support(indices=True)
    selected_vocab = [vocab[i] for i in idx]
    X_sel_tr = selector.transform(X_tf_tr)
    X_sel_te = selector.transform(X_tf_te)
    names = [f"tfidf_chi2:{w}" for w in selected_vocab]
    return sp.hstack([sp.csr_matrix(X_tr), X_sel_tr], format="csr"), \
        sp.hstack([sp.csr_matrix(X_te), X_sel_te], format="csr"), names, selected_vocab


def build_final_matrix(df: pd.DataFrame, tr_idx: list, te_idx: list, representation: str,
                        fold_num: int, y_tr: np.ndarray):
    """The single, tested matrix-building path for every representation.
    Correctly replicates each arm's EXISTING scaling precedent (verified
    against FinBERT/BASELINE_768.py, FinBERT/BASELINE_PCA50.py,
    TF-IDF/tfidf_structured_ablation.py, and Dictionarys' own lexicon
    ablation):

      structured, structured_has_desc, lexicon:
        ONE StandardScaler over the whole combined dense block (structured
        [+ has_desc] [+ lexicon]) -- matches the lexicon ablation's own
        approach of scaling STRUCT_COLS+LEXICON_COLS together. has_desc is
        new to structured/structured_has_desc (no prior arm to match
        against); combined-scale treatment is a documented, reasonable
        default, not a change to an existing decision.

      finbert_768, finbert_pca50:
        TWO separate scalers -- one for the structured block alone, one for
        (has_desc + embedding/PCA block) -- exactly matching
        FinBERT/BASELINE_768.py and BASELINE_PCA50.py's own build_matrix().

      tfidf_full, tfidf_chi2:
        ONE scaler for (structured + has_desc) only; the TF-IDF block is
        never rescaled (relies on its own L2-normalized weighting) and is
        sparse-hstacked on afterward -- exactly matching
        TF-IDF/tfidf_structured_ablation.py.

    BUG FOUND AND FIXED (fold-14 dry run): the original draft of this
    module skipped scaling entirely on the sparse (TF-IDF) path, leaving
    the structured block's raw, wildly-different-magnitude values (e.g.
    loan amounts in the thousands next to 0/1 dummies) unscaled next to
    TF-IDF's small L2-normalized weights. This broke Logistic Regression's
    saga solver -- both TF-IDF arms converged to an identical, badly
    degraded AUC=0.5834, while XGBoost (scale-invariant per feature) was
    unaffected and stayed ~0.70. Fixed by scaling the structured block
    before the sparse hstack, as above.

    Returns (X_train, X_test, feature_names, audit_dict) -- X_train/X_test
    are dense np.ndarray for all representations except tfidf_full/chi2,
    which are scipy.sparse.csr_matrix.
    """
    result = sp_prep.preprocess_fold(df, tr_idx, te_idx)
    X_struct_tr = result.X_train.values.astype(np.float32)
    X_struct_te = result.X_test.values.astype(np.float32)
    struct_names = list(result.feature_names)
    audit = dict(result.audit)

    hd_tr = _has_desc_block(df.loc[tr_idx, "has_desc"])
    hd_te = _has_desc_block(df.loc[te_idx, "has_desc"])

    if representation == "structured":
        X_tr, X_te = _scale(X_struct_tr, X_struct_te)
        names = struct_names

    elif representation == "structured_has_desc":
        combo_tr = np.hstack([X_struct_tr, hd_tr])
        combo_te = np.hstack([X_struct_te, hd_te])
        X_tr, X_te = _scale(combo_tr, combo_te)
        names = struct_names + ["has_desc"]

    elif representation == "lexicon":
        lex_tr, lex_te = _load_lexicon_block(df.loc[tr_idx, "id"], df.loc[te_idx, "id"])
        combo_tr = np.hstack([X_struct_tr, hd_tr, lex_tr])
        combo_te = np.hstack([X_struct_te, hd_te, lex_te])
        X_tr, X_te = _scale(combo_tr, combo_te)
        names = struct_names + ["has_desc"] + LEXICON_COLS

    elif representation in ("tfidf_full", "tfidf_chi2"):
        combo_tr = np.hstack([X_struct_tr, hd_tr])
        combo_te = np.hstack([X_struct_te, hd_te])
        X_struct_scaled_tr, X_struct_scaled_te = _scale(combo_tr, combo_te)
        if representation == "tfidf_full":
            X_tr, X_te, tf_names = attach_tfidf_full(X_struct_scaled_tr, X_struct_scaled_te, fold_num)
            audit["tfidf_vocab_size"] = len(tf_names)
        else:
            X_tr, X_te, tf_names, sel_vocab = attach_tfidf_chi2(
                X_struct_scaled_tr, X_struct_scaled_te, fold_num, y_tr)
            audit["chi2_selected_terms"] = sel_vocab
        names = struct_names + ["has_desc"] + tf_names

    elif representation in ("finbert_768", "finbert_pca50"):
        X_struct_scaled_tr, X_struct_scaled_te = _scale(X_struct_tr, X_struct_te)
        sem_tr, sem_te = _load_finbert_sem_block(df.loc[tr_idx, "id"], df.loc[te_idx, "id"])

        if representation == "finbert_768":
            fb_raw_tr, fb_raw_te = np.hstack([hd_tr, sem_tr]), np.hstack([hd_te, sem_te])
            fb_names = ["has_desc"] + [f"fb768:{c}" for c in SEM_COLS]
        else:
            pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver="randomized", random_state=PCA_RANDOM_STATE)
            pca_tr = pca.fit_transform(sem_tr)  # includes zero-vector rows (no desc)
            pca_te = pca.transform(sem_te)
            audit["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
            fb_raw_tr, fb_raw_te = np.hstack([hd_tr, pca_tr]), np.hstack([hd_te, pca_te])
            fb_names = ["has_desc"] + [f"fbpca:{i:03d}" for i in range(N_PCA_COMPONENTS)]

        fb_scaled_tr, fb_scaled_te = _scale(fb_raw_tr, fb_raw_te)
        X_tr = np.hstack([X_struct_scaled_tr, fb_scaled_tr])
        X_te = np.hstack([X_struct_scaled_te, fb_scaled_te])
        names = struct_names + fb_names

    else:
        raise ValueError(representation)

    audit["final_feature_count"] = len(names)
    return X_tr, X_te, names, audit
