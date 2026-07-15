"""
FinBERT Embeddings for Loan Descriptions — CLS + Semantic (mean-pooled)
========================================================================
Extracts TWO deep representations for every loan request text (the `desc`
column) using FinBERT, in a single forward pass per text:

  1. CLS      — final-layer hidden state of the [CLS] token (768-dim).
  2. Semantic — attention-mask-weighted mean of all final-layer token
                states (768-dim). Standard sentence-embedding pooling;
                usually the better representation of the text's meaning.

Requirements: pip install torch transformers pyarrow  (all installed)

INPUT
    PATH_CSV — the post-03_advanced_prep CSV (same file preliminary_results_v2.py
    reads). Only `id` and `desc` are loaded.

OUTPUT (written to data/04_finbert/)
    finbert_desc_embeddings.parquet — ONE file, one row per loan application:
        id          LendingClub loan id (primary key — merge on this)
        desc        the CLEANED description (boilerplate/HTML removed;
                    empty string if the borrower wrote nothing)
        has_desc    True if desc is non-empty after cleaning
        n_tokens    FinBERT token count (0 if empty)
        cls_000..cls_767    CLS embedding      (float16; all-zero if empty)
        sem_000..sem_767    semantic embedding (float16; all-zero if empty)

MERGING DOWNSTREAM
    Merge on `id` — robust to any sorting or row filtering
    (preliminary_results_v2.py sorts by issue_month_start, so positional
    alignment breaks after its sort; never merge by row position).

NOTES
    - Text cleaning strips the LendingClub boilerplate ("Borrower added on
      MM/DD/YY >" and its variants), drops <br> tags and decodes HTML
      entities — leaving only the borrower's original request. No stopword
      removal or other NLP munging: FinBERT expects natural text.
    - Inference only (no dropout) — deterministic, no seed needed.
    - CPU-only machine: expect roughly 1.5-3 hours for ~120K non-empty
      descriptions. Progress is checkpointed in chunks, so the script can
      be stopped and resumed (already-done chunks are skipped).
"""

import html
import os
import re
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# PARAMETERS — edit these before running
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
PATH_CSV   = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260715_2119.csv"
TEXT_COL   = "desc"

# FinBERT checkpoint. "yiyanghkust/finbert-pretrain" is the domain-pretrained
# encoder (best suited for general-purpose embeddings). Alternative:
# "ProsusAI/finbert" — fine-tuned for sentiment; its CLS space is shaped by
# that task, which may or may not help default prediction.
MODEL_NAME = "yiyanghkust/finbert-pretrain"

MAX_LENGTH = 256    # tokens per text; LC descriptions are rarely longer
BATCH_SIZE = 32     # lower to 8-16 if RAM is tight
CHUNK_SIZE = 5_000  # rows per checkpoint file (resume granularity)

OUT_DIR    = BASE_DIR / "data" / "04_finbert"
CHUNK_DIR  = OUT_DIR / "chunks"
OUT_PATH   = OUT_DIR / "finbert_desc_embeddings.parquet"

# ============================================================
# 1. LOAD TEXTS
# ============================================================
print("Loading data...")
ID_COL = "id"  # LendingClub loan id — primary key, carried through the pipeline since 02_data_prep
df = pd.read_csv(PATH_CSV, usecols=[ID_COL, TEXT_COL], low_memory=False)
n_rows = len(df)
print(f"  Rows: {n_rows:,}")
if df[ID_COL].isna().any() or df[ID_COL].duplicated().any():
    raise ValueError(f"'{ID_COL}' is not a valid primary key in {PATH_CSV.name} "
                     f"(nulls or duplicates found) — aborting to avoid unkeyed embeddings.")

# Cleaning: drop LC boilerplate + HTML artifacts, keep the borrower's own text.
# The boilerplate prefix appears in three variants: "Borrower added on MM/DD/YY >",
# "<listing id> added on MM/DD/YY >", and bare "added on MM/DD/YY >". The date is
# what anchors the match — a borrower's own "added on ..." (no date) is never touched.
BOILERPLATE_RE = re.compile(r"(?:borrower\s+|\d+\s+)?added\s+on\s+\d{2}/\d{2}/\d{2}\s*>?", re.IGNORECASE)

def clean_for_bert(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = html.unescape(s)  # &quot;/&amp;/&#39;... -> the characters the borrower actually typed
    s = BOILERPLATE_RE.sub(" ", s)
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

texts    = df[TEXT_COL].map(clean_for_bert)
has_desc = (texts.str.len() > 0).values
print(f"  Non-empty descriptions: {int(has_desc.sum()):,} ({has_desc.mean():.1%})")

# ============================================================
# 2. MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {MODEL_NAME} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

HIDDEN_SIZE = model.config.hidden_size  # 768 for FinBERT (BERT-base)

@torch.no_grad()
def embed_batch(batch_texts):
    """One forward pass -> (CLS, semantic mean-pooled, n_tokens) for the batch."""
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    out    = model(**enc)
    hidden = out.last_hidden_state                      # (B, T, 768)
    mask   = enc["attention_mask"].unsqueeze(-1)        # (B, T, 1) — 1 on real tokens, 0 on padding

    cls = hidden[:, 0, :]                               # [CLS] token, final layer
    sem = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean over non-padding tokens

    n_tok = enc["attention_mask"].sum(dim=1)
    return (cls.cpu().numpy().astype(np.float32),
            sem.cpu().numpy().astype(np.float32),
            n_tok.cpu().numpy())

# ============================================================
# 3. EMBED IN CHUNKS (checkpointed — safe to stop and resume)
# ============================================================
os.makedirs(CHUNK_DIR, exist_ok=True)
n_chunks = int(np.ceil(n_rows / CHUNK_SIZE))
print(f"Embedding {n_rows:,} rows in {n_chunks} chunks of {CHUNK_SIZE:,}...")

token_counts = np.zeros(n_rows, dtype=np.int32)
t0 = time.time()

for ci in range(n_chunks):
    chunk_path = CHUNK_DIR / f"chunk_{ci:04d}.npz"
    lo, hi = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_rows)

    if chunk_path.exists():  # resume: skip chunks already done
        z = np.load(chunk_path)
        if "sem" not in z.files:  # stale chunk from the CLS-only version of this script
            raise RuntimeError(f"{chunk_path} is from an older CLS-only run — "
                               f"delete the chunks/ folder and rerun.")
        token_counts[lo:hi] = z["n_tokens"]
        print(f"  Chunk {ci + 1}/{n_chunks} already exists — skipped.")
        continue

    chunk_cls = np.zeros((hi - lo, HIDDEN_SIZE), dtype=np.float32)
    chunk_sem = np.zeros((hi - lo, HIDDEN_SIZE), dtype=np.float32)
    chunk_tok = np.zeros(hi - lo, dtype=np.int32)

    # Only non-empty texts go through the model; empty rows stay all-zeros
    idx_local  = np.where(has_desc[lo:hi])[0]
    batch_text = texts.iloc[lo:hi].values

    for bs in range(0, len(idx_local), BATCH_SIZE):
        sel = idx_local[bs:bs + BATCH_SIZE]
        cls, sem, ntok = embed_batch([batch_text[i] for i in sel])
        chunk_cls[sel] = cls
        chunk_sem[sel] = sem
        chunk_tok[sel] = ntok

    np.savez_compressed(chunk_path,
                        cls=chunk_cls.astype(np.float16),
                        sem=chunk_sem.astype(np.float16),
                        n_tokens=chunk_tok)
    token_counts[lo:hi] = chunk_tok
    elapsed = time.time() - t0
    done_chunks = ci + 1
    eta_min = elapsed / done_chunks * (n_chunks - done_chunks) / 60
    print(f"  Chunk {done_chunks}/{n_chunks} done ({int(has_desc[lo:hi].sum()):,} texts embedded) "
          f"— elapsed {elapsed/60:.0f}m, ETA {eta_min:.0f}m.")

# ============================================================
# 4. ASSEMBLE & SAVE — one merged parquet: id + desc + both embeddings
# ============================================================
print("Assembling final file...")
cls_all = np.zeros((n_rows, HIDDEN_SIZE), dtype=np.float16)
sem_all = np.zeros((n_rows, HIDDEN_SIZE), dtype=np.float16)
for ci in range(n_chunks):
    lo, hi = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_rows)
    z = np.load(CHUNK_DIR / f"chunk_{ci:04d}.npz")
    cls_all[lo:hi] = z["cls"]
    sem_all[lo:hi] = z["sem"]

out = pd.concat(
    [
        pd.DataFrame({
            "id":       df[ID_COL].values,
            "desc":     texts.values,          # cleaned text — the borrower's original request
            "has_desc": has_desc,
            "n_tokens": token_counts,
        }),
        pd.DataFrame(cls_all, columns=[f"cls_{i:03d}" for i in range(HIDDEN_SIZE)]),
        pd.DataFrame(sem_all, columns=[f"sem_{i:03d}" for i in range(HIDDEN_SIZE)]),
    ],
    axis=1,
)
out.to_parquet(OUT_PATH, index=False)

print(f"\nSaved: {OUT_PATH}")
print(f"  {out.shape[0]:,} rows x {out.shape[1]:,} cols "
      f"(id, desc, has_desc, n_tokens, 768 cls_*, 768 sem_*)")
print(f"  (chunk checkpoints kept in {CHUNK_DIR}; delete the folder to reclaim ~1.3GB)")
