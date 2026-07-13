"""
FinBERT CLS Embeddings for Loan Descriptions
=============================================
Extracts a deep representation for every loan request text (the `desc`
column) using FinBERT: each text is passed through the encoder and the
final-layer hidden state of the [CLS] token (768-dim) is kept.

Requirements (not yet installed in this environment):
    pip install torch transformers

INPUT
    PATH_CSV — the post-03_advanced_prep CSV (same file preliminary_results_v2.py
    reads). Row order of the outputs matches the row order of this file.

OUTPUT (written to data/04_finbert/)
    finbert_cls_embeddings.npy   — float16 array, shape (n_rows, 768).
                                   Row i corresponds to row i of PATH_CSV.
                                   Rows with empty/missing desc are all-zeros.
    finbert_cls_meta.parquet     — per-row metadata: id (LendingClub loan
                                   id, the primary key), row_id (position
                                   in PATH_CSV), has_desc flag, n_tokens.

MERGING DOWNSTREAM
    Preferred: merge on `id` — the loan-level primary key carried through
    the pipeline since 02_data_prep. This is robust to any sorting or row
    filtering. Positional row_id is kept as a fallback/sanity check; note
    that preliminary_results_v2.py sorts by issue_month_start, so positional
    alignment is only valid immediately after pd.read_csv, before that sort.

NOTES
    - Text cleaning is minimal on purpose: FinBERT expects natural text,
      so we only strip the LendingClub boilerplate ("Borrower added on
      MM/DD/YY >") and HTML breaks. No stopword removal, no lowercasing
      beyond what the tokenizer does.
    - Embedding extraction is deterministic (inference only, no dropout),
      so no random seed is needed.
    - Runs on GPU if available, otherwise CPU (expect hours on CPU for
      ~100K non-empty descriptions; progress is checkpointed in chunks,
      so the script can be stopped and resumed).
"""

import os
import re
import sys
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
PATH_CSV   = BASE_DIR / "data" / "03_advanced_prep" / "lc_after_03_advanced_prep_basic+test_20260709_1340.csv"
TEXT_COL   = "desc"

# FinBERT checkpoint. "yiyanghkust/finbert-pretrain" is the domain-pretrained
# encoder (best suited for general-purpose embeddings). Alternative:
# "ProsusAI/finbert" — fine-tuned for sentiment; its CLS space is shaped by
# that task, which may or may not help default prediction.
MODEL_NAME = "yiyanghkust/finbert-pretrain"

MAX_LENGTH = 256    # tokens per text; LC descriptions are rarely longer
BATCH_SIZE = 32     # lower to 8-16 if RAM/VRAM is tight
CHUNK_SIZE = 5_000  # rows per checkpoint file (resume granularity)

OUT_DIR    = BASE_DIR / "data" / "04_finbert"
CHUNK_DIR  = OUT_DIR / "chunks"
EMB_PATH   = OUT_DIR / "finbert_cls_embeddings.npy"
META_PATH  = OUT_DIR / "finbert_cls_meta.parquet"

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

# Minimal cleaning: drop LC boilerplate + HTML breaks, keep natural text
BOILERPLATE_RE = re.compile(r"borrower\s+added\s+on\s+\d{2}/\d{2}/\d{2}\s*>?", re.IGNORECASE)

def clean_for_bert(text):
    if pd.isna(text):
        return ""
    s = str(text)
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
    """Returns (len(batch_texts), HIDDEN_SIZE) float32 array of CLS embeddings."""
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]  # [CLS] token, final layer
    return cls.cpu().numpy().astype(np.float32), enc["attention_mask"].sum(dim=1).cpu().numpy()

# ============================================================
# 3. EMBED IN CHUNKS (checkpointed — safe to stop and resume)
# ============================================================
os.makedirs(CHUNK_DIR, exist_ok=True)
n_chunks = int(np.ceil(n_rows / CHUNK_SIZE))
print(f"Embedding {n_rows:,} rows in {n_chunks} chunks of {CHUNK_SIZE:,}...")

token_counts = np.zeros(n_rows, dtype=np.int32)

for ci in range(n_chunks):
    chunk_path = CHUNK_DIR / f"chunk_{ci:04d}.npz"
    lo, hi = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_rows)

    if chunk_path.exists():  # resume: skip chunks already done
        token_counts[lo:hi] = np.load(chunk_path)["n_tokens"]
        print(f"  Chunk {ci + 1}/{n_chunks} already exists — skipped.")
        continue

    chunk_emb = np.zeros((hi - lo, HIDDEN_SIZE), dtype=np.float32)
    chunk_tok = np.zeros(hi - lo, dtype=np.int32)

    # Only non-empty texts go through the model; empty stay all-zeros
    idx_local  = np.where(has_desc[lo:hi])[0]
    batch_text = texts.iloc[lo:hi].values

    for bs in range(0, len(idx_local), BATCH_SIZE):
        sel = idx_local[bs:bs + BATCH_SIZE]
        emb, ntok = embed_batch([batch_text[i] for i in sel])
        chunk_emb[sel] = emb
        chunk_tok[sel] = ntok

    np.savez_compressed(chunk_path, emb=chunk_emb.astype(np.float16), n_tokens=chunk_tok)
    token_counts[lo:hi] = chunk_tok
    print(f"  Chunk {ci + 1}/{n_chunks} done ({int(has_desc[lo:hi].sum()):,} texts embedded).")

# ============================================================
# 4. ASSEMBLE & SAVE
# ============================================================
print("Assembling final matrix...")
embeddings = np.zeros((n_rows, HIDDEN_SIZE), dtype=np.float16)
for ci in range(n_chunks):
    lo, hi = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_rows)
    embeddings[lo:hi] = np.load(CHUNK_DIR / f"chunk_{ci:04d}.npz")["emb"]

np.save(EMB_PATH, embeddings)

meta = pd.DataFrame({
    "id":       df[ID_COL].values,
    "row_id":   np.arange(n_rows, dtype=np.int64),
    "has_desc": has_desc,
    "n_tokens": token_counts,
})
meta.to_parquet(META_PATH, index=False)

print(f"\nSaved:")
print(f"  {EMB_PATH}  — shape {embeddings.shape}, dtype float16")
print(f"  {META_PATH} — id / row_id / has_desc / n_tokens")
print(f"  (chunk checkpoints kept in {CHUNK_DIR}; delete the folder to reclaim space)")
