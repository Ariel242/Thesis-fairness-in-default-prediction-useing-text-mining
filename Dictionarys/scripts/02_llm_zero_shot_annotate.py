# -*- coding: utf-8 -*-
"""
02_llm_zero_shot_annotate.py
=============================
Step 2 of the Distress Lexicon protocol (see
Dictionarys/thesis_lexicon_plan_updated_260820_023907.pdf, section 2,
"Zero-Shot LLM Annotation via Structured Rubric").

Submits the frozen vocabulary (Dictionarys/dictionaries/frozen_vocabulary_initial_window.csv,
produced by 01_extract_frozen_vocabulary.py) to Claude via the real Anthropic
API, using the exact prompt stored in
Dictionarys/prompts/zero_shot_lexicon_rubric_v1.json (single frozen source of
truth -- never re-typed inline here).

Requires:
  pip install anthropic
  ANTHROPIC_API_KEY set in the environment (or another credential source the
  SDK resolves automatically -- see `anthropic.Anthropic()` docs).

Known deviation from the PDF: the PDF's audit requirement says "Temperature
fixed at 0.0". Claude Opus 5 no longer accepts a `temperature` parameter at
all (400 error) -- there is no substitute knob for byte-identical
determinism on this model generation. This is logged explicitly in the audit
file instead of a temperature value.

Usage:
  python 02_llm_zero_shot_annotate.py            # full run
  python 02_llm_zero_shot_annotate.py --limit 20  # small test slice first
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import anthropic
except ImportError:
    print("ERROR: the 'anthropic' package is not installed. Run: pip install anthropic")
    raise SystemExit(1)

# ============================================================
# PARAMETERS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent

VOCAB_CSV    = DICT_DIR / "dictionaries" / "frozen_vocabulary_initial_window.csv"
PROMPT_JSON  = DICT_DIR / "prompts" / "zero_shot_lexicon_rubric_v1.json"

OUT_LEXICON_CSV = DICT_DIR / "dictionaries" / "LC_Distress_Lexicon_v1.csv"
OUT_RAW_CSV      = DICT_DIR / "results" / "llm_raw_annotations.csv"
OUT_AUDIT_LOG     = DICT_DIR / "results" / "annotation_audit_log.md"

MODEL       = "claude-opus-5"
CHUNK_SIZE  = 50
MAX_TOKENS  = 4000
EFFORT      = "low"  # rubric-following classification, not deep reasoning

VALID_CATEGORIES = {
    "DEBT_PRESSURE", "LIQUIDITY_SHORTAGE", "SHOCK_EMERGENCY",
    "HIGH_RISK_REFINANCING", "NEUTRAL_BENIGN",
}

CATEGORY_COUNT_COL = {
    "DEBT_PRESSURE": "distress_debt_pressure_count",
    "LIQUIDITY_SHORTAGE": "distress_liquidity_shortage_count",
    "SHOCK_EMERGENCY": "distress_medical_life_shock_count",
    "HIGH_RISK_REFINANCING": "distress_refinancing_burden_count",
}


def resolve_api_key() -> str | None:
    """Prefer the process environment; fall back to the Windows User-scope
    env var (registry) for cases where this script's own process launched
    before `setx ANTHROPIC_API_KEY ...` was run elsewhere. Never logs or
    prints the value."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')"],
            capture_output=True, text=True, timeout=10,
        )
        key = result.stdout.strip()
        return key or None
    except Exception:
        return None


def load_prompt() -> dict:
    with open(PROMPT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def call_with_retry(client, max_retries=5, base_delay=1.0, max_delay=30.0, **kwargs):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e
            else:
                raise
        delay = min(base_delay * (2 ** attempt), max_delay)
        print(f"    retry {attempt + 1}/{max_retries} after {delay:.1f}s ({last_exc})")
        time.sleep(delay)
    raise last_exc


def parse_csv_rows(text: str, expected_terms: list[str]) -> list[dict]:
    """Parse the model's CSV response robustly (handles quoted commas)."""
    text = text.strip()
    # strip accidental markdown fences if the model added them anyway
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    reader = csv.reader(io.StringIO(text))
    rows = []
    for parts in reader:
        if not parts or not parts[0].strip():
            continue
        if len(parts) < 3:
            print(f"    WARNING: malformed row (skipped): {parts}")
            continue
        token = parts[0].strip()
        category = parts[1].strip().upper()
        is_distress_raw = parts[2].strip()
        rationale = parts[3].strip() if len(parts) > 3 else ""

        if category not in VALID_CATEGORIES:
            print(f"    WARNING: unexpected category '{category}' for token '{token}'")
        try:
            is_distress = int(is_distress_raw)
        except ValueError:
            is_distress = 1 if category != "NEUTRAL_BENIGN" else 0

        rows.append({
            "token": token,
            "category": category,
            "is_distress": is_distress,
            "rationale": rationale,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only classify the first N terms (smoke test).")
    args = parser.parse_args()

    prompt = load_prompt()
    system_prompt = prompt["system"]
    user_template = prompt["user_template"]

    vocab = pd.read_csv(VOCAB_CSV)
    terms = vocab["Term"].astype(str).tolist()
    if args.limit:
        terms = terms[:args.limit]
    print(f"Classifying {len(terms):,} terms in chunks of {CHUNK_SIZE}...")

    api_key = resolve_api_key()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in the process environment "
              "or the Windows User environment. Set it with:\n"
              '  setx ANTHROPIC_API_KEY "sk-ant-..."\n'
              "then open a new terminal (or restart this app) and try again.")
        raise SystemExit(1)
    client = anthropic.Anthropic(api_key=api_key)

    all_rows = []
    n_chunks = 0
    t_start = time.time()

    for chunk in chunk_list(terms, CHUNK_SIZE):
        n_chunks += 1
        user_msg = user_template.format(n=len(chunk), terms="\n".join(chunk))

        response = call_with_retry(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": EFFORT},
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        if response.stop_reason == "refusal":
            print(f"  chunk {n_chunks}: REFUSED -- {response.stop_details}")
            continue

        text = "".join(b.text for b in response.content if b.type == "text")
        rows = parse_csv_rows(text, chunk)

        got = {r["token"] for r in rows}
        missing = [t for t in chunk if t not in got]
        if missing:
            print(f"  chunk {n_chunks}: {len(missing)} terms missing from response: {missing[:5]}...")

        all_rows.extend(rows)
        print(f"  chunk {n_chunks}: {len(rows)}/{len(chunk)} terms classified")

    elapsed = time.time() - t_start

    if not all_rows:
        print("ERROR: no rows were classified. Aborting without writing output.")
        raise SystemExit(1)

    raw_df = pd.DataFrame(all_rows)

    # ------------------------------------------------------------
    # Raw output (PDF's "Raw & Curated Tables" requirement)
    # ------------------------------------------------------------
    OUT_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Curated frozen lexicon (dedupe defensively, keep first occurrence)
    # ------------------------------------------------------------
    curated = raw_df.drop_duplicates(subset="token", keep="first").reset_index(drop=True)
    OUT_LEXICON_CSV.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(OUT_LEXICON_CSV, index=False, encoding="utf-8-sig")

    print(f"\nTotal classified: {len(curated):,} / {len(terms):,} requested")
    print("Category counts:")
    cat_counts = curated["category"].value_counts()
    for cat, n in cat_counts.items():
        print(f"  {cat}: {n}")
    print(f"Raw output:     {OUT_RAW_CSV}")
    print(f"Curated lexicon: {OUT_LEXICON_CSV}")

    # ------------------------------------------------------------
    # Audit log (PDF section 4, Scientific Reproducibility & Audit Tracking)
    # ------------------------------------------------------------
    OUT_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AUDIT_LOG, "w", encoding="utf-8") as f:
        f.write("# Zero-Shot LLM Annotation -- Audit Log\n\n")
        f.write(f"- Executed: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write(f"- anthropic package version: `{anthropic.__version__}`\n")
        f.write(f"- Python version: `{sys.version.split()[0]}`\n")
        f.write(f"- output_config.effort: `{EFFORT}` "
                f"(no `thinking` override -- adaptive by default)\n")
        f.write(
            "- **Temperature deviation from the PDF**: the PDF's protocol specifies "
            "\"Temperature fixed at 0.0 (strict determinism)\". Claude Opus 5 no longer "
            "accepts a `temperature` parameter (returns HTTP 400 if sent) -- there is no "
            "equivalent knob for byte-identical determinism on this model generation. "
            "No `temperature` parameter was sent; the settings above (effort=low, adaptive "
            "thinking) are the closest available approximation and should be documented "
            "as a methodological footnote/adjustment in the thesis text.\n"
        )
        f.write(f"- Prompt artifact: `{PROMPT_JSON.relative_to(DICT_DIR.parent)}` (version `{prompt['version']}`)\n")
        f.write(f"- Input vocabulary: `{VOCAB_CSV.relative_to(DICT_DIR.parent)}` "
                f"({len(terms):,} terms requested"
                + (f", limited to first {args.limit} via --limit" if args.limit else "")
                + ")\n")
        f.write(f"- Chunk size: {CHUNK_SIZE} terms/request | Chunks sent: {n_chunks}\n")
        f.write(f"- Wall-clock time: {elapsed:.1f}s\n")
        f.write(f"- Terms classified (post-dedupe): {len(curated):,} / {len(terms):,} requested\n")
        f.write(f"- Category counts:\n")
        for cat, n in cat_counts.items():
            f.write(f"  - {cat}: {n}\n")
        f.write(f"- Raw output: `{OUT_RAW_CSV.relative_to(DICT_DIR.parent)}`\n")
        f.write(f"- Curated lexicon output: `{OUT_LEXICON_CSV.relative_to(DICT_DIR.parent)}`\n")
    print(f"Audit log: {OUT_AUDIT_LOG}")


if __name__ == "__main__":
    main()
