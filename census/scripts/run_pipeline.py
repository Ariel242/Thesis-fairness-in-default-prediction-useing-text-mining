# -*- coding: utf-8 -*-
"""
run_pipeline.py
====================================
Orchestrator for the ZIP3 grouping pipeline. Runs Stages 01-06 in order, exactly as
if each script had been executed one at a time from the command line:

  01_load_and_validate_census_data.py     -> census/results/01_census_data_validated.csv
  02_build_ses_disadvantage_index.py      -> census/results/02_ses_disadvantage_index.csv
  03_build_ses_tertiles.py                -> census/results/03_ses_groups.csv
  04_build_demographic_concentration.py   -> census/results/04_demographic_concentration_groups.csv
  05_balance_diagnostics.py               -> census/results/05_balance_diagnostics_{detail,summary}.csv
  06_join_zip3_labels_to_loans.py         -> census/results/06_loan_level_zip3_group_labels.csv

Stage 07 (robustness_variants.py) is intentionally NOT run here -- it is a set of
optional sensitivity-check helpers meant to be run manually later (see its own
docstring and doc section 11), not part of the primary group-construction pipeline.

Each stage's own script is still fully runnable on its own (`python 03_build_ses_tertiles.py`)
for inspecting/debugging one step in isolation -- this orchestrator just chains all six
by importing each module's main() in sequence, so the whole pipeline can also be run with
a single command:

    python census/scripts/run_pipeline.py

See ../../הסבר והצדקה מתודולוגית לאשכולות.docx for the full methodology this pipeline
implements, and README.md in this folder for a one-page map of stage -> output file.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import importlib.util
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = [
    "01_load_and_validate_census_data",
    "02_build_ses_disadvantage_index",
    "03_build_ses_tertiles",
    "04_build_demographic_concentration",
    "05_balance_diagnostics",
    "06_join_zip3_labels_to_loans",
]


def load_stage_module(stage_name: str):
    """Loads e.g. 03_build_ses_tertiles.py by file path. A plain `import 03_foo` is a
    SyntaxError in Python (module names can't start with a digit), so each stage
    script is loaded directly from its file path instead."""
    spec = importlib.util.spec_from_file_location(stage_name, SCRIPT_DIR / f"{stage_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    for stage_name in STAGES:
        print("\n" + "=" * 70)
        print(f"STAGE: {stage_name}")
        print("=" * 70)
        t0 = time.perf_counter()

        module = load_stage_module(stage_name)
        module.main()

        print(f"({time.perf_counter() - t0:.1f}s)")

    print("\nPipeline complete. All outputs in census/results/.")


if __name__ == "__main__":
    main()
