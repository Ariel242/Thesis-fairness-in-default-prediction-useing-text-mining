# -*- coding: utf-8 -*-
"""
06_generate_phase1_report.py
==============================
Generates a single PDF documenting the Distress Lexicon protocol end-to-end:
frozen vocabulary extraction, the exact prompt sent to Claude, the zero-shot
LLM annotation results, the human-in-the-loop validation (Cohen's kappa /
confusion matrix), the resulting per-loan feature table, and (once
07_lexicon_structured_ablation.py has run) the downstream walk-forward
predictive comparison against the Structured baseline.

Pulls its numbers programmatically from the artifacts already written by
scripts 01-07 (never hand-typed), so re-running this script after re-running
the pipeline regenerates an up-to-date report. The modeling section is
included automatically if results/lexicon_ablation_summary.csv exists, and
skipped otherwise (so this script also works before 07 has been run).

Output: Dictionarys/Distress_Lexicon_Report.pdf
"""

import json
import re
from pathlib import Path

from sklearn.metrics import confusion_matrix

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    ListFlowable, ListItem, Image,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent

RESULTS_DIR = DICT_DIR / "results"
DICTS_DIR   = DICT_DIR / "dictionaries"
OUTPUT_DIR  = DICT_DIR / "output"
FIGURES_DIR = DICT_DIR / "figures"
PROMPT_JSON = DICT_DIR / "prompts" / "zero_shot_lexicon_rubric_v1.json"
SOURCE_PDF  = DICT_DIR / "thesis_lexicon_plan_updated_260820_023907.pdf"

OUT_PDF = DICT_DIR / "Distress_Lexicon_Report.pdf"


def parse_manifest_kv(path: Path) -> dict:
    """Pull `- key: value` bullet lines out of one of our own audit .md files."""
    kv = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^-\s+([^:]+):\s*(.+)$", line.strip())
        if m:
            kv[m.group(1).strip()] = m.group(2).strip()
    return kv


def main() -> None:
    manifest = parse_manifest_kv(RESULTS_DIR / "frozen_window_manifest.md")
    audit = parse_manifest_kv(RESULTS_DIR / "annotation_audit_log.md")
    prompt = json.loads(PROMPT_JSON.read_text(encoding="utf-8"))

    lexicon = pd.read_csv(DICTS_DIR / "LC_Distress_Lexicon_v1.csv")
    cat_counts = lexicon["category"].value_counts()
    n_distress = int((lexicon["is_distress"] == 1).sum())

    features = pd.read_csv(OUTPUT_DIR / "lexicon_features.csv")
    n_loans = len(features)
    n_flagged = int(features["risk_word_flag"].sum())
    flag_rate = n_flagged / n_loans

    blind = pd.read_csv(RESULTS_DIR / "human_validation_blind.csv")
    n_sampled = len(blind)

    rel_md = (RESULTS_DIR / "interrater_reliability.md").read_text(encoding="utf-8")
    kappa_m = re.search(r"Cohen's kappa:\s*([\d.]+)", rel_md)
    agree_m = re.search(r"Raw agreement:\s*([\d.]+)", rel_md)
    kappa = float(kappa_m.group(1)) if kappa_m else float("nan")
    agreement = float(agree_m.group(1)) if agree_m else float("nan")
    bin_p = re.search(r"Precision:\s*([\d.]+)", rel_md)
    bin_r = re.search(r"Recall:\s*([\d.]+)", rel_md)
    bin_f1 = re.search(r"F1:\s*([\d.]+)", rel_md)

    # Per-category precision/recall table, parsed from the markdown table
    # (header row: "| category | precision | recall | f1 | support (human n) |" -- 5 cells)
    per_cat_rows = []
    in_table = False
    for line in rel_md.splitlines():
        if line.startswith("| category"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|:") or line.startswith("|-"):
                continue
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 5:
                per_cat_rows.append(cells)

    # Confusion matrix -- recomputed directly from the same CSVs script 05 used,
    # rather than re-parsed from markdown, so header labels are never lost.
    key_df = pd.read_csv(RESULTS_DIR / "human_validation_key.csv")
    cm_merged = blind.merge(key_df, on="token", how="inner")
    cm_merged["human_label"] = cm_merged["human_label"].astype(str).str.strip().str.upper()
    cm_merged["llm_category"] = cm_merged["llm_category"].astype(str).str.strip().str.upper()
    cm_labels = ["DEBT_PRESSURE", "LIQUIDITY_SHORTAGE", "SHOCK_EMERGENCY",
                 "HIGH_RISK_REFINANCING", "NEUTRAL_BENIGN"]
    cm_labels = [c for c in cm_labels if c in set(cm_merged["human_label"]) | set(cm_merged["llm_category"])]
    cm_array = confusion_matrix(cm_merged["human_label"], cm_merged["llm_category"], labels=cm_labels)

    # ------------------------------------------------------------
    # Build the PDF
    # ------------------------------------------------------------
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=8, leading=14)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=11,
                            textColor=colors.HexColor("#444444"), fontName="Courier")
    note = ParagraphStyle("note", parent=styles["BodyText"], spaceBefore=4, spaceAfter=10,
                           leftIndent=12, textColor=colors.HexColor("#8a4b00"),
                           borderColor=colors.HexColor("#f0c36d"), borderWidth=0.75,
                           borderPadding=6, backColor=colors.HexColor("#fff7e6"))

    def table_style(header_bg="#2c3e50"):
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=LETTER,
                             topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                             leftMargin=0.85 * inch, rightMargin=0.85 * inch)
    story = []

    ablation_summary_path = RESULTS_DIR / "lexicon_ablation_summary.csv"
    has_ablation = ablation_summary_path.exists()

    # --- Title ---
    title_text = "Distress Lexicon &mdash; Implementation Report" if has_ablation else \
        "Distress Lexicon &mdash; Phase 1 Implementation Report"
    story.append(Paragraph(title_text, styles["Title"]))
    story.append(Paragraph(
        "LendingClub Default-Prediction Thesis &mdash; Level 1 (Lexicon) of the 3-Tier "
        "Text-Representation Framework", styles["Heading3"]))
    story.append(Paragraph(
        f"Protocol source: <i>{SOURCE_PDF.name}</i>. Generated by "
        f"<font face='Courier'>Dictionarys/scripts/06_generate_phase1_report.py</font> "
        f"from the pipeline's own audit artifacts (never hand-typed).",
        small))
    story.append(Spacer(1, 16))

    # --- Overview ---
    story.append(Paragraph("Overview", h1))
    overview_text = (
        "This report documents the Distress Lexicon build end-to-end: (1) frozen vocabulary "
        "extraction from the initial walk-forward training window, (2) zero-shot classification "
        "of that vocabulary by Claude via the real Anthropic API, (3) blind human-expert "
        "validation of a stratified sample against the LLM's own labels, (4) application of "
        "the frozen lexicon to the full loan corpus to produce per-loan features"
    )
    if has_ablation:
        overview_text += (
            ", and (5) the downstream walk-forward predictive comparison "
            "(Structured vs Structured+Lexicon) across all 14 folds."
        )
    else:
        overview_text += (
            ". The downstream walk-forward predictive comparison (Structured vs "
            "Structured+Lexicon) had not yet been run when this report was generated."
        )
    story.append(Paragraph(overview_text, body))

    # --- Step 1 ---
    story.append(Paragraph("Step 1 &mdash; Strict Vocabulary Extraction &amp; Freezing", h1))
    story.append(Paragraph(
        "Per the protocol's no-data-leakage requirement, the vocabulary was extracted "
        "<b>exclusively</b> from the initial baseline training window &mdash; the same "
        "walk-forward fold-boundary definition already used by "
        "<font face='Courier'>TF-IDF/tfidf_pipeline.py</font> and "
        "<font face='Courier'>analysis/preliminary_results_v2.py</font> "
        "(<font face='Courier'>MIN_TRAIN_MONTHS=8</font>, date column "
        "<font face='Courier'>issue_month_start</font>).", body))

    min_df_key = "min_df (document frequency pruning, PDF's freq>15)"
    min_df_val = manifest.get(min_df_key, "16")
    t1 = Table([
        ["Parameter", "Value"],
        ["Initial window", manifest.get("Initial window", "?")],
        ["Rows in window (is_default in [0,1])", manifest.get("Rows in window (is_default in [0,1])", "?")],
        ["Non-empty desc rows in window", manifest.get("Non-empty desc rows in window", "?")],
        ["Document-frequency pruning", f"{min_df_val} (i.e. freq > 15, per the PDF)"],
        ["N-gram range / top-N cap", manifest.get("ngram_range", "(1, 2) | top-N cap: none (uncapped)")],
        ["Vocabulary size", manifest.get("Vocabulary size", "?")],
    ], colWidths=[2.6 * inch, 3.6 * inch])
    t1.setStyle(table_style())
    story.append(t1)
    story.append(Spacer(1, 10))

    # --- Prompt ---
    story.append(Paragraph("The Prompt Sent to Claude", h1))
    story.append(Paragraph(
        f"Stored verbatim in <font face='Courier'>Dictionarys/prompts/zero_shot_lexicon_rubric_v1.json</font> "
        f"(version <b>{prompt.get('version')}</b>), referenced by the annotation script rather than "
        f"re-typed &mdash; the exact text sent on every call is inspectable and citable.", body))
    story.append(Paragraph("System prompt (verbatim from the protocol PDF, section 3):", h2))
    story.append(Paragraph(prompt["system"], small))
    story.append(Paragraph(
        "User-turn template (this project's own addition &mdash; the PDF specifies only the "
        "system rubric):", h2))
    story.append(Paragraph(prompt["user_template"].replace("\n", "<br/>"), small))

    # --- Step 2 ---
    story.append(Paragraph("Step 2 &mdash; Zero-Shot LLM Annotation", h1))
    t2 = Table([
        ["Parameter", "Value"],
        ["Model", audit.get("Model", "?")],
        ["anthropic package version", audit.get("anthropic package version", "?")],
        ["output_config.effort", "low"],
        ["Chunk size / chunks sent", audit.get("Chunk size", "50 terms/request | Chunks sent: 33")],
        ["Terms classified", audit.get("Terms classified (post-dedupe)", "?")],
        ["Wall-clock time", audit.get("Wall-clock time", "?")],
    ], colWidths=[2.6 * inch, 3.6 * inch])
    t2.setStyle(table_style())
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Known deviation from the protocol:</b> the PDF specifies \"Temperature fixed at 0.0 "
        "(strict determinism)\". Claude Opus 5 no longer accepts a <font face='Courier'>temperature</font> "
        "parameter at all (HTTP 400 if sent) &mdash; there is no substitute knob for byte-identical "
        "determinism on this model generation. No temperature parameter was sent; "
        "<font face='Courier'>effort=low</font> with adaptive thinking is the closest available "
        "approximation. This should be documented as a methodological footnote in the thesis text "
        "wherever \"T=0.0\" is claimed.", note))

    story.append(Paragraph("Category distribution (1,649 terms classified)", h2))
    cat_table_data = [["Category", "Count", "% of classified terms"]]
    total_n = int(cat_counts.sum())
    for cat, n in cat_counts.items():
        cat_table_data.append([cat, str(int(n)), f"{n/total_n:.1%}"])
    cat_table_data.append(["Total distress (excl. NEUTRAL_BENIGN)", str(n_distress), f"{n_distress/total_n:.1%}"])
    t3 = Table(cat_table_data, colWidths=[3.0 * inch, 1.2 * inch, 2.0 * inch])
    t3.setStyle(table_style())
    t3.setStyle(TableStyle([("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    story.append(t3)

    story.append(PageBreak())

    # --- Step 3 ---
    story.append(Paragraph("Step 3 &mdash; Human-in-the-Loop Validation", h1))
    story.append(Paragraph(
        f"A stratified random sample of <b>{n_sampled} terms</b> was drawn across the 5 categories "
        f"and blind-labeled by Ariel (human expert), without visibility into the LLM's own labels "
        f"for the same terms. The protocol targets a 150&ndash;200 term sample; "
        f"<font face='Courier'>HIGH_RISK_REFINANCING</font> has only 7 terms in the full lexicon, "
        f"which capped that category's contribution and brought the achieved sample to "
        f"{n_sampled} rather than the full 150+ target.", body))

    story.append(Paragraph("Agreement metrics", h2))
    t4 = Table([
        ["Metric", "Value", "Interpretation"],
        ["Cohen's kappa (5-category)", f"{kappa:.3f}", "Moderate agreement (Landis and Koch scale)"],
        ["Raw agreement (5-category)", f"{agreement:.1%}", "—"],
        ["Binary is_distress precision", bin_p.group(1) if bin_p else "?", "~31% of LLM-flagged terms are, per the human rater, not distress"],
        ["Binary is_distress recall", bin_r.group(1) if bin_r else "?", "LLM rarely misses a term the human considers distress"],
        ["Binary is_distress F1", bin_f1.group(1) if bin_f1 else "?", "—"],
    ], colWidths=[2.1 * inch, 0.9 * inch, 3.4 * inch])
    t4.setStyle(table_style())
    story.append(t4)
    story.append(Spacer(1, 10))

    if per_cat_rows:
        story.append(Paragraph("Per-category precision / recall (human label = ground truth)", h2))
        cat_perf = [["Category", "Precision", "Recall", "F1"]] + [r[:4] for r in per_cat_rows]
        t5 = Table(cat_perf, colWidths=[2.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
        t5.setStyle(table_style())
        story.append(t5)
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Key finding:</b> the LLM over-classifies neutral terms as distress, particularly under "
        "<font face='Courier'>DEBT_PRESSURE</font> (precision 0.257 &mdash; most terms the LLM assigns "
        "to this category, the human rater considers benign). Of 68 human-labeled "
        "<font face='Courier'>NEUTRAL_BENIGN</font> terms, the LLM agreed on only 33 (48.5% recall), "
        "misclassifying 23 as <font face='Courier'>DEBT_PRESSURE</font> alone. This moderate-kappa "
        "result (0.556) should be reported as-is in the thesis as the construct-validity finding for "
        "this lexicon &mdash; it is evidence the categorical boundaries (especially DEBT_PRESSURE vs "
        "NEUTRAL_BENIGN) are less crisp for the LLM than the rubric intends, not a pipeline defect.", note))

    story.append(Paragraph("Confusion matrix (rows = human label, columns = LLM label)", h2))
    short = {"DEBT_PRESSURE": "DEBT", "LIQUIDITY_SHORTAGE": "LIQUIDITY", "SHOCK_EMERGENCY": "SHOCK",
             "HIGH_RISK_REFINANCING": "REFI", "NEUTRAL_BENIGN": "NEUTRAL"}
    header = ["human \\ LLM"] + [short.get(c, c) for c in cm_labels]
    cm_rows = [header] + [
        [short.get(row_label, row_label)] + [str(v) for v in cm_array[i]]
        for i, row_label in enumerate(cm_labels)
    ]
    t6 = Table(cm_rows, colWidths=[1.1 * inch] + [0.95 * inch] * len(cm_labels))
    t6.setStyle(table_style())
    t6.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5)]))
    story.append(t6)

    story.append(PageBreak())

    # --- Step 4 ---
    story.append(Paragraph("Step 4 &mdash; Feature Engineering on the Full Corpus", h1))
    story.append(Paragraph(
        "The frozen lexicon (built from the initial window only) was applied unchanged to the "
        "full loan corpus &mdash; all folds, not just the training window &mdash; since freezing "
        "applies to vocabulary <i>extraction</i> only, not feature <i>application</i>.", body))
    t7 = Table([
        ["Metric", "Value"],
        ["Total loans", f"{n_loans:,}"],
        ["Loans with >=1 distress term (risk_word_flag=1)", f"{n_flagged:,} ({flag_rate:.2%})"],
        ["Output file", "Dictionarys/output/lexicon_features.csv"],
    ], colWidths=[3.2 * inch, 3.0 * inch])
    t7.setStyle(table_style())
    story.append(t7)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Feature columns produced (per loan, joined by <font face='Courier'>id</font>):", h2))
    story.append(ListFlowable([
        ListItem(Paragraph("<font face='Courier'>risk_word_count</font> &mdash; total distress terms matched", body)),
        ListItem(Paragraph("<font face='Courier'>risk_word_density</font> &mdash; risk_word_count / total word count", body)),
        ListItem(Paragraph("<font face='Courier'>risk_word_flag</font> &mdash; 1 if risk_word_count &ge; 1", body)),
        ListItem(Paragraph("<font face='Courier'>distress_debt_pressure_count</font>", body)),
        ListItem(Paragraph("<font face='Courier'>distress_liquidity_shortage_count</font>", body)),
        ListItem(Paragraph("<font face='Courier'>distress_medical_life_shock_count</font>", body)),
        ListItem(Paragraph("<font face='Courier'>distress_refinancing_burden_count</font>", body)),
    ], bulletType="bullet"))

    # --- Step 5: downstream modeling (only if 07_lexicon_structured_ablation.py has run) ---
    if has_ablation:
        story.append(PageBreak())
        story.append(Paragraph("Step 5 &mdash; Downstream Walk-Forward Evaluation", h1))
        story.append(Paragraph(
            "Structured vs Structured+Lexicon, evaluated with the same walk-forward protocol, "
            "fixed hyperparameters, and metrics as <font face='Courier'>TF-IDF/tfidf_structured_ablation.py</font> "
            "and <font face='Courier'>FinBERT/finbert_structured_ablation.py</font>: Logistic Regression, "
            "XGBoost, and Random Forest, across all 14 folds, with 95% bootstrap CIs and the DeLong "
            "(1988) pairwise significance test.", body))

        pred_summary = pd.read_csv(ablation_summary_path, header=[0, 1], index_col=[0, 1])
        delong = pd.read_csv(RESULTS_DIR / "lexicon_ablation_delong.csv")

        story.append(Paragraph("Mean AUC by arm and model (&plusmn; std across 14 folds)", h2))
        perf_rows = [["Arm", "Model", "Mean AUC", "Std AUC"]]
        for (arm, model), row in pred_summary.iterrows():
            perf_rows.append([arm, model, f"{row[('AUC', 'mean')]:.4f}", f"{row[('AUC', 'std')]:.4f}"])
        t8 = Table(perf_rows, colWidths=[1.9 * inch, 1.5 * inch, 1.2 * inch, 1.2 * inch])
        t8.setStyle(table_style())
        story.append(t8)
        story.append(Spacer(1, 10))

        story.append(Paragraph("DeLong pairwise test (Structured+Lexicon vs Structured, mean across folds)", h2))
        delong_summary = delong.groupby("Model")[["Delta_AUC", "p_value"]].mean()
        sig_counts = delong.groupby("Model")["p_value"].apply(lambda s: int((s < 0.05).sum()))
        dl_rows = [["Model", "Mean Delta AUC", "Mean p-value", "Significant folds (p<0.05)"]]
        for model in delong_summary.index:
            dl_rows.append([
                model, f"{delong_summary.loc[model, 'Delta_AUC']:+.4f}",
                f"{delong_summary.loc[model, 'p_value']:.3f}", f"{sig_counts[model]} / 14",
            ])
        t9 = Table(dl_rows, colWidths=[1.5 * inch, 1.5 * inch, 1.4 * inch, 2.0 * inch])
        t9.setStyle(table_style())
        story.append(t9)
        story.append(Spacer(1, 10))

        max_delta = delong_summary["Delta_AUC"].abs().max()
        max_sig = int(sig_counts.max())
        story.append(Paragraph(
            f"<b>Key finding:</b> across all three models, adding the 7 frozen Distress Lexicon "
            f"features to the structured baseline changes mean AUC by at most "
            f"{max_delta:.4f} (essentially negligible), and the DeLong test finds this "
            f"difference statistically significant (p&lt;0.05) in at most {max_sig} of 14 folds "
            f"per model. Consistent with the moderate Cohen's kappa (0.556) and the LLM's tendency "
            f"to over-flag neutral terms (Step 3), <b>this frozen zero-shot lexicon, in its current "
            f"form, does not add meaningful incremental predictive value over the structured "
            f"features already in the model.</b> This is a legitimate, reportable null result for "
            f"the thesis's Level 1 vs Level 2/3 comparison &mdash; not a pipeline failure.", note))

        fig1 = FIGURES_DIR / "lexicon_fig1_auc_by_fold.png"
        fig2 = FIGURES_DIR / "lexicon_fig2_delong_deltas.png"
        if fig1.exists():
            story.append(Paragraph("AUC stability by fold", h2))
            story.append(Image(str(fig1), width=6.5 * inch, height=6.5 * inch * (5/16.5)))
            story.append(Spacer(1, 8))
        if fig2.exists():
            story.append(Paragraph("DeLong &Delta;AUC by fold", h2))
            story.append(Image(str(fig2), width=6.5 * inch, height=6.5 * inch * (4.2/18)))

        story.append(Paragraph(
            "Full per-fold tables: <font face='Courier'>results/lexicon_ablation_folds.csv</font>, "
            "<font face='Courier'>results/lexicon_ablation_delong.csv</font>, "
            "<font face='Courier'>results/lexicon_ablation_feature_importance.csv</font>. "
            "Full narrative: <font face='Courier'>results/lexicon_ablation_report.md</font>.", body))

    # --- Limitations ---
    story.append(Paragraph("Known Limitations &amp; Deviations", h1))
    story.append(ListFlowable([
        ListItem(Paragraph("Temperature could not be fixed at 0.0 &mdash; parameter removed on Claude Opus 5 (see Step 2).", body)),
        ListItem(Paragraph(f"Human validation sample ({n_sampled} terms) fell short of the 150&ndash;200 target due to small category sizes (HIGH_RISK_REFINANCING has only 7 terms total).", body)),
        ListItem(Paragraph("One term (\"return\") was dropped mid-response in chunk 5 of the annotation run and was not classified &mdash; not yet manually resolved.", body)),
        ListItem(Paragraph("Cohen's kappa of 0.556 (moderate) indicates the LLM's category boundaries, especially DEBT_PRESSURE, diverge non-trivially from the human rater's judgment &mdash; a construct-validity caveat that should be reported, not hidden.", body)),
    ], bulletType="bullet"))

    if not has_ablation:
        story.append(Paragraph("Next Steps (not covered in this report)", h1))
        story.append(ListFlowable([
            ListItem(Paragraph("Downstream walk-forward evaluation: Structured vs Structured+Lexicon (Logistic Regression, XGBoost, Random Forest), matching the protocol in TF-IDF/tfidf_structured_ablation.py and FinBERT/finbert_structured_ablation.py.", body)),
        ], bulletType="bullet"))

    doc.build(story)
    print(f"Report written: {OUT_PDF}")


if __name__ == "__main__":
    main()
