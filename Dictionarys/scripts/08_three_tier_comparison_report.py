# -*- coding: utf-8 -*-
"""
08_three_tier_comparison_report.py
=====================================
Synthesizes the three independently-run text-representation ablations
(Lexicon, TF-IDF, FinBERT) into a single cross-algorithm comparison, per the
protocol PDF's "3-Tier Representation Framework" (Level 1 Lexicon / Level 2
TF-IDF / Level 3 FinBERT). Each tier was built and evaluated by a separate,
self-contained script with no shared modules (this repo's established
convention) -- this script only READS their already-written result CSVs
(never recomputes anything) and produces one consolidated PDF.

Read-only inputs (outside Dictionarys/, per the repo's no-shared-module
convention -- nothing is written back to these locations):
  Dictionarys/results/lexicon_ablation_summary.csv, lexicon_ablation_delong.csv
  TF-IDF/results/ablation_summary.csv, ablation_delong.csv
  FinBERT/results/{baseline_768,gridsearch_768,baseline_pca50,gridsearch_pca50}/wf_predictive_summary.csv
  FinBERT/results/delong_vs_structured/delong_raw.csv (2026-08-29: added -- the FinBERT
    baseline/gridsearch scripts themselves never ran a DeLong test against Structured;
    FinBERT/DELONG_VS_STRUCTURED.py fills that gap by refitting Structured + both FinBERT
    arms side by side per fold, for both fixed and grid-search configs, reusing each arm's
    already-saved winning hyperparameters -- see that script's own docstring)

Output: Dictionarys/Three_Tier_Algorithm_Comparison.pdf
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DICT_DIR   = SCRIPT_DIR.parent
BASE_DIR   = DICT_DIR.parent

OUT_PDF = DICT_DIR / "Three_Tier_Algorithm_Comparison.pdf"

MODELS = ["Logistic", "XGBoost", "RandomForest"]


def load_wide_summary(path: Path, has_arm_col: bool) -> pd.DataFrame:
    """Reads a mean/std AUC summary CSV written by pandas' MultiIndex
    .to_csv() (2 or 3 header rows depending on whether an Arm level exists)."""
    header_rows = [0, 1, 2] if has_arm_col else [0, 1]
    df = pd.read_csv(path, header=header_rows, index_col=list(range(2 if has_arm_col else 1)))
    return df


def get_auc(df, model, arm=None):
    if arm is not None:
        row = df.loc[(arm, model)]
    else:
        row = df.loc[model]
    return float(row[("AUC", "mean")]), float(row[("AUC", "std")])


def main() -> None:
    # ------------------------------------------------------------
    # Load all sources
    # ------------------------------------------------------------
    lex_summary = load_wide_summary(DICT_DIR / "results" / "lexicon_ablation_summary.csv", has_arm_col=True)
    lex_delong  = pd.read_csv(DICT_DIR / "results" / "lexicon_ablation_delong.csv")

    tfidf_summary = load_wide_summary(BASE_DIR / "TF-IDF" / "results" / "ablation_summary.csv", has_arm_col=True)
    tfidf_delong  = pd.read_csv(BASE_DIR / "TF-IDF" / "results" / "ablation_delong.csv")

    fb_base768 = load_wide_summary(BASE_DIR / "FinBERT" / "results" / "baseline_768" / "wf_predictive_summary.csv", has_arm_col=False)
    fb_grid768 = load_wide_summary(BASE_DIR / "FinBERT" / "results" / "gridsearch_768" / "wf_predictive_summary.csv", has_arm_col=False)
    fb_basepca = load_wide_summary(BASE_DIR / "FinBERT" / "results" / "baseline_pca50" / "wf_predictive_summary.csv", has_arm_col=False)
    fb_gridpca = load_wide_summary(BASE_DIR / "FinBERT" / "results" / "gridsearch_pca50" / "wf_predictive_summary.csv", has_arm_col=False)

    fb_delong_all = pd.read_csv(BASE_DIR / "FinBERT" / "results" / "delong_vs_structured" / "delong_raw.csv")
    fb_delong_fixed = fb_delong_all[fb_delong_all["config"] == "fixed"]  # both Arm_B values (768 / PCA50)
    fb_delong_grid  = fb_delong_all[fb_delong_all["config"] == "grid"]

    # Canonical Structured-only reference: TF-IDF's "Structured" arm (identical
    # script family / numbers to the Lexicon ablation's own Structured arm).
    struct_ref = {m: get_auc(tfidf_summary, m, arm="Structured") for m in MODELS}

    def delong_sig(delong_df, model, arm_b, arm_a="Structured"):
        sub = delong_df[(delong_df["Model"] == model) & (delong_df["Arm_A"] == arm_a) & (delong_df["Arm_B"] == arm_b)]
        if sub.empty:
            return None, None
        return float(sub["Delta_AUC"].mean()), int((sub["p_value"] < 0.05).sum())

    # ------------------------------------------------------------
    # Build the master comparison table
    # ------------------------------------------------------------
    # Each row: (label, config, source, has_delong)
    rows_spec = [
        ("Lexicon (7 features)", "fixed hyperparams", "Dictionarys", lex_summary, "arm", "Structured+Lexicon", lex_delong, "Structured+Lexicon"),
        ("TF-IDF (full, uncapped)", "fixed hyperparams", "TF-IDF", tfidf_summary, "arm", "Structured+TFIDF_full", tfidf_delong, "Structured+TFIDF_full"),
        ("TF-IDF (chi2 top-500)", "fixed hyperparams", "TF-IDF", tfidf_summary, "arm", "Structured+TFIDF_chi2", tfidf_delong, "Structured+TFIDF_chi2"),
        ("FinBERT-768 (raw)", "fixed hyperparams", "FinBERT", fb_base768, "flat", None, fb_delong_fixed, "Structured+FinBERT-768"),
        ("FinBERT-768 (raw)", "grid search", "FinBERT", fb_grid768, "flat", None, fb_delong_grid, "Structured+FinBERT-768"),
        ("FinBERT-PCA50", "fixed hyperparams", "FinBERT", fb_basepca, "flat", None, fb_delong_fixed, "Structured+FinBERT-PCA50"),
        ("FinBERT-PCA50", "grid search", "FinBERT", fb_gridpca, "flat", None, fb_delong_grid, "Structured+FinBERT-PCA50"),
    ]

    table_rows = []
    for label, config, source, df, mode, arm, delong_df, delong_arm in rows_spec:
        for model in MODELS:
            try:
                auc, std = get_auc(df, model, arm=arm if mode == "arm" else None)
            except KeyError:
                auc, std = float("nan"), float("nan")
            struct_auc, _ = struct_ref[model]
            delta = auc - struct_auc
            if delong_df is not None:
                mean_delta, n_sig = delong_sig(delong_df, model, delong_arm)
                sig_str = f"{n_sig}/14" if n_sig is not None else "n/a"
            else:
                sig_str = "not run"
            table_rows.append({
                "Representation": label, "Config": config, "Source": source, "Model": model,
                "AUC": auc, "Std": std, "Delta_vs_Structured": delta, "Significant_folds": sig_str,
            })
    master = pd.DataFrame(table_rows)
    master.to_csv(DICT_DIR / "results" / "three_tier_master_comparison.csv", index=False)

    # ------------------------------------------------------------
    # Figure: grouped bar chart, AUC by representation x model
    # ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = DICT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    reps = [("Structured\n(reference)", None, None)] + [
        (f"{r[0]}\n({r[1]})", r[0], r[1]) for r in rows_spec
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(reps))
    width = 0.25
    colors_m = {"Logistic": "#2166ac", "XGBoost": "#d6604d", "RandomForest": "#1a9850"}
    for i, model in enumerate(MODELS):
        ys = []
        for label, rep_label, config in reps:
            if rep_label is None:
                ys.append(struct_ref[model][0])
            else:
                row = master[(master["Representation"] == rep_label) & (master["Config"] == config) & (master["Model"] == model)]
                ys.append(row["AUC"].values[0] if len(row) else np.nan)
        ax.bar(x + (i - 1) * width, ys, width, label=model, color=colors_m[model])
    ax.axhline(struct_ref["Logistic"][0], color="#2166ac", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in reps], fontsize=8)
    ax.set_ylabel("Mean AUC (walk-forward, 14 folds)")
    ax.set_ylim(0.64, 0.71)
    ax.set_title("AUC by Text Representation and Algorithm (fixed hyperparameters unless noted)", fontsize=12, fontweight="bold")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = figures_dir / "three_tier_comparison_bar.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Build the PDF
    # ------------------------------------------------------------
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=8, leading=14)
    note = ParagraphStyle("note", parent=styles["BodyText"], spaceBefore=4, spaceAfter=10,
                           leftIndent=12, textColor=colors.HexColor("#8a4b00"),
                           borderColor=colors.HexColor("#f0c36d"), borderWidth=0.75,
                           borderPadding=6, backColor=colors.HexColor("#fff7e6"))
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10,
                            textColor=colors.HexColor("#555555"))

    def table_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=LETTER,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Text Representation Comparison Across Three Algorithms", styles["Title"]))
    story.append(Paragraph(
        "LendingClub Default-Prediction Thesis &mdash; Lexicon vs TF-IDF vs FinBERT, "
        "Logistic Regression vs XGBoost vs Random Forest", styles["Heading3"]))
    story.append(Paragraph(
        "Synthesized by <font face='Courier'>Dictionarys/scripts/08_three_tier_comparison_report.py</font> "
        "purely by reading the three tiers' already-written result files (Dictionarys/, TF-IDF/, FinBERT/) "
        "&mdash; no models were refit for this report.", small))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Purpose", h1))
    story.append(Paragraph(
        "Each text-representation tier (Level 1 Lexicon, Level 2 TF-IDF, Level 3 FinBERT) was built and "
        "evaluated independently, by design (this repo's no-shared-module convention: each ablation script "
        "owns its own copy of the walk-forward fold logic, DeLong test, and bootstrap CI). This means no "
        "single existing document shows all three tiers against all three algorithms side by side. This "
        "report closes that gap by consolidating the already-computed results &mdash; it does not run any "
        "new model.", body))

    story.append(Paragraph(
        "<b>Important scope note:</b> no combined arm (e.g. Structured+TF-IDF+FinBERT together) has been "
        "tested anywhere in this repo. Every row below adds exactly one text representation to the same "
        "Structured baseline. Stacking representations together is a distinct, untested question.", note))

    story.append(Paragraph("Structured-only reference (mean AUC, 14 folds)", h2))
    ref_rows = [["Model", "AUC"]] + [[m, f"{struct_ref[m][0]:.4f} +/- {struct_ref[m][1]:.4f}"] for m in MODELS]
    t0 = Table(ref_rows, colWidths=[2.5 * inch, 2.5 * inch])
    t0.setStyle(table_style())
    story.append(t0)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Master comparison table", h2))
    story.append(Paragraph(
        "&Delta;AUC is relative to the Structured-only reference above. \"Significant folds\" is the count "
        "(of 14) where a per-fold DeLong test found the difference significant at p&lt;0.05 (raw, not "
        "multiple-testing corrected). For FinBERT this comes from a dedicated script "
        "(<font face='Courier'>FinBERT/DELONG_VS_STRUCTURED.py</font>, added 2026-08-29) that refits "
        "Structured and both FinBERT arms side by side per fold, since the original baseline/gridsearch "
        "scripts only saved aggregate AUC, not paired per-fold predictions.", body))

    tbl = [["Representation", "Config", "Model", "AUC", "Delta AUC", "Sig. folds"]]
    for _, r in master.iterrows():
        tbl.append([
            r["Representation"], r["Config"], r["Model"],
            f"{r['AUC']:.4f}", f"{r['Delta_vs_Structured']:+.4f}", r["Significant_folds"],
        ])
    t1 = Table(tbl, colWidths=[1.7*inch, 1.05*inch, 0.95*inch, 0.65*inch, 0.65*inch, 0.75*inch])
    t1.setStyle(table_style())
    # highlight negative deltas
    for i, r in enumerate(master.itertuples(), start=1):
        if r.Delta_vs_Structured < 0:
            t1.setStyle(TableStyle([("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#b02a2a"))]))
        else:
            t1.setStyle(TableStyle([("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#1a7a1a"))]))
    story.append(t1)

    story.append(PageBreak())
    story.append(Paragraph("AUC by representation and algorithm", h1))
    story.append(Image(str(fig_path), width=7.0 * inch, height=7.0 * inch * (6/13)))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Cross-cutting findings", h1))
    findings = [
        "<b>RandomForest is structurally penalized by wide/dense text additions.</b> TF-IDF (full), "
        "FinBERT-768, and even FinBERT-PCA50 all reduce its AUC below the Structured-only reference; "
        "the only representation that helps it at all is the small 7-feature Lexicon (+0.0014). This "
        "matches FinBERT/RESULTS_COMPARISON.md &sect;6a's direct finding: not one FinBERT dimension cracks "
        "RandomForest's global top-50 features, while Logistic and XGBoost both draw roughly half their "
        "top 50 from FinBERT &mdash; a model/algorithm mismatch given the fixed `min_samples_leaf`, not a "
        "lack of signal in the text itself.",
        "<b>FinBERT-768 raw is the worst-performing representation tested, for every algorithm</b> "
        "(&Delta;AUC from -0.0064 to -0.0275 with fixed hyperparameters) &mdash; worse than adding nothing. "
        "It only becomes competitive after PCA-50 compression, and grid-search tuning recovers relatively "
        "little more on top of that. This is also the one case in the whole comparison with strong, "
        "consistent DeLong significance: RandomForest+FinBERT-768 loses to Structured-only in 12/14 folds "
        "at fixed hyperparameters (8/14 even after grid-search tuning) &mdash; not a borderline or noisy "
        "result, a reliably worse one.",
        "<b>Under fixed hyperparameters (the fairest apples-to-apples comparison, since this is how the "
        "Lexicon and TF-IDF arms were run), TF-IDF outperforms FinBERT-PCA50 for Logistic and XGBoost</b> "
        "(0.6965 vs 0.6948, 0.6936 vs 0.6887) &mdash; FinBERT only closes part of that gap with grid search, "
        "which TF-IDF was never given the chance to use here.",
        "<b>The Lexicon is the smallest and weakest representation in absolute terms, but the only one that "
        "never hurt any of the three algorithms.</b> Its effect sizes are consistently the smallest "
        "(+0.0009 to +0.0014) and, per Dictionarys/results/lexicon_ablation_report.md, rarely reach "
        "statistical significance (at most 2/14 folds) &mdash; consistent with the moderate Cohen's kappa "
        "(0.556) found during its own human validation.",
        "<b>Across all three tiers, effect sizes are small and mostly non-significant.</b> Even TF-IDF's "
        "best case (Logistic, full vocabulary) is only significant in 7 of 14 folds. The honest reading "
        "across this entire comparison is that unstructured loan-description text adds, at best, a modest "
        "and algorithm-dependent increment over the 126 structured features already in the model &mdash; not "
        "a transformative one.",
    ]
    for f in findings:
        story.append(Paragraph("&bull; " + f, body))

    story.append(Paragraph("Data provenance / known asymmetries", h1))
    story.append(Paragraph(
        "The Structured-only reference AUC used throughout (0.6930 / 0.6904 / 0.6882) is taken from "
        "TF-IDF/results/ablation_summary.csv's own \"Structured\" arm, which is numerically identical to "
        "Dictionarys' own Structured arm (both scripts fit it independently from the same STRUCT_COLS "
        "definition). FinBERT/RESULTS_COMPARISON.md reports a very slightly different Structured-only "
        "RandomForest figure (0.6886 vs 0.6882) from its own separate analysis/BASELINE.py run &mdash; a "
        "negligible (0.0004) cross-run numerical difference, not a methodological one, since all three "
        "reuse the identical STRUCT_COLS definition and walk-forward parameters.", body))
    story.append(Paragraph(
        "The FinBERT vs Structured DeLong test (FinBERT/results/delong_vs_structured/, added 2026-08-29) "
        "refits both arms independently of the original baseline_768/gridsearch_768/baseline_pca50/"
        "gridsearch_pca50 runs, so its own AUC numbers can differ slightly (typically &lt;0.002) from the "
        "AUC column above -- most likely from a scikit-learn version drift between when those scripts were "
        "first run and this later DeLong-only run (a 'penalty' deprecation warning appears in the newer run's "
        "logs that does not appear in the older ones). Only the Significant-folds count is taken from this "
        "test; the AUC/&Delta;AUC columns still come from each arm's own original run.", note))

    doc.build(story)
    print(f"Report written: {OUT_PDF}")
    print(f"Master CSV: {DICT_DIR / 'results' / 'three_tier_master_comparison.csv'}")


if __name__ == "__main__":
    main()
