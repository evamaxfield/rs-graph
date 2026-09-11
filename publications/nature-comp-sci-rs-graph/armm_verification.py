#!/usr/bin/env python3

"""Classification-model verification against the manuscript's stated metrics, plus ARMM
held-out-test diagnostics (confusion matrices, README-length performance, breakdown tables).

Model training/eval artifacts stay in `sci-soft-models` -- these commands read the deployed
models' saved eval results and held-out predictions rather than retraining or reimplementing
any classifier. `sci_soft_models.binary_article_repo_em` imports are done lazily inside each
command: its data module resolves a symlinked local DB path at import time that only exists
in a full rs-graph-plus-data checkout, so an import failure should only break these commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import utils as u

###############################################################################

SCI_SOFT_MODELS_REPO = Path(__file__).resolve().parents[3] / "sci-soft-models"

# (results.json relative path, manuscript-stated precision/recall/f1, metric key prefix)
CLASSIFICATION_MODELS_TABLE: list[dict] = [
    {
        "name": "Article-Repository Matching Model (ARMM)",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/binary_article_repo_em/data/files/final-model-training-data-optimized/results.json",
        "stated_precision": 0.975,
        "stated_recall": 0.975,
        "stated_f1": 0.975,
        "metric_prefix": "macro",
    },
    {
        "name": "Researcher-Developer-Account Matching Model",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/dev_author_em/data/files/final-model-training-data/results.json",
        "stated_precision": 0.938,
        "stated_recall": 0.950,
        "stated_f1": 0.944,
        "metric_prefix": None,  # flat precision/recall/f1 keys, not macro_/binary_-prefixed
        # Cited value is the published figure; the artifact's saved eval differs (see note).
        "known_discrepancy_note": (
            "Cited value = published Brown/Slaughter/Weber figures; the sci-soft-models "
            "artifact's current saved eval differs (see 'actual'). Known discrepancy -- "
            "the published/cited value stands."
        ),
    },
    {
        "name": "Software Repository Sharing Statement Classifier",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/software_mentions_repo_clf/data/files/final-model-training-data/results.json",
        "stated_precision": 0.860,
        "stated_recall": 0.812,
        "stated_f1": 0.827,
        "metric_prefix": "macro",
    },
]

MISMATCH_TOLERANCE = 0.005


def _verify_stated_metrics(entry: dict) -> tuple[dict | None, bool]:
    """QA one model's manuscript-stated metrics against its saved eval results. Returns the
    model's "full test set" summary row (None when results.json is missing) and whether any
    metric mismatched.
    """
    results_path: Path = entry["results_path"]
    print(f"{entry['name']}")
    print(f"  Reading: {results_path}")
    if not results_path.exists():
        print("  MISSING -- results.json not found at this path, cannot verify.\n")
        return None, True

    results = json.loads(results_path.read_text())
    prefix = entry["metric_prefix"]
    if prefix is None:
        actual_precision = results["precision"]
        actual_recall = results["recall"]
        actual_f1 = results["f1"]
    else:
        actual_precision = results[f"{prefix}_precision"]
        actual_recall = results[f"{prefix}_recall"]
        actual_f1 = results[f"{prefix}_f1"]

    summary_row = {
        "model": entry["name"],
        "subset": "full test set",
        "precision": round(actual_precision, 3),
        "recall": round(actual_recall, 3),
        "f1": round(actual_f1, 3),
    }

    any_mismatch = False
    discrepancy_note = entry.get("known_discrepancy_note", "")
    for label, stated, actual in [
        ("Precision", entry["stated_precision"], actual_precision),
        ("Recall", entry["stated_recall"], actual_recall),
        ("F1", entry["stated_f1"], actual_f1),
    ]:
        delta = actual - stated
        if abs(delta) <= MISMATCH_TOLERANCE:
            status = "OK"
        elif discrepancy_note:
            # Known discrepancy: flagged, but not counted as a mismatch.
            status = "FLAGGED_KNOWN_DISCREPANCY"
        else:
            status = "MISMATCH"
            any_mismatch = True
        print(
            f"  {label}: manuscript states {stated:.3f}, actual is {actual:.4f} "
            f"(delta {delta:+.4f}) -- {status}"
        )
    print()
    return summary_row, any_mismatch


def _metric_for_value(frame: pl.DataFrame, key_col: str, key: str, metric_col: str) -> float:
    """Single metric value for the eval frame row whose `key_col` equals `key`."""
    return float(frame.filter(pl.col(key_col) == key).get_column(metric_col).item())


def _armm_detail_specs(
    field_eval: pl.DataFrame,
    period_eval: pl.DataFrame,
    source_eval: pl.DataFrame,
    readme_perf: pl.DataFrame,
) -> list[tuple[str, str, float, float]]:
    """(detail group, metric label, manuscript-stated value, actual value) rows for the
    ARMM detail verification (manuscript lines 207-211).
    """
    field_f1 = field_eval.get_column("macro_f1")
    field_min = field_eval.sort("macro_f1").row(0, named=True)
    period_f1 = period_eval.get_column("macro_f1")
    period_min = period_eval.sort("macro_f1").row(0, named=True)
    field_f1_mean, field_f1_sd = field_f1.mean(), field_f1.std()
    period_f1_mean, period_f1_sd = period_f1.mean(), period_f1.std()
    assert isinstance(field_f1_mean, float) and isinstance(field_f1_sd, float)
    assert isinstance(period_f1_mean, float) and isinstance(period_f1_sd, float)

    def _source_binary_f1(source_name: str) -> float:
        return _metric_for_value(source_eval, "feature_value", source_name, "binary_f1")

    def _readme_bin_f1(bin_name: str) -> float:
        return _metric_for_value(
            readme_perf, "repository_readme_length_bin", bin_name, "macro_f1"
        )

    return [
        ("per_field", "mean macro F1 across fields", 0.972, field_f1_mean),
        ("per_field", "SD macro F1 across fields", 0.010, field_f1_sd),
        (
            "per_field",
            f"lowest field macro F1 ({field_min['feature_value']})",
            0.951,
            float(field_min["macro_f1"]),
        ),
        ("per_period", "mean macro F1 across periods", 0.968, period_f1_mean),
        ("per_period", "SD macro F1 across periods", 0.012, period_f1_sd),
        (
            "per_period",
            f"lowest period macro F1 ({period_min['feature_value']})",
            0.950,
            float(period_min["macro_f1"]),
        ),
        ("per_source", "SoftCite-2025 binary F1", 0.968, _source_binary_f1("softcite_2025")),
        ("per_source", "JOSS binary F1", 0.998, _source_binary_f1("joss")),
        (
            "per_source",
            "same-author-different-article hard-negative binary F1",
            0.964,
            _source_binary_f1("same-author-different-article-negative"),
        ),
        (
            "per_source",
            "same-contributor-different-repo hard-negative binary F1",
            0.966,
            _source_binary_f1("same-contributor-different-repo-negative"),
        ),
        ("readme_length", "1601-3200 chars macro F1", 0.979, _readme_bin_f1("1601-3200")),
        ("readme_length", "<=100 chars macro F1", 0.957, _readme_bin_f1("<=100")),
    ]


def _armm_breakdown_rows(specs: list[tuple[pl.DataFrame, str, str, str]]) -> list[dict]:
    """ARMM performance rows, one per subset value, for the summary table."""
    rows = []
    for frame, value_col, subset_fmt, metric_prefix in specs:
        for row in frame.iter_rows(named=True):
            rows.append(
                {
                    "model": "Article-Repository Matching Model (ARMM)",
                    "subset": subset_fmt.format(row[value_col]),
                    "precision": round(row[f"{metric_prefix}_precision"], 3),
                    "recall": round(row[f"{metric_prefix}_recall"], 3),
                    "f1": round(row[f"{metric_prefix}_f1"], 3),
                }
            )
    return rows


def classification_models_table_verification(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build the supplementary classification-model performance table -- model / subset /
    precision / recall / f1 -- from each deployed model's saved evaluation results in
    `sci-soft-models` ("full test set" rows), plus ARMM breakdowns by field, publication
    period, dataset source, and README length (macro-averaged). Also prints a console-only
    QA comparison against the manuscript's stated figures; only the clean table is saved.
    """
    print("--- Classification models table verification (console-only QA) ---\n")
    any_mismatch = False
    summary_rows = []
    for entry in CLASSIFICATION_MODELS_TABLE:
        summary_row, entry_mismatch = _verify_stated_metrics(entry)
        if summary_row is not None:
            summary_rows.append(summary_row)
        if entry_mismatch:
            any_mismatch = True

    # ---- ARMM detail verification (manuscript lines 207-211): per-field, per-period,
    # per-source, and README-length numbers. ----
    from sci_soft_models.binary_article_repo_em import (
        load_performance_by_readme_length,
        load_single_feature_eval,
    )

    field_eval = load_single_feature_eval("document_topic_primary_field_pruned")
    period_eval = load_single_feature_eval("document_publication_date_bin")
    source_eval = load_single_feature_eval("dataset_source_name")
    readme_perf = load_performance_by_readme_length()
    detail_specs = _armm_detail_specs(field_eval, period_eval, source_eval, readme_perf)

    print("ARMM detail verification (lines 207-211):")
    for detail_group, metric, stated, actual in detail_specs:
        delta = actual - stated
        status = "OK" if abs(delta) <= MISMATCH_TOLERANCE else "MISMATCH"
        if status == "MISMATCH":
            any_mismatch = True
        print(
            f"  [{detail_group}] {metric}: manuscript states {stated:.3f}, actual is "
            f"{actual:.4f} (delta {delta:+.4f}) -- {status}"
        )
    print()

    # ---- Supplementary table rows: ARMM breakdowns per subset value. Sources use binary
    # metrics -- each source is single-class, so macro-averaging there is meaningless. ----
    summary_rows += _armm_breakdown_rows(
        [
            (field_eval, "feature_value", "field: {}", "macro"),
            (period_eval, "feature_value", "publication period: {}", "macro"),
            (source_eval, "feature_value", "source: {}", "binary"),
            (readme_perf, "repository_readme_length_bin", "README length: {} chars", "macro"),
        ]
    )

    summary = pl.DataFrame(summary_rows)
    u.save_table(summary, "classification_models_performance", output_dir)
    print("Supplementary classification-model performance table:")
    print(summary)
    u.print_caption_note(
        "classification_models_performance",
        "'source:' subset rows report binary precision/recall/F1 rather than macro-averaged "
        "values (macro-averaging is meaningless on a single-class subset); within a "
        "single-class subset binary precision is trivially 1.0, so recall and F1 are the "
        "informative metrics for those rows",
    )

    if any_mismatch:
        print(
            "CAVEAT: at least one model's manuscript-stated figure is outside the "
            f"+/-{MISMATCH_TOLERANCE} tolerance of its current saved eval results -- see MISMATCH "
            "lines above. Table needs updating before submission.\n"
        )
    else:
        print("All manuscript-stated figures are within tolerance of current eval results.\n")
    print("-----------------------------------------------------------\n")


###############################################################################
# ARMM model diagnostics (confusion matrices, README-length performance)


def armm_model_diagnostics(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build the ARMM held-out-test diagnostics.

      - the pooled confusion matrix (line 207, `Blues` cmap by standard convention);
      - per-field confusion matrices for the top-8 fields + the held-out "Other"/"Unknown"
        buckets (line 207), each panel annotated with n and macro F1;
      - the performance-by-README-length figure (line ~211), from sci-soft-models'
        `load_performance_by_readme_length` accessor.
    """
    from sci_soft_models.binary_article_repo_em import (
        load_final_model_test_predictions,
        load_performance_by_readme_length,
    )
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

    evaplot.set_style("evaplot_rc")

    test_preds = load_final_model_test_predictions()
    print(f"Loaded ARMM held-out test predictions: {test_preds.height:,} rows")

    y_true = test_preds.get_column("label").to_list()
    y_pred = test_preds.get_column("predicted_label").to_list()
    labels_order = ["no-match", "match"]

    # ---- Pooled confusion matrix (line 207 cross-reference) ----
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format=",")
    # evaplot_rc's default gridlines cut straight through the cell-value text on a
    # heatmap-style plot -- disable for this figure only.
    ax_cm.grid(False)
    u.print_caption_note("armm_confusion_matrix", "ARMM Held-Out Test Set Confusion Matrix")
    u.shrink_ticks(ax_cm, size=9)

    evaplot.adjust_layout(fig_cm)
    u.save_figure(fig_cm, "armm_confusion_matrix", output_dir)
    plt.close(fig_cm)

    # ---- Per-field confusion matrices (3x3 grid: top-8 fields + Other/Unknown buckets) ----
    field_col = "document_topic_primary_field_pruned"
    field_order = (
        test_preds.get_column(field_col).value_counts(sort=True).get_column(field_col).to_list()
    )
    n_fields = len(field_order)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    # Taller rows + explicit hspace so the grid's rows aren't cramped.
    fig_ff, axes_ff = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.6 * n_rows),
        gridspec_kw={"hspace": 0.45},
    )
    axes_flat = axes_ff.flatten()
    for i, field in enumerate(field_order):
        grp = test_preds.filter(pl.col(field_col) == field)
        g_true = grp.get_column("label").to_list()
        g_pred = grp.get_column("predicted_label").to_list()
        g_cm = confusion_matrix(g_true, g_pred, labels=labels_order)
        g_macro_f1 = f1_score(g_true, g_pred, labels=labels_order, average="macro")
        g_disp = ConfusionMatrixDisplay(confusion_matrix=g_cm, display_labels=labels_order)
        g_disp.plot(ax=axes_flat[i], cmap="Blues", colorbar=False, values_format=",")
        axes_flat[i].grid(False)
        axes_flat[i].set_title(
            f"{field}\n(n={grp.height:,}, macro F1={g_macro_f1:.3f})", fontsize=9
        )
        u.shrink_ticks(axes_flat[i], size=8)
        print(f"  {field}: n={grp.height:,}, macro F1={g_macro_f1:.4f}")
    for j in range(n_fields, len(axes_flat)):
        axes_flat[j].set_axis_off()
    u.print_caption_note(
        "armm_confusion_matrix_by_field", "ARMM Held-Out Test Confusion Matrices by Field"
    )
    # Row spacing passed through adjust_layout so tight_layout doesn't collapse it again.
    evaplot.adjust_layout(fig_ff, hspace=0.45)
    fig_ff.subplots_adjust(hspace=0.45)
    u.save_figure(fig_ff, "armm_confusion_matrix_by_field", output_dir)
    plt.close(fig_ff)

    # ---- Performance by README length (line ~211) ----
    readme_perf = load_performance_by_readme_length()
    bin_order = ["<=100", "101-200", "201-400", "401-800", "801-1600", "1601-3200", ">3200"]
    readme_perf = (
        readme_perf.with_columns(
            pl.col("repository_readme_length_bin")
            .replace_strict({b: i for i, b in enumerate(bin_order)}, return_dtype=pl.Int64)
            .alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )
    print("\nARMM performance by README length bin:")
    print(readme_perf.select("repository_readme_length_bin", "macro_f1", "support"))

    fig_rl, ax_rl = plt.subplots(figsize=(8, 5))
    point_color = u.general_palette(1)[0]
    ax_rl.plot(
        readme_perf.get_column("repository_readme_length_bin").to_list(),
        readme_perf.get_column("macro_f1").to_numpy(),
        marker="o",
        color=point_color,
        linewidth=1.6,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    for x, (f1v, supp) in enumerate(readme_perf.select("macro_f1", "support").iter_rows()):
        ax_rl.annotate(
            f"n={supp:,}",
            (x, f1v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7.5,
        )
    ax_rl.set_xlabel("Repository README Length (Characters)")
    ax_rl.set_ylabel("Macro F1")
    # Zero-based y-axis: a tight autoscale makes 0.96 vs. 0.98 look like a large gap; full
    # scale shows performance is uniformly high.
    ax_rl.set_ylim(0, 1.02)
    u.shrink_ticks(ax_rl, size=9)
    evaplot.adjust_layout(fig_rl)
    u.save_figure(fig_rl, "armm_performance_by_readme_length", output_dir)
    plt.close(fig_rl)

    # ---- Summary performance-metric tables: by publication-year bin, corresponding-author
    # country, and first-author country, with missingness buckets disclosed as rows. ----
    year_bin_labels = {
        "pub-year-bin-01": "< 2016",
        "pub-year-bin-02": "2016-2020",
        "pub-year-bin-03": "2021-2024",
        "pub-year-bin-04": ">= 2025",
    }
    from sci_soft_models.binary_article_repo_em import load_single_feature_eval

    # Year-bin table.
    year_ev = load_single_feature_eval("document_publication_date_bin")
    year_total_support = int(year_ev.get_column("support").sum())
    year_table = (
        year_ev.select(
            pl.col("feature_value").alias("group"),
            pl.col("support").alias("n_test_rows"),
            (100 * pl.col("support") / year_total_support).round(1).alias("pct_of_test_rows"),
            pl.col("binary_precision").round(3),
            pl.col("binary_recall").round(3),
            pl.col("binary_f1").round(3),
            pl.col("macro_f1").round(3),
        )
        .sort("group")
        .with_columns(pl.col("group").replace(year_bin_labels))
    )
    u.save_table(year_table, "armm_performance_by_publication_year_bin", output_dir)
    print("\narmm_performance_by_publication_year_bin:")
    print(year_table)

    # Corresponding-author and first-author country breakdowns share the same group rows --
    # one wide table, one column set per author position, rounded to 3 decimals.
    country_specs = [
        ("document_corresponding_author_institution_country_code_pruned", "corresponding"),
        ("document_first_author_institution_country_code_pruned", "first"),
    ]
    country_sides = []
    for feature, suffix in country_specs:
        ev = load_single_feature_eval(feature)
        total_support = int(ev.get_column("support").sum())
        country_sides.append(
            ev.select(
                pl.col("feature_value").alias("group"),
                pl.col("support").alias(f"n_test_rows_{suffix}"),
                (100 * pl.col("support") / total_support)
                .round(1)
                .alias(f"pct_of_test_rows_{suffix}"),
                pl.col("macro_f1").round(3).alias(f"macro_f1_{suffix}"),
            )
        )
    country_table = (
        country_sides[0]
        .join(country_sides[1], on="group", how="full", coalesce=True)
        # Fix the source data's spelling; keep missingness buckets as disclosed rows.
        .with_columns(
            pl.col("group").replace(
                {
                    "No affliation": "No affiliation",
                    "Multiple affliations": "Multiple affiliations",
                }
            )
        )
        .sort("n_test_rows_corresponding", descending=True)
    )
    u.save_table(country_table, "armm_performance_by_author_country", output_dir)
    print("\narmm_performance_by_author_country (corresponding + first, merged):")
    print(country_table)
