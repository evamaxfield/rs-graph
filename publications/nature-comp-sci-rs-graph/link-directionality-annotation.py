#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
import utils as u
from scipy.stats import binomtest

################################################################################

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATORS = ["eva", "sarah", "anna"]
# Each annotated column and the full category set it is rated on (Gwet's AC1 uses the set size)
ANNOTATION_CATEGORIES = {
    "label": ["match", "no-match", "unsure", "not-available"],
    "full-text-available": ["yes", "no"],
    "this-repo-linked-in-article": ["yes", "no"],
    "this-article-linked-in-repo": ["yes", "no"],
}
ANNOTATED_FILENAME_TEMPLATE = "link-directionality-annotated-{annotator}.csv"
OPEN_ACCESS_CATEGORIES = ["0", "1"]
AGREED_FILENAME_PATH = DATA_DIR / "link-directionality-training-agreed-labels.csv"
FULL_TO_ANNOTATE_FILENAME_TEMPLATE = "link-directionality-to-annotate-{annotator}.csv"
TRAINING_ANNOTATION_FILENAME_PATH = DATA_DIR / "link-directionality-training-annotation-set.csv"
TRAINING_ANNOTATION_FILENAME_TEMPLATE = "link-directionality-training-annotated-{annotator}.csv"
TRAINING_ANNOTATION_FILENAMES = [
    DATA_DIR / TRAINING_ANNOTATION_FILENAME_TEMPLATE.format(annotator=annotator)
    for annotator in ANNOTATORS
]

app = typer.Typer()

################################################################################


def _tagged_path(path: Path, output_tag: str | None) -> Path:
    """Append `-{output_tag}` to a filename stem, or return the path unchanged."""
    if output_tag is None:
        return path
    return path.with_name(f"{path.stem}-{output_tag}{path.suffix}")


@app.command()
def create_annotation_set(output_tag: str | None = None) -> None:
    """
    Sample pairs per field and write the agreement + per-annotator annotation CSVs.

    `output_tag` (e.g. a draw date) is appended to every output filename so an earlier
    sample is never overwritten. Also reports the sample's mining-round mix and each
    stratum's share of all mined pairs (the weights for a stratum-weighted precision).
    """
    # Load dataset with top 5 fields (5 + Other)
    df = u.load_filtered_pairs(top_n_fields=5)

    # Filter to only rows with a predictive model confidence that isn't null
    df = df.filter(pl.col("predictive_model_confidence").is_not_null())
    # Fixed row order so the seeded draws below are reproducible run to run.
    df = df.sort("document_repository_link_id")

    # Iter top fields
    top_fields = df.get_column("document_field_name_pruned").unique().sort().to_list()

    # We annotate 600 overall
    # 24 from training agreement set, 576 from all independent annotations
    agreement_subsets = []
    independent_subsets = []
    for field in top_fields:
        field_subset = df.filter(pl.col("document_field_name_pruned") == field)
        agreement_subset = field_subset.sample(
            n=4,
            seed=1,
        )
        independent_subset = field_subset.sample(
            n=96,
            seed=2,
        )
        agreement_subsets.append(agreement_subset)
        independent_subsets.append(independent_subset)

    # Prep for annotation with select column
    select_statements = [
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
        pl.col("document_field_name_pruned"),
        (pl.lit("https://doi.org/") + pl.col("document_doi")).alias("document_url"),
        pl.col("document_is_open_access"),
        (
            pl.lit("https://github.com/")
            + pl.col("repository_owner")
            + pl.lit("/")
            + pl.col("repository_name")
        ).alias("repository_url"),
        pl.lit(None).alias("label"),
        pl.lit(None).alias("full-text-available"),
        pl.lit(None).alias("this-repo-linked-in-article"),
        pl.lit(None).alias("this-article-linked-in-repo"),
        pl.lit(None).alias("notes"),
    ]

    # Combine subsets to single dataframes
    # Shuffle the rows of both
    agreement_df = (
        pl.concat(agreement_subsets)
        .select(*select_statements)
        .sample(fraction=1.0, seed=3, shuffle=True)
    )
    independent_df = (
        pl.concat(independent_subsets)
        .select(*select_statements)
        .sample(fraction=1.0, seed=4, shuffle=True)
    )

    # The open-access column comes from the document table via the base join, so no extra join
    assert len(agreement_df) == 24
    assert len(independent_df) == 576
    assert agreement_df.get_column("document_is_open_access").null_count() == 0
    assert independent_df.get_column("document_is_open_access").null_count() == 0
    assert (
        pl.concat([agreement_df, independent_df])
        .get_column("document_repository_link_id")
        .n_unique()
        == 600
    )

    # Save training set to be annotated by all annotators for agreement analysis
    agreement_df.write_csv(_tagged_path(TRAINING_ANNOTATION_FILENAME_PATH, output_tag))

    # Get the number of rows each annotator should annotate from the independent set
    num_annotators = len(ANNOTATORS)
    num_rows_per_annotator = len(independent_df) // num_annotators
    for i, annotator in enumerate(ANNOTATORS):
        start_idx = i * num_rows_per_annotator
        end_idx = (i + 1) * num_rows_per_annotator
        annotator_subset = independent_df[start_idx:end_idx]
        assert len(annotator_subset) == 192
        print(
            f"Annotator {annotator} assigned {len(annotator_subset)} rows for independent annotation."
        )

        # Save output CSV for this annotator
        output_path = DATA_DIR / FULL_TO_ANNOTATE_FILENAME_TEMPLATE.format(annotator=annotator)
        annotator_subset.write_csv(_tagged_path(output_path, output_tag))

    # Report the round mix of the drawn sample and each stratum's mined-pair share
    sampled = pl.concat(
        [
            pl.concat(agreement_subsets).with_columns(pl.lit("agreement").alias("subset")),
            pl.concat(independent_subsets).with_columns(pl.lit("independent").alias("subset")),
        ]
    )
    n_overlap = sampled.filter(pl.col("document_repository_link_id").is_duplicated()).height
    print(f"Pairs drawn into both the agreement and independent subsets: {n_overlap // 2}")
    round_mix = (
        sampled.group_by("subset", "link_processing_iteration")
        .agg(pl.len().alias("n_pairs"))
        .with_columns(
            (100 * pl.col("n_pairs") / pl.col("n_pairs").sum().over("subset")).alias(
                "pct_of_subset"
            )
        )
        .sort("subset", "link_processing_iteration")
    )
    print(round_mix)
    u.save_table(
        round_mix,
        _tagged_path(Path("link_directionality_sample_round_mix"), output_tag).name,
        u.OUTPUT_DIR,
    )
    stratum_weights = (
        df.group_by("document_field_name_pruned")
        .agg(pl.len().alias("n_mined_pairs"))
        .with_columns(
            (pl.col("n_mined_pairs") / pl.col("n_mined_pairs").sum()).alias("stratum_weight")
        )
        .sort("n_mined_pairs", descending=True)
    )
    print(stratum_weights)
    u.save_table(
        stratum_weights,
        _tagged_path(Path("link_directionality_stratum_weights"), output_tag).name,
        u.OUTPUT_DIR,
    )


def _agreement_stats(rows: list[list[str]], categories: list[str]) -> tuple[float, float]:
    """Percent agreement and Gwet's AC1 for rows that every rater labeled."""
    n_raters = len(rows[0])
    pairs_per_row = n_raters * (n_raters - 1)
    # Share of agreeing rater pairs per row, averaged over rows
    p_a = (
        sum(sum(row.count(c) * (row.count(c) - 1) for c in categories) for row in rows)
        / pairs_per_row
        / len(rows)
    )
    pi = [sum(row.count(c) for row in rows) / (len(rows) * n_raters) for c in categories]
    p_e = sum(p * (1 - p) for p in pi) / (len(categories) - 1)
    return p_a, (p_a - p_e) / (1 - p_e)


def _agreement_str(rows: list[list[str]], categories: list[str]) -> str:
    """Format percent agreement and AC1 (with N), or N/A when too few rows."""
    if len(rows) < 2:
        return "N/A (insufficient shared non-null rows)"
    p_a, ac1 = _agreement_stats(rows, categories)
    return f"agreement {100 * p_a:.1f}%, AC1 {ac1:.3f}  (N={len(rows)})"


def _load_annotated_sets() -> tuple[dict[str, pl.DataFrame], list[str]]:
    """Load each annotator's trial sheet and the columns present in all of them."""
    dfs = {
        annotator: pl.read_csv(
            DATA_DIR / TRAINING_ANNOTATION_FILENAME_TEMPLATE.format(annotator=annotator),
            infer_schema=False,
        )
        for annotator in ANNOTATORS
    }

    compare_columns = []
    for col in ANNOTATION_CATEGORIES:
        if all(col in df.columns for df in dfs.values()):
            compare_columns.append(col)
        else:
            typer.echo(f"Skipping {col}: not present in every annotated set")
    return dfs, compare_columns


def _column_values(base: pl.DataFrame, col: str) -> dict[str, list[str | None]]:
    """Each annotator's values for a column, empty cells as None, unknown categories rejected."""
    annotator_vals: dict[str, list[str | None]] = {}
    for annotator in ANNOTATORS:
        vals = [
            str(v).strip() if (v is not None and str(v).strip()) else None
            for v in base[f"{col}_{annotator}"].to_list()
        ]
        unknown = {v for v in vals if v is not None} - set(ANNOTATION_CATEGORIES[col])
        if unknown:
            raise ValueError(f"{annotator} has unexpected {col} values: {sorted(unknown)}")
        annotator_vals[annotator] = vals
    return annotator_vals


@app.command()
def compare_annotation_sets() -> None:
    """Report percent agreement, Gwet's AC1 and disagreements across the trial sets."""
    # Inner join on the shared row identifier keeps only rows present in all files
    dfs, compare_columns = _load_annotated_sets()

    base = (
        dfs["eva"]
        .select(
            [
                "document_repository_link_id",
                "document_url",
                "repository_url",
                *compare_columns,
            ]
        )
        .rename({col: f"{col}_eva" for col in compare_columns})
    )

    for annotator in ["sarah", "anna"]:
        other = (
            dfs[annotator]
            .select(["document_repository_link_id", *compare_columns])
            .rename({col: f"{col}_{annotator}" for col in compare_columns})
        )
        base = base.join(other, on="document_repository_link_id", how="inner")

    doc_urls = base["document_url"].to_list()
    repo_urls = base["repository_url"].to_list()

    summary_rows = []
    for col in compare_columns:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Column: {col}")
        typer.echo("=" * 60)

        categories = ANNOTATION_CATEGORIES[col]
        annotator_vals = _column_values(base, col)

        # --- Pairwise agreement ---
        typer.echo("\n--- Inter-Rater Agreement ---")
        typer.echo("Pairwise:")
        pairs = [("eva", "sarah"), ("eva", "anna"), ("sarah", "anna")]
        for name_a, name_b in pairs:
            pair_rows = [
                [a, b]
                for a, b in zip(annotator_vals[name_a], annotator_vals[name_b], strict=True)
                if a is not None and b is not None
            ]
            typer.echo(f"  {name_a} vs {name_b}: {_agreement_str(pair_rows, categories)}")

        # --- Three-way agreement ---
        # Keep only rows every annotator labeled
        shared_rows: list[list[str]] = []
        for i in range(len(doc_urls)):
            row_vals = [annotator_vals[ann][i] for ann in ANNOTATORS]
            non_null = [v for v in row_vals if v is not None]
            if len(non_null) == len(ANNOTATORS):
                shared_rows.append(non_null)
        typer.echo(f"Three-way: {_agreement_str(shared_rows, categories)}")
        if len(shared_rows) >= 2:
            p_a, ac1 = _agreement_stats(shared_rows, categories)
            summary_rows.append(
                {
                    "column": col,
                    "n_pairs": len(shared_rows),
                    "n_all_agree": sum(len(set(row)) == 1 for row in shared_rows),
                    "percent_agreement": 100 * p_a,
                    "gwet_ac1": ac1,
                }
            )

        # --- Disagreements ---
        disagreements = [
            i
            for i in range(len(doc_urls))
            if len({annotator_vals[ann][i] for ann in ANNOTATORS}) > 1
        ]
        typer.echo(f"\n--- Disagreements ({len(disagreements)} rows) ---")
        for i in disagreements:
            typer.echo(f"- document doi: {doc_urls[i]}")
            typer.echo(f"- repository url: {repo_urls[i]}")
            for ann in ANNOTATORS:
                v = annotator_vals[ann][i]
                typer.echo(f"\t- {ann}: {v if v is not None else '(empty)'}")
            typer.echo("")

    summary = pl.DataFrame(summary_rows)
    typer.echo(summary)
    u.save_table(summary, "link_directionality_trial_agreement", u.OUTPUT_DIR)


def _load_annotated_pairs() -> pl.DataFrame:
    """Stack every present annotator's sheet plus the agreed trial set, blank cells as null."""
    categories = {**ANNOTATION_CATEGORIES, "document_is_open_access": OPEN_ACCESS_CATEGORIES}
    frames = []
    sources = [
        (annotator, DATA_DIR / ANNOTATED_FILENAME_TEMPLATE.format(annotator=annotator))
        for annotator in ANNOTATORS
    ]
    for annotator, path in [*sources, ("agreed", AGREED_FILENAME_PATH)]:
        if not path.exists():
            typer.echo(f"Skipping {annotator}: {path.name} not found")
            continue
        df = pl.read_csv(path, infer_schema=False).with_columns(
            pl.col(col).str.strip_chars().replace("", None) for col in categories
        )
        for col, allowed in categories.items():
            unknown = set(df[col].drop_nulls().to_list()) - set(allowed)
            if unknown:
                raise ValueError(f"{annotator} has unexpected {col} values: {sorted(unknown)}")
        frames.append(df.with_columns(pl.lit(annotator).alias("annotator")))
    if not frames:
        raise FileNotFoundError("No annotated link-directionality files found")
    return pl.concat(frames, how="diagonal")


def _stat_row(
    subset: str, statistic: str, count: int, denominator: int, with_ci: bool = False
) -> dict[str, str | int | float | None]:
    """One summary row: count, denominator, percent and optionally an exact 95% CI."""
    row: dict[str, str | int | float | None] = {
        "subset": subset,
        "statistic": statistic,
        "count": count,
        "denominator": denominator,
        "percent": 100 * count / denominator if denominator else None,
        "ci_low_percent": None,
        "ci_high_percent": None,
    }
    if with_ci and denominator:
        ci = binomtest(count, denominator).proportion_ci(method="exact")
        row["ci_low_percent"] = 100 * ci.low
        row["ci_high_percent"] = 100 * ci.high
    return row


def _stat_rows(df: pl.DataFrame, subset: str) -> list[dict[str, str | int | float | None]]:
    """Every statistic for one set of annotated pairs."""
    rows = []
    n_all = df.height
    labeled = df.filter(pl.col("label").is_not_null())
    n_labeled = labeled.height
    n_match = labeled.filter(pl.col("label") == "match").height
    n_no_match = labeled.filter(pl.col("label") == "no-match").height

    rows.append(_stat_row(subset, "precision_strict", n_match, n_labeled, with_ci=True))
    rows.append(
        _stat_row(subset, "precision_lenient", n_match, n_match + n_no_match, with_ci=True)
    )
    rows.append(_stat_row(subset, "excluded", n_labeled - n_match - n_no_match, n_labeled))

    # Directionality and access are over matched pairs only
    matches = labeled.filter(pl.col("label") == "match")
    a2r_col, r2a_col = "this-repo-linked-in-article", "this-article-linked-in-repo"
    d = matches.drop_nulls([a2r_col, r2a_col])
    a2r, r2a = pl.col(a2r_col) == "yes", pl.col(r2a_col) == "yes"
    counts = {
        "neither": d.filter(~a2r & ~r2a).height,
        "one_way": d.filter(a2r ^ r2a).height,
        "both": d.filter(a2r & r2a).height,
        "one_way_article_to_repo_only": d.filter(a2r & ~r2a).height,
        "one_way_repo_to_article_only": d.filter(~a2r & r2a).height,
    }
    for name, count in counts.items():
        rows.append(_stat_row(subset, f"directionality_match_{name}", count, d.height))

    for name, col in [("article_links_repo", a2r_col), ("repo_links_article", r2a_col)]:
        filled = matches.drop_nulls(col)
        rows.append(
            _stat_row(
                subset,
                f"{name}_match",
                filled.filter(pl.col(col) == "yes").height,
                filled.height,
            )
        )

    n_closed_match = matches.filter(pl.col("document_is_open_access") == "0").height
    rows.append(_stat_row(subset, "closed_access_match", n_closed_match, matches.height))
    n_no_full_text_match = matches.filter(pl.col("full-text-available") == "no").height
    rows.append(_stat_row(subset, "no_full_text_match", n_no_full_text_match, matches.height))

    rows.append(_stat_row(subset, "article_link_unknown", df[a2r_col].null_count(), n_all))

    no_full_text = df.filter(pl.col("full-text-available") == "no").height
    rows.append(_stat_row(subset, "no_full_text", no_full_text, n_all))
    rows.append(
        _stat_row(subset, "full_text_blank", df["full-text-available"].null_count(), n_all)
    )
    return rows


@app.command()
def summarize_annotated_pairs() -> None:
    """Report precision, directionality and access stats from the independent annotation sets."""
    df = _load_annotated_pairs()
    annotators = df["annotator"].unique(maintain_order=True).to_list()
    typer.echo(f"Pairs: {df.height} from {annotators}")

    blank_labels = df.filter(pl.col("label").is_null())
    typer.echo(f"\n--- Blank labels ({blank_labels.height} rows) ---")
    for r in blank_labels.iter_rows(named=True):
        typer.echo(
            f"- {r['annotator']} link {r['document_repository_link_id']}: "
            f"{r['document_url']} | {r['repository_url']} | notes: {r['notes'] or '(none)'}"
        )

    rows = _stat_rows(df, "all")
    for annotator in annotators:
        rows.extend(_stat_rows(df.filter(pl.col("annotator") == annotator), annotator))

    for subset in ["all", *annotators]:
        typer.echo(f"\n--- {subset} ---")
        for r in (r for r in rows if r["subset"] == subset):
            pct = "n/a" if r["percent"] is None else f"{r['percent']:.1f}%"
            ci = (
                ""
                if r["ci_low_percent"] is None
                else f" (95% CI {r['ci_low_percent']:.1f}-{r['ci_high_percent']:.1f}%)"
            )
            typer.echo(f"{r['statistic']}: {r['count']}/{r['denominator']} = {pct}{ci}")

    u.save_table(pl.DataFrame(rows), "link_directionality_annotation_summary", u.OUTPUT_DIR)


if __name__ == "__main__":
    app()
