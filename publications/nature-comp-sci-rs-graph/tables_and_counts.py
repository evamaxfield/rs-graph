#!/usr/bin/env python3

"""Manuscript tables and placeholder-filling counts: Table 1, repository contributor
counts, the mining-rounds table (with its per-seed-source breakdown), and dataset
coverage counts.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import utils as u

from rs_graph.utils.software_alignment import align_software_names

###############################################################################

TABLE1_TOP_N_PER_ECOSYSTEM = 15


def _thousands(column: str) -> pl.Expr:
    """Format an integer column with thousands separators."""
    return pl.col(column).map_elements(lambda n: f"{n:,}", return_dtype=pl.String)


SEED_SOURCE_NAMES: list[str] = ["joss", "plos", "pwc", "softcite_2025", "softwarex"]


def _load_links_with_source_names() -> pl.DataFrame:
    """Load document_repository_link joined with dataset_source names."""
    raw_links = u.load_table("document_repository_link")
    dataset_sources = u.load_table("dataset_source")
    return raw_links.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"), pl.col("name").alias("dataset_source_name")
        ),
        on="dataset_source_id",
        how="left",
    )


###############################################################################
# Table 1 -- top software by mentions / imports / dependents


def table1_top_software_by_usage(
    output_dir: Path = u.OUTPUT_DIR,
    cutoff: float = 85.0,
    top_n_per_ecosystem: int = TABLE1_TOP_N_PER_ECOSYSTEM,
) -> None:
    """
    Build Table 1: one row per software, anchored on the import-normalized software name,
    split by ecosystem (Python / R) and ranked within each by import count. Dependency count
    comes from a second, separate import-vs-dependency Hungarian alignment -- same tool,
    cutoff, and two-views-at-a-time constraint as Figure 4's import-vs-mention pass, never
    combined into one three-way alignment. Mention count reuses Figure 4's per-pair
    import-vs-mention alignment logic, aggregated per software instead of per field/year.
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print(
        "\nLoading repository_import, repository_dependency, document_software_mention from "
        "HuggingFace..."
    )
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  repository_dependency: {len(deps):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")
    deps = u.clean_dependency_names(deps)
    mentions = u.clean_mention_names(mentions)

    # ---- Restrict to Python/R ecosystem repositories ----
    # `repository_primary_language` is GitHub's byte-count-based language classification.
    # "Jupyter Notebook"-classified repos are likely Python-ecosystem in practice, but are
    # excluded rather than assumed -- the count is reported below.
    n_before_lang = df.n_unique("repository_id")
    lang_df = df.filter(pl.col("repository_primary_language").is_in(["Python", "R"]))
    n_after_lang = lang_df.n_unique("repository_id")
    n_jupyter = df.filter(pl.col("repository_primary_language") == "Jupyter Notebook").n_unique(
        "repository_id"
    )
    print(
        f"\nRestricting to repositories with primary_language in {{Python, R}}: "
        f"{n_after_lang:,} of {n_before_lang:,} repositories remain "
        f"({n_jupyter:,} additional 'Jupyter Notebook'-primary repos excluded -- likely "
        "Python-heavy in practice but not assumed here)"
    )

    repo_ecosystem: dict[int, str] = dict(
        lang_df.select("repository_id", "repository_primary_language")
        .unique(subset="repository_id")
        .iter_rows()
    )

    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible_repo_ids = set(repo_ecosystem.keys()) & repo_with_import
    print(
        f"Of those, {len(eligible_repo_ids):,} repositories have >=1 extracted import and are "
        "eligible for this table."
    )

    imports_by_repo = u.normalized_names_by_id(
        imports.filter(pl.col("repository_id").is_in(eligible_repo_ids)), "repository_id"
    )
    mentions_by_doc = u.normalized_names_by_id(mentions, "document_id")
    # Pre-grouped per ecosystem's own manifest ecosystems, so the per-repo loop below never
    # has to re-filter the (multi-million-row) full deps table.
    deps_by_repo_by_ecosystem: dict[str, dict[int, list[str]]] = {
        eco: u.normalized_names_by_id(
            deps.filter(
                pl.col("repository_id").is_in(eligible_repo_ids)
                & pl.col("ecosystem").is_in(dep_ecosystems)
            ),
            "repository_id",
        )
        for eco, dep_ecosystems in u.MANIFEST_ECOSYSTEMS_BY_LANGUAGE.items()
    }

    # ---- Import counts: # of eligible repos importing each canonical name ----
    # Keyed by (name, ecosystem), not bare name -- the same normalized name can legitimately
    # appear as an import in both a Python-primary and an R-primary repository, and this
    # table is explicitly split by ecosystem, so counts must never merge across ecosystems.
    import_counts: dict[tuple[str, str], int] = {}
    for repo_id, names in imports_by_repo.items():
        eco = repo_ecosystem[repo_id]
        for name in set(names):
            key = (name, eco)
            import_counts[key] = import_counts.get(key, 0) + 1

    # ---- Dependency counts: second, separate import-vs-dependency alignment, per repo ----
    dependency_counts: dict[tuple[str, str], int] = {}
    n_repo_dep_aligned = 0
    for repo_id, import_names in imports_by_repo.items():
        eco = repo_ecosystem[repo_id]
        dep_names = deps_by_repo_by_ecosystem[eco].get(repo_id, [])
        if not dep_names:
            continue
        matches = align_software_names(
            items_a=import_names,
            items_b=dep_names,
            source_a="import",
            source_b="dependency",
            cutoff=cutoff,
            method="global_min_diff",
        )
        n_repo_dep_aligned += 1
        for name in {m.normalized_item_one for m in matches}:
            key = (name, eco)
            dependency_counts[key] = dependency_counts.get(key, 0) + 1
    print(
        f"\nImport-vs-dependency alignment ran on {n_repo_dep_aligned:,} repositories with both "
        "imports and manifest dependencies in their ecosystem."
    )

    # ---- Mention counts: per-pair import-vs-mention alignment, aggregated per software ----
    mention_doc_sets: dict[tuple[str, str], set[int]] = {}
    all_mention_names_seen: set[str] = set()
    matched_mention_names: set[str] = set()
    # Mention extraction is absent/partial after the cap year, so mention counts only
    # consider pairs published at or before it (imports/dependencies stay uncapped).
    eligible_pairs = (
        df.filter(
            pl.col("repository_id").is_in(eligible_repo_ids)
            & (pl.col("document_publication_year") <= u.MENTION_EXTRACTION_YEAR_CAP)
        )
        .select("document_id", "repository_id")
        .unique()
    )
    for row in eligible_pairs.iter_rows(named=True):
        repo_id, doc_id = row["repository_id"], row["document_id"]
        eco = repo_ecosystem[repo_id]
        import_names = imports_by_repo.get(repo_id, [])
        if not import_names:
            continue
        mention_names = mentions_by_doc.get(doc_id, [])
        all_mention_names_seen.update(mention_names)
        if not mention_names:
            continue
        matches = align_software_names(
            items_a=import_names,
            items_b=mention_names,
            source_a="import",
            source_b="mention",
            cutoff=cutoff,
            method="global_min_diff",
        )
        for m in matches:
            mention_doc_sets.setdefault((m.normalized_item_one, eco), set()).add(doc_id)
            matched_mention_names.add(m.normalized_item_two)
    mention_counts = {k: len(v) for k, v in mention_doc_sets.items()}

    n_unmatched_mentions = len(all_mention_names_seen - matched_mention_names)
    print(
        f"\n{n_unmatched_mentions:,} distinct mentioned-software names never matched any "
        f"import at cutoff={cutoff} (software mentioned but never imported cannot appear in "
        "this import-anchored table)."
    )

    # ---- Assemble table ----
    rows = [
        {
            "ecosystem": eco,
            "software_name": name,
            "import_count": n_import,
            "dependency_count": dependency_counts.get((name, eco), 0),
            "mention_count": mention_counts.get((name, eco), 0),
        }
        for (name, eco), n_import in import_counts.items()
    ]
    table = pl.DataFrame(rows).with_columns(
        (1000 * pl.col("mention_count") / pl.col("import_count"))
        .round(1)
        .alias("mentions_per_1000_imports")
    )

    top_tables = []
    for eco in ["Python", "R"]:
        eco_table = (
            table.filter(pl.col("ecosystem") == eco)
            .sort("import_count", descending=True)
            .head(top_n_per_ecosystem)
        )
        top_tables.append(eco_table)
        print(f"\nTop {top_n_per_ecosystem} {eco} software by import count:")
        print(eco_table)

    # Publication shape: human headers, thousands separators, ecosystem shown once per block.
    final_table = (
        pl.concat(top_tables)
        .with_columns(
            pl.when(pl.int_range(pl.len()).over("ecosystem") == 0)
            .then(pl.col("ecosystem"))
            .otherwise(pl.lit(""))
            .alias("ecosystem")
        )
        .select(
            pl.col("ecosystem").alias("Ecosystem"),
            pl.col("software_name").alias("Software"),
            _thousands("import_count").alias("Importing repositories"),
            _thousands("dependency_count").alias("Declaring repositories"),
            _thousands("mention_count").alias("Mentioning articles"),
            pl.col("mentions_per_1000_imports").alias("Mentions per 1,000 imports"),
        )
    )
    print("\nTable 1, publication shape:")
    print(final_table)
    u.save_table(final_table, "table1_top_software_by_usage", output_dir)
    u.print_caption_note(
        "table1_top_software_by_usage",
        f"Mention counts include only articles published through "
        f"{u.MENTION_EXTRACTION_YEAR_CAP}: SoftCite-2025 mention-extraction coverage "
        "collapses after that year (normal rates through May 2023, then exactly 0% from "
        "July 2023 onward), so the cap gives mentions their fairest representation while "
        "imports and dependencies each use their own full reliable range",
    )


###############################################################################
# Median repository contributor count


def median_repository_contributor_count(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Fill line 33's `X%` placeholder -- "The median scientific repository has only a single
    contributor (X%)...". Filters at the pair level first (standard filters), derives the
    surviving repository set, then computes per-repository contributor counts from
    `repository_contributor` (repositories with no `repository_contributor` rows at all count
    as 0 contributors, not dropped).
    """
    df = u.load_filtered_pairs()
    filtered_repo_ids = df.get_column("repository_id").unique()
    print(
        f"\nRepositories surviving standard pair-level filtering: {filtered_repo_ids.len():,}"
    )

    print("Loading repository_contributor from HuggingFace...")
    repo_contributors = u.load_table("repository_contributor")
    print(f"  repository_contributor: {len(repo_contributors):,} rows")

    contributor_counts = (
        repo_contributors.filter(pl.col("repository_id").is_in(filtered_repo_ids.implode()))
        .group_by("repository_id")
        .agg(pl.n_unique("developer_account_id").alias("n_contributors"))
    )

    repos_frame = (
        pl.DataFrame({"repository_id": filtered_repo_ids})
        .join(contributor_counts, on="repository_id", how="left")
        .with_columns(pl.col("n_contributors").fill_null(0))
    )

    n_with_zero = repos_frame.filter(pl.col("n_contributors") == 0).height
    print(
        f"Repositories with no repository_contributor rows at all (treated as 0 contributors): "
        f"{n_with_zero:,}"
    )

    median_contributors = repos_frame.get_column("n_contributors").median()
    n_single = repos_frame.filter(pl.col("n_contributors") == 1).height
    pct_single = 100 * n_single / repos_frame.height

    u.save_table(repos_frame, "repository_contributor_counts", output_dir)

    print("\n--- Median repository contributor count ---")
    print(f"Median contributors per repository: {median_contributors}")
    print(
        f"Repositories with exactly one contributor: {n_single:,} of {repos_frame.height:,} "
        f"({pct_single:.1f}%)"
    )
    print(f"Line 33 placeholder fill: X = {pct_single:.1f}%")
    print("---------------------------------------------\n")

    # ---- Repository development characteristics (line 33's medians + FOOTNOTE 3's tables) ----
    # Commit/development metrics are repo-level (dedup to first-seen pair);
    # publication-relative deltas are pair-level.
    pair_char = df.with_columns(
        pl.col("repository_last_pushed_datetime")
        .str.to_datetime(strict=False)
        .alias("_last_pushed"),
        pl.col("document_publication_date_parsed").cast(pl.Datetime("us")).alias("_pub_dt"),
    ).with_columns(
        (pl.col("_pub_dt") - pl.col("repository_creation_datetime_parsed"))
        .dt.total_days()
        .alias("days_created_before_publication"),
        (pl.col("_last_pushed") - pl.col("_pub_dt"))
        .dt.total_days()
        .alias("days_last_push_after_publication"),
        (pl.col("_last_pushed") - pl.col("repository_creation_datetime_parsed"))
        .dt.total_days()
        .alias("development_days_creation_to_last_push"),
        pl.col("repository_commits_count").cast(pl.Float64).alias("commit_count"),
    )
    repo_char = pair_char.unique(subset="repository_id", keep="first").join(
        repos_frame.select("repository_id", "n_contributors"), on="repository_id", how="left"
    )

    metric_specs = [
        ("n_contributors", repo_char),
        ("commit_count", repo_char),
        ("development_days_creation_to_last_push", repo_char),
        ("days_created_before_publication", pair_char),
        ("days_last_push_after_publication", pair_char),
    ]
    summary_rows = []
    for doctype in [None, "article", "preprint", "other"]:
        for metric, frame in metric_specs:
            sub = (
                frame
                if doctype is None
                else frame.filter(pl.col("document_type_bucket") == doctype)
            )
            vals = sub.get_column(metric).drop_nulls()
            if vals.len() == 0:
                continue
            summary_rows.append(
                {
                    "document_type": doctype or "all",
                    "metric": metric,
                    "n": vals.len(),
                    "median": float(vals.median()),
                    "p10": float(vals.quantile(0.10)),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                    "p90": float(vals.quantile(0.90)),
                }
            )
    char_summary = pl.DataFrame(summary_rows)
    u.save_table(char_summary, "repository_characteristics_summary", output_dir)
    print("Repository development characteristics (line 33 / FOOTNOTE 3):")
    print(char_summary.filter(pl.col("document_type") == "all"))


###############################################################################
# Mining-rounds table (iterations 1-5)


def mining_rounds_table(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Fill Table X (line 237) and line 254's `X` placeholder. Two halves:
      (a) new article-repository pairs per source/iteration -- a group-by on
          `document_repository_link`'s (dataset_source_id, iteration).
      (b) new researcher-developer-account identity links per iteration -- a structural join
          (not timestamp-based) attributing each identity link to the earliest iteration
          whose document-repository pair could have produced it.
    The combined table's "Seed" row is broken down by original seed source in indented
    sub-rows; only mining_rounds_table.csv is saved.
    """
    print("Loading document_repository_link, dataset_source from HuggingFace...")
    raw_links = _load_links_with_source_names()

    # ---- (a) New article-repository pairs per source/iteration -- simple group-by ----
    links_with_bucket = raw_links.with_columns(
        pl.col("iteration").fill_null(-1).alias("iteration_bucket")
    )
    pairs_by_source_iteration = (
        links_with_bucket.group_by(["dataset_source_name", "iteration_bucket"])
        .agg(pl.len().alias("n_pairs"))
        .sort(["iteration_bucket", "dataset_source_name"])
    )
    print("\n--- (a) New article-repository pairs by source and iteration (raw group-by) ---")
    print(pairs_by_source_iteration)

    pairs_by_iteration_total = (
        links_with_bucket.group_by("iteration_bucket")
        .agg(pl.len().alias("n_pairs"))
        .sort("iteration_bucket")
    )
    print("\nNew article-repository pairs per iteration, totaled across sources:")
    print(pairs_by_iteration_total)

    mining_rounds_total = int(
        pairs_by_iteration_total.filter(pl.col("iteration_bucket") != -1)
        .get_column("n_pairs")
        .sum()
    )
    extended_mining_total = int(
        pairs_by_iteration_total.filter(pl.col("iteration_bucket") >= 4)
        .get_column("n_pairs")
        .sum()
    )
    print(
        f"\nTotal new pairs across all mining iterations (1-5), RAW candidates: {mining_rounds_total:,}"
    )
    print(
        f"Line 254 placeholder fill, RAW candidates (Extended Mining Round, iterations 4+5): "
        f"{extended_mining_total:,}"
    )

    # ---- Same group-by, restricted to the standard-filtered pairs table. The raw group-by
    # counts every candidate row regardless of confidence; most predicted (non-seed) rows do
    # NOT meet 0.9994, so raw per-iteration counts overstate what's retained. The filtered
    # version is what "new pairs added to the dataset" in Table X and line 254 means.
    filtered_pairs_for_iteration = u.load_filtered_pairs()
    pairs_by_iteration_filtered = (
        filtered_pairs_for_iteration.with_columns(
            pl.col("link_processing_iteration").fill_null(-1).alias("iteration_bucket")
        )
        .group_by("iteration_bucket")
        .agg(pl.len().alias("n_pairs"))
        .sort("iteration_bucket")
    )
    print(
        "\nNew article-repository pairs per iteration, RETAINED (standard-filtered, "
        ">=0.9994-or-NULL confidence + post-2008):"
    )
    print(pairs_by_iteration_filtered)
    mining_rounds_total_filtered = int(
        pairs_by_iteration_filtered.filter(pl.col("iteration_bucket") != -1)
        .get_column("n_pairs")
        .sum()
    )
    extended_mining_total_filtered = int(
        pairs_by_iteration_filtered.filter(pl.col("iteration_bucket") >= 4)
        .get_column("n_pairs")
        .sum()
    )
    print(
        f"Total new RETAINED pairs across all mining iterations (1-5): "
        f"{mining_rounds_total_filtered:,}"
    )
    print(
        f"Line 254 placeholder fill, RETAINED (Extended Mining Round, iterations 4+5): "
        f"{extended_mining_total_filtered:,}"
    )
    # ---- (b) New researcher-developer-account identity links per iteration ----
    print(
        "\nLoading researcher_developer_account_link, document_contributor, "
        "repository_contributor from HuggingFace..."
    )
    rdal = u.load_table("researcher_developer_account_link")
    document_contributors = u.load_table("document_contributor")
    repository_contributors = u.load_table("repository_contributor")
    print(f"  researcher_developer_account_link: {len(rdal):,} rows")
    print(f"  document_contributor: {len(document_contributors):,} rows")
    print(f"  repository_contributor: {len(repository_contributors):,} rows")

    rdal_filtered = rdal.filter(
        pl.col("predictive_model_confidence") >= u.DEFAULT_RDAL_CONFIDENCE_THRESHOLD
    ).select("researcher_id", "developer_account_id")
    print(
        f"After filtering researcher_developer_account_link to confidence >= "
        f"{u.DEFAULT_RDAL_CONFIDENCE_THRESHOLD}: {rdal_filtered.height:,} identity links remain"
    )

    # Reuse the already-loaded standard-filtered pairs table instead of re-running the full
    # HuggingFace load/join/filter pipeline for the same default args.
    drl_filtered = filtered_pairs_for_iteration.select(
        "document_id", "repository_id", "link_processing_iteration"
    )
    print(
        f"Using the standard-filtered article-repository pairs table as the join's candidate "
        f"pair pool: {drl_filtered.height:,} pairs"
    )

    # Expand each identity to its researcher's candidate documents, then narrow to the
    # document-repository pairs whose repository is also in that identity's developer
    # account's contribution list.
    rdal_docs = rdal_filtered.join(
        document_contributors.select("researcher_id", "document_id"), on="researcher_id"
    )
    rdal_docs_pairs = rdal_docs.join(drl_filtered, on="document_id")
    candidate_pairs = rdal_docs_pairs.join(
        repository_contributors.select("developer_account_id", "repository_id"),
        on=["developer_account_id", "repository_id"],
        how="inner",
    )
    print(
        f"Candidate document-repository pairs connecting an identity's author and developer "
        f"account: {candidate_pairs.height:,} rows (before dedup/earliest-iteration reduction)"
    )

    attribution = (
        candidate_pairs.with_columns(
            pl.col("link_processing_iteration").fill_null(-1).alias("iteration_bucket")
        )
        .group_by(["researcher_id", "developer_account_id"])
        .agg(pl.min("iteration_bucket").alias("earliest_iteration_bucket"))
    )

    n_identities_total = rdal_filtered.height
    n_identities_attributed = attribution.height
    n_identities_unattributed = n_identities_total - n_identities_attributed
    print(
        f"\nOf {n_identities_total:,} filtered identity links, {n_identities_attributed:,} "
        f"had >=1 qualifying candidate document-repository pair and could be attributed to an "
        f"iteration; {n_identities_unattributed:,} had none (not in the standard-filtered pairs "
        f"table's candidate pool -- e.g. discovered via a pair that didn't clear the 0.9994 "
        f"confidence or post-2008 filters) and are excluded from the per-iteration counts below."
    )

    identities_by_iteration = (
        attribution.group_by("earliest_iteration_bucket")
        .agg(pl.len().alias("n_new_identities"))
        .sort("earliest_iteration_bucket")
    )
    print("\n--- (b) New researcher-developer-account identity links per iteration ---")
    print(identities_by_iteration)

    identities_mining_total = int(
        identities_by_iteration.filter(pl.col("earliest_iteration_bucket") != -1)
        .get_column("n_new_identities")
        .sum()
    )
    print(
        f"\nTotal new identity links attributed to mining iterations (1-5): {identities_mining_total:,}"
    )

    print(
        "\nAttribution caveats:\n"
        "  (a) An identity connected through pairs spanning multiple iterations attributes "
        "to the earliest ('first possible discovery', not a certainty).\n"
        "  (b) An identity whose earliest qualifying pair came from a seed source attributes "
        "to the seed even if it also links via later mining rounds.\n"
        "  (c) Ties within one iteration are unambiguous (min() is well-defined).\n"
    )

    print(
        "--- Combined mining-rounds table (Table X), using RETAINED (standard-filtered) pair "
        "counts -- this is the version that matches 'new pairs added to the dataset' ---"
    )
    # Iterations above 4 merge into a single "5" bucket; -1 is the seed ingestion.
    combined = (
        pairs_by_iteration_filtered.rename({"n_pairs": "new_pairs"})
        .join(
            identities_by_iteration.rename({"n_new_identities": "new_identities"}),
            left_on="iteration_bucket",
            right_on="earliest_iteration_bucket",
            how="full",
            coalesce=True,
        )
        .with_columns(
            pl.col("new_pairs").fill_null(0),
            pl.col("new_identities").fill_null(0),
            pl.when(pl.col("iteration_bucket") > 4)
            .then(5)
            .otherwise(pl.col("iteration_bucket"))
            .alias("iteration_bucket"),
        )
        .group_by("iteration_bucket")
        .agg(pl.sum("new_pairs"), pl.sum("new_identities"))
        .sort("iteration_bucket")
        .with_columns(
            pl.col("new_pairs").cum_sum().alias("cumulative_pairs"),
            pl.col("new_identities").cum_sum().alias("cumulative_identities"),
        )
        .select(
            pl.when(pl.col("iteration_bucket") == -1)
            .then(pl.lit("Seed"))
            .otherwise(pl.lit("Mining round ") + pl.col("iteration_bucket").cast(pl.String))
            .alias("round"),
            pl.col("new_pairs").cast(pl.Int64),
            pl.col("cumulative_pairs").cast(pl.Int64),
            pl.col("new_identities").cast(pl.Int64),
            pl.col("cumulative_identities").cast(pl.Int64),
        )
    )

    # Per-source breakdown of the Seed total, indented under the "Seed" group-header row.
    # Identity attribution is not source-scoped, so identity columns stay blank on sub-rows.
    seed_source_display = {
        "joss": "JOSS",
        "plos": "PLOS",
        "pwc": "Papers with Code",
        "softcite_2025": "SoftCite 2025",
        "softwarex": "SoftwareX",
    }
    seed_by_source = (
        filtered_pairs_for_iteration.filter(
            pl.col("link_processing_iteration").is_null()
            & pl.col("dataset_source_name").is_in(SEED_SOURCE_NAMES)
        )
        .group_by("dataset_source_name")
        .agg(pl.len().alias("new_pairs"))
        .sort("new_pairs", descending=True)
        .select(
            (
                pl.lit("  ") + pl.col("dataset_source_name").replace_strict(seed_source_display)
            ).alias("round"),
            pl.col("new_pairs").cast(pl.Int64),
            pl.lit(None, dtype=pl.Int64).alias("cumulative_pairs"),
            pl.lit(None, dtype=pl.Int64).alias("new_identities"),
            pl.lit(None, dtype=pl.Int64).alias("cumulative_identities"),
        )
    )

    # Seed header first, its per-source sub-rows, then the mining rounds.
    combined = pl.concat([combined.head(1), seed_by_source, combined.slice(1)])
    print(combined)
    u.save_table(combined, "mining_rounds_table", output_dir)


###############################################################################
# Dataset coverage counts (imports / dependencies / mentions presence)


def data_coverage_counts(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Count how many standard-filtered pairs have each data view present: articles with >=1
    extracted software mention, repositories with >=1 import, repositories with >=1
    pypi/conda/cran manifest dependency, and every pairwise/three-way combination
    ("complete coverage" = imports AND dependencies AND mentions).
    """
    df = u.load_filtered_pairs()

    print("\nLoading imports/dependencies/mentions from HuggingFace...")
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  repository_dependency: {len(deps):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")
    deps = u.clean_dependency_names(deps)
    deps_pr = deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))

    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    repo_with_dep = set(deps_pr.get_column("repository_id").unique().to_list())
    doc_with_mention = set(mentions.get_column("document_id").unique().to_list())

    # Mention extraction is absent/partial after the cap year, so mention presence only
    # counts for pairs published at or before it.
    flagged = df.with_columns(
        pl.col("repository_id").is_in(repo_with_import).alias("has_import"),
        pl.col("repository_id").is_in(repo_with_dep).alias("has_dependency"),
        (
            pl.col("document_id").is_in(doc_with_mention)
            & (pl.col("document_publication_year") <= u.MENTION_EXTRACTION_YEAR_CAP)
        ).alias("has_mention"),
    )

    n_pairs = flagged.height
    n_articles = flagged.n_unique("document_id")
    n_repositories = flagged.n_unique("repository_id")
    print(
        f"\nStandard-filtered pairs: {n_pairs:,} ({n_articles:,} articles, {n_repositories:,} repositories)"
    )

    n_articles_with_mention = flagged.filter(pl.col("has_mention")).n_unique("document_id")
    print(f"Articles with >=1 extracted software mention: {n_articles_with_mention:,}")
    n_repos_with_import = flagged.filter(pl.col("has_import")).n_unique("repository_id")
    print(f"Repositories with >=1 extracted import: {n_repos_with_import:,}")
    n_repos_with_dep = flagged.filter(pl.col("has_dependency")).n_unique("repository_id")
    print(
        f"Repositories with >=1 {'/'.join(u.ALL_MANIFEST_ECOSYSTEMS)} manifest dependency: "
        f"{n_repos_with_dep:,}"
    )

    n_pairs_import_mention = flagged.filter(pl.col("has_import") & pl.col("has_mention")).height
    print(f"Pairs with imports AND mentions: {n_pairs_import_mention:,}")
    n_pairs_dep_mention = flagged.filter(
        pl.col("has_dependency") & pl.col("has_mention")
    ).height
    print(f"Pairs with dependencies AND mentions: {n_pairs_dep_mention:,}")
    n_pairs_import_dep = flagged.filter(pl.col("has_import") & pl.col("has_dependency")).height
    print(f"Pairs with imports AND dependencies: {n_pairs_import_dep:,}")
    n_pairs_complete = flagged.filter(
        pl.col("has_import") & pl.col("has_dependency") & pl.col("has_mention")
    ).height
    print(f"Pairs with all three (complete coverage): {n_pairs_complete:,}")

    summary = pl.DataFrame(
        {
            "statistic": [
                "n_pairs_total",
                "n_articles_total",
                "n_repositories_total",
                "n_articles_with_gte1_mention",
                "n_repositories_with_gte1_import",
                "n_repositories_with_gte1_manifest_dependency",
                "n_pairs_imports_and_mentions",
                "n_pairs_dependencies_and_mentions",
                "n_pairs_imports_and_dependencies",
                "n_pairs_complete_coverage",
            ],
            "value": [
                n_pairs,
                n_articles,
                n_repositories,
                n_articles_with_mention,
                n_repos_with_import,
                n_repos_with_dep,
                n_pairs_import_mention,
                n_pairs_dep_mention,
                n_pairs_import_dep,
                n_pairs_complete,
            ],
        }
    )
    u.save_table(summary, "data_coverage_counts", output_dir)
    u.print_caption_note(
        "data_coverage_counts",
        f"Mention-related counts include only articles published through "
        f"{u.MENTION_EXTRACTION_YEAR_CAP}: SoftCite-2025 mention-extraction coverage "
        "collapses after that year (normal rates through May 2023, then exactly 0% from "
        "July 2023 onward), so the cap gives mentions their fairest representation while "
        "imports and dependencies each use their own full reliable range",
    )
