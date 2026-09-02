#!/usr/bin/env python3

"""Shared loading, filtering, and plotting-support utilities for the Nature Computational
Science `rs-graph` figures/tables. Every figure/table function in
`main-figures-and-tables-rewrite.py` loads its data through `load_table` /
`load_filtered_pairs` below rather than reading from a local database, per the paper's
replication-package policy: every unit must be runnable standalone directly from the published
HuggingFace dataset, with no dependency on a local SQLite checkout.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl
import seaborn as sns
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from rapidfuzz import fuzz

from rs_graph.utils.software_alignment import align_software_names
from rs_graph.utils.software_alternates import are_alternates, are_known_distinct

###############################################################################

HF_DATASET = "sci-soft-collections/rs-graph-v2-full"

THIS_DIR = Path(__file__).parent
load_dotenv(str(THIS_DIR.parents[1] / ".env"))

DEFAULT_CONFIDENCE_THRESHOLD = 0.9994
DEFAULT_MIN_YEAR = 2008
# Paper-local override of the repo-wide 0.9 identity-link convention -- this manuscript
# states/uses 0.97 throughout (Methods, following Brown, Slaughter & Weber).
DEFAULT_RDAL_CONFIDENCE_THRESHOLD = 0.97

SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "pwc": "Papers with Code",
    "plos": "PLOS",
    "joss": "JOSS",
    "softwarex": "SoftwareX",
    "softcite_2025": "SoftCite 2025",
    "snowball-sampling-discovery": "Mined",
}

###############################################################################
# Loading


def load_table(table: str) -> pl.DataFrame:
    """Load a single rs-graph table directly from HuggingFace as a polars DataFrame."""
    ds = load_dataset(HF_DATASET, table, split="train", token=os.environ.get("HF_TOKEN"))
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _bucket_document_type(col: str = "document_document_type") -> pl.Expr:
    """Collapse the ~19 OpenAlex document types down to article / preprint / other."""
    return (
        pl.when(pl.col(col) == "article")
        .then(pl.lit("article"))
        .when(pl.col(col) == "preprint")
        .then(pl.lit("preprint"))
        .otherwise(pl.lit("other"))
        .alias("document_type_bucket")
    )


def load_filtered_pairs(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_year: int = DEFAULT_MIN_YEAR,
    top_n_fields: int = 10,
    apply_researcher_developer_filter: bool = False,
    rdal_confidence_threshold: float = DEFAULT_RDAL_CONFIDENCE_THRESHOLD,
) -> pl.DataFrame:
    """
    Load the standard, filtered article-repository pair base table used across (almost) every
    figure/table in this paper.

    Filtering order (per the plan doc's general build instructions):
      1. article-repository pair confidence >= `confidence_threshold`, or NULL.
      2. published after `min_year` (GitHub's launch year, 2008, by default).
      3. researcher-developer identity confidence >= `rdal_confidence_threshold`, only when
         `apply_researcher_developer_filter=True` -- none of Figures 2/3/4 touch researcher/
         developer-account identity, so this defaults to off and is provided for future units.
      4. Filtering is always done at the pair level first; entity-level subsets (repositories,
         documents) are always derived *from* the filtered pairs, never filtered directly.

    Every step prints how much data was filtered and how much remains.
    """
    print(f"Loading base tables from HuggingFace ({HF_DATASET})...")
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    dataset_sources = load_table("dataset_source")
    print(f"  document: {len(documents):,} rows")
    print(f"  document_repository_link: {len(article_repo_links):,} rows")
    print(f"  repository: {len(repositories):,} rows")

    # Top topic per document (highest score) -> field name + domain name.
    document_top_topics = (
        document_topics.sort("score", descending=True)
        .unique(subset="document_id", keep="first")
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select("document_id", "document_field_name", "document_domain_name")
    )

    pairs = article_repo_links.select(
        pl.col("id").alias("document_repository_link_id"),
        "document_id",
        "repository_id",
        "dataset_source_id",
        "predictive_model_confidence",
        pl.col("iteration").alias("link_processing_iteration"),
    )
    print(f"Starting pairs: {len(pairs):,}")

    # ---- Filter 1: pair confidence >= threshold, or NULL ----
    pairs = pairs.filter(
        (pl.col("predictive_model_confidence") >= confidence_threshold)
        | pl.col("predictive_model_confidence").is_null()
    )
    print(
        f"After filtering to pair confidence >= {confidence_threshold} or NULL: "
        f"{len(pairs):,} pairs remain"
    )

    # ---- Join in document/repository/topic metadata ----
    merged = (
        pairs.join(
            documents.select(*[pl.col(c).alias(f"document_{c}") for c in documents.columns]),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(c).alias(f"repository_{c}") for c in repositories.columns]
            ),
            on="repository_id",
        )
        .join(document_top_topics, on="document_id", how="left")
    )

    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d", strict=False)
        .alias("document_publication_date_parsed"),
        pl.col("repository_creation_datetime")
        .str.to_datetime(strict=False)
        .alias("repository_creation_datetime_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
        pl.col("repository_creation_datetime_parsed")
        .dt.year()
        .alias("repository_creation_year"),
    )

    # ---- Filter 2: published after min_year ----
    merged = merged.filter(pl.col("document_publication_year") >= min_year)
    print(f"After filtering to published >= {min_year}: {len(merged):,} pairs remain")

    # ---- Filter 3: researcher-developer identity confidence, only if requested ----
    if apply_researcher_developer_filter:
        document_contributors = load_table("document_contributor")
        repository_contributors = load_table("repository_contributor")
        rdal = load_table("researcher_developer_account_link").filter(
            pl.col("predictive_model_confidence") >= rdal_confidence_threshold
        )
        linked_researcher_ids = set(rdal.get_column("researcher_id").unique().to_list())
        linked_developer_ids = set(rdal.get_column("developer_account_id").unique().to_list())

        docs_with_linked_author = set(
            document_contributors.filter(pl.col("researcher_id").is_in(linked_researcher_ids))
            .get_column("document_id")
            .unique()
            .to_list()
        )
        repos_with_linked_contributor = set(
            repository_contributors.filter(
                pl.col("developer_account_id").is_in(linked_developer_ids)
            )
            .get_column("repository_id")
            .unique()
            .to_list()
        )

        merged = merged.filter(
            pl.col("document_id").is_in(docs_with_linked_author)
            & pl.col("repository_id").is_in(repos_with_linked_contributor)
        )
        print(
            f"After filtering to pairs with a researcher-developer identity link "
            f">= {rdal_confidence_threshold}: {len(merged):,} pairs remain"
        )

    # ---- Derived columns used across figures ----
    merged = merged.with_columns(_bucket_document_type())

    top_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields)
        .get_column("document_field_name")
        .to_list()
    )
    merged = merged.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_field_names))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    merged = merged.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"),
            pl.col("name").alias("dataset_source_name"),
        ),
        on="dataset_source_id",
        how="left",
    ).with_columns(
        pl.col("dataset_source_name")
        .replace(SOURCE_DISPLAY_NAMES)
        .alias("dataset_source_name_canonical")
    )

    print(f"Final filtered/derived pairs table: {len(merged):,} rows")
    print(
        "  n unique documents:",
        merged.n_unique("document_id"),
        "| n unique repositories:",
        merged.n_unique("repository_id"),
    )

    return merged


###############################################################################
# Dependency-name cleaning (round 4, R2.3)

# Ecosystems whose names may legitimately carry residual comparator/junk characters from
# manifest parsing. GitHub-Actions/npm names legitimately contain `/` and `@`, so cleaning is
# scoped to these three ecosystems only, never applied globally.
CLEANED_DEPENDENCY_ECOSYSTEMS: tuple[str, ...] = ("pypi", "conda", "cran")
# Everything from the first of these characters onward is a version spec, environment marker,
# URL ref, or comment fragment, not part of the software name.
_DEP_NAME_TRUNCATE_PATTERN = r"[<>=!~;@,()\[\]{}#%*:/\\|$?\"'`].*$"


def clean_dependency_names(deps: pl.DataFrame) -> pl.DataFrame:
    """
    Clean residual comparator/junk characters out of `software_name_normalized` for
    pypi/conda/cran rows only (round 4, R2.3): strip a leading conda channel prefix
    (everything up to and including the last `::`), truncate at the first
    comparator/marker/junk character, and drop rows (in those ecosystems) whose cleaned name
    is empty or doesn't start with [a-z0-9]. Rows in other ecosystems pass through unchanged.
    """
    in_scope = pl.col("ecosystem").is_in(CLEANED_DEPENDENCY_ECOSYSTEMS)
    n_before = deps.height
    n_changed_candidates = deps.filter(
        in_scope & pl.col("software_name_normalized").str.contains(r"[^a-z0-9.+-]")
    ).height

    cleaned = deps.with_columns(
        pl.when(in_scope)
        .then(
            pl.col("software_name_normalized")
            .str.replace(r"^.*::", "")
            .str.replace(_DEP_NAME_TRUNCATE_PATTERN, "")
        )
        .otherwise(pl.col("software_name_normalized"))
        .alias("software_name_normalized")
    ).filter(~in_scope | pl.col("software_name_normalized").str.contains(r"^[a-z0-9]"))
    print(
        f"Dependency-name cleaning ({'/'.join(CLEANED_DEPENDENCY_ECOSYSTEMS)} rows only): "
        f"{n_changed_candidates:,} rows carried residual non-[a-z0-9.] characters; "
        f"{n_before - cleaned.height:,} rows dropped as empty/pure junk after cleaning; "
        f"{cleaned.height:,} of {n_before:,} rows remain"
    )
    return cleaned


###############################################################################
# Import-vs-mention long frame (shared by Figure 4, Table 1, and Unit 5's regression)


def _independent_matched_names(
    pair_imports: list[str], pair_mentions: list[str], cutoff: float
) -> set[str]:
    """Independent (non-Hungarian) matching: an import is mentioned iff ANY of the document's
    mention names scores >= cutoff against it (same alias-group + rapidfuzz scoring as
    `align_software_names`, but no one-to-one assignment).
    """
    matched: set[str] = set()
    unique_mentions = set(pair_mentions)
    for name in set(pair_imports):
        for mention in unique_mentions:
            if are_alternates(name, mention):
                matched.add(name)
                break
            # Registry veto: both names known, in different groups -- fuzzy skipped.
            if are_known_distinct(name, mention):
                continue
            if fuzz.ratio(name, mention) >= cutoff:
                matched.add(name)
                break
    return matched


def build_import_mention_pair_library_frame(
    df: pl.DataFrame,
    imports: pl.DataFrame,
    mentions: pl.DataFrame,
    cutoff: float = 85.0,
    alignment: Literal["grouped_hungarian", "independent"] = "grouped_hungarian",
) -> pl.DataFrame:
    """
    Build the (document, repository, library) row-level frame that Figure 4, Table 1, and
    Unit 5's logistic regression all ultimately derive from: for every article-repository pair
    whose repository has >=1 extracted import, one row per unique imported library
    (import-normalized name, always canonical -- never the mention name), with `is_mentioned`
    set from the same per-pair, two-view Hungarian alignment
    (`align_software_names(..., method="global_min_diff")`, imports as `items_a`) used
    elsewhere in this replication package. Restricted to pairs whose repository has >=1 import,
    matching Figure 4's denominator choice -- pairs with zero mentions correctly contribute
    `is_mentioned=False` rows rather than being dropped or skipped.

    `alignment="independent"` drops the per-pair one-to-one assignment: each import is scored
    against every mention name independently, so one mention name can satisfy several imports.
    """
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible = df.filter(pl.col("repository_id").is_in(repo_with_import))
    print(
        f"Building import/mention long frame (alignment={alignment}): "
        f"{eligible.height:,} of {df.height:,} pairs have >=1 import"
    )

    imports_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in imports.group_by("repository_id")
    }
    mentions_by_doc: dict[int, list[str]] = {
        did[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for did, grp in mentions.group_by("document_id")
    }

    rows = []
    for row in eligible.select("document_id", "repository_id").unique().iter_rows(named=True):
        repo_id, doc_id = row["repository_id"], row["document_id"]
        pair_imports = imports_by_repo.get(repo_id, [])
        if not pair_imports:
            continue
        pair_mentions = mentions_by_doc.get(doc_id, [])
        matched_names: set[str] = set()
        if pair_mentions:
            if alignment == "grouped_hungarian":
                matches = align_software_names(
                    items_a=pair_imports,
                    items_b=pair_mentions,
                    source_a="import",
                    source_b="mention",
                    cutoff=cutoff,
                    method="global_min_diff",
                )
                matched_names = {m.normalized_item_one for m in matches}
            else:
                matched_names = _independent_matched_names(pair_imports, pair_mentions, cutoff)
        for name in set(pair_imports):
            rows.append(
                {
                    "document_id": doc_id,
                    "repository_id": repo_id,
                    "library_name_normalized": name,
                    "is_mentioned": name in matched_names,
                }
            )

    out = pl.DataFrame(rows)
    print(f"Built long frame: {out.height:,} (document, repository, library) rows")
    return out


###############################################################################
# Modified FWCI / FWSI
#
# OpenAlex's `fwci` field is computed using a fixed citation window relative to a work's
# publication date, not the work's full lifetime citation count. The manuscript's Methods
# describe a "modified FWCI" that substitutes lifetime citations in the numerator instead.
# We don't have OpenAlex's internal windowed-actual-citation figure stored anywhere in
# rs-graph (only the final `fwci` ratio and lifetime `cited_by_count` are persisted), so a
# true "same denominator, different numerator" reconstruction isn't possible from this data
# alone. Instead we rebuild the whole ratio from scratch, using lifetime counts on both sides:
#
#   modified_fwci = document_cited_by_count / mean(document_cited_by_count) within a
#                   (field, publication year, doctype bucket) peer group
#
# This mirrors what OpenAlex's FWCI construction is doing conceptually (a work's citations
# relative to its same-field/same-age/same-type peers) while using lifetime citation counts
# consistently for the paper itself *and* its peer-group baseline, rather than mixing a
# lifetime numerator with a windowed-baseline denominator we can't reconstruct. Modified FWSI
# is built the same way, symmetrically, using repository stargazer counts and repository
# creation year in place of citations and publication year, restricted to repositories at
# least `min_age_years` old (so young repos with little time to accumulate stars aren't
# compared against long-lived peers).
###############################################################################


def raw_fwci_docs(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Document-level frame (one row per document_id) carrying OpenAlex's own pre-computed FWCI
    (`document_fwci`, aliased to `document_raw_fwci`) plus the peer-group descriptor columns
    the Figure 3 panels join/group on. FWCI is only computed by OpenAlex for qualifying works,
    so the non-null share is reported for the Methods text.
    """
    docs = pairs_df.select(
        "document_id",
        pl.col("document_fwci").alias("document_raw_fwci"),
        "document_field_name_pruned",
        "document_publication_year",
        "document_type_bucket",
    ).unique(subset="document_id", keep="first")
    n_nonnull = docs.filter(pl.col("document_raw_fwci").is_not_null()).height
    print(
        f"Raw OpenAlex FWCI: {n_nonnull:,} of {docs.height:,} unique documents have a "
        f"non-null fwci ({100 * n_nonnull / docs.height:.1f}%)"
    )
    return docs


def compute_modified_fwci(
    pairs_df: pl.DataFrame,
    peer_group_cols: tuple[str, str, str] = (
        "document_field_name_pruned",
        "document_publication_year",
        "document_type_bucket",
    ),
) -> pl.DataFrame:
    """
    Compute modified FWCI (lifetime citations / peer-group mean lifetime citations) at the
    document level. Returns a document-level frame (one row per document_id) with a new
    `document_modified_fwci` column, ready to be joined back onto a pairs frame.
    """
    docs = pairs_df.select(
        "document_id",
        "document_cited_by_count",
        *peer_group_cols,
    ).unique(subset="document_id", keep="first")

    peer_means = docs.group_by(list(peer_group_cols)).agg(
        pl.mean("document_cited_by_count").alias("_peer_group_mean_citations"),
        pl.len().alias("_peer_group_n"),
    )

    docs = docs.join(peer_means, on=list(peer_group_cols), how="left").with_columns(
        pl.when(pl.col("_peer_group_mean_citations") > 0)
        .then(pl.col("document_cited_by_count") / pl.col("_peer_group_mean_citations"))
        .otherwise(None)
        .alias("document_modified_fwci")
    )

    print(
        f"Computed modified FWCI for {docs.height:,} documents across "
        f"{peer_means.height:,} peer groups "
        f"(grouping on {list(peer_group_cols)})"
    )

    return docs


def compute_modified_fwsi(
    pairs_df: pl.DataFrame,
    peer_group_cols: tuple[str, str, str] = (
        "document_field_name_pruned",
        "repository_creation_year",
        "document_type_bucket",
    ),
    min_age_years: float = 2.0,
    as_of: date | None = None,
) -> pl.DataFrame:
    """
    Compute modified FWSI (lifetime stargazers / peer-group mean lifetime stargazers) at the
    repository level, restricted to repositories >= `min_age_years` old as of `as_of` (defaults
    to today). A repository's field/doctype for peer-grouping purposes is taken from its first
    linked document-repository pair (repositories can link to more than one document; this
    picks one deterministically rather than double-counting the repository once per pair).
    """
    if as_of is None:
        as_of = date.today()

    select_cols = list(
        dict.fromkeys(
            ["repository_id", "repository_stargazers_count", "repository_creation_year"]
            + list(peer_group_cols)
        )
    )
    repos = pairs_df.select(select_cols).unique(subset="repository_id", keep="first")

    repos = repos.with_columns(
        (pl.lit(as_of.year) - pl.col("repository_creation_year")).alias("repository_age_years")
    )

    n_before_age_filter = repos.height
    repos = repos.filter(pl.col("repository_age_years") >= min_age_years)
    print(
        f"Restricting FWSI to repositories >= {min_age_years} years old (as of {as_of}): "
        f"{repos.height:,} of {n_before_age_filter:,} repositories remain"
    )

    peer_means = repos.group_by(list(peer_group_cols)).agg(
        pl.mean("repository_stargazers_count").alias("_peer_group_mean_stars"),
        pl.len().alias("_peer_group_n"),
    )

    repos = repos.join(peer_means, on=list(peer_group_cols), how="left").with_columns(
        pl.when(pl.col("_peer_group_mean_stars") > 0)
        .then(pl.col("repository_stargazers_count") / pl.col("_peer_group_mean_stars"))
        .otherwise(None)
        .alias("repository_modified_fwsi")
    )

    print(
        f"Computed modified FWSI for {repos.height:,} repositories across "
        f"{peer_means.height:,} peer groups "
        f"(grouping on {list(peer_group_cols)})"
    )

    return repos


###############################################################################
# Plotting support


def save_figure(fig, stem: str, output_dir: Path) -> None:
    """Save a figure as PNG, TIFF, and PDF to `output_dir` at 300 dpi."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_dir / stem} (.png/.tiff/.pdf)")


def save_table(df: pl.DataFrame, stem: str, output_dir: Path) -> None:
    """Save a polars DataFrame as CSV to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_dir / f"{stem}.csv")
    print(f"Saved table: {output_dir / stem}.csv ({df.height:,} rows)")


def shrink_ticks(ax, size: int = 8) -> None:
    """Shrink tick label font size -- evaplot's default 15pt tick labels are too large for
    multi-panel figures with long categorical labels (field/domain names).
    """
    ax.tick_params(axis="both", labelsize=size)


# dark2[0]/dark2[1] -- the same teal-green / burnt-orange pair evaplot's `set_cat_palette`
# hands Figures 2 and 3 when they slice to n=1/n=2. Anchoring Figure 4's multi-category
# palette on the same two colors keeps the whole figure set in one visual family instead of
# Figure 4 falling back to evaplot's full 8-hue dark2/vivid cycle (green/orange/purple/pink/
# olive/mustard/brown/gray), which read as a different paper's color scheme.
_FAMILY_GREEN = "#1b9e77"
_FAMILY_ORANGE = "#d95f02"


def field_palette(n: int) -> list[str]:
    """Build an n-color categorical palette anchored on the same green/orange family used
    throughout Figures 2 and 3, for figures (like Figure 4) that need more than two colors.
    """
    return list(sns.blend_palette([_FAMILY_GREEN, _FAMILY_ORANGE], n_colors=n))


# A second, distinct 2-color qualitative pair -- reserved for binary contrasts that sit directly
# beside a teal/orange-colored panel within the *same* figure, where reusing teal/orange would
# let a reader pattern-match one binary's meaning onto an unrelated one three panels over (round-3
# figure critique, priority item 2: Figure 3 Panel A's Before/After legend and Panel C's
# Testing/Linting legend previously shared the exact same two colors). Not applied paper-wide --
# the cross-figure reuse of teal/orange for other, non-adjacent binaries (article/preprint, Ours/
# PwC) is a separate, more structural question the critique flagged as Eva's call, not auto-fixed
# here.
_CONTRAST_BLUE = "#3b6fb6"
_CONTRAST_PURPLE = "#8456ce"
SECONDARY_BINARY_PALETTE: list[str] = [_CONTRAST_BLUE, _CONTRAST_PURPLE]

# A third, distinct 2-color qualitative pair -- for the article/preprint split in the Figure 3
# supplemental, which sits alongside (not within) the main Figure 3 that already uses teal/orange
# (Before/After, Panel A) and blue/purple (Testing/Linting, Panel C). Reusing either pair here
# would recreate the same cross-panel color-collision problem those two were split apart to fix,
# just one hop further out (round-4 figure critique, priority item 3).
_CONTRAST_GOLD = "#c9a227"
_CONTRAST_MAGENTA = "#c2438a"
TERTIARY_BINARY_PALETTE: list[str] = [_CONTRAST_GOLD, _CONTRAST_MAGENTA]


def style_legend(legend, fontsize: int = 8) -> None:
    """Give a legend a consistent bordered-box look. Standardizes across the figure set --
    some panels previously produced bare/unboxed legends (evaplot's `legend.frameon: False`
    rcParam default) while others looked boxed, an inconsistency flagged in figure review.
    """
    if legend is None:
        return
    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_edgecolor("#333333")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.9)
    for text in legend.get_texts():
        text.set_fontsize(fontsize)


def add_panel_label(ax, label: str) -> None:
    """Add a bold panel label (A, B, C...) to the upper-left corner of an axes."""
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


_FOOTNOTE_POSITIONS: dict[str, dict] = {
    "lower left": {"x": 0.02, "y": 0.02, "va": "bottom", "ha": "left"},
    "upper left": {"x": 0.02, "y": 0.98, "va": "top", "ha": "left"},
    "upper right": {"x": 0.98, "y": 0.98, "va": "top", "ha": "right"},
    "lower right": {"x": 0.98, "y": 0.02, "va": "bottom", "ha": "right"},
    # Outside the plotted data area entirely -- for panels where every corner is occupied by
    # data (dense multi-line plots, tall bars) and an in-panel box would sit on top of a mark.
    "top center outside": {"x": 0.5, "y": 1.13, "va": "bottom", "ha": "center"},
    "bottom center outside": {"x": 0.5, "y": -0.22, "va": "top", "ha": "center"},
    # Nudged further up/right than plain "upper right" -- for panels whose rightmost data label
    # (e.g. an n= annotation) sits close enough to the in-panel box's bottom-left corner at print
    # resolution to read as crowded (round-3 figure critique, priority item 5).
    "upper right outside": {"x": 1.0, "y": 1.1, "va": "bottom", "ha": "right"},
}


def add_footnote(ax, text: str, loc: str = "lower left") -> None:
    """Add a standardized in-panel caveat/footnote annotation -- one consistent light-gray
    boxed-italic style used everywhere a panel needs to flag an edge case (partial years,
    axis-range differences, taxonomy mismatches) instead of each panel inventing its own ad hoc
    treatment (round-2 figure critique, cross-figure item 3).
    """
    pos = _FOOTNOTE_POSITIONS[loc]
    ax.text(
        pos["x"],
        pos["y"],
        text,
        transform=ax.transAxes,
        fontsize=6.5,
        style="italic",
        va=pos["va"],
        ha=pos["ha"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#f2f2f2",
            "edgecolor": "#999999",
            "linewidth": 0.6,
            "alpha": 0.9,
        },
    )


def cap_ylim_to_quantiles(
    ax,
    series,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    pad_frac: float = 0.12,
    axis: Literal["x", "y"] = "y",
) -> tuple[float, float]:
    """Cap an axes' limits to a data quantile range rather than the full min/max -- fixes
    boxplot whisker compression, where a handful of long-tail whiskers stretch the axis so far
    that the interquartile boxes (the actual signal) get squeezed into a thin unreadable band
    near the bottom (round-2 figure critique, priority item 1). `axis="x"` applies the same cap
    to the x-axis for horizontal boxplots. Returns the applied (lo, hi) so the caller can
    report it in a footnote.
    """
    lo = float(series.quantile(lower_q))
    hi = float(series.quantile(upper_q))
    pad = (hi - lo) * pad_frac
    lo_padded, hi_padded = lo - pad, hi + pad
    if axis == "y":
        ax.set_ylim(lo_padded, hi_padded)
    else:
        ax.set_xlim(lo_padded, hi_padded)
    return lo_padded, hi_padded
