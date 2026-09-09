#!/usr/bin/env python3

"""Shared loading, filtering, and plotting-support utilities for the Nature Computational
Science `rs-graph` figures and tables. Every figure/table function loads its data through
`load_table` / `load_filtered_pairs` below rather than a local database, so each command is
runnable standalone directly from the published HuggingFace dataset.
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

OUTPUT_DIR = THIS_DIR / "outputs"

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

# Dependency-manifest ecosystems considered part of each language bucket -- restricts
# import-vs-dependency alignment to the manifests that belong to that language, rather than
# e.g. matching an npm frontend-tooling dependency declared inside a Python repo.
MANIFEST_ECOSYSTEMS_BY_LANGUAGE: dict[str, list[str]] = {
    "Python": ["pypi", "conda"],
    "R": ["cran"],
}
ALL_MANIFEST_ECOSYSTEMS: list[str] = sorted(
    {e for ecosystems in MANIFEST_ECOSYSTEMS_BY_LANGUAGE.values() for e in ecosystems}
)

# Mention extraction is absent/partial after this publication year (see the
# mentions-coverage diagnostic), so mention-dependent analyses cap at it.
MENTION_EXTRACTION_YEAR_CAP = 2022

###############################################################################
# Loading


def load_table(table: str) -> pl.DataFrame:
    """Load a single rs-graph table directly from HuggingFace as a polars DataFrame."""
    ds = load_dataset(HF_DATASET, table, split="train", token=os.environ.get("HF_TOKEN"))
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def normalized_names_by_id(frame: pl.DataFrame, id_column: str) -> dict[int, list[str]]:
    """Group non-null `software_name_normalized` values into a list per `id_column` value."""
    return {
        key[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for key, grp in frame.group_by(id_column)
    }


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

    Filtering order:
      1. article-repository pair confidence >= `confidence_threshold`, or NULL.
      2. published after `min_year` (GitHub's launch year, 2008, by default).
      3. researcher-developer identity confidence >= `rdal_confidence_threshold`, only when
         `apply_researcher_developer_filter=True` (off by default; most figures never touch
         researcher/developer-account identity).
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
# Dependency-name cleaning

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
    pypi/conda/cran rows only: strip a leading conda channel prefix
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


def clean_mention_names(mentions: pl.DataFrame) -> pl.DataFrame:
    """
    Clean residual punctuation out of mention `software_name_normalized` -- the mention-side
    analog of `clean_dependency_names`: truncate at the first comparator/marker/junk
    character (e.g. `mgcv()` -> `mgcv`), strip leading/trailing dots (e.g. `scipy.` ->
    `scipy`), and drop rows whose cleaned name is empty or doesn't start with [a-z0-9].
    """
    n_before = mentions.height
    n_changed_candidates = mentions.filter(
        pl.col("software_name_normalized").str.contains(r"[^a-z0-9.+-]")
        | pl.col("software_name_normalized").str.contains(r"^\.|\.$")
    ).height

    cleaned = mentions.with_columns(
        pl.col("software_name_normalized")
        .str.replace(_DEP_NAME_TRUNCATE_PATTERN, "")
        .str.strip_chars(".")
        .alias("software_name_normalized")
    ).filter(pl.col("software_name_normalized").str.contains(r"^[a-z0-9]"))
    print(
        f"Mention-name cleaning: {n_changed_candidates:,} rows carried residual punctuation; "
        f"{n_before - cleaned.height:,} rows dropped as empty/pure junk after cleaning; "
        f"{cleaned.height:,} of {n_before:,} rows remain"
    )
    return cleaned


def format_p_value(p: float) -> str:
    """Format a p-value for a published table; below-float-underflow values print as a bound
    rather than a literal 0.0.
    """
    return "< 1e-300" if p < 1e-300 else f"{p:.3g}"


###############################################################################
# Import-vs-mention long frame (shared by Figure 4, Table 1, and the mention-predictors
# regression)


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
    the mention-predictors logistic regression all derive from: for every article-repository
    pair whose repository has >=1 extracted import, one row per unique imported library
    (import-normalized name, always canonical -- never the mention name), with `is_mentioned`
    set from the per-pair, two-view Hungarian alignment
    (`align_software_names(..., method="global_min_diff")`, imports as `items_a`).
    Pairs with zero mentions contribute `is_mentioned=False` rows rather than being dropped.

    `alignment="independent"` drops the per-pair one-to-one assignment: each import is scored
    against every mention name independently, so one mention name can satisfy several imports.
    """
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible = df.filter(pl.col("repository_id").is_in(repo_with_import))
    print(
        f"Building import/mention long frame (alignment={alignment}): "
        f"{eligible.height:,} of {df.height:,} pairs have >=1 import"
    )

    imports_by_repo = normalized_names_by_id(imports, "repository_id")
    mentions_by_doc = normalized_names_by_id(mentions, "document_id")

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
# OpenAlex's `fwci` uses a fixed citation window; rs-graph only stores the final ratio and
# lifetime `cited_by_count`, so the windowed figure can't be reconstructed. Instead the whole
# ratio is rebuilt from scratch using lifetime counts on both sides:
#
#   modified_fwci = document_cited_by_count / mean(document_cited_by_count) within a
#                   (field, publication year, doctype bucket) peer group
#
# Modified FWSI is built the same way, symmetrically, using repository stargazer counts and
# repository creation year, restricted to repositories at least `min_age_years` old (so young
# repos with little time to accumulate stars aren't compared against long-lived peers).
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
            [
                "repository_id",
                "repository_stargazers_count",
                "repository_creation_year",
                *peer_group_cols,
            ]
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
    """Save a figure as PNG, TIFF (LZW-compressed), and PDF to `output_dir` at 300 dpi."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "pdf"):
        # LZW is lossless and cuts TIFF sizes ~5-10x.
        extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", **extra)
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
# hands general statistical comparisons at n=1/n=2; multi-category general palettes are
# anchored on these two colors so those panels stay in one visual family.
_FAMILY_GREEN = "#1b9e77"
_FAMILY_ORANGE = "#d95f02"


def general_palette(n: int) -> list[str]:
    """Build an n-color categorical palette anchored on the green/orange family used for
    general statistical comparisons (tooling categories, licenses, manifest bands).
    """
    return list(sns.blend_palette([_FAMILY_GREEN, _FAMILY_ORANGE], n_colors=n))


# One shared field-to-color assignment across every per-field figure: seaborn's colorblind
# palette, assigned in global pair-count prevalence order, with "Other" always gray. These
# colors are reserved for differentiating fields and nothing else.
FIELD_OTHER_COLOR = "#bbbbbb"


def field_color_map(fields: list[str]) -> dict[str, str]:
    """Map field names (in global prevalence order) to the shared colorblind field palette;
    "Other" maps to gray.
    """
    named = [f for f in fields if f != "Other"]
    palette = sns.color_palette("colorblind", n_colors=len(named)).as_hex()
    colors = dict(zip(named, palette, strict=True))
    colors["Other"] = FIELD_OTHER_COLOR
    return colors


# One shared color per data view (mentions / imports / manifest dependencies), distinct from
# both the field palette and the teal/orange general-comparison family.
DATA_VIEW_COLORS: dict[str, str] = {
    "mentions": "#c2438a",
    "imports": "#3b6fb6",
    "dependencies": "#8456ce",
}

# A distinct 2-color qualitative pair for the article/preprint split in the Figure 3
# supplemental, so this unrelated binary isn't pattern-matched onto the teal/orange binaries
# in the main figure.
_CONTRAST_GOLD = "#c9a227"
_CONTRAST_MAGENTA = "#c2438a"
TERTIARY_BINARY_PALETTE: list[str] = [_CONTRAST_GOLD, _CONTRAST_MAGENTA]


def style_legend(legend, fontsize: int = 8) -> None:
    """Give a legend a consistent bordered-box look across the figure set (evaplot's
    `legend.frameon: False` rcParam default otherwise leaves some legends unboxed).
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


def print_caption_note(figure_stem: str, text: str) -> None:
    """Print a caption-bound note (title or caveat) for a figure, for copying into the
    manuscript -- in-figure titles and footnote boxes are not used; captions carry them.
    """
    print(f"Caption ({figure_stem}): {text}")


def cap_ylim_to_quantiles(
    ax,
    series,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    pad_frac: float = 0.12,
    axis: Literal["x", "y"] = "y",
) -> tuple[float, float]:
    """Cap an axes' limits to a data quantile range rather than the full min/max, so a handful
    of long-tail boxplot whiskers can't squeeze the interquartile boxes into an unreadable
    band. `axis="x"` applies the cap to the x-axis for horizontal boxplots. Returns the
    applied (lo, hi) so the caller can report it in a footnote.
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
