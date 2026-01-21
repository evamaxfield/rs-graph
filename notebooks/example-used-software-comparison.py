#!/usr/bin/env python

import shutil
from collections.abc import Callable
from pathlib import Path

import connectorx  # noqa: F401
import numpy as np
import polars as pl
import typer
from eil import Extractor  # type: ignore[import-not-found]
from git import GitCommandError, Repo
from git.remote import RemoteProgress
from nb_to_src import convert_directory as convert_nb_to_src_in_dir
from py_ascii_tree import ascii_tree
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from rs_graph.db import constants as db_constants

THIS_DIR = Path(__file__).parent.resolve()
TEMP_REPO_PATH = THIS_DIR / "temp-repo"

###############################################################################

app = typer.Typer()

###############################################################################


def _read_table(table: str) -> pl.DataFrame:
    return pl.read_database_uri(
        f"SELECT * FROM {table}",
        f"sqlite:///{db_constants.V2_DATABASE_PATHS.dev}",
    )


def _normalize_doi_expr(col_name: str) -> pl.Expr:
    """
    Create a Polars expression to normalize DOIs.

    Normalization:
    - Strip whitespace
    - Lowercase
    - Remove URL prefixes (https://doi.org/, http://dx.doi.org/, etc.)
    - Remove 'doi:' prefix
    """
    return (
        pl.col(col_name)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace(r"^https?://(dx\.)?doi\.org/", "")
        .str.replace(r"^doi:", "")
        .str.strip_chars()
    )


def load_pairs() -> pl.DataFrame:
    # Read all the tables we need
    dataset_sources = _read_table("dataset_source")
    docs = _read_table("document")
    repos = _read_table("repository")
    pairs = _read_table("document_repository_link")
    doc_topics = _read_table("document_topic")
    topics = _read_table("topic")
    # Drop to unique doc and unique repo in pairs
    pairs = pairs.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )

    # Create a table of document_authors
    # and their country institutional affiliations
    # then group by document to get to document_country_affiliation
    doc_contribs = _read_table("document_contributor")
    doc_contrib_institutions = _read_table("document_contributor_institution")
    institutions = _read_table("institution")

    doc_author_countries = (
        doc_contribs.select(
            pl.col("id").alias("document_contributor_id"),
            pl.col("researcher_id"),
            pl.col("document_id"),
        )
        .join(
            doc_contrib_institutions.select(
                "document_contributor_id",
                "institution_id",
            ),
            on="document_contributor_id",
            how="left",
        )
        .join(
            institutions.select(
                pl.col("id").alias("institution_id"),
                "country_code",
            ),
            on="institution_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("document_n_authors"),
            pl.when(pl.col("country_code").n_unique() == 1)
            .then(pl.col("country_code").get(0))
            .otherwise(pl.lit("Multiple")),
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
            .alias("country_code")
        )
    )

    # Get repo_contribs count
    repo_contribs = _read_table("repository_contributor")
    repo_contribs = repo_contribs.group_by("repository_id").len("repository_n_contributors")

    # Join the tables to get the positive examples
    return (
        pairs.select(
            "document_id",
            "repository_id",
            "dataset_source_id",
            pl.col("predictive_model_confidence").alias("document_repository_link_confidence"),
        )
        .join(
            docs.select(
                pl.col("id").alias("document_id"),
                pl.col("doi").alias("document_doi"),
                pl.col("cited_by_count").alias("document_cited_by_count"),
                pl.col("fwci").alias("document_fwci"),
                pl.col("is_open_access").alias("document_is_open_access"),
                pl.col("publication_date").alias("document_publication_date"),
            ).with_columns(_normalize_doi_expr("document_doi").alias("document_doi")),
            on="document_id",
            how="left",
        )
        .join(
            repos.select(
                pl.col("id").alias("repository_id"),
                pl.col("owner").alias("repository_owner"),
                pl.col("name").alias("repository_name"),
                pl.col("stargazers_count").alias("repository_stargazers_count"),
                pl.col("commits_count").alias("repository_commits_count"),
                pl.col("primary_language").alias("repository_primary_language"),
                pl.col("creation_datetime").alias("repository_creation_datetime"),
                pl.col("last_pushed_datetime").alias("repository_last_pushed_datetime"),
            ),
            on="repository_id",
            how="left",
        )
        .join(
            dataset_sources.select(
                pl.col("id").alias("dataset_source_id"),
                pl.col("name").alias("dataset_source_name"),
            ),
            on="dataset_source_id",
            how="left",
        )
        .join(
            doc_topics.sort(
                "score",
                descending=True,
            )
            .unique(
                "document_id",
                maintain_order=True,
            )
            .select(
                pl.col("document_id").alias("document_id"),
                pl.col("topic_id").alias("topic_id"),
            ),
            on="document_id",
            how="left",
        )
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("domain_name").alias("document_domain_name"),
                pl.col("field_name").alias("document_field_name"),
            ),
            on="topic_id",
            how="left",
        )
        .join(
            doc_author_countries,
            on="document_id",
            how="left",
        )
        .join(
            repo_contribs,
            on="repository_id",
            how="left",
        )
    ).filter(
        (pl.col("document_repository_link_confidence") >= 0.995)
        | (pl.col("document_repository_link_confidence").is_null())
    )


def normalize_name(name: str) -> str:
    """
    Normalize a software name for comparison.

    - Converts to lowercase
    - Removes hyphens, underscores, and spaces
    - Preserves alphanumeric characters and dots
    """
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def align_dependencies(
    imported: set[str],
    mentioned: set[str],
    score_cutoff: float = 70.0,
    scorer: Callable[[str, str], float] = fuzz.ratio,
) -> tuple[dict[str, tuple[str, float]], list[str], dict[str, str], dict[str, str]]:
    """
    Align imported dependencies with mentioned software names using optimal assignment.

    Uses the Hungarian algorithm to find the globally optimal matching that
    maximizes the total similarity score across all pairs.

    Args:
        imported: Set of imported dependency names (e.g., from code analysis)
        mentioned: Set of mentioned software names (e.g., from papers)
        score_cutoff: Minimum similarity score (0-100) to consider a match
        scorer: Scoring function from rapidfuzz.fuzz (default: fuzz.ratio)

    Returns:
        Tuple of:
        - matched: Mapping mentioned software (original) -> (imported dep original, score)
        - unmatched: List of imported dependencies (original) that weren't matched
        - imported_lut: Dict mapping original imported name -> normalized name
        - mentioned_lut: Dict mapping original mentioned name -> normalized name
    """
    if not imported or not mentioned:
        imported_lut = {orig: normalize_name(orig) for orig in imported}
        mentioned_lut = {orig: normalize_name(orig) for orig in mentioned}
        return {}, list(imported), imported_lut, mentioned_lut

    # Create lookup tables: original -> normalized
    imported_lut = {orig: normalize_name(orig) for orig in imported}
    mentioned_lut = {orig: normalize_name(orig) for orig in mentioned}

    # Convert to lists to maintain consistent ordering
    imported_list = list(imported)
    mentioned_list = list(mentioned)

    # Get normalized versions
    imported_norm = [imported_lut[imp] for imp in imported_list]
    mentioned_norm = [mentioned_lut[men] for men in mentioned_list]

    # Build cost matrix
    n_mentioned = len(mentioned_list)
    n_imported = len(imported_list)

    # Create a square cost matrix by padding with dummy entries if needed
    max_size = max(n_mentioned, n_imported)
    cost_matrix = np.full((max_size, max_size), -score_cutoff)

    # Fill in actual similarity scores (negated for minimization)
    for i, men_norm in enumerate(mentioned_norm):
        for j, imp_norm in enumerate(imported_norm):
            score = scorer(men_norm, imp_norm)
            cost_matrix[i, j] = -score

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract matches above threshold
    matched = {}
    used_imported_indices = set()

    for i, j in zip(row_ind, col_ind, strict=False):
        if i >= n_mentioned or j >= n_imported:
            continue

        score = -cost_matrix[i, j]

        if score >= score_cutoff:
            mention_orig = mentioned_list[i]
            import_orig = imported_list[j]
            matched[mention_orig] = (import_orig, score)
            used_imported_indices.add(j)

    # Find unmatched imports
    unmatched = [imported_list[j] for j in range(n_imported) if j not in used_imported_indices]

    return matched, unmatched, imported_lut, mentioned_lut


class _TqdmProgress(RemoteProgress):
    """Progress handler for git clone operations with tqdm integration."""

    def __init__(self, desc: str):
        super().__init__()
        self._tqdm = tqdm(unit="objects", desc=desc, leave=False)

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ) -> None:
        try:
            cur = int(cur_count or 0)
            if max_count and int(max_count) > 0:
                total = int(max_count)
                if self._tqdm.total is None or self._tqdm.total < total:
                    self._tqdm.total = total
                self._tqdm.update(cur - self._tqdm.n)
            else:
                self._tqdm.update(cur - self._tqdm.n)
            if message:
                self._tqdm.set_postfix_str(str(message))
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tqdm.clear()
        self._tqdm.close()
        return False


@app.command()
def get_imported_libraries(
    softcite_dataset_dir: str,
    sample_n: int = 50,
    results_path: str = str(THIS_DIR / "used-software.parquet"),
    cache_every_n: int = 1,
    debug: bool = False,
) -> None:
    # Load softcite papers to filter to only repos with known mentions
    softcite_dir = Path(softcite_dataset_dir)
    softcite_papers_path = softcite_dir / "papers.parquet"
    softcite_papers_df = pl.read_parquet(softcite_papers_path)

    # Normalize softcite DOIs
    softcite_papers_df = softcite_papers_df.with_columns(
        _normalize_doi_expr("doi").alias("doi")
    )
    softcite_dois = softcite_papers_df.select("doi").drop_nulls().unique()["doi"].to_list()
    print(f"Found {len(softcite_dois)} unique DOIs in softcite dataset")

    # Load the pairs data and normalize DOIs
    df = load_pairs()

    # Filter to only documents with softcite mentions
    df = df.filter(pl.col("document_doi").is_in(softcite_dois))
    print(f"Filtered to {df.height} document-repository pairs with softcite mentions")

    # Filter to repositories with primary language of "Python", "Jupyter Notebook", or "R"
    df = df.filter(
        pl.col("repository_primary_language").is_in(["Python", "Jupyter Notebook", "R"])
    )
    print(f"Filtered to {df.height} pairs with target programming languages")

    # Sample
    df_sample = df.sample(n=min(sample_n, df.height), seed=12)

    # Clean prior temp repo
    if TEMP_REPO_PATH.exists():
        shutil.rmtree(TEMP_REPO_PATH)

    # Iter through, clone, convert, eil
    results = []
    for row_dict in tqdm(
        df_sample.iter_rows(named=True),
        total=df_sample.height,
    ):
        # Get the repo full name
        repo_full_name = f"{row_dict['repository_owner']}/{row_dict['repository_name']}"

        # Clone the repo using GitPython with a progress bar
        try:
            with _TqdmProgress(desc=f"Cloning {repo_full_name}") as progress:
                Repo.clone_from(
                    f"https://github.com/{repo_full_name}.git",
                    str(TEMP_REPO_PATH),
                    progress=progress,
                )

            # Print tree of files
            if debug:
                print(f"Repository: {repo_full_name}")
                all_repo_files = [
                    str(p.relative_to(TEMP_REPO_PATH)) for p in TEMP_REPO_PATH.glob("**/*")
                ]
                print(ascii_tree(all_repo_files, max_depth=3, max_files_per_dir=3))
                print()

            # Convert notebooks to source code
            convert_nb_to_src_in_dir(
                TEMP_REPO_PATH,
                recursive=True,
                progress_leave=False,
            )

            # Extract imported libraries using EIL
            extractor = Extractor()
            extracted_lib_results = extractor.extract_from_directory(
                TEMP_REPO_PATH,
                recursive=True,
                progress_leave=False,
            )

            # Condense third party libraries
            third_party_libs = list(
                {
                    lib
                    for _file_path, imported_libs_result in extracted_lib_results.extracted.items()
                    for lib in imported_libs_result.third_party
                }
            )

            # Store as long format
            for lib in third_party_libs:
                results.append(
                    {
                        **row_dict,
                        "imported_library": lib,
                    }
                )

            # If no third party libs found, still record the repo with no imports
            if not third_party_libs:
                if debug:
                    print(f"No third-party libraries found in {repo_full_name}")
                    print(extracted_lib_results)

                results.append(
                    {
                        **row_dict,
                        "imported_library": None,
                    }
                )

            # Cache every n
            if len(results) % cache_every_n == 0:
                pl.DataFrame(results).write_parquet(results_path)

        except GitCommandError as e:
            print(f"Failed to clone repository: {repo_full_name} ({e})")
            continue

        finally:
            # Clean up the temp repo
            if TEMP_REPO_PATH.exists():
                shutil.rmtree(TEMP_REPO_PATH)


@app.command()
def compare_imported_vs_mentioned(
    softcite_dataset_dir: str,
    used_software_path: str = str(THIS_DIR / "used-software.parquet"),
    output_path: str = str(THIS_DIR / "comparison-results.parquet"),
    score_cutoff: float = 75.0,
    debug: bool = False,
) -> None:
    """
    Compare imported libraries from code analysis with software mentioned in papers.

    Args:
        softcite_dataset_dir: Path to directory containing softcite dataset files (papers.parquet, etc.)
        used_software_path: Path to the parquet file with imported libraries
        output_path: Path to save comparison results
        score_cutoff: Minimum similarity score (0-100) for matching
        debug: Whether to print detailed debug information
    """
    # Load the imported libraries data
    used_software_df = pl.read_parquet(used_software_path)
    print(
        f"Loaded {used_software_df.height} imported library records from "
        f"{used_software_df.n_unique(subset=['repository_owner', 'repository_name'])} "
        f"repositories"
    )

    # Load software mentions from softcite dataset and normalize DOIs
    softcite_dir = Path(softcite_dataset_dir)
    softcite_papers_path = softcite_dir / "papers.parquet"
    softcite_mentions_path = softcite_dir / "mentions.pdf.parquet"
    softcite_papers_df = pl.read_parquet(softcite_papers_path)
    softcite_papers_df = softcite_papers_df.with_columns(
        _normalize_doi_expr("doi").alias("doi")
    )

    softcite_mentions_df = pl.scan_parquet(softcite_mentions_path)
    softcite_mentions_df = (
        softcite_mentions_df.select(
            pl.col("software_mention_id").str.strip_chars().alias("softcite_mention_id"),
            pl.col("paper_id").alias("softcite_paper_id"),
            pl.col("software_raw").alias("softcite_software_mention_raw"),
        )
        .collect()
        .join(
            softcite_papers_df.select(
                pl.col("paper_id").alias("softcite_paper_id"),
                pl.col("doi").alias("softcite_paper_doi"),
            ).filter(
                pl.col("softcite_paper_doi").is_in(
                    used_software_df["document_doi"].unique().to_list()
                )
            ),
            on="softcite_paper_id",
            how="right",
        )
        .drop_nulls()
    )

    print(
        f"Found {softcite_mentions_df.height} software mentions from "
        f"{softcite_mentions_df['softcite_paper_id'].n_unique()} relevant papers"
    )

    if debug:
        print()
        print("-" * 80)
        print()

    # Get all DOIs from used_software_df (includes repos without softcite mentions)
    used_dois = used_software_df["document_doi"].unique().to_list()

    # Iter over DOIs get subset of each dataframe and align
    results = []
    for doi in tqdm(
        used_dois,
        desc="Comparing imported vs mentioned",
    ):
        # Subset data
        repo_subset = used_software_df.filter(pl.col("document_doi") == doi)
        mention_subset = softcite_mentions_df.filter(pl.col("softcite_paper_doi") == doi)

        # Get repository info for this DOI
        repo_owner = repo_subset["repository_owner"][0]
        repo_name = repo_subset["repository_name"][0]

        # Get imported libraries from the FULL used_software_df
        imported_set = {v for v in repo_subset["imported_library"].to_list() if v is not None}
        mentioned_set = set(mention_subset["softcite_software_mention_raw"].to_list())

        # Align
        matched, unmatched, _, _ = align_dependencies(
            imported=imported_set,
            mentioned=mentioned_set,
            score_cutoff=score_cutoff,
        )
        matched_names = set(matched.keys())

        # Store results per repo
        article_doi = doi
        n_imported = len(imported_set)
        n_mentioned = len(mentioned_set)
        n_matched = len(matched)
        n_unmatched_imported = len(unmatched)
        total_score = sum(score for _, score in matched.values())
        avg_score = total_score / n_matched if n_matched > 0 else 0.0
        median_score = (
            np.median([score for _, score in matched.values()]) if n_matched > 0 else 0.0
        )

        # Show the whole match lut
        if debug:
            print()
            print(f"Article DOI: {article_doi}")
            print(f"Repository URL: https://github.com/{repo_owner}/{repo_name}")
            print(f"Imported libraries ({n_imported}): {imported_set}")
            print(f"Mentioned software ({n_mentioned}): {mentioned_set}")
            print(f"Matched ({n_matched}): {matched_names}")
            print(f"Unmatched imported ({n_unmatched_imported}): {unmatched}")
            print()

        results.append(
            {
                **repo_subset.drop("imported_library").row(0, named=True),
                "n_imported": n_imported,
                "n_mentioned": n_mentioned,
                "n_matched": n_matched,
                "n_unmatched_imported": n_unmatched_imported,
                "avg_match_score": avg_score,
                "median_match_score": median_score,
                "imported_libraries": [
                    f"<imported-library>{name}</imported-library>" for name in imported_set
                ],
                "mentioned_software": [
                    f"<mentioned-software>{name}</mentioned-software>" for name in mentioned_set
                ],
                "matched_imports_to_mentions": [
                    f"<matched-imported-to-mentioned-software>{name}</matched-imported-to-mentioned-software>"
                    for name in matched_names
                ],
            }
        )

    # Save results
    results_df = pl.DataFrame(results)
    results_df.write_parquet(output_path)
    if debug:
        print()
        print("-" * 80)
        print()

    # Print summary
    print("Summary of comparison results:")
    print(f"Total repositories analyzed: {results_df.height}")
    print(f"Average number of imported libraries: {results_df['n_imported'].mean():.2f}")
    print(f"Average number of mentioned software: {results_df['n_mentioned'].mean():.2f}")
    print(f"Average number of matched software: {results_df['n_matched'].mean():.2f}")
    comparisons_with_both_imports_and_mentions = results_df.filter(
        (pl.col("n_imported") > 0) & (pl.col("n_mentioned") > 0)
    )
    print(
        f"Average matched software (repos with both imports and mentions): {comparisons_with_both_imports_and_mentions['n_matched'].mean():.2f}"
    )
    print(
        f"Average match score: {results_df.filter(pl.col('avg_match_score') > 0)['avg_match_score'].mean():.2f}"
    )
    print(
        f"Median match score: {results_df.filter(pl.col('median_match_score') > 0)['median_match_score'].mean():.2f}"
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
