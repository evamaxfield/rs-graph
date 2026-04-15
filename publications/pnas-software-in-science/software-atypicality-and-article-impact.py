import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import torch
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from tqdm import tqdm

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "software-atypicality-and-article-impact"

###############################################################################


# Helper to load a table as a polars DataFrame (zero-copy via Arrow)
def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset("evamxb/rs-graph-v2-full", table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _load_our_dataset(
    top_n_fields: int = 5,
) -> pl.DataFrame:
    # Load all pair info
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")

    # Create dataframe of document_id,
    # document_author_count, and document_author_mean_citations
    document_contributors = document_contributors.join(
        researchers.select(
            pl.col("id").alias("researcher_id"),
            pl.col("cited_by_count").alias("researcher_cited_by_count"),
        ),
        on="researcher_id",
        how="left",
    )
    document_author_stats = (
        document_contributors.group_by("document_id")
        .agg(
            pl.count("researcher_id").alias("document_author_count"),
            pl.mean("researcher_cited_by_count").alias("document_author_mean_citations"),
        )
        .with_columns(
            pl.col("document_author_mean_citations")
            .log1p()
            .alias("document_log_author_mean_citations")
        )
    )

    # Sort document topics by score (descending)
    # Drop duplicated by document_id to get the top topic for each document
    # Join topic info to get the field and domain name
    document_topics = document_topics.sort("score", descending=True)
    document_topics = document_topics.unique(subset="document_id", keep="first")
    document_topics = (
        document_topics.select(
            pl.col("document_id"),
            pl.col("topic_id"),
        )
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select(
            pl.col("document_id"),
            pl.col("document_field_name").alias("document_field_name"),
            pl.col("document_domain_name").alias("document_domain_name"),
        )
    )

    # Construct the merged dataframe with all basic details
    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
            pl.col("dataset_source_id"),
            pl.col("predictive_model_confidence"),
        )
        .join(
            documents.select(
                *[pl.col(col).alias(f"document_{col}") for col in documents.columns]
            ),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(col).alias(f"repository_{col}") for col in repositories.columns]
            ),
            on="repository_id",
        )
        .join(
            document_topics,
            on="document_id",
        )
        .join(
            document_author_stats,
            on="document_id",
        )
    )

    # Create document publication year column as integer (extract year from date)
    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
    )

    # Filter to only pairs published after 2008 (the year GitHub was founded)
    merged = merged.filter(pl.col("document_publication_year") >= 2008).with_columns(
        (pl.col("document_publication_year") - pl.col("document_publication_year").min()).alias(
            "document_years_since_earliest"
        )
    )

    # Reduce to only pairs with confidence of 0.9994
    merged = merged.filter(
        (pl.col("predictive_model_confidence") > 0.9994)
        | (pl.col("predictive_model_confidence").is_null())
    )

    # Create a "document_field_name_pruned" column
    # that takes the top N most common field names, and then labels the rest as "other"
    top_n_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields)
        .get_column("document_field_name")
        .to_list()
    )
    merged = merged.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_n_field_names))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    return merged


def _get_imported_software(
    repository_ids: list[int],
    remove_extremely_rare_imports: bool = True,
    rare_import_threshold: int = 5,
) -> pl.DataFrame:
    # Load all import data
    repository_imports = load_table("repository_import")

    # Filter to only the repositories in our dataset
    repository_imports = repository_imports.filter(
        pl.col("repository_id").is_in(repository_ids)
    )

    # Create column called "ecosystem_normalized_software_name"
    # Which prepends "py:", "r:", "mixed:" to the "software_name_normalized" column
    # based on the "file_paths" column
    # The "file_paths" column is a semi-colon separated list of file paths

    # Pre-construct the py and r checks
    check_for_py = (
        pl.col("file_paths_lower").str.contains(r"\.py\;")
        | pl.col("file_paths_lower").str.contains(r"\.py$")
        | pl.col("file_paths_lower").str.contains(r"\.ipynb\;")
        | pl.col("file_paths_lower").str.contains(r"\.ipynb$")
    )
    check_for_r = (
        pl.col("file_paths_lower").str.contains(r"\.r\;")
        | pl.col("file_paths_lower").str.contains(r"\.r$")
        | pl.col("file_paths_lower").str.contains(r"\.rmd\;")
        | pl.col("file_paths_lower").str.contains(r"\.rmd$")
    )

    # Now compute ecosystem and ecosystem normalized software name
    repository_imports = (
        repository_imports.with_columns(
            pl.col("file_paths").str.to_lowercase().alias("file_paths_lower")
        )
        .with_columns(
            # Check contains at least one .py or .ipynb
            # AND at least one .r or .rmd to determine if it's mixed
            pl.when(check_for_py & check_for_r)
            .then(pl.lit("mixed"))
            # Check py and ipynb
            .when(check_for_py)
            .then(pl.lit("py"))
            # Check r and rmd
            .when(check_for_r)
            .then(pl.lit("r"))
            # Other
            .otherwise(pl.lit("other"))
            .alias("ecosystem")
        )
        .with_columns(
            (pl.col("ecosystem") + pl.lit(":") + pl.col("software_name_normalized")).alias(
                "ecosystem_normalized_software_name"
            )
        )
    )

    # Drop "mixed" and "other" ecosystems
    repository_imports = repository_imports.filter(pl.col("ecosystem").is_in(["py", "r"]))

    # Remove any imports that were imported less than 3 times across the entire dataset (to remove noise)
    if remove_extremely_rare_imports:
        # Count imports
        import_counts = repository_imports.group_by("ecosystem_normalized_software_name").agg(
            pl.len().alias("import_count")
        )

        # Filter to only imports that were imported at least 3 times
        pre_filter_unique_package_count = repository_imports.get_column(
            "ecosystem_normalized_software_name"
        ).n_unique()
        non_rare_imports = (
            import_counts.filter(pl.col("import_count") >= rare_import_threshold)
            .get_column("ecosystem_normalized_software_name")
            .to_list()
        )

        # Filter repository imports to only non-rare imports
        repository_imports = repository_imports.filter(
            pl.col("ecosystem_normalized_software_name").is_in(non_rare_imports)
        )
        post_filter_unique_package_count = repository_imports.get_column(
            "ecosystem_normalized_software_name"
        ).n_unique()

        # Log how many unique packages were removed by this filter
        print(
            f"Removed {pre_filter_unique_package_count - post_filter_unique_package_count} unique packages"
        )
        print(
            f"Number of remaining unique imported packages: {post_filter_unique_package_count}"
        )

    return repository_imports


def _construct_article_to_software_mapping(
    our_dataset: pl.DataFrame,
    repository_imports: pl.DataFrame,
) -> dict[int, set[str]]:
    # Join document id to repository imports
    repository_imports = repository_imports.join(
        our_dataset.select(
            pl.col("document_id"),
            pl.col("repository_id"),
        ),
        on="repository_id",
        how="inner",
    )

    # Now group by document_id and aggregate software into sets
    article_to_software_df = repository_imports.group_by("document_id").agg(
        pl.col("ecosystem_normalized_software_name").unique().alias("imported_software")
    )

    # Convert article_to_software to a dict of document_id to set of software
    article_to_software_dict = {
        row["document_id"]: set(row["imported_software"])
        for row in article_to_software_df.iter_rows(named=True)
    }

    return article_to_software_dict


def _create_article_vecs(
    article_to_software_mapping: dict[int, set[str]],
    article_id_to_index: dict[int, int],
    software_name_to_index: dict[str, int],
    total_articles: int,
    total_software_names: int,
) -> np.ndarray:
    """
    Create the article vectors for each software based on the article to software mapping.

    Each software gets a binary vector of length num_papers.
    software_article_vecs[s, p] = 1 if software s is used in paper p, else 0.

    This is the "software co-usage" representation: software that tend to appear
    in the same papers will have similar vectors, and thus high cosine similarity.
    """
    software_article_vecs = np.zeros((total_software_names, total_articles), dtype=float)

    for document_id, software_used in article_to_software_mapping.items():
        article_index = article_id_to_index[document_id]
        for software in software_used:
            software_index = software_name_to_index[software]
            software_article_vecs[software_index, article_index] = 1.0

    return software_article_vecs


def _compute_pairwise_cosine_similarity(
    software_article_vecs: np.ndarray,
    all_software_names: list[str],
    log_software_pair_examples: bool = True,
) -> np.ndarray:
    # Compute pairwise cosine similarity between software article vectors
    # High similarity means software i and j are often used together in the same papers,
    # thus they are a conventional pairing.
    # Low similarity means software i and j are rarely used together,
    # thus they are an unconventional pairing.
    # Move the dataset-article matrix to a torch tensor (on GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    software_articles_vectors_tensor = torch.tensor(
        software_article_vecs, dtype=torch.float32, device=device
    )

    # Normalize each row (each dataset's article vector) to unit length
    row_norms = software_articles_vectors_tensor.norm(dim=1, keepdim=True)

    # Avoid division by zero for datasets that appear in zero papers
    row_norms = row_norms.clamp(min=1e-8)
    normalized_vectors = software_articles_vectors_tensor / row_norms

    # Cosine similarity = dot product of unit vectors
    # This is a single matrix multiply: (num_datasets x num_papers) @ (num_papers x num_datasets)
    pairwise_software_cosine_similarity_tensor = normalized_vectors @ normalized_vectors.T

    # If you need it back as numpy for downstream work:
    pairwise_software_cosine_similarity = (
        pairwise_software_cosine_similarity_tensor.cpu().numpy()
    )

    # Take a random sample of 20 non-zero similarity pairs and
    # print them out with their software names for qualitative inspection
    if log_software_pair_examples:
        non_zero_indices = np.argwhere(pairwise_software_cosine_similarity > 0)
        sampled_indices = non_zero_indices[
            np.random.choice(non_zero_indices.shape[0], size=20, replace=False)
        ]
        print()
        print("Sample of software pairs with non-zero cosine similarity:")
        for i, j in sampled_indices:
            software_i = all_software_names[i]
            software_j = all_software_names[j]
            similarity_score = pairwise_software_cosine_similarity[i, j]
            print(f"{software_i} - {software_j}: {similarity_score:.4f}")

    return pairwise_software_cosine_similarity


def _compute_article_atypicality(
    article_to_software_mapping: dict[int, set[str]],
    software_name_to_index: dict[str, int],
    pairwise_dataset_cosine_similarity: np.ndarray,
) -> dict[int, float | None]:
    """
    Compute article atypicality based on the article to software mapping and the
    pairwise dataset cosine similarity matrix.
    """
    article_atypicality_scores: dict[int, float | None] = {}

    for document_id, software_names in article_to_software_mapping.items():
        count_unique_software_used = len(software_names)

        if count_unique_software_used < 2:
            # Atypicality is only meaningful for papers using 2+ datasets.
            # (With 1 dataset, the only pair is (i,i) with D_ii=1,
            #  giving atypicality = 1 - 1/1 * 1 = 0, which is trivial.)
            article_atypicality_scores[document_id] = None
            continue

        # Get the row indices for each dataset used in this paper
        software_indices_in_article = [
            software_name_to_index[software] for software in software_names
        ]

        # Extract the sub-matrix of D_ij for just the datasets in this paper.
        # This is an (N_c x N_c) matrix of all pairwise similarities.
        similarity_submatrix = pairwise_dataset_cosine_similarity[
            np.ix_(software_indices_in_article, software_indices_in_article)
        ]

        # Eq. [2]: Atypicality = 1 - (1/N_c^2) * sum of all D_ij
        sum_of_all_pairwise_similarities = similarity_submatrix.sum()
        atypicality = 1.0 - (sum_of_all_pairwise_similarities / (count_unique_software_used**2))

        article_atypicality_scores[document_id] = atypicality

    return article_atypicality_scores


def _compute_document_atypicality_for_ecosystem(
    ecosystem_label: str,
    ecosystem_specific_article_to_software_mapping: dict[int, set[str]],
    pair_metadata: pl.DataFrame,
    log_software_pair_examples: bool = True,
) -> pl.DataFrame:
    # Get total counts and construct document id to index and software name to index mappings
    # for downstream matrix construction
    all_document_ids = sorted(ecosystem_specific_article_to_software_mapping.keys())
    all_software_names = sorted(
        {
            software
            for software_list in ecosystem_specific_article_to_software_mapping.values()
            for software in software_list
        }
    )

    # Get counts
    total_documents = len(all_document_ids)
    total_software_names = len(all_software_names)

    # Create LUTs
    document_id_to_index = {pid: idx for idx, pid in enumerate(all_document_ids)}
    software_name_to_index = {name: idx for idx, name in enumerate(all_software_names)}

    # Create the article vectors for each software
    software_article_vecs = _create_article_vecs(
        ecosystem_specific_article_to_software_mapping,
        document_id_to_index,
        software_name_to_index,
        total_documents,
        total_software_names,
    )

    # Compute pairwise cosine similarity between software
    pairwise_software_cosine_similarity = _compute_pairwise_cosine_similarity(
        software_article_vecs,
        all_software_names,
        log_software_pair_examples=log_software_pair_examples,
    )

    # Compute article atypicality scores
    article_atypicality_scores = _compute_article_atypicality(
        ecosystem_specific_article_to_software_mapping,
        software_name_to_index,
        pairwise_software_cosine_similarity,
    )

    # Z-score the atypicality scores (ignoring None values)
    atypicality_values = np.array(
        [score for score in article_atypicality_scores.values() if score is not None]
    )
    atypicality_mean = atypicality_values.mean()
    atypicality_std = atypicality_values.std()
    article_atypicality_scores_zscored = {
        doc_id: (score - atypicality_mean) / atypicality_std if score is not None else None
        for doc_id, score in article_atypicality_scores.items()
    }

    # Merge this data with the original details
    atypicality_df = pl.DataFrame(
        {
            "document_id": list(article_atypicality_scores_zscored.keys()),
            "document_atypicality_score": list(article_atypicality_scores.values()),
            "document_atypicality_z_score": list(article_atypicality_scores_zscored.values()),
        }
    )
    results_df = atypicality_df.join(pair_metadata, on="document_id", how="left")

    # Add in a column for the ecosystem label
    results_df = results_df.with_columns(pl.lit(ecosystem_label).alias("ecosystem_label"))

    return results_df


def _plot_coefficient_forest(
    summary_df: pl.DataFrame,
    model_types: tuple[str, str],
    output_filename: str,
    title: str,
) -> None:
    """Forest plot of atypicality coefficients across ecosystems, faceted by outcome."""
    outcomes = [
        (
            "document_cited_by_count",
            model_types[0],
            "NegBin coefficient for atypicality z-score (95% CI)",
        ),
        (
            "document_log_fwci",
            model_types[1],
            "OLS coefficient for atypicality z-score (95% CI)",
        ),
    ]
    ecosystems = sorted(summary_df.get_column("ecosystem").unique().to_list())

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(ecosystems) * 1.0)), sharey=True)
    for ax, (outcome, model_type, xlabel) in zip(axes, outcomes, strict=True):
        sub = summary_df.filter(
            (pl.col("outcome_variable") == outcome) & (pl.col("model_type") == model_type)
        )
        for i, eco in enumerate(ecosystems):
            rows = sub.filter(pl.col("ecosystem") == eco).to_dicts()
            if not rows:
                continue
            r = rows[0]
            ax.errorbar(
                r["atypicality_coefficient"],
                i,
                xerr=[
                    [r["atypicality_coefficient"] - r["ci_lower"]],
                    [r["ci_upper"] - r["atypicality_coefficient"]],
                ],
                fmt="o",
                capsize=4,
            )
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(ecosystems)))
        ax.set_yticklabels([e.title() if "-" in e else e.upper() for e in ecosystems])
        ax.set_xlabel(xlabel)
        ax.set_title(outcome)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Coefficient forest plot saved to {output_filename}.")


@app.command()
def main(  # noqa: C901
    remove_extremely_rare_imports: bool = True,
    rare_import_threshold: int = 3,
    top_n_fields: int = 5,
    sample: bool = False,
    sample_size: int = 5000,
    log_software_pair_examples: bool = True,
) -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create data dir
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load our dataset
    pair_metadata = _load_our_dataset(
        top_n_fields=top_n_fields,
    )

    # Remove any pairs that have null FWCI and less than 2 citations
    # Log how many we are removing by this filter
    pre_filter_count = len(pair_metadata)
    pair_metadata = (
        pair_metadata.filter(
            pl.col("document_fwci").is_not_null(),
        )
        .filter(pl.col("document_fwci") > 0)
        .filter(
            pl.col("document_cited_by_count") >= 2,
        )
    )
    post_filter_count = len(pair_metadata)
    low_or_null_citation_impact_diff_count = pre_filter_count - post_filter_count
    print(
        f"Removed {low_or_null_citation_impact_diff_count} pairs "
        f"with null FWCI or less than 2 citations"
    )

    # Select down to only 1:1 article-repository pairs
    pair_metadata = pair_metadata.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )
    print(
        f"Using {len(pair_metadata)} unique article-repository pairs "
        f"for atypicality calculation"
    )

    # Take a sample to speed up development
    if sample:
        pair_metadata = pair_metadata.sample(sample_size, seed=42)

    # Get the repository IDs in our dataset
    repository_ids = pair_metadata.get_column("repository_id").unique().to_list()

    # Load the repository imports for the repositories in our dataset
    repository_imports = _get_imported_software(
        repository_ids,
        remove_extremely_rare_imports=remove_extremely_rare_imports,
        rare_import_threshold=rare_import_threshold,
    )

    # Construct a mapping from article ID to the set of software imported by its linked repositories
    article_to_software_mapping = _construct_article_to_software_mapping(
        pair_metadata,
        repository_imports,
    )

    # Create cross-ecosystem mapping to add to results later
    ecosystem_to_documents: dict[str, set[int]] = {}
    for document_id, software_names in article_to_software_mapping.items():
        ecosystems = {s.split(":")[0] for s in software_names}
        if len(ecosystems) > 1:
            ecosystem_label = "cross-ecosystem"
        else:
            ecosystem_label = ecosystems.pop()  # "py" or "r"

        # Add ecosystem label if not already there
        if ecosystem_label not in ecosystem_to_documents:
            ecosystem_to_documents[ecosystem_label] = set()

        # Add document id to the set of documents in this ecosystem
        ecosystem_to_documents[ecosystem_label].add(document_id)

    # Filter the article_to_software_mapping to remove articles
    # in the top 1% of number of unique software imported
    # Do this per-ecosystem so that each ecosystem can have a different threshold for what counts
    # as an extreme number of software imported
    filtered_article_to_software_mapping = {}
    filtered_ecosystem_to_documents: dict[str, set[int]] = {}
    article_software_counts_rows = []
    for ecosystem_label, documents_in_ecosystem in ecosystem_to_documents.items():
        # Get the subset of the article_to_software_mapping for just the documents in this ecosystem
        ecosystem_specific_article_to_software_mapping = {
            doc_id: article_to_software_mapping[doc_id] for doc_id in documents_in_ecosystem
        }

        # This ecosystem-specific software counts
        ecosystem_article_software_counts_df = pl.DataFrame(
            [
                {"document_id": doc_id, "num_unique_software_imported": len(software_set)}
                for doc_id, software_set in ecosystem_specific_article_to_software_mapping.items()
            ]
        )

        # Get the threshold for number of unique software imported for this ecosystem
        software_count_threshold = (
            ecosystem_article_software_counts_df.filter(
                pl.col("document_id").is_in(documents_in_ecosystem)
            )
            .get_column("num_unique_software_imported")
            .quantile(0.99)
        )

        # Get the articles in this ecosystem that are above this threshold
        articles_to_exclude = (
            ecosystem_article_software_counts_df.filter(
                pl.col("num_unique_software_imported") > software_count_threshold
            )
            .get_column("document_id")
            .to_list()
        )

        # Add the docs and their software to the new filtered mapping
        for doc_id, software_set in ecosystem_specific_article_to_software_mapping.items():
            if doc_id not in articles_to_exclude:
                filtered_article_to_software_mapping[doc_id] = software_set

        # Remake the filtered ecosystem to documents mapping
        filtered_ecosystem_to_documents[ecosystem_label] = {
            doc_id for doc_id in documents_in_ecosystem if doc_id not in articles_to_exclude
        }

        # Store this document id, num software, and ecosystem label
        for doc_id, software_set in ecosystem_specific_article_to_software_mapping.items():
            if doc_id not in articles_to_exclude:
                article_software_counts_rows.append(
                    {
                        "document_id": doc_id,
                        "num_unique_software_imported": len(software_set),
                        "ecosystem_label": ecosystem_label,
                    }
                )

    # Create a dataframe of document_id to num_unique_software_imported
    article_software_counts_df = pl.DataFrame(article_software_counts_rows)

    # Describe the distribution of number of unique software imported per article
    print("Distribution of number of unique software imported per article:")
    print(
        article_software_counts_df.group_by("ecosystem_label").agg(
            pl.col("num_unique_software_imported").min().alias("min"),
            pl.col("num_unique_software_imported").quantile(0.25).alias("25%"),
            pl.col("num_unique_software_imported").quantile(0.5).alias("50%"),
            pl.col("num_unique_software_imported").quantile(0.75).alias("75%"),
            pl.col("num_unique_software_imported").max().alias("max"),
            pl.col("num_unique_software_imported").mean().alias("mean"),
            pl.col("num_unique_software_imported").std().alias("std"),
        )
    )

    # Calculate atypicality scores per-article
    # but stratified by software ecosystem
    all_ecosystem_results = []
    for ecosystem_label, documents_in_ecosystem in tqdm(
        filtered_ecosystem_to_documents.items(),
        total=len(filtered_ecosystem_to_documents),
        desc="Computing atypicality scores for each ecosystem",
    ):
        # Get the ecosystem specific article to software mapping
        ecosystem_specific_article_to_software_mapping = {
            doc_id: filtered_article_to_software_mapping[doc_id]
            for doc_id in documents_in_ecosystem
        }

        # Compute atypicality scores for this ecosystem
        ecosystem_results_df = _compute_document_atypicality_for_ecosystem(
            ecosystem_label=ecosystem_label,
            ecosystem_specific_article_to_software_mapping=ecosystem_specific_article_to_software_mapping,
            pair_metadata=pair_metadata,
            log_software_pair_examples=log_software_pair_examples,
        )
        all_ecosystem_results.append(ecosystem_results_df)

    # Combine results for all ecosystems into one dataframe
    results_df = pl.concat(all_ecosystem_results)

    # Add number of unique software imported to the results dataframe
    # by joining with article_software_counts_df
    results_df = results_df.join(
        article_software_counts_df.select(
            pl.col("document_id"),
            pl.col("num_unique_software_imported"),
        ),
        on="document_id",
        how="left",
    )

    # Store results to atypicality parquet
    results_df.write_parquet(RESULTS_DIR / "article-atypicality-scores.parquet")

    # Filter to not_null atypicality score
    # Log how many we are filtering out
    pre_filter_count = len(results_df)
    results_df = results_df.filter(pl.col("document_atypicality_score").is_not_null())
    post_filter_count = len(results_df)
    null_atypicality_count = pre_filter_count - post_filter_count
    print(
        f"Removed {null_atypicality_count} pairs with null atypicality score "
        f"(papers with only 0 or 1 software, for which atypicality is not defined)"
    )

    # Log the final N after all filtering
    print(f"Final number of papers with atypicality scores: {len(results_df)}")

    # Print the counts per-ecosystem after all filtering
    print("Counts per ecosystem after all filtering:")
    print(results_df.get_column("ecosystem_label").value_counts(sort=True))

    # Plot the distribution of atypicality scores
    g = sns.displot(
        results_df,
        kind="hist",
        stat="percent",
        common_norm=False,
        x="document_atypicality_score",
        col="ecosystem_label",
        hue="ecosystem_label",
        bins=50,
        facet_kws={"sharey": False},
    )

    # Change the subplot titles
    g.set_titles(col_template="{col_name}")

    g.fig.savefig(RESULTS_DIR / "atypicality-score-distribution.png")

    # Take the log of citations and add 1 to avoid log(0)
    results_df = results_df.with_columns(
        (pl.col("document_cited_by_count").cast(pl.Float64).log()).alias(
            "document_log_cited_by_count"
        ),
        (pl.col("document_fwci").log()).alias("document_log_fwci"),
    ).filter(
        pl.col("document_log_fwci").is_not_nan(),
    )

    # Select down to just atypicality z-score, log citations, and FWCI
    # Melt to long format so that we have:
    # "document_id", "atypicality_z_score", "citation_impact_metric", "citation_impact_value"
    analysis_df = results_df.select(
        "document_id",
        "document_log_cited_by_count",
        "document_log_fwci",
    ).unpivot(
        on=["document_log_cited_by_count", "document_log_fwci"],
        index="document_id",
        variable_name="citation_impact_metric",
        value_name="citation_impact_value",
    )

    # Join metadata and atypicality scores to the original pair metadata for downstream analysis
    analysis_df = analysis_df.join(
        results_df.select(
            "document_id",
            "document_author_count",
            "document_author_mean_citations",
            "document_log_author_mean_citations",
            "document_publication_year",
            "document_years_since_earliest",
            "document_field_name",
            "document_field_name_pruned",
            "document_domain_name",
            "document_atypicality_z_score",
            "ecosystem_label",
        ),
        on="document_id",
        how="left",
    ).sort("document_id", descending=False)

    # Create facet grid of atypicality z-score vs. log citations and FWCI
    g = sns.lmplot(
        data=analysis_df,
        x="document_atypicality_z_score",
        y="citation_impact_value",
        col="citation_impact_metric",
        hue="ecosystem_label",
        row="ecosystem_label",
        scatter_kws={"alpha": 0.3},
        facet_kws={
            "sharey": False,
            "sharex": True,
        },
    )

    # Change the subplot titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Save fig
    g.fig.savefig(RESULTS_DIR / "atypicality-vs-citation-impact.png")

    # Run statsmodels
    # Keep track of summary stats for most important models
    # Important models are the Negative Binomial (controlled with log citations) and OLS (FWCI)
    # Metrics to track are:
    # - ecosystem
    # - n
    # - outcome variable (log citations or log FWCI)
    # - atypicality coefficient
    # - p-value for atypicality coefficient
    summary_stats_rows = []
    for ecosystem_label in results_df.get_column("ecosystem_label").unique():
        # Get ecosystem specific dataframe
        ecosystem_df = results_df.filter(
            pl.col("ecosystem_label") == ecosystem_label
        ).to_pandas()

        # Raw correlation
        ols_model_raw = smf.ols(
            "document_log_cited_by_count ~ document_atypicality_z_score", data=ecosystem_df
        ).fit()

        # With controls
        ols_model_controlled = smf.ols(
            "document_log_cited_by_count "
            "~ document_atypicality_z_score "
            "+ document_years_since_earliest "
            "+ num_unique_software_imported "
            "+ document_author_count "
            "+ document_log_author_mean_citations "
            "+ C(document_field_name_pruned)",
            data=ecosystem_df,
        ).fit()

        # Negative Bin raw
        negative_binomial_model_raw = NegativeBinomial.from_formula(
            "document_cited_by_count ~ document_atypicality_z_score",
            data=ecosystem_df,
        ).fit(maxiter=1000)

        # Negative Bin with controls
        negative_binomial_model_controlled = NegativeBinomial.from_formula(
            "document_cited_by_count "
            "~ document_atypicality_z_score "
            "+ document_years_since_earliest "
            "+ num_unique_software_imported "
            "+ document_author_count "
            "+ document_log_author_mean_citations "
            "+ C(document_field_name_pruned)",
            data=ecosystem_df,
        ).fit(maxiter=1000)

        # Poisson raw
        poisson_model_raw = Poisson.from_formula(
            "document_cited_by_count ~ document_atypicality_z_score",
            data=ecosystem_df,
        ).fit(maxiter=1000)

        # Poisson with controls
        poisson_model_controlled = Poisson.from_formula(
            "document_cited_by_count "
            "~ document_atypicality_z_score "
            "+ document_years_since_earliest "
            "+ num_unique_software_imported "
            "+ document_author_count "
            "+ document_log_author_mean_citations "
            "+ C(document_field_name_pruned)",
            data=ecosystem_df,
        ).fit(maxiter=1000)

        # OLS with log FWCI as outcome
        ols_fwci = smf.ols(
            "document_log_fwci "
            "~ document_atypicality_z_score "
            "+ document_years_since_earliest "
            "+ num_unique_software_imported "
            "+ document_author_count "
            "+ document_log_author_mean_citations ",
            data=ecosystem_df,
        ).fit()

        # OLS with log FWCI as outcome (raw / uncontrolled)
        ols_fwci_raw = smf.ols(
            "document_log_fwci ~ document_atypicality_z_score",
            data=ecosystem_df,
        ).fit()

        # Store each model to its own CSV in a per-ecosystem subdirectory
        eco_dir = RESULTS_DIR / "modeling-results" / ecosystem_label
        eco_dir.mkdir(exist_ok=True, parents=True)
        for filename, model in [
            ("ols-raw.txt", ols_model_raw),
            ("ols-controlled.txt", ols_model_controlled),
            ("negbin-raw.txt", negative_binomial_model_raw),
            ("negbin-controlled.txt", negative_binomial_model_controlled),
            ("poisson-raw.txt", poisson_model_raw),
            ("poisson-controlled.txt", poisson_model_controlled),
            ("ols-fwci.txt", ols_fwci),
            ("ols-fwci-raw.txt", ols_fwci_raw),
        ]:
            with open(eco_dir / filename, "w") as f:
                f.write(model.summary().as_text())
            if hasattr(model, "get_margeff"):
                margeff_path = eco_dir / filename.replace(".txt", "-marginal-effects.txt")
                with open(margeff_path, "w") as f:
                    f.write(model.get_margeff(at="overall").summary().as_text())

        # Add summary stats for the negative binomial model with controls to our summary stats list
        predictor = "document_atypicality_z_score"
        negbin_ci = negative_binomial_model_controlled.conf_int()
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_cited_by_count",
                "model_type": "Negative Binomial with controls",
                "atypicality_coefficient": negative_binomial_model_controlled.params[predictor],
                "atypicality_p_value": negative_binomial_model_controlled.pvalues[predictor],
                "std_err": negative_binomial_model_controlled.bse[predictor],
                "ci_lower": negbin_ci.loc[predictor, 0],
                "ci_upper": negbin_ci.loc[predictor, 1],
            }
        )

        # Add summary stats for the OLS model with log FWCI as outcome to our summary stats list
        ols_ci = ols_fwci.conf_int()
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_log_fwci",
                "model_type": "OLS with controls",
                "atypicality_coefficient": ols_fwci.params[predictor],
                "atypicality_p_value": ols_fwci.pvalues[predictor],
                "std_err": ols_fwci.bse[predictor],
                "ci_lower": ols_ci.loc[predictor, 0],
                "ci_upper": ols_ci.loc[predictor, 1],
            }
        )

        # Raw (uncontrolled) Negative Binomial for document_cited_by_count
        negbin_raw_ci = negative_binomial_model_raw.conf_int()
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_cited_by_count",
                "model_type": "Negative Binomial",
                "atypicality_coefficient": negative_binomial_model_raw.params[predictor],
                "atypicality_p_value": negative_binomial_model_raw.pvalues[predictor],
                "std_err": negative_binomial_model_raw.bse[predictor],
                "ci_lower": negbin_raw_ci.loc[predictor, 0],
                "ci_upper": negbin_raw_ci.loc[predictor, 1],
            }
        )

        # Raw (uncontrolled) OLS for document_log_fwci
        ols_fwci_raw_ci = ols_fwci_raw.conf_int()
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_log_fwci",
                "model_type": "OLS",
                "atypicality_coefficient": ols_fwci_raw.params[predictor],
                "atypicality_p_value": ols_fwci_raw.pvalues[predictor],
                "std_err": ols_fwci_raw.bse[predictor],
                "ci_lower": ols_fwci_raw_ci.loc[predictor, 0],
                "ci_upper": ols_fwci_raw_ci.loc[predictor, 1],
            }
        )

    # Create a summary stats dataframe and save to CSV
    summary_stats_df = pl.DataFrame(summary_stats_rows)
    summary_stats_df.write_csv(RESULTS_DIR / "modeling-summary-stats.csv")

    # Forest plot of atypicality coefficient across ecosystems and outcomes
    _plot_coefficient_forest(
        summary_stats_df,
        model_types=("Negative Binomial with controls", "OLS with controls"),
        output_filename="coefficient-forest-plot.png",
        title="Atypicality coefficient across ecosystems and outcomes (with controls)",
    )
    _plot_coefficient_forest(
        summary_stats_df,
        model_types=("Negative Binomial", "OLS"),
        output_filename="coefficient-forest-plot-raw.png",
        title="Atypicality coefficient across ecosystems and outcomes (raw / uncontrolled)",
    )

    # Create dataframes of high-atypicality papers in each ecosystem
    for ecosystem_label in results_df.get_column("ecosystem_label").unique():
        ecosystem_df = results_df.filter(pl.col("ecosystem_label") == ecosystem_label)

        # Get the top 10 most atypical papers in this ecosystem
        top_atypical_papers_df = ecosystem_df.sort(
            "document_atypicality_z_score", descending=True
        ).head(10)

        # Save to CSV
        top_atypical_papers_df.write_csv(
            RESULTS_DIR / f"top-atypical-papers-{ecosystem_label}.csv"
        )


###############################################################################

if __name__ == "__main__":
    app()
