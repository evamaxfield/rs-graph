#!/usr/bin/env python

import logging
import os

import coiled
import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_defaults, setup_logger
from rs_graph.data import DATA_FILES_DIR, load_basic_repos_dataset
from rs_graph.data.enrichment import github, semantic_scholar
from rs_graph.distributed_utils import use_coiled

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def get_upstreams(debug: bool = False) -> None:
    """Get upstream dependencies from repos."""
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Get upstream deps
    upstream_deps, failed = github.get_upstream_dependencies_for_repos(
        repos=rs_graph_repos.repo.tolist(),
    )

    # Convert to dataframe
    upstream_deps_df = pd.DataFrame([d.to_dict() for d in upstream_deps])
    failed_df = pd.DataFrame([f.to_dict() for f in failed])

    # Store upstream deps
    output_filepath_for_successful_repos = (
        DATA_FILES_DIR / "rs-graph-upstream-deps.parquet"
    )
    upstream_deps_df.to_parquet(output_filepath_for_successful_repos)
    log.info(f"Stored upstream deps to: '{output_filepath_for_successful_repos}'")

    # Store failed repos
    output_filepath_for_failed_repos = (
        DATA_FILES_DIR / "rs-graph-upstream-deps-failed.parquet"
    )
    failed_df.to_parquet(output_filepath_for_failed_repos)
    log.info(f"Stored failed repos to: '{output_filepath_for_failed_repos}'")


@app.command()
def get_extended_paper_details(debug: bool = False) -> None:
    """
    Get the extended paper details for each paper in the dataset.

    Be sure to set the `SEMANTIC_SCHOLAR_API_KEY` environment variable.
    """
    # Setup logger
    storage_prefix = setup_defaults(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Get semantic_scholar_api_key
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    # Handle distributed
    if use_coiled():
        # Get extended paper details
        @coiled.function(
            container="ghcr.io/evamaxfield/rs-graph:distributed.enrichment",
            vm_type="n1-standard-1",
            local=not use_coiled(),
        )
        def wrapped_get_paper_details() -> list:
            return semantic_scholar.get_extended_paper_details(
                paper_dois=rs_graph_repos.doi.tolist(),
                semantic_scholar_api_key=semantic_scholar_api_key,
            )

        # Set adaptive limit
        wrapped_get_paper_details.cluster.adapt(minimum=1, maximum=48)

        # Get extended paper details
        extended_paper_details = wrapped_get_paper_details()

    # Run locally
    else:
        extended_paper_details = semantic_scholar.get_extended_paper_details(
            paper_dois=rs_graph_repos.doi.tolist(),
            semantic_scholar_api_key=semantic_scholar_api_key,
        )

    # Convert to dataframe
    extended_paper_details_df = pd.DataFrame(
        [d.to_dict() for d in extended_paper_details]
    )

    # Store extended paper details
    if storage_prefix:
        output_filepath_for_author_details = (
            f"{storage_prefix}/rs-graph-extended-paper-details.parquet"
        )
    else:
        output_filepath_for_author_details = str(
            DATA_FILES_DIR / "rs-graph-extended-paper-details.parquet"
        )
    extended_paper_details_df.to_parquet(output_filepath_for_author_details)
    log.info(
        f"Stored extended paper details to: '{output_filepath_for_author_details}'"
    )


@app.command()
def get_repo_contributors(debug: bool = False) -> None:
    """
    Get the contributors for each repo in the dataset.

    Be sure to set the `GITHUB_TOKEN` environment variable.
    """
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Store repo contributors
    output_filepath_for_repo_contributors = (
        DATA_FILES_DIR / "rs-graph-repo-contributors.parquet"
    )

    # Get repo contributors
    repo_contributors = github.get_repo_contributors_for_repos(
        repo_urls=rs_graph_repos.repo.tolist(),
        cache_file=output_filepath_for_repo_contributors,
    )

    # Convert to dataframe
    repo_contributors_df = pd.DataFrame([d.to_dict() for d in repo_contributors])

    # Store repo contributors
    repo_contributors_df.to_parquet(output_filepath_for_repo_contributors)
    log.info(f"Stored repo contributors to: '{output_filepath_for_repo_contributors}'")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
