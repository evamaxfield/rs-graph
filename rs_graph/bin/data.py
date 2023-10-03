#!/usr/bin/env python

import logging
import shutil
from datetime import datetime

import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data.github import (
    get_upstream_dependencies_for_repos,
)
from rs_graph.data.joss import get_joss_dataset

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def joss(
    output_filepath: str = "joss-short-paper-details.parquet",
    start_page: int = 1,
    copy_to_lib: bool = False,
    debug: bool = False,
) -> None:
    """Download the JOSS dataset."""
    # Setup logger
    setup_logger(debug=debug)

    # Download JOSS dataset
    final_stored_dataset = get_joss_dataset(
        output_filepath=output_filepath,
        start_page=start_page,
    )
    log.info(f"Stored JOSS dataset to: '{final_stored_dataset}'")

    # Copy the final to the repo / library
    if copy_to_lib:
        current_date_str = datetime.now().date().isoformat()
        lib_storage_path = f"rs_graph/data/files/joss-{current_date_str}.parquet"
        shutil.copy2(
            final_stored_dataset,
            lib_storage_path,
        )
        log.info(f"Copied JOSS dataset to: '{lib_storage_path}'")


@app.command()
def get_upstream_deps_from_repos_dataset(
    repos_dataset_path: str = "joss-short-paper-details.parquet",
    repos_column: str = "repo",
    output_filepath_for_successful_repos: str = "joss-upstream-deps.parquet",
    output_filepath_for_failed_repos: str = "joss-upstream-deps-failed.parquet",
    debug: bool = False,
) -> None:
    """Get upstream dependencies from repos."""
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    repos_dataset = pd.read_parquet(repos_dataset_path)

    # Get upstream deps
    upstream_deps, failed = get_upstream_dependencies_for_repos(
        repos=repos_dataset[repos_column].tolist(),
    )

    # Convert to dataframe
    upstream_deps_df = pd.DataFrame([d.to_dict() for d in upstream_deps])
    failed_df = pd.DataFrame([f.to_dict() for f in failed])

    # Store upstream deps
    upstream_deps_df.to_parquet(output_filepath_for_successful_repos)
    log.info(f"Stored upstream deps to: '{output_filepath_for_successful_repos}'")

    # Store failed repos
    failed_df.to_parquet(output_filepath_for_failed_repos)
    log.info(f"Stored failed repos to: '{output_filepath_for_failed_repos}'")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
