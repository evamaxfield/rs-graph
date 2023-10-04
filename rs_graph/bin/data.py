#!/usr/bin/env python

import logging
import shutil
from datetime import datetime

import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import load_rs_graph_repos_dataset
from rs_graph.data.github import (
    get_upstream_dependencies_for_repos,
)
from rs_graph.data.joss import get_joss_dataset as _get_joss_dataset
from rs_graph.data.softwarex import get_softwarex_dataset as _get_softwarex_dataset
from rs_graph.data import DATA_FILES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _copy_to_lib(
    final_stored_dataset: str,
    name_prefix: str,
) -> None:
    # Create lib storage path
    lib_storage_path = (
        DATA_FILES_DIR / f"{name_prefix}-{datetime.now().date().isoformat()}.parquet"
    )

    # Copy
    shutil.copy2(
        final_stored_dataset,
        lib_storage_path,
    )
    log.info(f"Copied dataset to: '{lib_storage_path}'")


@app.command()
def get_joss_dataset(
    output_filepath: str = "joss-short-paper-details.parquet",
    start_page: int = 1,
    copy_to_lib: bool = False,
    debug: bool = False,
) -> None:
    """Download the JOSS dataset."""
    # Setup logger
    setup_logger(debug=debug)

    # Download JOSS dataset
    final_stored_dataset = _get_joss_dataset(
        output_filepath=output_filepath,
        start_page=start_page,
    )
    log.info(f"Stored JOSS dataset to: '{final_stored_dataset}'")

    # Copy the final to the repo / library
    if copy_to_lib:
        _copy_to_lib(
            final_stored_dataset=final_stored_dataset,
            name_prefix="joss",
        )


@app.command()
def get_softwarex_dataset(
    output_filepath: str = "softwarex-short-paper-details.parquet",
    copy_to_lib: bool = False,
    debug: bool = False,
) -> None:
    """Download the SoftwareX dataset."""
    # Setup logger
    setup_logger(debug=debug)

    # Download SoftwareX dataset
    final_stored_dataset = _get_softwarex_dataset(
        output_filepath=output_filepath,
    )
    log.info(f"Stored SoftwareX dataset to: '{final_stored_dataset}'")

    # Copy the final to the repo / library
    if copy_to_lib:
        _copy_to_lib(
            final_stored_dataset=final_stored_dataset,
            name_prefix="softwarex",
        )


def _get_upstream_deps_from_repos_dataset(
    repos_dataset: pd.DataFrame,
    repos_column: str = "repo",
    output_filepath_for_successful_repos: str = "upstream-deps.parquet",
    output_filepath_for_failed_repos: str = "upstream-deps-failed.parquet",
    debug: bool = False,
) -> tuple[str, str]:
    # Setup logger
    setup_logger(debug=debug)

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

    return (
        output_filepath_for_successful_repos,
        output_filepath_for_failed_repos,
    )


@app.command()
def get_upstream_deps_from_repos_dataset(
    repos_dataset_path: str = "joss-short-paper-details.parquet",
    repos_column: str = "repo",
    output_filepath_for_successful_repos: str = "joss-upstream-deps.parquet",
    output_filepath_for_failed_repos: str = "joss-upstream-deps-failed.parquet",
    debug: bool = False,
) -> None:
    """Get upstream dependencies from repos."""
    # Read repos dataset
    repos_dataset = pd.read_parquet(repos_dataset_path)

    # Process
    _get_upstream_deps_from_repos_dataset(
        repos_dataset=repos_dataset,
        repos_column=repos_column,
        output_filepath_for_successful_repos=output_filepath_for_successful_repos,
        output_filepath_for_failed_repos=output_filepath_for_failed_repos,
        debug=debug,
    )


@app.command()
def get_upstream_repos_for_rs_graph_dataset(
    output_filepath_for_successful_repos: str = "rs-graph-upstream-deps.parquet",
    output_filepath_for_failed_repos: str = "rs-graph-upstream-deps-failed.parquet",
    copy_to_lib: bool = False,
    debug: bool = False,
) -> None:
    """Get upstream dependencies from repos."""
    # Read repos dataset
    repos_dataset = load_rs_graph_repos_dataset()

    # Process
    (
        output_filepath_for_successful_repos,
        output_filepath_for_failed_repos,
    ) = _get_upstream_deps_from_repos_dataset(
        repos_dataset=repos_dataset,
        repos_column="repo",
        output_filepath_for_successful_repos=output_filepath_for_successful_repos,
        output_filepath_for_failed_repos=output_filepath_for_failed_repos,
        debug=debug,
    )

    # Copy the final to the repo / library
    if copy_to_lib:
        _copy_to_lib(
            final_stored_dataset=output_filepath_for_successful_repos,
            name_prefix="rs-graph-upstream-deps",
        )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
