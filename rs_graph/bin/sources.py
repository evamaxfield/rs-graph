#!/usr/bin/env python

import json
import logging
import pkgutil
from pathlib import Path

import coiled
import typer

from rs_graph.bin.typer_utils import setup_defaults
from rs_graph.data import DATA_FILES_DIR, DATASET_SOURCE_FILE_PATTERN, sources
from rs_graph.distributed_utils import use_coiled

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

ALL_DATASET_SOURCE_MODULES = [i.name for i in pkgutil.iter_modules(sources.__path__)]

###############################################################################

app = typer.Typer()

###############################################################################


def _get_single_dataset(
    source: str,
    storage_prefix: str | Path = DATA_FILES_DIR,
    gh_api_keys_file: str = ".github-tokens.json",
) -> None:
    """Download a single dataset."""
    # Get the dataset loader function from the module
    source_module = getattr(sources, source)

    # Check for gh_api_keys_file
    assert (
        Path(gh_api_keys_file).resolve().exists()
    ), f"Could not find '{gh_api_keys_file}'"

    # Setup function / handle distributed
    @coiled.function(
        container="ghcr.io/evamaxfield/rs-graph:distributed.data-sources",
        vm_type="n1-standard-1",
        local=not use_coiled(),
    )
    def get_dataset_func() -> str:
        # Download the dataset
        return source_module.get_dataset(
            output_filepath=f"{storage_prefix}/{source}{DATASET_SOURCE_FILE_PATTERN}",
        )

    # Need to do some adaptive handling due to number of keys available
    with open(gh_api_keys_file) as f:
        loaded_github_api_keys = json.load(f)

    # Adapt cluster to number of keys
    get_dataset_func.cluster.adapt(minimum=1, maximum=len(loaded_github_api_keys))

    # Actual run
    final_stored_dataset = get_dataset_func()
    log.info(f"Stored {source} dataset to: '{final_stored_dataset}'")


@app.command()
def get_single(
    source: str,
    debug: bool = False,
    gh_api_keys_file: str = ".github-tokens.json",
) -> None:
    """Download the JOSS dataset."""
    # Setup logger
    storage_prefix = setup_defaults(debug=debug)

    # Check storage path
    if storage_prefix:
        # Run download
        _get_single_dataset(
            source=source,
            storage_prefix=storage_prefix,
            gh_api_keys_file=gh_api_keys_file,
        )

    else:
        # Run download
        _get_single_dataset(
            source=source,
            gh_api_keys_file=gh_api_keys_file,
        )


###############################################################################


@app.command()
def get_all(
    debug: bool = False,
    gh_api_keys_file: str = ".github-tokens.json",
) -> None:
    """Download all source datasets."""
    # Setup logger
    storage_prefix = setup_defaults(debug=debug)

    # Run download
    for source in ALL_DATASET_SOURCE_MODULES:
        # Check storage path
        if storage_prefix:
            # Run download
            _get_single_dataset(
                source=source,
                storage_prefix=storage_prefix,
                gh_api_keys_file=gh_api_keys_file,
            )

        else:
            # Run download
            _get_single_dataset(
                source=source,
                gh_api_keys_file=gh_api_keys_file,
            )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
