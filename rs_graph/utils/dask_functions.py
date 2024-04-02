#!/usr/bin/env python

import logging
from collections.abc import Callable

from distributed import Client, LocalCluster, progress
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def process_func(
    name: str,
    func: Callable,
    func_iterables: list,
    cluster_kwargs: dict,
    use_dask: bool = False,
) -> list:
    """Process a function using Dask."""
    if use_dask:
        # Create a cluster
        with LocalCluster(**cluster_kwargs) as cluster:
            # Log dashboard address
            log.info(f"Dask dashboard for '{name}': {cluster.dashboard_link}")

            # Create a Dask client
            with Client(cluster) as client:
                # Map the function
                futures = client.map(func, *func_iterables)

                # Show progress
                progress(futures)

                return client.gather(futures)

    # Process without Dask
    return [
        func(*func_iterable)
        for func_iterable in tqdm(zip(*func_iterables, strict=True), desc=name)
    ]
