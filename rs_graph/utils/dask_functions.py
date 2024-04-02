#!/usr/bin/env python

from distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from typing import Callable
import logging

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

def process_func(name: str, func: Callable, func_iterables: list, cluster_kwargs: dict) -> list:
    """Process a function using Dask."""
    # Create a cluster
    with LocalCluster(**cluster_kwargs) as cluster:
        # Log dashboard address
        log.info(f"Dask dashboard for '{name}': {cluster.dashboard_link}")

        # Create a Dask client
        with Client(cluster) as client:
            with ProgressBar():
                # Map the function
                futures = client.map(func, *func_iterables)

                return client.gather(futures)