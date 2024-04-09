#!/usr/bin/env python

import logging
from collections.abc import Callable

from distributed import Client, progress
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def process_func(
    name: str,
    func: Callable,
    func_iterables: list,
    cluster_address: str | None,
) -> list:
    """Process a function using Dask."""
    if cluster_address is not None:
        with Client(address=cluster_address) as client:
            # Map the function
            futures = client.map(func, *func_iterables)

            # Show progress
            progress(futures)

            return client.gather(futures)

    else:
        # Process without Dask
        return [
            func(*func_iterable)
            for func_iterable in tqdm(
                zip(
                    *func_iterables,
                    strict=True,
                ),
                desc=name,
                total=len(func_iterables[0]),
            )
        ]
