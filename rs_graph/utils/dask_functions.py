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
    use_tqdm: bool = True,
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
        if use_tqdm:
            iterable = tqdm(
                zip(
                    *func_iterables,
                    strict=True,
                ),
                desc=name,
                total=len(func_iterables[0]),
            )
        else:
            iterable = zip(
                *func_iterables,
                strict=True,
            )

        return [func(*func_iterable) for func_iterable in iterable]
