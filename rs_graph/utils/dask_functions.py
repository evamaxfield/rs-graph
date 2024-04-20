#!/usr/bin/env python

import logging
import time
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
    use_dask: bool = True,
    cluster_kwargs: dict | None = None,
    display_tqdm: bool = True,
) -> list:
    """Process a function using Dask."""
    if use_dask:
        if cluster_kwargs is None:
            cluster_kwargs = {}

        with Client(**cluster_kwargs) as client:
            # Log dashboard address
            log.info(f"Dask dashboard for '{name}': {client.dashboard_link}")

            # Wait for warm up
            time.sleep(4)

            # Map the function
            # Start the timer
            start_time = time.time()
            futures = client.map(func, *func_iterables, batch_size=500)

            # Show progress
            progress(futures)

            results = client.gather(futures)

            # Log time taken
            duration = time.time() - start_time
            log.info(
                f"Time taken for '{name}': {duration:.2f} seconds "
                f"({duration / len(futures):.2f} seconds per item)"
            )

            # Wait for cooldown
            time.sleep(4)

        # Wait for full shutdown
        time.sleep(2)

        return results

    else:
        # Process without Dask
        if display_tqdm:
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
