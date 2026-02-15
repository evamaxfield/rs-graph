#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path
from typing import Any

import coiled
import yaml
from prefect import Task, task

###############################################################################

DEFAULT_RESULTS_DIR = Path("processing-results")
DEFAULT_GITHUB_TOKENS_FILE = ".github-tokens.yml"
DEFAULT_OPEN_ALEX_TOKENS_FILE = ".open-alex-tokens.yml"
DEFAULT_ELSEVIER_API_KEYS_FILE = ".elsevier-api-keys.yml"
DEFAULT_ERRORS_CACHE_FILE = DEFAULT_RESULTS_DIR / "errors-cache.parquet"

###############################################################################


def _get_small_cpu_api_cluster(
    n_workers: int,
    use_coiled: bool,
    coiled_region: str,
    keepalive: str = "15m",
) -> dict:
    return {
        "keepalive": keepalive,
        "vm_type": "t4g.large",
        # One worker per token to avoid rate limiting
        # This isn't deterministic, that is,
        # a single token might be used by multiple workers,
        # but this does spread the load out a bit
        # and should help avoid rate limiting
        "n_workers": n_workers,
        "threads_per_worker": 1,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _get_basic_gpu_cluster_config(
    use_coiled: bool,
    coiled_region: str,
    keepalive: str = "15m",
) -> dict:
    return {
        "keepalive": keepalive,
        "vm_type": "g4dn.xlarge",
        "n_workers": [4, 12],
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = None,
    coiled_func_name: str | None = None,
    coiled_kwargs: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    timeout_seconds: int = 1200,  # 20 minutes
) -> Task:
    if coiled_kwargs is None:
        coiled_kwargs = {}
    if prefect_kwargs is None:
        prefect_kwargs = {}

    @task(
        **prefect_kwargs,
        name=func.__name__,  # type: ignore[attr-defined]
        log_prints=True,
        timeout_seconds=timeout_seconds,
    )
    @coiled.function(
        **coiled_kwargs,
        name=coiled_func_name if coiled_func_name is not None else func.__name__,  # type: ignore[attr-defined]
        environ=environ,
    )
    def wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped_func


def _load_open_alex_tokens(
    open_alex_tokens_file: str,
) -> list[str]:
    # Load tokens
    try:
        with open(open_alex_tokens_file) as f:
            tokens_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Open Alex tokens file not found at path: {open_alex_tokens_file}"
        ) from e

    # Get tokens
    tokens_dict = tokens_file["tokens"]
    return [token_details["token"] for _user, token_details in tokens_dict.items()]


def _load_elsevier_api_keys(
    elsevier_api_keys_file: str,
) -> list[str]:
    # Load tokens
    try:
        with open(elsevier_api_keys_file) as f:
            tokens_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Elsevier tokens file not found at path: {elsevier_api_keys_file}"
        ) from e

    # Get tokens
    tokens_list = tokens_file["keys"].values()

    return tokens_list
