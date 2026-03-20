#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

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
    host_setup_script: str | None = None,
) -> dict:
    return {
        "keepalive": keepalive,
        "vm_type": "t4g.xlarge",
        "n_workers": round(n_workers / 4),
        "threads_per_worker": 4,  # t4g.xlarge has 4 vCPUs, so we can use all of them
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
        "extra_kwargs": {
            "host_setup_script": host_setup_script,
        },
    }


def _get_basic_gpu_cluster_config(
    use_coiled: bool,
    coiled_region: str,
    keepalive: str = "15m",
) -> dict:
    return {
        "keepalive": keepalive,
        "vm_type": "g4dn.xlarge",
        "n_workers": [2, 12],
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


@overload
def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = ...,
    coiled_func_name: str | None = ...,
    coiled_kwargs: dict[str, Any] | None = ...,
    environ: dict[str, str] | None = ...,
    timeout_seconds: int = ...,
    *,
    warmup_timeout_seconds: None = ...,
) -> Task: ...


@overload
def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = ...,
    coiled_func_name: str | None = ...,
    coiled_kwargs: dict[str, Any] | None = ...,
    environ: dict[str, str] | None = ...,
    timeout_seconds: int = ...,
    *,
    warmup_timeout_seconds: int,
) -> tuple[Task, Task]: ...


def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = None,
    coiled_func_name: str | None = None,
    coiled_kwargs: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    timeout_seconds: int = 1200,  # 20 minutes
    *,
    warmup_timeout_seconds: int | None = None,
) -> Task | tuple[Task, Task]:
    if coiled_kwargs is None:
        coiled_kwargs = {}
    if prefect_kwargs is None:
        prefect_kwargs = {}

    # Create the coiled function once — all Prefect task wrappers call this
    # same object so they share the same cluster.
    @coiled.function(
        **coiled_kwargs,
        name=coiled_func_name if coiled_func_name is not None else func.__name__,  # type: ignore[attr-defined]
        environ=environ,
    )
    def coiled_func(*args, **kwargs):
        return func(*args, **kwargs)

    @task(
        **prefect_kwargs,
        name=func.__name__,  # type: ignore[attr-defined]
        log_prints=True,
        timeout_seconds=timeout_seconds,
    )
    def normal_task(*args, **kwargs):
        return coiled_func(*args, **kwargs)

    if warmup_timeout_seconds is not None:

        @task(
            **prefect_kwargs,
            name=func.__name__,  # type: ignore[attr-defined]
            log_prints=True,
            timeout_seconds=warmup_timeout_seconds,
        )
        def warmup_task(*args, **kwargs):
            return coiled_func(*args, **kwargs)

        return warmup_task, normal_task

    return normal_task


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
