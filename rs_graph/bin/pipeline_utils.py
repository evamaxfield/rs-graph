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
DEFAULT_OPEN_ALEX_EMAILS_FILE = ".open-alex-emails.yml"
DEFAULT_ELSEVIER_API_KEYS_FILE = ".elsevier-api-keys.yml"

###############################################################################


def _get_small_cpu_api_cluster(
    n_workers: int,
    use_coiled: bool,
    coiled_region: str,
    keepalive: str = "15m",
    software_env_name: str | None = None,
) -> dict:
    return {
        "software": software_env_name,
        "keepalive": keepalive,
        "vm_type": "t4g.small",
        # One worker per token to avoid rate limiting
        # This isn't deterministic, that is,
        # a single token might be used by multiple workers,
        # but this does spread the load out a bit
        # and should help avoid rate limiting
        "n_workers": n_workers,
        "threads_per_worker": 1,
        "spot_policy": "on-demand",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _get_basic_gpu_cluster_config(
    use_coiled: bool,
    coiled_region: str,
    keepalive: str = "15m",
    software_env_name: str | None = None,
) -> dict:
    return {
        "software": software_env_name,
        "keepalive": keepalive,
        "vm_type": "g4dn.xlarge",
        "n_workers": [4, 6],
        "spot_policy": "on-demand",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = None,
    coiled_func_name: str | None = None,
    coiled_kwargs: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    timeout_seconds: int = 600,  # 10 minutes
) -> Task:
    if coiled_kwargs is None:
        coiled_kwargs = {}
    if prefect_kwargs is None:
        prefect_kwargs = {}

    @task(
        **prefect_kwargs,
        name=func.__name__,
        log_prints=True,
        timeout_seconds=timeout_seconds,
    )
    @coiled.function(
        **coiled_kwargs,
        name=coiled_func_name if coiled_func_name is not None else func.__name__,
        environ=environ,
    )
    def wrapped_func(*args, **kwargs):  # type: ignore
        return func(*args, **kwargs)

    return wrapped_func


def _load_open_alex_emails(
    open_alex_emails_file: str,
) -> list[str]:
    # Load emails
    try:
        with open(open_alex_emails_file) as f:
            emails_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Open Alex emails file not found at path: {open_alex_emails_file}"
        ) from e

    # Get emails
    emails_list = emails_file["emails"].values()

    return emails_list


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


def _load_coiled_software_envs(
    coiled_software_envs_file: str,
) -> dict[str, str] | None:
    # Load software envs
    try:
        with open(coiled_software_envs_file) as f:
            envs_file = yaml.safe_load(f)

        return envs_file["envs"]

    except FileNotFoundError:
        print(f"Coiled software envs file not found at path: {coiled_software_envs_file}")
        return None
