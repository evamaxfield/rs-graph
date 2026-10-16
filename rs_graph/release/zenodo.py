#!/usr/bin/env python

"""Create a Zenodo draft deposit of the public variant. Never publishes."""

# Disabled until the dataset is released.
# Every function below raises NotImplementedError; the original bodies are commented out.
# To re-enable: restore the commented imports, constants and bodies, remove the raises,
# and wire a CLI command and the Justfile recipe back to `deposit_draft`.

# import logging
# import os
from pathlib import Path
from typing import Any

import requests

# from dotenv import load_dotenv

###############################################################################

# Fill in before release: {"name": "Family, Given", "affiliation": "...", "orcid": "0000-..."}
ZENODO_CREATORS: list[dict[str, str]] = []

###############################################################################

# Disabled until the dataset is released.
# log = logging.getLogger(__name__)
#
# ZENODO_API = "https://zenodo.org/api"
# ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"
#
# # Zenodo's default per-record limits
# MAX_FILES = 100
# MAX_TOTAL_BYTES = 50 * 10**9
#
# # (connect, read) seconds; reads are long because parquet files are streamed
# REQUEST_TIMEOUT = (60, 3600)
#
#
# def zenodo_metadata(version: str) -> dict[str, Any]:
#     """Zenodo deposition metadata (creators are added by the deposit code)."""
#     return {
#         "title": "RS-Graph v2 (public)",
#         "upload_type": "dataset",
#         "description": (
#             "RS-Graph v2 links scientific articles to the code repositories they describe, "
#             "with article, researcher, repository and developer-account metadata and model "
#             "confidence scores. This is the public variant: article abstracts and developer "
#             "display names and emails are withheld. Mirrored on HuggingFace at "
#             "https://huggingface.co/datasets/sci-soft-collections/rs-graph-v2."
#         ),
#         "access_right": "open",
#         "license": "cc-by-4.0",
#         "keywords": [
#             "research software",
#             "scientometrics",
#             "science of science",
#             "GitHub",
#             "OpenAlex",
#         ],
#         "version": version,
#     }

###############################################################################


def _deposit_files(public_dir: Path) -> list[Path]:
    """Every non-hidden file in the public directory (parquet files and card)."""
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # return [
    #     p for p in sorted(public_dir.iterdir()) if p.is_file() and not p.name.startswith(".")
    # ]


def check_limits(files: list[Path]) -> None:
    """Fail before any request if the files exceed Zenodo's per-record limits."""
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # total = sum(p.stat().st_size for p in files)
    # if len(files) > MAX_FILES:
    #     raise ValueError(f"{len(files)} files exceeds the {MAX_FILES}-file limit")
    # if total > MAX_TOTAL_BYTES:
    #     raise ValueError(f"{total} bytes exceeds the {MAX_TOTAL_BYTES}-byte limit")


def _get_token() -> str:
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # load_dotenv()
    # token = os.getenv("ZENODO_TOKEN")
    # if not token:
    #     raise ValueError("Environment variable 'ZENODO_TOKEN' is not set")
    # return token


def _create_draft(
    session: requests.Session, base: str, concept_record_id: str | None
) -> dict[str, Any]:
    """Open a draft: a new record, or a new version of an existing one."""
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # if concept_record_id is None:
    #     response = session.post(
    #         f"{base}/deposit/depositions", json={}, timeout=REQUEST_TIMEOUT
    #     )
    #     response.raise_for_status()
    #     return response.json()
    #
    # latest = session.get(
    #     f"{base}/records/{concept_record_id}/versions/latest", timeout=REQUEST_TIMEOUT
    # )
    # latest.raise_for_status()
    # response = session.post(
    #     f"{base}/deposit/depositions/{latest.json()['id']}/actions/newversion",
    #     timeout=REQUEST_TIMEOUT,
    # )
    # response.raise_for_status()
    # draft_response = session.get(
    #     response.json()["links"]["latest_draft"], timeout=REQUEST_TIMEOUT
    # )
    # draft_response.raise_for_status()
    # draft = draft_response.json()
    #
    # # A new version inherits the previous files; remove them so only this build is uploaded
    # for inherited in draft.get("files", []):
    #     deleted = session.delete(
    #         f"{base}/deposit/depositions/{draft['id']}/files/{inherited['id']}",
    #         timeout=REQUEST_TIMEOUT,
    #     )
    #     deleted.raise_for_status()
    # return draft


def _upload_files(session: requests.Session, draft: dict[str, Any], files: list[Path]) -> None:
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # bucket_url = draft["links"]["bucket"]
    # for path in files:
    #     log.info(f"Uploading {path.name}")
    #     with path.open("rb") as f:
    #         response = session.put(
    #             f"{bucket_url}/{path.name}", data=f, timeout=REQUEST_TIMEOUT
    #         )
    #     response.raise_for_status()


def _set_metadata(
    session: requests.Session, base: str, draft_id: int, metadata: dict[str, Any]
) -> dict[str, Any]:
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # response = session.put(
    #     f"{base}/deposit/depositions/{draft_id}",
    #     json={"metadata": metadata},
    #     timeout=REQUEST_TIMEOUT,
    # )
    # response.raise_for_status()
    # return response.json()


def deposit_draft(
    public_dir: Path,
    version: str,
    concept_record_id: str | None,
    sandbox: bool,
) -> str:
    """
    Upload the public variant to a Zenodo draft and return the draft's URL.

    This never publishes. The draft is published by hand in the Zenodo UI.
    """
    # Disabled until the dataset is released.
    raise NotImplementedError("Zenodo deposit is disabled.")
    # if not ZENODO_CREATORS:
    #     raise ValueError("ZENODO_CREATORS is empty. Fill it in before depositing.")
    #
    # files = _deposit_files(public_dir)
    # check_limits(files)
    #
    # base = ZENODO_SANDBOX_API if sandbox else ZENODO_API
    # session = requests.Session()
    # session.headers["Authorization"] = f"Bearer {_get_token()}"
    #
    # draft = _create_draft(session, base, concept_record_id)
    # _upload_files(session, draft, files)
    # metadata = {**zenodo_metadata(version), "creators": ZENODO_CREATORS}
    # draft = _set_metadata(session, base, draft["id"], metadata)
    #
    # url: str = draft["links"]["html"]
    # log.info(f"Draft ready for review: {url}")
    # return url
