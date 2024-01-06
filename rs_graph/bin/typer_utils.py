#!/usr/bin/env python

import logging
import shutil
from datetime import datetime

from dotenv import load_dotenv

from rs_graph.data import DATA_FILES_DIR, REMOTE_STORAGE_BUCKET
from rs_graph.distributed_utils import use_coiled

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def setup_logger(debug: bool = False) -> None:
    lvl = logging.INFO
    if debug:
        lvl = logging.DEBUG

    # If debug, allow lots of logging from backoff
    if debug:
        logging.getLogger("backoff").setLevel(logging.DEBUG)
    # Otherwise ignore basically all backoff logging
    else:
        logging.getLogger("backoff").setLevel(logging.ERROR)

    logging.basicConfig(
        level=lvl,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )


def copy_data_to_lib(
    final_stored_dataset: str,
    name_prefix: str,
) -> None:
    # Create lib storage path
    lib_storage_path = DATA_FILES_DIR / f"{name_prefix}.parquet"

    # Copy
    shutil.copy2(
        final_stored_dataset,
        lib_storage_path,
    )
    log.info(f"Copied dataset to: '{lib_storage_path}'")


def setup_remote_path() -> str:
    # Create prefix
    utcnow = datetime.utcnow().replace(microsecond=0).isoformat()
    remote_storage_prefix = f"distributed-{utcnow}".replace(":", "-")

    # Clean prefix
    if remote_storage_prefix.startswith("/"):
        remote_storage_prefix = remote_storage_prefix.strip("/")
    elif remote_storage_prefix.endswith("/"):
        remote_storage_prefix = remote_storage_prefix.rstrip("/")

    # Handle final path
    full_storage_prefix = f"{REMOTE_STORAGE_BUCKET}/remote/{remote_storage_prefix}"

    return full_storage_prefix


def setup_defaults(
    debug: bool = False,
) -> str | None:
    setup_logger(debug=debug)

    # Load env
    load_dotenv()

    # Setup remote path
    if use_coiled():
        full_storage_prefix = setup_remote_path()
        return full_storage_prefix

    return None
