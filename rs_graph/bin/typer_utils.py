#!/usr/bin/env python

import logging
import shutil

from rs_graph.data import DATA_FILES_DIR

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
