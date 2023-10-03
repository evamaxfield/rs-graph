#!/usr/bin/env python

import logging

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
