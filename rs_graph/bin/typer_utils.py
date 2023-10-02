#!/usr/bin/env python

import logging

###############################################################################

def setup_logger(debug: bool = False) -> None:
    lvl = logging.INFO
    if debug:
        lvl = logging.DEBUG
    
    logging.basicConfig(
        level=lvl,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )