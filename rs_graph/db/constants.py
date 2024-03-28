#!/usr/bin/env python

from ..data import DATA_FILES_DIR

PROD_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-prod.db").resolve()
DEV_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-dev.db").resolve()
