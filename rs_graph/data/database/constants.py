#!/usr/bin/env python

from .. import DATA_FILES_DIR

PROD_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-prod.db").resolve(strict=True)
DEV_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-dev.db").resolve(strict=True)
