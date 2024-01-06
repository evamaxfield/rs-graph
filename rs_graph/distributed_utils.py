#!/usr/bin/env python

import os


def use_coiled() -> bool:
    return os.environ.get("USE_COILED", "False") == "True"
