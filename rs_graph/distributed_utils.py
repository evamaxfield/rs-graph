#!/usr/bin/env python

import os

def use_coiled() -> bool:
    return bool(os.environ.get("USE_COILED", False))