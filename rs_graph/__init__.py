"""Top-level package for rs-graph."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rs-graph")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
