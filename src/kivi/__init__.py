"""Tabular risk modeling tools."""

import importlib.metadata as _metadata

try:
    __version__ = _metadata.version("kivi")
except _metadata.PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "0+unknown"
