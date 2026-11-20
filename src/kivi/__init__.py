"""Tabular risk modeling tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kivi")
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "0+unknown"
