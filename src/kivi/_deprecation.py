"""Deprecation warnings that point at the caller's code.

A deprecated argument may be checked several calls deep inside kivi, for
example when ``monitoring_table`` passes ``smoothing`` on. ``warn`` therefore
attributes the warning to the first frame outside the package instead of
taking a fixed ``stacklevel``.
"""

import os
import sys
import warnings

_PACKAGE = os.path.dirname(__file__) + os.sep


def warn(message: str) -> None:
    """Issue a ``FutureWarning`` for the line that called kivi."""
    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back
        level += 1
    warnings.warn(message, FutureWarning, stacklevel=level)
