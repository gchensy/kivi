"""The helper that deprecations warn through.

kivi 1.0 has no deprecations: it removed those of 0.9. A deprecation in a later
1.x release warns through ``kivi._deprecation.warn`` and is tested here until
2.0 removes it.
"""

import os
import warnings

import pytest

from kivi import _deprecation


def test_warn_gives_a_future_warning_for_the_calling_line():
    with pytest.warns(FutureWarning, match="^gone in 2.0$") as record:
        _deprecation.warn("gone in 2.0")
    assert len(record) == 1
    assert record[0].filename == __file__


def test_warn_skips_the_frames_inside_kivi():
    # Code compiled under a file name in the package stands for a kivi function that
    # checks a deprecated argument two calls deep.
    inner = os.path.join(os.path.dirname(_deprecation.__file__), "example.py")
    code = compile(
        "def check():\n    warn('gone in 2.0')\n\ndef public():\n    check()\n", inner, "exec"
    )
    namespace = {"warn": _deprecation.warn}
    exec(code, namespace)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        namespace["public"]()
    assert len(record) == 1
    assert record[0].category is FutureWarning
    assert record[0].filename == __file__
