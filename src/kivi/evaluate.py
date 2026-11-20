"""Deprecated alias of :mod:`kivi.stability`, removed in kivi 0.6.0."""

import warnings

from . import stability as _stability

__all__ = ["psi_table"]  # noqa: F822 -- provided by __getattr__ with a FutureWarning


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "kivi.evaluate is deprecated and stops working in kivi 0.6.0; "
            f"import {name} from kivi.stability instead",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(_stability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
