"""Helpers for deprecating call patterns."""

import functools
import inspect
import warnings


def positional_deprecated(*names: str, removed_in: str):
    """Accept the keyword-only parameters ``names`` by position, with a FutureWarning.

    The decorated function keeps its keyword-only signature for documentation.
    Extra positional arguments are passed on as ``names``, in order.
    """

    def decorate(func):
        parameters = inspect.signature(func).parameters.values()
        n_positional = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extra = args[n_positional:]
            if extra and len(extra) <= len(names):
                moved = names[: len(extra)]
                for name in moved:
                    if name in kwargs:
                        raise TypeError(f"{func.__name__}() got multiple values for {name!r}")
                warnings.warn(
                    f"passing {', '.join(moved)} to {func.__name__} by position is deprecated "
                    f"and stops working in kivi {removed_in}; use "
                    + ", ".join(f"{name}=..." for name in moved),
                    FutureWarning,
                    stacklevel=2,
                )
                kwargs.update(zip(moved, extra, strict=True))
                args = args[:n_positional]
            return func(*args, **kwargs)

        return wrapper

    return decorate
