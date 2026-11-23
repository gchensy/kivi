"""Public modules bind no names besides their API, private names and modules."""

import importlib
import pkgutil
import types

import pytest

import kivi

PUBLIC = [
    info.name for info in pkgutil.iter_modules(kivi.__path__) if not info.name.startswith("_")
]


def leaked(module, api=()):
    """Names that neither belong to the API, nor are private, nor are modules."""
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and name not in api and not isinstance(value, types.ModuleType)
    ]


def test_every_public_module_is_checked():
    assert len(PUBLIC) == 16
    assert "binning" in PUBLIC and "datasets" in PUBLIC


@pytest.mark.parametrize("name", PUBLIC)
def test_public_modules_bind_only_their_api_and_modules(name):
    module = importlib.import_module(f"kivi.{name}")
    assert leaked(module, module.__all__) == []


def test_the_package_binds_only_its_version_and_its_modules():
    assert leaked(kivi) == []
    assert isinstance(kivi.__version__, str)
