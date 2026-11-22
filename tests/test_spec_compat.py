"""Specifications saved by earlier releases load, compare and encode as they did.

The files in ``tests/data`` were written by the code of the release tags, not
by the current code. For each release, extract the tag and give it the
version metadata that an installed release reports, so that the file records
that ``kivi_version``, then run this module as a script with the tag's code on
the path::

    git archive v0.6.0 | tar -x -C "$TMP/kivi-0.6.0"
    mkdir -p "$TMP/meta-0.6.0/kivi-0.6.0.dist-info"
    printf 'Metadata-Version: 2.1\\nName: kivi\\nVersion: 0.6.0\\n' \\
        > "$TMP/meta-0.6.0/kivi-0.6.0.dist-info/METADATA"
    PYTHONPATH="$TMP/meta-0.6.0:$TMP/kivi-0.6.0/src" python tests/test_spec_compat.py 0.6.0

and the same with 0.8.0. The script fits the specification of `fit` and
writes ``binning_spec_<release>.json`` with ``to_json(indent=2)`` and
``binning_spec_<release>.woe.json`` with the release's ``woe_transform`` of
`encoded_frame`. The 0.6.0 file has format version 1 and the 0.8.0 file
format version 2, with a column grouped by ``"rare"``, one grouped by
``"chimerge"`` and an ``unseen`` of ``{"group": ...}``. Both were fitted with
weights, a special code, an ``unseen`` number and every numeric method.

A saved file is read and applied exactly. A fresh fit recomputes logarithms
and sums, whose last digit NumPy can round differently on another processor,
because it picks its vectorized routines by the instructions a CPU has. The
fresh fit is therefore compared with the file to a relative ``RTOL``, while
its structure, keys and labels must match exactly.
"""

import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from kivi.datasets import load_bank
from kivi.spec import BinningSpec, fit_spec

DATA = pathlib.Path(__file__).parent / "data"
# The release that wrote each file and the format version it wrote.
RELEASES = {"0.6.0": 1, "0.8.0": 2}
NUMERIC = ["balance", "age", "pdays", "duration", "campaign"]
# The last digits of a fresh fit's floats may differ from the file's on another CPU.
RTOL = 1e-12


def columns(release):
    """The columns of the specification of ``release``."""
    grouped = ["job", "education"] if release == "0.8.0" else []
    return [*NUMERIC, "housing", "contact", *grouped]


def fit(release):
    """Fit the specification that the file of ``release`` was written from.

    It uses only arguments that ``release`` accepts, so the same call runs on
    the code of the tag and on the current code.
    """
    data = load_bank().iloc[:3000]
    method = {"age": "chimerge", "pdays": "tree", "duration": "monotonic", "campaign": "uniform"}
    options = {"age": {"initial_bins": 10}}
    unseen = {"contact": 0.0}
    if release == "0.8.0":
        method.update(job="rare", education="chimerge")
        options["job"] = {"min_share": 0.05}
        unseen["job"] = {"group": "other"}
    return fit_spec(
        data[columns(release)],
        data["target"],
        method=method,
        bins={"balance": 4},
        special={"pdays": [-1]},
        options=options,
        unseen=unseen,
        weights=1.0 + np.arange(len(data)) % 3,
    )


def encoded_frame(release):
    """Later rows to encode, with a contact and a job that the fit never saw."""
    frame = load_bank().iloc[3000:3200][columns(release)].copy()
    frame.iloc[0, frame.columns.get_loc("contact")] = "satellite"
    if "job" in frame:
        frame.iloc[1, frame.columns.get_loc("job")] = "astronaut"
    return frame


def write(release):
    """Write the two files of ``release`` with the kivi on the path."""
    import kivi

    if kivi.__version__ != release:
        raise SystemExit(f"kivi on the path is {kivi.__version__}, not {release}")
    spec = fit(release)
    text = spec.to_json(indent=2) + "\n"
    (DATA / f"binning_spec_{release}.json").write_text(text, encoding="utf-8")
    encoded = spec.woe_transform(encoded_frame(release))
    # One row per line; json writes each float exactly, as the shortest repr that reads back.
    rows = ",\n  ".join(json.dumps(row) for row in encoded.to_numpy().tolist())
    text = (
        f'{{"columns": {json.dumps(list(encoded.columns))},\n'
        f' "index": {json.dumps(encoded.index.tolist())},\n'
        f' "woe": [\n  {rows}\n ]}}\n'
    )
    (DATA / f"binning_spec_{release}.woe.json").write_text(text, encoding="utf-8")


def assert_close(fresh, saved, path="spec"):
    """Compare two parsed JSON specifications: floats to ``RTOL``, all else exactly."""
    if isinstance(saved, dict):
        assert isinstance(fresh, dict), path
        assert list(fresh) == list(saved), path
        for key, value in saved.items():
            assert_close(fresh[key], value, f"{path}[{key!r}]")
    elif isinstance(saved, list):
        assert isinstance(fresh, list), path
        assert len(fresh) == len(saved), path
        for position, (left, right) in enumerate(zip(fresh, saved, strict=True)):
            assert_close(left, right, f"{path}[{position}]")
    elif isinstance(saved, float):
        assert isinstance(fresh, float), path
        assert math.isclose(fresh, saved, rel_tol=RTOL, abs_tol=RTOL), (path, fresh, saved)
    else:
        assert type(fresh) is type(saved), path
        assert fresh == saved, (path, fresh, saved)


def golden(release):
    return (DATA / f"binning_spec_{release}.json").read_text(encoding="utf-8")


def golden_encoding(release):
    content = json.loads((DATA / f"binning_spec_{release}.woe.json").read_text(encoding="utf-8"))
    return pd.DataFrame(content["woe"], index=content["index"], columns=content["columns"])


@pytest.mark.parametrize("release", RELEASES)
def test_a_saved_file_loads_with_its_release_and_format(release):
    loaded = BinningSpec.from_json(golden(release))
    assert loaded.kivi_version == release
    assert loaded.format_version == RELEASES[release]
    assert list(loaded.columns) == columns(release)
    assert bool(loaded.groups) == (RELEASES[release] == 2)


@pytest.mark.parametrize("release", RELEASES)
def test_a_saved_file_matches_a_fresh_fit_apart_from_the_kivi_version(release):
    loaded = BinningSpec.from_json(golden(release))
    fresh = fit(release)
    # Equality includes the version of kivi that fitted a specification.
    if fresh.kivi_version != release:
        assert loaded != fresh
    # A fit that needs no newer feature still writes the release's format and content.
    assert fresh.format_version == RELEASES[release]
    saved = json.loads(golden(release))
    saved["kivi_version"] = fresh.kivi_version
    assert_close(json.loads(fresh.to_json()), saved)


@pytest.mark.parametrize("release", RELEASES)
def test_a_saved_file_encodes_as_the_release_did(release):
    frame = encoded_frame(release)
    encoded = BinningSpec.from_json(golden(release)).woe_transform(frame)
    expected = golden_encoding(release).set_axis(frame.index)
    pd.testing.assert_frame_equal(encoded, expected, check_exact=True)
    # A fresh fit recomputes the WOE, so it gives the same encoding up to rounding.
    pd.testing.assert_frame_equal(
        fit(release).woe_transform(frame), expected, check_exact=False, rtol=RTOL, atol=RTOL
    )
    # The unseen contact gets the number of the file; the unseen job the WOE of its group.
    assert encoded["contact"].iloc[0] == 0.0
    if "job" in frame:
        other = BinningSpec.from_json(golden(release)).tables["job"].loc["other", "woe"]
        assert encoded["job"].iloc[1] == other


def test_a_newer_format_version_is_rejected_with_its_number():
    content = json.loads(golden("0.8.0"))
    newer = BinningSpec.FORMAT_VERSION + 1
    content["format_version"] = newer
    message = (
        f"^format_version {newer} is newer than {BinningSpec.FORMAT_VERSION}, the newest that "
        r"kivi \S+ reads; upgrade kivi to load this specification$"
    )
    with pytest.raises(ValueError, match=message):
        BinningSpec.from_json(json.dumps(content))


if __name__ == "__main__":
    write(sys.argv[1])
