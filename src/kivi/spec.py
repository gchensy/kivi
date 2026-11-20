"""Binning specifications for all columns of a data frame.

``fit_spec`` bins every column of a development sample and measures the
weight of evidence of each group. The resulting `BinningSpec` groups and
encodes any other sample in the same way. It can be saved as JSON and loaded
again, so that production scoring reproduces the development encoding
exactly.
"""

import inspect
import json
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

import kivi

from . import binning, woe
from ._validation import as_series, as_weights

__all__ = ["BinningSpec", "fit_spec"]

_FORMAT = "kivi.BinningSpec"
_FORMAT_VERSION = 1
_KINDS = ("numeric", "categorical")
_FITTERS = {
    "quantile": binning.bin_edges,
    "uniform": binning.bin_edges,
    "chimerge": binning.chimerge_edges,
    "tree": binning.tree_edges,
    "monotonic": binning.monotonic_edges,
}
# Arguments that fit_spec passes itself and that options must therefore not contain.
_SET_BY_FIT = {"bins", "max_bins", "method", "special", "weights", "smoothing"}


def _installed_version() -> str:
    return kivi.__version__


@dataclass(frozen=True, eq=False, repr=False)
class BinningSpec:
    """Binning and WOE encoding of every column, fitted on a development sample.

    For each column, a specification records whether it is numeric or
    categorical, the edges and special codes of a numeric column, the WOE
    table of its groups and the policy for groups that are not in the table.
    `cut` groups any sample with the same edges, `woe_transform` encodes it
    with the same WOE, and `to_json` and `from_json` save and load the
    specification without changing either. Fit a specification with
    `fit_spec` rather than calling the constructor.

    A specification is immutable. Its mappings are read-only, its edges are
    read-only arrays, and ``tables`` returns a new copy of a table each time.
    To change a field, such as the unseen policy, use `dataclasses.replace`,
    which validates the result again.

    Parameters
    ----------
    kinds : mapping
        ``"numeric"`` or ``"categorical"`` for each column, in the order of
        the specification.
    edges : mapping
        Edges of each numeric column, increasing from ``-inf`` to ``inf``.
    special : mapping
        Special codes of each numeric column, possibly none. Each code forms
        its own group, as in `kivi.binning.cut`.
    tables : mapping
        The `kivi.woe.woe_table` result of each column. The groups of a
        numeric column are its special codes and the intervals of its edges.
        Missing values form a group labelled NaN.
    unseen : mapping
        ``"raise"`` or a finite WOE for each column, used for values whose
        group is not in the column's table.
    smoothing : float, default 0.5
        Smoothing with which the tables were computed.
    kivi_version : str, optional
        Version of kivi that fitted the specification. By default the
        installed version.

    Attributes
    ----------
    format_version : int
        Version of the JSON format that `to_json` writes.

    Raises
    ------
    ValueError
        If `kinds` is empty or has a name that is not a string or a kind
        other than the two above, if `edges` and `special` do not have one
        entry per numeric column or `tables` and `unseen` one entry per
        column, or if an entry is invalid: edges that do not increase from
        ``-inf`` to ``inf``, a missing or unsupported special code, a table
        without a finite ``woe`` column or with a repeated group, a group of
        a numeric column that is neither one of its special codes nor an
        interval of its edges, or an unseen policy that is neither
        ``"raise"`` nor a finite number. Also if `smoothing` is not finite
        and positive or `kivi_version` is not a string.

    See Also
    --------
    fit_spec : Fit a specification on a development sample.
    kivi.binning.cut : Group one numeric column.
    kivi.woe.woe_transform : Encode one column with its WOE table.

    Notes
    -----
    Labels of categorical columns must be strings, integers, floats, booleans
    or intervals, which the JSON format keeps apart. Within a column, labels
    that compare equal, such as ``True`` and ``1``, form one group, because
    `kivi.woe.woe_table` and `kivi.woe.woe_transform` match labels by
    equality.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.spec import fit_spec
    >>> frame = pd.DataFrame({
    ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
    ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
    ... })
    >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    >>> spec = fit_spec(frame, target, bins=2)
    >>> spec
    BinningSpec(columns=['age', 'housing'])
    >>> dict(spec.kinds)
    {'age': 'numeric', 'housing': 'categorical'}
    >>> spec.edges["age"]
    array([-inf, 41.5,  inf])
    """

    kinds: Mapping[str, Literal["numeric", "categorical"]]
    edges: Mapping[str, npt.NDArray[np.float64]]
    special: Mapping[str, tuple]
    tables: Mapping[str, pd.DataFrame]
    unseen: Mapping[str, float | Literal["raise"]]
    smoothing: float = 0.5
    kivi_version: str = field(default_factory=_installed_version)
    format_version: ClassVar[int] = _FORMAT_VERSION

    def __post_init__(self):
        """Validate the fields and replace them with read-only copies."""
        kinds = _checked_kinds(self.kinds)
        names = list(kinds)
        numeric = [name for name in names if kinds[name] == "numeric"]
        edges = {
            name: _checked_edges(value, f"edges of column {name!r}")
            for name, value in _entries(self.edges, numeric, "edges", "numeric ").items()
        }
        special = {
            name: _checked_codes(value, f"special codes of column {name!r}")
            for name, value in _entries(self.special, numeric, "special", "numeric ").items()
        }
        tables = {
            name: _checked_table(
                value, name, (edges[name], special[name]) if name in edges else None
            )
            for name, value in _entries(self.tables, names, "tables").items()
        }
        unseen = {
            name: _checked_unseen(value, f"unseen of column {name!r}")
            for name, value in _entries(self.unseen, names, "unseen").items()
        }
        _check_smoothing(self.smoothing)
        if not isinstance(self.kivi_version, str):
            raise ValueError(f"kivi_version must be a string, got {self.kivi_version!r}")
        object.__setattr__(self, "kinds", MappingProxyType(kinds))
        object.__setattr__(self, "edges", MappingProxyType(edges))
        object.__setattr__(self, "special", MappingProxyType(special))
        object.__setattr__(self, "tables", _Tables(tables))
        object.__setattr__(self, "unseen", MappingProxyType(unseen))
        object.__setattr__(self, "smoothing", float(self.smoothing))

    def __repr__(self) -> str:
        """Name the columns, which is shorter than listing every field."""
        return f"{type(self).__name__}(columns={list(self.kinds)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare the JSON text, which keeps labels such as True and 1 apart."""
        if not isinstance(other, BinningSpec):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        """Hash the JSON text, consistently with equality."""
        return hash(self.to_json())

    def __reduce__(self):
        """Pickle through the dict format, because read-only mappings cannot be pickled."""
        return (type(self).from_dict, (self.to_dict(),))

    @property
    def columns(self) -> pd.Index:
        """Names of the columns in the specification.

        The columns are in the order in which they were fitted, which is the
        order of the columns of `cut` and `woe_transform`.

        See Also
        --------
        select : A specification of some of the columns.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52], "housing": ["yes", "no", "yes", "no"]})
        >>> fit_spec(frame, [1, 0, 0, 1], bins=2).columns
        Index(['age', 'housing'], dtype='object')
        """
        return pd.Index(list(self.kinds), dtype=object)

    def cut(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Group every column of a sample as in development.

        Numeric columns are assigned to their bins with `kivi.binning.cut`,
        using the fitted edges and special codes. Categorical columns are
        returned unchanged, because their labels are their groups.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the specification. Other columns
            are ignored.

        Returns
        -------
        pandas.DataFrame
            The groups of each column of the specification, in its order,
            with the index of `frame`. Numeric columns are categorical, with
            the special codes followed by every interval of the edges.

        Raises
        ------
        ValueError
            If `frame` is not a DataFrame, lacks a column of the
            specification or has it twice, or a numeric column cannot be
            binned, for example because it has values that are neither
            numbers nor special codes. The message names the column.

        See Also
        --------
        woe_transform : Encode the groups with the fitted WOE.
        kivi.binning.cut : Group one numeric column.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52], "housing": ["yes", "no", "yes", "no"]})
        >>> spec = fit_spec(frame, [1, 0, 0, 1], bins=2)
        >>> spec.cut(pd.DataFrame({"age": [30, 70], "housing": ["no", "yes"]}))
                    age housing
        0  (-inf, 38.0]      no
        1   (38.0, inf]     yes
        """
        frame = self._sample(frame)
        groups = {}
        for name, kind in self.kinds.items():
            values = frame[name]
            if kind == "categorical":
                groups[name] = values.array
                continue
            codes = list(self.special[name]) or None
            try:
                grouped = binning.cut(values, self.edges[name], special=codes)
            except (TypeError, ValueError) as error:
                raise ValueError(f"column {name!r}: {error}") from error
            groups[name] = grouped.array
        return pd.DataFrame(groups, index=frame.index)

    def woe_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode every column of a sample with the WOE fitted in development.

        The columns are grouped as by `cut`, and each group is replaced with
        its WOE by `kivi.woe.woe_transform`. A group that is not in a
        column's table, such as a new category, gets the column's unseen
        policy, which by default raises an error.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the specification. Other columns
            are ignored.

        Returns
        -------
        pandas.DataFrame
            Float WOE values of each column of the specification, in its
            order, with the index of `frame`.

        Raises
        ------
        ValueError
            If `frame` is not a DataFrame, lacks a column of the
            specification or has it twice, or a numeric column cannot be
            binned, or a column with the unseen policy ``"raise"`` has values
            whose group is not in its table. The message names the column.

        See Also
        --------
        cut : Group the columns without encoding them.
        kivi.woe.woe_transform : Encode one column.
        kivi.scorecard.woe_to_points : Turn the WOE values into points.

        Notes
        -----
        The intervals of the edges are closed on the right, so ``-inf`` falls
        outside every interval. `cut` leaves it missing, as ``pandas.cut``
        does, and `woe_transform` treats it as unseen rather than giving it
        the WOE of missing values.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> new = pd.DataFrame({"age": [30, 70], "housing": ["no", "rent"]})
        >>> fit_spec(frame, target, bins=2).woe_transform(new)
        Traceback (most recent call last):
        ...
        ValueError: column 'housing': 1 group(s) are not in table, such as 'rent'; ...

        With an unseen WOE for ``housing``, the new label gets 0:

        >>> spec = fit_spec(frame, target, bins=2, unseen={"housing": 0.0})
        >>> spec.woe_transform(new).round(4)
              age  housing
        0  1.4553  -0.5878
        1 -1.7228   0.0000
        """
        frame = self._sample(frame)
        groups = self.cut(frame)
        encoded = {}
        for name, kind in self.kinds.items():
            unseen = self.unseen[name]
            try:
                values = woe.woe_transform(groups[name], self.tables[name]["woe"], unseen=unseen)
            except ValueError as error:
                raise ValueError(f"column {name!r}: {error}") from error
            values = values.to_numpy(dtype=float, copy=True)
            if kind == "numeric":
                outside = groups[name].isna().to_numpy(dtype=bool) & frame[name].notna().to_numpy(
                    dtype=bool
                )
                if outside.any():
                    if unseen == "raise":
                        example = frame[name][outside].iloc[0]
                        raise ValueError(
                            f"column {name!r}: {int(outside.sum())} value(s) fall outside the "
                            f"edges, such as {_shown(example)}; give the column an unseen WOE"
                        )
                    values[outside] = unseen
            encoded[name] = values
        return pd.DataFrame(encoded, index=frame.index)

    def select(self, columns: Sequence[str]) -> "BinningSpec":
        """Return the specification of some of the columns.

        Use it to keep only the variables of the final model before saving
        the specification for production. The selected columns keep their
        edges, special codes, tables and unseen policies.

        Parameters
        ----------
        columns : sequence of str
            Columns to keep, in the order of the new specification.

        Returns
        -------
        BinningSpec
            A specification of `columns` only.

        Raises
        ------
        ValueError
            If `columns` is a string, is empty, repeats a column or names a
            column that is not in the specification.

        See Also
        --------
        columns : The columns of the specification.
        to_json : Save the specification.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52], "housing": ["yes", "no", "yes", "no"]})
        >>> fit_spec(frame, [1, 0, 0, 1], bins=2).select(["housing"])
        BinningSpec(columns=['housing'])
        """
        names = _column_list(columns, "columns")
        absent = [name for name in names if name not in self.kinds]
        if absent:
            raise ValueError(f"columns must be in the specification, got {_shown_list(absent)}")
        numeric = [name for name in names if self.kinds[name] == "numeric"]
        return BinningSpec(
            kinds={name: self.kinds[name] for name in names},
            edges={name: self.edges[name] for name in numeric},
            special={name: self.special[name] for name in numeric},
            tables={name: self.tables[name] for name in names},
            unseen={name: self.unseen[name] for name in names},
            smoothing=self.smoothing,
            kivi_version=self.kivi_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the specification as a dict of JSON types.

        The result holds only dicts, lists, strings, numbers, booleans and
        None, so ``json.dumps`` can write it without further conversion. Its
        keys are always in the same order. `to_json` writes this dict.

        Returns
        -------
        dict
            The keys ``format``, ``format_version``, ``kivi_version``,
            ``smoothing`` and ``variables``, a list with one dict per column.
            See the user guide for the format.

        See Also
        --------
        from_dict : Rebuild the specification.
        to_json : Write the dict as JSON.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52]})
        >>> data = fit_spec(frame, [1, 0, 0, 1], bins=2).to_dict()
        >>> list(data)
        ['format', 'format_version', 'kivi_version', 'smoothing', 'variables']
        >>> data["variables"][0]["edges"]
        [{'float': '-inf'}, 38.0, {'float': 'inf'}]
        """
        variables = []
        for name, kind in self.kinds.items():
            entry: dict[str, Any] = {"name": name, "kind": kind}
            if kind == "numeric":
                entry["edges"] = [_encode(float(edge)) for edge in self.edges[name]]
                entry["special"] = [_encode(code) for code in self.special[name]]
            entry["unseen"] = self.unseen[name]
            entry["table"] = _encode_table(self.tables[name])
            variables.append(entry)
        return {
            "format": _FORMAT,
            "format_version": self.format_version,
            "kivi_version": self.kivi_version,
            "smoothing": self.smoothing,
            "variables": variables,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BinningSpec":
        """Rebuild a specification from the result of `to_dict`.

        The dict is checked before the specification is built, so that a
        damaged or hand-edited file fails with a message that names the
        problem instead of encoding wrong values.

        Parameters
        ----------
        data : mapping
            A dict in the format of `to_dict`, for example from
            ``json.load``.

        Returns
        -------
        BinningSpec
            The specification, equal to the one that wrote `data`.

        Raises
        ------
        ValueError
            If `data` is not a kivi binning specification, has a
            ``format_version`` newer than this version of kivi reads, lacks a
            key or has an unknown one, or holds an invalid value. The message
            names the column.

        See Also
        --------
        to_dict : The dict format.
        from_json : Read the specification from JSON text.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import BinningSpec, fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52]})
        >>> spec = fit_spec(frame, [1, 0, 0, 1], bins=2)
        >>> BinningSpec.from_dict(spec.to_dict()) == spec
        True
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"data must be a mapping, got {type(data).__name__}")
        if data.get("format") != _FORMAT:
            raise ValueError(
                f"data is not a kivi binning specification: format must be {_FORMAT!r}, "
                f"got {data.get('format')!r}"
            )
        version = data.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"format_version must be a positive integer, got {version!r}")
        if version > _FORMAT_VERSION:
            raise ValueError(
                f"format_version {version} is newer than {_FORMAT_VERSION}, the newest that "
                f"kivi {kivi.__version__} reads; upgrade kivi to load this specification"
            )
        _check_keys(
            data, {"format", "format_version", "kivi_version", "smoothing", "variables"}, ""
        )
        variables = data["variables"]
        if not isinstance(variables, list):
            raise ValueError(f"variables must be a list, got {type(variables).__name__}")
        kinds, edges, special, tables, unseen = {}, {}, {}, {}, {}
        for entry in variables:
            if not isinstance(entry, Mapping):
                raise ValueError(f"each variable must be a mapping, got {type(entry).__name__}")
            name = entry.get("name")
            if not isinstance(name, str):
                raise ValueError(f"each variable must have a string name, got {name!r}")
            if name in kinds:
                raise ValueError(f"variables must not repeat a name, got {name!r} twice")
            kind = entry.get("kind")
            where = f"column {name!r}"
            if kind == "numeric":
                _check_keys(entry, {"name", "kind", "edges", "special", "unseen", "table"}, where)
                edges[name] = _decode_list(entry["edges"], f"edges of {where}")
                special[name] = tuple(_decode_list(entry["special"], f"special codes of {where}"))
            elif kind == "categorical":
                _check_keys(entry, {"name", "kind", "unseen", "table"}, where)
            else:
                raise ValueError(f"kind of {where} must be one of {_KINDS}, got {kind!r}")
            kinds[name] = kind
            unseen[name] = entry["unseen"]
            tables[name] = _decode_table(entry["table"], where)
        return cls(
            kinds=kinds,
            edges=edges,
            special=special,
            tables=tables,
            unseen=unseen,
            smoothing=data["smoothing"],
            kivi_version=data["kivi_version"],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Return the specification as JSON text.

        The text is the dict of `to_dict` written by ``json.dumps``. The same
        specification always gives the same text, so saved specifications
        can be compared and kept under version control. Non-ASCII labels are
        written as they are, so save the text as UTF-8.

        Parameters
        ----------
        indent : int, optional
            Indentation of nested values, as in ``json.dumps``. By default
            the text is a single line.

        Returns
        -------
        str
            JSON text that `from_json` reads back into an equal
            specification.

        See Also
        --------
        from_json : Read the specification back.
        to_dict : The same content as a dict.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52]})
        >>> spec = fit_spec(frame, [1, 0, 0, 1], bins=2)
        >>> print(spec.to_json(indent=2))
        {
          "format": "kivi.BinningSpec",
          "format_version": 1,
          "kivi_version": "...",
          "smoothing": 0.5,
          "variables": [
            {
              "name": "age",
              "kind": "numeric",
              "edges": [
                {
                  "float": "-inf"
                },
                38.0,
        ...

        Write the text to a file with
        ``Path("spec.json").write_text(text, encoding="utf-8")``.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)

    @classmethod
    def from_json(cls, text: str | bytes) -> "BinningSpec":
        """Read a specification from the JSON text of `to_json`.

        The loaded specification groups and encodes every sample exactly as
        the one that was saved. A file written by a newer version of the
        format is rejected rather than read in part.

        Parameters
        ----------
        text : str or bytes
            JSON text, for example from ``Path(...).read_text(encoding="utf-8")``.

        Returns
        -------
        BinningSpec
            The specification, equal to the one that wrote `text`.

        Raises
        ------
        ValueError
            If `text` is not valid JSON, contains ``NaN`` or ``Infinity``,
            or does not describe a valid specification; see `from_dict`.

        See Also
        --------
        to_json : Write the specification.
        from_dict : Rebuild the specification from a dict.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.spec import BinningSpec, fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52]})
        >>> spec = fit_spec(frame, [1, 0, 0, 1], bins=2)
        >>> loaded = BinningSpec.from_json(spec.to_json())
        >>> loaded == spec
        True
        >>> loaded.woe_transform(frame).equals(spec.woe_transform(frame))
        True
        """
        return cls.from_dict(json.loads(text, parse_constant=_reject_constant))

    def _sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
        absent = [name for name in self.kinds if name not in frame.columns]
        if absent:
            raise ValueError(f"frame lacks columns of the specification: {_shown_list(absent)}")
        repeated = [name for name in self.kinds if (frame.columns == name).sum() > 1]
        if repeated:
            raise ValueError(f"frame has more than one column named {_shown_list(repeated)}")
        return frame


def fit_spec(
    frame: pd.DataFrame,
    target: npt.ArrayLike,
    *,
    columns: Sequence[str] | None = None,
    method: str | Mapping[str, str] = "quantile",
    bins: int | Mapping[str, int | None | npt.ArrayLike] = 5,
    special: Mapping[str, npt.ArrayLike] | None = None,
    options: Mapping[str, Mapping[str, Any]] | None = None,
    unseen: float | Literal["raise"] | Mapping[str, float | Literal["raise"]] = "raise",
    smoothing: float = 0.5,
    weights: npt.ArrayLike | None = None,
) -> BinningSpec:
    """Fit the binning and WOE of every column of a development sample.

    Numeric columns are binned with the method chosen for them and grouped
    with `kivi.binning.cut`; other columns keep their labels as groups. The
    WOE of each group is then measured with `kivi.woe.woe_table`. Nothing is
    re-binned automatically: the method, the number of bins and the options
    of each column decide how coarse its bins are.

    Parameters
    ----------
    frame : pandas.DataFrame
        Development sample with one column per variable. Columns with a
        numeric dtype other than bool are binned. Columns of any other dtype,
        such as object, string, category or bool, are categorical: each label
        is a group. Convert integer codes to the category dtype to keep them
        as labels.
    target : array_like
        Binary labels, with 1 indicating the event, one per row of `frame`.
        A Series must have the same index as `frame`; other inputs are
        matched by position.
    columns : sequence of str, optional
        Columns to fit, in the order of the specification. By default every
        column of `frame`.
    method : str or dict, default "quantile"
        Binning method of the numeric columns: ``"quantile"`` or
        ``"uniform"`` for `kivi.binning.bin_edges`, or ``"chimerge"``,
        ``"tree"`` or ``"monotonic"`` for `kivi.binning.chimerge_edges`,
        `kivi.binning.tree_edges` or `kivi.binning.monotonic_edges`. A dict
        maps columns to methods, and the numeric columns it leaves out use
        ``"quantile"``.
    bins : int or dict, default 5
        Number of bins, passed as ``bins`` to `kivi.binning.bin_edges` and as
        ``max_bins`` to the supervised methods. A dict maps columns to a
        number of bins or to edges, which must increase from ``-inf`` to
        ``inf`` and are used as given. With ``"monotonic"``, None merges bins
        only until the WOE is monotonic. The numeric columns that a dict
        leaves out get 5 bins.
    special : dict, optional
        Special codes of numeric columns, such as ``{"pdays": [-1]}``. They
        are left out of the fit, and each code forms its own group.
    options : dict, optional
        Further keyword arguments for the binning function of each column,
        such as ``{"balance": {"alpha": None}}`` with ``"chimerge"``. The
        special codes, the weights, the number of bins and, for
        ``"monotonic"``, the smoothing are passed by `fit_spec` and cannot be
        given here.
    unseen : float, "raise" or dict, default "raise"
        WOE for values whose group is not in a column's table when a sample
        is encoded: a new category, a special code or an interval without
        development observations, or missing values when development had
        none. By default such values raise an error. A dict maps columns to
        a WOE or ``"raise"``, and the columns it leaves out raise.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating the WOE, as in
        `kivi.woe.woe_table`. ``"monotonic"`` binning uses the same value, so
        its trend holds in the tables.
    weights : array_like, optional
        Non-negative weight of each row, matched like `target`, for the
        binning functions and `kivi.woe.woe_table`. Counts are then sums of
        weights, and `smoothing` is in units of weight.

    Returns
    -------
    BinningSpec
        The kind, edges, special codes, WOE table and unseen policy of every
        column.

    Raises
    ------
    ValueError
        If `frame` is not a DataFrame or has no columns to fit, `columns`
        repeats a column or names one that `frame` lacks, a column name is
        not a string, the columns include the target, or `target` does not
        give each row a label 0 or 1 with both classes present. If a dict
        names a column that is not fitted, `method`, `bins`, `special` or
        `options` names a categorical column, or a value is invalid. Also if
        `smoothing` or `weights` are invalid, or the binning of a column
        fails, in which case the message names the column.

    Warns
    -----
    UserWarning
        If tied quantiles leave a column with fewer bins than requested, as
        in `kivi.binning.bin_edges`, or if a numeric column has no values to
        fit apart from missing, special and zero-weight values, so that its
        edges are ``[-inf, inf]``. The message names the column.

    See Also
    --------
    BinningSpec : Group, encode and save the fitted columns.
    kivi.binning.bin_edges : Fit the edges of one column.
    kivi.woe.woe_table : The WOE table of one column.

    Notes
    -----
    Each column is fitted on its own with the public functions, so a column
    of the specification has the same edges and table as the steps of
    :doc:`/user_guide/binning` and :doc:`/user_guide/woe_iv` applied to it
    alone.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.spec import fit_spec
    >>> frame = pd.DataFrame({
    ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
    ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
    ... })
    >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    >>> spec = fit_spec(frame, target, bins=2)
    >>> spec.tables["age"][["non_event", "event", "woe"]].round(4)
                  non_event  event     woe
    bin
    (-inf, 41.5]          1      3  1.4553
    (41.5, inf]           4      0 -1.7228
    >>> spec.woe_transform(frame).head(3).round(4)
          age  housing
    0  1.4553   0.5108
    1  1.4553  -0.5878
    2 -1.7228   0.5108
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    names = _fitted_columns(frame, columns)
    labels = _target_labels(target, frame, names)
    _check_smoothing(smoothing)
    mass = as_weights(weights, frame.index.to_series(), of="frame")
    if mass is not None and not (np.bincount(labels, weights=mass, minlength=2) > 0).all():
        raise ValueError("weights must give both target classes a positive total")
    kinds = {name: "numeric" if _is_numeric(frame[name].dtype) else "categorical" for name in names}
    numeric = [name for name in names if kinds[name] == "numeric"]
    if not isinstance(bins, Mapping) and np.ndim(bins) != 0:
        raise ValueError(f"bins must be an integer or a dict by column, got {bins!r}")
    methods = _per_column(method, numeric, kinds, "method", "quantile", _checked_method)
    counts = _per_column(bins, numeric, kinds, "bins", 5)
    codes = _per_column({} if special is None else special, numeric, kinds, "special", ())
    codes = {
        name: _checked_codes(value, f"special of column {name!r}") for name, value in codes.items()
    }
    extra = _per_column({} if options is None else options, numeric, kinds, "options", {})
    policies = _per_column(unseen, names, kinds, "unseen", "raise", _checked_unseen)
    edges = {}
    for name in numeric:
        if isinstance(method, Mapping) and name in method and np.ndim(counts[name]) == 1:
            raise ValueError(f"method must not name column {name!r}, whose edges bins gives")
        fitter = _fitter(name, methods[name], counts[name], extra[name])
        edges[name] = _fit_edges(
            frame[name], labels, fitter, codes[name], smoothing, mass, name, methods[name]
        )
    tables = {}
    for name in names:
        groups = frame[name]
        try:
            if name in edges:
                groups = binning.cut(groups, edges[name], special=list(codes[name]) or None)
            tables[name] = _canonical_table(
                woe.woe_table(groups, labels, smoothing=smoothing, weights=mass), "labels"
            )
        except ValueError as error:
            raise ValueError(f"column {name!r}: {error}") from error
    return BinningSpec(
        kinds=kinds,
        edges=edges,
        special={name: codes[name] for name in numeric},
        tables=tables,
        unseen=policies,
        smoothing=smoothing,
    )


class _Tables(Mapping):
    """Read-only mapping that returns a copy of a table on each access."""

    __slots__ = ("_tables",)

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables = tables

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self._tables[name].copy()

    def __iter__(self) -> Iterator[str]:
        return iter(self._tables)

    def __len__(self) -> int:
        return len(self._tables)

    def __repr__(self) -> str:
        return f"<tables of {list(self._tables)!r}>"


def _fitted_columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        names = list(frame.columns)
        if not names:
            raise ValueError("frame must have at least one column to fit")
        for name in names:
            if not isinstance(name, str):
                raise ValueError(f"column names must be strings, got {name!r}")
    else:
        names = _column_list(columns, "columns")
        absent = [name for name in names if name not in frame.columns]
        if absent:
            raise ValueError(f"columns must be columns of frame, got {_shown_list(absent)}")
    repeated = [name for name in names if (frame.columns == name).sum() > 1]
    if repeated:
        raise ValueError(f"frame has more than one column named {_shown_list(repeated)}")
    return names


def _column_list(columns: Sequence[str], argument: str) -> list[str]:
    if isinstance(columns, str) or not isinstance(columns, (Sequence, pd.Index)):
        raise ValueError(
            f"{argument} must be a sequence of column names, got {type(columns).__name__}"
        )
    names = list(columns)
    if not names:
        raise ValueError(f"{argument} must name at least one column")
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"column names must be strings, got {name!r}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{argument} must not repeat a column, got {_shown_list(repeated)}")
    return names


def _target_labels(target: npt.ArrayLike, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    if isinstance(target, pd.Series):
        if target.name is not None and target.name in names:
            raise ValueError(
                f"columns include the target {target.name!r}; leave it out of the fitted columns"
            )
        if not target.index.equals(frame.index):
            raise ValueError("frame and target must have identical indices")
    series = as_series(target, "target")
    if len(series) != len(frame):
        raise ValueError(
            f"frame and target must have the same length, got {len(frame)} and {len(series)}"
        )
    if not series.isin([0, 1]).all() or series.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    return series.to_numpy(dtype=np.intp)


def _is_numeric(dtype) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _per_column(value, names, kinds, argument, default, check=None):
    """Resolve a setting of the columns ``names``: one value for all or a dict by column."""
    if not isinstance(value, Mapping):
        if argument in ("special", "options"):
            raise ValueError(f"{argument} must be a dict by column, got {type(value).__name__}")
        value = check(value, argument) if check is not None else value
        return dict.fromkeys(names, value)
    unknown = [key for key in value if key not in kinds]
    if unknown:
        raise ValueError(f"{argument} names columns that are not fitted: {_shown_list(unknown)}")
    categorical = [key for key in value if key not in names]
    if categorical:
        raise ValueError(
            f"{argument} names categorical columns, which keep their labels: "
            f"{_shown_list(categorical)}"
        )
    resolved = {}
    for name in names:
        if name in value:
            setting = value[name]
            resolved[name] = (
                check(setting, f"{argument} of column {name!r}") if check is not None else setting
            )
        else:
            resolved[name] = default
    return resolved


def _checked_method(value, argument: str) -> str:
    if not isinstance(value, str) or value not in _FITTERS:
        names = ", ".join(repr(name) for name in _FITTERS)
        raise ValueError(f"{argument} must be one of {names}, got {value!r}")
    return value


def _fitter(name: str, method: str, bins, extra):
    """Explicit edges, or the binning function of a column and its keyword arguments."""
    if np.ndim(bins) == 1 and not isinstance(bins, str):
        if extra:
            raise ValueError(f"options of column {name!r} must be empty when bins gives its edges")
        return _checked_edges(bins, f"edges of column {name!r}")
    if not (
        (bins is None and method == "monotonic")
        or (isinstance(bins, Integral) and not isinstance(bins, bool) and bins >= 1)
    ):
        allowed = "a positive integer, None" if method == "monotonic" else "a positive integer"
        raise ValueError(f"bins of column {name!r} must be {allowed} or edges, got {bins!r}")
    if not isinstance(extra, Mapping):
        raise ValueError(f"options of column {name!r} must be a dict, got {type(extra).__name__}")
    function = _FITTERS[method]
    accepted = {
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind is parameter.KEYWORD_ONLY
    } - _SET_BY_FIT
    rejected = [key for key in extra if key not in accepted]
    if rejected:
        accepts = ", ".join(sorted(accepted)) if accepted else "no options"
        raise ValueError(
            f"options of column {name!r} must be arguments of {function.__name__} other than "
            f"those fit_spec sets ({accepts}), got {_shown_list(rejected)}"
        )
    if method in ("quantile", "uniform"):
        return function, {"bins": bins, "method": method}
    return function, {"max_bins": bins, **extra}


def _fit_edges(values, labels, fitter, codes, smoothing, mass, name, method):
    if isinstance(fitter, np.ndarray):
        return fitter
    function, arguments = fitter
    observed = values.notna().to_numpy(dtype=bool)
    if codes:
        observed = observed & ~values.isin(list(codes)).to_numpy(dtype=bool)
    if mass is not None:
        observed = observed & (mass > 0)
    if not observed.any():
        warnings.warn(
            f"column {name!r} has no values to fit apart from missing, special and zero-weight "
            "values; its edges are [-inf, inf]",
            UserWarning,
            stacklevel=3,
        )
        return np.array([-np.inf, np.inf])
    arguments = {**arguments, "special": list(codes) or None, "weights": mass}
    if method == "monotonic":
        arguments["smoothing"] = smoothing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if function is binning.bin_edges:
                edges = function(values, **arguments)
            else:
                edges = function(values, labels, **arguments)
        except ValueError as error:
            raise ValueError(f"column {name!r}: {error}") from error
    for warning in caught:
        warnings.warn(f"column {name!r}: {warning.message}", warning.category, stacklevel=3)
    return edges


def _check_smoothing(smoothing) -> None:
    if (
        isinstance(smoothing, bool)
        or not isinstance(smoothing, Real)
        or not (math.isfinite(smoothing) and smoothing > 0)
    ):
        raise ValueError(f"smoothing must be finite and positive, got {smoothing!r}")


def _checked_kinds(kinds) -> dict[str, str]:
    if not isinstance(kinds, Mapping) or not kinds:
        raise ValueError("kinds must be a non-empty mapping from column names to kinds")
    for name, kind in kinds.items():
        if not isinstance(name, str):
            raise ValueError(f"column names must be strings, got {name!r}")
        if kind not in _KINDS:
            raise ValueError(f"kind of column {name!r} must be one of {_KINDS}, got {kind!r}")
    return dict(kinds)


def _entries(mapping, names: list[str], argument: str, kind: str = "") -> dict:
    """Return the entries of a mapping in the order of ``names``, which must be its keys."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{argument} must be a mapping, got {type(mapping).__name__}")
    missing = [name for name in names if name not in mapping]
    extra = [key for key in mapping if key not in set(names)]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"no entry for {_shown_list(missing)}")
        if extra:
            problems.append(f"an entry for {_shown_list(extra)}, which are not {kind}columns")
        raise ValueError(
            f"{argument} must have one entry per {kind}column, but has " + " and ".join(problems)
        )
    return {name: mapping[name] for name in names}


def _checked_edges(value, argument: str) -> npt.NDArray[np.float64]:
    try:
        edges = np.array(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{argument} must be numbers, got {value!r}") from error
    with np.errstate(invalid="ignore"):
        increasing = edges.ndim == 1 and len(edges) >= 2 and bool(np.all(np.diff(edges) > 0))
    if (
        not increasing
        or edges[0] != -np.inf
        or edges[-1] != np.inf
        or not np.isfinite(edges[1:-1]).all()
    ):
        raise ValueError(f"{argument} must increase from -inf to inf, got {np.asarray(value)!r}")
    edges.flags.writeable = False
    return edges


def _checked_codes(value, argument: str) -> tuple:
    if isinstance(value, str):
        raise ValueError(f"{argument} must be a list of codes, got {value!r}")
    codes = []
    for code in pd.unique(as_series(value, argument)):
        label = _label(code, argument)
        if isinstance(label, pd.Interval) or (isinstance(label, float) and math.isnan(label)):
            raise ValueError(f"{argument} must be numbers or strings, not missing, got {code!r}")
        codes.append(label)
    return tuple(codes)


def _checked_unseen(value, argument: str) -> float | str:
    if isinstance(value, str) and value == "raise":
        return value
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{argument} must be 'raise' or a finite number, got {value!r}")
    return float(value)


def _checked_table(table, name: str, numeric: tuple | None) -> pd.DataFrame:
    where = f"table of column {name!r}"
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{where} must be a DataFrame, got {type(table).__name__}")
    if "woe" not in table.columns:
        raise ValueError(f"{where} must have a woe column, as returned by woe_table")
    for column in table.columns:
        if not isinstance(column, str) or column == "bin":
            raise ValueError(f"{where} must have string column names other than 'bin'")
        values = table[column]
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all()
        ):
            raise ValueError(f"{where} must hold finite numbers, but column {column!r} does not")
    table = _canonical_table(table, f"labels of the {where}")
    labels = table.index
    missing = pd.isna(labels)
    if missing.sum() > 1 or labels[~missing].has_duplicates:
        raise ValueError(f"{where} must have one row per group")
    if numeric is not None:
        edges, codes = numeric
        groups = list(codes) + [
            pd.Interval(float(left), float(right), closed="right")
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ]
        observed = labels[~missing]
        foreign = observed[pd.Index(groups, dtype=object).get_indexer(observed) < 0].tolist()
        if foreign:
            raise ValueError(
                f"{where} has groups that are neither special codes nor intervals of its "
                f"edges: {_shown_list(foreign)}"
            )
    return table


def _canonical_table(table: pd.DataFrame, where: str) -> pd.DataFrame:
    """Copy a table and give it an index of Python labels, with NaN for the missing group."""
    labels = [_label(label, where) for label in table.index]
    table = table.copy()
    table.index = pd.Index(labels, dtype=object, name="bin")
    return table


def _label(value, where: str):
    """Convert a label to a Python str, int, float, bool or interval, or NaN when missing."""
    if isinstance(value, pd.Interval):
        left, right = (_label(bound, where) for bound in (value.left, value.right))
        if not all(_is_number(bound) for bound in (left, right)):
            raise ValueError(f"intervals in {where} must have numeric bounds, got {value!r}")
        return pd.Interval(left, right, closed=value.closed)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return np.nan
    raise ValueError(
        f"{where} must be strings, integers, floats, booleans or intervals, "
        f"got {value!r} of type {type(value).__name__}"
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _encode(value):
    """Encode a label for JSON: NaN as null, infinities and intervals as tagged objects."""
    if isinstance(value, pd.Interval):
        return {"interval": [_encode(value.left), _encode(value.right)], "closed": value.closed}
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
    return value


def _decode(value, where: str):
    if value is None:
        return np.nan
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Mapping):
        if set(value) == {"float"} and value["float"] in ("inf", "-inf"):
            return float(value["float"])
        if set(value) == {"interval", "closed"}:
            bounds = value["interval"]
            if isinstance(bounds, list) and len(bounds) == 2:
                left, right = (_decode(bound, where) for bound in bounds)
                closed = value["closed"]
                if _is_number(left) and _is_number(right) and closed in _CLOSED:
                    return pd.Interval(left, right, closed=closed)
    raise ValueError(f"{where} has an invalid value {value!r}")


_CLOSED = ("right", "left", "both", "neither")


def _decode_list(values, where: str) -> list:
    if not isinstance(values, list):
        raise ValueError(f"{where} must be a list, got {type(values).__name__}")
    return [_decode(value, where) for value in values]


def _encode_table(table: pd.DataFrame) -> list[dict[str, Any]]:
    columns = {column: table[column].tolist() for column in table.columns}
    return [
        {"bin": _encode(label), **{column: values[row] for column, values in columns.items()}}
        for row, label in enumerate(table.index)
    ]


def _decode_table(rows, where: str) -> pd.DataFrame:
    where = f"table of {where}"
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{where} must be a non-empty list of groups")
    keys = None
    for row in rows:
        if not isinstance(row, Mapping) or "bin" not in row:
            raise ValueError(f"{where} must have a mapping with a bin for each group")
        if keys is None:
            keys = list(row)
        elif list(row) != keys:
            raise ValueError(f"{where} must have the same keys for every group")
        for key, value in row.items():
            if key != "bin" and not _is_number(value):
                raise ValueError(f"{where} must hold numbers, got {key}={value!r}")
    columns = [key for key in keys if key != "bin"]
    labels = [_decode(row["bin"], where) for row in rows]
    return pd.DataFrame(
        {column: [row[column] for row in rows] for column in columns},
        index=pd.Index(labels, dtype=object, name="bin"),
    )


def _check_keys(data: Mapping, expected: set[str], where: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected, key=str)
    if missing or unknown:
        subject = f"{where} of the specification" if where else "the specification"
        problems = []
        if missing:
            problems.append(f"lacks {_shown_list(missing)}")
        if unknown:
            problems.append(f"has unknown keys {_shown_list(unknown)}")
        raise ValueError(f"{subject} " + " and ".join(problems))


def _reject_constant(name: str):
    raise ValueError(
        f"JSON must not contain {name}; kivi writes missing labels as null and infinite "
        'numbers as {"float": "inf"}'
    )


def _shown(value) -> str:
    if isinstance(value, (np.generic,)):
        value = value.item()
    return repr(value)


def _shown_list(values) -> str:
    return "[" + ", ".join(_shown(value) for value in values) + "]"
