"""Time-window features of event data, computed point in time.

``window_features`` aggregates the events of each entity, such as the
transactions of a customer, over windows that end at an observation time,
such as the date of an application: the number of transactions in the last
three months, their total, the days since the last one. Only events before
the observation time count, so a feature never uses what was not yet known.
A `FeatureSpec` keeps the definitions of the features and saves them as
JSON, so that development and production compute the same numbers.
"""

import dataclasses
import json
import re
import types
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

import kivi

from . import _validation

__all__ = ["FeatureSpec", "window_features"]

_FORMAT = "kivi.FeatureSpec"
# The newest version of the JSON format that from_dict reads and to_dict writes.
_FORMAT_VERSION = 1
_AGGREGATIONS = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "nunique",
    "first",
    "last",
    "days_since_last",
)
# Aggregations of the values themselves, which must therefore be numbers.
_NUMERIC = ("sum", "mean", "min", "max", "std", "first", "last")
# Aggregations that reduce every value of a window rather than two cumulative sums.
_REDUCED = ("min", "max", "std", "nunique")
_DEFAULT_WINDOWS = ("3M", "6M", "12M")
_BOUND = re.compile(r"([1-9][0-9]*)([MD])")
_GROUP_KEYS = ("table", "time", "windows", "aggregations", "closed", "prefix")
_DEFAULT_TABLE = "events"
_NANOSECONDS_PER_DAY = 86_400 * 10**9


class _Window(typing.NamedTuple):
    """A window: its definition, its label in column names and its two bounds.

    A bound is a number of months or days before the observation time, and
    an ``end`` of None is the observation time itself.
    """

    definition: str | tuple[str, str]
    label: str
    start: tuple[int, str]
    end: tuple[int, str] | None


class _Group(typing.NamedTuple):
    """The checked definitions of one call of `window_features`."""

    table: str
    time: str
    windows: tuple[_Window, ...]
    aggregations: dict[str, tuple[str, ...]]
    closed: str
    prefix: str


class _Feature(typing.NamedTuple):
    """One column of the result; ``column`` is None for the number of events."""

    name: str
    stem: str
    window: _Window
    column: str | None
    aggregation: str | None


class _Ratio(typing.NamedTuple):
    name: str
    numerator: str
    denominator: str


class _Observed(typing.NamedTuple):
    """The observations, read once for every group of a specification."""

    ids: pd.Series
    instants: npt.NDArray[np.int64]
    local: pd.Series
    zone: typing.Any
    labels: pd.Index
    bounds: dict[tuple[int, str], npt.NDArray[np.int64]]


def window_features(
    events: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    id: str,
    time: str,
    observation_time: str,
    windows: abc.Sequence[str | tuple[str, str]] = _DEFAULT_WINDOWS,
    aggregations: abc.Mapping[str, abc.Sequence[str]] | None = None,
    ratios: abc.Sequence[tuple[str, str]] | None = None,
    closed: typing.Literal["left", "right"] = "left",
    prefix: str = "",
) -> pd.DataFrame:
    """Aggregate the events of each entity over windows before an observation time.

    For each row of `observations`, an entity and a time :math:`t`, every
    window collects the events of that entity in an interval that ends at
    or before :math:`t`, such as the last three months :math:`[t - 3M, t)`,
    and aggregates the values of the chosen columns: their number, sum,
    mean, extremes and so on. Events at or after :math:`t` are never used,
    so the features are known at the observation time.

    Parameters
    ----------
    events : pandas.DataFrame
        One row per event, such as a transaction, in any order, with the
        columns `id` and `time` and the columns to aggregate. Several events
        of an entity may share one time.
    observations : pandas.DataFrame
        One row per entity and observation time, with the columns `id` and
        `observation_time`, such as the applicants and their application
        dates. An entity may appear in several rows, for example once a
        month to compute its features over time, and two rows may be equal.
    id : str
        Column of both tables that names the entity, such as a customer
        number. Values are matched by equality, so ``1`` and ``"1"`` are
        different entities.
    time : str
        Column of `events` with the time of each event, of a datetime dtype.
    observation_time : str
        Column of `observations` with the time of each observation, of a
        datetime dtype, tz-naive or in the time zone of `time`.
    windows : sequence, default ("3M", "6M", "12M")
        Windows before the observation time. ``"3M"`` is the last three
        calendar months and ``"90D"`` the last 90 days. A pair such as
        ``("6M", "3M")`` is the earlier period from six to three months
        before the observation time; its two bounds must have the same unit
        and the first must be further back. A pair is one window only as an
        item of the list, as in ``windows=[("6M", "3M")]``:
        ``windows=("6M", "3M")`` is the two windows ``"6M"`` and ``"3M"``.
    aggregations : dict, optional
        The aggregations of each column of `events`, such as
        ``{"amount": ["count", "sum", "max"]}``, among ``"count"``,
        ``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"std"``,
        ``"nunique"``, ``"first"``, ``"last"`` and ``"days_since_last"``;
        see the Notes. By default, only the number of events of each window
        is computed.
    ratios : sequence of pairs of str, optional
        Ratios of two features, each a pair of column names of the result
        such as ``("amount_sum_3m", "amount_sum_12m")``. A ratio of a recent
        window to an earlier one, such as ``("amount_sum_3m",
        "amount_sum_6m_3m")``, measures the change from one period to the
        next.
    closed : {"left", "right"}, default "left"
        ``"left"`` takes the windows as :math:`[t - k, t)`, without the
        events at the observation time. ``"right"`` takes them as
        :math:`(t - k, t]`, for events that are known at their own time,
        such as balances at the end of a day; whether they are known at the
        observation time is the caller's judgment. A pair of windows is
        closed on the same side.
    prefix : str, default ""
        Text put in front of every column name, as given, such as
        ``"cash_"`` for the features of cash withdrawals. Ratios name columns
        with the prefix.

    Returns
    -------
    pandas.DataFrame
        One row per row of `observations`, on its index and in its order.
        For each window in turn, the column ``{prefix}events_{window}`` and
        then, column by column and aggregation by aggregation, the columns
        ``{prefix}{column}_{aggregation}_{window}``, such as
        ``amount_sum_3m`` or ``amount_sum_6m_3m``. The prefix and the column
        names are kept as given, and only kivi's own parts, the aggregations
        and the windows, are in lower case, so ``Amount`` and ``amount`` give
        different columns. The ratios
        follow, named ``{numerator}_over_{window}`` when the numerator and
        the denominator differ only in their windows, such as
        ``amount_sum_3m_over_12m``, and ``{numerator}_over_{denominator}``
        otherwise. ``events``, ``count`` and ``nunique`` are integers and
        all other columns floats.

    Raises
    ------
    ValueError
        If `events` or `observations` is not a DataFrame, if `id`, `time` or
        `observation_time` is not the name of a single column of its table,
        if an id or a time is missing, if a time column does not hold
        datetimes, or if the two time columns are not both tz-naive or in
        one time zone. Also if a window, an aggregation, a ratio, `closed`
        or `prefix` is invalid or repeated, if `aggregations` names a column
        that `events` lacks, if a column with an aggregation other than
        ``count``, ``nunique`` and ``days_since_last`` holds values that are
        not numbers or are infinite, or if two columns of the result would
        have one name.

    See Also
    --------
    FeatureSpec : Keep the definitions and compute them again.
    kivi.univariate.profile_table : Screen the features against a target.
    kivi.datasets.make_applications : Applications with dates to observe.

    Notes
    -----
    A window of :math:`k` months starts :math:`k` calendar months before
    the observation time, as ``pandas.DateOffset(months=k)`` counts them:
    one month before 31 March is 29 February in a leap year, and the time
    of day is kept. A window of :math:`k` days starts :math:`k` days before.
    With time zones, the bounds are computed on the local calendar and the
    events compared as instants. A bound that falls into the hour skipped
    when clocks go forward moves to the end of that hour, and one in the
    hour repeated when they go back takes its first occurrence.

    The aggregations leave out the missing values of their column; the
    ``events`` columns count every event of the window. In a window
    without values, ``count``, ``sum`` and ``nunique`` are 0 and the other
    aggregations NaN:

    * ``count``: the number of values;
    * ``sum`` and ``mean``: their sum and mean;
    * ``min`` and ``max``: the smallest and the largest value;
    * ``std``: the sample standard deviation, as ``pandas.Series.std``
      gives it;
    * ``nunique``: the number of distinct values;
    * ``first`` and ``last``: the value of the earliest and of the latest
      event;
    * ``days_since_last``: the days from the latest event to the
      observation time.

    ``std`` is NaN for fewer than two values. ``first`` and ``last`` take
    events of one time in the order of the rows of `events`.
    ``days_since_last`` counts fractions of a day, so an event twelve hours
    before the observation is 0.5 days old; for a pair of windows it is
    still measured from the observation time. It is the time that has
    passed, so with a time zone it can differ by the hour that clocks change
    from the local calendar days that windows such as ``"90D"`` count.
    ``count``, ``nunique`` and
    ``days_since_last`` take columns of any dtype, such as the type of a
    merchant; the other aggregations need numbers, and booleans count as 0
    and 1. A ratio is NaN where its denominator is 0 or NaN.

    Entities of `observations` without events get the values of empty
    windows, and events of entities that `observations` lacks are ignored.
    The events are sorted once, and the bounds of every window are found by
    binary search. Counts, sums, means, first and last values and
    ``days_since_last`` then take a time that grows with the number of rows
    of the two tables, and a sum adds only values inside its window, so a
    large earlier value does not round away the small ones in it.
    ``min``, ``max``, ``std`` and ``nunique`` visit each event once for every
    window that contains it, so their time grows with the events of all
    windows together; they are expanded a bounded number at a time, so
    memory does not.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.features import window_features
    >>> events = pd.DataFrame({
    ...     "customer": [1, 1, 1, 2],
    ...     "time": pd.to_datetime(["2024-01-15", "2024-03-02", "2024-03-31", "2024-02-10"]),
    ...     "amount": [100.0, 40.0, 60.0, 80.0],
    ... })
    >>> observations = pd.DataFrame({
    ...     "customer": [1, 2, 3],
    ...     "date": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-03-31"]),
    ... })
    >>> window_features(
    ...     events, observations, id="customer", time="time", observation_time="date",
    ...     windows=["1M", "3M"], aggregations={"amount": ["sum"]},
    ... )
       events_1m  amount_sum_1m  events_3m  amount_sum_3m
    0          1           40.0          2          140.0
    1          0            0.0          1           80.0
    2          0            0.0          0            0.0

    The event of 31 March is on the observation date, so no window uses
    it, and the month before 31 March starts on 29 February.

    A pair of bounds is one window only as an item of the list of windows;
    on its own, it is two windows:

    >>> pair = window_features(
    ...     events, observations, id="customer", time="time", observation_time="date",
    ...     windows=[("3M", "1M")],
    ... )
    >>> pair
       events_3m_1m
    0             1
    1             1
    2             0
    >>> two = window_features(
    ...     events, observations, id="customer", time="time", observation_time="date",
    ...     windows=("3M", "1M"),
    ... )
    >>> list(two.columns)
    ['events_3m', 'events_1m']
    """
    group = _checked_group(
        table=_DEFAULT_TABLE,
        time=time,
        windows=windows,
        aggregations=aggregations,
        closed=closed,
        prefix=prefix,
        arguments={name: name for name in _GROUP_KEYS},
    )
    features = _features(group)
    checked = _checked_ratios(ratios, features, "ratios")
    _check_unique(features, checked, "the result", "remove the repeated ratio")
    observed = _observed(observations, id, observation_time)
    columns = _group_values(group, events, "events", id, observed)
    return _frame(columns, checked, observed.labels)


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class FeatureSpec:
    """Definitions of time-window features, to compute them the same way again.

    A specification records the columns that identify the entities and the
    observation times, and one or more groups of window features. A group
    holds the arguments of one call of `window_features`: the event table
    it reads, its time column, its windows, its aggregations, the side on
    which its windows are closed and the prefix of its column names. Groups
    that read filtered events, such as only the cash withdrawals, use their
    own table and prefix. `compute` gives the features of any sample, and
    `to_json` and `from_json` save and load the definitions. Nothing is
    fitted to data, so a specification can be written by hand.

    A specification is immutable: its groups are read-only mappings with
    tuples for lists. To change a field, use `dataclasses.replace`, which
    validates the result again.

    Parameters
    ----------
    id : str
        Column of the observations and of every event table that names the
        entity; all groups share it.
    observation_time : str
        Column of the observations with the observation times.
    events : sequence of dict
        One dict per group, with the keys ``time`` (required),
        ``windows``, ``aggregations``, ``closed`` and ``prefix``, which mean
        and default to what the arguments of `window_features` do, and
        ``table``, the name of the event table that the group reads,
        ``"events"`` by default.
    ratios : sequence of pairs of str, default ()
        Ratios of two features of any groups, named by their columns, as in
        `window_features`.
    panel : None, default None
        Reserved for the features of monthly status panels, which
        ``panel_features`` adds before kivi 1.3.0 is released, completing
        version 1 of the JSON format; until then it must be None.
    kivi_version : str, optional
        Version of kivi that wrote the specification. By default, or if
        None, the installed version.

    Attributes
    ----------
    FORMAT_VERSION : int
        Version of the JSON format that `to_json` writes and the newest
        that `from_json` reads.

    Raises
    ------
    ValueError
        If `id` or `observation_time` is not a string, if `events` is not a
        non-empty list of dicts with valid entries, if two features of the
        groups have one name, if a ratio is invalid, if `panel` is not None
        or if `kivi_version` is not a string. The checks of each entry are
        those of the arguments of `window_features`.

    See Also
    --------
    window_features : Compute the features of one group directly.
    kivi.spec.BinningSpec : The binning of the features, saved the same way.

    Notes
    -----
    Two specifications are equal when `to_json` gives the same text for
    both, which includes `kivi_version`.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.features import FeatureSpec
    >>> spec = FeatureSpec(
    ...     id="customer",
    ...     observation_time="date",
    ...     events=[{"time": "time", "windows": ["1M", "3M"], "aggregations": {"amount": ["sum"]}}],
    ...     ratios=[("amount_sum_1m", "amount_sum_3m")],
    ... )
    >>> spec
    FeatureSpec(columns=['events_1m', 'amount_sum_1m', 'events_3m', 'amount_sum_3m',
                         'amount_sum_1m_over_3m'])
    >>> events = pd.DataFrame({
    ...     "customer": [1, 1, 2],
    ...     "time": pd.to_datetime(["2024-01-15", "2024-03-02", "2024-02-10"]),
    ...     "amount": [100.0, 40.0, 80.0],
    ... })
    >>> observations = pd.DataFrame({
    ...     "customer": [1, 2],
    ...     "date": pd.to_datetime(["2024-03-31", "2024-03-31"]),
    ... })
    >>> spec.compute(observations, events=events)
       events_1m  amount_sum_1m  events_3m  amount_sum_3m  amount_sum_1m_over_3m
    0          1           40.0          2          140.0               0.285714
    1          0            0.0          1           80.0               0.000000
    """

    id: str
    observation_time: str
    events: abc.Sequence[abc.Mapping[str, typing.Any]]
    ratios: abc.Sequence[tuple[str, str]] = ()
    panel: None = None
    kivi_version: str | None = None
    FORMAT_VERSION: typing.ClassVar[int] = _FORMAT_VERSION

    def __post_init__(self) -> None:
        """Validate the fields and replace them with read-only copies."""
        for argument in ("id", "observation_time"):
            _check_name(getattr(self, argument), argument)
        entries = self.events
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError(
                "events must be a non-empty list of dicts, one per group of features, got "
                f"{_validation.shown_argument(entries)}"
            )
        groups = tuple(_entry_group(entry, number) for number, entry in enumerate(entries))
        features = [feature for group in groups for feature in _features(group)]
        ratios = _checked_ratios(self.ratios, features, "ratios")
        _check_unique(
            features,
            ratios,
            "the groups of events",
            "give the groups different prefixes or remove the repeated ratio",
        )
        if self.panel is not None:
            raise ValueError(
                f"panel must be None, got {_validation.shown_argument(self.panel)}; panel "
                "features are not available yet"
            )
        version = kivi.__version__ if self.kivi_version is None else self.kivi_version
        if not isinstance(version, str):
            raise ValueError(
                f"kivi_version must be a string, got {_validation.shown_argument(version)}"
            )
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "observation_time", str(self.observation_time))
        object.__setattr__(self, "events", tuple(_frozen_group(group) for group in groups))
        object.__setattr__(
            self, "ratios", tuple((ratio.numerator, ratio.denominator) for ratio in ratios)
        )
        object.__setattr__(self, "kivi_version", version)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_ratios", ratios)
        object.__setattr__(self, "_features", features)

    def __repr__(self) -> str:
        """Name the columns that `compute` returns."""
        return f"{type(self).__name__}(columns={list(self.columns)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare the JSON text."""
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        """Hash the JSON text, consistently with equality."""
        return hash(self.to_json())

    def __reduce__(
        self,
    ) -> tuple[
        abc.Callable[[abc.Mapping[str, typing.Any]], "FeatureSpec"], tuple[dict[str, typing.Any]]
    ]:
        """Pickle through the dict format, which validates the fields again."""
        return (type(self).from_dict, (self.to_dict(),))

    @property
    def columns(self) -> pd.Index:
        """Names of the columns that `compute` returns, in their order.

        The features of each group come in the order of the groups, and the
        ratios last, so the names can select or check the columns of a
        result before any data is read.

        See Also
        --------
        compute : Compute the columns.

        Examples
        --------
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(
        ...     id="customer",
        ...     observation_time="date",
        ...     events=[
        ...         {"time": "time", "windows": ["3M"], "aggregations": {"amount": ["sum"]}},
        ...         {"table": "cash", "time": "time", "windows": ["3M"], "prefix": "cash_"},
        ...     ],
        ... )
        >>> spec.columns
        Index(['events_3m', 'amount_sum_3m', 'cash_events_3m'], dtype='object')
        """
        features: list[_Feature] = self.__dict__["_features"]
        ratios: list[_Ratio] = self.__dict__["_ratios"]
        names = [feature.name for feature in features] + [ratio.name for ratio in ratios]
        return pd.Index(names, dtype=object)

    def compute(
        self,
        observations: pd.DataFrame,
        *,
        events: pd.DataFrame | abc.Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute the features of every observation.

        Each group gives the columns that `window_features` gives for its
        definitions and its event table, without ratios, and the ratios of
        the specification follow.

        Parameters
        ----------
        observations : pandas.DataFrame
            One row per entity and observation time, with the columns `id`
            and `observation_time` of the specification.
        events : pandas.DataFrame or dict of pandas.DataFrame
            The event table, or a dict from the ``table`` name of every
            group to its event table. A single table is allowed when all
            groups read the same table.

        Returns
        -------
        pandas.DataFrame
            The columns of `columns`, one row per row of `observations`, on
            its index and in its order.

        Raises
        ------
        ValueError
            If `events` is neither a DataFrame nor a dict of DataFrames with
            one entry per table of the groups, or for the reasons that
            `window_features` gives.

        See Also
        --------
        window_features : The features of one group.
        columns : The names of the columns.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(
        ...     id="customer",
        ...     observation_time="date",
        ...     events=[
        ...         {"time": "time", "windows": ["3M"], "aggregations": {"amount": ["sum"]}},
        ...         {"table": "cash", "time": "time", "windows": ["3M"],
        ...          "aggregations": {"amount": ["sum"]}, "prefix": "cash_"},
        ...     ],
        ...     ratios=[("cash_amount_sum_3m", "amount_sum_3m")],
        ... )
        >>> events = pd.DataFrame({
        ...     "customer": [1, 1, 1],
        ...     "time": pd.to_datetime(["2024-01-15", "2024-02-03", "2024-03-02"]),
        ...     "kind": ["purchase", "cash", "purchase"],
        ...     "amount": [100.0, 60.0, 40.0],
        ... })
        >>> observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-03-31"])})
        >>> tables = {"events": events, "cash": events[events["kind"] == "cash"]}
        >>> spec.compute(observations, events=tables).T
                                                0
        events_3m                             3.0
        amount_sum_3m                       200.0
        cash_events_3m                        1.0
        cash_amount_sum_3m                   60.0
        cash_amount_sum_3m_over_amount_sum_3m 0.3
        """
        groups: tuple[_Group, ...] = self.__dict__["_groups"]
        tables = _tables(events, list(dict.fromkeys(group.table for group in groups)))
        observed = _observed(observations, self.id, self.observation_time)
        columns: dict[str, npt.NDArray[typing.Any]] = {}
        for group in groups:
            table, where = tables[group.table]
            columns.update(_group_values(group, table, where, self.id, observed))
        return _frame(columns, self.__dict__["_ratios"], observed.labels)

    def to_dict(self) -> dict[str, typing.Any]:
        """Return the feature specification as a dict of JSON types.

        The dict holds the arguments of the constructor as the constructor
        takes them, with lists for tuples, so they rebuild the specification.
        It holds only dicts, lists, strings and None, so ``json.dumps`` can
        write it without further conversion, and its keys are always in the
        same order. `to_json` writes this dict.

        Returns
        -------
        dict
            The keys ``format``, ``format_version`` and ``kivi_version``,
            then the arguments ``id``, ``observation_time``, ``events``,
            ``ratios`` and ``panel``. Each group of ``events`` has all its
            keys: ``table``, ``time``, ``windows``, ``aggregations``, a dict
            from each column to the list of its aggregations in the order of
            the columns of the result, ``closed`` and ``prefix``.

        See Also
        --------
        from_dict : Rebuild the feature specification.
        to_json : Write the dict as JSON.

        Examples
        --------
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(
        ...     id="customer",
        ...     observation_time="date",
        ...     events=[{"time": "time", "windows": ["3M", ("6M", "3M")]}],
        ... )
        >>> data = spec.to_dict()
        >>> list(data)
        ['format', 'format_version', 'kivi_version', 'id', 'observation_time', 'events', 'ratios',
         'panel']
        >>> data["events"][0]
        {'table': 'events', 'time': 'time', 'windows': ['3M', ['6M', '3M']], 'aggregations': {},
         'closed': 'left', 'prefix': ''}
        >>> arguments = ["id", "observation_time", "events", "ratios", "panel", "kivi_version"]
        >>> FeatureSpec(**{name: data[name] for name in arguments}) == spec
        True
        """
        groups = []
        for group in self.__dict__["_groups"]:
            groups.append(
                {
                    "table": group.table,
                    "time": group.time,
                    "windows": [
                        list(window.definition)
                        if isinstance(window.definition, tuple)
                        else window.definition
                        for window in group.windows
                    ],
                    "aggregations": {
                        column: list(names) for column, names in group.aggregations.items()
                    },
                    "closed": group.closed,
                    "prefix": group.prefix,
                }
            )
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "kivi_version": self.kivi_version,
            "id": self.id,
            "observation_time": self.observation_time,
            "events": groups,
            "ratios": [list(pair) for pair in self.ratios],
            "panel": None,
        }

    @classmethod
    def from_dict(cls, data: abc.Mapping[str, typing.Any]) -> "FeatureSpec":
        """Rebuild a feature specification from the result of `to_dict`.

        The dict is checked before the specification is built, so that a
        damaged or hand-edited file fails with a message that names the
        problem.

        Parameters
        ----------
        data : mapping
            A dict in the format of `to_dict`, for example from
            ``json.load``. Every key that `to_dict` writes must be present,
            also in each group.

        Returns
        -------
        FeatureSpec
            The feature specification, equal to the one that wrote `data`.

        Raises
        ------
        ValueError
            If `data` is not a kivi feature specification, has a
            ``format_version`` newer than this version of kivi reads, lacks a
            key or has an unknown one, or holds an invalid value.

        See Also
        --------
        to_dict : The dict format.
        from_json : Read the feature specification from JSON text.

        Examples
        --------
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(id="customer", observation_time="date", events=[{"time": "time"}])
        >>> FeatureSpec.from_dict(spec.to_dict()) == spec
        True
        """
        if not isinstance(data, abc.Mapping):
            raise ValueError(f"data must be a mapping, got {type(data).__name__}")
        if data.get("format") != _FORMAT:
            raise ValueError(
                f"data is not a kivi feature specification: format must be {_FORMAT!r}, "
                f"got {_validation.shown(data.get('format'))}"
            )
        version = data.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(
                f"format_version must be a positive integer, got {_validation.shown(version)}"
            )
        if version > _FORMAT_VERSION:
            raise ValueError(
                f"format_version {version} is newer than {_FORMAT_VERSION}, the newest that "
                f"kivi {kivi.__version__} reads; upgrade kivi to load this specification"
            )
        keys = {"format", "format_version", "kivi_version", "id", "observation_time"}
        _check_keys(data, keys | {"events", "ratios", "panel"}, "the specification")
        if not isinstance(data["kivi_version"], str):
            raise ValueError(
                f"kivi_version must be a string, got {_validation.shown(data['kivi_version'])}"
            )
        entries = data["events"]
        if not isinstance(entries, list):
            raise ValueError(f"events must be a list, got {type(entries).__name__}")
        groups = []
        for number, entry in enumerate(entries):
            where = f"events[{number}]"
            if not isinstance(entry, abc.Mapping):
                raise ValueError(f"{where} must be a mapping, got {type(entry).__name__}")
            _check_keys(entry, set(_GROUP_KEYS), where)
            groups.append(entry)
        return cls(
            id=data["id"],
            observation_time=data["observation_time"],
            events=groups,
            ratios=data["ratios"],
            panel=data["panel"],
            kivi_version=data["kivi_version"],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Return the feature specification as JSON text.

        The text is the dict of `to_dict` written by ``json.dumps``. The same
        specification always gives the same text, so saved specifications
        can be compared and kept under version control. Non-ASCII names are
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
            feature specification.

        See Also
        --------
        from_json : Read the feature specification back.
        to_dict : The same content as a dict.

        Examples
        --------
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(
        ...     id="customer",
        ...     observation_time="date",
        ...     events=[{"time": "time", "windows": ["3M"], "aggregations": {"amount": ["sum"]}}],
        ... )
        >>> print(spec.to_json(indent=2))
        {
          "format": "kivi.FeatureSpec",
          "format_version": 1,
          "kivi_version": "...",
          "id": "customer",
          "observation_time": "date",
          "events": [
            {
              "table": "events",
              "time": "time",
              "windows": [
                "3M"
              ],
              "aggregations": {
                "amount": [
                  "sum"
                ]
              },
              "closed": "left",
              "prefix": ""
            }
          ],
          "ratios": [],
          "panel": null
        }

        Write the text to a file with
        ``Path("features.json").write_text(text, encoding="utf-8")``.
        """
        if indent is not None:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        # The specification is frozen, so the compact text can be kept for equality and hashing.
        text = self.__dict__.get("_compact_json")
        if text is None:
            text = json.dumps(self.to_dict(), ensure_ascii=False)
            object.__setattr__(self, "_compact_json", text)
        return text

    @classmethod
    def from_json(cls, text: str | bytes) -> "FeatureSpec":
        """Read a feature specification from the JSON text of `to_json`.

        A file written by a newer version of the format is rejected rather
        than read in part, and so is a JSON object that repeats a key, which
        ``json.loads`` would otherwise read as its last value.

        Parameters
        ----------
        text : str or bytes
            JSON text, for example from ``Path(...).read_text(encoding="utf-8")``.

        Returns
        -------
        FeatureSpec
            The feature specification, equal to the one that wrote `text`.

        Raises
        ------
        ValueError
            If `text` is neither a str nor bytes, is not valid JSON, repeats
            a key of an object, or does not describe a valid specification;
            see `from_dict`.

        See Also
        --------
        to_json : Write the feature specification.
        from_dict : Rebuild the feature specification from a dict.

        Examples
        --------
        >>> from kivi.features import FeatureSpec
        >>> spec = FeatureSpec(id="customer", observation_time="date", events=[{"time": "time"}])
        >>> FeatureSpec.from_json(spec.to_json()) == spec
        True
        """
        if not isinstance(text, (str, bytes, bytearray)):
            raise ValueError(
                f"text must be JSON text as a str or bytes, got {type(text).__name__}; read a "
                'file first, for example with Path(...).read_text(encoding="utf-8")'
            )
        return cls.from_dict(json.loads(text, object_pairs_hook=_unique_keys))


def _checked_group(
    *,
    table: object,
    time: object,
    windows: object,
    aggregations: object,
    closed: object,
    prefix: object,
    arguments: abc.Mapping[str, str],
) -> _Group:
    """Check the definitions of one group; ``arguments`` names each argument in messages."""
    _check_name(table, arguments["table"])
    _check_name(time, arguments["time"])
    if not isinstance(closed, str) or closed not in ("left", "right"):
        raise ValueError(
            f"{arguments['closed']} must be 'left' or 'right', got "
            f"{_validation.shown_argument(closed)}"
        )
    if not isinstance(prefix, str):
        raise ValueError(
            f"{arguments['prefix']} must be a string, got {_validation.shown_argument(prefix)}"
        )
    return _Group(
        table=str(table),
        time=str(time),
        windows=_checked_windows(windows, arguments["windows"]),
        aggregations=_checked_aggregations(aggregations, arguments["aggregations"]),
        closed=str(closed),
        prefix=str(prefix),
    )


def _entry_group(entry: object, number: int) -> _Group:
    """Check one entry of `FeatureSpec.events`."""
    where = f"events[{number}]"
    if not isinstance(entry, abc.Mapping):
        raise ValueError(f"{where} must be a dict, got {_validation.shown_argument(entry)}")
    unknown = [key for key in entry if key not in _GROUP_KEYS]
    if unknown:
        raise ValueError(
            f"{where} has unknown keys {_validation.shown_list(unknown)}; the keys are "
            f"{_validation.shown_list(_GROUP_KEYS)}"
        )
    if "time" not in entry:
        raise ValueError(f"{where} lacks 'time', the column with the time of each event")
    return _checked_group(
        table=entry.get("table", _DEFAULT_TABLE),
        time=entry["time"],
        windows=entry.get("windows", _DEFAULT_WINDOWS),
        aggregations=entry.get("aggregations"),
        closed=entry.get("closed", "left"),
        prefix=entry.get("prefix", ""),
        arguments={key: f"{key} of {where}" for key in _GROUP_KEYS},
    )


def _frozen_group(group: _Group) -> types.MappingProxyType[str, typing.Any]:
    """Return a group as the read-only mapping that `FeatureSpec.events` holds."""
    return types.MappingProxyType(
        {
            "table": group.table,
            "time": group.time,
            "windows": tuple(window.definition for window in group.windows),
            "aggregations": types.MappingProxyType(dict(group.aggregations)),
            "closed": group.closed,
            "prefix": group.prefix,
        }
    )


def _check_name(value: object, argument: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{argument} must be a string, got {_validation.shown_argument(value)}")


def _checked_windows(windows: object, argument: str) -> tuple[_Window, ...]:
    if not isinstance(windows, (list, tuple)):
        hint = f"; pass [{windows!r}] for one window" if isinstance(windows, str) else ""
        raise ValueError(
            f"{argument} must be a list of windows such as '3M', '90D' or ('6M', '3M'), got "
            f"{_validation.shown_argument(windows)}{hint}"
        )
    if not windows:
        raise ValueError(f"{argument} must have at least one window")
    checked = [_checked_window(window, argument) for window in windows]
    labels = [window.label for window in checked]
    for window in checked:
        if labels.count(window.label) > 1:
            raise ValueError(
                f"{argument} must not repeat a window, got {_validation.shown(window.definition)}"
            )
    return tuple(checked)


def _checked_window(window: object, argument: str) -> _Window:
    if isinstance(window, str):
        start = _bound(window, argument)
        return _Window(str(window), _bound_label(start), start, None)
    if (
        isinstance(window, (list, tuple))
        and len(window) == 2
        and all(isinstance(bound, str) for bound in window)
    ):
        start, end = _bound(window[0], argument), _bound(window[1], argument)
        if start[1] != end[1] or start[0] <= end[0]:
            raise ValueError(
                f"a pair of {argument} must reach further back with its first bound, in the "
                f"unit of its second, such as ('6M', '3M'), got "
                f"{_validation.shown_argument(tuple(window))}"
            )
        label = f"{_bound_label(start)}_{_bound_label(end)}"
        return _Window((str(window[0]), str(window[1])), label, start, end)
    raise ValueError(
        f"each window of {argument} must be a string such as '3M' or a pair such as "
        f"('6M', '3M'), got {_validation.shown_argument(window)}"
    )


def _bound(text: str, argument: str) -> tuple[int, str]:
    match = _BOUND.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{argument} must count months or days such as '3M' or '90D', got "
            f"{_validation.shown(text)}"
        )
    return int(match[1]), match[2]


def _bound_label(bound: tuple[int, str]) -> str:
    return f"{bound[0]}{bound[1].lower()}"


def _checked_aggregations(aggregations: object, argument: str) -> dict[str, tuple[str, ...]]:
    if aggregations is None:
        return {}
    if not isinstance(aggregations, abc.Mapping):
        raise ValueError(
            f"{argument} must be a dict from column names to lists of aggregations, got "
            f"{_validation.shown_argument(aggregations)}"
        )
    checked = {}
    for column, names in aggregations.items():
        if not isinstance(column, str):
            raise ValueError(f"column names must be strings, got {_validation.shown(column)}")
        if not isinstance(names, (list, tuple)):
            hint = f"; pass [{names!r}] for one aggregation" if isinstance(names, str) else ""
            raise ValueError(
                f"{argument} of column {_validation.shown(column)} must be a list such as "
                f"['sum', 'max'], got {_validation.shown_argument(names)}{hint}"
            )
        if not names:
            raise ValueError(
                f"{argument} of column {_validation.shown(column)} must name at least one "
                "aggregation"
            )
        for name in names:
            if not isinstance(name, str) or name not in _AGGREGATIONS:
                raise ValueError(
                    f"{argument} must be among {_validation.shown_list(_AGGREGATIONS)}, got "
                    f"{_validation.shown(name)}"
                )
            if names.count(name) > 1:
                raise ValueError(
                    f"{argument} of column {_validation.shown(column)} must not repeat an "
                    f"aggregation, got {_validation.shown(name)}"
                )
        checked[str(column)] = tuple(str(name) for name in names)
    return checked


def _features(group: _Group) -> list[_Feature]:
    """Return the columns of a group by window, then by column, then by aggregation."""
    features = []
    for window in group.windows:
        stem = f"{group.prefix}events"
        features.append(_Feature(f"{stem}_{window.label}", stem, window, None, None))
        for column, names in group.aggregations.items():
            for name in names:
                stem = f"{group.prefix}{column}_{name}"
                features.append(_Feature(f"{stem}_{window.label}", stem, window, column, name))
    return features


def _checked_ratios(ratios: object, features: list[_Feature], argument: str) -> list[_Ratio]:
    if ratios is None:
        return []
    if not isinstance(ratios, (list, tuple)):
        raise ValueError(
            f"{argument} must be a list of pairs of column names, got "
            f"{_validation.shown_argument(ratios)}"
        )
    known = {feature.name: feature for feature in features}
    checked = []
    for pair in ratios:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(name, str) for name in pair)
        ):
            raise ValueError(
                f"each ratio of {argument} must be a pair of column names such as "
                f"('amount_sum_3m', 'amount_sum_12m'), got {_validation.shown_argument(pair)}"
            )
        numerator, denominator = str(pair[0]), str(pair[1])
        for name in (numerator, denominator):
            if name not in known:
                raise ValueError(
                    f"{argument} must name features of the windows and aggregations, such as "
                    f"{_validation.shown(features[0].name)}, got {_validation.shown(name)}"
                )
        if numerator == denominator:
            raise ValueError(
                f"{argument} must divide two different features, got "
                f"{_validation.shown_argument(tuple(pair))}"
            )
        top, bottom = known[numerator], known[denominator]
        if top.stem == bottom.stem:
            name = f"{numerator}_over_{bottom.window.label}"
        else:
            name = f"{numerator}_over_{denominator}"
        checked.append(_Ratio(name, numerator, denominator))
    return checked


def _check_unique(features: list[_Feature], ratios: list[_Ratio], where: str, hint: str) -> None:
    names = [feature.name for feature in features] + [ratio.name for ratio in ratios]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"{where} would have two columns named {_validation.shown(name)}; {hint}"
            )
        seen.add(name)


def _check_keys(data: abc.Mapping[str, typing.Any], expected: set[str], where: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected, key=str)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"lacks {_validation.shown_list(missing)}")
        if unknown:
            problems.append(f"has unknown keys {_validation.shown_list(unknown)}")
        raise ValueError(f"{where} " + " and ".join(problems))


def _unique_keys(pairs: list[tuple[str, typing.Any]]) -> dict[str, typing.Any]:
    """Return the pairs of a JSON object as a dict, rejecting a repeated key."""
    result: dict[str, typing.Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"JSON objects must not repeat a key, got {_validation.shown(key)} twice"
            )
        result[key] = value
    return result


def _tables(events: object, names: list[str]) -> dict[str, tuple[pd.DataFrame, str]]:
    """Map the table name of each group to its event table and its name in messages."""
    if isinstance(events, pd.DataFrame):
        if len(names) > 1:
            raise ValueError(
                "events must be a dict from table names to DataFrames, because the groups read "
                f"the tables {_validation.shown_list(names)}"
            )
        return {names[0]: (events, "events")}
    if not isinstance(events, abc.Mapping):
        raise ValueError(
            f"events must be a DataFrame or a dict of DataFrames, got {type(events).__name__}"
        )
    missing = [name for name in names if name not in events]
    if missing:
        raise ValueError(
            f"events must have a table for every group, but lacks {_validation.shown(missing[0])}"
        )
    unknown = [name for name in events if name not in names]
    if unknown:
        raise ValueError(
            f"events has tables that no group reads: {_validation.shown_list(unknown)}"
        )
    return {name: (events[name], f"events[{name!r}]") for name in names}


def _column(frame: pd.DataFrame, name: str, argument: str, where: str) -> pd.Series:
    """Return the column ``name`` of ``frame``, which must be its only column of that name."""
    count = int(np.sum(frame.columns == name))
    if not count:
        raise ValueError(f"{argument} must name a column of {where}, got {_validation.shown(name)}")
    if count > 1:
        raise ValueError(f"{where} has more than one column named {_validation.shown(name)}")
    return frame[name]


def _check_frame(frame: object, where: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{where} must be a DataFrame, got {type(frame).__name__}")


def _check_complete(values: pd.Series, argument: str, where: str) -> None:
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"{argument} column {_validation.shown(values.name)} of {where} must not contain "
            f"missing values, got {missing}"
        )


def _times(
    values: pd.Series, argument: str, where: str
) -> tuple[npt.NDArray[np.int64], pd.Series, typing.Any]:
    """Return the nanoseconds since 1970 in UTC, the local wall times and the time zone."""
    dtype = values.dtype
    if not pd.api.types.is_datetime64_any_dtype(dtype):
        raise ValueError(
            f"{argument} column {_validation.shown(values.name)} of {where} must hold datetimes, "
            f"got dtype {dtype}; convert it with pandas.to_datetime"
        )
    _check_complete(values, argument, where)
    zone = getattr(dtype, "tz", None)
    if zone is None:
        local = values.astype("datetime64[ns]")
        instants = local
    else:
        local = values.dt.tz_localize(None).astype("datetime64[ns]")
        instants = values.dt.tz_convert(None).astype("datetime64[ns]")
    return _nanoseconds(instants), local, zone


def _nanoseconds(values: pd.Series) -> npt.NDArray[np.int64]:
    """Return the integer nanoseconds of tz-naive datetime64[ns] values."""
    return np.asarray(values.to_numpy(dtype="datetime64[ns]")).view(np.int64)


def _zone_name(zone: object) -> str:
    return "tz-naive" if zone is None else repr(str(zone))


def _observed(observations: object, id: str, observation_time: str) -> _Observed:
    _check_frame(observations, "observations")
    frame = typing.cast(pd.DataFrame, observations)
    _check_name(id, "id")
    _check_name(observation_time, "observation_time")
    ids = _column(frame, id, "id", "observations")
    _check_complete(ids, "id", "observations")
    times = _column(frame, observation_time, "observation_time", "observations")
    instants, local, zone = _times(times, "observation_time", "observations")
    return _Observed(ids, instants, local, zone, frame.index, {})


def _bound_instants(observed: _Observed, bound: tuple[int, str]) -> npt.NDArray[np.int64]:
    """Return the instants of a bound before every observation, on the local calendar."""
    if bound not in observed.bounds:
        count, unit = bound
        offset = pd.DateOffset(months=count) if unit == "M" else pd.Timedelta(days=count)
        local = observed.local - offset
        if observed.zone is not None:
            local = local.dt.tz_localize(
                observed.zone,
                ambiguous=np.ones(len(local), dtype=bool),
                nonexistent="shift_forward",
            ).dt.tz_convert(None)
        observed.bounds[bound] = _nanoseconds(local)
    return observed.bounds[bound]


def _group_values(
    group: _Group, events: object, where: str, id: str, observed: _Observed
) -> dict[str, npt.NDArray[typing.Any]]:
    """Return the columns of one group for every observation, by name."""
    _check_frame(events, where)
    frame = typing.cast(pd.DataFrame, events)
    ids = _column(frame, id, "id", where)
    _check_complete(ids, "id", where)
    instants, _, zone = _times(_column(frame, group.time, "time", where), "time", where)
    if _zone_name(zone) != _zone_name(observed.zone):
        raise ValueError(
            f"the times of {where} and of observations must both be tz-naive or share one time "
            f"zone, got {_zone_name(zone)} and {_zone_name(observed.zone)}"
        )
    values = {
        column: _column(frame, column, "aggregations", where) for column in group.aggregations
    }

    codes, uniques = pd.factorize(ids)
    targets = pd.Index(uniques).get_indexer(observed.ids).astype(np.int64)
    order = np.lexsort((instants, codes))
    codes = codes[order].astype(np.int64)
    times = instants[order]
    distinct, rank = np.unique(times, return_inverse=True)
    width = len(distinct) + 1
    keys = codes * width + rank.reshape(-1)
    side: typing.Literal["left", "right"] = "left" if group.closed == "left" else "right"

    def position(bound: tuple[int, str] | None) -> npt.NDArray[np.intp]:
        instant = observed.instants if bound is None else _bound_instants(observed, bound)
        found = np.searchsorted(keys, targets * width + np.searchsorted(distinct, instant, side))
        return np.where(targets >= 0, found, 0)

    columns = {
        column: _ColumnValues(values[column], order, column, where, names)
        for column, names in group.aggregations.items()
    }
    result: dict[str, npt.NDArray[typing.Any]] = {}
    for window in group.windows:
        start, end = position(window.start), position(window.end)
        label = window.label
        result[f"{group.prefix}events_{label}"] = (end - start).astype(np.int64)
        reduced = []
        for column, names in group.aggregations.items():
            for name in names:
                key = f"{group.prefix}{column}_{name}_{label}"
                if name in _REDUCED:
                    reduced.append((key, columns[column], name))
                else:
                    result[key] = columns[column].aggregate(
                        name, start, end, times=times, instants=observed.instants
                    )
        for key, _, name in reduced:
            result[key] = (
                np.zeros(len(start), dtype=np.int64)
                if name == "nunique"
                else np.full(len(start), np.nan)
            )
        # The rows of the windows are expanded a bounded number at a time, so that memory stays
        # bounded however many events the windows hold together.
        for chunk in _chunks(end - start):
            expansion = _expansion(start[chunk], end[chunk])
            for key, column_values, name in reduced:
                result[key][chunk] = column_values.reduce(name, start[chunk], end[chunk], expansion)
    return {feature.name: result[feature.name] for feature in _features(group)}


# The largest number of event rows of windows that are expanded at once.
_CHUNK = 1 << 20


def _chunks(lengths: npt.NDArray[np.intp]) -> abc.Iterator[slice]:
    """Yield slices of consecutive windows that hold at most `_CHUNK` rows together.

    A window that is longer on its own forms a slice of its own.
    """
    ends = np.cumsum(lengths)
    first = 0
    while first < len(lengths):
        before = ends[first - 1] if first else 0
        last = max(int(np.searchsorted(ends, before + _CHUNK, side="right")), first + 1)
        yield slice(first, last)
        first = last


class _Expansion(typing.NamedTuple):
    """The event rows of some windows, window after window.

    ``heads`` are the positions in ``rows`` where the non-empty windows
    begin, ``filled`` marks those windows and ``lengths`` are their lengths.
    """

    rows: npt.NDArray[np.intp]
    heads: npt.NDArray[np.intp]
    filled: npt.NDArray[np.bool_]
    lengths: npt.NDArray[np.intp]


def _expansion(start: npt.NDArray[np.intp], end: npt.NDArray[np.intp]) -> _Expansion:
    """Return the event rows of the windows [start, end) of the sorted rows."""
    lengths = end - start
    offsets = np.cumsum(lengths) - lengths
    rows = np.arange(int(lengths.sum())) - np.repeat(offsets - start, lengths)
    filled = lengths > 0
    return _Expansion(rows, offsets[filled], filled, lengths[filled])


class _ColumnValues:
    """One column of the events in the sorted order, read once for all its windows."""

    def __init__(
        self,
        values: pd.Series,
        order: npt.NDArray[np.intp],
        column: str,
        where: str,
        names: tuple[str, ...],
    ) -> None:
        present = values.notna().to_numpy(dtype=bool)[order]
        self.counts = np.concatenate(([0], np.cumsum(present, dtype=np.int64)))
        # Rows with a value, for the first and the last value of a window.
        self.kept = np.flatnonzero(present)
        self.numbers = np.empty(0)
        self.blocks: list[npt.NDArray[np.float64]] = []
        self.codes = np.empty(0, dtype=np.intp)
        self.previous = np.empty(0, dtype=np.intp)
        if any(name in _NUMERIC for name in names):
            self.numbers = _numbers(values, column, where)[order]
        if "sum" in names or "mean" in names:
            self.blocks = _blocks(np.where(present, self.numbers, 0.0))
        if "nunique" in names:
            codes = pd.factorize(values)[0][order]
            self.codes = codes
            self.previous = _previous(codes)

    def aggregate(
        self,
        name: str,
        start: npt.NDArray[np.intp],
        end: npt.NDArray[np.intp],
        *,
        times: npt.NDArray[np.int64],
        instants: npt.NDArray[np.int64],
    ) -> npt.NDArray[typing.Any]:
        """Return an aggregation other than those of `_REDUCED` over the windows [start, end)."""
        before, until = self.counts[start], self.counts[end]
        count = until - before
        if name == "count":
            return count
        if name in ("sum", "mean"):
            sums = _block_sums(self.blocks, start, end)
            if name == "sum":
                return sums
            return np.divide(sums, count, out=np.full(len(sums), np.nan), where=count > 0)
        result = np.full(len(start), np.nan)
        has = count > 0
        if name == "first":
            rows = self.kept[before[has]]
            result[has] = self.numbers[rows]
        else:
            rows = self.kept[until[has] - 1]
            if name == "last":
                result[has] = self.numbers[rows]
            else:
                result[has] = (instants[has] - times[rows]) / _NANOSECONDS_PER_DAY
        return result

    def reduce(
        self,
        name: str,
        start: npt.NDArray[np.intp],
        end: npt.NDArray[np.intp],
        expansion: _Expansion,
    ) -> npt.NDArray[typing.Any]:
        """Return an aggregation of `_REDUCED` over the windows [start, end) of an expansion."""
        rows, heads, filled, lengths = expansion
        if name == "nunique":
            # A value is new in a window when its previous occurrence lies before the window.
            new = (self.codes[rows] >= 0) & (
                self.previous[rows] < np.repeat(start[filled], lengths)
            )
            result = np.zeros(len(start), dtype=np.int64)
            if len(heads):
                result[filled] = np.add.reduceat(new.astype(np.int64), heads)
            return result
        result = np.full(len(start), np.nan)
        if not len(heads):
            return result
        values = self.numbers[rows]
        if name == "min":
            result[filled] = np.fmin.reduceat(values, heads)
        elif name == "max":
            result[filled] = np.fmax.reduceat(values, heads)
        else:
            missing = np.isnan(values)
            n = (self.counts[end] - self.counts[start])[filled]
            sums = np.add.reduceat(np.where(missing, 0.0, values), heads)
            means = np.divide(sums, n, out=np.zeros(len(n)), where=n > 0)
            deviations = np.where(missing, 0.0, values - np.repeat(means, lengths))
            squares = np.add.reduceat(deviations * deviations, heads)
            variance = np.divide(squares, n - 1, out=np.full(len(n), np.nan), where=n > 1)
            result[filled] = np.sqrt(variance)
        return result


def _blocks(values: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
    """Return the sums of aligned blocks of 1, 2, 4, ... values, one array per block length.

    Each block sums two blocks of half its length, so that the sum of a
    window, assembled from the blocks that lie inside it, is as precise as
    a pairwise sum of the window alone.
    """
    blocks = [values]
    while len(blocks[-1]) > 1:
        level = blocks[-1]
        if len(level) % 2:
            level = np.append(level, 0.0)
        blocks.append(level[0::2] + level[1::2])
    return blocks


def _block_sums(
    blocks: list[npt.NDArray[np.float64]],
    start: npt.NDArray[np.intp],
    end: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Return the sums of the windows [start, end), each from the aligned blocks inside it.

    At every block length, a window takes the block at its left end when
    that end is not aligned to the next length, and likewise at its right
    end, as a segment tree is read from the leaves up. Values outside a
    window never enter its sum, so a large value before a window cannot
    swamp the small ones inside it.
    """
    sums = np.zeros(len(start))
    low, high = start.astype(np.intp), end.astype(np.intp)
    for level in blocks:
        left = (low & 1).astype(bool) & (low < high)
        sums[left] += level[low[left]]
        low = low + left
        right = (high & 1).astype(bool) & (low < high)
        high = high - right
        sums[right] += level[high[right]]
        low, high = low >> 1, high >> 1
    return sums


def _numbers(values: pd.Series, column: str, where: str) -> npt.NDArray[np.float64]:
    """Return a column as floats, missing values as NaN; booleans count as 0 and 1."""
    try:
        numbers = _validation.as_real(
            values, f"column {_validation.shown(column)} of {where}", booleans=True
        )
    except ValueError as error:
        raise ValueError(
            f"{error}; only count, nunique and days_since_last take values other than numbers"
        ) from None
    infinite = np.isinf(numbers)
    if infinite.any():
        raise ValueError(
            f"column {_validation.shown(column)} of {where} must hold finite numbers or missing "
            f"values, got {_validation.shown(numbers[infinite][0])}"
        )
    return numbers


def _previous(codes: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """Return the row of the previous occurrence of each code, or -1 for its first one."""
    grouped = np.argsort(codes, kind="stable")
    previous = np.full(len(codes), -1, dtype=np.intp)
    same = codes[grouped[1:]] == codes[grouped[:-1]]
    previous[grouped[1:][same]] = grouped[:-1][same]
    return previous


def _frame(
    columns: dict[str, npt.NDArray[typing.Any]], ratios: list[_Ratio], index: pd.Index
) -> pd.DataFrame:
    """Return the features and their ratios as a DataFrame on the index of the observations."""
    data = dict(columns)
    for ratio in ratios:
        top = columns[ratio.numerator].astype(float)
        bottom = columns[ratio.denominator].astype(float)
        data[ratio.name] = np.divide(
            top, bottom, out=np.full(len(top), np.nan), where=(bottom != 0) & ~np.isnan(bottom)
        )
    return pd.DataFrame(data, index=index)
