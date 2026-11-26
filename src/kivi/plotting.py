"""Charts of kivi's result tables for risk reports.

Each function draws one result of another kivi module on a matplotlib
Axes: ``woe_chart`` a ``kivi.woe.woe_table``, ``correlation_chart`` a
correlation matrix such as ``X.corr()``, ``roc_chart`` and
``precision_recall_chart`` a ``kivi.metrics.roc_table``, ``gains_chart`` a
``kivi.metrics.gains_table``, ``distribution_chart`` the event and
non-event counts of a gains or WOE table,
``calibration_chart`` a ``kivi.calibration.calibration_table``,
``cutoff_chart`` a ``kivi.strategy.cutoff_table``, ``psi_chart`` a
``kivi.stability.psi_table``, ``monitoring_chart`` a column of a
``kivi.monitoring.monitoring_table``, ``vintage_chart`` a
``kivi.vintage.vintage_table``, ``survival_chart`` a
``kivi.lifetime.survival_table``, ``transition_chart`` a
``kivi.vintage.transition_matrix``, ``rule_set_chart`` a
``kivi.rules.rule_set_table``, ``profit_chart`` a
``kivi.strategy.cutoff_table`` with revenue and loss, and ``lgd_chart`` a
``kivi.lgd.lgd_table``.

A function draws on ``ax`` when one is given, such as a panel of a larger
figure, and creates a figure otherwise. It returns the Axes it drew on, so
that titles, limits and styles can be changed afterwards. It never calls
``matplotlib.pyplot.show``, never changes ``matplotlib.rcParams`` and never
modifies its input. Text inside a chart is limited to column names and data
values.

matplotlib is an optional dependency, imported when a chart is drawn.
Install it with ``pip install "kivi[plot]"``.
"""

import math
import typing
from collections import abc

import numpy as np
import pandas as pd

from . import _validation

if typing.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.axis
    import matplotlib.typing

__all__ = [
    "woe_chart",
    "correlation_chart",
    "roc_chart",
    "precision_recall_chart",
    "gains_chart",
    "distribution_chart",
    "calibration_chart",
    "cutoff_chart",
    "psi_chart",
    "monitoring_chart",
    "vintage_chart",
    "survival_chart",
    "transition_chart",
    "rule_set_chart",
    "profit_chart",
    "lgd_chart",
]

_BLUE = "#2f5bd3"
_LIGHT_BLUE = "#9fb4ee"
_AMBER = "#d97706"
_RED = "#dc2626"
_GRAY = "#9ca3af"
_TEXT = "#1f2937"
_LEVEL_COLORS = (_AMBER, _RED, _GRAY)
# vintage_chart names at most this many cohorts in its legend.
_LEGEND_COHORTS = 8
_INSTALL = 'kivi.plotting needs matplotlib; install it with pip install "kivi[plot]"'


def woe_chart(
    table: pd.DataFrame,
    *,
    line: typing.Literal["event_rate", "woe"] = "event_rate",
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple["matplotlib.axes.Axes", "matplotlib.axes.Axes"]:
    """Draw the population share of each group with its event rate or WOE.

    The bars show ``total_share``, the share of the sample in each group,
    on the left y-axis, and a line shows ``event_rate`` or ``woe`` on a
    second y-axis on the right. This is the one chart of the module with
    two y-axes, and so the one that returns two Axes. The groups keep the
    order of the table, and the legend gives the IV of the whole table.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.woe.woe_table`, or one of the tables of a
        `kivi.spec.BinningSpec`. It needs the columns ``total_share``,
        ``iv`` and the column named by `line`.
    line : {'event_rate', 'woe'}, default 'event_rate'
        Column drawn as the line.
    ax : matplotlib.axes.Axes, optional
        Axes of the bars. A new figure is created when it is not given.

    Returns
    -------
    bars : matplotlib.axes.Axes
        Axes of the bars, `ax` when it is given.
    line : matplotlib.axes.Axes
        Axes of the line, a twin of `bars` that shares its x-axis.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, `line` is neither
        ``'event_rate'`` nor ``'woe'``, or `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.woe.woe_table : The table that this chart draws.
    psi_chart : Compare the shares of two samples in the same groups.

    Notes
    -----
    The tick labels are the group labels of the index, with ``NaN`` for the
    group of missing values. Sort the table first, for example with
    ``table.sort_index()`` or ``table.sort_values("woe")``, to draw the
    groups in another order.

    Each y-axis is labeled in the color of what it shows, and the dotted
    line of zero WOE belongs to the right axis. Both axes leave room at the
    top for the legend, so that it covers no bar and no point.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.plotting import woe_chart
    >>> from kivi.woe import woe_table
    >>> groups = pd.Series(["a", "b", "b", "c", "c", "c"])
    >>> target = pd.Series([1, 0, 1, 0, 0, 1])
    >>> bars, line = woe_chart(woe_table(groups, target))
    >>> [label.get_text() for label in bars.get_xticklabels()]
    ['a', 'b', 'c']
    >>> line.get_ylabel()
    'event_rate'
    """
    if not isinstance(line, str) or line not in ("event_rate", "woe"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            f"line must be 'event_rate' or 'woe', got {_validation.shown_argument(line)}"
        )
    table = _table(table, ["total_share", line, "iv"], "kivi.woe.woe_table")
    ax = _axes(ax, (7, 4))
    x = np.arange(len(table))
    shares = table["total_share"].to_numpy(dtype=float)
    values = table[line].to_numpy(dtype=float)
    bars = ax.bar(x, shares, width=0.6, color=_LIGHT_BLUE, label="total_share")
    columns = _legend_columns(ax, ["total_share", line])
    room = _legend_room(ax, 1 + 2 // columns)
    ax.set_ylim(0, _top(shares, 0.0, room))
    ax.set_ylabel("total_share", color=_BLUE)
    _percent(ax.yaxis)
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    twin = ax.twinx()
    if line == "woe":
        twin.axhline(0, color=_AMBER, linestyle=":", linewidth=1)
        low, high = _span(values)
        bottom = low - 0.08 * (high - low)
    else:
        _percent(twin.yaxis)
        bottom = 0.0
    (curve,) = twin.plot(x, values, color=_AMBER, marker="o", linewidth=2, label=line)
    twin.set_ylim(bottom, _top(np.r_[values, 0.0], bottom, room))
    twin.set_ylabel(line, color=_AMBER)
    twin.grid(False)
    twin.legend(
        [bars, curve],
        ["total_share", line],
        loc="upper left",
        ncols=columns,
        title=f"iv = {table['iv'].sum():.3f}",
    )
    return ax, twin


def correlation_chart(
    matrix: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw a correlation matrix as a heatmap with the value in each cell.

    Positive correlations are blue and negative ones amber, darker as they
    approach 1 or -1, so that pairs of variables that carry the same
    information stand out before a model is fitted.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Square matrix of correlations with the same labels, in the same
        order, in its index and columns, such as ``X.corr()`` or
        ``X.corr(method="spearman")`` of the WOE-encoded candidates ``X``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure, sized to the number of variables, is
        created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `matrix` is not a DataFrame, has no rows, does not have the same
        labels in its index and columns, has a column that is not numeric
        or a value outside [-1, 1], such as a covariance, or `ax` is not a
        matplotlib Axes.

    See Also
    --------
    kivi.selection.correlation_filter : Drop one variable of each highly
        correlated pair.
    kivi.selection.vif : Variance inflation factor of each variable.

    Notes
    -----
    The colors run from -1 to 1 whatever the values, and a missing value,
    such as the correlation of a constant column, leaves its cell empty.
    The heatmap is ``ax.images[0]``, so ``ax.figure.colorbar(ax.images[0],
    ax=ax)`` adds a color bar.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.plotting import correlation_chart
    >>> X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 4, 3], "c": [4, 3, 2, 1]})
    >>> ax = correlation_chart(X.corr())
    >>> [label.get_text() for label in ax.get_yticklabels()]
    ['a', 'b', 'c']
    >>> ax.texts[1].get_text()
    '0.60'
    """
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError(
            f"matrix must be a DataFrame such as X.corr(), got {type(matrix).__name__}"
        )
    if not len(matrix):
        raise ValueError("matrix must have at least one row")
    if not matrix.index.equals(matrix.columns):
        raise ValueError(
            "matrix must have the same labels in its index and columns, such as X.corr(), got "
            f"{_validation.shown_argument(matrix.index)} and "
            f"{_validation.shown_argument(matrix.columns)}"
        )
    for column in range(matrix.shape[1]):
        _validation.check_real_dtype(
            matrix.iloc[:, column].dtype, f"column {_validation.shown(matrix.columns[column])}"
        )
    values = matrix.to_numpy(dtype=float)
    # Rounding can put a correlation a few units in the last place beyond 1.
    outside = np.abs(values) > 1 + 1e-9
    if outside.any():
        raise ValueError(
            "matrix must hold correlations between -1 and 1, got "
            f"{_validation.shown(values[outside][0])}"
        )
    size = len(matrix)
    ax = _axes(ax, (max(4.0, 0.6 * size + 1.6), max(3.2, 0.6 * size + 1.0)), "compressed")
    import matplotlib.colors

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "kivi_correlation", [_AMBER, "white", _BLUE]
    )
    ax.imshow(values, cmap=colors, vmin=-1, vmax=1)
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize="small",
                color="white" if abs(value) > 0.6 else _TEXT,
            )
    labels = _labels(matrix.index)
    _ticks(ax, np.arange(size), labels)
    ax.set_yticks(np.arange(size), labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def roc_chart(
    table: pd.DataFrame,
    *,
    ks: bool = True,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the ROC curve, with the largest gap between the two rates marked.

    The curve plots ``true_positive_rate`` against ``false_positive_rate``
    from the strictest threshold to the loosest, next to the diagonal of a
    score that ranks at random. With `ks`, a vertical segment marks the KS
    statistic: the largest distance between the curve and the diagonal.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.metrics.roc_table`. It needs the columns
        ``false_positive_rate`` and ``true_positive_rate``.
    ks : bool, default True
        Whether to mark the KS statistic.
    label : str, optional
        Legend entry of the curve. A legend is drawn only when a curve has
        a label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, `ks` is not True or False,
        or `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.metrics.roc_table : The table that this chart draws.
    kivi.metrics.roc_auc : Area under the curve.
    gains_chart : The same separation against the share of the sample.

    Notes
    -----
    The curve takes the next color of the Axes' color cycle, so that ROC
    curves of several scores drawn on the same `ax` are told apart by
    color and by `label`. The KS segment and its value share the color of
    their curve.

    Examples
    --------
    >>> from kivi.metrics import roc_table
    >>> from kivi.plotting import roc_chart
    >>> table = roc_table([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    >>> ax = roc_chart(table)
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('false_positive_rate', 'true_positive_rate')
    >>> ax.texts[0].get_text()
    'ks = 0.500'
    """
    if not isinstance(ks, (bool, np.bool_)):
        raise ValueError(f"ks must be True or False, got {_validation.shown_argument(ks)}")
    table = _table(table, ["false_positive_rate", "true_positive_rate"], "kivi.metrics.roc_table")
    ax = _axes(ax, (5, 4.5))
    false_positive = table["false_positive_rate"].to_numpy(dtype=float)
    true_positive = table["true_positive_rate"].to_numpy(dtype=float)
    ax.plot([0, 1], [0, 1], color=_GRAY, linestyle="--", linewidth=1)
    (curve,) = ax.plot(false_positive, true_positive, linewidth=2, label=label)
    if ks:
        gap = true_positive - false_positive
        largest = int(np.argmax(np.abs(gap)))
        _gap(
            ax,
            false_positive[largest],
            false_positive[largest],
            true_positive[largest],
            curve.get_color(),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("false_positive_rate")
    ax.set_ylabel("true_positive_rate")
    if label is not None:
        ax.legend(loc="lower right")
    return ax


def precision_recall_chart(
    table: pd.DataFrame,
    *,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    r"""Draw the precision against the recall of every threshold of a score.

    Flagging the observations from the highest score down, the curve plots
    ``precision``, the event rate of the flagged observations, against
    ``true_positive_rate``, the share of all events flagged, also called
    the recall, as steps whose area is the average precision. A dashed line
    marks the event rate of the sample, the precision of a score that ranks
    at random.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.metrics.roc_table`. It needs the columns
        ``true_positive_rate`` and ``precision``.
    label : str, optional
        Legend entry of the curve. A legend is drawn only when a curve has
        a label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, no row has a precision, or
        `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.metrics.roc_table : The table that this chart draws.
    roc_chart : The true positive rate against the false positive rate.
    kivi.strategy.confusion_table : Precision and recall at chosen
        cut-offs.

    Notes
    -----
    The first row of a roc table, at the threshold ``inf``, flags nobody
    and has no precision. From a recall of 0, the curve steps through the
    other rows: from the recall of one row to that of the next, it stays at
    the precision of the next row. Straight lines between the rows would
    overstate the precision between them [1]_. With :math:`R_n` and
    :math:`P_n` the recall and the precision of the n-th row, and
    :math:`R_0 = 0`, the area under the steps is the average precision

    .. math::

       \mathrm{AP} = \sum_n (R_n - R_{n-1})\, P_n.

    The last row flags every observation, so its precision is the event
    rate: the curve ends on the dashed line, which is labeled with that
    rate at its left end.

    The curve takes the next color of the Axes' color cycle, so that the
    curves of several scores drawn on the same `ax` are told apart by
    color and by `label`. The dashed line is gray, as the diagonal of
    `roc_chart`, and is drawn once for each event rate shown, so that the
    scores of one sample share it and two samples get one line each.

    References
    ----------
    .. [1] J. Davis and M. Goadrich, "The Relationship Between
       Precision-Recall and ROC Curves", Proceedings of the 23rd
       International Conference on Machine Learning, 233-240, 2006.

    Examples
    --------
    >>> from kivi.metrics import roc_table
    >>> from kivi.plotting import precision_recall_chart
    >>> table = roc_table([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    >>> ax = precision_recall_chart(table)
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('true_positive_rate', 'precision')
    >>> ax.texts[0].get_text()
    'event_rate = 0.5'
    """
    table = _table(table, ["true_positive_rate", "precision"], "kivi.metrics.roc_table")
    precision = table["precision"].to_numpy(dtype=float)
    flagged = ~np.isnan(precision)
    if not flagged.any():
        raise ValueError(
            "table must have a row with a precision, below the first row of a roc table"
        )
    recall = table["true_positive_rate"].to_numpy(dtype=float)[flagged]
    precision = precision[flagged]
    ax = _axes(ax, (5, 4.5))
    rate = float(precision[-1])
    text = f"event_rate = {rate:.3g}"
    if all(drawn.get_text() != text for drawn in ax.texts):
        ax.axhline(rate, color=_GRAY, linestyle="--", linewidth=1)
        ax.annotate(
            text,
            (0, rate),
            xycoords=ax.get_yaxis_transform(),
            xytext=(4, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize="small",
            color=_GRAY,
        )
    # Steps from a recall of 0, each at the precision of the row it ends on.
    ax.plot(
        np.r_[0.0, recall],
        np.r_[precision[0], precision],
        drawstyle="steps-pre",
        linewidth=2,
        label=label,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("true_positive_rate")
    ax.set_ylabel("precision")
    if label is not None:
        ax.legend(loc="best")
    return ax


def gains_chart(
    table: pd.DataFrame,
    *,
    ks: bool = True,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the cumulative shares of events and non-events by score band.

    Reviewing the bands from the highest scores down, the curves show the
    share of all events and of all non-events captured against the share
    of the sample reviewed, ``cum_total_share``. With `ks`, a vertical
    segment marks the band with the largest ``ks``, the gap between the
    two curves.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.metrics.gains_table`, with the highest band
        first. It needs the columns ``cum_total_share``,
        ``cum_event_share``, ``cum_non_event_share`` and ``ks``.
    ks : bool, default True
        Whether to mark the largest ``ks``.
    label : str, optional
        Name of the score, added to the legend entries of both curves, as
        in ``cum_event_share (label)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, `ks` is not True or False,
        or `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.metrics.gains_table : The table that this chart draws.
    roc_chart : The same separation over all thresholds.

    Notes
    -----
    Both curves start at the origin, before the first band is reviewed,
    and each band adds one point. The diagonal is the ``cum_event_share``
    of a score that ranks at random.

    The ``cum_event_share`` curve is solid and the ``cum_non_event_share``
    curve dashed, both in the next color of the Axes' color cycle, so that
    the gains of several scores drawn on the same `ax` are told apart by
    color and by `label`. The KS segment and its value share that color.
    The legend goes where it covers the fewest points; with several scores
    it has two entries for each, and ``ax.legend(loc=...)`` moves it.

    Examples
    --------
    >>> from kivi.metrics import gains_table
    >>> from kivi.plotting import gains_chart
    >>> score = [1, 2, 3, 3, 4, 5, 1, 2]
    >>> target = [0, 0, 1, 0, 1, 1, 0, 0]
    >>> ax = gains_chart(gains_table(score, target, bins=3))
    >>> ax.get_xlabel()
    'cum_total_share'
    >>> ax.texts[0].get_text()
    'ks = 0.800'
    """
    if not isinstance(ks, (bool, np.bool_)):
        raise ValueError(f"ks must be True or False, got {_validation.shown_argument(ks)}")
    table = _table(
        table,
        ["cum_total_share", "cum_event_share", "cum_non_event_share", "ks"],
        "kivi.metrics.gains_table",
    )
    ax = _axes(ax, (5.6, 4.5))
    share = np.r_[0.0, table["cum_total_share"].to_numpy(dtype=float)]
    event = np.r_[0.0, table["cum_event_share"].to_numpy(dtype=float)]
    non_event = np.r_[0.0, table["cum_non_event_share"].to_numpy(dtype=float)]
    suffix = "" if label is None else f" ({label})"
    ax.plot([0, 1], [0, 1], color=_GRAY, linestyle="--", linewidth=1)
    (curve,) = ax.plot(
        share, event, marker="o", markersize=4, linewidth=2, label=f"cum_event_share{suffix}"
    )
    color = curve.get_color()
    ax.plot(
        share,
        non_event,
        color=color,
        linestyle="--",
        marker="o",
        markersize=4,
        linewidth=2,
        label=f"cum_non_event_share{suffix}",
    )
    if ks:
        largest = int(np.argmax(table["ks"].to_numpy(dtype=float))) + 1
        _gap(ax, share[largest], non_event[largest], event[largest], color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("cum_total_share")
    ax.set_ylabel("cum_event_share, cum_non_event_share")
    ax.legend(loc="best")
    return ax


def distribution_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw the share of all non-events and of all events in each group side by side.

    Each row of the table, such as a score band of a gains table or a bin
    of a WOE table, gets a pair of bars: its share of all the non-events of
    the table and its share of all the events, from the raw ``non_event``
    and ``event`` counts. The further apart the two distributions lie, the
    better the score or the variable tells the classes apart.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.metrics.gains_table` or `kivi.woe.woe_table`,
        or one of the tables of a `kivi.spec.BinningSpec`. It needs the
        columns ``non_event`` and ``event``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, a count is negative or not
        finite, the table counts no event or no non-event, or `ax` is not a
        matplotlib Axes.

    See Also
    --------
    kivi.metrics.gains_table : Score bands, with the highest first.
    kivi.woe.woe_table : Bins or groups of a variable.
    gains_chart : The cumulative shares of the same bands.
    psi_chart : The shares of two samples in the same groups.

    Notes
    -----
    The bars keep the order of the table, the highest band first for a
    gains table, and its empty rows, so that the charts of two samples
    binned on the same edges line up. The shares are not smoothed, so for
    a group without events or without non-events they differ from the
    ``event_share`` and ``non_event_share`` columns of a WOE table, which
    give an empty cell the smoothing. The tick labels are the group labels
    of the index, with ``NaN`` for the group of missing values.

    The non-events are blue and the events red. The y-axis leaves room at
    the top for the legend, so that it covers no bar.

    Examples
    --------
    >>> from kivi.metrics import gains_table
    >>> from kivi.plotting import distribution_chart
    >>> score = [1, 2, 3, 3, 4, 5, 1, 2]
    >>> target = [0, 0, 1, 0, 1, 1, 0, 0]
    >>> ax = distribution_chart(gains_table(score, target, bins=3))
    >>> [label.get_text() for label in ax.get_xticklabels()]
    ['(3.0, inf]', '(2.0, 3.0]', '(-inf, 2.0]']
    >>> [f"{bar.get_height():.2f}" for bar in ax.patches]
    ['0.00', '0.20', '0.80', '0.67', '0.33', '0.00']
    """
    table = _table(table, ["non_event", "event"], "kivi.metrics.gains_table or kivi.woe.woe_table")
    counts = _validation.as_count_columns(
        table["non_event"], table["event"], ("non_event of table", "event of table")
    )
    shares = counts / counts.sum(axis=0)
    ax = _axes(ax, (7, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, shares[:, 0], width=0.4, color=_BLUE, label="non_event_share")
    ax.bar(x + 0.2, shares[:, 1], width=0.4, color=_RED, label="event_share")
    columns = _legend_columns(ax, ["non_event_share", "event_share"])
    # The room of a row more than the legend has: slanted labels of many bands make the
    # layout shrink the Axes more, and the highest band of a gains table, on the left,
    # often holds the tallest event bar.
    room = _legend_room(ax, 1 + 2 // columns)
    ax.set_ylim(0, _top(shares.ravel(), 0.0, room))
    _percent(ax.yaxis)
    ax.set_ylabel("non_event_share, event_share")
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    ax.legend(loc="upper left", ncols=columns)
    return ax


def calibration_chart(
    table: pd.DataFrame,
    *,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the observed event rate of each group against its mean predicted PD.

    Each group of the table, such as a grade or a PD band, is one point
    with ``mean_probability`` on the x-axis and ``event_rate`` on the
    y-axis. Points on the diagonal are calibrated, points above it have
    more events than predicted and points below it fewer.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.calibration.calibration_table`. It needs the
        columns ``mean_probability`` and ``event_rate``.
    label : str, optional
        Legend entry of the points. A legend is drawn only when a line has
        a label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.calibration.calibration_table : The table that this chart draws.
    kivi.calibration.binomial_test : Test the default rate of each grade.

    Notes
    -----
    Both axes run from 0 to just above the largest rate drawn on `ax`, so
    that the diagonal stays at 45 degrees. The points are joined in the
    order of the table and take the next color of the Axes' color cycle,
    so that the PDs of several models, or the same PDs before and after a
    correction, can be compared on one `ax`.

    Examples
    --------
    >>> from kivi.calibration import calibration_table
    >>> from kivi.plotting import calibration_chart
    >>> grade = ["A", "A", "B", "B", "B", "C"]
    >>> target = [0, 0, 0, 1, 0, 1]
    >>> probability = [0.05, 0.1, 0.2, 0.3, 0.25, 0.6]
    >>> ax = calibration_chart(calibration_table(grade, target, probability))
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('mean_probability', 'event_rate')
    """
    table = _table(table, ["mean_probability", "event_rate"], "kivi.calibration.calibration_table")
    ax = _axes(ax, (5, 4.5))
    ax.axline((0, 0), slope=1, color=_GRAY, linestyle="--", linewidth=1)
    ax.plot(
        table["mean_probability"].to_numpy(dtype=float),
        table["event_rate"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.5,
        label=label,
    )
    # Square limits from everything drawn on ax, so that earlier curves stay in view.
    upper = min(1.0, 1.1 * max(ax.dataLim.x1, ax.dataLim.y1))
    upper = upper if upper > 0 else 1.0
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("mean_probability")
    ax.set_ylabel("event_rate")
    if label is not None:
        ax.legend(loc="best")
    return ax


def cutoff_chart(
    table: pd.DataFrame,
    *,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the event rate of the approved book against the approval rate.

    Each cut-off of the table is one point, with ``approval_rate`` on the
    x-axis and ``approved_event_rate``, the bad rate of the approved book,
    on the y-axis. A higher cut-off approves more applicants and lets more
    events through, so the curve shows the price in risk of each extra
    point of approval rate.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.strategy.cutoff_table`. It needs the columns
        ``approval_rate`` and ``approved_event_rate``.
    label : str, optional
        Legend entry of the curve. A legend is drawn only when a curve has
        a label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.strategy.cutoff_table : The table that this chart draws.
    kivi.strategy.swap_set : Compare the decisions of two scores.

    Notes
    -----
    A cut-off below every score approves nothing and has a missing
    ``approved_event_rate``, so it is left out of the curve. The curve
    takes the next color of the Axes' color cycle, so that the curves of a
    current and a new score can be compared on one `ax`. The y-axis runs
    from 0 to just above the largest rate drawn on `ax`.

    Examples
    --------
    >>> from kivi.plotting import cutoff_chart
    >>> from kivi.strategy import cutoff_table
    >>> score = [0.1, 0.2, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    >>> target = [0, 0, 1, 0, 0, 1, 1, 1]
    >>> ax = cutoff_chart(cutoff_table(score, target, cutoffs=[0.2, 0.5, 1.0]))
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('approval_rate', 'approved_event_rate')
    """
    table = _table(table, ["approval_rate", "approved_event_rate"], "kivi.strategy.cutoff_table")
    ax = _axes(ax, (5.6, 4.2))
    ax.plot(
        table["approval_rate"].to_numpy(dtype=float),
        table["approved_event_rate"].to_numpy(dtype=float),
        marker="o",
        markersize=4,
        linewidth=2,
        label=label,
    )
    upper = min(1.0, 1.1 * ax.dataLim.y1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, upper if upper > 0 else 1.0)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("approval_rate")
    ax.set_ylabel("approved_event_rate")
    if label is not None:
        ax.legend(loc="best")
    return ax


def psi_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw the reference and the current share of each group side by side.

    Each group of the table gets a pair of bars, ``reference_share`` and
    ``current_share``, and the legend gives the PSI of the whole table, so
    that the groups whose shares moved most can be read off directly.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.stability.psi_table`. It needs the columns
        ``reference_share``, ``current_share`` and ``psi``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.stability.psi_table : The table that this chart draws.
    monitoring_chart : The PSI of a score in each period.

    Notes
    -----
    The shares are those of the table, so with smoothing a group that is
    empty in one sample shows the small share that smoothing gave it. The
    tick labels are the group labels of the index, with ``NaN`` for the
    group of missing values.

    The y-axis leaves room at the top for the legend, so that it covers no
    bar.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.plotting import psi_chart
    >>> from kivi.stability import psi_table
    >>> reference = pd.Series(["a", "a", "b", "b", "c", "c"])
    >>> current = pd.Series(["a", "b", "b", "c", "c", "c"])
    >>> ax = psi_chart(psi_table(reference, current))
    >>> ax.get_legend().get_title().get_text()
    'psi = 0.183'
    """
    table = _table(table, ["reference_share", "current_share", "psi"], "kivi.stability.psi_table")
    ax = _axes(ax, (7, 4))
    x = np.arange(len(table))
    reference = table["reference_share"].to_numpy(dtype=float)
    current = table["current_share"].to_numpy(dtype=float)
    ax.bar(x - 0.2, reference, width=0.4, color=_LIGHT_BLUE, label="reference_share")
    ax.bar(x + 0.2, current, width=0.4, color=_BLUE, label="current_share")
    columns = _legend_columns(ax, ["reference_share", "current_share"])
    ax.set_ylim(0, _top(np.r_[reference, current], 0.0, _legend_room(ax, 1 + 2 // columns)))
    _percent(ax.yaxis)
    ax.set_ylabel("reference_share, current_share")
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    ax.legend(loc="upper left", ncols=columns, title=f"psi = {table['psi'].sum():.3f}")
    return ax


def monitoring_chart(
    table: pd.DataFrame,
    *,
    column: str = "psi",
    kind: typing.Literal["line", "bar"] = "line",
    label: str | None = None,
    levels: float | abc.Sequence[float] = (),
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw one column of a monitoring report period by period.

    The periods of the table run along the x-axis in its order, and
    `column` is drawn as a line or as bars, as `kind` says. Dashed lines can
    mark the `levels` that the column is compared with, such as the PSI
    thresholds or the Gini of the development sample.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.monitoring.monitoring_table`, or another table
        with one row per period, such as one from
        `kivi.monitoring.stability_table`. It needs the column named by
        `column`.
    column : str, default 'psi'
        Column to draw.
    kind : {'line', 'bar'}, default 'line'
        Draw the column as a line, which suits rates and measures such as
        ``event_rate`` and ``gini`` and several columns on one `ax`, or as
        bars, which suit counts such as ``total`` and a single ``psi``.
    label : str, optional
        Legend entry of the line or the bars. A legend is drawn only when
        something on the Axes has a label.
    levels : float or sequence of float, default ()
        Values of `column` to mark with dashed horizontal lines, such as
        the PSI thresholds ``[0.1, 0.25]`` or the Gini of the development
        sample. Each line is labeled with its value at its right end.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, has no rows or has no numeric column
        named `column`, `kind` is neither ``'line'`` nor ``'bar'``, a level
        is not a finite number, or `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.monitoring.monitoring_table : The table that this chart draws.
    psi_chart : The shares behind the PSI of one period.

    Notes
    -----
    The y-axis is labeled with `column`, or with ``psi`` when the columns
    of `table` are named ``variable``, as those of
    `kivi.monitoring.stability_table` are, whose columns are the PSIs of
    the variables. The x-axis is labeled with the name of the index.

    A missing value, such as the ``gini`` of a period whose outcomes are
    not known yet, leaves a gap in the line and no bar. To draw several
    columns, call the function once for each: on the panels of one figure
    for columns on different scales, or on one `ax` with a `label` each,
    as for the variables of a stability table. The line or the bars take
    the next color of the Axes' color cycle, and the legend sits to the
    right of the Axes, so that it covers no line.

    Examples
    --------
    >>> from kivi.monitoring import monitoring_table
    >>> from kivi.plotting import monitoring_chart
    >>> score = [0.1, 0.3, 0.6, 0.8, 0.2, 0.4, 0.7, 0.9]
    >>> period = ["2025-01"] * 4 + ["2025-02"] * 4
    >>> target = [0, 0, 1, 1, 0, 1, 0, 1]
    >>> report = monitoring_table(score, period, target, reference_period="2025-01", bins=2)
    >>> ax = monitoring_chart(report, column="gini", levels=0.5)
    >>> [label.get_text() for label in ax.get_xticklabels()]
    ['2025-01', '2025-02']
    >>> ax.get_ylabel()
    'gini'
    """
    if not isinstance(column, str):
        raise ValueError(f"column must be a column name, got {_validation.shown_argument(column)}")
    if not isinstance(kind, str) or kind not in ("line", "bar"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(f"kind must be 'line' or 'bar', got {_validation.shown_argument(kind)}")
    marks = _levels(levels)
    table = _table(table, [column], "kivi.monitoring.monitoring_table")
    ax = _axes(ax, (7, 3.6))
    x = np.arange(len(table))
    values = table[column].to_numpy(dtype=float)
    if kind == "bar":
        ax.bar(x, values, width=0.6, label=label)
    else:
        ax.plot(x, values, marker="o", linewidth=2, label=label)
    for number, level in enumerate(marks):
        color = _LEVEL_COLORS[min(number, len(_LEVEL_COLORS) - 1)]
        ax.axhline(level, color=color, linestyle="--", linewidth=1)
        ax.annotate(
            f"{level:g}",
            (1, level),
            xycoords=ax.get_yaxis_transform(),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize="small",
            color=color,
        )
    ax.set_ylabel("psi" if table.columns.name == "variable" else column)
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    if label is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return ax


def vintage_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw one curve per cohort of its cumulative bad rate by month on book.

    Each row of the table is a cohort and each column a month on book.
    Later cohorts are drawn in darker blue, so that a cohort whose curve
    rises faster than those before it stands out.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.vintage.vintage_table`, with one row per cohort
        and one numeric column per month on book, or another table of the
        same shape, such as cumulative PDs from `kivi.lifetime.convert_pd`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame or has no rows, its columns are not
        numbers, it holds a value that is not numeric, or `ax` is not a
        matplotlib Axes.

    See Also
    --------
    kivi.vintage.vintage_table : The table that this chart draws.
    kivi.lifetime.convert_pd : Turn the curves into marginal or conditional
        PDs.

    Notes
    -----
    The missing months of a cohort, those it has not reached or that come
    before its first observed month on book, are left out of its curve.
    The x-axis is labeled with the name of the columns.

    The legend sits to the right of the Axes, so that it covers no curve,
    and is titled with the name of the index. It names every cohort when
    there are at most eight, and otherwise eight cohorts evenly spread from
    the first to the last, since the shades of the others lie between
    theirs. In a figure of several panels, use a layout engine, such as
    ``layout="constrained"``, to make room for it.

    Examples
    --------
    >>> from kivi.plotting import vintage_chart
    >>> from kivi.vintage import vintage_table
    >>> loan = [1, 1, 1, 2, 2, 3, 3, 3]
    >>> cohort = ["2024-01", "2024-01", "2024-01", "2024-01", "2024-01",
    ...           "2024-02", "2024-02", "2024-02"]
    >>> month_on_book = [1, 2, 3, 1, 2, 1, 2, 3]
    >>> bad = [0, 1, 1, 0, 0, 0, 0, 1]
    >>> ax = vintage_chart(vintage_table(loan, cohort, month_on_book, bad))
    >>> ax.get_xlabel()
    'month_on_book'
    >>> [text.get_text() for text in ax.get_legend().get_texts()]
    ['2024-01', '2024-02']
    """
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            f"table must be a DataFrame from kivi.vintage.vintage_table, got {type(table).__name__}"
        )
    if not len(table):
        raise ValueError("table must have at least one row")
    if not _validation.is_real_dtype(table.columns.dtype):
        raise ValueError(
            "table must have one numeric column per month on book, got the column "
            f"{_validation.shown(table.columns[0])}"
        )
    for column in table.columns:
        _validation.check_real_dtype(table[column].dtype, f"month {column} of table")
    ax = _axes(ax, (7, 4))
    import matplotlib

    shades = matplotlib.colormaps["Blues"](np.linspace(0.95, 0.35, len(table))[::-1])
    months = table.columns.to_numpy(dtype=float)
    curves = []
    for (_, row), color in zip(table.iterrows(), shades, strict=True):
        values = row.to_numpy(dtype=float)
        observed = ~np.isnan(values)
        (curve,) = ax.plot(months[observed], values[observed], color=color, linewidth=1.8)
        curves.append(curve)
    _percent(ax.yaxis)
    ax.set_ylim(bottom=0)
    _xlabel(ax, table.columns)
    named = np.unique(np.linspace(0, len(table) - 1, _LEGEND_COHORTS).round().astype(int))
    labels = _labels(table.index)
    ax.legend(
        [curves[row] for row in named],
        [labels[row] for row in named],
        title=None if table.index.name is None else str(table.index.name),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize="small",
    )
    return ax


def survival_chart(
    table: pd.DataFrame,
    *,
    column: typing.Literal["cumulative_pd", "survival"] = "cumulative_pd",
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the cumulative PD or the survival curve of a survival table as steps.

    The curve starts at period 0 with 0 for ``cumulative_pd`` and 1 for
    ``survival`` and changes at each period, as steps, which is how the
    Kaplan-Meier estimate behaves between observations. When the table was
    computed with competing risks, the cumulative incidence of the competing
    cause is drawn next to that of the event.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.lifetime.survival_table`. It needs the column
        named by `column`, and with competing risks, which is when it has a
        ``competing`` column, also ``at_risk`` and ``survival``.
    column : {'cumulative_pd', 'survival'}, default 'cumulative_pd'
        Column to draw. ``survival`` is the all-cause survival, one curve
        whether or not there are competing risks.
    label : str, optional
        Legend entry of the curve, in place of `column`. With competing
        risks it is put in front of the name of each cause, as in
        ``'high cumulative_pd'`` and ``'high competing'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, `column` is neither
        ``'cumulative_pd'`` nor ``'survival'``, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.lifetime.survival_table : The table that this chart draws.
    vintage_chart : Cumulative bad rates by cohort.

    Notes
    -----
    With competing risks and ``column='cumulative_pd'`` the legend names
    the curves ``cumulative_pd`` and ``competing``. The cumulative
    incidence of the competing cause is the sum of the all-cause survival
    of the previous period times ``competing`` over ``at_risk``, as that of
    the event is, so the two curves and the survival add up to 1. Periods
    after the last one with accounts at risk have no value and end the
    curves.

    The curve starts one period before the first row of the table, so a
    slice of a table starts at its own first period, and it is drawn one
    period beyond the last row so that the last change is visible. The
    limits of the axes cover every curve already drawn on `ax`, so the
    curves of several tables, each with a `label`, can share one. A table
    without a single value, which is what `kivi.lifetime.survival_table`
    gives when nobody is at risk in its first period, draws an empty
    chart: a flat line at the start value.

    Examples
    --------
    >>> from kivi.lifetime import survival_table
    >>> from kivi.plotting import survival_chart
    >>> table = survival_table([1, 2, 3, 3, 4], [1, 0, 1, 1, 0])
    >>> ax = survival_chart(table)
    >>> ax.lines[0].get_ydata().round(3).tolist()
    [0.0, 0.2, 0.2, 0.733, 0.733, 0.733]
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('period', 'cumulative_pd')
    """
    if not isinstance(column, str) or column not in ("cumulative_pd", "survival"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            "column must be 'cumulative_pd' or 'survival', "
            f"got {_validation.shown_argument(column)}"
        )
    source = "kivi.lifetime.survival_table"
    needed: list[str] = [column]
    if isinstance(table, pd.DataFrame) and "competing" in table.columns and column != "survival":  # type: ignore[redundant-expr]  # table may not be a DataFrame yet
        needed += ["competing", "at_risk", "survival"]
    table = _table(table, needed, source)
    ax = _axes(ax, (6, 4))
    index = table.index.to_numpy(dtype=float)
    # One step before the first period, and one after the last so that its change is visible.
    periods = np.r_[index[0] - 1, index, index[-1] + 1]
    curves: list[tuple[str, float, np.ndarray]] = [
        (column, 1.0 if column == "survival" else 0.0, table[column].to_numpy(dtype=float))
    ]
    if len(needed) > 1:
        survival = table["survival"].to_numpy(dtype=float)
        previous = np.r_[1.0, survival[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = table["competing"].to_numpy(dtype=float) / table["at_risk"].to_numpy(
                dtype=float
            )
        curves.append(("competing", 0.0, np.cumsum(previous * share)))
    for name, start, values in curves:
        entry = name if label is None else f"{label} {name}" if len(curves) > 1 else label
        ax.plot(
            periods,
            np.r_[start, values, values[-1]],
            drawstyle="steps-post",
            linewidth=2,
            label=entry if len(curves) > 1 or label is not None else None,
        )
    # dataLim covers the finite points of every curve drawn on ax so far.
    left, right = ax.dataLim.x0, ax.dataLim.x1
    ax.set_xlim(left, right if right > left else left + 1)
    if column == "survival":
        ax.set_ylim(0, 1)
    else:
        upper = min(1.0, 1.1 * ax.dataLim.y1)
        ax.set_ylim(0, upper if upper > 0 else 1.0)
    _percent(ax.yaxis)
    ax.set_xlabel("period" if table.index.name is None else str(table.index.name))
    ax.set_ylabel(column)
    if len(curves) > 1 or label is not None:
        ax.legend(loc="best")
    return ax


def transition_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw a transition matrix as a heatmap with the value in each cell.

    The rows are the states before and the columns the states after. The
    shade of a cell is the share of its row that moved there, so that a
    table of counts and one of shares look alike, and the diagonal, the
    entities that stayed, is outlined in amber.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.vintage.transition_matrix`, counts or shares
        (``normalize=True``), with the states before in its index and the
        states after in its columns. Missing values leave their cells
        empty.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure, sized to the number of states, is
        created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, has no rows or fewer than two
        columns, has a column that is not numeric or a negative value, or
        `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.vintage.transition_matrix : The table that this chart draws.
    kivi.vintage.roll_rates : The share of each state that moved to the
        next worse one.
    correlation_chart : A heatmap of a correlation matrix.

    Notes
    -----
    A cell is outlined when its row label equals its column label, so a
    matrix whose states before and after are not the same set has an
    outline only where they meet. Whole numbers below 10,000 are written
    in full, shares, when no value exceeds 1, with two decimals, and any
    other value with three significant digits and a suffix ``K``, ``M`` or
    ``B``, so that weighted counts fit their cells. The
    heatmap is ``ax.images[0]``. The axes are labeled with the names of
    the index and the columns, ``before`` and ``after`` for the output of
    `kivi.vintage.transition_matrix`.

    Examples
    --------
    >>> from kivi.plotting import transition_chart
    >>> from kivi.vintage import transition_matrix
    >>> before = ["a", "b", "a", "c", "b", "c", "a", "c"]
    >>> after = ["a", "b", "b", "c", "c", "c", "a", "b"]
    >>> ax = transition_chart(transition_matrix(before, after))
    >>> [text.get_text() for text in ax.texts]
    ['2', '1', '0', '0', '1', '1', '0', '1', '2']
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('after', 'before')
    """
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            "table must be a DataFrame from kivi.vintage.transition_matrix, "
            f"got {type(table).__name__}"
        )
    if not len(table) or not table.shape[1]:
        raise ValueError("table must have at least one row and one column")
    if table.shape[1] < 2:
        raise ValueError(
            "table must have at least two columns, one for each state after; "
            "a Series such as the result of kivi.vintage.roll_rates is read as a table, "
            "not drawn"
        )
    for column in range(table.shape[1]):
        _validation.check_real_dtype(
            table.iloc[:, column].dtype, f"column {_validation.shown(table.columns[column])}"
        )
    values = table.to_numpy(dtype=float)
    if (values < 0).any():
        raise ValueError(
            f"table must not have negative values, got {_validation.shown(values[values < 0][0])}"
        )
    rows, columns = values.shape
    ax = _axes(ax, (max(4.0, 0.7 * columns + 1.8), max(3.2, 0.7 * rows + 1.2)), "compressed")
    import matplotlib.colors
    import matplotlib.patches

    totals = np.nansum(values, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(totals > 0, values / totals, 0.0)
    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "kivi_transition", ["white", _BLUE]
    )
    ax.imshow(np.where(np.isnan(values), np.nan, shares), cmap=colors, vmin=0, vmax=1)
    finite = values[np.isfinite(values)]
    whole = bool(np.all(finite == np.round(finite)))
    share = bool(finite.size and finite.max() <= 1)
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(
                column,
                row,
                _cell(value, whole, share),
                ha="center",
                va="center",
                fontsize="small",
                color="white" if shares[row, column] > 0.6 else _TEXT,
            )
    for row, state in enumerate(table.index):
        for column, other in enumerate(table.columns):
            if state == other:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=_AMBER,
                        linewidth=2.5,
                    )
                )
    _ticks(ax, np.arange(columns), _labels(table.columns))
    ax.set_yticks(np.arange(rows), _labels(table.index))
    ax.set_xlabel("" if table.columns.name is None else str(table.columns.name))
    ax.set_ylabel("" if table.index.name is None else str(table.index.name))
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def rule_set_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw the share of the book that each rule declines and the approval rate left.

    A rule set is a waterfall: each rule declines only the applicants that
    no earlier rule declined. The bars show the ``new_hits`` of each rule,
    in the order of the table, as a share of all rows, and a line shows the
    ``approval_rate`` of the book after the rule, on the same axis. The
    approval rate after a rule is therefore the one before it minus its
    bar, and the first starts from 100%.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.rules.rule_set_table`. It needs the columns
        ``new_hits``, ``approved`` and ``approval_rate``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.rules.rule_set_table : The table that this chart draws.
    cutoff_chart : The same trade-off for a score instead of rules.

    Notes
    -----
    The total that the bars are shares of is the ``approved`` of the first
    rule plus its ``new_hits``, so it counts every row of the rules,
    outcomes known or not, and is a total weight when the table was
    computed with weights. The tick labels are the rule names of the index.
    The legend sits at the top of the Axes, where it covers no bar and no
    point.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.plotting import rule_set_chart
    >>> from kivi.rules import rule_set_table
    >>> rules = pd.DataFrame(
    ...     {"high_debt": [True, False, True, False], "young": [False, False, True, True]}
    ... )
    >>> ax = rule_set_chart(rule_set_table(rules))
    >>> [float(patch.get_height()) for patch in ax.patches]
    [0.5, 0.25]
    >>> ax.lines[0].get_ydata().tolist()
    [0.5, 0.25]
    """
    table = _table(table, ["new_hits", "approved", "approval_rate"], "kivi.rules.rule_set_table")
    ax = _axes(ax, (7, 4))
    x = np.arange(len(table))
    hits = table["new_hits"].to_numpy(dtype=float)
    total = hits[0] + float(table["approved"].iloc[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = hits / total
    rates = table["approval_rate"].to_numpy(dtype=float)
    bars = ax.bar(x, shares, width=0.6, color=_LIGHT_BLUE, label="new_hits")
    (curve,) = ax.plot(x, rates, color=_AMBER, marker="o", linewidth=2, label="approval_rate")
    columns = _legend_columns(ax, ["new_hits", "approval_rate"])
    ax.set_ylim(0, _top(np.r_[shares, rates], 0.0, _legend_room(ax, 1 + 2 // columns)))
    _percent(ax.yaxis)
    ax.set_ylabel("new_hits, approval_rate")
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    ax.legend([bars, curve], ["new_hits", "approval_rate"], loc="upper left", ncols=columns)
    return ax


def profit_chart(
    table: pd.DataFrame,
    *,
    label: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw the profit against the approval rate and mark the best cut-off.

    Each cut-off of the table is one point, with ``approval_rate`` on the
    x-axis and ``profit``, the revenue minus the loss of the approved book,
    on the y-axis. The cut-off with the highest profit is marked in amber
    and labeled with its value and its profit, and a dotted line marks
    zero profit.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.strategy.cutoff_table` computed with both
        `loss` and `revenue`. It needs the columns ``approval_rate`` and
        ``profit``.
    label : str, optional
        Legend entry of the curve. A legend is drawn only when a curve has
        a label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, no row has a profit, or
        `ax` is not a matplotlib Axes.

    See Also
    --------
    kivi.strategy.cutoff_table : The table that this chart draws.
    kivi.metrics.expected_maximum_profit : The profit of the best cut-off
        under a loss distribution.
    cutoff_chart : The event rate against the approval rate.

    Notes
    -----
    The best cut-off is the first row with the largest profit, in the
    order of the table, and rows with a missing profit are left out. Its
    label shows the index of the row, the cut-off, with four significant
    digits. The curve takes the next color of the Axes' color cycle, so
    that the tables of a current and a new score can be compared on one
    `ax`, each with a `label`; the best cut-off of each is marked in amber.

    Examples
    --------
    >>> from kivi.plotting import profit_chart
    >>> from kivi.strategy import cutoff_table
    >>> score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    >>> target = [0, 0, 0, 1, 0, 1]
    >>> loss = [0, 0, 0, 5, 0, 5]
    >>> table = cutoff_table(score, target, cutoffs=[0.2, 0.3, 0.6], loss=loss, revenue=[1] * 6)
    >>> ax = profit_chart(table)
    >>> ax.texts[0].get_text()
    'cutoff = 0.3, profit = 3'
    """
    table = _table(table, ["approval_rate", "profit"], "kivi.strategy.cutoff_table")
    profit = table["profit"].to_numpy(dtype=float)
    if np.isnan(profit).all():
        raise ValueError("table must have at least one row with a profit")
    ax = _axes(ax, (5.6, 4.2))
    rate = table["approval_rate"].to_numpy(dtype=float)
    ax.plot(rate, profit, marker="o", markersize=4, linewidth=2, label=label)
    ax.axhline(0, color=_GRAY, linestyle=":", linewidth=1)
    best = int(np.nanargmax(profit))
    ax.plot([rate[best]], [profit[best]], marker="o", markersize=9, color=_AMBER, linestyle="")
    name = "cutoff" if table.index.name is None else str(table.index.name)
    ax.annotate(
        f"{name} = {_cutoff_label(table.index[best])}, profit = {profit[best]:,.10g}",
        (rate[best], profit[best]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize="small",
        color=_AMBER,
    )
    ax.margins(y=0.15)
    _percent(ax.xaxis)
    ax.set_xlabel("approval_rate")
    ax.set_ylabel("profit")
    if label is not None:
        ax.legend(loc="best")
    return ax


def lgd_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw the mean LGD of each group with its shares of zero and full losses.

    Each group of the table gets three bars: ``mean``, the average LGD,
    ``zero_share``, the share of its exposures that lost nothing, and
    ``full_share``, the share that lost everything or more. The shares make
    visible the bimodal shape that a mean hides.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.lgd.lgd_table`. It needs the columns ``mean``,
        ``zero_share`` and ``full_share``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when it is not given.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, `ax` when it is given.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If `table` is not a DataFrame, lacks a column it needs or has no
        rows, a column it needs is not numeric, or `ax` is not a matplotlib
        Axes.

    See Also
    --------
    kivi.lgd.lgd_table : The table that this chart draws.
    kivi.lgd.workout_lgd : The realized LGDs of the loans.
    psi_chart : Bars of two shares side by side for each group.

    Notes
    -----
    All three bars are on one percent axis, which runs below 0 or above
    100% when a mean LGD does, as realized LGDs can. The tick labels are
    the group labels of the index, and the legend sits at the top, where it
    covers no bar. A group with a missing ``mean`` has no such bar.

    Examples
    --------
    >>> from kivi.lgd import lgd_table
    >>> from kivi.plotting import lgd_chart
    >>> table = lgd_table(["a", "a", "b", "b"], [0.0, 0.5, 1.0, 1.2])
    >>> ax = lgd_chart(table)
    >>> [round(float(patch.get_height()), 3) for patch in ax.patches]
    [0.25, 1.1, 0.5, 0.0, 0.0, 1.0]
    """
    table = _table(table, ["mean", "zero_share", "full_share"], "kivi.lgd.lgd_table")
    ax = _axes(ax, (7, 4))
    x = np.arange(len(table))
    series = [
        ("mean", _BLUE, -0.27),
        ("zero_share", _GRAY, 0.0),
        ("full_share", _AMBER, 0.27),
    ]
    values = np.concatenate([table[name].to_numpy(dtype=float) for name, _, _ in series])
    for name, color, shift in series:
        ax.bar(x + shift, table[name].to_numpy(dtype=float), width=0.26, color=color, label=name)
    ax.axhline(0, color=_TEXT, linewidth=0.8)
    columns = _legend_columns(ax, [name for name, _, _ in series])
    room = _legend_room(ax, 1 + 3 // columns - 1)
    low = min(0.0, float(np.nanmin(values))) if np.isfinite(values).any() else 0.0
    ax.set_ylim(low, _top(values, low, room))
    _percent(ax.yaxis)
    ax.set_ylabel("mean, zero_share, full_share")
    _xlabel(ax, table.index)
    _ticks(ax, x, _labels(table.index))
    ax.legend(loc="upper left", ncols=columns)
    return ax


def _axes(
    ax: "matplotlib.axes.Axes | None",
    size: tuple[float, float],
    layout: typing.Literal["constrained", "compressed"] = "constrained",
) -> "matplotlib.axes.Axes":
    """Return `ax`, or the Axes of a new figure of `size` inches; import pyplot only for that."""
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(_INSTALL) from error
        _, created = plt.subplots(figsize=size, layout=layout)
        return created
    try:
        import matplotlib.axes
    except ImportError as error:
        raise ImportError(_INSTALL) from error
    if not isinstance(ax, matplotlib.axes.Axes):
        raise ValueError(f"ax must be a matplotlib Axes, got {type(ax).__name__}")
    return ax


def _table(table: pd.DataFrame, columns: abc.Sequence[str], source: str) -> pd.DataFrame:
    """Check that `table` is a DataFrame with rows and the numeric `columns` of `source`."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"table must be a DataFrame from {source}, got {type(table).__name__}")
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"table must have the columns {_validation.shown_list(columns)} of {source}, "
            f"missing {_validation.shown_list(missing)}"
        )
    if not len(table):
        raise ValueError("table must have at least one row")
    for column in columns:
        _validation.check_real_dtype(table[column].dtype, f"{column} of table")
    return table


def _cutoff_label(cutoff: object) -> str:
    """Return a cut-off with four significant digits, or a label as text."""
    if _validation.is_real(cutoff) and not isinstance(cutoff, (bool, np.bool_)):
        return f"{cutoff:.4g}"
    return str(cutoff)


def _cell(value: float, whole: bool, share: bool) -> str:
    """Return the text of a heatmap cell that fits it: whole numbers, shares or K, M, B."""
    if whole and abs(value) < 10_000:
        return f"{value:.0f}"
    if share and not whole:
        return f"{value:.2f}"
    magnitude = abs(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def _levels(levels: float | abc.Sequence[float]) -> list[float]:
    """Return the `levels` of `monitoring_chart` as a list of finite floats."""
    values: abc.Iterable[object] = (
        levels if isinstance(levels, abc.Iterable) and np.ndim(levels) == 1 else [levels]
    )
    checked: list[float] = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not _validation.is_real(value)
            or not math.isfinite(value)
        ):
            raise ValueError(f"levels must be finite numbers, got {_validation.shown(value)}")
        checked.append(float(value))
    return checked


def _labels(index: pd.Index) -> list[str]:
    """Return the tick labels of an index, with ``NaN`` for a missing label."""
    return ["NaN" if _validation.is_missing(value) else str(value) for value in index]


def _xlabel(ax: "matplotlib.axes.Axes", index: pd.Index) -> None:
    """Label the x-axis with the name of `index`, when it has one."""
    if index.name is not None:
        ax.set_xlabel(str(index.name))


def _ticks(ax: "matplotlib.axes.Axes", x: np.ndarray, labels: list[str]) -> None:
    """Label the x ticks, slanted when the longest label is wider than the space of a tick."""
    import matplotlib.font_manager

    size = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["xtick.labelsize"]
    ).get_size_in_points()
    # About half an em per character, with two characters of space between labels.
    needed = (max(map(len, labels)) + 2) * 0.5 * size / 72
    available = ax.bbox.width / ax.figure.dpi / len(labels)
    if needed > available:
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize="small")
    else:
        ax.set_xticks(x, labels)


def _legend_columns(ax: "matplotlib.axes.Axes", labels: list[str]) -> int:
    """Return how many columns a legend of `labels` can have, side by side or one."""
    import matplotlib.font_manager

    size = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["legend.fontsize"]
    ).get_size_in_points()
    # Measured: about 0.53 font sizes per character and 4 more per entry for the handle
    # and the spacing, plus the frame.
    width = (sum(0.55 * len(label) + 4 for label in labels) + 1) * size / 72
    return len(labels) if width <= ax.bbox.width / ax.figure.dpi else 1


def _legend_room(ax: "matplotlib.axes.Axes", rows: int) -> float:
    """Return the share of the height of `ax` that a legend of `rows` text rows at its top takes."""
    import matplotlib.font_manager

    size = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["legend.fontsize"]
    ).get_size_in_points()
    # Measured: a row takes about 1.55 font sizes; add the padding, the gap to the
    # top of the Axes and a margin for the space that the layout later takes.
    height = (1.6 * rows + 1.5) * size / 72
    return min(0.6, height / (ax.bbox.height / ax.figure.dpi))


def _top(values: np.ndarray, bottom: float, room: float) -> float:
    """Return the upper limit that keeps every value below the top `room` of the height."""
    finite = values[np.isfinite(values)]
    largest = float(finite.max()) if finite.size else bottom
    if largest <= bottom:
        return bottom + 1.0
    return bottom + 1.03 * (largest - bottom) / (1 - room)


def _span(values: np.ndarray) -> tuple[float, float]:
    """Return the smallest and the largest finite value, with 0 between them."""
    finite = values[np.isfinite(values)]
    low = min(0.0, float(finite.min())) if finite.size else 0.0
    high = max(0.0, float(finite.max())) if finite.size else 0.0
    return low, high


def _percent(axis: "matplotlib.axis.Axis") -> None:
    """Show the ticks of an axis of shares as percentages, such as 0.125 as 12.5%."""
    import matplotlib.ticker

    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(_percent_text))


def _percent_text(value: float, position: int | None = None) -> str:
    """Return a share as a percentage without trailing zeros or rounding noise."""
    return f"{100 * value + 0.0:g}%"  # + 0.0 turns -0.0 into 0.0


def _gap(
    ax: "matplotlib.axes.Axes",
    x: float,
    low: float,
    high: float,
    color: "matplotlib.typing.ColorType",
) -> None:
    """Mark the KS gap from `low` to `high` at `x` with a segment and its value."""
    ax.plot([x, x], [low, high], color=color, linestyle=":", linewidth=1.8)
    ax.annotate(
        f"ks = {abs(high - low):.3f}",
        (x, (low + high) / 2),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize="small",
        color=color,
    )
