"""Charts of kivi's result tables for risk reports.

Each function draws one result of another kivi module on a matplotlib
Axes: ``woe_chart`` a ``kivi.woe.woe_table``, ``correlation_chart`` a
correlation matrix such as ``X.corr()``, ``roc_chart`` a
``kivi.metrics.roc_table``, ``gains_chart`` a ``kivi.metrics.gains_table``,
``calibration_chart`` a ``kivi.calibration.calibration_table``,
``cutoff_chart`` a ``kivi.strategy.cutoff_table``, ``psi_chart`` a
``kivi.stability.psi_table``, ``monitoring_chart`` a column of a
``kivi.monitoring.monitoring_table`` and ``vintage_chart`` a
``kivi.vintage.vintage_table``.

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
    "gains_chart",
    "calibration_chart",
    "cutoff_chart",
    "psi_chart",
    "monitoring_chart",
    "vintage_chart",
]

_BLUE = "#2f5bd3"
_LIGHT_BLUE = "#9fb4ee"
_TEAL = "#14a397"
_AMBER = "#d97706"
_RED = "#dc2626"
_GRAY = "#9ca3af"
_TEXT = "#1f2937"
_LEVEL_COLORS = (_AMBER, _RED, _GRAY)
# monitoring_chart draws these columns as bars and the others as a line.
_BAR_COLUMNS = ("total", "known", "event", "psi")
_INSTALL = 'kivi.plotting needs matplotlib; install it with pip install "kivi[plot]"'


def woe_chart(
    table: pd.DataFrame,
    *,
    line: typing.Literal["event_rate", "woe"] = "event_rate",
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple["matplotlib.axes.Axes", "matplotlib.axes.Axes"]:
    """Draw the population share of each group with its event rate or WOE.

    The bars show ``total_share``, the share of the sample in each group,
    and a line on a second y-axis shows ``event_rate`` or ``woe``. The
    groups keep the order of the table, and the legend gives the IV of the
    whole table.

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
    bars = ax.bar(
        x,
        table["total_share"].to_numpy(dtype=float),
        width=0.6,
        color=_LIGHT_BLUE,
        label="total_share",
    )
    ax.set_ylabel("total_share")
    _percent(ax.yaxis)
    ax.margins(y=0.3)
    _ticks(ax, x, _labels(table.index))
    twin = ax.twinx()
    values = table[line].to_numpy(dtype=float)
    if line == "woe":
        twin.axhline(0, color=_GRAY, linewidth=1)
    (curve,) = twin.plot(x, values, color=_AMBER, marker="o", linewidth=2, label=line)
    twin.margins(y=0.3)
    if line == "event_rate":
        _percent(twin.yaxis)
        twin.set_ylim(0, _upper(values, 1.3))
    twin.set_ylabel(line)
    twin.grid(False)
    twin.legend(
        [bars, curve],
        ["total_share", line],
        loc="upper left",
        ncols=2,
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
        labels in its index and columns, or has a column that is not
        numeric, or `ax` is not a matplotlib Axes.

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
    size = len(matrix)
    ax = _axes(ax, (max(4.0, 0.6 * size + 1.6), max(3.2, 0.6 * size + 1.0)), "compressed")
    import matplotlib.colors

    values = matrix.to_numpy(dtype=float)
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


def gains_chart(
    table: pd.DataFrame, *, ax: "matplotlib.axes.Axes | None" = None
) -> "matplotlib.axes.Axes":
    """Draw the cumulative shares of events and non-events by score band.

    Reviewing the bands from the highest scores down, the curves show the
    share of all events and of all non-events captured against the share
    of the sample reviewed, ``cum_total_share``. A vertical segment marks
    the band with the largest ``ks``, the gap between the two curves.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.metrics.gains_table`, with the highest band
        first. It needs the columns ``cum_total_share``,
        ``cum_event_share``, ``cum_non_event_share`` and ``ks``.
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
    kivi.metrics.gains_table : The table that this chart draws.
    roc_chart : The same separation over all thresholds.

    Notes
    -----
    Both curves start at the origin, before the first band is reviewed,
    and each band adds one point. The diagonal is the ``cum_event_share``
    of a score that ranks at random.

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
    table = _table(
        table,
        ["cum_total_share", "cum_event_share", "cum_non_event_share", "ks"],
        "kivi.metrics.gains_table",
    )
    ax = _axes(ax, (5.6, 4.5))
    share = np.r_[0.0, table["cum_total_share"].to_numpy(dtype=float)]
    event = np.r_[0.0, table["cum_event_share"].to_numpy(dtype=float)]
    non_event = np.r_[0.0, table["cum_non_event_share"].to_numpy(dtype=float)]
    ax.plot([0, 1], [0, 1], color=_GRAY, linestyle="--", linewidth=1)
    ax.plot(
        share, event, color=_BLUE, marker="o", markersize=4, linewidth=2, label="cum_event_share"
    )
    ax.plot(
        share,
        non_event,
        color=_TEAL,
        marker="o",
        markersize=4,
        linewidth=2,
        label="cum_non_event_share",
    )
    largest = int(np.argmax(table["ks"].to_numpy(dtype=float))) + 1
    _gap(ax, share[largest], non_event[largest], event[largest], _RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent(ax.xaxis)
    _percent(ax.yaxis)
    ax.set_xlabel("cum_total_share")
    ax.legend(loc="lower right")
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
        ax.legend(loc="upper left")
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
        ax.legend(loc="upper left")
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
    ax.bar(
        x - 0.2,
        table["reference_share"].to_numpy(dtype=float),
        width=0.4,
        color=_LIGHT_BLUE,
        label="reference_share",
    )
    ax.bar(
        x + 0.2,
        table["current_share"].to_numpy(dtype=float),
        width=0.4,
        color=_BLUE,
        label="current_share",
    )
    _percent(ax.yaxis)
    ax.margins(y=0.2)
    _ticks(ax, x, _labels(table.index))
    ax.legend(title=f"psi = {table['psi'].sum():.3f}")
    return ax


def monitoring_chart(
    table: pd.DataFrame,
    *,
    column: str = "psi",
    levels: float | abc.Sequence[float] = (),
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw one column of a monitoring report period by period.

    The periods of the table run along the x-axis in its order. The counts
    ``total``, ``known`` and ``event`` and the ``psi`` are drawn as bars,
    and the rates and measures of discrimination, such as ``event_rate``
    and ``gini``, as a line. Dashed lines can mark the `levels` that the
    column is compared with.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from `kivi.monitoring.monitoring_table`, or another table
        with one row per period, such as one from
        `kivi.monitoring.stability_table`. It needs the column named by
        `column`.
    column : str, default 'psi'
        Column to draw.
    levels : float or sequence of float, default ()
        Values of `column` to mark with dashed horizontal lines, such as
        the PSI thresholds ``[0.1, 0.25]`` or the Gini of the development
        sample. Each line is labeled with its value.
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
        named `column`, a level is not a finite number, or `ax` is not a
        matplotlib Axes.

    See Also
    --------
    kivi.monitoring.monitoring_table : The table that this chart draws.
    psi_chart : The shares behind the PSI of one period.

    Notes
    -----
    A missing value, such as the ``gini`` of a period whose outcomes are
    not known yet, leaves a gap in the line and no bar. To draw several
    columns, call the function once for each, on the panels of one figure.

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
    marks = _levels(levels)
    table = _table(table, [column], "kivi.monitoring.monitoring_table")
    ax = _axes(ax, (7, 3.6))
    x = np.arange(len(table))
    values = table[column].to_numpy(dtype=float)
    if column in _BAR_COLUMNS:
        ax.bar(x, values, width=0.6, color=_BLUE)
    else:
        ax.plot(x, values, color=_BLUE, marker="o", linewidth=2)
    for number, level in enumerate(marks):
        color = _LEVEL_COLORS[min(number, len(_LEVEL_COLORS) - 1)]
        ax.axhline(level, color=color, linestyle="--", linewidth=1)
        ax.text(
            1.005,
            level,
            f"{level:g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize="small",
            color=color,
        )
    if column == "event_rate":
        _percent(ax.yaxis)
    ax.set_ylabel(column)
    _ticks(ax, x, _labels(table.index))
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
    The legend names every cohort, titled with the name of the index, and
    the x-axis is labeled with the name of the columns. Select fewer
    cohorts first, for example every third one with ``table.iloc[::3]``,
    when there are too many curves to tell apart.

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
    for (cohort, row), color in zip(table.iterrows(), shades, strict=True):
        values = row.to_numpy(dtype=float)
        observed = ~np.isnan(values)
        ax.plot(months[observed], values[observed], color=color, linewidth=1.8, label=str(cohort))
    _percent(ax.yaxis)
    ax.set_ylim(bottom=0)
    if table.columns.name is not None:
        ax.set_xlabel(str(table.columns.name))
    ax.legend(
        title=None if table.index.name is None else str(table.index.name),
        ncols=2 if len(table) > 8 else 1,
        fontsize="small",
    )
    return ax


def _axes(
    ax: "matplotlib.axes.Axes | None",
    size: tuple[float, float],
    layout: typing.Literal["constrained", "compressed"] = "constrained",
) -> "matplotlib.axes.Axes":
    """Import matplotlib and return `ax`, or the Axes of a new figure of `size` inches."""
    try:
        import matplotlib.axes
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(_INSTALL) from error
    if ax is None:
        _, created = plt.subplots(figsize=size, layout=layout)
        return created
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


def _percent(axis: "matplotlib.axis.Axis") -> None:
    """Show the ticks of an axis of shares as percentages, such as 0.125 as 12.5%."""
    import matplotlib.ticker

    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(_percent_text))


def _percent_text(value: float, position: int | None = None) -> str:
    """Return a share as a percentage without trailing zeros or rounding noise."""
    return f"{100 * value + 0.0:g}%"  # + 0.0 turns -0.0 into 0.0


def _upper(values: np.ndarray, factor: float) -> float:
    """Return `factor` times the largest finite value, or 1 when there is none above 0."""
    finite = values[np.isfinite(values)]
    largest = float(finite.max()) if finite.size else 0.0
    return factor * largest if largest > 0 else 1.0


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
