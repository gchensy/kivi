"""Diagrams that show how kivi computes its results.

Each ``figure_*`` recipe recomputes the example of the doctest just above it and draws the
computation as tables: the input rows, the call and the result that kivi returns, with its
numbers. Like the charts of _plotting.py, the diagrams contain only kivi's column and
parameter names, data values and formulas, so the same images serve the English and Chinese
editions. The pages call the recipes from ``.. plot::`` directives as ``kd.figure_*()``.

The colors keep one meaning in every diagram:

* groups and bins: one tint per category, a light-to-dark blue ramp for ordered intervals,
  and gray for missing values;
* target: events (1) in red, unknown outcomes (NaN) in gray;
* the result that the diagram is about: amber;
* the reference sample in blue, the current sample in teal.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import _plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator

from kivi.binning import bin_edges, ks_edges
from kivi.calibration import isotonic_fit
from kivi.datasets import load_bank, make_applications, make_loan_panel
from kivi.expert import ahp_consistency, ahp_weights
from kivi.features import panel_features, window_features
from kivi.lifetime import assign_stages, convert_pd, expected_credit_loss
from kivi.metrics import generalized_auc, roc_auc, roc_table
from kivi.monitoring import monitoring_table
from kivi.reject_inference import fuzzy_augmentation, parceling, reweighting
from kivi.rules import rule_set_table
from kivi.scorecard import score_scaling, woe_to_points
from kivi.selection import correlation_filter, logistic_regression, stepwise
from kivi.spec import BinningSpec, fit_spec
from kivi.stability import psi_table
from kivi.strategy import confusion_table, cutoff_table, swap_set
from kivi.univariate import profile_table
from kivi.vintage import (
    delinquency_panel,
    migration_bandwidth,
    migration_test,
    roll_rates,
    transition_matrix,
)
from kivi.woe import woe_table, woe_transform

UNIT = 0.25  # inches per grid unit, 18 points
WIDTH = 32  # grid units, so that diagrams are as wide as the charts (8 inches)
ROW = 1.0  # distance between two table rows
CELL = 0.82  # height of a table cell
GAP = 0.16  # horizontal gap between two table columns
HEADER = 1.05  # height of a table header
CHAR = 0.602 / 18  # width of a DejaVu Sans Mono character per point of font size
MONO = "DejaVu Sans Mono"

INK = "#1f2937"
MUTED = "#6b7280"
FAINT = "#9ca3af"
RULE = "#d1d5db"
CELL_FILL = "#ffffff"
CELL_EDGE = "#e5e7eb"
CARD_FILL = "#f7f9ff"
CARD_EDGE = "#d5def5"
ARROW = "#94a3b8"
CHIP = "#e6edfc"
CHIP_TEXT = "#1d3d9e"
EVENT = "#dc2626"
EVENT_FILL = "#fdecec"
MISSING_FILL = "#f3f4f6"
RESULT_FILL = "#fdf0dc"
RESULT_EDGE = "#f1c27d"
RESULT_TEXT = "#b45309"
AMBER = "#d97706"
BLUE = "#2f5bd3"
BLUE_FILL = "#e3e9fb"
TEAL = "#0b7a70"
TEAL_FILL = "#d3efeb"

# One tint per category, with the text color that goes with it.
CATEGORY = [
    ("#dfe7fb", "#1d3d9e"),
    ("#d3efeb", "#0b5c55"),
    ("#fbe8cc", "#92400e"),
    ("#ece6fb", "#5b21b6"),
    ("#fbe1e6", "#9f1239"),
    ("#e5e7eb", "#374151"),
]
MISSING = (MISSING_FILL, FAINT)
# Ordered intervals, from the lowest to the highest.
RAMP = ["#eef2fd", "#dbe4fb", "#c3d2f7", "#a6bcf1", "#86a2ea", "#6788e2"]


def ramp(n: int) -> list[tuple[str, str]]:
    """``n`` tints of the blue ramp, spread over its whole range, with dark text."""
    if n == 1:
        return [(RAMP[2], INK)]
    return [(RAMP[round(i * (len(RAMP) - 1) / (n - 1))], INK) for i in range(n)]


def number(value, digits: int = 3) -> str:
    """Format a number as the doctests print it, with NaN for missing values."""
    if pd.isna(value):
        return "NaN"
    return f"{round(float(value), digits) + 0.0:.{digits}f}"


def label(value) -> str:
    return "NaN" if pd.isna(value) else str(value)


def text_width(s: str, size: float = 9.5) -> float:
    return len(s) * CHAR * size


@dataclass
class Column:
    """One column of a drawn table: a header and one cell per row."""

    name: str
    cells: Sequence[str]
    width: float
    fill: str | Sequence[str] = CELL_FILL
    color: str | Sequence[str] = INK
    bold: bool | Sequence[bool] = False
    edge: str | Sequence[str | None] | None = None
    header_color: str = INK

    def get(self, attribute: str, row: int):
        value = getattr(self, attribute)
        if value is None or isinstance(value, str | bool):
            return value
        return value[row]


def result(name: str, cells: Sequence[str], width: float, **style) -> Column:
    """A column of the result that the diagram is about, drawn in amber."""
    style = {"fill": RESULT_FILL, "edge": RESULT_EDGE, "header_color": RESULT_TEXT} | style
    return Column(name, cells, width, **style)


def target_column(values, width: float = 2.4, name: str = "target") -> Column:
    """A target column: events in red, unknown outcomes in gray."""
    cells = [label(value) if pd.isna(value) else str(int(value)) for value in values]
    return Column(
        name,
        cells,
        width,
        fill=[MISSING_FILL if c == "NaN" else EVENT_FILL if c == "1" else CELL_FILL for c in cells],
        color=[FAINT if c == "NaN" else EVENT if c == "1" else INK for c in cells],
        bold=[c == "1" for c in cells],
    )


def group_styles(labels, palette: str = "category") -> dict:
    """A tint and a text color for each distinct label that is not missing."""
    distinct = [value for value in dict.fromkeys(labels) if not pd.isna(value)]
    if palette == "ramp":
        return dict(zip(distinct, ramp(len(distinct)), strict=True))
    return {value: CATEGORY[i % len(CATEGORY)] for i, value in enumerate(distinct)}


def group_column(name: str, labels, width: float, styles: dict, **extra) -> Column:
    """A column of group labels, each in the tint of its group."""
    labels = list(labels)
    tints = [MISSING if pd.isna(value) else styles[value] for value in labels]
    return Column(
        name,
        [label(value) for value in labels],
        width,
        fill=[tint[0] for tint in tints],
        color=[tint[1] for tint in tints],
        **extra,
    )


@dataclass
class Layout:
    """Where a drawn table ended up."""

    left: float
    right: float
    top: float
    bottom: float
    rows: list[float]
    columns: dict[str, tuple[float, float]]

    def center(self, name: str) -> float:
        low, high = self.columns[name]
        return (low + high) / 2


class Canvas:
    """A figure in grid units, with the origin at the top left and y pointing down."""

    def __init__(self, height: float, width: float = WIDTH):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width * UNIT, height * UNIT))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_axis_off()

    def box(self, x, y, width, height, *, fill, edge=None, radius=0.14, linewidth=0.8, z=1):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=edge or fill,
                linewidth=linewidth,
                zorder=z,
            )
        )

    def card(self, x, y, width, height):
        self.box(x, y, width, height, fill=CARD_FILL, edge=CARD_EDGE, radius=0.45, z=0)

    def text(self, x, y, s, *, size=9.5, color=INK, bold=False, ha="center", z=4):
        self.ax.text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            fontweight="bold" if bold else "normal",
            ha=ha,
            va="center",
            family=MONO,
            zorder=z,
        )

    def line(self, x0, y0, x1, y1, *, color=RULE, linewidth=0.9, style="-", z=2):
        self.ax.plot(
            [x0, x1], [y0, y1], color=color, linewidth=linewidth, linestyle=style, zorder=z
        )

    def arrow(self, start, end, *, color=ARROW, linewidth=1.4, bend=0.0, both=False, z=3):
        style = "<|-|>" if both else "-|>"
        self.ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=f"{style},head_length=0.55,head_width=0.3",
                mutation_scale=12,
                color=color,
                linewidth=linewidth,
                connectionstyle=f"arc3,rad={bend}",
                shrinkA=0,
                shrinkB=0,
                zorder=z,
            )
        )

    def call(self, x, y, s, *, size=9.5, ha="left") -> tuple[float, float]:
        """A function call in a chip, from (x, y) at its top; returns its left and right ends."""
        width = text_width(s, size) + 0.9
        left = {"left": x, "center": x - width / 2, "right": x - width}[ha]
        self.box(left, y, width, 1.0, fill=CHIP, radius=0.25, z=2)
        self.text(left + 0.45, y + 0.5, s, size=size, color=CHIP_TEXT, ha="left")
        return left, left + width

    def note(self, x, y, s, *, size=8.8, color=MUTED, ha="left", bold=False):
        self.text(x, y, s, size=size, color=color, ha=ha, bold=bold)

    def notes(self, x, y, lines: Sequence[str], *, step=0.95, **style) -> float:
        """Several lines of notes from (x, y); returns the y of the line after the last."""
        for i, s in enumerate(lines):
            self.note(x, y + i * step, s, **style)
        return y + len(lines) * step

    def axes(self, x, y, width, height):
        """Matplotlib axes over the given area of the grid."""
        return self.fig.add_axes(
            (
                x / self.width,
                1 - (y + height) / self.height,
                width / self.width,
                height / self.height,
            )
        )

    def table(self, x, y, columns: Sequence[Column], *, header=True) -> Layout:
        """Draw the columns side by side from (x, y), the top left corner of the header."""
        n = len(columns[0].cells)
        top = y + (HEADER if header else 0.0)
        rows = [top + (i + 0.5) * ROW for i in range(n)]
        spans = {}
        left = x
        for column in columns:
            right = left + column.width
            spans[column.name] = (left, right)
            if header:
                self.text(
                    (left + right) / 2,
                    y + HEADER / 2 - 0.05,
                    column.name,
                    size=9,
                    bold=True,
                    color=column.header_color,
                )
            for i, center in enumerate(rows):
                fill = column.get("fill", i)
                self.box(
                    left,
                    center - CELL / 2,
                    column.width,
                    CELL,
                    fill=fill,
                    edge=column.get("edge", i) or (CELL_EDGE if fill == CELL_FILL else fill),
                )
                self.text(
                    (left + right) / 2,
                    center,
                    column.cells[i],
                    color=column.get("color", i),
                    bold=column.get("bold", i),
                )
            left = right + GAP
        right = left - GAP
        if header:
            self.line(x, top - 0.06, right, top - 0.06)
        return Layout(x, right, y, top + n * ROW, rows, spans)


def _style_axes(ax) -> None:
    ax.set_facecolor("none")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    for text in (ax.xaxis.label, ax.yaxis.label):
        text.set_fontsize(8.5)
        text.set_family(MONO)
        text.set_color(MUTED)


# Binning


def figure_bin_edges_cut():
    """Edges fitted on a reference sample, then applied to a new sample with pandas.cut."""
    reference = pd.Series([10, 20, 30, 40, 50, 60])
    edges = bin_edges(reference, bins=2)
    sample = pd.Series([5, 35, 99, None])
    groups = pd.cut(sample, edges)
    labels = list(groups.cat.categories)
    styles = dict(zip(labels, ramp(len(labels)), strict=True))

    canvas = Canvas(height=10.2)
    column = group_column("reference", pd.cut(reference, edges), 3.2, styles, header_color=BLUE)
    column.cells = [str(v) for v in reference]
    ref = canvas.table(0.6, 2.0, [column])
    canvas.call(0.6, 0.2, "bin_edges(reference, bins=2)")
    canvas.note(10.4, 0.7, f"-> array({np.array2string(edges, separator=', ')})", size=9)

    # A number line from 0 to 100 that extends to -inf and inf.
    x0, x1, line_y = 6.6, 20.4, 6.2

    def at(value):
        return x0 + (value / 100) * (x1 - x0)

    cut = at(edges[1])
    for (low, high), name in zip(((x0 - 1.0, cut), (cut, x1 + 1.0)), labels, strict=True):
        canvas.box(
            low, 3.0, high - low, 5.2, fill=_blend("#ffffff", styles[name][0], 0.7), radius=0.3, z=0
        )
        canvas.note((low + high) / 2, 3.6, str(name), ha="center", color=INK)
    canvas.line(cut, 2.2, cut, line_y + 1.0, color=AMBER, linewidth=1.6, style="--", z=3)
    canvas.note(cut, 2.0, number(edges[1], 1), ha="center", color=RESULT_TEXT, bold=True)
    canvas.arrow((x0 - 0.9, line_y), (x1 + 0.9, line_y), color=MUTED, linewidth=1.1, both=True)
    canvas.note(x0 - 0.5, line_y - 0.75, "-inf", ha="left")
    canvas.note(x1 + 0.5, line_y - 0.75, "inf", ha="right")
    for value in reference:
        canvas.ax.plot(at(value), line_y - 0.75, "o", color=BLUE, markersize=6.5, zorder=5)
    for value in sample.dropna():
        x = at(min(value, 100))
        canvas.ax.plot(x, line_y + 0.75, "D", color=TEAL, markersize=6, zorder=5)
        canvas.note(x, line_y + 1.6, f"{value:g}", ha="center", color=TEAL, bold=True)
    canvas.arrow((ref.right + 0.3, line_y), (x0 - 1.2, line_y), linewidth=1.2)

    canvas.call(31.4, 0.2, "pd.cut(sample, edges)", ha="right")
    out = canvas.table(
        23.4,
        2.0,
        [
            Column(
                "sample",
                [label(v) if pd.isna(v) else f"{v:g}" for v in sample],
                2.6,
                color=TEAL,
                header_color=TEAL,
            ),
            group_column("groups", groups, 5.4, styles),
        ],
    )
    canvas.arrow((x1 + 1.2, line_y), (out.left - 0.3, line_y), linewidth=1.2)
    return canvas.fig


def _best_ks_cut(values, target, low, high, min_count):
    """The cut of the bin (low, high] with the largest KS, tried value by value."""
    inside = (values > low) & (values <= high)
    values, target = values[inside], target[inside]
    events, non_events = int(target.sum()), int((1 - target).sum())
    best = None
    for cut in np.unique(values)[:-1]:
        left = values <= cut
        if left.sum() < min_count or (~left).sum() < min_count:
            continue
        e, n = int(target[left].sum()), int((1 - target[left]).sum())
        # e n KS: the pairs of an event and a non-event that the cut ranks apart; ties go to
        # the first cut.
        pairs = abs(e * (non_events - n) - (events - e) * n)
        if best is None or pairs > best["pairs"]:
            best = {"cut": cut, "e": e, "n": n, "pairs": pairs}
    best["ks"] = abs(best["e"] / events - best["n"] / non_events)
    return {"low": low, "high": high, "event": events, "non_event": non_events} | best


def figure_ks_second_cut():
    """The second step of Best KS: each bin of the first cut with its best cut and its pairs."""
    reference = load_bank().iloc[:3000]
    balance, target = reference["balance"], reference["target"]
    min_bin_share = 0.05
    first = ks_edges(balance, target, max_bins=2)
    values, labels = balance.to_numpy(dtype=float), target.to_numpy()
    bins = [
        _best_ks_cut(values, labels, low, high, min_bin_share * len(values))
        for low, high in zip(first[:-1], first[1:], strict=True)
    ]
    chosen = max(range(len(bins)), key=lambda i: bins[i]["pairs"])
    second = ks_edges(balance, target, max_bins=3)
    third = ks_edges(balance, target, max_bins=4)
    if list(second) != sorted([*first, bins[chosen]["cut"]]) or list(third) != sorted(
        [*first, *(b["cut"] for b in bins)]
    ):
        raise AssertionError("the drawn cuts are not those of ks_edges")

    def interval(low, high):
        return str(pd.Interval(float(low), float(high), closed="right"))

    parents = [interval(b["low"], b["high"]) for b in bins]
    halves = [
        name for b in bins for name in (interval(b["low"], b["cut"]), interval(b["cut"], b["high"]))
    ]
    styles = dict(zip(parents, ramp(len(parents)), strict=True))
    styles |= dict(zip(halves, ramp(len(halves)), strict=True))

    canvas = Canvas(height=12.4)
    left, right = canvas.call(0.6, 0.15, "ks_edges(balance, target, max_bins=2)")
    canvas.note(right + 0.4, 0.65, f"-> array({np.array2string(first, separator=', ')})", size=9)
    width, top, bottom = 15.0, 1.9, 10.0
    for i, b in enumerate(bins):
        x = 0.6 + i * (width + 0.8)
        picked = i == chosen
        canvas.box(
            x,
            top,
            width,
            bottom - top,
            fill=CARD_FILL,
            edge=AMBER if picked else CARD_EDGE,
            radius=0.45,
            linewidth=1.6 if picked else 0.8,
            z=0,
        )
        tx = x + 0.9
        counts = canvas.table(
            tx,
            top + 0.35,
            [
                group_column("bin", [parents[i]], 5.4, styles),
                Column("event", [str(b["event"])], 2.2),
                Column("non_event", [str(b["non_event"])], 3.0),
            ],
        )
        split = canvas.table(
            tx,
            counts.bottom + 0.9,
            [
                group_column("bin", halves[2 * i : 2 * i + 2], 5.4, styles),
                Column("event", [str(b["e"]), str(b["event"] - b["e"])], 2.2),
                Column("non_event", [str(b["n"]), str(b["non_event"] - b["n"])], 3.0),
            ],
            header=False,
        )
        centre = counts.center("bin")
        canvas.arrow((centre, counts.bottom + 0.08), (centre, split.top - 0.08), linewidth=1.2)
        boundary = (split.rows[0] + split.rows[1]) / 2
        canvas.line(
            split.left - 0.3, boundary, split.right + 0.3, boundary, color=AMBER, linewidth=1.6
        )
        canvas.note(
            split.right + 0.5, boundary, f"{b['cut']:.1f}", color=RESULT_TEXT, bold=True, size=9.5
        )
        canvas.note(
            tx,
            split.bottom + 0.7,
            f"ks = |{b['e']} / {b['event']} - {b['n']} / {b['non_event']}| = {b['ks']:.4f}",
        )
        pairs = f"event * non_event * ks = {b['pairs']}"
        canvas.box(
            tx,
            split.bottom + 1.2,
            text_width(pairs, 9.5) + 0.8,
            CELL,
            fill=RESULT_FILL if picked else CELL_FILL,
            edge=RESULT_EDGE if picked else CELL_EDGE,
        )
        canvas.text(
            tx + 0.4,
            split.bottom + 1.2 + CELL / 2,
            pairs,
            color=RESULT_TEXT if picked else INK,
            bold=picked,
            ha="left",
        )

    centre = 0.6 + chosen * (width + 0.8) + width / 2
    left, right = canvas.call(0.6, bottom + 1.1, "ks_edges(balance, target, max_bins=3)")
    canvas.arrow((min(centre, right - 1.0), bottom + 0.1), (min(centre, right - 1.0), bottom + 1.0))
    canvas.note(
        right + 0.4,
        bottom + 1.6,
        f"-> array({np.array2string(second, separator=', ')})",
        size=9,
        color=RESULT_TEXT,
        bold=True,
    )
    return canvas.fig


# WOE and IV


def figure_woe_table():
    """The worked example of the WOE guide: counts, shares, WOE and IV of each group."""
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    target = pd.Series([1, 1, 0, 1, 0, 0])
    table = woe_table(groups, target)
    styles = group_styles(groups)

    canvas = Canvas(height=10.7)
    inputs = canvas.table(
        0.6,
        1.4,
        [group_column("groups", groups, 2.6, styles), target_column(target, 2.6)],
    )
    x = 8.4
    canvas.call(x, 0.15, "woe_table(groups, target)")
    columns = [
        group_column("bin", table.index, 1.8, styles),
        Column("non_event", [str(v) for v in table["non_event"]], 3.1),
        Column("event", [str(v) for v in table["event"]], 2.3),
        Column("non_event_share", [number(v, 4) for v in table["non_event_share"]], 5.0),
        Column("event_share", [number(v, 4) for v in table["event_share"]], 3.9),
        result("woe", [number(v, 4) for v in table["woe"]], 2.9),
        result("iv", [number(v, 4) for v in table["iv"]], 2.6),
    ]
    out = canvas.table(x, 1.4, columns)
    canvas.arrow((inputs.right + 0.4, inputs.rows[2]), (out.left - 0.4, inputs.rows[2]))

    total = out.bottom + 0.45
    canvas.note(out.center("bin"), total, "sum", ha="center")
    sums = {
        "non_event": str(table["non_event"].sum()),
        "event": str(table["event"].sum()),
        "non_event_share": "1",
        "event_share": "1",
        "iv": number(table["iv"].sum(), 4),
    }
    for name, value in sums.items():
        canvas.note(out.center(name), total, value, ha="center", bold=name == "iv")
    canvas.notes(
        x,
        total + 1.4,
        [
            "non_event_share = non_event / non_event.sum()",
            "event_share     = event / event.sum()",
            "woe             = ln(event_share / non_event_share)",
            "iv              = (event_share - non_event_share) * woe",
        ],
    )
    return canvas.fig


def figure_woe_transform():
    """New rows are grouped with the development edges and take the WOE of their group."""
    data = load_bank()
    reference, current = data.iloc[:3000], data.iloc[3000:]
    edges = bin_edges(reference["age"], bins=5)
    age_table = woe_table(pd.cut(reference["age"], edges), reference["target"])
    rows = current.head(4)
    groups = pd.cut(rows["age"], edges)
    encoded = woe_transform(groups, age_table)
    styles = group_styles(age_table.index, palette="ramp")

    canvas = Canvas(height=9.4)
    canvas.call(0.6, 0.2, "woe_transform(groups, age_table)")
    sample = canvas.table(
        0.6,
        1.6,
        [
            Column("index", [str(i) for i in rows.index], 2.0, color=MUTED),
            Column("age", [str(v) for v in rows["age"]], 1.8),
            group_column("groups", groups, 4.4, styles),
        ],
    )
    lookup = canvas.table(
        13.2,
        1.6,
        [
            group_column("bin", age_table.index, 4.4, styles),
            Column("woe", [number(v) for v in age_table["woe"]], 2.6),
        ],
    )
    out = canvas.table(
        25.0,
        1.6,
        [
            Column("index", [str(i) for i in rows.index], 2.0, color=MUTED),
            result("woe", [number(v) for v in encoded], 2.8),
        ],
    )
    position = {value: i for i, value in enumerate(age_table.index)}
    for row, group in zip(sample.rows, groups, strict=True):
        source = lookup.rows[position[group]]
        color = _blend(styles[group][0], INK, 0.45)
        canvas.arrow(
            (sample.right + 0.3, row), (lookup.left - 0.3, source), color=color, linewidth=1.0
        )
        canvas.arrow(
            (lookup.right + 0.3, source), (out.left - 0.3, row), color=color, linewidth=1.0
        )
    canvas.note(
        lookup.left,
        lookup.bottom + 0.6,
        "age_table = woe_table(pd.cut(reference['age'], edges), ...)",
        ha="left",
    )
    return canvas.fig


# PSI


def figure_psi_table():
    """The worked example of the PSI guide, with the smoothed empty cells marked."""
    reference = pd.Series(["A", "A", None])
    current = pd.Series(["B", None, None])
    table = psi_table(reference, current)
    styles = group_styles(table.index)

    canvas = Canvas(height=11.0)
    inputs = canvas.table(
        0.6,
        1.4,
        [
            group_column("reference", reference, 3.2, styles, header_color=BLUE),
            group_column("current", current, 3.0, styles, header_color=TEAL),
        ],
    )
    x = 10.0
    canvas.call(x, 0.15, "psi_table(reference, current)")

    def counts(name, color):
        values = table[name]
        return Column(
            name,
            [str(v) for v in values],
            2.9 if name == "current" else 3.2,
            edge=[AMBER if v == 0 else None for v in values],
            color=[RESULT_TEXT if v == 0 else INK for v in values],
            bold=[bool(v == 0) for v in values],
            header_color=color,
        )

    out = canvas.table(
        x,
        1.4,
        [
            group_column("bin", table.index, 1.8, styles),
            counts("reference", BLUE),
            counts("current", TEAL),
            Column(
                "reference_share",
                [number(v, 4) for v in table["reference_share"]],
                4.7,
                header_color=BLUE,
            ),
            Column(
                "current_share",
                [number(v, 4) for v in table["current_share"]],
                4.2,
                header_color=TEAL,
            ),
            result("psi", [number(v, 4) for v in table["psi"]], 2.6),
        ],
    )
    canvas.arrow((inputs.right + 0.4, inputs.rows[1]), (out.left - 0.4, inputs.rows[1]))
    canvas.note(
        out.center("psi"), out.bottom + 0.45, number(table["psi"].sum(), 4), ha="center", bold=True
    )
    canvas.note(out.center("bin"), out.bottom + 0.45, "sum", ha="center")
    canvas.notes(
        0.6,
        out.bottom + 1.6,
        [
            "reference_share = r / r.sum(),  r = reference.where(reference > 0, smoothing)",
            "current_share   = c / c.sum(),  c = current.where(current > 0, smoothing)",
            "psi = (current_share - reference_share) * ln(current_share / reference_share)",
            "smoothing = 0.5",
        ],
    )
    return canvas.fig


# Scorecard


def _scorecard_example():
    data = load_bank()
    development, monitoring = data.iloc[:3000], data.iloc[3000:]
    age_edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])

    def to_groups(frame):
        groups = frame[["poutcome", "contact", "housing"]].copy()
        groups["age"] = pd.cut(frame["age"], age_edges)
        return groups

    dev_groups = to_groups(development)
    tables = {column: woe_table(dev_groups[column], development["target"]) for column in dev_groups}
    coefficients = {"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714}
    return monitoring, to_groups, tables, coefficients, -2.028


def figure_scorecard_points():
    """One applicant: raw values, groups, WOE, points and the score."""
    monitoring, to_groups, tables, coefficients, intercept = _scorecard_example()
    scale = dict(pdo=20, base_score=600, base_odds=50)
    applicant = monitoring.loc[[3001]]
    groups = to_groups(applicant)
    woe = pd.DataFrame({column: woe_transform(groups[column], tables[column]) for column in groups})
    points = woe_to_points(woe, coefficients, intercept, **scale)
    factor, offset = score_scaling(**scale)
    variables = list(groups)

    canvas = Canvas(height=11.6)
    left, width = 8.6, 4.5
    canvas.call(left, 0.15, "monitoring.loc[3001]")
    stages = [
        ("value", [str(applicant[v].iloc[0]) for v in variables], CELL_FILL, INK),
        ("to_groups", [str(groups[v].iloc[0]) for v in variables], CATEGORY[0][0], CATEGORY[0][1]),
        ("woe_transform", [number(woe[v].iloc[0], 4) for v in variables], CELL_FILL, INK),
        ("coefficients", [f"{coefficients[v]:.3f}" for v in variables], CELL_FILL, MUTED),
        ("woe_to_points", [number(points[v].iloc[0], 1) for v in variables], RESULT_FILL, INK),
    ]
    header = 2.1
    for j, variable in enumerate(variables):
        x = left + j * (width + GAP)
        canvas.text(x + width / 2, header, variable, size=9, bold=True)
    canvas.line(left, header + 0.55, left + 4 * width + 3 * GAP, header + 0.55)
    for i, (name, cells, fill, color) in enumerate(stages):
        y = header + 1.25 + i * 1.25
        is_result = name == "woe_to_points"
        canvas.text(
            left - 0.4, y, name, size=9, ha="right", color=RESULT_TEXT if is_result else MUTED
        )
        for j, cell in enumerate(cells):
            x = left + j * (width + GAP)
            canvas.box(
                x,
                y - CELL / 2,
                width,
                CELL,
                fill=fill,
                edge=RESULT_EDGE if is_result else (CELL_EDGE if fill == CELL_FILL else fill),
            )
            canvas.text(x + width / 2, y, cell, color=color)
    last = header + 1.25 + 4 * 1.25
    score = points.sum(axis=1).iloc[0]
    end = left + 4 * width + 3 * GAP
    canvas.arrow((end + 0.25, last), (end + 1.0, last), linewidth=1.2)
    canvas.box(end + 1.15, last - CELL / 2, 3.0, CELL, fill=RESULT_FILL, edge=AMBER, linewidth=1.2)
    canvas.text(end + 2.65, last, number(score, 1), bold=True, color=RESULT_TEXT)
    canvas.note(end + 2.65, last - 0.95, "score", ha="center", bold=True, color=RESULT_TEXT)
    base = (offset - factor * intercept) / len(variables)
    canvas.notes(
        left,
        last + 1.35,
        [
            f"points = (offset - factor * intercept) / {len(variables)} - factor * coef * woe",
            f"       = {base:.2f} - {factor:.2f} * coef * woe",
        ],
    )
    return canvas.fig


# Expert scorecards


def _expert_matrices():
    """The comparison matrices of the two analysts in the expert guide."""
    factors = ["leverage", "coverage", "liquidity", "profitability", "industry"]

    def comparison(rows):
        return pd.DataFrame(rows, index=factors, columns=factors)

    first = comparison(
        [
            [1, 2, 4, 2, 5],
            [1 / 2, 1, 2, 1, 3],
            [1 / 4, 1 / 2, 1, 1 / 2, 2],
            [1 / 2, 1, 2, 1, 3],
            [1 / 5, 1 / 3, 1 / 2, 1 / 3, 1],
        ]
    )
    second = comparison(
        [
            [1, 2, 4, 1 / 3, 5],
            [1 / 2, 1, 2, 1, 3],
            [1 / 4, 1 / 2, 1, 4, 2],
            [3, 1, 1 / 4, 1, 3],
            [1 / 5, 1 / 3, 1 / 2, 1 / 3, 1],
        ]
    )
    return first, second


def _principal_eigenvector(matrix: pd.DataFrame) -> np.ndarray:
    """Power iteration, apart from the eigensolver that ahp_weights uses."""
    values = matrix.to_numpy(dtype=float)
    vector = np.full(len(values), 1 / len(values))
    for _ in range(500):
        vector = values @ vector
        vector /= vector.sum()
    return vector


def _comparison_grid(canvas, x, top, matrix, cells, *, fill, color, edge, bold, width=2.0):
    """A square matrix with the factors on its rows and, slanted, on its columns."""
    labels = list(matrix.index)
    rows = [top + HEADER + (i + 0.5) * ROW for i in range(len(labels))]
    for j, name in enumerate(labels):
        center = x + j * (width + GAP) + width / 2
        canvas.ax.text(
            center - 0.2,
            top + HEADER - 0.25,
            name,
            fontsize=8.5,
            color=INK,
            fontweight="bold",
            rotation=30,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
            family=MONO,
            zorder=4,
        )
    for i, name in enumerate(labels):
        canvas.text(x - 0.4, rows[i], name, size=9, ha="right", bold=True)
        for j in range(len(labels)):
            left = x + j * (width + GAP)
            canvas.box(
                left,
                rows[i] - CELL / 2,
                width,
                CELL,
                fill=fill[i][j],
                edge=edge[i][j] or (CELL_EDGE if fill[i][j] == CELL_FILL else fill[i][j]),
                linewidth=1.3 if edge[i][j] else 0.8,
            )
            canvas.text(left + width / 2, rows[i], cells[i][j], color=color[i][j], bold=bold[i][j])
    right = x + len(labels) * (width + GAP) - GAP
    canvas.line(x, top + HEADER - 0.06, right, top + HEADER - 0.06)
    return rows, right


def figure_ahp_weights():
    """The weights of the five factors are the principal eigenvector: A w = lambda_max w."""
    matrix, _ = _expert_matrices()
    weights = ahp_weights(matrix)
    statistics = ahp_consistency(matrix)
    product = matrix.to_numpy() @ weights.to_numpy()
    ratio = product / weights.to_numpy()
    if not np.allclose(_principal_eigenvector(matrix), weights, rtol=1e-9, atol=0):
        raise AssertionError("the drawn weights do not match ahp_weights")
    if not np.allclose(ratio, statistics["lambda_max"], rtol=1e-9, atol=0):
        raise AssertionError("the drawn lambda_max does not match ahp_consistency")
    n = len(matrix)
    consistency = (statistics["lambda_max"] - n) / (n - 1)
    if not np.isclose(consistency, statistics["ci"], rtol=1e-12, atol=1e-15) or not np.isclose(
        consistency / statistics["ri"], statistics["cr"], rtol=1e-12, atol=1e-15
    ):
        raise AssertionError("the drawn ci and cr do not match ahp_consistency")

    size = range(n)
    cells = [[str(Fraction(matrix.iat[i, j]).limit_denominator(9)) for j in size] for i in size]
    judged = [[j > i for j in size] for i in size]
    canvas = Canvas(height=11.6)
    x, top = 5.0, 2.2
    canvas.call(x, 0.15, "ahp_weights(matrix)")
    rows, right = _comparison_grid(
        canvas,
        x,
        top,
        matrix,
        cells,
        fill=[[CELL_FILL] * n for _ in size],
        color=[[INK if judged[i][j] else FAINT if i == j else MUTED for j in size] for i in size],
        edge=[[None] * n for _ in size],
        bold=judged,
        width=1.8,
    )
    out = canvas.table(
        right + 0.65,
        top,
        [
            result("weight", [number(w, 4) for w in weights], 2.4),
            Column("(A w)_i", [number(v, 4) for v in product], 2.4),
            Column("(A w)_i / w_i", [number(v, 4) for v in ratio], 4.0),
        ],
    )
    canvas.call(31.6, 0.15, "ahp_consistency(matrix)", ha="right")
    # The Series as the doctest prints it: the index name heads the labels, and the values
    # have no header because the Series has no name.
    canvas.table(
        out.right + 0.6,
        top,
        [
            Column(statistics.index.name, list(statistics.index), 3.6, color=MUTED),
            result(statistics.name or "", [number(v, 4) for v in statistics], 2.3),
        ],
    )
    lam, ri = statistics["lambda_max"], statistics["ri"]
    canvas.notes(
        x,
        rows[-1] + 1.15,
        [
            "a_ji = 1 / a_ij",
            f"ci = (lambda_max - n) / (n - 1) = ({lam:.4f} - {n}) / {n - 1} = {consistency:.4f}",
            f"cr = ci / ri = {consistency:.4f} / {ri:.2f} = {consistency / ri:.4f}",
        ],
    )
    return canvas.fig


def _box_exit(centre, towards, half_width, half_height):
    """Where the line from a box's centre towards a point leaves the box."""
    dx, dy = towards[0] - centre[0], towards[1] - centre[1]
    scale = min(
        half_width / abs(dx) if dx else np.inf,
        half_height / abs(dy) if dy else np.inf,
    )
    return centre[0] + dx * scale, centre[1] + dy * scale


def figure_ahp_ratios():
    """Each judgment of the second analyst against the ratio of the weights it implies."""
    _, second = _expert_matrices()
    weights = ahp_weights(second)
    if not np.allclose(weights, _principal_eigenvector(second), rtol=1e-9, atol=0):
        raise AssertionError("the drawn weights do not match ahp_weights")
    values = (second / np.outer(weights, 1 / weights)).to_numpy()
    n = len(second)
    size = range(n)
    above_one = [(values[i, j], i, j) for i in size for j in size if values[i, j] > 1]
    circle = [(i, j) for _, i, j in sorted(above_one, reverse=True)[:3]]
    marked = set(circle) | {(j, i) for i, j in circle}
    fill = [
        [
            CELL_FILL if i == j else _blend(CELL_FILL, RESULT_FILL, abs(np.log(values[i, j])) / 0.9)
            for j in size
        ]
        for i in size
    ]
    canvas = Canvas(height=10.2)
    x, top = 5.0, 2.6
    canvas.call(x, 0.15, "second / np.outer(w, 1 / w)")
    rows, right = _comparison_grid(
        canvas,
        x,
        top,
        second,
        [[number(values[i, j], 2) for j in size] for i in size],
        fill=fill,
        color=[
            [RESULT_TEXT if (i, j) in circle else FAINT if i == j else INK for j in size]
            for i in size
        ],
        edge=[[AMBER if (i, j) in marked else None for j in size] for i in size],
        bold=[[(i, j) in circle for j in size] for i in size],
        width=2.2,
    )
    canvas.call(31.6, 0.15, "w = ahp_weights(second)", ha="right")
    # The three judgments that run in a circle, each arrow from the factor judged more important.
    labels = list(second.index)
    first = circle[0][0]
    order = [first, dict(circle)[first], dict(circle)[dict(circle)[first]]]
    centre_x, centre_y = right + 7.4, (rows[0] + rows[-1]) / 2
    places = [
        (centre_x, centre_y - 2.3),
        (centre_x + 3.7, centre_y + 1.9),
        (centre_x - 3.7, centre_y + 1.9),
    ]
    nodes = {}
    for k, (px, py) in zip(order, places, strict=True):
        half = (text_width(labels[k], 9) + 0.8) / 2
        canvas.box(px - half, py - 0.45, 2 * half, 0.9, fill=CHIP, radius=0.25, z=2)
        canvas.text(px, py, labels[k], size=9, color=CHIP_TEXT, bold=True)
        nodes[k] = (px, py, half)
    for i, j in circle:
        (x0, y0, h0), (x1, y1, h1) = nodes[i], nodes[j]
        start = _box_exit((x0, y0), (x1, y1), h0 + 0.2, 0.65)
        end = _box_exit((x1, y1), (x0, y0), h1 + 0.2, 0.65)
        canvas.arrow(start, end, color=AMBER, linewidth=1.4)
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        out_x, out_y = mid_x - centre_x, mid_y - centre_y
        length = np.hypot(out_x, out_y)
        judgment = Fraction(second.iat[i, j]).limit_denominator(9)
        canvas.text(
            mid_x + 1.3 * out_x / length,
            mid_y + 0.6 * out_y / length,
            f"a_ij = {judgment}",
            size=8.8,
            color=RESULT_TEXT,
            bold=True,
        )
    canvas.notes(x, rows[-1] + 1.15, ["a_ij / (w_i / w_j)"])
    return canvas.fig


# Metrics


def figure_roc_table():
    """The worked example of the metrics guide: the ROC table, the AUC and the KS."""
    score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    target = [0, 0, 1, 0, 1, 1]
    table = roc_table(score, target)
    fpr, tpr = table["false_positive_rate"], table["true_positive_rate"]
    gap = (tpr - fpr).abs()
    widest = np.isclose(gap, gap.max())

    canvas = Canvas(height=16.4)
    inputs = canvas.table(
        0.6, 1.4, [Column("score", [f"{v:g}" for v in score], 2.2), target_column(target, 2.4)]
    )
    x = 7.6
    canvas.call(x, 0.15, "roc_table(score, target)")
    edge = [AMBER if w else None for w in widest]
    out = canvas.table(
        x,
        1.4,
        [
            Column("threshold", [f"{v:g}" for v in table.index], 2.9, color=MUTED),
            Column("cum_non_event", [str(v) for v in table["cum_non_event"]], 4.1),
            Column("cum_event", [str(v) for v in table["cum_event"]], 3.0),
            result("false_positive_rate", [number(v) for v in fpr], 6.2, edge=edge),
            result("true_positive_rate", [number(v) for v in tpr], 5.9, edge=edge),
        ],
    )
    canvas.arrow((inputs.right + 0.4, inputs.rows[2]), (out.left - 0.4, inputs.rows[2]))

    ax = canvas.axes(3.0, 8.9, 6.0, 6.0)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=FAINT, linestyle="--", linewidth=1)
    points = list(zip(fpr, tpr, strict=True))
    areas = []
    for (f0, t0), (f1, t1) in zip(points[:-1], points[1:], strict=True):
        if f1 > f0:
            areas.append(Fraction((f1 - f0) * (t0 + t1) / 2).limit_denominator(100))
            tint = BLUE_FILL if len(areas) % 2 else "#cad7f8"
            ax.fill_between([f0, f1], [t0, t1], color=tint, zorder=0, linewidth=0)
            ax.text(
                (f0 + f1) / 2,
                (t0 + t1) / 4,
                str(areas[-1]),
                ha="center",
                va="center",
                fontsize=8.5,
                family=MONO,
                color=CHIP_TEXT,
            )
    ax.plot(fpr, tpr, color=BLUE, marker="o", markersize=4, linewidth=1.8)
    for f, t, w in zip(fpr, tpr, widest, strict=True):
        if w:
            ax.plot([f, f], [f, t], color=AMBER, linewidth=2.4, solid_capstyle="butt")
    thirds = FixedLocator([0, 1 / 3, 2 / 3, 1])
    names = {0: "0", 1: "1/3", 2: "2/3", 3: "1"}
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(thirds)
        axis.set_major_formatter(lambda v, _: names[round(v * 3)])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false_positive_rate")
    ax.set_ylabel("true_positive_rate")

    auc = roc_auc(score, target)
    total = " + ".join(str(a) for a in areas)
    canvas.notes(
        11.4,
        10.0,
        [
            f"roc_auc = {total} = {sum(areas)} = {auc:.4f}",
            f"gini    = 2 * roc_auc - 1 = {2 * auc - 1:.4f}",
        ],
        step=1.1,
    )
    canvas.line(11.4, 13.0, 12.3, 13.0, color=AMBER, linewidth=2.4)
    canvas.note(12.6, 13.0, "|true_positive_rate - false_positive_rate|")
    canvas.note(
        11.4, 14.1, f"ks      = max |true_positive_rate - false_positive_rate| = {gap.max():.4f}"
    )
    return canvas.fig


def figure_generalized_auc():
    """The table of generalized_auc, with the cells that agree and disagree with one cell."""
    estimated = [1, 1, 1, 2, 2, 2, 3, 3]
    realized = [1, 1, 2, 1, 2, 3, 2, 3]
    result = generalized_auc(estimated, realized)
    table = pd.crosstab(
        pd.Series(estimated, name="estimated"), pd.Series(realized, name="realized")
    )
    pairs = [
        (realized[j] - realized[i]) * (estimated[j] - estimated[i])
        for i in range(len(estimated))
        for j in range(i + 1, len(estimated))
        if estimated[i] != estimated[j]
    ]
    same, other = sum(p > 0 for p in pairs), sum(p < 0 for p in pairs)
    tied = len(pairs) - same - other
    somers_d = Fraction(same - other, len(pairs))
    if not np.isclose(result.loc["somers_d", "estimate"], float(somers_d), rtol=1e-12, atol=0):
        raise AssertionError("the drawn pairs do not match generalized_auc")
    # The cell of estimated segment 2 and realized segment 1, and the cells that agree with it in
    # order (A) and that disagree (D).
    cell = (1, 0)
    agree = {(2, 1), (2, 2)}
    disagree = {(0, 1), (0, 2)}
    a_count = sum(int(table.iat[i, j]) for i, j in agree)
    d_count = sum(int(table.iat[i, j]) for i, j in disagree)

    canvas = Canvas(height=8.4)
    left, top, width = 7.2, 1.9, 2.4
    canvas.call(0.6, 0.15, "generalized_auc(estimated, realized)")
    canvas.text(
        left - 0.4,
        top + HEADER / 2 - 0.05,
        "estimated \\ realized",
        size=8.5,
        ha="right",
        color=MUTED,
    )
    for j, column in enumerate(table.columns):
        x = left + j * (width + GAP) + width / 2
        canvas.text(x, top + HEADER / 2 - 0.05, str(column), size=9, bold=True)
    for i, row in enumerate(table.index):
        y = top + HEADER + (i + 0.5) * ROW
        canvas.text(left - 0.4, y, str(row), size=9, ha="right", bold=True)
        for j in range(table.shape[1]):
            x = left + j * (width + GAP)
            if (i, j) == cell:
                fill, edge, color = RESULT_FILL, AMBER, RESULT_TEXT
            elif (i, j) in agree:
                fill, edge, color = BLUE_FILL, BLUE, BLUE
            elif (i, j) in disagree:
                fill, edge, color = EVENT_FILL, EVENT, EVENT
            else:
                fill, edge, color = CELL_FILL, CELL_EDGE, INK
            canvas.box(x, y - CELL / 2, width, CELL, fill=fill, edge=edge, linewidth=1.2)
            canvas.text(
                x + width / 2, y, str(int(table.iat[i, j])), color=color, bold=fill != CELL_FILL
            )
    right = left + 3 * width + 2 * GAP
    canvas.line(left, top + HEADER - 0.06, right, top + HEADER - 0.06)
    canvas.note(left, top + HEADER + 3 * ROW + 0.6, f"A_21 = {a_count}", color=BLUE, bold=True)
    canvas.note(
        left + 3.4, top + HEADER + 3 * ROW + 0.6, f"D_21 = {d_count}", color=EVENT, bold=True
    )
    x = right + 1.6
    canvas.notes(
        x,
        2.4,
        [
            f"pairs in different estimated  {len(pairs)}",
            f"  ordered the same way        {same}",
            f"  ordered the other way       {other}",
            f"  tied in realized            {tied}",
        ],
        step=1.0,
        size=8.6,
    )
    canvas.notes(
        x,
        6.6,
        [
            f"somers_d = ({same} - {other}) / {len(pairs)} = {number(float(somers_d), 4)}",
            f"gauc     = (somers_d + 1) / 2 = {number(result.loc['gauc', 'estimate'], 4)}",
        ],
        step=1.0,
        size=8.6,
        color=RESULT_TEXT,
        bold=True,
    )
    return canvas.fig


# Calibration


def _pool_adjacent_violators(target: Sequence[int]):
    """The blocks of PAV after each observation that triggered a merge."""
    blocks: list[list[int]] = []  # [first, last, events]
    states = []
    for i, value in enumerate(target):
        blocks.append([i, i, value])
        merged = False
        while len(blocks) > 1:
            (a0, a1, ae), (b0, b1, be) = blocks[-2], blocks[-1]
            if ae / (a1 - a0 + 1) <= be / (b1 - b0 + 1):
                break
            blocks[-2:] = [[a0, b1, ae + be]]
            merged = True
        if merged:
            states.append((i, [tuple(b) for b in blocks]))
    states.append((len(target) - 1, [tuple(b) for b in blocks]))
    return states


def figure_isotonic_pav():
    """Pool-adjacent-violators on ten scores, from the outcomes to the fitted bands."""
    scores = list(range(1, 11))
    outcomes = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    edges, probability = isotonic_fit(scores, outcomes)
    states = _pool_adjacent_violators(outcomes)
    final = states[-1][1]
    if not np.allclose([e / (b - a + 1) for a, b, e in final], probability):
        raise AssertionError("the drawn PAV does not match isotonic_fit")

    n = len(scores)
    left, width = 5.2, 2.36
    step = width + GAP

    def x_of(i):
        return left + i * step

    canvas = Canvas(height=12.6)
    canvas.call(left, 0.15, "isotonic_fit(scores, outcomes)")
    rows = [2.0, 3.0] + [4.35 + k * 1.1 for k in range(len(states))]
    canvas.text(left - 0.4, rows[0], "scores", size=9, ha="right", bold=True)
    canvas.text(left - 0.4, rows[1], "outcomes", size=9, ha="right", bold=True)
    for i, (s, t) in enumerate(zip(scores, outcomes, strict=True)):
        canvas.text(x_of(i) + width / 2, rows[0], str(s), bold=True)
        canvas.box(
            x_of(i),
            rows[1] - CELL / 2,
            width,
            CELL,
            fill=EVENT_FILL if t else CELL_FILL,
            edge=None if t else CELL_EDGE,
        )
        canvas.text(x_of(i) + width / 2, rows[1], str(t), color=EVENT if t else INK, bold=bool(t))
    canvas.line(left, rows[0] + 0.5, x_of(n - 1) + width, rows[0] + 0.5)

    previous: set[tuple] = set()
    for k, (seen, blocks) in enumerate(states):
        y = rows[2 + k]
        last = k == len(states) - 1
        for a, b, events in blocks:
            rate = events / (b - a + 1)
            new = (a, b, events) not in previous and b > a
            fill = RESULT_FILL if last else ("#fff7ed" if new else CELL_FILL)
            edge = AMBER if new and not last else RESULT_EDGE if last else CELL_EDGE
            canvas.box(
                x_of(a),
                y - CELL / 2,
                x_of(b) + width - x_of(a),
                CELL,
                fill=fill,
                edge=edge,
                linewidth=1.2 if new else 0.8,
            )
            canvas.text(
                (x_of(a) + x_of(b) + width) / 2,
                y,
                number(rate),
                color=RESULT_TEXT if last else INK,
                bold=last,
            )
        for i in range(seen + 1, n):
            canvas.text(x_of(i) + width / 2, y, str(outcomes[i]), color=FAINT)
        previous = set(blocks)
        if not last:
            canvas.text(left - 0.4, y, f"<= {scores[seen]}", size=8.8, ha="right", color=MUTED)
    canvas.text(
        left - 0.4, rows[-1], "probability", size=9, ha="right", bold=True, color=RESULT_TEXT
    )

    y = rows[-1] + 1.2
    canvas.line(left, y, x_of(n - 1) + width, y, color=MUTED, linewidth=1.0)
    for edge in edges[1:-1]:
        i = scores.index(int(edge))
        x = x_of(i) + width + GAP / 2
        canvas.line(x, y - 0.35, x, y + 0.35, color=AMBER, linewidth=1.6)
        canvas.note(x, y + 0.85, f"{edge:g}", ha="center", color=RESULT_TEXT, bold=True)
    canvas.note(left - 0.4, y, "edges", ha="right", color=INK, bold=True, size=9)
    canvas.note(left, y + 0.85, "-inf", ha="left")
    canvas.note(x_of(n - 1) + width, y + 0.85, "inf", ha="right")
    return canvas.fig


# Reject inference


def _reject_example():
    applications = make_applications(rng=0)
    population = applications[~applications["policy_decline"]]
    accepts = population[population["approved"]]
    rejects = population[~population["approved"]]
    features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment"]
    edges = {
        column: bin_edges(population[column], bins=5)
        for column in ["age", "income", "debt_to_income"]
    }
    edges["inquiries"] = [-np.inf, 0, 1, 2, np.inf]
    spec = fit_spec(accepts[features], accepts["target"], bins=edges)
    coef = logistic_regression(spec.woe_transform(accepts), accepts["target"])["coef"]

    def predict(frame):
        log_odds = coef["intercept"] + spec.woe_transform(frame) @ coef.drop("intercept")
        return 1 / (1 + np.exp(-log_odds))

    return accepts, predict(accepts), predict(rejects)


def figure_fuzzy_augmentation():
    """Each reject becomes a bad row weighted with its PD and a good row with the rest."""
    _, _, reject_pd = _reject_example()
    shown = reject_pd.head(3)
    inferred = fuzzy_augmentation(shown)
    styles = group_styles(shown.index)

    canvas = Canvas(height=11.6)
    inputs = canvas.table(
        3.0,
        2.6,
        [
            group_column("index", shown.index, 2.2, styles),
            Column("reject_pd", [number(v) for v in shown], 3.4),
        ],
    )
    x = 16.4
    canvas.call(x, 0.2, "fuzzy_augmentation(reject_pd)")
    out = canvas.table(
        x,
        1.6,
        [
            group_column("index", inferred.index, 2.2, styles),
            target_column(inferred["target"], 2.4),
            result("weight", [number(v) for v in inferred["weight"]], 2.8),
        ],
    )
    for i, row in enumerate(inputs.rows):
        color = _blend(styles[shown.index[i]][0], INK, 0.45)
        for j in (2 * i, 2 * i + 1):
            canvas.arrow(
                (inputs.right + 0.3, row), (out.left - 0.3, out.rows[j]), linewidth=1.0, color=color
            )
    canvas.notes(
        x,
        out.bottom + 0.6,
        [
            "target 1: weight = min(factor * reject_pd, 1)",
            "target 0: weight = 1 - min(factor * reject_pd, 1)",
            "factor = 1",
        ],
    )
    return canvas.fig


def _interval(low: float, high: float) -> str:
    def end(value):
        return "inf" if value == np.inf else "-inf" if value == -np.inf else f"{value:.3f}"

    return f"({end(low)}, {end(high)}]"


def figure_reject_bands():
    """Score bands of accepts and rejects: the rates of parceling and the weights of reweighting."""
    accepts, accept_pd, reject_pd = _reject_example()
    factor = 1.5
    edges = bin_edges(pd.concat([accept_pd, reject_pd]), bins=10)
    accept_band = pd.cut(accept_pd, edges, labels=False)
    reject_band = pd.cut(reject_pd, edges, labels=False)
    bands = pd.DataFrame(
        {
            "accepts": accept_band.value_counts().sort_index(),
            "rejects": reject_band.value_counts().sort_index(),
            "event_rate": accepts["target"].groupby(accept_band).mean(),
        }
    )
    bands["parceling"] = np.minimum(bands["event_rate"] * factor, 1)
    bands["reweighting"] = (bands["accepts"] + bands["rejects"]) / bands["accepts"]

    parcels = parceling(accept_pd, accepts["target"], reject_pd, factor=factor, expected=True)
    bad = parcels.loc[parcels["target"] == 1, "weight"]
    weights = reweighting(accept_pd, reject_pd)
    if not (
        np.allclose(bad.to_numpy(), bands["parceling"].to_numpy()[reject_band.loc[bad.index]])
        and np.allclose(weights.to_numpy(), bands["reweighting"].to_numpy()[accept_band])
    ):
        raise AssertionError("the drawn bands do not match parceling and reweighting")

    canvas = Canvas(height=15.8)
    x = 0.6
    canvas.call(x, 0.15, "bin_edges(pd.concat([accept_pd, reject_pd]), bins=10)")
    names = [_interval(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    styles = dict(zip(names, ramp(len(names)), strict=True))
    out = canvas.table(
        x,
        1.6,
        [
            group_column("bin", names, 5.6, styles),
            Column("accepts", [str(v) for v in bands["accepts"]], 2.8, header_color=BLUE),
            Column("rejects", [str(v) for v in bands["rejects"]], 2.8, header_color=MUTED),
            Column("event_rate", [number(v) for v in bands["event_rate"]], 3.4),
            result("parceling", [number(v) for v in bands["parceling"]], 3.6),
            result("reweighting", [number(v, 2) for v in bands["reweighting"]], 3.9),
        ],
    )
    # The accepts and rejects of each band as one bar.
    left, span = out.right + 0.6, WIDTH - out.right - 1.2
    largest = (bands["accepts"] + bands["rejects"]).max()
    header = 1.6 + HEADER / 2 - 0.05
    canvas.note(left, header, "accepts", size=8.5, color=BLUE, bold=True)
    canvas.note(left + 2.6, header, "rejects", size=8.5, color=MUTED, bold=True)
    for row, (a, r) in zip(out.rows, bands[["accepts", "rejects"]].to_numpy(), strict=True):
        wa, wr = span * a / largest, span * r / largest
        canvas.box(left, row - 0.3, wa, 0.6, fill=BLUE, radius=0.05)
        canvas.box(left + wa, row - 0.3, wr, 0.6, fill=FAINT, radius=0.05)
    canvas.notes(
        x,
        out.bottom + 0.75,
        [
            f"parceling   = min(factor * event_rate, 1),  factor = {factor:g}",
            "reweighting = (accepts + rejects) / accepts",
        ],
    )
    return canvas.fig


# Vintage


def _repayment_example():
    """The schedule and the payments of the four loans of the vintage guide."""
    schedule = pd.DataFrame(
        {
            "loan": np.repeat(["A", "B", "C", "D"], 3),
            "due_date": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"] * 4),
            "amount_due": 100.0,
        }
    )
    payments = pd.DataFrame(
        {
            "loan": ["A", "A", "A", "B", "B", "C", "C", "D", "D"],
            "date": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-02-15",
                    "2024-03-14",
                    "2024-01-15",
                    "2024-03-20",
                    "2024-01-10",
                    "2024-03-15",
                    "2024-01-15",
                    "2024-02-15",
                ]
            ),
            "amount": [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 100.0, 100.0, 99.99],
        }
    )
    return schedule, payments


def _month_end_rows(due_date, amount_due, payment_date, payment_amount, months, statuses):
    """days_past_due, overdue_amount, outstanding and status at each month end, by hand."""
    order = np.argsort(due_date.to_numpy(), kind="stable")
    due_date, amount_due = due_date.to_numpy()[order], amount_due.to_numpy()[order]
    owed = np.cumsum(amount_due)
    rows = []
    for month in months:
        end = month.to_timestamp(how="end").normalize()
        paid = payment_amount[payment_date <= end].sum()
        unpaid = np.flatnonzero(owed - paid > 1e-9)
        oldest = due_date[unpaid[0]] if len(unpaid) else None
        days = max((end - oldest).days, 0) if oldest is not None else 0
        bucket = sum(days >= start for start in (1, 30, 60, 90))
        rows.append(
            {
                "days_past_due": days,
                "overdue_amount": max(amount_due[due_date < end].sum() - paid, 0.0),
                "outstanding": amount_due.sum() - min(paid, amount_due.sum()),
                "status": statuses[bucket],
            }
        )
    return pd.DataFrame(rows)


def figure_delinquency_timeline():
    """Loan D of the vintage guide: instalments, payments and the columns at each month end."""
    schedule, payments = _repayment_example()
    book = delinquency_panel(
        schedule["loan"],
        schedule["due_date"],
        schedule["amount_due"],
        payments["loan"],
        payments["date"],
        payments["amount"],
        end="2024-06",
    )
    loan = "D"
    rows = book[book["loan_id"] == loan].reset_index(drop=True)
    due = schedule[schedule["loan"] == loan]
    paid = payments[payments["loan"] == loan]
    statuses = list(rows["status"].cat.categories)
    expected = _month_end_rows(
        due["due_date"], due["amount_due"], paid["date"], paid["amount"], rows["month"], statuses
    )
    numbers = ["days_past_due", "overdue_amount", "outstanding"]
    if not (
        np.allclose(rows[numbers].to_numpy(dtype=float), expected[numbers].to_numpy(dtype=float))
        and list(rows["status"]) == list(expected["status"])
    ):
        raise AssertionError(f"the drawn month ends of {loan} do not match delinquency_panel")

    canvas = Canvas(height=16.6)
    origin, scale, x0 = pd.Timestamp("2024-01-01"), 0.1265, 6.8
    labels = 6.2

    def at(time):
        return x0 + (time - origin).days * scale

    ends = [month.to_timestamp(how="end").normalize() for month in rows["month"]]
    canvas.call(0.6, 0.15, 'delinquency_panel(*instalments, *paid, end="2024-06")')
    canvas.note(31.4, 0.65, f'loan_id == "{loan}"', ha="right", color=INK, bold=True)

    # Days past due run from the due date of the oldest unpaid instalment to each month end.
    oldest = due["due_date"].iloc[1]
    bracket, instalments, payment_row, axis = 2.1, 3.5, 4.9, 6.2
    canvas.line(at(oldest), bracket, at(ends[-1]), bracket, color=AMBER, linewidth=1.6, z=3)
    canvas.line(
        at(oldest), bracket - 0.28, at(oldest), bracket + 0.28, color=AMBER, linewidth=1.6, z=3
    )
    canvas.line(at(oldest), bracket, at(oldest), instalments, color=AMBER, style="--", z=1)
    canvas.note(
        at(oldest) - 0.4, bracket, "days_past_due", ha="right", color=RESULT_TEXT, bold=True
    )
    for end in ends:
        late = end > oldest
        start = bracket if late else instalments - 1.0
        canvas.line(at(end), start, at(end), axis, color=AMBER, linewidth=0.9, style="--", z=1)
        if late:
            canvas.ax.plot(at(end), bracket, "o", markersize=4.5, color=AMBER, zorder=4)

    canvas.text(labels, instalments, "amount_due", size=9, bold=True, ha="right")
    canvas.text(labels, payment_row, "payment_amount", size=9, bold=True, ha="right")
    for y in (instalments, payment_row):
        canvas.line(labels + 0.4, y, at(ends[-1]) + 0.6, y, color=CELL_EDGE)
    fill, color = CATEGORY[0]
    for time, amount in zip(due["due_date"], due["amount_due"], strict=True):
        is_oldest = time == oldest
        canvas.ax.plot(
            at(time),
            instalments,
            "s",
            markersize=9,
            markerfacecolor=fill,
            markeredgecolor=AMBER if is_oldest else color,
            markeredgewidth=1.8 if is_oldest else 1.2,
            zorder=5,
        )
        canvas.note(at(time), instalments - 0.62, f"{amount:g}", ha="center", color=INK)
    fill, color = CATEGORY[3]
    for time, amount in zip(paid["date"], paid["amount"], strict=True):
        canvas.ax.plot(
            at(time),
            payment_row,
            "o",
            markersize=9,
            markerfacecolor=fill,
            markeredgecolor=color,
            markeredgewidth=1.2,
            zorder=5,
        )
        canvas.note(at(time), payment_row - 0.62, f"{amount:g}", ha="center", color=INK)

    canvas.arrow((labels + 0.4, axis), (at(ends[-1]) + 1.2, axis), color=MUTED)
    for time in due["due_date"]:
        canvas.line(at(time), axis - 0.15, at(time), axis + 0.15, color=MUTED)
        canvas.note(at(time), axis + 0.6, time.strftime("%Y-%m-%d"), ha="center", size=8.3)
    for end in ends:
        canvas.line(at(end), axis - 0.2, at(end), axis + 0.2, color=AMBER, linewidth=1.4)
        canvas.note(
            at(end), axis + 1.35, end.strftime("%Y-%m-%d"), ha="center", size=8.3, color=RESULT_TEXT
        )

    # The rows of the loan, one column per month end.
    top = axis + 2.2
    tints = dict(zip(statuses, ramp(len(statuses)), strict=True))
    canvas.text(labels, top + HEADER / 2 - 0.05, "month", size=9, bold=True, ha="right")
    cells = {
        "days_past_due": [str(v) for v in rows["days_past_due"]],
        "overdue_amount": [f"{v:.2f}" for v in rows["overdue_amount"]],
        "outstanding": [f"{v:.2f}" for v in rows["outstanding"]],
        "status": [str(v) for v in rows["status"]],
    }
    width = 3.4
    for j, (month, end) in enumerate(zip(rows["month"], ends, strict=True)):
        centre = at(end)
        canvas.text(centre, top + HEADER / 2 - 0.05, str(month), size=9, bold=True)
        for i, (name, values) in enumerate(cells.items()):
            y = top + HEADER + (i + 0.5) * ROW
            if j == 0:
                canvas.text(labels, y, name, size=9, bold=True, ha="right", color=RESULT_TEXT)
            if name == "status":
                fill, color = tints[values[j]]
                edge = None
            else:
                fill, color, edge = RESULT_FILL, INK, RESULT_EDGE
            canvas.box(centre - width / 2, y - CELL / 2, width, CELL, fill=fill, edge=edge)
            canvas.text(centre, y, values[j], color=color)
    canvas.line(labels + 0.4, top + HEADER - 0.06, at(ends[-1]) + width / 2, top + HEADER - 0.06)

    # The month end of February by hand.
    end = ends[1]
    before = due.loc[due["due_date"] < end, "amount_due"]
    made = paid.loc[paid["date"] <= end, "amount"]

    def total(values):
        return " + ".join(f"{v:g}" for v in values)

    canvas.notes(
        labels + 0.4,
        top + HEADER + len(cells) * ROW + 0.75,
        [
            f"{rows['month'][1]}:  days_past_due  = {end:%Y-%m-%d} - {oldest:%Y-%m-%d}"
            f" = {cells['days_past_due'][1]}",
            f"{'':9}overdue_amount = {total(before)} - ({total(made)})"
            f" = {cells['overdue_amount'][1]}",
            f"{'':9}outstanding    = {total(due['amount_due'])} - ({total(made)})"
            f" = {cells['outstanding'][1]}",
        ],
    )
    return canvas.fig


def _monthly_pairs():
    panel = make_loan_panel(rng=0)
    later = panel[["loan_id", "month_on_book", "status"]].assign(
        month_on_book=panel["month_on_book"] - 1
    )
    pairs = panel.merge(later, on=["loan_id", "month_on_book"], suffixes=("", "_next"))
    return pairs["status"], pairs["status_next"]


def _blend(light: str, dark: str, share: float) -> str:
    """Mix two hex colors: ``share`` 0 gives the light one, 1 the dark one."""
    a = np.array([int(light[i : i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(dark[i : i + 2], 16) for i in (1, 3, 5)])
    mixed = np.round(a + (b - a) * min(max(share, 0.0), 1.0)).astype(int)
    return "#" + "".join(f"{v:02x}" for v in mixed)


def figure_transition_matrix():
    """The monthly transition matrix, with the roll rates just right of its diagonal."""
    before, after = _monthly_pairs()
    monthly = transition_matrix(before, after, normalize=True)
    rolls = roll_rates(before, after)
    states = list(monthly.index)

    canvas = Canvas(height=8.2)
    left, top, width = 5.6, 1.7, 3.1
    canvas.call(left, 0.15, "transition_matrix(before, after, normalize=True)")
    canvas.text(
        left - 0.4, top + HEADER / 2 - 0.05, "before \\ after", size=8.5, ha="right", color=MUTED
    )
    for j, state in enumerate(states):
        canvas.text(
            left + j * (width + GAP) + width / 2, top + HEADER / 2 - 0.05, state, size=9, bold=True
        )
    for i, state in enumerate(states):
        y = top + HEADER + (i + 0.5) * ROW
        canvas.text(left - 0.4, y, state, size=9, ha="right", bold=True)
        for j, value in enumerate(monthly.loc[state]):
            x = left + j * (width + GAP)
            if j == i + 1:
                fill, edge, color = RESULT_FILL, AMBER, RESULT_TEXT
            elif value > 0:
                fill, edge = _blend(BLUE_FILL, BLUE, value**0.8), None
                color = "#ffffff" if value > 0.5 else INK
            else:
                fill, edge, color = CELL_FILL, CELL_EDGE, FAINT
            canvas.box(
                x,
                y - CELL / 2,
                width,
                CELL,
                fill=fill,
                edge=edge,
                linewidth=1.3 if j == i + 1 else 0.8,
            )
            canvas.text(x + width / 2, y, number(value), color=color, bold=j == i + 1)
    right = left + 5 * width + 4 * GAP
    canvas.line(left, top + HEADER - 0.06, right, top + HEADER - 0.06)
    x = right + 1.4
    canvas.call(31.4, 0.15, "roll_rates(before, after)", ha="right")
    canvas.text(x + 1.6, top + HEADER / 2 - 0.05, "roll_rate", size=9, bold=True, color=RESULT_TEXT)
    for i, state in enumerate(states[:-1]):
        y = top + HEADER + (i + 0.5) * ROW
        canvas.box(x, y - CELL / 2, 3.2, CELL, fill=RESULT_FILL, edge=RESULT_EDGE)
        canvas.text(x + 1.6, y, number(rolls[state]))
    return canvas.fig


def _migration_example():
    """The yearly migrations of the vintage guide: grades at the start and statuses at the end."""
    rng = np.random.default_rng(0)
    grades = ["A", "B", "C", "D", "E"]
    statuses = grades + ["default", "closed"]
    yearly = np.array(
        [
            [0.86, 0.07, 0.02, 0.01, 0.00, 0.00, 0.04],
            [0.06, 0.80, 0.06, 0.01, 0.03, 0.01, 0.03],
            [0.01, 0.07, 0.78, 0.07, 0.02, 0.02, 0.03],
            [0.00, 0.01, 0.08, 0.76, 0.08, 0.04, 0.03],
            [0.00, 0.00, 0.02, 0.09, 0.73, 0.13, 0.03],
        ]
    )
    counts = [300, 600, 700, 400, 200]
    start = np.repeat(grades, counts)
    end = np.concatenate(
        [rng.choice(statuses, size=count, p=row) for count, row in zip(counts, yearly, strict=True)]
    )
    return start, end, grades, statuses


def figure_migration_test():
    """The migration shares, with the cell that migration_test flags and its neighbour."""
    start, end, grades, statuses = _migration_example()
    shares = transition_matrix(start, end, states=statuses, normalize=True).loc[grades]
    total = pd.Series(start).value_counts().reindex(grades)
    test = migration_test(start, end, grades=grades)
    flagged = test.index[test["p_value"] < 0.05]
    if len(flagged) != 1:
        raise AssertionError("the guide flags one cell")
    before, after = flagged[0]
    row, column = grades.index(before), statuses.index(after)
    neighbour = column - 1 if column > row else column + 1
    p, q, n = shares.iat[row, column], shares.iat[row, neighbour], int(total[before])
    se = np.sqrt((p * (1 - p) + q * (1 - q) + 2 * p * q) / n)
    z = (q - p) / se
    if not np.isclose(z, test.loc[(before, after), "statistic"], rtol=1e-12, atol=0):
        raise AssertionError("the drawn statistic does not match migration_test")

    canvas = Canvas(height=12.3)
    left, top, width = 2.6, 1.9, 2.25
    canvas.call(0.6, 0.15, "migration_test(start, end, grades=grades)")
    for j, status in enumerate(statuses):
        x = left + j * (width + GAP) + width / 2
        canvas.text(
            x,
            top + HEADER / 2 - 0.05,
            status,
            size=8.5,
            bold=True,
            color=INK if status in grades else MUTED,
        )
    x_total = left + len(statuses) * (width + GAP) + 0.3
    canvas.text(x_total + 1.0, top + HEADER / 2 - 0.05, "total", size=8.5, bold=True)
    for i, grade in enumerate(grades):
        y = top + HEADER + (i + 0.5) * ROW
        canvas.text(left - 0.4, y, grade, size=9, ha="right", bold=True)
        for j, value in enumerate(shares.loc[grade]):
            x = left + j * (width + GAP)
            if (i, j) == (row, column):
                fill, edge, color = RESULT_FILL, AMBER, RESULT_TEXT
            elif (i, j) == (row, neighbour):
                fill, edge, color = CELL_FILL, AMBER, RESULT_TEXT
            elif j >= len(grades):
                fill, edge, color = MISSING_FILL, None, MUTED
            elif value > 0:
                fill, edge = _blend(BLUE_FILL, BLUE, value**0.8), None
                color = "#ffffff" if value > 0.5 else INK
            else:
                fill, edge, color = CELL_FILL, CELL_EDGE, FAINT
            canvas.box(
                x,
                y - CELL / 2,
                width,
                CELL,
                fill=fill,
                edge=edge,
                linewidth=1.4 if edge == AMBER else 0.8,
            )
            canvas.text(x + width / 2, y, number(value), size=8.8, color=color)
        canvas.box(x_total, y - CELL / 2, 2.0, CELL, fill=CELL_FILL, edge=CELL_EDGE)
        canvas.text(x_total + 1.0, y, str(int(total[grade])), size=8.8)
    right = x_total + 2.0
    canvas.line(left, top + HEADER - 0.06, right, top + HEADER - 0.06)
    y_arrow = top + HEADER + (row + 0.5) * ROW - CELL / 2 - 0.05
    x_from = left + neighbour * (width + GAP) + width / 2
    x_to = left + column * (width + GAP) + width / 2
    canvas.arrow((x_from, y_arrow), (x_to, y_arrow), color=AMBER, linewidth=1.2, bend=-0.35)
    pair = f"p_{before}{after}"
    near = f"p_{before}{statuses[neighbour]}"
    y = top + HEADER + len(grades) * ROW + 0.7
    canvas.notes(
        left,
        y,
        [
            f"N_{before} = {n},  {pair} = {number(p, 4)},  {near} = {number(q, 4)}",
            f"se = sqrt(({pair} (1 - {pair}) + {near} (1 - {near}) + 2 {pair} {near}) / N_{before})"
            f" = {number(se, 4)}",
            f"statistic = ({near} - {pair}) / se = {number(z, 4)}",
        ],
        size=8.4,
    )
    canvas.note(
        left,
        y + 3 * 0.95,
        f"p_value = Phi(statistic) = {number(test.loc[(before, after), 'p_value'], 4)}",
        size=8.4,
        color=RESULT_TEXT,
        bold=True,
    )
    return canvas.fig


def figure_migration_bandwidth():
    """The moves of the three-grade example and how migration_bandwidth weighs them."""
    before = ["A", "A", "A", "B", "B", "B", "C", "C"]
    after = ["A", "B", "C", "A", "B", "default", "B", "C"]
    grades = ["A", "B", "C"]
    result = migration_bandwidth(before, after, grades=grades)
    columns = grades + ["default"]
    counts = pd.crosstab(pd.Series(before), pd.Series(after)).reindex(
        index=grades, columns=columns, fill_value=0
    )
    k = len(grades)
    furthest = {g: max(abs(i - (k - 1)), i) for i, g in enumerate(grades)}
    sums = {"upper": [0, 0], "lower": [0, 0]}
    terms = {"upper": ([], []), "lower": ([], [])}
    for i, g in enumerate(grades):
        for j, h in enumerate(grades):
            n = int(counts.loc[g, h])
            if i == j or n == 0:
                continue
            side = "upper" if j > i else "lower"
            sums[side][0] += abs(i - j) * n
            sums[side][1] += furthest[g] * n
            terms[side][0].append(f"{abs(i - j)}×{n}")
            terms[side][1].append(f"{furthest[g]}×{n}")
    for side, (top_sum, bottom_sum) in sums.items():
        if not np.isclose(result[side], top_sum / bottom_sum, rtol=1e-12, atol=0):
            raise AssertionError("the drawn bandwidth does not match migration_bandwidth")

    canvas = Canvas(height=11.6)
    left, top, width, height = 3.0, 1.9, 3.4, 1.5
    canvas.call(0.6, 0.15, "migration_bandwidth(before, after, grades=grades)")
    canvas.text(left - 0.4, top + HEADER / 2 - 0.05, "i", size=8.5, ha="right", color=MUTED)
    for j, status in enumerate([*columns, "furthest"]):
        x = left + j * (width + GAP) + width / 2
        color = INK if status in grades else MUTED if status == "default" else RESULT_TEXT
        canvas.text(x, top + HEADER / 2 - 0.05, status, size=9, bold=True, color=color)
    tints = {"upper": CATEGORY[2], "lower": CATEGORY[0]}
    for i, grade in enumerate(grades):
        y = top + HEADER + i * (height + GAP)
        canvas.text(left - 0.4, y + height / 2, f"{i + 1} {grade}", size=9, ha="right", bold=True)
        for j, status in enumerate(columns):
            x = left + j * (width + GAP)
            n = int(counts.loc[grade, status])
            distance = None
            if status not in grades:
                fill, color = MISSING
            elif j == i:
                fill, color = CELL_FILL, INK
            else:
                fill, color = tints["upper" if j > i else "lower"]
                distance = abs(i - j)
            canvas.box(x, y, width, height, fill=fill, edge=CELL_EDGE)
            if distance is None:
                canvas.text(x + width / 2, y + height / 2, str(n), size=10, color=color, bold=True)
            else:
                canvas.text(x + width / 2, y + 0.5, str(n), size=10, color=color, bold=True)
                canvas.text(x + width / 2, y + 1.1, f"|i - j| = {distance}", size=7.6, color=color)
        x = left + len(columns) * (width + GAP)
        canvas.box(x, y, width, height, fill=RESULT_FILL, edge=RESULT_EDGE)
        canvas.text(x + width / 2, y + height / 2, str(furthest[grade]), size=10, bold=True)
    canvas.line(left, top + HEADER - 0.06, x + width, top + HEADER - 0.06)
    y = top + HEADER + k * (height + GAP) + 0.5
    canvas.note(left, y, f"furthest = max(|i - K|, |i - 1|),  K = {k}", size=8.6)
    y += 1.15
    for side in ("upper", "lower"):
        moves, weights = terms[side]
        canvas.note(
            left,
            y,
            f"{side} = ({' + '.join(moves)}) / ({' + '.join(weights)}) = "
            f"{sums[side][0]} / {sums[side][1]} = {number(result[side], 4)}",
            size=8.8,
            color=tints[side][1],
            bold=True,
        )
        y += 1.05
    return canvas.fig


# Lifetime PD and ECL


def figure_expected_credit_loss():
    """The worked example of the lifetime guide: PDs per year and the discounted losses."""
    cumulative = [0.02, 0.05, 0.08]
    marginal = convert_pd(cumulative, kind="cumulative", to="marginal")
    conditional = convert_pd(cumulative, kind="cumulative", to="conditional")
    lgd, ead, rate = 0.5, [1000.0, 700.0, 400.0], 0.05
    t = np.arange(1, 4)
    discount = (1 + rate) ** -t
    terms = marginal * lgd * np.array(ead) * discount
    lifetime = expected_credit_loss([marginal], lgd, [ead], rate=rate)[0]
    twelve = expected_credit_loss([marginal], lgd, [ead], rate=rate, horizon=1)[0]
    if not np.isclose(terms.sum(), lifetime) or not np.isclose(terms[0], twelve):
        raise AssertionError("the drawn terms do not match expected_credit_loss")

    canvas = Canvas(height=12.0)
    x = 0.6
    canvas.call(x, 0.15, "convert_pd(cumulative, ...)")
    canvas.call(31.4, 0.15, "expected_credit_loss(marginal, lgd, ead, rate=rate)", ha="right")
    name = "expected_credit_loss"
    out = canvas.table(
        x,
        1.6,
        [
            Column("t", [str(v) for v in t], 1.3, color=MUTED),
            Column("cumulative", [f"{v:.2f}" for v in cumulative], 3.6),
            result("marginal", [f"{v:.2f}" for v in marginal], 3.2),
            result("conditional", [number(v, 4) for v in conditional], 3.9),
            Column("lgd", [f"{lgd:g}"] * 3, 1.8),
            Column("ead", [f"{v:g}" for v in ead], 2.2),
            Column("(1 + rate)^-t", [number(v, 4) for v in discount], 4.4),
            result(name, [number(v, 2) for v in terms], 6.6),
        ],
    )
    low, high = out.columns[name]
    for i, (horizon, value) in enumerate((("horizon=1", twelve), ("horizon=None", lifetime))):
        y = out.bottom + 0.55 + i * 1.0
        canvas.note(low - 0.3, y, horizon, ha="right", color=INK)
        canvas.note(
            (low + high) / 2, y, number(value, 2), ha="center", color=RESULT_TEXT, bold=True
        )
    canvas.notes(
        x,
        out.bottom + 2.75,
        [
            "marginal             = cumulative - cumulative.shift(fill_value=0)",
            "conditional          = marginal / (1 - cumulative.shift(fill_value=0))",
            "expected_credit_loss = sum(marginal * lgd * ead * (1 + rate)^-t)",
            f"lgd = {lgd:g},  rate = {rate:g}",
        ],
    )
    return canvas.fig


def figure_stage_rules():
    """The days-past-due backstops of assign_stages and the boundary cases of the guide."""
    days = [0, 30, 31, 89, 90]
    stages = assign_stages([False] * len(days), days_past_due=days)

    canvas = Canvas(height=11.4)
    x0, x1, axis = 4.6, 27.4, 4.4

    def at(value):
        return x0 + value / 120 * (x1 - x0)

    zones = [(0, 30, TEAL_FILL), (30, 90, "#fbe8cc"), (90, 120, EVENT_FILL)]
    for stage, (low, high, fill) in enumerate(zones, start=1):
        canvas.box(at(low), 1.6, at(high) - at(low), axis - 1.6, fill=fill, radius=0.2, z=0)
        canvas.note(
            (at(low) + at(high)) / 2, 2.4, f"stage {stage}", ha="center", color=INK, bold=True
        )
    canvas.arrow((at(0), axis), (at(120) + 0.9, axis), color=MUTED, linewidth=1.1)
    canvas.note(at(120) + 0.9, axis + 0.75, "days_past_due", ha="right")
    for value, name in ((30, "stage_2_days = 30"), (90, "stage_3_days = 90")):
        canvas.line(
            at(value), 1.3, at(value), axis + 0.35, color=AMBER, linewidth=1.5, style="--", z=3
        )
        canvas.note(at(value), 0.8, name, ha="center", color=RESULT_TEXT, bold=True)

    width = 2.6
    cells = [x0 + 2.0 + i * 4.4 for i in range(len(days))]
    top = 6.6
    canvas.text(cells[0] - 0.5, top, "days_past_due", size=9, ha="right", bold=True)
    canvas.text(
        cells[0] - 0.5, top + 1.1, "stage", size=9, ha="right", bold=True, color=RESULT_TEXT
    )
    for cell, value, stage in zip(cells, days, stages, strict=True):
        canvas.ax.plot(at(value), axis, "o", color=INK, markersize=5, zorder=5)
        canvas.line(
            at(value),
            axis + 0.2,
            cell + width / 2,
            top - CELL / 2 - 0.1,
            color=FAINT,
            linewidth=0.9,
        )
        canvas.box(cell, top - CELL / 2, width, CELL, fill=CELL_FILL, edge=CELL_EDGE)
        canvas.text(cell + width / 2, top, str(value))
        canvas.box(cell, top + 1.1 - CELL / 2, width, CELL, fill=RESULT_FILL, edge=RESULT_EDGE)
        canvas.text(cell + width / 2, top + 1.1, str(stage), bold=True, color=RESULT_TEXT)
    canvas.notes(
        x0,
        top + 2.6,
        [
            "assign_stages(significant_increase, days_past_due, defaulted)",
            "significant_increase = True -> stage >= 2,   defaulted = True -> stage 3",
        ],
    )
    return canvas.fig


# Binning specifications


def figure_spec_roundtrip():
    """A specification from the fit to JSON and back, followed for the pdays column."""
    data = load_bank()
    development, monitoring = data.iloc[:3000], data.iloc[3000:]
    shortlist = ["balance", "pdays", "age", "poutcome", "contact", "housing", "job"]
    spec = fit_spec(
        development,
        development["target"],
        columns=shortlist,
        method={"balance": "monotonic", "pdays": "tree"},
        bins={"balance": 4, "age": [-np.inf, 25, 35, 45, 55, 65, np.inf]},
        special={"pdays": [-1]},
        options={"pdays": {"min_bin_share": 0.1}},
    )
    text = spec.to_json()
    loaded = BinningSpec.from_json(text)
    entry = json.loads(text)["variables"][1]
    rows = monitoring.loc[[3000, 3007, 3013]]
    groups = loaded.cut(rows)["pdays"]
    encoded = loaded.woe_transform(rows)["pdays"]
    table = spec.tables["pdays"]
    styles = group_styles(table.index)

    canvas = Canvas(height=21.0)
    left_card, right_card = (0.4, 11.0), (14.4, 17.2)
    # Top left: the development sample.
    canvas.card(left_card[0], 0.4, left_card[1], 9.4)
    canvas.note(1.0, 1.15, "development", color=INK, bold=True, size=9)
    sample = development["pdays"].loc[[0, 1, 2, 24, 25]]
    canvas.table(
        1.2,
        1.9,
        [
            Column("index", [str(i) for i in sample.index], 2.6, color=MUTED),
            Column("pdays", [str(v) for v in sample], 2.8),
            target_column(development["target"].loc[sample.index], 2.6),
        ],
    )
    canvas.note(1.2, 8.8, f"development.shape = {development.shape}", size=8.5)
    # Top right: the fitted specification.
    x = right_card[0]
    canvas.card(x, 0.4, right_card[1], 9.4)
    canvas.note(x + 0.6, 1.15, "spec.tables['pdays']", color=INK, bold=True, size=9)
    canvas.table(
        x + 0.6,
        1.9,
        [
            group_column("bin", table.index, 4.9, styles),
            Column("total", [str(v) for v in table["total"]], 2.3),
            result("woe", [number(v) for v in table["woe"]], 2.6),
        ],
    )
    ends = [
        "-inf" if v == -np.inf else "inf" if v == np.inf else f"{v:g}" for v in spec.edges["pdays"]
    ]
    canvas.notes(
        x + 11.3,
        2.4,
        [
            "edges",
            f"[{', '.join(ends[:3])},",
            f" {', '.join(ends[3:])}]",
            "",
            "special",
            str(list(spec.special["pdays"])),
        ],
        size=8.5,
    )
    canvas.arrow((left_card[0] + left_card[1] + 0.2, 5.2), (x - 0.2, 5.2))
    canvas.call((left_card[0] + left_card[1] + x) / 2, 3.6, "fit_spec", ha="center", size=9)
    # Bottom right: the JSON text.
    canvas.card(x, 11.8, right_card[1], 8.8)
    lines = [
        '{"name": "pdays", "kind": "numeric",',
        ' "edges": [{"float": "-inf"}, 100.0, 176.0, ...],',
        f' "special": {json.dumps(entry["special"])}, "unseen": {json.dumps(entry["unseen"])},',
        ' "table": [{"bin": -1, ...},',
        '           {"bin": {"interval": [{"float": "-inf"}, 100.0],',
        '                    "closed": "right"},',
        f'            "woe": {entry["table"][1]["woe"]!r}, ...}},',
        "           ...]}",
    ]
    canvas.notes(x + 0.6, 12.8, lines, step=0.9, size=8.0, color=INK)
    middle = x + right_card[1] / 2
    canvas.arrow((middle, 9.9), (middle, 11.7))
    canvas.call(middle + 0.4, 10.3, "to_json()", size=9)
    # Bottom left: a monitoring sample encoded with the loaded specification.
    canvas.card(left_card[0], 11.8, left_card[1], 8.8)
    canvas.note(1.0, 12.55, "monitoring", color=INK, bold=True, size=9)
    canvas.table(
        1.2,
        13.3,
        [
            Column("pdays", [str(v) for v in rows["pdays"]], 2.0),
            group_column("bin", groups, 4.6, styles),
            result("woe", [number(v) for v in encoded], 2.6),
        ],
    )
    canvas.call(
        left_card[0] + left_card[1] / 2,
        18.7,
        "loaded.woe_transform(monitoring)",
        ha="center",
        size=8.5,
    )
    canvas.arrow((x - 0.2, 16.2), (left_card[0] + left_card[1] + 0.2, 16.2))
    canvas.call((left_card[0] + left_card[1] + x) / 2, 16.6, "from_json", ha="center", size=9)
    return canvas.fig


# Monitoring


def figure_monitoring_table():
    """The worked example of the monitoring guide: bands from January, one row per month."""
    score = [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.6, 0.7, 0.5, 0.6, 0.8, 0.9]
    target = [0, 0, 1, 1, 0, 1, 0, 1, 0, np.nan, 1, np.nan]
    period = ["2024-01"] * 4 + ["2024-02"] * 4 + ["2024-03"] * 4
    table = monitoring_table(score, period, target, reference_period="2024-01", bins=2)
    edges = bin_edges(score[:4], bins=2)
    low, high = ramp(2)

    canvas = Canvas(height=8.9)
    left, width = 3.8, 1.6
    step = width + GAP
    canvas.call(
        0.6,
        0.15,
        "monitoring_table(score, period, target, reference_period='2024-01', bins=2)",
        size=9,
    )
    top = 1.9
    header = top + HEADER / 2 - 0.05
    canvas.text(2.1, header, "period", size=9, bold=True)
    canvas.text(left + 2 * step - GAP / 2, header, "score", size=9, bold=True)
    canvas.text(left + 6 * step + 0.4 - GAP / 2, header, "target", size=9, bold=True)
    for k, month in enumerate(dict.fromkeys(period)):
        y = top + HEADER + (k + 0.5) * ROW
        canvas.text(2.1, y, month, color=BLUE if k == 0 else INK, bold=k == 0)
        for i in range(4):
            s, t = score[4 * k + i], target[4 * k + i]
            x = left + i * step
            tint = low if s <= edges[1] else high
            canvas.box(
                x,
                y - CELL / 2,
                width,
                CELL,
                fill=tint[0],
                edge=BLUE if k == 0 else None,
                linewidth=1.0,
            )
            canvas.text(x + width / 2, y, f"{s:g}")
            x = left + (4 + i) * step + 0.4
            known = not pd.isna(t)
            fill = MISSING_FILL if not known else EVENT_FILL if t else CELL_FILL
            canvas.box(
                x,
                y - CELL / 2,
                width,
                CELL,
                fill=fill,
                edge=CELL_EDGE if fill == CELL_FILL else None,
            )
            canvas.text(
                x + width / 2,
                y,
                str(int(t)) if known else "NaN",
                color=FAINT if not known else EVENT if t else INK,
                bold=bool(known and t),
            )
    right = left + 8 * step + 0.4 - GAP
    canvas.line(0.6, top + HEADER - 0.06, right, top + HEADER - 0.06)
    y = top + HEADER + 3 * ROW + 0.7
    for i, (name, tint) in enumerate(
        zip([f"(-inf, {edges[1]:g}]", f"({edges[1]:g}, inf]"], (low, high), strict=True)
    ):
        x = left + i * 6.4
        canvas.box(x, y - 0.3, 0.6, 0.6, fill=tint[0], radius=0.08)
        canvas.note(x + 0.9, y, name, color=INK)
    shown = np.array2string(edges, separator=", ")
    canvas.note(0.6, y + 1.1, f"edges = bin_edges(score[:4], bins=2) = {shown}")

    x = right + 1.6
    out = canvas.table(
        x,
        top,
        [
            Column("total", [str(v) for v in table["total"]], 1.9),
            Column("known", [str(v) for v in table["known"]], 1.9),
            Column("event", [str(v) for v in table["event"]], 1.9),
            Column("auc", [f"{v:g}" for v in table["auc"]], 1.9),
            result("psi", [number(v, 4) for v in table["psi"]], 2.6),
        ],
    )
    canvas.arrow((right + 0.25, out.rows[1]), (x - 0.25, out.rows[1]), linewidth=1.2)
    return canvas.fig


# Univariate profile


def figure_profile_rows():
    """profile_table turns every column of a frame into one row of the profile."""
    data = load_bank()
    development = data.iloc[:3000]
    candidates = development.drop(columns="target").replace("unknown", np.nan)
    profile = profile_table(candidates, development["target"])
    columns = ["age", "job", "contact", "pdays", "poutcome"]
    styles = group_styles(columns)
    sample = candidates[columns].head(6)

    canvas = Canvas(height=10.2)
    left = 0.6
    widths = {"age": 1.5, "job": 3.9, "contact": 2.9, "pdays": 1.7, "poutcome": 2.9}
    cells = []
    for column in columns:
        values = sample[column]
        missing = values.isna().to_numpy()
        cells.append(
            Column(
                column,
                ["NaN" if m else str(v) for v, m in zip(values, missing, strict=True)],
                widths[column],
                fill=[MISSING_FILL if m else styles[column][0] for m in missing],
                color=[FAINT if m else styles[column][1] for m in missing],
                header_color=styles[column][1],
            )
        )
    frame = canvas.table(left, 1.6, cells)
    canvas.call(left, 0.15, "candidates.head(6)")
    canvas.note(left, frame.bottom + 0.5, f"candidates.shape = {candidates.shape}")

    x = frame.right + 1.4
    canvas.call(31.4, 0.15, "profile_table(candidates, target)", ha="right")
    shown = profile.loc[columns]
    out = canvas.table(
        x,
        1.6,
        [
            group_column("variable", columns, 3.0, styles),
            Column("kind", list(shown["kind"]), 3.9),
            Column("missing_share", [number(v) for v in shown["missing_share"]], 4.3),
            Column("distinct", [str(v) for v in shown["distinct"]], 2.4),
            result("iv", [number(v) for v in shown["iv"]], 1.9),
        ],
    )
    canvas.arrow((frame.right + 0.3, out.rows[2]), (x - 0.3, out.rows[2]))
    canvas.note(x, out.bottom + 0.5, f"profile.shape = {profile.shape}")
    return canvas.fig


# Variable selection


def _selection_example():
    data = load_bank()
    development = data.iloc[:3000]
    target = development["target"]
    edges = {
        "age": bin_edges(development["age"], bins=5),
        "balance": bin_edges(development["balance"], bins=5),
        "day": bin_edges(development["day"], bins=5),
        "campaign": [-np.inf, 1, 2, 3, np.inf],
        "pdays": [-np.inf, -1, 100, 200, 300, np.inf],
        "previous": [-np.inf, 0, 1, 3, np.inf],
    }
    groups = pd.DataFrame({c: pd.cut(development[c], e) for c, e in edges.items()})
    for column in [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "poutcome",
    ]:
        groups[column] = development[column]
    tables = {column: woe_table(groups[column], target) for column in groups}
    iv = pd.Series({column: table["iv"].sum() for column, table in tables.items()})
    iv = iv.sort_values(ascending=False)
    candidates = iv.index[iv >= 0.02]
    X = pd.DataFrame({c: woe_transform(groups[c], tables[c]) for c in candidates})
    return X, target, iv


def figure_correlation_filter():
    """correlation_filter visits the variables by IV and drops those too close to a kept one."""
    X, _, iv = _selection_example()
    threshold = 0.7
    correlation = X.corr().abs()
    kept, largest, partner = [], [], []
    for column in X.columns:
        if kept:
            value = correlation.loc[column, kept].max()
            largest.append(value)
            partner.append(correlation.loc[column, kept].idxmax())
        else:
            largest.append(np.nan)
            partner.append(None)
        if not kept or largest[-1] <= threshold:
            kept.append(column)
    if kept != correlation_filter(X, priority=iv).tolist():
        raise AssertionError("the drawn filter does not match correlation_filter")

    dropped = [column not in kept for column in X.columns]
    canvas = Canvas(height=14.6)
    x = 2.6
    canvas.call(x, 0.15, "correlation_filter(X, priority=iv)  # threshold=0.7")
    out = canvas.table(
        x,
        1.6,
        [
            Column("variable", list(X.columns), 3.6),
            Column("iv", [number(iv[c]) for c in X.columns], 2.4, color=MUTED),
            Column(
                "max |corr| with kept",
                [number(v, 2) for v in largest],
                6.4,
                color=[RESULT_TEXT if d else INK for d in dropped],
                bold=dropped,
                edge=[AMBER if d else None for d in dropped],
            ),
            result(
                "kept",
                [str(not d) for d in dropped],
                2.6,
                fill=[EVENT_FILL if d else RESULT_FILL for d in dropped],
                color=[EVENT if d else INK for d in dropped],
                edge=[EVENT if d else RESULT_EDGE for d in dropped],
            ),
        ],
    )
    canvas.arrow((x - 0.6, out.rows[0] - 0.3), (x - 0.6, out.rows[-1] + 0.3), color=MUTED)
    canvas.note(x - 1.1, (out.rows[0] + out.rows[-1]) / 2, "iv", ha="right", size=8.5)
    for i, column in enumerate(X.columns):
        if dropped[i]:
            j = list(X.columns).index(partner[i])
            near, far = out.right + 0.3, out.right + 1.0
            canvas.line(near, out.rows[i], far, out.rows[i], color=AMBER, linewidth=1.3)
            canvas.line(far, out.rows[i], far, out.rows[j], color=AMBER, linewidth=1.3)
            canvas.arrow((far, out.rows[j]), (near, out.rows[j]), color=AMBER, linewidth=1.3)
            canvas.note(
                far + 0.5,
                (out.rows[i] + out.rows[j]) / 2,
                f"|corr({column}, {partner[i]})| = {largest[i]:.2f} > {threshold}",
                color=RESULT_TEXT,
            )
    return canvas.fig


def figure_stepwise_path():
    """Forward steps of stepwise: the z statistic of each candidate, and the one that enters."""
    X, target, iv = _selection_example()
    kept = list(correlation_filter(X, priority=iv))
    selected, steps = [], []
    while True:
        trials, best, best_z = {}, None, -np.inf
        for column in kept:
            if column in selected:
                continue
            fit = logistic_regression(X[[*selected, column]], target)
            z, p_value = fit.loc[column, "z"], fit.loc[column, "p_value"]
            eligible = p_value < 0.05 and (fit["coef"].drop("intercept") > 0).all()
            trials[column] = (z, eligible)
            if eligible and abs(z) > best_z:
                best, best_z = column, abs(z)
        steps.append((list(selected), trials, best))
        if best is None:
            break
        selected.append(best)
        fit = logistic_regression(X[selected], target).drop("intercept")
        if (fit["p_value"] > 0.1).any() or (fit["coef"] <= 0).any():
            raise AssertionError("stepwise would remove a variable, which the diagram omits")
    if sorted(selected) != sorted(stepwise(X[kept], target)):
        raise AssertionError("the drawn steps do not match stepwise")

    canvas = Canvas(height=15.4)
    left, width = 3.4, 2.62
    canvas.call(left, 0.15, "stepwise(X[kept], target)  # p_enter=0.05, positive=True")
    header = 2.0
    for j, column in enumerate(kept):
        canvas.text(left + j * (width + GAP) + width / 2, header, column, size=8.3, bold=True)
    right = left + len(kept) * (width + GAP) - GAP
    canvas.line(left, header + 0.55, right, header + 0.55)
    for i, (before, trials, best) in enumerate(steps):
        y = header + 1.25 + i * ROW
        canvas.text(left - 0.5, y, str(i + 1), size=9, ha="right", color=MUTED)
        for j, column in enumerate(kept):
            x = left + j * (width + GAP)
            if column in before:
                canvas.box(x, y - CELL / 2, width, CELL, fill=BLUE_FILL, edge=BLUE_FILL)
                continue
            z, eligible = trials[column]
            chosen = column == best
            fill = RESULT_FILL if chosen else CELL_FILL
            edge = AMBER if chosen else CELL_EDGE
            color = RESULT_TEXT if chosen else INK if eligible else FAINT
            canvas.box(
                x, y - CELL / 2, width, CELL, fill=fill, edge=edge, linewidth=1.3 if chosen else 0.8
            )
            canvas.text(x + width / 2, y, f"{z:.2f}", color=color, bold=chosen, size=9)
    bottom = header + 1.25 + len(steps) * ROW
    legend = [
        (BLUE_FILL, BLUE_FILL, "selected"),
        (RESULT_FILL, AMBER, "max |z|"),
        (CELL_FILL, CELL_EDGE, "p_value >= p_enter"),
    ]
    x = left
    for fill, edge, name in legend:
        canvas.box(x, bottom, 0.8, 0.6, fill=fill, edge=edge, radius=0.08)
        canvas.note(x + 1.1, bottom + 0.3, name, color=FAINT if name.startswith("p_") else INK)
        x += 1.6 + text_width(name, 8.8) + 0.6
    canvas.note(
        left,
        bottom + 1.5,
        "z = logistic_regression(X[[*selected, column]], target).loc[column, 'z']",
    )
    return canvas.fig


# Credit strategy


def figure_confusion_cutoff():
    """The four counts of confusion_table at one cut-off, and the cutoff_table columns they make."""
    current, _, target = _plotting._strategy_scores()
    cutoff = 0.1
    row = confusion_table(current, target, cutoffs=[cutoff]).iloc[0]
    approved = cutoff_table(current, target, cutoffs=[cutoff]).iloc[0]
    tp, fp, tn, fn = (
        int(row[c]) for c in ("true_positive", "false_positive", "true_negative", "false_negative")
    )
    if approved["approved"] != tn + fn or approved["approved_event"] != fn:
        raise AssertionError("the drawn counts do not match cutoff_table")

    canvas = Canvas(height=9.0)
    left, top, width, height = 4.6, 2.2, 5.0, 2.2
    canvas.call(0.6, 0.15, f"confusion_table(current, y, cutoffs=[{cutoff}])")
    canvas.text(left + width / 2, top - 0.5, f"current <= {cutoff}", size=9, bold=True, color=BLUE)
    canvas.text(
        left + width + GAP + width / 2,
        top - 0.5,
        f"current > {cutoff}",
        size=9,
        bold=True,
        color=RESULT_TEXT,
    )
    cells = {
        (0, 0): ("true_negative", tn, CELL_FILL, INK),
        (0, 1): ("false_positive", fp, CELL_FILL, INK),
        (1, 0): ("false_negative", fn, RESULT_FILL, RESULT_TEXT),
        (1, 1): ("true_positive", tp, EVENT_FILL, EVENT),
    }
    for (i, j), (name, count, fill, color) in cells.items():
        x = left + j * (width + GAP)
        y = top + i * (height + GAP)
        canvas.box(
            x, y, width, height, fill=fill, edge=RESULT_EDGE if fill == RESULT_FILL else CELL_EDGE
        )
        canvas.text(x + width / 2, y + 0.8, str(count), size=12, bold=True, color=color)
        canvas.note(x + width / 2, y + 1.6, name, ha="center", size=8.3)
    for i, label_ in enumerate(("y = 0", "y = 1")):
        canvas.text(
            left - 0.4,
            top + i * (height + GAP) + height / 2,
            label_,
            size=9,
            ha="right",
            bold=True,
            color=EVENT if i else INK,
        )
    middle = left + width + GAP / 2
    canvas.line(
        middle,
        top - 0.9,
        middle,
        top + 2 * height + GAP + 0.3,
        color=AMBER,
        linewidth=1.6,
        style="--",
        z=3,
    )
    total = len(target)
    canvas.notes(
        16.0,
        2.4,
        [
            f"approved             = {tn} + {fn} = {tn + fn}",
            f"approved_event       = false_negative = {fn}",
            f"rejected_event_share = recall = {tp} / {tp + fn} = {tp / (tp + fn):.3f}",
            f"precision            = {tp} / {tp + fp} = {tp / (tp + fp):.3f}",
            f"accuracy             = ({tp} + {tn}) / {total} = {(tp + tn) / total:.3f}",
        ],
        step=1.05,
        size=8.6,
    )
    canvas.note(16.0, 8.0, f"cutoff_table(current, y, cutoffs=[{cutoff}])", color=CHIP_TEXT)
    return canvas.fig


def figure_swap_set():
    """The four groups of swap_set, with all outcomes known and as they are observed."""
    current, new, target = _plotting._strategy_scores()
    approved = int((current <= 0.1).sum())
    new_cutoff = float(np.sort(new)[approved - 1])
    observed = target.where(current <= 0.1)
    tables = {
        "y": swap_set(current, new, target, current_cutoff=0.1, new_cutoff=new_cutoff),
        "observed": swap_set(current, new, observed, current_cutoff=0.1, new_cutoff=new_cutoff),
    }

    canvas = Canvas(height=11.4)
    width, height = 5.2, 2.5
    fills = {
        ("approve", "approve"): (CELL_FILL, CELL_EDGE),
        ("approve", "decline"): (EVENT_FILL, "#f3b4b4"),
        ("decline", "approve"): (TEAL_FILL, "#8fd3c9"),
        ("decline", "decline"): (CELL_FILL, CELL_EDGE),
    }
    for k, (name, table) in enumerate(tables.items()):
        left = 4.2 + k * 15.6
        top = 3.0
        canvas.call(left - 3.6, 0.15, f"swap_set(current, new, {name}, ...)", size=8.8)
        canvas.note(left + width + GAP / 2, top - 1.3, "new", ha="center", color=INK, bold=True)
        for j, decision in enumerate(("approve", "decline")):
            canvas.text(
                left + j * (width + GAP) + width / 2, top - 0.5, decision, size=9, bold=True
            )
        canvas.text(left - 2.6, top + height + GAP / 2, "current", size=9, bold=True, ha="center")
        for i, row in enumerate(("approve", "decline")):
            canvas.text(
                left - 0.4,
                top + i * (height + GAP) + height / 2,
                row,
                size=9,
                ha="right",
                bold=True,
            )
            for j, column in enumerate(("approve", "decline")):
                cell = table.loc[(row, column)]
                fill, edge = fills[(row, column)]
                x = left + j * (width + GAP)
                y = top + i * (height + GAP)
                canvas.box(x, y, width, height, fill=fill, edge=edge)
                canvas.text(x + width / 2, y + 0.75, f"{int(cell['total'])}", size=11, bold=True)
                known = int(cell["known"])
                rate = "NaN" if known == 0 else number(cell["event_rate"])
                canvas.note(
                    x + width / 2,
                    y + 1.7,
                    f"event_rate {rate}",
                    ha="center",
                    size=8.3,
                    color=FAINT if known == 0 else INK,
                )
    canvas.notes(
        0.6,
        9.0,
        [
            f"current_cutoff=0.1, new_cutoff={new_cutoff:.6f}",
            "observed = y.where(current <= 0.1)",
        ],
    )
    return canvas.fig


# Policy rules


def figure_rule_set_table():
    """The worked example of the rules guide: the first rule that hits each row gets the decline."""
    example = pd.DataFrame(
        {
            "high_debt": [1, 1, 0, 0, 0, 1, 0, 0],
            "young": [0, 1, 1, 0, 0, 0, 0, 0],
            "inquiries": [0, 0, 1, 1, 0, 0, 0, 0],
        }
    )
    example_target = [1, 1, 0, 1, 0, 0, 0, 0]
    table = rule_set_table(example, example_target)
    names = list(example.columns)
    hits = example.to_numpy(dtype=bool)
    first = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    if [int((first == j).sum()) for j in range(len(names))] != table["new_hits"].tolist():
        raise AssertionError("the first hits drawn do not match rule_set_table")
    # Leave out the orange tint, which would read as the amber of the first hits.
    styles = dict(zip(names, [CATEGORY[0], CATEGORY[1], CATEGORY[3]], strict=True))

    def rule_column(j: int, width: float) -> Column:
        tint, text = styles[names[j]]
        cells = [str(int(value)) for value in hits[:, j]]
        credited = first == j
        return Column(
            names[j],
            cells,
            width,
            fill=[
                RESULT_FILL if c else tint if h else CELL_FILL
                for c, h in zip(credited, hits[:, j], strict=True)
            ],
            color=[
                RESULT_TEXT if c else text if h else FAINT
                for c, h in zip(credited, hits[:, j], strict=True)
            ],
            bold=list(credited),
            edge=[RESULT_EDGE if c else None for c in credited],
            header_color=text,
        )

    canvas = Canvas(height=12.0)
    inputs = canvas.table(
        0.6,
        1.4,
        [
            Column("index", [str(i) for i in example.index], 1.6, color=MUTED),
            rule_column(0, 2.9),
            rule_column(1, 2.0),
            rule_column(2, 2.9),
            target_column(example_target, 2.2, name="target"),
        ],
    )
    x = 13.9
    canvas.call(x, 0.15, "rule_set_table(example, example_target)")
    out = canvas.table(
        x,
        1.4,
        [
            group_column("rule", table.index, 3.0, styles),
            result("new_hits", [str(v) for v in table["new_hits"]], 2.6),
            Column("new_event", [str(v) for v in table["new_event"]], 2.9),
            Column("approved", [str(v) for v in table["approved"]], 2.6),
            Column(
                "approved_event_rate", [number(v, 2) for v in table["approved_event_rate"]], 5.9
            ),
        ],
    )
    canvas.arrow((inputs.right + 0.2, out.rows[1]), (out.left - 0.2, out.rows[1]))
    canvas.notes(
        x,
        out.bottom + 0.9,
        [
            "new_hits  = rows whose first hit is the rule",
            "new_event = events among them",
            "approved  = rows that no rule up to it hits",
            "approved_event_rate = approved_event / approved",
        ],
    )
    approved = [str(i) for i in example.index[first == -1]]
    canvas.note(
        0.6,
        inputs.bottom + 0.7,
        f"approved after {names[-1]}: rows {', '.join(approved)}",
    )
    return canvas.fig


# Time-window features


def _features_example():
    """The events, applications and panel of the worked examples of the features guide."""
    events = pd.DataFrame(
        {
            "customer": ["a", "a", "a", "a", "a", "b", "b", "b"],
            "time": pd.to_datetime(
                [
                    "2023-11-20",
                    "2024-01-10",
                    "2024-02-29",
                    "2024-03-15",
                    "2024-03-31",
                    "2024-01-05",
                    "2024-03-20",
                    "2024-03-20",
                ]
            ),
            "amount": [300.0, 120.0, 80.0, 50.0, 200.0, 50.0, 40.0, 25.0],
            "kind": ["card", "card", "cash", "card", "card", "cash", "cash", "card"],
        }
    )
    applications = pd.DataFrame(
        {
            "customer": ["a", "b", "c"],
            "application_date": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-03-31"]),
        }
    )
    months = ["2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]
    panel = pd.DataFrame(
        {
            "customer": ["a"] * 7 + ["b"] * 7,
            "month": pd.PeriodIndex(
                months
                + ["2023-10", "2023-11", "2023-12", "2024-01", "2024-01", "2024-02", "2024-03"],
                freq="M",
            ),
            "status": pd.Categorical(
                ["current", "current", "1-29", "30-59", "30-59", "current", "current"]
                + ["current", "current", "current", "current", "1-29", "60-89", "90+"],
                categories=["current", "1-29", "30-59", "60-89", "90+"],
                ordered=True,
            ),
        }
    )
    return events, applications, panel


def _bracket(canvas, x0, x1, y, name, *, above=True):
    """An amber window [x0, x1): a closed tick on the left and an open circle on the right."""
    canvas.line(x0, y, x1, y, color=AMBER, linewidth=1.6, z=3)
    canvas.line(x0, y - 0.28, x0, y + 0.28, color=AMBER, linewidth=1.6, z=3)
    canvas.ax.plot(
        x1, y, "o", markersize=6, markerfacecolor="white", markeredgecolor=AMBER, zorder=4
    )
    if above:
        canvas.note(x0 + 0.3, y - 0.48, name, color=RESULT_TEXT, bold=True)
    else:
        canvas.note(x0 - 0.35, y, name, color=RESULT_TEXT, bold=True, ha="right")


def figure_window_features():
    """The worked example of the features guide: the events of each window and their sums."""
    events, applications, _ = _features_example()
    keys = {"id": "customer", "time": "time", "observation_time": "application_date"}
    windows = {"1m": pd.Timestamp("2024-02-29"), "3m": pd.Timestamp("2023-12-31")}
    t = pd.Timestamp("2024-03-31")
    features = window_features(
        events, applications, **keys, windows=["1M", "3M"], aggregations={"amount": ["sum"]}
    )
    # The sums again from the drawn rows: the events from the start of a window up to, but
    # not including, the application date.
    for name, start in windows.items():
        for i, customer in enumerate(applications["customer"]):
            own = events[events["customer"] == customer]
            inside = own[(own["time"] >= start) & (own["time"] < t)]
            if (len(inside), inside["amount"].sum()) != (
                features[f"events_{name}"].iloc[i],
                features[f"amount_sum_{name}"].iloc[i],
            ):
                raise AssertionError(f"the {name} window of {customer} does not match")

    canvas = Canvas(height=16.6)
    origin, scale = pd.Timestamp("2023-11-01"), 23.5 / 161
    left = 3.6

    def at(time):
        return left + (time - origin).days * scale

    rows = {customer: 4.1 + i * 1.45 for i, customer in enumerate(applications["customer"])}
    axis = 8.4
    tints = {"card": CATEGORY[0], "cash": CATEGORY[2]}
    for start in windows.values():
        canvas.line(at(start), 2.9, at(start), axis, color=AMBER, linewidth=0.9, style="--", z=1)
    _bracket(canvas, at(windows["3m"]), at(t), 1.1, "3m = [2023-12-31, 2024-03-31)")
    _bracket(canvas, at(windows["1m"]), at(t), 2.5, "1m = [2024-02-29, 2024-03-31)", above=False)
    canvas.line(at(t), 0.4, at(t), axis + 0.3, color=AMBER, linewidth=1.8, z=2)
    canvas.note(at(t) + 0.3, 0.45, "application_date", color=RESULT_TEXT, bold=True)

    canvas.text(left - 2.0, 3.25, "customer", size=9, bold=True)
    for customer, y in rows.items():
        canvas.text(left - 2.0, y, customer, bold=True)
        canvas.line(left - 0.6, y, at(pd.Timestamp("2024-04-10")), y, color=CELL_EDGE)
        own = events[events["customer"] == customer]
        repeated = own["time"].duplicated(keep=False)
        offset = 0
        for time, amount, kind, twin in zip(
            own["time"], own["amount"], own["kind"], repeated, strict=True
        ):
            used = windows["3m"] <= time < t
            fill, color = tints[kind] if used else ("white", FAINT)
            edge = color if used else FAINT
            dy = (offset - 0.5) * 0.62 if twin else 0.0
            offset += twin
            canvas.ax.plot(
                at(time),
                y + dy,
                "o",
                markersize=9,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=1.2,
                zorder=5,
            )
            text = f"{amount:g}"
            if twin:
                canvas.note(at(time) + 0.4, y + dy, text, color=color if used else FAINT)
            else:
                canvas.note(at(time), y - 0.6, text, ha="center", color=INK if used else FAINT)

    canvas.arrow((left - 0.6, axis), (at(pd.Timestamp("2024-04-10")), axis), color=MUTED)
    for month in pd.date_range("2023-11-01", "2024-04-01", freq="MS"):
        canvas.line(at(month), axis - 0.15, at(month), axis + 0.15, color=MUTED)
        canvas.note(at(month), axis + 0.6, month.strftime("%Y-%m"), ha="center", size=8.3)
    for i, (kind, (fill, color)) in enumerate(tints.items()):
        x = left + 17.0 + i * 4.4
        canvas.ax.plot(
            x, axis + 1.5, "o", markersize=8, markerfacecolor=fill, markeredgecolor=color
        )
        canvas.note(x + 0.45, axis + 1.5, f"kind = {kind}", color=INK)

    canvas.call(
        0.6,
        axis + 2.35,
        'window_features(events, applications, **keys, windows=["1M", "3M"], '
        'aggregations={"amount": ["sum"]})',
        size=8.6,
    )
    columns = [Column("customer", list(applications["customer"]), 2.6)]
    for name in windows:
        columns.append(result(f"events_{name}", [str(v) for v in features[f"events_{name}"]], 3.5))
        columns.append(
            result(
                f"amount_sum_{name}",
                [f"{v:.1f}" for v in features[f"amount_sum_{name}"]],
                4.6,
            )
        )
    canvas.table(4.4, axis + 3.8, columns)
    return canvas.fig


def _runs(flags) -> int:
    """The longest run of True in a list of True, False and None, where None ends a run."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def figure_panel_features():
    """The repayment example of the features guide: the months of each window and the runs."""
    _, applications, panel = _features_example()
    features = panel_features(
        panel,
        applications,
        id="customer",
        month="month",
        status="status",
        observation_time="application_date",
        windows=["3M", "6M"],
    )
    categories = list(panel["status"].cat.categories)
    bad = categories.index("30-59")
    months = pd.period_range("2023-09", "2024-03", freq="M")
    observed = pd.Period("2024-03", freq="M")
    # The worst status of each customer and month, as a position in the categories.
    worst = {}
    for customer, month, code in zip(
        panel["customer"], panel["month"], panel["status"].cat.codes, strict=True
    ):
        worst[customer, month] = max(worst.get((customer, month), -1), int(code))
    # The columns again from the drawn months: only the months before the observation month.
    for i, customer in enumerate(applications["customer"]):
        for name, k in (("3m", 3), ("6m", 6)):
            codes = [worst.get((customer, observed - j)) for j in range(k, 0, -1)]
            flags = [None if c is None else c >= bad for c in codes]
            expected = {
                "months": sum(c is not None for c in codes),
                "bad_months": sum(f is True for f in flags),
                "longest_bad_run": _runs(flags),
                "longest_good_run": _runs([None if f is None else not f for f in flags]),
            }
            for column, value in expected.items():
                if features[f"{column}_{name}"].iloc[i] != value:
                    raise AssertionError(f"{column}_{name} of {customer} does not match")

    canvas = Canvas(height=14.2)
    left, width, step, height, pitch = 2.6, 2.1, 2.24, 1.0, 1.25
    tints = dict(zip(categories, ramp(len(categories)), strict=True))
    top = 5.0

    def x_of(month):
        return left + (month - months[0]).n * step

    canvas.call(
        0.6,
        0.15,
        'panel_features(panel, applications, ..., windows=["3M", "6M"])',
        size=8.8,
    )
    end = x_of(observed) - GAP / 2
    _bracket(canvas, x_of(months[0]), end, 2.35, "6m = [2023-09, 2024-03)")
    _bracket(canvas, x_of(observed - 3), end, 3.55, "3m = [2023-12, 2024-03)", above=False)
    canvas.line(end, 1.75, end, top + 2 * pitch + height + 0.25, color=AMBER, linewidth=1.8, z=2)
    for month in months:
        canvas.text(x_of(month) + width / 2, top - 0.45, str(month), size=8.3, color=MUTED)
    for row, customer in enumerate(applications["customer"]):
        y = top + row * pitch
        canvas.text(left - 1.2, y + height / 2, customer, bold=True)
        for month in months:
            x = x_of(month)
            loans = panel[(panel["customer"] == customer) & (panel["month"] == month)]
            statuses = list(loans["status"])
            used = month < observed
            code = worst.get((customer, month))
            edge = EVENT if used and code is not None and code >= bad else None
            if not statuses:
                canvas.box(x, y, width, height, fill=CELL_FILL, edge=CELL_EDGE)
            elif len(statuses) == 1:
                fill, color = tints[statuses[0]]
                canvas.box(x, y, width, height, fill=fill, edge=edge, linewidth=1.6)
                canvas.text(
                    x + width / 2,
                    y + height / 2,
                    statuses[0],
                    size=8,
                    color=color if used else FAINT,
                )
            else:
                # Two loans in one month: the worse status counts.
                for k, status in enumerate(statuses):
                    fill, color = tints[status]
                    canvas.box(x, y + k * height / 2, width, height / 2, fill=fill, radius=0.08)
                    chosen = categories.index(status) == code
                    canvas.text(
                        x + width / 2,
                        y + (k + 0.5) * height / 2,
                        status,
                        size=7.2,
                        color=INK if chosen else FAINT,
                        bold=chosen,
                    )
                canvas.box(x, y, width, height, fill="none", edge=edge or CELL_EDGE, z=2)
            if not used:
                canvas.ax.add_patch(
                    FancyBboxPatch(
                        (x, y),
                        width,
                        height,
                        boxstyle="round,pad=0,rounding_size=0.14",
                        facecolor="none",
                        edgecolor=FAINT,
                        hatch="////",
                        linewidth=0.6,
                        zorder=3,
                    )
                )
    y = top + 3 * pitch + 0.55
    for i, status in enumerate(categories):
        fill, color = tints[status]
        x = left + i * 3.15
        canvas.box(x, y - 0.3, 0.6, 0.6, fill=fill, radius=0.08)
        canvas.note(x + 0.85, y, status, color=INK)
    canvas.box(left, y + 0.75, 0.6, 0.6, fill=CELL_FILL, edge=EVENT, radius=0.08, linewidth=1.6)
    canvas.note(left + 0.85, y + 1.05, f'status >= "{categories[bad]}"', color=INK)

    names = list(features.columns)
    in_3m = [name.endswith("_3m") for name in names]

    def values(i):
        cells = [
            label(v) if isinstance(v, str) or pd.isna(v) else f"{v:g}" for v in features.iloc[i]
        ]
        months_since = features["months_since_bad"].iloc[i]
        cells[-1] = label(months_since) if pd.isna(months_since) else f"{months_since:.1f}"
        return Column(
            applications["customer"].iloc[i],
            cells,
            1.75,
            fill=[
                MISSING_FILL if c == "NaN" else RESULT_FILL if amber else CELL_FILL
                for c, amber in zip(cells, in_3m, strict=True)
            ],
            edge=[RESULT_EDGE if amber else None for amber in in_3m],
            color=[FAINT if c == "NaN" else INK for c in cells],
        )

    canvas.table(
        19.4,
        1.2,
        [
            Column(
                "",
                names,
                6.2,
                fill="none",
                edge="none",
                color=[RESULT_TEXT if amber else INK for amber in in_3m],
            ),
            *(values(i) for i in range(len(applications))),
        ],
    )
    return canvas.fig
