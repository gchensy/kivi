"""Charts for the kivi documentation.

The plotting functions take kivi result tables. The ``figure_*`` recipes
recompute the examples shown in the pages and draw them; the pages call them
from ``.. plot::`` directives. Chart labels use column and parameter names so
that the same images serve the English and Chinese editions.

``python docs/_plotting.py`` refreshes the README images in docs/_static.
"""

import warnings
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter

from kivi.binning import bin_edges, chimerge_edges, monotonic_edges, tree_edges
from kivi.calibration import (
    assign_grades,
    calibration_table,
    fit_grade_pd,
    master_scale,
    prior_correction,
)
from kivi.datasets import Dataset, load_bank, make_applications, make_loan_panel
from kivi.metrics import gains_table
from kivi.reject_inference import fuzzy_augmentation
from kivi.selection import logistic_regression
from kivi.spec import fit_spec
from kivi.stability import psi_table
from kivi.strategy import cutoff_table
from kivi.univariate import profile_table
from kivi.vintage import vintage_table
from kivi.woe import woe_table, woe_transform

BLUE = "#2f5bd3"
LIGHT_BLUE = "#9fb4ee"
LIGHT_TEAL = "#a7e0da"
TEAL = "#14a397"
AMBER = "#d97706"
RED = "#dc2626"
GRAY = "#9ca3af"
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRAY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=[BLUE, TEAL, AMBER, RED]),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "kivi",
        }
    )


def _labels(index) -> list[str]:
    return ["NaN" if pd.isna(value) else str(value) for value in index]


def _bar_values(ax, x, values) -> None:
    size = 8.5 if len(x) <= 7 else 6.8
    for xi, value in zip(x, values, strict=True):
        value = round(float(value), 2) + 0.0
        ax.annotate(
            f"{value:+.2f}",
            (xi, value),
            xytext=(0, 3 if value >= 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=size,
            color=TEXT,
        )
    ax.margins(y=0.18)


def _ticks(ax, x, labels) -> None:
    if len(labels) > 6 or max(map(len, labels)) > 14:
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8.5)
    else:
        ax.set_xticks(x, labels)


def woe_chart(table: pd.DataFrame, title: str = ""):
    """Population share and event rate per bin (top) and WOE per bin (bottom)."""
    share, event_rate = table["total_share"], table["event_rate"]
    x = np.arange(len(table))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]}
    )
    top.bar(x, share, color=LIGHT_BLUE, width=0.6, label="total_share")
    top.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    top.set_ylabel("total_share")
    rate = top.twinx()
    rate.plot(x, event_rate, color=AMBER, marker="o", linewidth=2, label="event_rate")
    rate.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    rate.set_ylim(0, max(event_rate.max() * 1.25, 0.01))
    rate.set_ylabel("event_rate")
    rate.grid(False)
    rate.spines["right"].set_visible(True)
    handles = top.get_legend_handles_labels()[0] + rate.get_legend_handles_labels()[0]
    top.legend(
        handles,
        ["total_share", "event_rate"],
        loc="lower right",
        bbox_to_anchor=(1, 1.01),
        ncols=2,
        borderaxespad=0,
    )
    if title:
        top.set_title(title)
    colors = [BLUE if value >= 0 else TEAL for value in table["woe"]]
    bottom.bar(x, table["woe"], color=colors, width=0.6)
    bottom.axhline(0, color=GRAY, linewidth=1)
    _bar_values(bottom, x, table["woe"])
    bottom.set_ylabel("woe")
    _ticks(bottom, x, _labels(table.index))
    fig.tight_layout()
    return fig


def _woe_bars(ax, table: pd.DataFrame, title: str = "") -> None:
    x = np.arange(len(table))
    colors = [BLUE if value >= 0 else TEAL for value in table["woe"]]
    ax.bar(x, table["woe"], color=colors, width=0.65)
    ax.axhline(0, color=GRAY, linewidth=1)
    _bar_values(ax, x, table["woe"])
    ax.set_xticks(x, _labels(table.index), rotation=35, ha="right", fontsize=8.5)
    ax.set_title(f"{title}    IV = {table['iv'].sum():.3f}".strip(), fontsize=10)
    ax.set_ylabel("woe")


def woe_panels(tables: dict[str, pd.DataFrame], height: float = 3.2):
    """WOE bars for several groupings side by side, one panel per table."""
    fig, axes = plt.subplots(1, len(tables), figsize=(8, height), squeeze=False)
    for ax, (title, table) in zip(axes[0], tables.items(), strict=True):
        _woe_bars(ax, table, title)
    fig.tight_layout()
    return fig


def _method_axes():
    return plt.subplots(1, 2, figsize=(8, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})


def _grouped_woe(values: pd.Series, target: pd.Series, edges: np.ndarray) -> pd.DataFrame:
    return woe_table(pd.cut(values, edges), target).sort_index()


def _pair_chi2(table: pd.DataFrame) -> np.ndarray:
    """Chi-square of each adjacent pair of rows in a WOE table, from the 2x2 formula."""
    a, b = table["non_event"].to_numpy()[:-1], table["event"].to_numpy()[:-1]
    c, d = table["non_event"].to_numpy()[1:], table["event"].to_numpy()[1:]
    n = a + b + c + d
    return n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))


def chimerge_path(
    values: pd.Series,
    target: pd.Series,
    max_bins: int = 5,
    alpha: float = 0.05,
    initial_bins: int = 20,
):
    """Smallest adjacent chi-square as ChiMerge merges (left) and the resulting WOE (right)."""
    start = (
        len(
            chimerge_edges(
                values, target, max_bins=initial_bins, alpha=None, initial_bins=initial_bins
            )
        )
        - 1
    )
    bins = np.arange(start, 1, -1)
    smallest = [
        _pair_chi2(
            _grouped_woe(
                values,
                target,
                chimerge_edges(values, target, max_bins=k, alpha=None, initial_bins=initial_bins),
            )
        ).min()
        for k in bins
    ]
    edges = chimerge_edges(
        values, target, max_bins=max_bins, alpha=alpha, initial_bins=initial_bins
    )
    chosen = len(edges) - 1
    critical = NormalDist().inv_cdf(1 - alpha / 2) ** 2
    fig, (left, right) = _method_axes()
    left.plot(bins, smallest, color=BLUE, marker="o", markersize=4, linewidth=1.6)
    left.axhline(critical, color=AMBER, linestyle="--", linewidth=1.2, label=f"alpha = {alpha:g}")
    left.axvline(max_bins, color=GRAY, linestyle=":", linewidth=1.4, label=f"max_bins = {max_bins}")
    stop = smallest[list(bins).index(chosen)]
    left.plot([chosen], [stop], marker="o", markersize=9, color=RED, zorder=3)
    left.annotate(
        f"{chosen}",
        (chosen, stop),
        xytext=(8, -4),
        textcoords="offset points",
        color=RED,
        fontsize=9,
        fontweight="bold",
    )
    left.text(
        1.005,
        critical,
        f"{critical:.2f}",
        transform=left.get_yaxis_transform(),
        va="center",
        fontsize=8.5,
        color=AMBER,
    )
    left.set_yscale("symlog", linthresh=1)
    left.yaxis.set_major_locator(FixedLocator([0, 0.5, 1, 2, 5, 10, 20]))
    left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    left.yaxis.set_minor_locator(FixedLocator([]))
    left.set_ylim(0, max(smallest) * 1.3)
    left.set_xlim(start + 0.5, 1.5)
    left.set_xticks([k for k in bins if k % 3 == 0 or k in (chosen, max_bins)])
    left.set_xlabel("bins")
    left.set_ylabel(r"$\chi^2$")
    left.legend(loc="upper left")
    _woe_bars(right, _grouped_woe(values, target, edges))
    fig.tight_layout()
    return fig


def _cumulative_share(values: pd.Series, points) -> np.ndarray:
    observed = np.sort(values.dropna().to_numpy(dtype=float))
    return np.searchsorted(observed, points, side="right") / len(observed)


def tree_splits(
    values: pd.Series, target: pd.Series, max_bins: int = 5, min_bin_share: float = 0.05
):
    """Information gain of every first cut (left) and the WOE of the tree's bins (right)."""
    frame = pd.DataFrame({"value": values, "event": target}).dropna()
    counts = frame.groupby("value")["event"].agg(["size", "sum"])
    n, events = counts["size"].cumsum().to_numpy()[:-1], counts["sum"].cumsum().to_numpy()[:-1]
    total, total_events = len(frame), frame["event"].sum()

    def entropy(rate):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate)))

    gain = (
        entropy(total_events / total)
        - n / total * entropy(events / n)
        - (total - n) / total * entropy((total_events - events) / (total - n))
    )
    share = n / total
    edges = tree_edges(values, target, max_bins=max_bins, min_bin_share=min_bin_share)
    order = {}
    for k in range(2, len(edges)):
        for edge in tree_edges(values, target, max_bins=k, min_bin_share=min_bin_share)[1:-1]:
            order.setdefault(edge, len(order) + 1)
    fig, (left, right) = _method_axes()
    left.plot(share, gain, color=BLUE, linewidth=1.4)
    for low, high in ((0, min_bin_share), (1 - min_bin_share, 1)):
        left.axvspan(low, high, color=GRID, zorder=0)
    left.fill_between([], [], color=GRID, label=f"min_bin_share = {min_bin_share:g}")
    top = gain.max() * 1.3
    for edge, rank in order.items():
        x = _cumulative_share(values, [edge])[0]
        right_side = x < 0.75
        left.axvline(x, color=AMBER, linewidth=1.2, linestyle="--")
        left.annotate(
            f"#{rank}  {edge:g}",
            (x, top * (0.97 - 0.09 * (rank - 1))),
            xytext=(4 if right_side else -4, 0),
            textcoords="offset points",
            ha="left" if right_side else "right",
            va="top",
            fontsize=8.5,
            color=TEXT,
            bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"},
        )
    left.set_ylim(0, top)
    left.set_xlim(0, 1)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel(f"quantile ({values.name})")
    left.set_ylabel(r"$\Delta$ entropy")
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    _woe_bars(right, _grouped_woe(values, target, edges))
    fig.tight_layout()
    return fig


def monotonic_pooling(
    values: pd.Series, target: pd.Series, max_bins: int = 5, initial_bins: int = 20
):
    """Fine-bin WOE with the pooled monotonic WOE (left) and the final bins (right)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # monotonic_edges also drops tied quantiles
        fine = _grouped_woe(values, target, bin_edges(values, bins=initial_bins))
    pooled_edges = monotonic_edges(values, target, max_bins=None, initial_bins=initial_bins)
    pooled = _grouped_woe(values, target, pooled_edges)
    fig, (left, right) = _method_axes()
    widths = (fine["event"] + fine["non_event"]) / len(values.dropna())
    starts = np.r_[0, widths.cumsum().to_numpy()[:-1]]
    colors = [LIGHT_BLUE if value >= 0 else LIGHT_TEAL for value in fine["woe"]]
    left.bar(
        starts,
        fine["woe"],
        width=widths,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        label=f"initial_bins = {initial_bins}",
    )
    bounds = _cumulative_share(values, pooled_edges[1:-1])
    left.stairs(
        pooled["woe"],
        np.r_[0, bounds, 1],
        color=BLUE,
        linewidth=2.2,
        baseline=None,
        label="max_bins = None",
    )
    left.axhline(0, color=GRAY, linewidth=1)
    left.set_xlim(0, 1)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel(f"quantile ({values.name})")
    left.set_ylabel("woe")
    left.legend(loc="upper left")
    edges = monotonic_edges(values, target, max_bins=max_bins, initial_bins=initial_bins)
    _woe_bars(right, _grouped_woe(values, target, edges), f"max_bins = {max_bins}")
    fig.tight_layout()
    return fig


def binning_methods(values: pd.Series, bins: int = 5):
    """Distribution with quantile and uniform edges (left) and counts per bin (right)."""
    quantile = bin_edges(values, bins=bins)
    uniform = bin_edges(values, bins=bins, method="uniform")
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    upper = values.quantile(0.99)
    left.hist(values[values <= upper], bins=60, color=LIGHT_BLUE)
    for edge in quantile[1:-1]:
        left.axvline(edge, color=BLUE, linewidth=1.6)
    for edge in uniform[1:-1]:
        if edge <= upper:
            left.axvline(edge, color=AMBER, linewidth=1.6, linestyle="--")
    left.plot([], [], color=BLUE, label="quantile")
    left.plot([], [], color=AMBER, linestyle="--", label="uniform")
    left.legend(loc="upper right")
    left.set_xlabel(values.name)
    left.set_ylabel("count")
    x = np.arange(1, bins + 1)
    q_counts = pd.cut(values, quantile).value_counts(sort=False).to_numpy()
    u_counts = pd.cut(values, uniform).value_counts(sort=False).to_numpy()
    right.bar(
        x - 0.2,
        np.pad(q_counts, (0, bins - len(q_counts))),
        width=0.4,
        color=BLUE,
        label="quantile",
    )
    right.bar(x + 0.2, u_counts, width=0.4, color=AMBER, label="uniform")
    right.set_xticks(x)
    right.set_xlabel("bin")
    right.set_ylabel("count")
    right.legend(loc="upper right")
    fig.tight_layout()
    return fig


def iv_ranking(iv: pd.Series, limit: float = 0.65):
    """Horizontal IV bars with the usual 0.02 / 0.1 / 0.3 reading bands."""
    iv = iv.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(iv) + 0.9))
    colors = [GRAY if v < 0.02 else TEAL if v < 0.1 else BLUE if v < 0.5 else AMBER for v in iv]
    y = np.arange(len(iv))
    ax.barh(y, np.minimum(iv, limit), color=colors, height=0.65)
    for yi, value in zip(y, iv, strict=True):
        label = f"{value:.3f}" + ("  →" if value > limit else "")
        ax.annotate(
            label,
            (min(value, limit), yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=AMBER if value > limit else TEXT,
        )
    for threshold in (0.02, 0.1, 0.3):
        ax.axvline(threshold, color=GRAY, linestyle="--", linewidth=1)
        ax.annotate(
            f"{threshold:g}", (threshold, len(iv) - 0.4), ha="center", fontsize=8.5, color=MUTED
        )
    ax.set_yticks(y, iv.index)
    ax.set_xlim(0, limit + 0.08)
    ax.set_xlabel("iv")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def psi_panels(tables: dict[str, pd.DataFrame]):
    """Reference and current shares per bin for several comparisons."""
    fig, axes = plt.subplots(1, len(tables), figsize=(8, 3.3), sharey=True, squeeze=False)
    for ax, (title, table) in zip(axes[0], tables.items(), strict=True):
        x = np.arange(len(table))
        ax.bar(
            x - 0.2, table["reference_share"], width=0.4, color=LIGHT_BLUE, label="reference_share"
        )
        ax.bar(x + 0.2, table["current_share"], width=0.4, color=BLUE, label="current_share")
        ax.set_xticks(x, _labels(table.index), rotation=35, ha="right", fontsize=8.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(f"{title}    PSI = {table['psi'].sum():.3f}", fontsize=10)
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def psi_monitor(psi: pd.Series, sizes: pd.Series):
    """PSI per period, coloured by the 0.1 / 0.25 conventions, with sample sizes."""
    x = np.arange(len(psi))
    colors = [BLUE if v < 0.1 else AMBER if v < 0.25 else RED for v in psi]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x, psi, color=colors, width=0.62)
    for threshold, color in ((0.1, AMBER), (0.25, RED)):
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1)
        ax.text(
            1.005,
            threshold,
            f"{threshold:g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=color,
        )
    ax.set_xticks(
        x, [f"{period}\nn={n}" for period, n in zip(psi.index, sizes, strict=True)], fontsize=8.5
    )
    ax.set_ylabel("psi")
    ax.set_ylim(0, max(psi.max() * 1.18, 0.3))
    fig.tight_layout()
    return fig


def gains_chart(table: pd.DataFrame):
    """Event rate per score band (left) and the cumulative gains curve (right)."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    x = np.arange(len(table))
    overall = table["event"].sum() / table["total"].sum()
    left.bar(x, table["event_rate"], color=LIGHT_BLUE, width=0.6, label="event_rate")
    left.axhline(overall, color=AMBER, linestyle="--", linewidth=1.4, label="lift = 1")
    for xi, lift, rate in zip(x, table["lift"], table["event_rate"], strict=True):
        left.annotate(
            f"lift {lift:.2f}",
            (xi, rate),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_ylabel("event_rate")
    left.margins(y=0.15)
    left.set_xticks(x, _labels(table.index), rotation=25, ha="right", fontsize=8.5)
    left.legend(loc="upper right")
    share = np.r_[0, table["cum_total_share"]]
    gains = np.r_[0, table["cum_event_share"]]
    right.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    right.plot(share, gains, color=BLUE, marker="o", linewidth=2)
    largest = int(np.argmax(table["ks"].to_numpy()))
    right.annotate(
        f"ks {table['ks'].iloc[largest]:.3f}",
        (share[largest + 1], gains[largest + 1]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8.5,
        color=MUTED,
    )
    for axis in (right.xaxis, right.yaxis):
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.set_xlim(0, 1)
    right.set_ylim(0, 1.02)
    right.set_xlabel("cum_total_share")
    right.set_ylabel("cum_event_share")
    right.grid(True, axis="both")
    fig.tight_layout()
    return fig


def reliability_chart(tables: dict[str, pd.DataFrame]):
    """Observed event rate against mean predicted probability per group."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    limit = max(max(t["event_rate"].max(), t["mean_probability"].max()) for t in tables.values())
    limit = min(1.0, limit * 1.1)
    ax.plot([0, limit], [0, limit], color=GRAY, linestyle="--", linewidth=1)
    for (label, table), color in zip(tables.items(), [BLUE, AMBER], strict=False):
        ax.plot(
            table["mean_probability"],
            table["event_rate"],
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=4,
            label=label,
        )
        ax.scatter(
            table["mean_probability"],
            table["event_rate"],
            s=20 + 300 * table["total"] / table["total"].sum(),
            color=color,
            zorder=3,
        )
        for grade, row in table.iterrows():
            ax.annotate(
                str(grade),
                (row["mean_probability"], row["event_rate"]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("mean_probability")
    ax.set_ylabel("event_rate")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def vintage_curves(table: pd.DataFrame, title: str = ""):
    """Cumulative bad rate of each cohort by month on book."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(table)))
    for (cohort, row), color in zip(table.iterrows(), shades, strict=True):
        observed = row.dropna()
        ax.plot(observed.index, observed.to_numpy(), color=color, linewidth=1.8, label=str(cohort))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("month_on_book")
    ax.set_xlim(0, table.columns.max())
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(loc="lower right", ncols=2, fontsize=8.5)
    fig.tight_layout()
    return fig


def cutoff_curves(tables: dict[str, pd.DataFrame], marks: dict[str, tuple[str, pd.DataFrame]]):
    """Bad rate of the approved book against the approval rate, one curve per score.

    ``marks`` maps a curve to a label and the one-row ``cutoff_table`` of a
    cut-off to highlight on it.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (name, table), color in zip(tables.items(), [GRAY, BLUE], strict=False):
        ax.plot(
            table["approval_rate"],
            table["approved_event_rate"],
            color=color,
            linewidth=2,
            label=name,
        )
        if name in marks:
            label, row = marks[name]
            point = (row["approval_rate"].iloc[0], row["approved_event_rate"].iloc[0])
            ax.scatter(*point, color=color, s=36, zorder=3)
            ax.annotate(
                label, point, xytext=(8, -14), textcoords="offset points", fontsize=8.5, color=color
            )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("approval_rate")
    ax.set_ylabel("approved_event_rate")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def factor_sensitivity(results: pd.DataFrame, true_rate: float, known_pd: float):
    """Inferred bad rate and refitted mean PD of the rejects for each ``factor``."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for column, color, reference, label in (
        ("inferred", AMBER, true_rate, 'results["known"]["inferred"]'),
        ("reject_pd", BLUE, known_pd, 'results["known"]["reject_pd"]'),
    ):
        ax.plot(results.index, results[column], color=color, marker="o", linewidth=2, label=column)
        ax.axhline(reference, color=color, linestyle="--", linewidth=1.2, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("factor")
    ax.set_xticks(results.index)
    ax.grid(True, axis="both")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def grade_fit(counts: pd.DataFrame, fitted: pd.Series, grid: pd.Series):
    """Observed default rate per grade against the PDs of ``fit_grade_pd``, on a log scale."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(fitted))
    ax.plot(x, grid, color=GRAY, linestyle="--", marker=".", linewidth=1.2, label="np.geomspace")
    ax.plot(x, fitted, color=BLUE, marker="o", linewidth=2, label="fit_grade_pd")
    observed = (counts["total"] > 0).to_numpy()
    rate = (counts["event"] / counts["total"].where(counts["total"] > 0)).to_numpy()
    ax.scatter(
        x[observed],
        rate[observed],
        s=20 + 400 * counts["total"][observed] / counts["total"].sum(),
        color=AMBER,
        zorder=3,
        label="event_rate",
    )
    for xi, total in zip(x[observed], counts["total"][observed], strict=True):
        ax.annotate(
            f"total {total:g}",
            (xi, rate[xi]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=MUTED,
        )
    ax.set_yscale("log")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
    ax.set_xticks(x, _labels(fitted.index))
    ax.set_xlabel("grade")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def missing_profile(profile: pd.DataFrame):
    """Share of missing values and their lift for the columns that have any."""
    shown = profile[profile["missing"] > 0].sort_values("missing_share")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 0.42 * len(shown) + 1.2), sharey=True)
    y = np.arange(len(shown))
    left.barh(y, shown["missing_share"], color=LIGHT_BLUE, height=0.6)
    colors = [AMBER if lift > 1 else TEAL for lift in shown["missing_lift"]]
    right.barh(y, shown["missing_lift"], color=colors, height=0.6)
    right.axvline(1, color=GRAY, linestyle="--", linewidth=1)
    for ax, column, text in (
        (left, "missing_share", "{:.1%}"),
        (right, "missing_lift", "{:.2f}"),
    ):
        for yi, value in zip(y, shown[column], strict=True):
            ax.annotate(
                text.format(value),
                (value, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
            )
        ax.set_xlabel(column)
        ax.margins(x=0.2)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_yticks(y, shown.index)
    fig.tight_layout()
    return fig


# Recipes for the documentation pages ---------------------------------------


def _samples():
    data = Dataset.bank_data()
    return data, data.iloc[:3000], data.iloc[3000:]


def figure_quickstart_woe():
    _, reference, _ = _samples()
    groups = pd.cut(reference["age"], bin_edges(reference["age"], bins=5))
    return woe_chart(woe_table(groups, reference["target"]).sort_index(), "age")


def figure_binning_methods():
    _, reference, _ = _samples()
    return binning_methods(reference["balance"])


def figure_custom_edges():
    _, reference, _ = _samples()
    age_edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])
    pdays_edges = np.array([-np.inf, -1, 100, 200, 300, np.inf])
    return woe_panels(
        {
            "age": woe_table(pd.cut(reference["age"], age_edges), reference["target"]).sort_index(),
            "pdays": woe_table(
                pd.cut(reference["pdays"], pdays_edges), reference["target"]
            ).sort_index(),
        },
        height=3.4,
    )


def figure_coarse_classing():
    _, reference, _ = _samples()
    fine = bin_edges(reference["balance"], bins=10)
    coarse = fine[[0, 3, 7, 10]]
    return woe_panels(
        {
            "bins=10": woe_table(
                pd.cut(reference["balance"], fine), reference["target"]
            ).sort_index(),
            "fine[[0, 3, 7, 10]]": woe_table(
                pd.cut(reference["balance"], coarse), reference["target"]
            ).sort_index(),
        },
        height=3.6,
    )


def figure_chimerge():
    _, reference, _ = _samples()
    return chimerge_path(reference["balance"], reference["target"])


def figure_tree():
    _, reference, _ = _samples()
    return tree_splits(reference["balance"], reference["target"])


def figure_monotonic():
    _, reference, _ = _samples()
    return monotonic_pooling(reference["balance"], reference["target"])


def figure_woe_job():
    _, reference, _ = _samples()
    table = woe_table(reference["job"], reference["target"]).sort_values("woe")
    return woe_chart(table, "job")


def figure_iv_ranking():
    _, reference, _ = _samples()
    profile = profile_table(reference.drop(columns="target"), reference["target"])
    return iv_ranking(profile["iv"])


def figure_missing_profile():
    development = load_bank().iloc[:3000]
    candidates = development.drop(columns="target").replace("unknown", np.nan)
    return missing_profile(profile_table(candidates, development["target"]))


def figure_psi_shift():
    data, reference, current = _samples()
    edges = bin_edges(reference["age"], bins=5)
    before = pd.cut(reference["age"], edges)
    single = data[data["marital"] == "single"]
    return psi_panels(
        {
            "data.iloc[3000:]": psi_table(before, pd.cut(current["age"], edges)).sort_index(),
            "marital == 'single'": psi_table(before, pd.cut(single["age"], edges)).sort_index(),
        }
    )


def monthly_psi(data: pd.DataFrame, reference: pd.DataFrame, column: str = "age"):
    edges = bin_edges(reference[column], bins=5)
    before = pd.cut(reference[column], edges)
    months = [m for m in MONTHS if m in set(data["month"])]
    psi = pd.Series(
        {
            m: psi_table(before, pd.cut(data.loc[data["month"] == m, column], edges))["psi"].sum()
            for m in months
        }
    )
    sizes = data["month"].value_counts().reindex(months)
    return psi, sizes


def figure_psi_monitor():
    data, reference, _ = _samples()
    return psi_monitor(*monthly_psi(data, reference))


def _woe_score_samples():
    """The four-variable WOE score of the metrics guide, on both samples."""
    _, development, validation = _samples()

    def to_groups(frame):
        return pd.DataFrame(
            {
                "poutcome": frame["poutcome"],
                "contact": frame["contact"],
                "housing": frame["housing"],
                "age": pd.cut(frame["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf]),
            }
        )

    dev_groups = to_groups(development)
    tables = {c: woe_table(dev_groups[c], development["target"]) for c in dev_groups}

    def woe_score(frame):
        groups = to_groups(frame)
        return pd.DataFrame({c: woe_transform(groups[c], tables[c]) for c in groups}).sum(axis=1)

    return woe_score(development), woe_score(validation), validation["target"]


def figure_gains():
    dev_score, val_score, target = _woe_score_samples()
    edges = bin_edges(dev_score, bins=5).round(2)
    return gains_chart(gains_table(val_score, target, bins=edges))


def _calibration_samples():
    """The samples and PDs of the calibration guide."""
    _, development, monitoring = _samples()
    age_edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])

    def to_groups(frame):
        groups = frame[["poutcome", "contact", "housing"]].copy()
        groups["age"] = pd.cut(frame["age"], age_edges)
        return groups

    dev_groups = to_groups(development)
    tables = {c: woe_table(dev_groups[c], development["target"]) for c in dev_groups}
    coefficients = pd.Series({"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714})

    def predict(frame):
        groups = to_groups(frame)
        woe = pd.DataFrame({c: woe_transform(groups[c], tables[c]) for c in groups})
        return 1 / (1 + np.exp(2.028 - woe @ coefficients))

    return development, monitoring, predict(development), predict(monitoring)


def figure_grade_fit():
    development, _, dev_prob, _ = _calibration_samples()
    scale = master_scale(np.geomspace(0.02, 0.64, 6), labels=list("ABCDEF"))
    counts = calibration_table(assign_grades(dev_prob, scale), development["target"], dev_prob)
    counts = counts.reindex(scale.index, fill_value=0)
    fitted = fit_grade_pd(counts["event"], counts["total"])
    return grade_fit(counts, fitted, scale["probability"])


def figure_calibration():
    _, monitoring, _, probability = _calibration_samples()
    low = prior_correction(probability, sample_rate=0.116, population_rate=0.06)
    pd_edges = [0, 0.05, 0.08, 0.12, 0.2, 0.4, 1]

    def grade_table(p):
        grades = pd.cut(p, pd_edges, labels=list("ABCDEF"), include_lowest=True)
        return calibration_table(grades, monitoring["target"], p)

    return reliability_chart(
        {
            "probability": grade_table(probability),
            "population_rate=0.06": grade_table(low),
        }
    )


def _strategy_scores():
    """The PDs of the current and the new scorecard of the strategy guide, with the target."""
    development, applicants = load_bank().iloc[:3000], load_bank().iloc[3000:]

    def to_groups(frame):
        groups = frame[["poutcome", "contact", "housing", "loan", "marital"]].copy()
        groups["age"] = pd.cut(frame["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf])
        return groups

    dev_groups = to_groups(development)
    tables = {c: woe_table(dev_groups[c], development["target"]) for c in dev_groups}

    def to_woe(frame, columns):
        groups = to_groups(frame)
        return pd.DataFrame({c: woe_transform(groups[c], tables[c]) for c in columns})

    def fit_pd(columns):
        coef = logistic_regression(to_woe(development, columns), development["target"])["coef"]
        log_odds = coef["intercept"] + to_woe(applicants, columns) @ coef[columns]
        return 1 / (1 + np.exp(-log_odds))

    four = ["poutcome", "contact", "housing", "age"]
    return fit_pd(four), fit_pd([*four, "loan", "marital"]), applicants["target"]


def figure_cutoff_curves():
    current, new, target = _strategy_scores()
    approved = int((current <= 0.1).sum())
    new_cutoff = float(np.sort(new)[approved - 1])
    grid = np.linspace(0.05, 1, 40)
    tables = {
        name: cutoff_table(score, target, cutoffs=np.unique(np.quantile(score, grid)))
        for name, score in (("current", current), ("new", new))
    }
    marks = {
        "current": ("current_cutoff=0.1", cutoff_table(current, target, cutoffs=[0.1])),
        "new": (f"new_cutoff={new_cutoff:.3f}", cutoff_table(new, target, cutoffs=[new_cutoff])),
    }
    return cutoff_curves(tables, marks)


def figure_reject_factor():
    applications = make_applications(rng=0)
    population = applications[~applications["policy_decline"]]
    accepts = population[population["approved"]]
    rejects = population[~population["approved"]]
    features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment"]
    edges = {c: bin_edges(population[c], bins=5) for c in ["age", "income", "debt_to_income"]}
    edges["inquiries"] = [-np.inf, 0, 1, 2, np.inf]
    spec = fit_spec(accepts[features], accepts["target"], bins=edges)

    def predict(model, frame):
        coef = model["coef"]
        log_odds = coef["intercept"] + spec.woe_transform(frame) @ coef.drop("intercept")
        return 1 / (1 + np.exp(-log_odds))

    model = logistic_regression(spec.woe_transform(accepts), accepts["target"])
    reject_pd = predict(model, rejects)
    results = {}
    for factor in [1, 1.25, 1.5, 1.75, 2, 2.5]:
        inferred = fuzzy_augmentation(reject_pd, factor=factor)
        frame = pd.concat([accepts[features], rejects.loc[inferred.index, features]])
        target = pd.concat([accepts["target"], inferred["target"]])
        weights = pd.concat([pd.Series(1.0, index=accepts.index), inferred["weight"]])
        refit = logistic_regression(spec.woe_transform(frame), target, weights=weights)
        results[factor] = {
            "inferred": np.average(inferred["target"], weights=inferred["weight"]),
            "reject_pd": predict(refit, rejects).mean(),
        }
    known = logistic_regression(spec.woe_transform(population), population["target"])
    return factor_sensitivity(
        pd.DataFrame(results).T, rejects["target"].mean(), predict(known, rejects).mean()
    )


def figure_vintage():
    panel = make_loan_panel(rng=0)
    ever_90 = vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
    )
    return vintage_curves(ever_90.iloc[::3], "status == '90+'")


README_FIGURES = {
    "readme-woe.svg": figure_quickstart_woe,
    "readme-iv.svg": figure_iv_ranking,
    "readme-psi.svg": figure_psi_monitor,
    "readme-gains.svg": figure_gains,
    "readme-vintage.svg": figure_vintage,
    "readme-strategy.svg": figure_cutoff_curves,
    "readme-reject.svg": figure_reject_factor,
}


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    apply_style()
    static = Path(__file__).resolve().parent / "_static"
    for name, recipe in README_FIGURES.items():
        figure = recipe()
        figure.savefig(static / name, format="svg", metadata={"Date": None})
        plt.close(figure)
        print(f"wrote docs/_static/{name}")
