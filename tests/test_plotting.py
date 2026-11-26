"""kivi.plotting draws the values of the tables it is given and checks its inputs."""

import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kivi import plotting
from kivi.binning import bin_edges
from kivi.calibration import calibration_table
from kivi.datasets import load_bank, make_loan_panel
from kivi.lgd import lgd_table
from kivi.lifetime import survival_table
from kivi.metrics import gains_table, ks, roc_table
from kivi.monitoring import monitoring_table, stability_table
from kivi.rules import rule_set_table
from kivi.stability import psi_table
from kivi.strategy import cutoff_table
from kivi.vintage import transition_matrix, vintage_table
from kivi.woe import woe_table

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

SCORE = pd.Series([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.9, 0.3, 0.5])
TARGET = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
BLUE, AMBER = "#2f5bd3", "#d97706"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def no_new_figure():
    """Fail a test that leaves a figure behind, as a chart that raises must not."""
    before = plt.get_fignums()
    yield
    assert plt.get_fignums() == before


def woe_example():
    groups = pd.Series(["a", "b", "b", "c", "c", "c", None, None, "a", "b"])
    return woe_table(groups, TARGET)


def correlation_example():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 4, 3], "c": [4, 3, 2, 1]})
    return frame.corr()


def roc_example():
    return roc_table(SCORE, TARGET)


def gains_example():
    return gains_table(SCORE, TARGET, bins=3)


def calibration_example():
    grade = pd.Series(np.where(SCORE <= 0.3, "A", np.where(SCORE <= 0.6, "B", "C")))
    return calibration_table(grade, TARGET, SCORE)


def cutoff_example():
    return cutoff_table(SCORE, TARGET, cutoffs=[0.05, 0.3, 0.6, 1.0])


def psi_example():
    return psi_table(pd.Series(list("aabbcc")), pd.Series(list("abbccc")))


PERIOD = pd.Series(["2025-01"] * 4 + ["2025-02"] * 4 + ["2025-03"] * 2)


def monitoring_example():
    target = TARGET.copy()
    target[8:] = 1  # 2025-03 has only events, so its gini is missing
    return monitoring_table(SCORE, PERIOD, target, reference_period="2025-01", bins=2)


def stability_example():
    groups = pd.DataFrame({"x": list("aabbabbbab"), "y": list("ccddcdcddd")})
    return stability_table(groups, PERIOD, reference_period="2025-01")


def vintage_example():
    loan = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    cohort = ["2024-01"] * 5 + ["2024-02"] * 3 + ["2024-03"] * 2
    month_on_book = [1, 2, 3, 1, 2, 1, 2, 3, 2, 3]
    bad = [0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
    return vintage_table(loan, cohort, month_on_book, bad)


def survival_example():
    return survival_table([1, 2, 3, 3, 4, 5], [1, 0, 1, 1, 0, 0])


def competing_example():
    return survival_table([1, 2, 3, 3, 4, 5], [1, 0, 0, 1, 0, 0], competing=[0, 0, 1, 0, 0, 0])


def transition_example():
    before = ["a", "b", "a", "c", "b", "c", "a", "c"]
    after = ["a", "b", "b", "c", "c", "c", "a", "b"]
    return transition_matrix(before, after)


def rule_set_example():
    rules = pd.DataFrame(
        {"high_debt": [True, False, True, False, False], "young": [False, False, True, True, False]}
    )
    return rule_set_table(rules, [1, 0, 1, 0, 0])


def profit_example():
    loss = [0, 0, 0, 5, 0, 5]
    return cutoff_table(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0, 0, 0, 1, 0, 1],
        cutoffs=[0.05, 0.2, 0.3, 0.6],
        loss=loss,
        revenue=[1] * 6,
    )


def lgd_example():
    return lgd_table(["a", "a", "b", "b", "c"], [0.0, 0.5, 1.0, 1.2, -0.1])


CASES = {
    "woe_chart": woe_example,
    "correlation_chart": correlation_example,
    "roc_chart": roc_example,
    "precision_recall_chart": roc_example,
    "gains_chart": gains_example,
    "distribution_chart": gains_example,
    "calibration_chart": calibration_example,
    "cutoff_chart": cutoff_example,
    "psi_chart": psi_example,
    "monitoring_chart": monitoring_example,
    "vintage_chart": vintage_example,
    "survival_chart": survival_example,
    "transition_chart": transition_example,
    "rule_set_chart": rule_set_example,
    "profit_chart": profit_example,
    "lgd_chart": lgd_example,
}


def drawn_axes(result):
    """The Axes a chart function drew on; woe_chart returns the bars and the line."""
    return result[0] if isinstance(result, tuple) else result


def test_every_chart_is_covered():
    assert sorted(CASES) == sorted(plotting.__all__)


@pytest.mark.parametrize("name", plotting.__all__)
def test_draws_on_a_new_figure_without_ax(name):
    before = set(plt.get_fignums())
    ax = drawn_axes(getattr(plotting, name)(CASES[name]()))
    assert isinstance(ax, matplotlib.axes.Axes)
    assert set(plt.get_fignums()) - before == {ax.figure.number}
    assert isinstance(
        ax.figure.get_layout_engine(), matplotlib.layout_engine.ConstrainedLayoutEngine
    )


@pytest.mark.parametrize("name", plotting.__all__)
def test_draws_on_the_given_ax(name):
    figure, ax = plt.subplots()
    before = plt.get_fignums()
    assert drawn_axes(getattr(plotting, name)(CASES[name](), ax=ax)) is ax
    assert plt.get_fignums() == before


@pytest.mark.parametrize("name", plotting.__all__)
def test_leaves_the_table_and_rcparams_alone(name):
    table = CASES[name]()
    copy = table.copy()
    params = matplotlib.rcParams.copy()
    getattr(plotting, name)(table)
    pd.testing.assert_frame_equal(table, copy)
    assert matplotlib.rcParams == params


@pytest.mark.parametrize("name", plotting.__all__)
def test_rejects_an_ax_that_is_not_an_axes(name, no_new_figure):
    with pytest.raises(ValueError, match="ax must be a matplotlib Axes, got str"):
        getattr(plotting, name)(CASES[name](), ax="left")


@pytest.mark.parametrize("name", plotting.__all__)
def test_asks_to_install_the_plot_extra_without_matplotlib(name, monkeypatch):
    table = CASES[name]()
    ax = matplotlib.figure.Figure().add_subplot()
    # A new figure needs pyplot; drawing on a given ax needs only matplotlib itself.
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib.axes", None)
    with pytest.raises(ImportError, match=r'pip install "kivi\[plot\]"'):
        getattr(plotting, name)(table)
    with pytest.raises(ImportError, match=r'pip install "kivi\[plot\]"'):
        getattr(plotting, name)(table, ax=ax)


def run_python(code):
    """Run `code` in a fresh interpreter and return what it prints."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_module_leaves_matplotlib_unimported():
    code = (
        "import sys, kivi.plotting\n"
        "print([name for name in sys.modules if name.split('.')[0] == 'matplotlib'])"
    )
    assert run_python(code) == "[]"


DRAW_ON_GIVEN_AXES = """
import sys
import matplotlib
from matplotlib.figure import Figure
import pandas as pd
from kivi import plotting
from kivi.calibration import calibration_table
from kivi.metrics import gains_table, roc_table
from kivi.monitoring import monitoring_table
from kivi.stability import psi_table
from kivi.strategy import cutoff_table
from kivi.vintage import vintage_table, transition_matrix
from kivi.woe import woe_table
from kivi.lgd import lgd_table
from kivi.lifetime import survival_table
from kivi.rules import rule_set_table

score = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6]
target = [0, 0, 1, 1, 0, 1]
groups = pd.Series(list("aabbcc"))
tables = {
    "woe_chart": woe_table(groups, pd.Series(target)),
    "correlation_chart": pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}).corr(),
    "roc_chart": roc_table(score, target),
    "precision_recall_chart": roc_table(score, target),
    "gains_chart": gains_table(score, target, bins=2),
    "distribution_chart": gains_table(score, target, bins=2),
    "calibration_chart": calibration_table(groups, target, score),
    "cutoff_chart": cutoff_table(score, target, cutoffs=[0.3, 1.0]),
    "psi_chart": psi_table(groups, pd.Series(list("abbccc"))),
    "monitoring_chart": monitoring_table(
        score, ["1", "1", "1", "2", "2", "2"], target, reference_period="1", bins=2
    ),
    "vintage_chart": vintage_table([1, 1, 2], ["a", "a", "b"], [1, 2, 1], [0, 1, 0]),
    "survival_chart": survival_table([1, 2, 3], [1, 0, 1]),
    "transition_chart": transition_matrix(list("aab"), list("abb")),
    "rule_set_chart": rule_set_table(pd.DataFrame({"r": [True, False, False]})),
    "profit_chart": cutoff_table(score, target, cutoffs=[0.3, 1.0], loss=[1] * 6, revenue=[2] * 6),
    "lgd_chart": lgd_table(groups, [0.1, 0.5, 1.0, 0.0, 0.3, 0.9]),
}
for name, table in tables.items():
    getattr(plotting, name)(table, ax=Figure().add_subplot())
print("matplotlib.pyplot" in sys.modules, len(tables))
"""


def test_drawing_on_a_given_ax_leaves_pyplot_unimported():
    assert run_python(DRAW_ON_GIVEN_AXES) == f"False {len(plotting.__all__)}"


TABLES = {
    "woe_chart": ("kivi.woe.woe_table", "iv"),
    "roc_chart": ("kivi.metrics.roc_table", "true_positive_rate"),
    "precision_recall_chart": ("kivi.metrics.roc_table", "precision"),
    "gains_chart": ("kivi.metrics.gains_table", "ks"),
    "distribution_chart": ("kivi.metrics.gains_table or kivi.woe.woe_table", "event"),
    "calibration_chart": ("kivi.calibration.calibration_table", "mean_probability"),
    "cutoff_chart": ("kivi.strategy.cutoff_table", "approved_event_rate"),
    "psi_chart": ("kivi.stability.psi_table", "psi"),
    "monitoring_chart": ("kivi.monitoring.monitoring_table", "psi"),
    "survival_chart": ("kivi.lifetime.survival_table", "cumulative_pd"),
    "rule_set_chart": ("kivi.rules.rule_set_table", "approved"),
    "profit_chart": ("kivi.strategy.cutoff_table", "profit"),
    "lgd_chart": ("kivi.lgd.lgd_table", "mean"),
}


@pytest.mark.parametrize("name", TABLES)
def test_rejects_a_table_that_is_not_a_dataframe(name, no_new_figure):
    source, _ = TABLES[name]
    with pytest.raises(ValueError, match=f"table must be a DataFrame from {source}, got list"):
        getattr(plotting, name)([[0.1, 0.2]])


@pytest.mark.parametrize("name", TABLES)
def test_names_the_missing_columns(name, no_new_figure):
    source, column = TABLES[name]
    table = CASES[name]().drop(columns=column)
    with pytest.raises(
        ValueError, match=rf"table must have the columns \[.*\] of {source}"
    ) as info:
        getattr(plotting, name)(table)
    assert str(info.value).endswith(f"missing ['{column}']")


@pytest.mark.parametrize("name", TABLES)
def test_rejects_an_empty_table(name, no_new_figure):
    with pytest.raises(ValueError, match="table must have at least one row"):
        getattr(plotting, name)(CASES[name]().iloc[:0])


@pytest.mark.parametrize("name", TABLES)
def test_rejects_a_column_that_is_not_numeric(name, no_new_figure):
    _, column = TABLES[name]
    table = CASES[name]().astype({column: str})
    with pytest.raises(ValueError, match=f"{column} of table must be real and numeric, got dtype"):
        getattr(plotting, name)(table)


def covered(legend, axes):
    """Whether `legend` covers a bar or a point drawn on any of `axes`."""
    renderer = legend.figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    for ax in axes:
        if any(box.overlaps(patch.get_window_extent(renderer)) for patch in ax.patches):
            return True
        for line in ax.get_lines():
            x, y = line.get_transform().transform(line.get_xydata()).T
            if ((x > box.x0) & (x < box.x1) & (y > box.y0) & (y < box.y1)).any():
                return True
    return False


def test_legends_cover_no_data_in_report_panels():
    data = load_bank()
    development, validation = data.iloc[:3000], data.iloc[3000:]
    edges = bin_edges(development["age"], bins=5)
    reference = pd.cut(development["age"], edges)
    panel = make_loan_panel(rng=0)
    vintage = vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
    )
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    twins = [plotting.woe_chart(woe_table(reference, development["target"]), ax=axes[0, 0])]
    twins += [
        plotting.woe_chart(
            woe_table(development["poutcome"], development["target"]).sort_values("woe"),
            line="woe",
            ax=axes[0, 1],
        )
    ]
    single = data.loc[data["marital"] == "single", "age"]
    plotting.psi_chart(psi_table(reference, pd.cut(single, edges)), ax=axes[1, 0])
    plotting.vintage_chart(vintage, ax=axes[1, 1])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for bars, line in twins:
        assert not covered(line.get_legend(), [bars, line])
    assert not covered(axes[1, 0].get_legend(), [axes[1, 0]])
    # The entries stand side by side when they fit in the Axes and are stacked otherwise.
    legends = [(bars, line.get_legend()) for bars, line in twins]
    legends.append((axes[1, 0], axes[1, 0].get_legend()))
    for ax, legend in legends:
        box = legend.get_window_extent(renderer)
        assert ax.bbox.x0 <= box.x0 and box.x1 <= ax.bbox.x1
    columns = [
        len({round(text.get_window_extent(renderer).x0) for text in legend.get_texts()})
        for _, legend in legends
    ]
    assert columns == [1, 2, 1]
    legend_box = axes[1, 1].get_legend().get_window_extent(figure.canvas.get_renderer())
    assert not legend_box.overlaps(axes[1, 1].bbox)
    # The PSI chart of a sample like the reference, whose bars are all about as high.
    figure, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    plotting.psi_chart(psi_table(reference, pd.cut(validation["age"], edges)), ax=ax)
    figure.canvas.draw()
    assert not covered(ax.get_legend(), [ax])


# woe_chart -------------------------------------------------------------------


def test_woe_chart_draws_the_shares_and_the_event_rate():
    table = woe_example()
    bars, line = plotting.woe_chart(table)
    assert [bar.get_height() for bar in bars.patches] == pytest.approx(table["total_share"])
    assert [label.get_text() for label in bars.get_xticklabels()] == ["a", "b", "c", "NaN"]
    (curve,) = line.get_lines()
    np.testing.assert_array_equal(curve.get_xdata(), np.arange(4))
    np.testing.assert_allclose(curve.get_ydata(), table["event_rate"])
    assert (bars.get_ylabel(), line.get_ylabel(), bars.get_xlabel()) == (
        "total_share",
        "event_rate",
        "bin",
    )
    assert matplotlib.colors.same_color(bars.yaxis.label.get_color(), BLUE)
    assert matplotlib.colors.same_color(line.yaxis.label.get_color(), AMBER)
    assert line.get_shared_x_axes().joined(bars, line)
    assert line.get_ylim()[0] == 0
    legend = line.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["total_share", "event_rate"]
    assert legend.get_title().get_text() == f"iv = {table['iv'].sum():.3f}"
    assert bars.yaxis.get_major_formatter()(0.125) == "12.5%"


def test_woe_chart_draws_the_woe_and_its_zero_on_the_right_axis():
    table = woe_example()
    _, line = plotting.woe_chart(table, line="woe")
    zero, curve = line.get_lines()
    np.testing.assert_array_equal(zero.get_ydata(), [0, 0])
    assert zero.get_linestyle() == ":"
    assert matplotlib.colors.same_color(zero.get_color(), curve.get_color())
    np.testing.assert_allclose(curve.get_ydata(), table["woe"])
    assert line.get_ylabel() == "woe"
    assert line.get_ylim()[0] < table["woe"].min()


def test_woe_chart_keeps_zero_in_view_when_every_woe_is_negative():
    table = pd.DataFrame({"total_share": [0.5, 0.5], "woe": [-0.4, -0.2], "iv": [0.1, 0.1]})
    _, line = plotting.woe_chart(table, line="woe")
    low, high = line.get_ylim()
    assert low < -0.4 and high > 0


@pytest.mark.parametrize("column", ["event_rate", "woe"])
def test_woe_chart_keeps_a_range_when_every_value_is_zero(column):
    table = pd.DataFrame({"total_share": [0.0, 0.0], column: [0.0, 0.0], "iv": [0.0, 0.0]})
    bars, line = plotting.woe_chart(table, line=column)
    assert bars.get_ylim() == (0, 1) and line.get_ylim() == (0, 1)


@pytest.mark.parametrize("line", ["iv", ["woe"], None])
def test_woe_chart_rejects_another_line(line, no_new_figure):
    with pytest.raises(ValueError, match="line must be 'event_rate' or 'woe', got"):
        plotting.woe_chart(woe_example(), line=line)


def shares(labels):
    """A WOE table of equal shares with the given group labels."""
    size = len(labels)
    return pd.DataFrame(
        {"total_share": np.full(size, 1 / size), "event_rate": np.full(size, 0.1), "iv": 0.0},
        index=labels,
    )


def test_slants_tick_labels_that_do_not_fit_side_by_side():
    intervals = ["(-inf, 32.0]", "(32.0, 36.0]", "(36.0, 42.0]", "(42.0, 51.0]", "(51.0, inf]"]
    cases = {
        "short": (shares(["a", "b"]), None),
        "intervals": (shares(intervals), None),
        "intervals in a narrow panel": (shares(intervals), plt.subplots(1, 2)[1][0]),
        "long": (shares(["a category with a long name"] * 3 + ["b"]), None),
        "many": (shares([f"group {i}" for i in range(20)]), None),
    }
    angles = {
        name: plotting.woe_chart(table, ax=ax)[0].get_xticklabels()[0].get_rotation()
        for name, (table, ax) in cases.items()
    }
    assert angles == {
        "short": 0,
        "intervals": 0,
        "intervals in a narrow panel": 35,
        "long": 35,
        "many": 35,
    }


# correlation_chart -----------------------------------------------------------


def test_correlation_chart_draws_every_value():
    matrix = correlation_example()
    ax = plotting.correlation_chart(matrix)
    np.testing.assert_allclose(ax.images[0].get_array(), matrix.to_numpy())
    assert ax.images[0].get_clim() == (-1, 1)
    texts = [text.get_text() for text in ax.texts]
    assert texts == [f"{value:.2f}" for value in matrix.to_numpy().ravel()]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b", "c"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b", "c"]
    # Dark cells get white text.
    assert ax.texts[0].get_color() == "white" and ax.texts[1].get_color() != "white"


def test_correlation_chart_leaves_missing_cells_empty():
    matrix = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    ax = plotting.correlation_chart(matrix)
    assert [text.get_text() for text in ax.texts] == ["1.00", "1.00"]


def test_correlation_chart_accepts_rounding_beyond_one():
    matrix = pd.DataFrame(
        [[1 + 1e-12, -1 - 1e-12], [-1 - 1e-12, 1.0]], index=["a", "b"], columns=["a", "b"]
    )
    ax = plotting.correlation_chart(matrix)
    assert [text.get_text() for text in ax.texts] == ["1.00", "-1.00", "-1.00", "1.00"]


def test_correlation_chart_sizes_a_new_figure_to_the_variables():
    small = plotting.correlation_chart(correlation_example())
    names = [f"x{i}" for i in range(10)]
    large = plotting.correlation_chart(pd.DataFrame(np.eye(10), index=names, columns=names))
    assert tuple(small.figure.get_size_inches()) == (4.0, 3.2)
    assert tuple(large.figure.get_size_inches()) == pytest.approx((7.6, 7.0))


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1.0]], "matrix must be a DataFrame such as X.corr(), got list"),
        (pd.DataFrame(), "matrix must have at least one row"),
        (
            pd.DataFrame([[1.0, 0.5]], index=["a"], columns=["a", "b"]),
            "matrix must have the same labels in its index and columns, such as X.corr(), "
            "got ['a'] and ['a', 'b']",
        ),
        (
            pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["b", "a"], columns=["a", "b"]),
            "same labels in its index and columns",
        ),
        (
            pd.DataFrame({"a": [1.0, "x"], "b": [0.5, 1.0]}, index=["a", "b"]),
            "column 'a' must be real and numeric, got dtype object",
        ),
        (
            pd.DataFrame([[4.0, 3.0], [3.0, 9.0]], index=["a", "b"], columns=["a", "b"]),
            "matrix must hold correlations between -1 and 1, got 4.0",
        ),
        (
            pd.DataFrame([[1.0, -1.5], [-1.5, 1.0]], index=["a", "b"], columns=["a", "b"]),
            "matrix must hold correlations between -1 and 1, got -1.5",
        ),
    ],
)
def test_correlation_chart_rejects_another_matrix(matrix, message, no_new_figure):
    with pytest.raises(ValueError, match=re.escape(message)):
        plotting.correlation_chart(matrix)


# roc_chart -------------------------------------------------------------------


def test_roc_chart_draws_the_curve_and_marks_the_ks():
    table = roc_example()
    ax = plotting.roc_chart(table)
    diagonal, curve, gap = ax.get_lines()
    np.testing.assert_array_equal(diagonal.get_xydata(), [[0, 0], [1, 1]])
    np.testing.assert_allclose(curve.get_xdata(), table["false_positive_rate"])
    np.testing.assert_allclose(curve.get_ydata(), table["true_positive_rate"])
    largest = (table["true_positive_rate"] - table["false_positive_rate"]).abs().argmax()
    x, y = table["false_positive_rate"].iloc[largest], table["true_positive_rate"].iloc[largest]
    np.testing.assert_allclose(gap.get_xydata(), [[x, x], [x, y]])
    (text,) = ax.texts
    assert text.get_text() == f"ks = {ks(SCORE, TARGET):.3f}"
    assert gap.get_color() == curve.get_color() == text.get_color()
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("false_positive_rate", "true_positive_rate")
    assert ax.get_legend() is None


def test_roc_chart_marks_the_largest_absolute_gap_of_a_reversed_score():
    ax = plotting.roc_chart(roc_table(-SCORE, TARGET))
    assert ax.texts[0].get_text() == f"ks = {ks(SCORE, TARGET):.3f}"


def test_roc_chart_compares_curves_on_one_ax():
    ax = plotting.roc_chart(roc_example(), label="score")
    plotting.roc_chart(roc_table(SCORE.rank(), TARGET), ks=False, label="rank", ax=ax)
    first, second = ax.get_lines()[1], ax.get_lines()[4]
    assert first.get_color() != second.get_color()
    assert len(ax.texts) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["score", "rank"]


# precision_recall_chart ------------------------------------------------------


def test_precision_recall_chart_draws_the_rows_that_flag_observations():
    table = roc_example()
    assert np.isinf(table.index[0]) and np.isnan(table["precision"].iloc[0])
    ax = plotting.precision_recall_chart(table)
    baseline, curve = ax.get_lines()
    recall, precision = table["true_positive_rate"].iloc[1:], table["precision"].iloc[1:]
    # Steps from a recall of 0, each at the precision of the row it ends on.
    assert curve.get_drawstyle() == "steps-pre"
    np.testing.assert_allclose(curve.get_xdata(), np.r_[0, recall])
    np.testing.assert_allclose(curve.get_ydata(), np.r_[precision.iloc[0], precision])
    rate = TARGET.mean()
    np.testing.assert_allclose(baseline.get_ydata(), [rate, rate])
    assert baseline.get_linestyle() == "--"
    assert matplotlib.colors.same_color(baseline.get_color(), "#9ca3af")
    # The last row flags everyone, so the curve ends on the line of the event rate.
    assert (curve.get_xdata()[-1], curve.get_ydata()[-1]) == (1, pytest.approx(rate))
    (text,) = ax.texts
    assert text.get_text() == "event_rate = 0.5"
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("true_positive_rate", "precision")
    assert ax.yaxis.get_major_formatter()(0.125) == "12.5%"
    assert ax.get_legend() is None


def test_precision_recall_chart_steps_enclose_the_average_precision():
    rng = np.random.default_rng(0)
    # Few distinct scores, so that rows add events and non-events together.
    score = rng.integers(0, 8, size=200)
    target = rng.binomial(1, 0.1 + 0.08 * score)
    table = roc_table(score, target).iloc[1:]
    recall, precision = table["true_positive_rate"], table["precision"]
    average_precision = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    (curve,) = plotting.precision_recall_chart(roc_table(score, target)).get_lines()[1:]
    x, y = matplotlib.cbook.pts_to_prestep(curve.get_xdata(), curve.get_ydata())
    # Vertical steps have no width, so each horizontal step adds its width times its height.
    assert np.sum(np.diff(x) * y[1:]) == pytest.approx(average_precision, rel=1e-12)
    # Straight lines between the rows would enclose more.
    linear = np.sum(np.diff(curve.get_xdata()) * (curve.get_ydata()[1:] + curve.get_ydata()[:-1]))
    assert linear / 2 > average_precision


def test_precision_recall_chart_draws_the_line_of_each_event_rate_once():
    ax = plotting.precision_recall_chart(roc_example(), label="score")
    plotting.precision_recall_chart(roc_table(SCORE.rank(), TARGET), label="rank", ax=ax)
    baseline, first, second = ax.get_lines()
    assert first.get_color() != second.get_color()
    assert [text.get_text() for text in ax.texts] == ["event_rate = 0.5"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["score", "rank"]
    # Another sample with another event rate gets a line of its own.
    plotting.precision_recall_chart(roc_table(SCORE[:5], TARGET[:5]), ax=ax)
    assert [text.get_text() for text in ax.texts] == ["event_rate = 0.5", "event_rate = 0.4"]
    np.testing.assert_allclose(ax.get_lines()[3].get_ydata(), [0.4, 0.4])


def test_precision_recall_chart_shows_a_small_event_rate_to_three_digits():
    target = np.r_[np.ones(3), np.zeros(1279)]
    score = np.linspace(1, 0, target.size)
    ax = plotting.precision_recall_chart(roc_table(score, target))
    assert ax.texts[0].get_text() == "event_rate = 0.00234"


def test_precision_recall_chart_needs_a_row_with_a_precision(no_new_figure):
    with pytest.raises(ValueError, match="table must have a row with a precision"):
        plotting.precision_recall_chart(roc_example().iloc[:1])


@pytest.mark.parametrize("chart", [plotting.roc_chart, plotting.gains_chart])
def test_rejects_a_ks_that_is_not_a_boolean(chart, no_new_figure):
    table = roc_example() if chart is plotting.roc_chart else gains_example()
    with pytest.raises(ValueError, match="ks must be True or False, got 'yes'"):
        chart(table, ks="yes")


# gains_chart -----------------------------------------------------------------


def test_gains_chart_draws_both_curves_and_the_largest_ks():
    table = gains_example()
    ax = plotting.gains_chart(table)
    diagonal, event, non_event, gap = ax.get_lines()
    share = np.r_[0, table["cum_total_share"]]
    np.testing.assert_allclose(event.get_xdata(), share)
    np.testing.assert_allclose(event.get_ydata(), np.r_[0, table["cum_event_share"]])
    np.testing.assert_allclose(non_event.get_xdata(), share)
    np.testing.assert_allclose(non_event.get_ydata(), np.r_[0, table["cum_non_event_share"]])
    assert (event.get_linestyle(), non_event.get_linestyle()) == ("-", "--")
    assert event.get_color() == non_event.get_color() == gap.get_color()
    row = table.iloc[table["ks"].argmax()]
    np.testing.assert_allclose(
        gap.get_xydata(),
        [
            [row["cum_total_share"], row["cum_non_event_share"]],
            [row["cum_total_share"], row["cum_event_share"]],
        ],
    )
    assert ax.texts[0].get_text() == f"ks = {table['ks'].max():.3f}"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "cum_event_share",
        "cum_non_event_share",
    ]
    assert (ax.get_xlabel(), ax.get_ylabel()) == (
        "cum_total_share",
        "cum_event_share, cum_non_event_share",
    )


def test_gains_chart_compares_scores_on_one_ax():
    ax = plotting.gains_chart(gains_example(), label="score")
    plotting.gains_chart(gains_table(SCORE.rank(), TARGET, bins=3), ks=False, label="rank", ax=ax)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "cum_event_share (score)",
        "cum_non_event_share (score)",
        "cum_event_share (rank)",
        "cum_non_event_share (rank)",
    ]
    first, second = ax.get_lines()[1], ax.get_lines()[5]
    assert first.get_color() != second.get_color()
    assert len(ax.texts) == 1


# distribution_chart ----------------------------------------------------------


def test_distribution_chart_draws_the_raw_shares_of_each_row():
    table = woe_example()
    # Group a has no event, so the smoothed event_share of the table is not 0.
    assert table["event"].iloc[0] == 0 and table["event_share"].iloc[0] > 0
    ax = plotting.distribution_chart(table)
    bars = ax.patches
    non_event = table["non_event"] / table["non_event"].sum()
    event = table["event"] / table["event"].sum()
    assert [bar.get_height() for bar in bars[:4]] == pytest.approx(non_event)
    assert [bar.get_height() for bar in bars[4:]] == pytest.approx(event)
    assert bars[4].get_height() == 0
    centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    assert centers == pytest.approx([-0.2, 0.8, 1.8, 2.8, 0.2, 1.2, 2.2, 3.2])
    assert matplotlib.colors.same_color(bars[0].get_facecolor(), BLUE)
    assert matplotlib.colors.same_color(bars[4].get_facecolor(), "#dc2626")
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b", "c", "NaN"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("bin", "non_event_share, event_share")
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["non_event_share", "event_share"]
    assert ax.yaxis.get_major_formatter()(0.125) == "12.5%"


def test_distribution_chart_draws_the_bands_of_a_gains_table_in_its_order():
    table = gains_table(SCORE, TARGET, bins=[-np.inf, 0.3, 0.6, 2.0, np.inf])
    # The highest band comes first, and it is empty.
    assert table["total"].tolist() == [0, 3, 4, 3]
    ax = plotting.distribution_chart(table)
    heights = [bar.get_height() for bar in ax.patches]
    assert heights == pytest.approx([0, 0.2, 0.2, 0.6, 0, 0.4, 0.6, 0])
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "(2.0, inf]",
        "(0.6, 2.0]",
        "(0.3, 0.6]",
        "(-inf, 0.3]",
    ]


def test_distribution_chart_legend_covers_no_bar():
    data = load_bank().iloc[:3000]
    groups = pd.cut(data["age"], bin_edges(data["age"], bins=5))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    plotting.distribution_chart(woe_table(groups, data["target"]), ax=axes[0])
    # The highest band of duration holds the largest share of events, under the legend.
    plotting.distribution_chart(gains_table(data["duration"], data["target"]), ax=axes[1])
    figure.canvas.draw()
    for ax in axes:
        assert not covered(ax.get_legend(), [ax])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("non_event", -1.0, "non_event of table must contain finite, non-negative counts"),
        ("event", np.inf, "event of table must contain finite, non-negative counts"),
        ("event", np.nan, "event of table must contain finite, non-negative counts"),
    ],
)
def test_distribution_chart_rejects_a_count_that_is_not_a_count(
    column, value, message, no_new_figure
):
    table = gains_example().astype({column: float})
    table.iloc[0, table.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match=message):
        plotting.distribution_chart(table)


@pytest.mark.parametrize("column", ["non_event", "event"])
def test_distribution_chart_needs_both_classes(column, no_new_figure):
    table = gains_example().assign(**{column: 0})
    with pytest.raises(ValueError, match=f"^{column} of table must contain a positive count"):
        plotting.distribution_chart(table)


# calibration_chart -----------------------------------------------------------


def test_calibration_chart_draws_the_rates_on_square_axes():
    table = calibration_example()
    ax = plotting.calibration_chart(table, label="model")
    diagonal, curve = ax.get_lines()
    np.testing.assert_allclose(curve.get_xdata(), table["mean_probability"])
    np.testing.assert_allclose(curve.get_ydata(), table["event_rate"])
    upper = 1.1 * max(table["mean_probability"].max(), table["event_rate"].max())
    assert ax.get_xlim() == ax.get_ylim() == pytest.approx((0, min(1, upper)))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("mean_probability", "event_rate")
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["model"]


def test_calibration_chart_keeps_earlier_curves_in_view():
    table = pd.DataFrame({"mean_probability": [0.1, 0.4], "event_rate": [0.2, 0.3]})
    ax = plotting.calibration_chart(table)
    plotting.calibration_chart(table / 4, ax=ax)
    assert ax.get_xlim() == pytest.approx((0, 0.44))
    assert ax.get_legend() is None


def test_calibration_chart_keeps_a_range_when_every_rate_is_zero():
    table = pd.DataFrame({"mean_probability": [0.0, 0.0], "event_rate": [0.0, 0.0]})
    assert plotting.calibration_chart(table).get_xlim() == (0, 1)


# cutoff_chart ----------------------------------------------------------------


def test_cutoff_chart_draws_the_approved_event_rate():
    table = cutoff_example()
    ax = plotting.cutoff_chart(table, label="current")
    (curve,) = ax.get_lines()
    np.testing.assert_allclose(curve.get_xdata(), table["approval_rate"])
    np.testing.assert_allclose(curve.get_ydata(), table["approved_event_rate"])
    assert np.isnan(table["approved_event_rate"].iloc[0])
    assert ax.get_xlim() == (0, 1)
    assert ax.get_ylim() == pytest.approx((0, 1.1 * table["approved_event_rate"].max()))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["current"]


def test_cutoff_chart_keeps_a_range_without_events():
    table = pd.DataFrame({"approval_rate": [0.5, 1.0], "approved_event_rate": [0.0, 0.0]})
    ax = plotting.cutoff_chart(table)
    assert ax.get_ylim() == (0, 1)
    assert ax.get_legend() is None


# psi_chart -------------------------------------------------------------------


def test_psi_chart_draws_both_shares_of_each_group():
    table = psi_example()
    ax = plotting.psi_chart(table)
    bars = ax.patches
    assert [bar.get_height() for bar in bars[:3]] == pytest.approx(table["reference_share"])
    assert [bar.get_height() for bar in bars[3:]] == pytest.approx(table["current_share"])
    centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    assert centers == pytest.approx([-0.2, 0.8, 1.8, 0.2, 1.2, 2.2])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b", "c"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("bin", "reference_share, current_share")
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["reference_share", "current_share"]
    assert legend.get_title().get_text() == f"psi = {table['psi'].sum():.3f}"


# monitoring_chart ------------------------------------------------------------


def test_monitoring_chart_draws_a_line_by_default():
    report = monitoring_example()
    assert np.isnan(report["gini"].iloc[-1])
    for column in ["psi", "gini"]:
        ax = plotting.monitoring_chart(report, column=column)
        (line,) = ax.get_lines()
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
        np.testing.assert_allclose(line.get_ydata(), report[column])
        assert len(ax.patches) == 0
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("period", column)
        assert ax.get_legend() is None
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "2025-01",
            "2025-02",
            "2025-03",
        ]


def test_monitoring_chart_draws_bars_on_request():
    report = monitoring_example()
    ax = plotting.monitoring_chart(report, column="total", kind="bar")
    assert [bar.get_height() for bar in ax.patches] == pytest.approx(report["total"])
    assert len(ax.get_lines()) == 0


def test_monitoring_chart_draws_the_variables_of_a_stability_table():
    stability = stability_example()
    assert stability.columns.name == "variable"
    ax = None
    for column in stability:
        ax = plotting.monitoring_chart(stability, column=column, label=column, ax=ax)
    first, second = ax.get_lines()
    np.testing.assert_allclose(second.get_ydata(), stability["y"])
    assert first.get_color() != second.get_color()
    assert ax.get_ylabel() == "psi"
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["x", "y"]
    ax.figure.canvas.draw()
    assert not legend.get_window_extent(ax.figure.canvas.get_renderer()).overlaps(ax.bbox)


def test_monitoring_chart_marks_the_levels():
    ax = plotting.monitoring_chart(monitoring_example(), kind="bar", levels=[0.1, 0.25, 0.5, 1])
    levels = [line.get_ydata()[0] for line in ax.get_lines()]
    assert levels == [0.1, 0.25, 0.5, 1]
    assert [text.get_text() for text in ax.texts] == ["0.1", "0.25", "0.5", "1"]
    colors = [line.get_color() for line in ax.get_lines()]
    assert len(set(colors[:3])) == 3 and colors[3] == colors[2]
    single = plotting.monitoring_chart(monitoring_example(), column="gini", levels=0.4)
    assert [text.get_text() for text in single.texts] == ["0.4"]


@pytest.mark.parametrize("levels", ["a", True, np.inf, [0.1, None], [[0.1]]])
def test_monitoring_chart_rejects_levels_that_are_not_finite_numbers(levels, no_new_figure):
    with pytest.raises(ValueError, match="levels must be finite numbers, got"):
        plotting.monitoring_chart(monitoring_example(), levels=levels)


@pytest.mark.parametrize("kind", ["bars", None, ["line"]])
def test_monitoring_chart_rejects_another_kind(kind, no_new_figure):
    with pytest.raises(ValueError, match="kind must be 'line' or 'bar', got"):
        plotting.monitoring_chart(monitoring_example(), kind=kind)


@pytest.mark.parametrize("column", [["psi"], 1, None])
def test_monitoring_chart_rejects_a_column_that_is_not_a_name(column, no_new_figure):
    with pytest.raises(ValueError, match="column must be a column name, got"):
        plotting.monitoring_chart(monitoring_example(), column=column)


def test_monitoring_chart_names_a_missing_column(no_new_figure):
    with pytest.raises(ValueError, match=r"missing \['auc_change'\]"):
        plotting.monitoring_chart(monitoring_example(), column="auc_change")


# vintage_chart ---------------------------------------------------------------


def test_vintage_chart_draws_the_observed_months_of_each_cohort():
    table = vintage_example()
    ax = plotting.vintage_chart(table)
    lines = ax.get_lines()
    assert len(lines) == len(table)
    for line, (_, row) in zip(lines, table.iterrows(), strict=True):
        observed = row.dropna()
        np.testing.assert_array_equal(line.get_xdata(), observed.index.to_numpy(dtype=float))
        np.testing.assert_allclose(line.get_ydata(), observed.to_numpy())
    # The cohort of 2024-03 has no month 1.
    np.testing.assert_array_equal(lines[2].get_xdata(), [2, 3])
    darkness = [-sum(matplotlib.colors.to_rgb(line.get_color())) for line in lines]
    assert darkness == sorted(darkness)
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["2024-01", "2024-02", "2024-03"]
    assert legend.get_title().get_text() == "cohort"
    assert ax.get_xlabel() == "month_on_book"
    assert ax.get_ylim()[0] == 0


def test_vintage_chart_names_eight_cohorts_from_the_first_to_the_last():
    table = pd.DataFrame(
        np.tile([0.0, 0.01, 0.02], (24, 1)),
        index=pd.Index([f"c{i:02d}" for i in range(24)], name="cohort"),
        columns=[1, 2, 3],
    )
    ax = plotting.vintage_chart(table)
    assert len(ax.get_lines()) == 24
    legend = ax.get_legend()
    names = [text.get_text() for text in legend.get_texts()]
    assert names == ["c00", "c03", "c07", "c10", "c13", "c16", "c20", "c23"]
    # Each entry has the color of its cohort's curve.
    for handle, name in zip(legend.legend_handles, names, strict=True):
        assert matplotlib.colors.same_color(
            handle.get_color(), ax.get_lines()[int(name[1:])].get_color()
        )


def test_vintage_chart_draws_an_unnamed_table():
    table = pd.DataFrame([[0.0, 0.1, 0.2]], columns=[1, 2, 3])
    ax = plotting.vintage_chart(table)
    assert ax.get_xlabel() == ""
    assert ax.get_legend().get_title().get_text() == ""
    assert ax.get_lines()[0].get_color() == pytest.approx(matplotlib.colormaps["Blues"](0.95))


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ([[0.1]], "table must be a DataFrame from kivi.vintage.vintage_table, got list"),
        (pd.DataFrame(columns=[1, 2]), "table must have at least one row"),
        (
            pd.DataFrame([[0.1, 0.2]], columns=["one", "two"]),
            "table must have one numeric column per month on book, got the column 'one'",
        ),
        (
            pd.DataFrame({1: [0.1], 2: ["x"]}),
            "month 2 of table must be real and numeric, got dtype",
        ),
    ],
)
def test_vintage_chart_rejects_another_table(table, message, no_new_figure):
    with pytest.raises(ValueError, match=re.escape(message)):
        plotting.vintage_chart(table)


def steps(start, values):
    """The y values of a survival curve: its start, the table and the last value repeated."""
    values = np.asarray(values, dtype=float)
    return np.r_[start, values, values[-1]]


def test_survival_chart_draws_the_steps_of_the_table():
    table = survival_example()
    ax = plotting.survival_chart(table)
    (line,) = ax.get_lines()
    assert line.get_drawstyle() == "steps-post"
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(line.get_ydata(), steps(0.0, table["cumulative_pd"]))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("period", "cumulative_pd")
    assert ax.get_legend() is None


def test_survival_chart_draws_the_survival_from_one():
    table = survival_example()
    ax = plotting.survival_chart(table, column="survival", label="men")
    (line,) = ax.get_lines()
    np.testing.assert_array_equal(line.get_ydata(), steps(1.0, table["survival"]))
    assert ax.get_ylim() == (0, 1)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["men"]


def test_survival_chart_draws_both_causes_with_competing_risks():
    table = competing_example()
    ax = plotting.survival_chart(table)
    event, competing = ax.get_lines()
    np.testing.assert_array_equal(event.get_ydata(), steps(0.0, table["cumulative_pd"]))
    # Period 1: one event of 6; period 3: one event and one competing of 4 left, S = 5/6 * ...
    previous = np.r_[1.0, table["survival"].to_numpy()[:-1]]
    expected = np.cumsum(previous * table["competing"] / table["at_risk"])
    np.testing.assert_allclose(competing.get_ydata(), steps(0.0, expected))
    assert competing.get_ydata()[-1] > 0
    # The two causes and the survival add up to one.
    np.testing.assert_allclose(
        event.get_ydata()[1:-1] + competing.get_ydata()[1:-1] + table["survival"], 1.0
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "cumulative_pd",
        "competing",
    ]
    labeled = plotting.survival_chart(table, label="high")
    assert [text.get_text() for text in labeled.get_legend().get_texts()] == [
        "high cumulative_pd",
        "high competing",
    ]


def test_survival_chart_competing_risks_leave_the_survival_a_single_curve():
    ax = plotting.survival_chart(competing_example(), column="survival")
    assert len(ax.get_lines()) == 1


def test_survival_chart_keeps_the_limits_of_the_curves_already_drawn():
    high = survival_table([1, 2, 3, 3, 4], [1, 1, 1, 1, 0])
    low = survival_table(range(1, 11), [0, 0, 1] + [0] * 7)
    ax = plotting.survival_chart(high, label="high")
    plotting.survival_chart(low, label="low", ax=ax)
    assert ax.get_ylim()[1] >= high["cumulative_pd"].max()
    assert ax.get_xlim()[1] >= 11
    reverse = plotting.survival_chart(low, label="low")
    plotting.survival_chart(high, label="high", ax=reverse)
    assert reverse.get_xlim()[1] >= 11 and reverse.get_ylim()[1] >= 0.8
    assert [line.get_label() for line in ax.get_lines()] == ["high", "low"]


def test_survival_chart_starts_a_slice_at_its_own_first_period():
    ax = plotting.survival_chart(survival_example().iloc[2:])
    assert ax.get_lines()[0].get_xdata()[0] == 2


def test_survival_chart_draws_an_empty_chart_for_a_table_without_values(recwarn):
    table = survival_table([3, 4, 5], [1, 0, 1], entry=[2, 2, 2])
    assert table["cumulative_pd"].isna().all()
    ax = plotting.survival_chart(table)
    assert ax.get_xlim()[1] > ax.get_xlim()[0]
    assert ax.get_ylim() == (0, 1.0)
    assert not recwarn.list


def test_survival_chart_ends_the_curve_where_no_account_is_at_risk():
    table = survival_table([1, 2], [1, 0], entry=[0, 0])
    table.loc[3] = [0, 0, 0, np.nan, np.nan, np.nan]
    ax = plotting.survival_chart(table)
    assert np.isnan(ax.get_lines()[0].get_ydata()[-2])


def test_survival_chart_rejects_another_column(no_new_figure):
    with pytest.raises(
        ValueError, match="column must be 'cumulative_pd' or 'survival', got 'hazard'"
    ):
        plotting.survival_chart(survival_example(), column="hazard")


def test_survival_chart_needs_the_columns_of_the_competing_cause(no_new_figure):
    table = competing_example().drop(columns="at_risk")
    with pytest.raises(ValueError, match=r"missing \['at_risk'\]"):
        plotting.survival_chart(table)


def test_transition_chart_draws_the_values_and_outlines_the_diagonal():
    table = transition_example()
    ax = plotting.transition_chart(table)
    assert [text.get_text() for text in ax.texts] == ["2", "1", "0", "0", "1", "1", "0", "1", "2"]
    # The shade is the share of the row.
    image = ax.images[0].get_array()
    np.testing.assert_allclose(image, table.to_numpy() / table.to_numpy().sum(axis=1)[:, None])
    corners = sorted(patch.get_xy() for patch in ax.patches)
    assert corners == [(-0.5, -0.5), (0.5, 0.5), (1.5, 1.5)]
    assert all(patch.get_edgecolor()[:3] == matplotlib.colors.to_rgb(AMBER) for patch in ax.patches)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("after", "before")
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b", "c"]


def test_transition_chart_writes_weighted_counts_compactly():
    table = pd.DataFrame(
        [[300_332_577.87, 1234.5], [12.0, 20_000.0]], index=["a", "b"], columns=["a", "b"]
    )
    ax = plotting.transition_chart(table)
    assert [text.get_text() for text in ax.texts] == ["300M", "1.23K", "12", "20K"]


def test_transition_chart_writes_shares_with_two_decimals():
    ax = plotting.transition_chart(
        transition_matrix(["a", "a", "a"], ["a", "b", "b"], normalize=True)
    )
    assert [text.get_text() for text in ax.texts] == ["0.33", "0.67"]


def test_transition_chart_outlines_only_the_states_that_meet():
    table = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["b", "c"])
    ax = plotting.transition_chart(table)
    assert [patch.get_xy() for patch in ax.patches] == [(-0.5, 0.5)]


def test_transition_chart_leaves_missing_cells_empty_and_shades_empty_rows_white():
    table = pd.DataFrame([[0.0, 0.0], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    ax = plotting.transition_chart(table)
    assert [text.get_text() for text in ax.texts] == ["0", "0", "1"]
    assert ax.images[0].get_array()[0].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ([[1]], "table must be a DataFrame from kivi.vintage.transition_matrix, got list"),
        (pd.DataFrame(), "table must have at least one row and one column"),
        (
            pd.DataFrame({"roll_rate": [0.1, 0.2]}),
            "table must have at least two columns, one for each state after; a Series such as the "
            "result of kivi.vintage.roll_rates is read as a table, not drawn",
        ),
        (pd.DataFrame({"a": ["x"], "b": [1]}), "column 'a' must be real and numeric, got dtype"),
        (
            pd.DataFrame({"a": [1.0, -0.5], "b": [1, 1]}),
            "table must not have negative values, got -0.5",
        ),
    ],
)
def test_transition_chart_rejects_another_table(table, message, no_new_figure):
    with pytest.raises(ValueError, match=re.escape(message)):
        plotting.transition_chart(table)


def test_rule_set_chart_draws_the_shares_and_the_approval_rate_on_one_axis():
    table = rule_set_example()
    ax = plotting.rule_set_chart(table)
    assert isinstance(ax, matplotlib.axes.Axes) and len(ax.figure.axes) == 1
    shares = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(shares, table["new_hits"] / 5)
    rates = ax.get_lines()[0].get_ydata()
    np.testing.assert_array_equal(rates, table["approval_rate"])
    np.testing.assert_allclose(rates, 1 - np.cumsum(shares))
    assert ax.get_lines()[0].get_color() == AMBER
    assert ax.get_ylabel() == "new_hits, approval_rate"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["high_debt", "young"]
    assert ax.get_xlabel() == "rule"
    assert not covered(ax.get_legend(), [ax])


def test_profit_chart_marks_the_cutoff_with_the_highest_profit():
    table = profit_example()
    ax = plotting.profit_chart(table)
    line, mark = ax.get_lines()[0], ax.get_lines()[2]
    np.testing.assert_array_equal(line.get_xdata(), table["approval_rate"])
    np.testing.assert_array_equal(line.get_ydata(), table["profit"])
    best = table["profit"].idxmax()
    assert best == 0.3 and table["profit"].max() > 0
    assert mark.get_xdata().tolist() == [table.loc[best, "approval_rate"]]
    assert mark.get_ydata().tolist() == [table.loc[best, "profit"]]
    assert mark.get_color() == AMBER
    assert ax.texts[0].get_text() == f"cutoff = 0.3, profit = {table.loc[best, 'profit']:,.10g}"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("approval_rate", "profit")


def test_profit_chart_prints_a_cutoff_with_four_significant_digits():
    table = profit_example()
    table.index = pd.Index([0.05, 0.2, 0.11883280369928576, 0.6], name="cutoff")
    ax = plotting.profit_chart(table)
    assert ax.texts[0].get_text().startswith("cutoff = 0.1188,")


def test_profit_chart_labels_a_curve_to_compare_two_scores():
    table = profit_example()
    ax = plotting.profit_chart(table, label="current")
    assert plotting.profit_chart(table, label="new", ax=ax) is ax
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["current", "new"]
    assert plotting.profit_chart(table).get_legend() is None


def test_profit_chart_takes_the_first_of_tied_profits_and_skips_missing_ones():
    table = pd.DataFrame(
        {"approval_rate": [0.2, 0.5, 0.9], "profit": [np.nan, 2.0, 2.0]},
        index=pd.Index(["x", "y", "z"], name="cut"),
    )
    ax = plotting.profit_chart(table)
    assert ax.texts[0].get_text() == "cut = y, profit = 2"


def test_profit_chart_needs_a_profit(no_new_figure):
    table = profit_example().assign(profit=np.nan)
    with pytest.raises(ValueError, match="table must have at least one row with a profit"):
        plotting.profit_chart(table)


def test_profit_chart_names_the_columns_of_a_table_without_revenue(no_new_figure):
    table = cutoff_table(SCORE, TARGET, cutoffs=[0.3, 1.0])
    with pytest.raises(ValueError, match=r"missing \['profit'\]"):
        plotting.profit_chart(table)


def test_lgd_chart_draws_three_bars_per_group():
    table = lgd_example()
    ax = plotting.lgd_chart(table)
    heights = np.array([patch.get_height() for patch in ax.patches]).reshape(3, 3)
    np.testing.assert_allclose(heights[0], table["mean"])
    np.testing.assert_allclose(heights[1], table["zero_share"])
    np.testing.assert_allclose(heights[2], table["full_share"])
    assert [patch.get_facecolor()[:3] for patch in ax.patches[::3]] == [
        matplotlib.colors.to_rgb(color) for color in (BLUE, "#9ca3af", AMBER)
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "mean",
        "zero_share",
        "full_share",
    ]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b", "c"]
    assert not covered(ax.get_legend(), [ax])


def test_lgd_chart_keeps_a_negative_mean_in_view():
    ax = plotting.lgd_chart(lgd_table(["a", "a"], [-0.4, -0.2], floor=None))
    assert ax.get_ylim()[0] <= -0.3
