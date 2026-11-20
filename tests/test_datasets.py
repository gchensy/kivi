import numpy as np
import pandas as pd
import pytest

from kivi.datasets import Dataset, load_bank, make_loan_panel

STATUSES = ["current", "1-29", "30-59", "60-89", "90+"]
COLUMNS = ["loan_id", "cohort", "month", "month_on_book", "term", "amount", "balance", "status"]


@pytest.fixture(scope="module")
def panel():
    return make_loan_panel(n_loans=1500, n_months=30, rng=7)


def test_load_bank_matches_dataset_bank_data():
    pd.testing.assert_frame_equal(load_bank(), Dataset.bank_data())


def test_load_bank_shape_and_target():
    bank = load_bank()
    assert bank.shape == (4521, 17)
    assert "y" not in bank.columns
    assert bank["target"].dtype == np.int64
    assert bank["target"].value_counts().to_dict() == {0: 4000, 1: 521}


def test_load_bank_returns_a_new_frame_on_every_call():
    first = load_bank()
    first.loc[0, "age"] = -1
    assert load_bank().loc[0, "age"] != -1


def test_panel_columns_and_dtypes(panel):
    assert list(panel.columns) == COLUMNS
    for column in ["loan_id", "month_on_book", "term"]:
        assert panel[column].dtype == np.int64
    for column in ["amount", "balance"]:
        assert panel[column].dtype == np.float64
    assert panel["cohort"].dtype == pd.PeriodDtype("M")
    assert panel["month"].dtype == pd.PeriodDtype("M")
    assert panel["status"].dtype == pd.CategoricalDtype(STATUSES, ordered=True)
    assert panel.index.equals(pd.RangeIndex(len(panel)))
    assert not panel.isna().any().any()


def test_panel_has_every_loan_with_consecutive_months_on_book(panel):
    assert np.array_equal(panel["loan_id"].unique(), np.arange(1, 1501))
    assert panel["loan_id"].is_monotonic_increasing
    position = panel.groupby("loan_id").cumcount() + 1
    assert np.array_equal(panel["month_on_book"], position)


def test_panel_months_follow_the_cohorts(panel):
    cohort = panel["cohort"].array.asi8
    month = panel["month"].array.asi8
    first = pd.Period("2024-01", freq="M").ordinal
    assert np.array_equal(month - cohort, panel["month_on_book"])
    assert cohort.min() == first and cohort.max() == first + 29
    assert month.max() == first + 30
    # Loans that are still on book are observed until the end of the window.
    last = panel.groupby("loan_id").tail(1)
    assert (last.loc[last["balance"] > 0, "month"].array.asi8 == first + 30).all()
    # Loan ids increase with the origination month.
    assert panel.groupby("loan_id")["cohort"].first().is_monotonic_increasing


def test_loan_attributes_are_constant(panel):
    attributes = panel.groupby("loan_id")[["cohort", "term", "amount"]].nunique()
    assert (attributes == 1).all().all()
    assert set(panel["term"].unique()) == {12, 24, 36}
    assert panel["amount"].min() >= 1000
    assert (panel["amount"] % 100 == 0).all()


def test_balance_follows_the_instalments_paid(panel):
    # Below 90+, the status counts the months since the oldest unpaid instalment fell due, so a
    # delinquent loan has paid month_on_book - status instalments and a current one every
    # instalment due; the balance is that of an annuity at 1% a month.
    code = panel["status"].cat.codes.to_numpy()
    rows = panel[code < 4]
    term = rows["term"].to_numpy()
    month_on_book = rows["month_on_book"].to_numpy()
    late = code[code < 4]
    paid = np.where(late == 0, np.minimum(month_on_book, term), month_on_book - late)
    expected = rows["amount"] * (1.01**term - 1.01**paid) / (1.01**term - 1)
    np.testing.assert_allclose(rows["balance"], expected.round(2), atol=1e-9)
    assert (panel["balance"] <= panel["amount"]).all()


def test_a_month_without_payment_rolls_the_loan_also_after_its_term(panel):
    # Every instalment lowers the balance, so an unchanged balance means that nothing was paid.
    code = panel["status"].cat.codes
    loan = panel["loan_id"]
    previous_code = code.groupby(loan).shift()
    unpaid = (panel["balance"] == panel["balance"].groupby(loan).shift()) & (previous_code < 4)
    assert (code[unpaid] == previous_code[unpaid] + 1).all()
    # Before the fix, a loan that was behind at the end of its term stopped rolling.
    after_term = panel["month_on_book"] > panel["term"]
    assert (unpaid & after_term).any()
    assert ((code == 4) & after_term).any()


def test_ninety_plus_is_absorbing_and_stops_payments(panel):
    code = panel["status"].cat.codes
    reached = code.eq(4).groupby(panel["loan_id"]).cummax()
    assert (code[reached] == 4).all()
    frozen = panel[reached].groupby("loan_id")["balance"].nunique()
    assert (frozen == 1).all()
    assert reached.any()


def test_status_worsens_by_at_most_one_bucket_a_month(panel):
    code = panel["status"].cat.codes
    step = code.groupby(panel["loan_id"]).diff().dropna()
    assert step.max() == 1
    # Cures jump straight to current, from every delinquent bucket.
    previous = code.groupby(panel["loan_id"]).shift()
    cured = previous[(code == 0) & (previous > 0)]
    assert set(cured.astype(int)) == {1, 2, 3}
    # The first month on book is current or 1-29.
    assert set(code[panel["month_on_book"] == 1]) == {0, 1}


def test_loans_leave_after_the_last_instalment(panel):
    repaid = panel[panel["balance"] == 0]
    assert (repaid["status"] == "current").all()
    # A loan that is behind at the end of its term pays off later.
    assert (repaid["month_on_book"] >= repaid["term"]).all()
    assert (repaid["month_on_book"] > repaid["term"]).any()
    last = panel.groupby("loan_id").tail(1)
    assert set(repaid.index) <= set(last.index)
    assert len(repaid) > 0


def test_default_panel_rolls_more_and_cures_less_the_later_the_bucket():
    panel = make_loan_panel(rng=0)
    code = panel["status"].cat.codes
    after = code.groupby(panel["loan_id"]).shift(-1)
    pairs = pd.DataFrame({"before": code, "after": after}).dropna()
    roll = [
        ((pairs.before == b) & (pairs.after == b + 1)).sum() / (pairs.before == b).sum()
        for b in range(4)
    ]
    cure = [
        ((pairs.before == b) & (pairs.after == 0)).sum() / (pairs.before == b).sum()
        for b in range(1, 4)
    ]
    assert roll == sorted(roll)
    assert cure == sorted(cure, reverse=True)
    assert 0.01 < roll[0] < 0.05


def test_panel_is_reproducible_and_leaves_global_state_alone():
    # Reads the legacy global state only to check that the generator does not touch it.
    state = np.random.get_state()  # noqa: NPY002
    first = make_loan_panel(n_loans=200, n_months=6, rng=3)
    after = np.random.get_state()  # noqa: NPY002
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
    pd.testing.assert_frame_equal(first, make_loan_panel(n_loans=200, n_months=6, rng=3))
    pd.testing.assert_frame_equal(
        first, make_loan_panel(n_loans=200, n_months=6, rng=np.random.default_rng(3))
    )
    assert not first.equals(make_loan_panel(n_loans=200, n_months=6, rng=4))


def test_panel_advances_a_generator():
    generator = np.random.default_rng(5)
    first = make_loan_panel(n_loans=50, n_months=4, rng=generator)
    second = make_loan_panel(n_loans=50, n_months=4, rng=generator)
    assert not first.equals(second)


def test_smallest_panel():
    panel = make_loan_panel(n_loans=1, n_months=1, rng=0)
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["loan_id"] == 1 and row["month_on_book"] == 1
    assert row["cohort"] == pd.Period("2024-01", freq="M")
    assert row["month"] == pd.Period("2024-02", freq="M")


def test_numpy_integer_sizes_are_accepted():
    pd.testing.assert_frame_equal(
        make_loan_panel(n_loans=np.int64(20), n_months=np.int32(3), rng=1),
        make_loan_panel(n_loans=20, n_months=3, rng=1),
    )


def test_sizes_are_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        make_loan_panel(20, 3)


@pytest.mark.parametrize("value", [0, -3, 2.5, True, "10", None])
@pytest.mark.parametrize("name", ["n_loans", "n_months"])
def test_sizes_must_be_positive_integers(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be a positive integer, got {value!r}"):
        make_loan_panel(**{name: value})
