import datetime
import re

import numpy as np
import pandas as pd
import pytest

from kivi.datasets import load_bank, make_applications, make_loan_panel, make_transactions
from kivi.metrics import roc_auc

STATUSES = ["current", "1-29", "30-59", "60-89", "90+"]
COLUMNS = ["loan_id", "cohort", "month", "month_on_book", "term", "amount", "balance", "status"]
TRANSACTION_COLUMNS = ["applicant_id", "timestamp", "amount", "kind", "merchant_category"]
KINDS = ["purchase", "cash_withdrawal", "repayment"]
MERCHANT_CATEGORIES = [
    "grocery",
    "fuel",
    "dining",
    "travel",
    "retail",
    "online",
    "health",
    "utilities",
]


@pytest.fixture(scope="module")
def panel():
    return make_loan_panel(n_loans=1500, n_months=30, rng=7)


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


# --- make_applications --------------------------------------------------------------------


@pytest.fixture(scope="module")
def applications():
    return make_applications(rng=0)


def test_applications_columns_and_dtypes(applications):
    assert applications.shape == (10000, 10)
    assert applications.dtypes.astype(str).to_dict() == {
        "age": "int64",
        "income": "float64",
        "debt_to_income": "float64",
        "inquiries": "int64",
        "housing": "category",
        "employment": "category",
        "incumbent_pd": "float64",
        "policy_decline": "bool",
        "approved": "bool",
        "target": "int64",
    }
    assert applications["housing"].cat.categories.tolist() == ["own", "mortgage", "rent"]
    assert applications["employment"].cat.categories.tolist() == [
        "permanent",
        "temporary",
        "self_employed",
    ]
    assert not applications["housing"].cat.ordered
    assert applications.index.equals(pd.RangeIndex(10000))
    assert not applications.isna().any().any()


def test_application_values_lie_in_their_ranges(applications):
    assert applications["age"].between(18, 75).all()
    assert (applications["income"] > 0).all()
    assert (applications["income"] % 10 == 0).all()
    assert applications["debt_to_income"].between(0, 1).all()
    np.testing.assert_array_equal(
        applications["debt_to_income"], applications["debt_to_income"].round(2)
    )
    assert (applications["inquiries"] >= 0).all()
    assert applications["incumbent_pd"].between(0, 1).all()
    assert applications["target"].isin([0, 1]).all()


def test_approval_follows_the_cut_off_and_the_policy_rules(applications):
    policy = (applications["debt_to_income"] > 0.6) | (applications["age"] < 21)
    pd.testing.assert_series_equal(applications["policy_decline"], policy, check_names=False)
    approved = (applications["incumbent_pd"] < 0.15) & ~policy
    pd.testing.assert_series_equal(applications["approved"], approved, check_names=False)


def test_rejected_applicants_are_riskier(applications):
    approved = applications["approved"]
    assert 0.6 < approved.mean() < 0.75
    assert 0.05 < applications["policy_decline"].mean() < 0.1
    accepted_rate = applications.loc[approved, "target"].mean()
    rejected_rate = applications.loc[~approved, "target"].mean()
    assert 0.07 < accepted_rate < 0.13
    assert rejected_rate > 2.5 * accepted_rate
    # The incumbent PD ranks the risk: its deciles have rising default rates on the whole.
    deciles = pd.qcut(applications["incumbent_pd"], 10, labels=False)
    rates = applications["target"].groupby(deciles).mean()
    assert rates.iloc[-1] > 4 * rates.iloc[0]
    assert np.corrcoef(rates.index, rates)[0, 1] > 0.9


def test_rejects_are_riskier_than_accepts_with_the_same_features(applications):
    # The incumbent sees a hidden risk factor, so within a band of the true drivers the
    # rejected applicants default more often: the data are missing not at random.
    eligible = applications[~applications["policy_decline"]]
    band = pd.qcut(eligible["debt_to_income"], 4, labels=False)
    rates = eligible.groupby([band, "approved"])["target"].mean().unstack()
    assert (rates[False] > 1.5 * rates[True]).all()


def test_features_drive_the_default_rate(applications):
    rate = applications.groupby("housing", observed=True)["target"].mean()
    assert rate["own"] < rate["mortgage"] < rate["rent"]
    rate = applications.groupby("employment", observed=True)["target"].mean()
    assert rate["permanent"] < rate["self_employed"] < rate["temporary"]
    for column, sign in [("debt_to_income", 1), ("inquiries", 1), ("age", -1), ("income", -1)]:
        assert sign * np.corrcoef(applications[column], applications["target"])[0, 1] > 0.05


def test_applications_are_reproducible_and_leave_global_state_alone():
    # Reads the legacy global state only to check that the generator does not touch it.
    state = np.random.get_state()  # noqa: NPY002
    first = make_applications(n_applicants=300, rng=3)
    after = np.random.get_state()  # noqa: NPY002
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
    pd.testing.assert_frame_equal(first, make_applications(n_applicants=300, rng=3))
    pd.testing.assert_frame_equal(
        first, make_applications(n_applicants=300, rng=np.random.default_rng(3))
    )
    assert not first.equals(make_applications(n_applicants=300, rng=4))


def test_applications_advance_a_generator():
    generator = np.random.default_rng(5)
    first = make_applications(n_applicants=50, rng=generator)
    second = make_applications(n_applicants=50, rng=generator)
    assert not first.equals(second)


def test_smallest_applications():
    applications = make_applications(n_applicants=1, rng=0)
    assert applications.shape == (1, 10)
    assert make_applications(n_applicants=np.int64(1), rng=0).equals(applications)


def test_applications_size_is_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        make_applications(20)


@pytest.mark.parametrize("value", [0, -3, 2.5, True, "10", None])
def test_n_applicants_must_be_a_positive_integer(value):
    message = rf"n_applicants must be a positive integer, got {value!r}"
    with pytest.raises(ValueError, match=message):
        make_applications(n_applicants=value)


# --- make_transactions --------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulated():
    applications = make_applications(n_applicants=4000, rng=0)
    transactions, observations = make_transactions(applications, rng=0)
    return applications, transactions, observations


def window_counts(transactions, observations, kind, months, *, skip=0, total=False):
    """Count (or sum) the transactions of a kind in [date - months, date - skip) per application."""
    merged = transactions[transactions["kind"] == kind].merge(observations, on="applicant_id")
    start = merged["application_date"] - pd.DateOffset(months=months)
    end = merged["application_date"] - pd.DateOffset(months=skip)
    inside = (merged["timestamp"] >= start) & (merged["timestamp"] < end)
    grouped = merged[inside].groupby("applicant_id")["amount" if total else "timestamp"]
    result = grouped.sum() if total else grouped.size()
    return result.reindex(observations["applicant_id"], fill_value=0).set_axis(observations.index)


def test_transactions_columns_and_dtypes(simulated):
    applications, transactions, observations = simulated
    assert list(transactions.columns) == TRANSACTION_COLUMNS
    assert transactions.dtypes.astype(str).to_dict() == {
        "applicant_id": "int64",
        "timestamp": "datetime64[ns]",
        "amount": "float64",
        "kind": "category",
        "merchant_category": "category",
    }
    assert transactions["kind"].cat.categories.tolist() == KINDS
    assert transactions["merchant_category"].cat.categories.tolist() == MERCHANT_CATEGORIES
    assert not transactions["kind"].cat.ordered
    assert transactions.index.equals(pd.RangeIndex(len(transactions)))
    assert list(observations.columns) == ["applicant_id", "application_date"]
    assert observations.dtypes.astype(str).to_dict() == {
        "applicant_id": "int64",
        "application_date": "datetime64[ns]",
    }
    assert observations.index.equals(applications.index)
    assert observations["applicant_id"].tolist() == applications.index.tolist()


def test_transaction_values(simulated):
    applications, transactions, _ = simulated
    assert not transactions[["applicant_id", "timestamp", "amount", "kind"]].isna().any().any()
    assert np.isfinite(transactions["amount"]).all() and (transactions["amount"] >= 1).all()
    np.testing.assert_array_equal(transactions["amount"], transactions["amount"].round(2))
    assert (transactions["timestamp"].dt.microsecond == 0).all()
    # Only purchases have a merchant category, and every category occurs.
    purchase = transactions["kind"] == "purchase"
    assert transactions["merchant_category"].notna().equals(purchase)
    shares = transactions.loc[purchase, "merchant_category"].value_counts(normalize=True)
    assert set(shares.index) == set(MERCHANT_CATEGORIES)
    assert shares["grocery"] > shares["travel"] > 0.03
    assert set(transactions["applicant_id"]) == set(applications.index)


def test_transactions_are_sorted_by_applicant_and_time(simulated):
    _, transactions, _ = simulated
    applicant = transactions["applicant_id"].to_numpy()
    time = transactions["timestamp"].to_numpy()
    assert (np.diff(applicant) >= 0).all()
    same = applicant[1:] == applicant[:-1]
    assert (time[1:][same] >= time[:-1][same]).all()


def test_a_year_has_about_forty_transactions_per_applicant(simulated):
    applications, transactions, _ = simulated
    per_year = transactions["kind"].value_counts() / len(applications)
    assert 35 < per_year.sum() < 55
    assert 20 < per_year["purchase"] < 32
    assert 4 < per_year["cash_withdrawal"] < 9
    assert 10 < per_year["repayment"] <= 12


@pytest.mark.parametrize("months", [1, 3, 12, 15])
def test_every_transaction_falls_in_the_months_before_the_application(months):
    applications = make_applications(n_applicants=600, rng=1)
    transactions, observations = make_transactions(applications, months=months, rng=2)
    merged = transactions.merge(observations, on="applicant_id")
    assert len(merged) == len(transactions)
    start = merged["application_date"] - pd.DateOffset(months=months)
    assert (merged["timestamp"] >= start).all()
    assert (merged["timestamp"] < merged["application_date"]).all()
    # Every month of the window holds transactions.
    for month in range(months):
        end = merged["application_date"] - pd.DateOffset(months=month)
        begin = merged["application_date"] - pd.DateOffset(months=month + 1)
        assert ((merged["timestamp"] >= begin) & (merged["timestamp"] < end)).sum() > 500


def test_the_window_follows_calendar_months_at_the_end_of_a_month():
    applications = make_applications(n_applicants=3000, rng=3)
    transactions, _ = make_transactions(
        applications, application_date="2024-03-31", months=1, rng=0
    )
    # One month before 31 March is 29 February, and 31 March itself is not in the window.
    first = pd.Timestamp("2024-02-29")
    last = pd.Timestamp("2024-03-31")
    assert first <= transactions["timestamp"].min() < first + pd.Timedelta(hours=2)
    assert last - pd.Timedelta(hours=2) <= transactions["timestamp"].max() < last
    transactions, _ = make_transactions(
        applications, application_date="2024-05-31", months=3, rng=0
    )
    first = pd.Timestamp("2024-02-29")
    assert first <= transactions["timestamp"].min() < first + pd.Timedelta(hours=2)


def test_each_application_can_have_its_own_date():
    applications = make_applications(n_applicants=4, rng=0)
    dates = ["2024-01-31", "2024-02-29", "2025-03-01", "2020-07-04"]
    transactions, observations = make_transactions(
        applications, application_date=dates, months=2, rng=0
    )
    assert observations["application_date"].tolist() == [pd.Timestamp(d) for d in dates]
    for label, date in zip(applications.index, pd.to_datetime(dates), strict=True):
        own = transactions.loc[transactions["applicant_id"] == label, "timestamp"]
        assert len(own) > 0
        assert own.between(date - pd.DateOffset(months=2), date, inclusive="left").all()


def test_the_application_dates_can_be_given_in_every_form():
    applications = make_applications(n_applicants=60, rng=0)
    day = pd.Timestamp("2024-06-30")
    expected = make_transactions(applications, application_date=day, rng=4)
    forms = [
        "2024-06-30",
        datetime.date(2024, 6, 30),
        datetime.datetime(2024, 6, 30, 23, 59),
        np.datetime64("2024-06-30"),
        pd.Timestamp("2024-06-30 15:30"),
        pd.Timestamp("2024-06-30", tz="Asia/Shanghai"),
        ["2024-06-30"] * 60,
        np.full(60, day.to_datetime64()),
        pd.Series(day, index=applications.index),
        pd.DatetimeIndex([day] * 60),
    ]
    for form in forms:
        got = make_transactions(applications, application_date=form, rng=4)
        pd.testing.assert_frame_equal(got[0], expected[0])
        pd.testing.assert_frame_equal(got[1], expected[1])
    for column in ["2024-06-30", day, datetime.date(2024, 6, 30)]:
        frame = applications.assign(decided=column)
        got = make_transactions(frame, application_date="decided", rng=4)
        pd.testing.assert_frame_equal(got[0], expected[0])
        pd.testing.assert_frame_equal(got[1], expected[1])
    assert expected[1]["application_date"].eq(day).all()


def test_a_list_of_dates_is_matched_by_position_and_a_series_by_index():
    applications = make_applications(n_applicants=4, rng=0)
    applications.index = pd.Index(["w", "x", "y", "z"])
    dates = ["2024-01-31", "2024-02-29", "2025-03-01", "2020-07-04"]
    by_position = make_transactions(applications, application_date=dates, months=1, rng=0)
    assert by_position[1]["application_date"].tolist() == [pd.Timestamp(d) for d in dates]
    series = pd.Series(pd.to_datetime(dates), index=["w", "x", "y", "z"])
    by_index = make_transactions(applications, application_date=series, months=1, rng=0)
    pd.testing.assert_frame_equal(by_index[0], by_position[0])
    # The same labels in another order are another index, however.
    with pytest.raises(ValueError, match="applications and application_date must have identical"):
        make_transactions(applications, application_date=series[::-1], rng=0)


def test_default_dates_spread_the_applications_over_2024_in_row_order():
    applications = make_applications(n_applicants=1200, rng=0)
    _, observations = make_transactions(applications, months=1, rng=0)
    dates = observations["application_date"]
    assert dates.is_monotonic_increasing
    assert dates.iloc[0] == pd.Timestamp("2024-01-01")
    assert dates.iloc[-1] <= pd.Timestamp("2024-12-31")
    # The month of the tutorial: the twelve months of 2024 in equal parts of the rows.
    months = pd.period_range("2024-01", periods=12, freq="M")[np.arange(1200) * 12 // 1200]
    assert (dates.dt.to_period("M").to_numpy() == months.to_numpy()).all()
    # Within a month the days are spread evenly.
    january = dates[dates.dt.month == 1]
    assert january.dt.day.iloc[[0, 50, 99]].tolist() == [1, 16, 31]


def test_default_dates_of_a_few_applications():
    for n, expected in [
        (1, ["2024-01-01"]),
        (5, ["2024-01-01", "2024-03-01", "2024-05-01", "2024-08-01", "2024-10-01"]),
        (13, ["2024-01-01", "2024-01-16"] + [f"2024-{m:02d}-01" for m in range(2, 13)]),
    ]:
        applications = make_applications(n_applicants=n, rng=0)
        _, observations = make_transactions(applications, months=1, rng=0)
        assert observations["application_date"].dt.strftime("%Y-%m-%d").tolist() == expected


def test_applicant_ids_are_the_index_labels():
    applications = make_applications(n_applicants=6, rng=0)
    applications.index = pd.Index(["f", "e", "d", "c", "b", "a"], name="applicant_id")
    transactions, observations = make_transactions(applications, rng=0)
    assert pd.unique(transactions["applicant_id"]).tolist() == ["f", "e", "d", "c", "b", "a"]
    assert observations.index.equals(applications.index)
    assert observations["applicant_id"].tolist() == ["f", "e", "d", "c", "b", "a"]
    # The index of the observations has no name, so that a column join is not ambiguous.
    assert observations.index.name is None
    assert len(transactions.merge(observations, on="applicant_id")) == len(transactions)
    assert applications.join(observations[["application_date"]])["application_date"].notna().all()


def test_transactions_are_reproducible_and_leave_global_state_alone():
    applications = make_applications(n_applicants=200, rng=0)
    # Reads the legacy global state only to check that the generator does not touch it.
    state = np.random.get_state()  # noqa: NPY002
    first = make_transactions(applications, rng=3)
    after = np.random.get_state()  # noqa: NPY002
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
    for other in [
        make_transactions(applications, rng=3),
        make_transactions(applications, rng=np.random.default_rng(3)),
    ]:
        pd.testing.assert_frame_equal(first[0], other[0])
        pd.testing.assert_frame_equal(first[1], other[1])
    assert not first[0].equals(make_transactions(applications, rng=4)[0])


def test_transactions_advance_a_generator():
    applications = make_applications(n_applicants=50, rng=0)
    generator = np.random.default_rng(5)
    first = make_transactions(applications, rng=generator)[0]
    second = make_transactions(applications, rng=generator)[0]
    assert not first.equals(second)


def test_defaulters_withdraw_more_cash_in_the_last_months(simulated):
    applications, transactions, observations = simulated
    target = applications["target"] == 1
    cash = window_counts(transactions, observations, "cash_withdrawal", 3)
    earlier = window_counts(transactions, observations, "cash_withdrawal", 12, skip=6) / 2
    assert cash[target].mean() > 1.5 * cash[~target].mean()
    # Their withdrawals grow towards the application, and those of the others hardly do.
    growth = cash[target].mean() / earlier[target].mean()
    assert growth > 1.3 * cash[~target].mean() / earlier[~target].mean()
    assert earlier[target].mean() < cash[target].mean()
    amount = window_counts(transactions, observations, "cash_withdrawal", 3, total=True)
    assert amount[target].mean() > 1.5 * amount[~target].mean()


def test_defaulters_repay_less_in_the_last_months(simulated):
    applications, transactions, observations = simulated
    target = applications["target"] == 1
    repayments = window_counts(transactions, observations, "repayment", 3)
    assert repayments[target].mean() < repayments[~target].mean() - 0.1
    assert repayments.max() <= 3
    # An applicant never makes more than the one scheduled repayment a month.
    assert window_counts(transactions, observations, "repayment", 12).max() <= 12


def test_the_signal_is_clear_but_imperfect(simulated):
    applications, transactions, observations = simulated
    cash = window_counts(transactions, observations, "cash_withdrawal", 3)
    auc = roc_auc(cash, applications["target"])
    assert 0.58 < auc < 0.78
    repayments = window_counts(transactions, observations, "repayment", 3)
    assert 0.55 < 1 - roc_auc(repayments, applications["target"]) < 0.78


def test_the_volumes_scale_with_income(simulated):
    applications, transactions, observations = simulated
    purchases = window_counts(transactions, observations, "purchase", 12, total=True)
    assert np.corrcoef(np.log(applications["income"]), np.log(purchases))[0, 1] > 0.6
    merged = transactions.merge(
        applications[["income", "debt_to_income"]], left_on="applicant_id", right_index=True
    )
    share = (merged["amount"] / merged["income"]).groupby(merged["kind"], observed=True).median()
    assert 0.035 < share["purchase"] < 0.055
    assert 0.02 < share["cash_withdrawal"] < 0.04
    # A repayment is the instalment that the ratio of debt to income implies.
    repayment = merged[merged["kind"] == "repayment"]
    due = repayment["income"] * repayment["debt_to_income"].clip(lower=0.01)
    assert 0.9 < (repayment["amount"] / due).median() < 1.1


def test_riskier_applicants_by_the_incumbent_withdraw_more_cash(simulated):
    applications, transactions, observations = simulated
    cash = window_counts(transactions, observations, "cash_withdrawal", 12)
    quartile = pd.qcut(applications["incumbent_pd"], 4, labels=False)
    mean = cash.groupby(quartile).mean()
    assert mean.iloc[3] > 1.3 * mean.iloc[0]


def test_purchases_peak_before_the_end_of_the_year():
    applications = make_applications(n_applicants=2000, rng=2)
    transactions, _ = make_transactions(applications, application_date="2025-01-01", rng=0)
    purchases = transactions[transactions["kind"] == "purchase"]
    monthly = purchases.groupby(purchases["timestamp"].dt.month).size()
    assert monthly.index.tolist() == list(range(1, 13))
    assert monthly[12] > 1.2 * monthly.loc[3:9].mean()
    assert monthly[11] > monthly.loc[3:9].mean()


def test_inputs_are_not_changed():
    applications = make_applications(n_applicants=50, rng=0).assign(decided="2024-06-30")
    series = pd.Series(pd.date_range("2024-01-01", periods=50), index=applications.index)
    dates = list(series)
    copy = applications.copy()
    make_transactions(applications, application_date="decided", rng=0)
    make_transactions(applications, application_date=series, rng=0)
    make_transactions(applications, application_date=dates, rng=0)
    pd.testing.assert_frame_equal(applications, copy)
    assert series.equals(pd.Series(dates, index=applications.index))
    # The index of the observations is a copy of the labels of the applications.
    _, observations = make_transactions(applications, rng=0)
    observations.index = observations.index + 1
    assert applications.index.equals(copy.index)


def test_smallest_transactions():
    applications = make_applications(n_applicants=1, rng=0)
    transactions, observations = make_transactions(applications, months=1, rng=0)
    assert observations.shape == (1, 2)
    assert len(transactions) > 0
    assert (transactions["applicant_id"] == 0).all()
    assert transactions["timestamp"].between("2023-12-01", "2024-01-01", inclusive="left").all()
    assert make_transactions(applications, months=np.int64(1), rng=0)[0].equals(transactions)


def test_no_applications_give_empty_tables():
    applications = make_applications(n_applicants=3, rng=0)
    full, _ = make_transactions(applications, rng=0)
    for date in [None, [], "2024-01-01"]:
        transactions, observations = make_transactions(
            applications.iloc[:0], application_date=date, rng=0
        )
        assert transactions.shape == (0, 5) and observations.shape == (0, 2)
        assert (transactions.dtypes == full.dtypes).all()
        assert observations["application_date"].dtype == "datetime64[ns]"
        assert transactions.index.equals(pd.RangeIndex(0))


def test_transactions_arguments_are_keyword_only():
    applications = make_applications(n_applicants=5, rng=0)
    with pytest.raises(TypeError, match="positional argument"):
        make_transactions(applications, "2024-01-01")


@pytest.fixture
def few():
    return make_applications(n_applicants=5, rng=0)


def test_applications_must_be_a_data_frame(few):
    for value in [few["income"], [1, 2, 3], None]:
        message = f"applications must be a DataFrame, got {type(value).__name__}"
        with pytest.raises(ValueError, match=message):
            make_transactions(value)


def test_applications_must_have_the_required_columns(few):
    message = (
        "applications must have the columns ['income', 'debt_to_income', 'incumbent_pd', "
        "'target'], got no column ['incumbent_pd', 'target']"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        make_transactions(few.drop(columns=["incumbent_pd", "target"]))


def test_applications_need_a_flat_unique_index_and_columns(few):
    multi = few.set_index(pd.MultiIndex.from_arrays([[1, 1, 2, 2, 3], [1, 2, 3, 4, 5]]))
    with pytest.raises(ValueError, match="applications must have a flat index, got a MultiIndex"):
        make_transactions(multi)
    with pytest.raises(ValueError, match="unique index labels, got 1 twice"):
        make_transactions(few.set_axis([0, 1, 1, 2, 3]))
    doubled = pd.concat([few, few[["income"]]], axis=1)
    with pytest.raises(ValueError, match="applications has more than one column named 'income'"):
        make_transactions(doubled)
    doubled = pd.concat(
        [few.assign(decided="2024-06-30"), few[["income"]].rename(columns={"income": "decided"})],
        axis=1,
    )
    with pytest.raises(ValueError, match="applications has more than one column named 'decided'"):
        make_transactions(doubled, application_date="decided")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("income", 0.0, "column 'income' must be finite and positive, got 0.0"),
        ("income", -5.0, "column 'income' must be finite and positive, got -5.0"),
        ("income", np.inf, "column 'income' must be finite and positive, got inf"),
        ("income", np.nan, "column 'income' must not contain missing values, got 1"),
        (
            "debt_to_income",
            -0.1,
            "column 'debt_to_income' must be finite and non-negative, got -0.1",
        ),
        (
            "debt_to_income",
            np.inf,
            "column 'debt_to_income' must be finite and non-negative, got inf",
        ),
        ("incumbent_pd", 1.5, "column 'incumbent_pd' must be between 0 and 1, got 1.5"),
        ("incumbent_pd", -0.2, "column 'incumbent_pd' must be between 0 and 1, got -0.2"),
        ("target", 2.0, "target must contain only 0 and 1, got 2.0"),
        ("target", np.nan, "target must contain only 0 and 1, got nan"),
    ],
)
def test_driver_columns_must_hold_valid_numbers(few, column, value, message):
    values = few[column].to_numpy(dtype=float).copy()
    values[0] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        make_transactions(few.assign(**{column: values}))


def test_driver_columns_must_be_numbers(few):
    with pytest.raises(ValueError, match="column 'income' must be real and numeric"):
        make_transactions(few.assign(income="high"))
    with pytest.raises(ValueError, match="column 'debt_to_income' must be real and numeric"):
        make_transactions(few.assign(debt_to_income=True))


@pytest.mark.parametrize("value", [0, -3, 2.5, True, "12", None])
def test_months_must_be_a_positive_integer(few, value):
    with pytest.raises(ValueError, match=rf"months must be a positive integer, got {value!r}"):
        make_transactions(few, months=value)


def test_months_must_not_reach_back_before_1678(few):
    message = "months must not reach back before the year 1678, got 1000000"
    with pytest.raises(ValueError, match=message):
        make_transactions(few, months=1_000_000)
    # 21 years before 1700 is still after the first date pandas holds, 23 years are not.
    make_transactions(few, application_date="1700-01-01", months=12 * 21)
    with pytest.raises(ValueError, match="months must not reach back"):
        make_transactions(few, application_date="1700-01-01", months=12 * 23)


def test_rng_must_be_a_seed_or_generator(few):
    message = "rng must be None, an integer, a SeedSequence or a Generator, got 'seed'"
    with pytest.raises(ValueError, match=message):
        make_transactions(few, rng="seed")


@pytest.mark.parametrize(
    ("date", "message"),
    [
        ("dated", "application_date must be a column of applications or a date, got 'dated'"),
        (5, "application_date must hold dates, got dtype int64"),
        (True, "application_date must hold dates, got dtype bool"),
        (2.5, "application_date must hold dates, got dtype float64"),
        (pd.NaT, "application_date must not contain missing values, got 1"),
        ([None] * 5, "application_date must not contain missing values, got 5"),
        (
            pd.period_range("2024-01", periods=5, freq="M"),
            "application_date must hold dates, got dtype period[M]",
        ),
        (
            ["2024-01-01"] * 3,
            "application_date must have one value per row of applications, got 3 for 5",
        ),
        (np.zeros((5, 2)), "application_date must be one-dimensional, got 2 dimensions"),
        (
            pd.Series(pd.date_range("2024-01-01", periods=5), index=range(10, 15)),
            "applications and application_date must have identical indices",
        ),
        (
            ["2024-01-01", "soon", "2024-01-03", "2024-01-04", "2024-01-05"],
            "application_date must hold dates, got 'soon'",
        ),
        (
            datetime.date(3000, 1, 1),
            r"application_date must (hold dates|lie between the years 1678 and 2261), got",
        ),
        (
            np.datetime64("1500-01-01"),
            r"application_date must (hold dates|lie between the years 1678 and 2261), got",
        ),
    ],
)
def test_application_dates_must_be_dates_for_every_application(few, date, message):
    with pytest.raises(ValueError, match=re.escape(message) if "(" not in message else message):
        make_transactions(few, application_date=date)


def test_a_date_column_must_hold_dates(few):
    for values, message in [
        (1, "application_date must hold dates, got dtype int64"),
        (
            ["2024-01-01", "soon", "2024-01-03", "2024-01-04", "2024-01-05"],
            "application_date must hold dates, got 'soon'",
        ),
        (pd.NaT, "application_date must not contain missing values, got 5"),
    ]:
        with pytest.raises(ValueError, match=re.escape(message)):
            make_transactions(few.assign(decided=values), application_date="decided")
