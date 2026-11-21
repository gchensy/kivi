import numpy as np
import pandas as pd
import pytest

from kivi.datasets import Dataset, load_bank, make_applications, make_loan_panel

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
