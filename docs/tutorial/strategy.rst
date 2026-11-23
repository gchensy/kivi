Cut-off and strategy
====================

The scorecard ranks the applicants, and the cut-off decides whom to approve. This chapter estimates what each cut-off would approve over all eligible applicants, compares the new decisions with the vendor's, and adds up the expected loss of the approved book; :doc:`/user_guide/strategy` describes each table.

.. dropdown:: Code from the previous chapters

   >>> import numpy as np
   >>> import pandas as pd
   >>> from kivi.datasets import make_applications
   >>> from kivi.reject_inference import fuzzy_augmentation
   >>> from kivi.selection import logistic_regression, stepwise
   >>> from kivi.spec import fit_spec
   >>> applications = make_applications(n_applicants=40000, rng=0)
   >>> months = pd.period_range("2024-01", periods=12, freq="M")
   >>> applications["month"] = months[np.arange(40000) * 12 // 40000]
   >>> eligible = applications[~applications["policy_decline"]]
   >>> accepts = eligible[eligible["approved"]]
   >>> development = accepts[accepts["month"] <= "2024-09"]
   >>> validation = accepts[accepts["month"] >= "2024-10"]
   >>> rejects = eligible[~eligible["approved"] & (eligible["month"] <= "2024-09")]
   >>> features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment", "incumbent_pd"]
   >>> method = {column: "monotonic" for column in ["age", "income", "debt_to_income", "incumbent_pd"]}
   >>> method["employment"] = "chimerge"
   >>> bins = {"inquiries": [-np.inf, 0, 1, 2, np.inf], "incumbent_pd": 8}
   >>> spec = fit_spec(development[features], development["target"], method=method, bins=bins)
   >>> selected = stepwise(spec.woe_transform(development), development["target"])
   >>> accepts_model = logistic_regression(spec.woe_transform(development)[selected], development["target"])
   >>> def predict(model, frame):
   ...     slopes = model["coef"].drop("intercept")
   ...     log_odds = model.loc["intercept", "coef"] + spec.woe_transform(frame)[slopes.index] @ slopes
   ...     return 1 / (1 + np.exp(-log_odds))
   >>> inferred = fuzzy_augmentation(predict(accepts_model, rejects), factor=1.5)
   >>> frame = pd.concat([development[features], rejects.loc[inferred.index, features]])
   >>> target = pd.concat([development["target"], inferred["target"]])
   >>> weights = pd.concat([pd.Series(1.0, index=development.index), inferred["weight"]])
   >>> model = logistic_regression(spec.woe_transform(frame)[selected], target, weights=weights)
   >>> from kivi.calibration import assign_grades, master_scale
   >>> master = master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE"))

Estimate each cut-off
---------------------

The cut-off applies to every eligible applicant, but the outcome is known only for those that the vendor approved. The declined ones get inferred outcomes, as in :doc:`reject_inference`: :func:`~kivi.reject_inference.fuzzy_augmentation` with a factor of 1, because the model's PDs already include the uplift of the rejects. Together with the approved applicants and their known outcomes, they stand for all eligible applicants of October to December:

>>> quarter = eligible[eligible["month"] >= "2024-10"]
>>> quarter_pd = predict(model, quarter)
>>> known = quarter["approved"]
>>> declined = fuzzy_augmentation(quarter_pd[~known])
>>> book_pd = pd.concat([quarter_pd[known], quarter_pd.loc[declined.index]])
>>> book_target = pd.concat([quarter["target"][known], declined["target"]])
>>> book_weight = pd.concat([pd.Series(1.0, index=quarter.index[known]), declined["weight"]])

:func:`~kivi.strategy.cutoff_table` gives, for each PD cut-off, the share of applicants that it approves and the default rate of the approved book. The upper bounds of the grades approve the grades up to each one:

>>> from kivi.strategy import cutoff_table, expected_loss, expected_loss_table, swap_set
>>> policies = cutoff_table(book_pd, book_target, cutoffs=master["upper"], weights=book_weight)
>>> columns = ["approval_rate", "approved_event_rate", "rejected_event_share"]
>>> policies.set_index(master.index)[columns].round(3)
       approval_rate  approved_event_rate  rejected_event_share
grade
A              0.073                0.025                 0.988
B              0.196                0.037                 0.951
C              0.438                0.063                 0.813
D              0.761                0.104                 0.467
E              1.000                0.149                 0.000

The vendor's scorecard approved 73.4% of the eligible applicants of the quarter, at a default rate of 10.6%. Approving the grades A to D, a PD of at most 22.6%, approves 76.1% at an estimated default rate of 10.4%: more loans at a slightly lower risk. The project takes that cut-off.

.. admonition:: Simulation check
   :class: kivi-check

   With the true outcomes of all eligible applicants, the same cut-offs give the same approval rates and almost the same default rates:

   >>> cutoff_table(quarter_pd, quarter["target"], cutoffs=master["upper"]).set_index(master.index)[
   ...     ["approval_rate", "approved_event_rate"]
   ... ].round(3)
          approval_rate  approved_event_rate
   grade
   A              0.073                0.025
   B              0.196                0.037
   C              0.438                0.063
   D              0.761                0.104
   E              1.000                0.153

Compare the decisions
---------------------

:func:`~kivi.strategy.swap_set` puts the new decisions next to the vendor's. The vendor approved a PD below 15%, while a cut-off approves PDs at or below it, so the vendor's cut-off is passed as 0.1499, just below 15% at the four decimals of ``incumbent_pd``. The outcome is known only where the vendor approved:

>>> cutoff = master.loc["D", "upper"]
>>> observed = quarter["target"].where(quarter["approved"])
>>> swap_set(
...     quarter["incumbent_pd"], quarter_pd, observed, current_cutoff=0.1499, new_cutoff=cutoff
... ).round(3)
                 total  share  known  event  event_rate
current new
approve approve   6086  0.659   6086    558       0.092
        decline    699  0.076    699    164       0.235
decline approve    951  0.103      0      0         NaN
        decline   1505  0.163      0      0         NaN

The new scorecard turns away 699 applicants whom the vendor approved, and who defaulted at 23.5%. It approves 951 applicants whom the vendor declined, the *swap-ins*, whose outcome no one knows. Their PDs say how risky they are expected to be:

>>> swap_ins = (quarter["incumbent_pd"] >= 0.15) & (quarter_pd <= cutoff)
>>> print(f"{quarter_pd[swap_ins].mean():.3f}")
0.183

.. admonition:: Simulation check
   :class: kivi-check

   The swap-ins default at 18.4%, close to their mean PD and below the 23.5% of the swap-outs, so the exchange lowers the risk of the book:

   >>> print(f"{quarter['target'][swap_ins].mean():.3f}")
   0.184

Add up the expected loss
------------------------

The expected loss of a loan is its PD times its loss given default times its exposure at default. The simulated applications have no loan amount, so the example lends each approved applicant six months of income and takes one LGD of 45%. :func:`~kivi.strategy.expected_loss` gives the loss of each loan, and :func:`~kivi.strategy.expected_loss_table` adds it up by grade:

>>> approved = quarter[quarter_pd <= cutoff]
>>> approved_pd = quarter_pd[quarter_pd <= cutoff]
>>> exposure = 6 * approved["income"]
>>> loss = expected_loss(approved_pd, 0.45, exposure)
>>> print(f"{loss.sum():,.0f} on {exposure.sum():,.0f}")
7,007,628 on 161,290,080
>>> by_grade = expected_loss_table(assign_grades(approved_pd, master), approved_pd, 0.45, exposure)
>>> by_grade[["total", "exposure", "expected_loss", "expected_loss_rate"]].round(
...     {"exposure": 0, "expected_loss": 0, "expected_loss_rate": 4}
... )
     total    exposure  expected_loss  expected_loss_rate
bin
A      679  21490440.0       172230.0              0.0080
B     1135  29518920.0       565187.0              0.0191
C     2234  50862660.0      1869213.0              0.0368
D     2989  59418060.0      4400998.0              0.0741

Grade D holds over a third of the exposure but almost two thirds of the expected loss. Which grades to approve, and at which price, is a business decision; the tables show its cost.
