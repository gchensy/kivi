Glossary
========

The terms of credit risk modeling as the guides and the docstrings use them. Each entry links to the guide that explains the term with an example.

.. glossary::

   Accepts and rejects
      The applicants that a previous policy approved and declined. The outcome is known only for the accepts, so a model developed on them alone may misjudge the rejects; see :doc:`/user_guide/reject_inference`.

   Area under the ROC curve (AUC)
      The probability that a randomly chosen event scores higher than a randomly chosen non-event, with ties counted as one half. A score unrelated to the target has an AUC of about 0.5; see :doc:`/user_guide/metrics`.

   Binning
      Dividing the values of a numeric variable into intervals, the bins, whose edges are fitted on a reference sample and then applied to every other sample. Scorecard development calls it classing: fine classing forms many small bins, and coarse classing merges them into a few that are large enough to be stable; see :doc:`/user_guide/binning`.

   Binning specification
      The edges, special codes and WOE tables of every column of a model, fitted together and saved as versioned JSON, so that production data are grouped and encoded exactly as the development sample; see :doc:`/user_guide/spec`.

   Characteristic stability index (CSI)
      The change of a variable's average scorecard points from a reference sample to a current one: the change of each group's share times the group's points, summed over the groups. Over all the variables of a scorecard, the CSI values add up to the change of the average score; see :ref:`characteristic-stability`.

   Cohort
      The loans originated in the same period, usually a month, followed together as they age in a vintage table; see :doc:`/user_guide/vintage`.

   Cut-off
      The score that separates approval from decline. kivi approves the scores at or below the cut-off, because its scores rise with the risk; see :doc:`/user_guide/strategy`.

   Event and non-event
      The two outcomes of a binary target: an event has target 1, such as a default or another bad outcome, and a non-event has target 0. The guides read the subscription of the bundled bank data as the event; see :doc:`/user_guide/woe_iv`.

   Event rate
      The share of events among the observations of a group, called the bad rate when the event is a default. It is computed from the raw counts, without smoothing; see :doc:`/user_guide/woe_iv`.

   Ever bad
      A loan is ever bad by a month on book if it was bad at any month on book up to then, even if it has cured since. The cumulative bad rates of a vintage table count the loans that are ever bad; see :doc:`/user_guide/vintage`.

   Expected loss (EL)
      The loss that an exposure is expected to cause: its probability of default times its loss given default times its exposure at default; see :doc:`/user_guide/strategy`.

   Exposure at default (EAD)
      The amount owed when a borrower defaults, such as the outstanding balance of a loan; see :doc:`/user_guide/strategy`.

   Gains table
      A table of score bands from the riskiest to the safest, with the event rate and lift of each band and the cumulative shares of events and non-events; see :doc:`/user_guide/metrics`.

   Gini coefficient
      Twice the AUC minus 1: 0 for a score unrelated to the target and 1 for a score that ranks every event above every non-event. Scorecard development reports it more often than the AUC itself; see :doc:`/user_guide/metrics`.

   Grade
      One class of a rating system or master scale, with a PD of its own and bounds that decide which predicted PDs fall into it; see :doc:`/user_guide/calibration`.

   Information value (IV)
      The sum of the differences between the event share and the non-event share of each group, each times the group's WOE. It measures how strongly a grouped variable separates events from non-events on its own; see :doc:`/user_guide/woe_iv`.

   Kolmogorov-Smirnov statistic (KS)
      The largest absolute gap between the distribution functions of the scores of events and of non-events; see :doc:`/user_guide/metrics`.

   Lift
      The event rate of a group divided by the event rate of the whole sample. A lift above 1 marks a group that is riskier than average; see :doc:`/user_guide/metrics`.

   Loss given default (LGD)
      The share of the exposure that is lost when a borrower defaults, after recoveries and collateral; see :doc:`/user_guide/strategy`.

   Master scale
      A fixed set of grades that divides the PD range from 0 to 1, so that ratings from different models share one meaning; see :doc:`/user_guide/calibration`.

   Month on book
      The number of months since a loan was originated. A vintage table compares cohorts at the same month on book; see :doc:`/user_guide/vintage`.

   Points to double the odds (PDO)
      The number of scorecard points by which the score rises when the good:bad odds double. Together with a base score and the odds at that score, it sets the scale of a scorecard; see :doc:`/user_guide/scorecard`.

   Population stability index (PSI)
      A measure of how much the distribution of a variable or a score has shifted between a reference sample and a current one, summed over the groups from the change of each group's share; see :doc:`/user_guide/psi`.

   Probability of default (PD)
      The probability that a borrower defaults within a given horizon, usually twelve months. kivi calls a predicted PD, or any other predicted event probability, ``probability``; see :doc:`/user_guide/calibration`.

   Reason codes
      The variables that lowered an applicant's score most, compared with the best group of each variable, as given to a declined applicant; see :doc:`/user_guide/scorecard`.

   Reject inference
      Methods that supply outcomes for rejected applicants, as rows with weights or as weights for the accepts, so that a model can be refitted on all applicants; see :doc:`/user_guide/reject_inference`.

   Roll rate
      The share of the loans in a delinquency bucket at one month end that are in the next worse bucket at the next month end; see :doc:`/user_guide/vintage`.

   Smoothing
      The value that replaces an empty count before shares are computed, 0.5 by default, so that the WOE, the IV and the PSI of a group stay finite. Counts are reported as observed, and only the shares are smoothed; see :doc:`/user_guide/woe_iv`.

   Special codes
      Values that stand for a state rather than a measurement, such as -1 for "never contacted" or 999 for "no record". Passed in ``special``, they are left out of the fit of the edges and each gets a group of its own; see :ref:`special-values`.

   Swap set
      The four groups of applicants that two scores, each with its cut-off, approve or decline. The swap-outs are approved only by the current score and the swap-ins only by the new one; see :doc:`/user_guide/strategy`.

   Transition matrix
      A table that counts the loans or obligors that moved from each state to each other state between two dates, such as delinquency buckets or rating grades; see :doc:`/user_guide/vintage`.

   Vintage table
      The cumulative ever-bad rate of each cohort by month on book; see :doc:`/user_guide/vintage`.

   Weight of evidence (WOE)
      The logarithm of a group's share of all events over its share of all non-events. A positive WOE marks a group that is relatively more common among events; see :doc:`/user_guide/woe_iv`.
