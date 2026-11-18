User guide
==========

Binning and missing values
--------------------------

Use quantile bins to divide the reference observations by rank, or uniform bins to divide their numeric range into equal widths. Tied cut points are collapsed. A constant reference produces one bin. Infinite endpoints cover finite future observations outside the training range. Use the saved edges with pandas.cut on every sample.

Pass categorical labels directly to the metric functions. A missing label is a separate group. Only observed groups appear, in first-observed order. For WOE the group and target indices must match exactly.

Reading WOE and IV
------------------

An event is target 1. For a bin, let p be its share of all events and q its share of all non-events. WOE is log(p/q); a positive value means the bin is relatively more common among events. Its IV contribution is (p-q) times WOE. Sum the iv column across bins. Both target classes must be present.

Smoothing replaces zero cells with 0.5 by default. Each column is then normalized separately. Counts in the result always report observations. Changing smoothing can materially change the result for sparse bins.

Reading PSI
-----------

For PSI, p and q are the current and reference shares. Each bin contributes (p-q) times log(p/q). Sum the psi column to obtain total drift. Categories appearing in only one sample are retained. PSI describes distribution change; it does not use the outcome label.

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_.
