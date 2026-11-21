Reject inference
================

.. module:: kivi.reject_inference

Supply outcomes for rejected applicants, so that a model developed on the approved applicants can be refitted on all of them. :func:`~kivi.reject_inference.fuzzy_augmentation` and :func:`~kivi.reject_inference.parceling` return inferred rows with weights, and :func:`~kivi.reject_inference.reweighting` returns weights for the approved applicants.

See :doc:`/user_guide/reject_inference` for a comparison of the methods on synthetic applications.

.. autosummary::

   fuzzy_augmentation
   parceling
   reweighting

.. autofunction:: fuzzy_augmentation

.. autofunction:: parceling

.. autofunction:: reweighting
