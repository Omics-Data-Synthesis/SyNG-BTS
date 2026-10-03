Release Notes
=============

3.6.0
-----

* Add :func:`~syng_bts.fit_sample_sizes` and :class:`~syng_bts.LearningCurveFit`
  for reusable parameters, covariance, predictions, confidence intervals and
  fit/interval statuses, with JSON-compatible export.
* Make :func:`~syng_bts.plot_sample_sizes` use the same numerical implementation,
  preserving its public signature and weighted fitting method.
* Reject malformed or non-finite inputs explicitly. Retain valid curves when
  confidence intervals cannot be estimated.

See :doc:`synthesize` for usage, result fields and uncertainty assumptions.
