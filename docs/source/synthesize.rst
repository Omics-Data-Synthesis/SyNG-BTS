Sample-Size Evaluation (SyntheSize)
====================================

SyNG-BTS integrates the `SyntheSize <https://github.com/LXQin/SyntheSize>`_
methodology for exploring how classifier performance changes across candidate
subset sizes. It visualizes learning-curve behavior; it does not calculate a
required or optimal sample size.

The integration provides two public functions:

- :func:`~syng_bts.evaluate_sample_sizes` — Evaluate classifiers across
  candidate sample sizes using stratified cross-validation or a fixed external
  evaluation set.
- :func:`~syng_bts.plot_sample_sizes` — Visualize inverse power-law (IPLF)
  learning curves fitted from evaluation metrics.

.. contents:: Table of Contents
   :local:
   :depth: 2

Background
----------

The SyntheSize approach trains multiple classifiers (logistic regression, SVM,
KNN, random forest, XGBoost) at varying sample sizes and fits inverse power-law
curves to the resulting metrics (F1, accuracy, AUC). This reveals how
classification performance changes with data volume and supports exploratory
assessment of whether generating more synthetic samples could improve
downstream analyses.

For more details on the methodology, see:

- **SyntheSize (R)**: https://github.com/LXQin/SyntheSize
- **SyntheSize (Python)**: https://github.com/LXQin/SyntheSize_py
- Qi Y, Wang X, Qin LX. *Optimizing sample size for supervised machine
  learning with bulk transcriptomic sequencing: a learning curve approach.*
  Brief Bioinform. 2025;26(2):bbaf097. https://doi.org/10.1093/bib/bbaf097

Quick Start
-----------

Evaluate a DataFrame
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from syng_bts import evaluate_sample_sizes, plot_sample_sizes, resolve_data

   # Load a bundled dataset
   data, groups = resolve_data("BRCASubtypeSel_test")

   # Evaluate classifiers across sample sizes
   metrics = evaluate_sample_sizes(
       data=data,
       sample_sizes=np.arange(25, 201, 25),
       groups=groups,
       n_draws=5,
   )
   print(metrics.head())

   # Plot learning curves
   fig = plot_sample_sizes(metrics)
   fig.savefig("learning_curves.png")

Evaluate a SyngResult
~~~~~~~~~~~~~~~~~~~~~

When you have a :class:`~syng_bts.SyngResult` with group information (e.g.,
from a CVAE run), you can pass it directly and groups are auto-resolved:

.. code-block:: python

   import numpy as np
   from syng_bts import generate, evaluate_sample_sizes, plot_sample_sizes

   # Generate synthetic data with a conditional model
   result = generate(
       data="BRCASubtypeSel_train",
       model="CVAE1-20",
       apply_log=True,
       epoch=50,
   )

   # Evaluate the generated data — groups are auto-resolved from result
   metrics_gen = evaluate_sample_sizes(
       data=result,
       sample_sizes=np.arange(25, 201, 25),
       which="generated",
   )

   # Compare real vs generated learning curves
   metrics_real = evaluate_sample_sizes(
       data=result,
       sample_sizes=np.arange(25, 201, 25),
       which="original",
   )

   fig = plot_sample_sizes(
       metric_real=metrics_real,
       metric_generated=metrics_gen,
   )
   fig.savefig("real_vs_generated.png")

Evaluate Against a Fixed Empirical Test Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``test_data`` and ``test_groups`` together to train each classifier on the
complete candidate subset and evaluate it once on fixed external rows. This is
useful for comparing real and generated candidate data against the same
empirical observations. The external observations should not have been used to
train the generative model.

.. code-block:: python

   metrics_real = evaluate_sample_sizes(
       data=real_candidate_data,
       sample_sizes=[50, 100, 150],
       groups=real_candidate_groups,
       test_data=empirical_test_data,
       test_groups=empirical_test_groups,
   )

   metrics_generated = evaluate_sample_sizes(
       data=generated_candidate_data,
       sample_sizes=[50, 100, 150],
       groups=generated_candidate_groups,
       test_data=empirical_test_data,
       test_groups=empirical_test_groups,
   )

Both calls return the same one-row-per-size/draw/method table used by the
internal cross-validation mode.

Workflow
--------

1. **Generate synthetic data** using :func:`~syng_bts.generate` (or load
   existing data).
2. **Evaluate** with :func:`~syng_bts.evaluate_sample_sizes` on both real
   and generated datasets, optionally using the same fixed empirical test set.
3. **Fit** with :func:`~syng_bts.fit_sample_sizes` to obtain numerical results.
4. **Visualize** with :func:`~syng_bts.plot_sample_sizes` to compare
   learning curves side by side.

Available Classifiers
---------------------

The following classifiers are available via the ``methods`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Name
     - Aliases
     - Description
   * - ``LOGIS``
     - ``LOGISTIC``, ``LR``
     - Ridge (L2-penalised) logistic regression via ``LogisticRegressionCV``
   * - ``SVM``
     -
     - Support Vector Machine with probability estimates
   * - ``KNN``
     -
     - K-Nearest Neighbors (k=5)
   * - ``RF``
     - ``RANDOM_FOREST``
     - Random Forest (100 trees)
   * - ``XGB``
     - ``XGBOOST``
     - XGBoost gradient-boosted trees

Without an external test set, classifiers are evaluated using 5-fold
stratified cross-validation. With an external test set, each classifier is
trained on the complete candidate subset and evaluated once on the fixed rows.

Meaning of Candidate Size
-------------------------

The ``total_size`` value and plotted x-axis represent the candidate subset
size before evaluation. In internal cross-validation mode, each classifier is
trained on about 80% of that subset in each fold. In external-evaluation mode,
each classifier is trained on the complete candidate subset.

Metrics
-------

Each evaluation returns three metrics per classifier per sample size:

- **F1 Score** (``f1_score``) — Macro-averaged F1
- **Accuracy** (``accuracy``) — Overall classification accuracy
- **AUC** (``auc``) — Area under ROC curve (one-vs-one, macro-averaged for multiclass)

Plot Scaling
------------

By default, :func:`~syng_bts.plot_sample_sizes` fixes every panel's y-axis to
``(0.4, 1)``. Pass a different two-value tuple with ``y_limits`` to choose a
custom range, or pass ``None`` to let Matplotlib choose limits automatically:

.. code-block:: python

   # Show the full metric range automatically
   fig = plot_sample_sizes(metrics, y_limits=None)

   # Use a custom fixed range for every panel
   fig = plot_sample_sizes(metrics, y_limits=(0.6, 1))

Log Transform
-------------

By default, :func:`~syng_bts.evaluate_sample_sizes` applies a
``log2(x + 1)`` transform (``apply_log=True``). Set ``apply_log=False``
when your input data is already log-transformed. The default behavior matches
the preprocessing convention used in SyNG-BTS training. In either evaluation
mode, feature standardization is fitted on the candidate training data and
then applied unchanged to the corresponding fold or external evaluation data.

Curve Fitting and Confidence Intervals
--------------------------------------

:func:`~syng_bts.fit_sample_sizes` fits each classifier separately, averaging
metric rows at each candidate size with equal weight per row. It fits
``1 - a - b*n**c`` to the *m* sorted means using weights ``1/m, ..., 1``,
independent of draw counts or SDs. Plotting uses the same implementation.

SciPy ``curve_fit`` uses unconstrained parameters, starting values
``[0, 1, -0.5]``, at most 50,000 function evaluations,
``sigma=1/sqrt(weight)`` and ``absolute_sigma=False``. Neither predictions nor
intervals are clipped to the metric range, and monotonicity is not imposed.

Bands are approximate pointwise 95% confidence intervals for the fitted mean,
not prediction intervals for individual classifier results. The delta method
uses the analytic parameter Jacobian, weighted residual variance with ``m - 3``
degrees of freedom, and the standard normal 0.975 quantile. Scaled, pivoted QR
and triangular solves reduce cancellation; exported covariance comes from the
same factorization.

At least three distinct sizes are required to fit a curve and four to estimate
intervals. Optimizer covariance warnings, non-finite covariance or a numerically
rank-deficient Jacobian leave the curve available without a band. Weak parameter
directions are never silently discarded.

Finite bands can still be unreliable for ill-conditioned or poorly identified
fits, including nearly zero exponents and effectively constant curves. These
local approximations do not account for all dependence between folds, draws or
overlapping subsets, and do not validate extrapolation.

Reusable Numerical Results
--------------------------

.. versionadded:: 3.6.0

.. code-block:: python

   import json
   import numpy as np
   from syng_bts import fit_sample_sizes

   fits = fit_sample_sizes(metrics, metric_name="auc")
   fit = fits["LOGIS"]
   print(fit.fit_status, fit.interval_status)
   print(fit.message)
   fit.observed.to_csv("observed.csv", index=False)
   fit.predictions.to_csv("predictions.csv", index=False)
   grid = np.linspace(fit.observed.n.min(), fit.observed.n.max(), 100)
   dense = fit.predict(grid)  # no optimizer call
   payload = json.dumps(fit.to_dict(), allow_nan=False)

The returned dictionary preserves first-seen classifier order. Result fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``observed``
     - DataFrame: sorted ``n``, ``observed_mean``, ``observed_std`` (sample SD,
       ddof=1), ``n_draws`` (row count), ``weight``. One draw gives unknown SD.
   * - ``parameters``
     - NumPy vector in ``a, b, c`` order, or None when unavailable.
   * - ``covariance``
     - NumPy 3-by-3 matrix in the same order, or None when unavailable.
       Analytic-Jacobian covariance when calculable; otherwise the optimizer's
       covariance is retained for inspection. See the rounding limitation below.
   * - ``predictions``
     - Stored DataFrame at observed sizes: ``n``, ``predicted``, ``ci_low``,
       ``ci_high``. Calculated once during fitting.
   * - ``fit_status``, ``interval_status``
     - Fit: ``ok`` or ``failed``. Intervals: ``ok`` or ``unavailable``.
       ``fit_ok`` and ``ci_ok`` are convenience booleans for status ``ok``.
   * - ``message``
     - Reason for a failed fit or unavailable intervals; empty on full success.

Treat attributes as read-only. Accessing ``predictions`` or exporting results
reuses the stored observed-size values without recalculating them.
``predict(grid)`` evaluates the fitted parameters at new sizes, returning only
the four numerical columns in the requested order. Repeated and fractional sizes
are allowed; the grid must be non-empty, one-dimensional, positive and finite.
A failed fit returns NaN predictions; unavailable intervals remain NaN. If a
previously usable curve or band cannot be evaluated on the new grid (for example,
because extrapolation overflows), ``predict`` raises ValueError and leaves the
stored results unchanged. Extrapolation is permitted, but remains unvalidated.
The small QR calculation is repeated from the stored observations and parameters
when evaluating a new grid; the nonlinear optimizer is not run again.

For ill-conditioned fits, rounding the dense covariance can lose information
needed to reproduce the intervals, even if its entries are individually accurate.
Use ``predict`` or the stored predictions instead of multiplying the exported
matrix by gradients. The exported observations, weights and parameters retain
the inputs needed to reconstruct the factorized calculation without refitting.

``to_dict()`` returns the observed data, fitted parameters, covariance,
predictions and fit status in a dictionary ready to save as JSON. Missing or
non-finite values become JSON ``null``; DataFrames use NaN for unavailable values.
The method does not write files. When saving results, also record the package
version and study settings used to produce them.


Verbosity
---------

The ``verbose`` parameter of :func:`~syng_bts.evaluate_sample_sizes` controls
console output during evaluation. It accepts the same levels used by the
training functions (:func:`~syng_bts.generate`, :func:`~syng_bts.pilot_study`,
:func:`~syng_bts.transfer`):

.. list-table::
   :header-rows: 1
   :widths: 10 15 75

   * - Level
     - Name
     - Behaviour
   * - ``0``
     - ``"silent"``
     - No output.
   * - ``1``
     - ``"minimal"``
     - One dynamically updated overall progress-bar line across all
       sample sizes, draws, and methods (default), while showing current
       size index/``n``, draw, and method.
   * - ``2``
     - ``"detailed"``
     - Per-draw / per-method metric lines (previous default behaviour).

Example:

.. code-block:: python

   # Detailed logging
   metrics = evaluate_sample_sizes(data, sample_sizes=[50, 100],
                                   groups=groups, verbose="detailed")

Reproducibility
---------------

Set ``random_seed`` to an integer to reproduce candidate sampling, shuffled
cross-validation splits, and stochastic classifier fits.

.. code-block:: python

   metrics = evaluate_sample_sizes(
       data,
       sample_sizes=[50, 100],
       groups=groups,
       random_seed=42,
   )

Sample-Size Shortcuts
---------------------

``sample_sizes`` accepts a **list**, **numpy array**, **pandas Series**, or a
**single integer**.  When a single integer *k* is provided it is interpreted as
the desired *number* of equidistant sizes — the maximum equals the number of
rows in the input data. The grid count *k* cannot exceed the number of rows.

.. code-block:: python

   # Equivalent to sample_sizes=[5, 10, 15] for 15-row data
   metrics = evaluate_sample_sizes(data, sample_sizes=3, groups=groups)

API Reference
-------------

.. autofunction:: syng_bts.evaluate_sample_sizes
   :no-index:

.. autofunction:: syng_bts.plot_sample_sizes
   :no-index:

.. autofunction:: syng_bts.fit_sample_sizes
   :no-index:

.. autoclass:: syng_bts.LearningCurveFit
   :members: predict, to_dict, fit_ok, ci_ok
   :no-index:
