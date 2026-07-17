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
3. **Visualize** with :func:`~syng_bts.plot_sample_sizes` to compare
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

:func:`~syng_bts.plot_sample_sizes` displays approximate pointwise 95%
confidence intervals for the fitted inverse-power-law mean curves. The bands
propagate fitted-parameter covariance with the delta method; they are not
prediction intervals for individual classifier results.

The nonlinear fit uses the same increasing row weights as the R
implementation. After ordering the *m* curve points by candidate size, their
weights are ``1/m, 2/m, ..., m/m``, giving larger candidate sizes greater
weight.

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
