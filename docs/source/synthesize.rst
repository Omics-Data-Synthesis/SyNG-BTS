Sample-Size Evaluation (SyntheSize)
====================================

SyNG-BTS integrates the `SyntheSize <https://github.com/LXQin/SyntheSize>`_
methodology for evaluating how classifier performance scales with sample size.
This is useful for answering the question: *"How many samples do I need for
reliable classification?"*

The integration provides two public functions:

- :func:`~syng_bts.evaluate_sample_sizes` — Evaluate classifiers across
  candidate sample sizes using stratified cross-validation.
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
classification performance scales with data volume and helps determine whether
generating more synthetic samples would improve downstream analyses.

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

   import pandas as pd
   from syng_bts import evaluate_sample_sizes, plot_sample_sizes, resolve_data

   # Load a bundled dataset
   data, groups = resolve_data("BRCASubtypeSel_test")

   # Evaluate classifiers across sample sizes
   metrics = evaluate_sample_sizes(
       data=data,
       sample_sizes=[50, 100, 150],
       groups=groups,
       n_draws=5,
   )
   print(metrics.head())

   # Plot learning curves
   fig = plot_sample_sizes(metrics, n_target=200)
   fig.savefig("learning_curves.png")

Evaluate a SyngResult
~~~~~~~~~~~~~~~~~~~~~

When you have a :class:`~syng_bts.SyngResult` with group information (e.g.,
from a CVAE run), you can pass it directly and groups are auto-resolved:

.. code-block:: python

  from syng_bts import (
     generate,
     evaluate_sample_sizes,
     plot_sample_sizes,
     resolve_data,
  )

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
       sample_sizes=[50, 100, 200],
       which="generated",
   )

   # Compare real vs generated learning curves
   real_data, real_groups = resolve_data("BRCASubtypeSel_test")
   metrics_real = evaluate_sample_sizes(
       data=real_data,
       sample_sizes=[50, 100, 150],
       groups=real_groups,
   )

   fig = plot_sample_sizes(
       metric_real=metrics_real,
       n_target=200,
       metric_generated=metrics_gen,
   )
   fig.savefig("real_vs_generated.png")

Workflow
--------

1. **Generate synthetic data** using :func:`~syng_bts.generate` (or load
   existing data).
2. **Evaluate** with :func:`~syng_bts.evaluate_sample_sizes` on both real
   and generated datasets.
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

All classifiers are evaluated using 5-fold stratified cross-validation.

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
the preprocessing convention used in SyNG-BTS training.

API Reference
-------------

.. autofunction:: syng_bts.evaluate_sample_sizes
   :no-index:

.. autofunction:: syng_bts.plot_sample_sizes
   :no-index:
