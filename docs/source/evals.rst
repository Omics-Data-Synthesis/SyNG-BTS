Evaluation Functions
====================

This page documents the evaluation and visualization functions in SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

SyNG-BTS provides multiple ways to evaluate generated data:

**On result objects** (recommended):

- ``result.plot_loss()`` — training loss curve with dual x-axes
- ``result.plot_heatmap()`` — heatmap of generated or reconstructed data

**Standalone functions** for comparing real vs. generated data:

- :func:`~syng_bts.heatmap_eval` — side-by-side heatmap comparison
- :func:`~syng_bts.UMAP_eval` — 2D UMAP scatter plot comparison
- :func:`~syng_bts.evaluation` — combined heatmap + UMAP pipeline

Result Object Plotting
----------------------

The simplest way to visualize results is through the
:class:`~syng_bts.SyngResult` methods:

.. code-block:: python

   from syng_bts import generate

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=500)

   # Training loss with running average and dual x-axes
   fig_loss = result.plot_loss(averaging_iterations=100)

   # Heatmap of generated data
   fig_heat = result.plot_heatmap(which="generated")

   # Heatmap of reconstructed data (AE/VAE/CVAE only)
   fig_recon = result.plot_heatmap(which="reconstructed")

For pilot studies, plot all runs at once:

.. code-block:: python

   from syng_bts import pilot_study

   pilot = pilot_study(data="SKCMPositive_4", pilot_size=[50, 100], model="VAE1-10")

   # One figure per run
   figs = pilot.plot_loss()

   # All runs overlaid on a single figure
   fig = pilot.plot_loss(aggregate=True)

.. _heatmap:

Heatmap Evaluation
------------------

Compare real and generated data with heatmaps.

.. autofunction:: syng_bts.heatmap_eval
   :no-index:

Examples
~~~~~~~~

**Example 1: Visualize only real data**

.. code-block:: python

   import numpy as np
   from syng_bts import resolve_data, heatmap_eval

   real_data = resolve_data("SKCMPositive_4").select_dtypes(include=[np.number])
   fig = heatmap_eval(real_data=real_data.head(50))

**Example 2: Compare real and generated data**

.. code-block:: python

   from syng_bts import generate, resolve_data, heatmap_eval

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)
   real_data = resolve_data("SKCMPositive_4").select_dtypes(include="number")

   fig = heatmap_eval(
       real_data=real_data.head(50),
       generated_data=result.generated_data.head(50),
   )

.. _umap:

UMAP Visualization
------------------

Visualize real and generated data distributions using UMAP.

.. autofunction:: syng_bts.UMAP_eval
   :no-index:

Examples
~~~~~~~~

**Example 1: Compare real and generated data**

.. code-block:: python

   import numpy as np
   from syng_bts import generate, resolve_data, UMAP_eval

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=500)
   real_data = resolve_data("SKCMPositive_4").select_dtypes(include=[np.number])

   fig = UMAP_eval(
       real_data=real_data,
       generated_data=result.generated_data,
       random_seed=42,
   )

**Example 2: UMAP with group labels**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import resolve_data, UMAP_eval

   real_data = resolve_data("SKCMPositive_4").select_dtypes(include=[np.number])

   groups_real = pd.Series(["Group A", "Group B"] * (len(real_data) // 2))

   fig = UMAP_eval(
       real_data=real_data,
       groups_real=groups_real,
       random_seed=42,
       legend_pos="best",
   )

.. _evaluation:

Comprehensive Evaluation
------------------------

Run combined heatmap + UMAP evaluation in a single call.

.. autofunction:: syng_bts.evaluation
   :no-index:

Example
~~~~~~~

The ``evaluation`` function accepts DataFrames, file paths, or bundled dataset
names (via ``resolve_data``) and returns a dict of figures:

.. code-block:: python

   from syng_bts import generate, evaluation

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)

   figs = evaluation(
       real_data="SKCMPositive_4",
       generated_data=result.generated_data,
       n_samples=200,
   )
   figs["heatmap"].savefig("heatmap.png")
   figs["umap"].savefig("umap.png")

Evaluation Workflow
-------------------

A typical end-to-end workflow:

.. code-block:: python

   from syng_bts import generate, resolve_data, heatmap_eval, UMAP_eval

   # Step 1: Generate synthetic data
   result = generate(
       data="SKCMPositive_4",
       model="VAE1-10",
       new_size=500,
       batch_frac=0.1,
       learning_rate=0.0005,
   )

   # Step 2: Load original data for comparison
   real_data = resolve_data("SKCMPositive_4").select_dtypes(include="number")

   # Step 3: Visualize training loss
   fig_loss = result.plot_loss()

   # Step 4: Compare with UMAP
   fig_umap = UMAP_eval(
       real_data=real_data,
       generated_data=result.generated_data,
       random_seed=42,
   )

   # Step 5: Compare with heatmap
   fig_heatmap = heatmap_eval(
       real_data=real_data.head(50),
       generated_data=result.generated_data.head(50),
   )

See :doc:`methods` for more information on running experiments.