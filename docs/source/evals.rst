Evaluation Functions
====================

This page documents the evaluation and visualization functions in SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

SyNG-BTS provides functions to evaluate and visualize generated data:

- :func:`~syng_bts.heatmap_eval` - Compare real and generated data with heatmaps
- :func:`~syng_bts.UMAP_eval` - Visualize data distribution with UMAP
- :func:`~syng_bts.evaluation` - Comprehensive evaluation metrics

.. _heatmap:

Heatmap Evaluation
------------------

Generate heatmap comparisons between real and synthetic data.

.. autofunction:: syng_bts.heatmap_eval
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import heatmap_eval

   # Evaluate VAE-generated data
   heatmap_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

   # Evaluate with custom data directories
   heatmap_eval(
       dataname="my_dataset",
       pilot_size=200,
       model="CVAE1-20",
       new_size=500,
       data_dir="./my_data/",
   )

.. _umap:

UMAP Visualization
------------------

Visualize real and generated data distributions using UMAP.

.. autofunction:: syng_bts.UMAP_eval
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import UMAP_eval

   # UMAP visualization of generated samples
   UMAP_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

   # Compare multiple generation sizes
   for size in [50, 100, 200]:
       UMAP_eval(
           dataname="SKCMPositive_4",
           pilot_size=100,
           model="VAE1-10",
           new_size=size,
       )

.. _evaluation:

Comprehensive Evaluation
------------------------

Run comprehensive evaluation metrics on generated data.

.. autofunction:: syng_bts.evaluation
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import evaluation

   # Run full evaluation
   evaluation(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

Evaluation Workflow
-------------------

A typical evaluation workflow after running experiments:

.. code-block:: python

   from syng_bts import (
       PilotExperiment,
       heatmap_eval,
       UMAP_eval,
   )

   # Step 1: Run experiment
   PilotExperiment(
       dataname="SKCMPositive_4",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
       early_stop_num=30,
   )

   # Step 2: Visualize with UMAP
   UMAP_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

   # Step 3: Compare with heatmap
   heatmap_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

See :doc:`methods` for more information on running experiments.