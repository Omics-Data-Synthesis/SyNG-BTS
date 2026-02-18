Synthetic Data Generation
=========================

This page documents the core synthetic data generation functions in SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

SyNG-BTS provides three core synthetic data generation functions. All accept
data as a pandas DataFrame, a CSV file path, or the name of a bundled dataset,
and return rich result objects (``SyngResult`` or ``PilotResult``).
See :doc:`configuration` for all available parameters and model choices.

- :func:`~syng_bts.generate` — Train a model and produce synthetic samples
- :func:`~syng_bts.pilot_study` — Sweep over pilot sizes with replicated draws
- :func:`~syng_bts.transfer` — Pre-train on source data, fine-tune on target data

.. _generate:

generate
--------

Train a generative model on a dataset and generate synthetic samples.
This is the primary entry point for single model training.

.. autofunction:: syng_bts.generate
   :no-index:

Examples
~~~~~~~~

.. code-block:: python

   from syng_bts import generate

   # Generate synthetic data using a bundled dataset
   result = generate(
       data="SKCMPositive_4",
       model="VAE1-10",
       new_size=500,
       batch_frac=0.1,
       learning_rate=0.0005,
   )

   # Access results
   print(result.generated_data.shape)  # (500, n_features)
   print(result.summary())

   # Plot training loss (one figure per loss column)
   figs = result.plot_loss()

   # Save to disk
   result.save("./my_output/")

.. code-block:: python

   import pandas as pd
   from syng_bts import generate

   # Use your own DataFrame
   my_data = pd.read_csv("my_dataset.csv")
   result = generate(
       data=my_data,
       name="my_dataset",
       model="WGANGP",
       new_size=1000,
       epoch=50,
   )

.. _pilot:

pilot_study
-----------

Run pilot studies to evaluate models across multiple pilot sizes.
For each pilot size, five random sub-samples are drawn and a model is
trained on each.

.. autofunction:: syng_bts.pilot_study
   :no-index:

Examples
~~~~~~~~

.. code-block:: python

   from syng_bts import pilot_study

   # Evaluate VAE across different pilot sizes
   pilot = pilot_study(
       data="SKCMPositive_4",
       pilot_size=[50, 100, 200],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
   )

   # Access individual run results
   run = pilot.runs[(100, 1)]  # (pilot_size, draw_index)
   print(run.generated_data.shape)

   # Aggregate loss plots (one figure per loss column)
   figs = pilot.plot_loss(aggregate=True)

   # Save all results
   pilot.save("./pilot_output/")

.. code-block:: python

   from syng_bts import pilot_study

   # Using custom data with CVAE
   pilot = pilot_study(
       data="BRCASubtypeSel_train",
       pilot_size=[50, 100],
       model="CVAE1-20",
       epoch=100,
       output_dir="./results/",
   )

.. _transfer:

transfer
--------

Transfer learning: pre-train on a source dataset, then fine-tune and
generate on a target dataset.

.. autofunction:: syng_bts.transfer
   :no-index:

Examples
~~~~~~~~

.. code-block:: python

   from syng_bts import transfer

   # Transfer from PRAD to BRCA using MAF
   result = transfer(
       source_data="PRAD",
       target_data="BRCA",
       new_size=500,
       model="maf",
       apply_log=True,
       epoch=10,
   )

   print(result.generated_data.shape)
   result.save("./transfer_output/")

``transfer()`` is a single-run operation and always returns a
``SyngResult``. For pilot sweeps over target sample sizes, use
``pilot_study()``.

Choosing a Model
----------------

SyNG-BTS supports several generative models:

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Model
     - Description
   * - ``VAE1-10``
     - VAE with 1:10 loss ratio
   * - ``VAE1-20``
     - VAE with 1:20 loss ratio
   * - ``CVAE1-10``
     - Conditional VAE (1:10)
   * - ``CVAE1-20``
     - Conditional VAE (1:20)
   * - ``GAN``
     - Standard GAN
   * - ``WGANGP``
     - Wasserstein GAN-GP
   * - ``maf``
     - Masked Autoregressive Flow

See :doc:`configuration` for detailed parameter descriptions.
