Experiment Methods
==================

This page documents the main experiment functions in SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

SyNG-BTS provides three main experiment functions:

- :func:`~syng_bts.PilotExperiment` - Train models on pilot data and generate samples
- :func:`~syng_bts.ApplyExperiment` - Apply trained models to case study datasets
- :func:`~syng_bts.TransferExperiment` - Transfer learning between datasets

.. _pilot:

PilotExperiment
---------------

Run pilot experiments to train generative models and generate synthetic data.

.. autofunction:: syng_bts.PilotExperiment
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import PilotExperiment

   # Train VAE on bundled example data
   PilotExperiment(
       dataname="SKCMPositive_4",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
       early_stop_num=30,
   )

   # Train with custom data and output location
   PilotExperiment(
       dataname="my_data",
       pilot_size=[50, 100, 200],
       model="CVAE1-20",
       data_dir="./input_data/",
       output_dir="./results/",
       batch_frac=0.1,
       learning_rate=0.0005,
       epoch=100,
   )

.. _apply:

ApplyExperiment
---------------

Apply generative models to case study datasets and generate samples.

.. autofunction:: syng_bts.ApplyExperiment
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import ApplyExperiment

   # Apply WGANGP to generate 1000 new samples
   ApplyExperiment(
       dataname="BRCASubtypeSel_train",
       new_size=[1000],
       model="WGANGP",
       data_dir="./case_study/",
       output_dir="./results/",
       apply_log=True,
       batch_frac=0.1,
       learning_rate=0.0005,
       epoch=10,
       early_stop_num=30,
   )

   # Using bundled BRCA subtype dataset
   ApplyExperiment(
       dataname="BRCASubtypeSel_train",
       new_size=[500, 1000, 2000],
       model="CVAE1-20",
       apply_log=True,
       batch_frac=0.1,
       learning_rate=0.0005,
       epoch=10,
   )

.. _transfer:

TransferExperiment
------------------

Transfer learning from one dataset to another.

.. autofunction:: syng_bts.TransferExperiment
   :no-index:

Example
~~~~~~~

.. code-block:: python

   from syng_bts import TransferExperiment

   # Transfer from PRAD to BRCA using MAF
   TransferExperiment(
       fromname="PRAD",
       toname="BRCA",
       fromsize=551,
       new_size=500,
       model="maf",
       apply_log=True,
       batch_frac=0.1,
       learning_rate=0.0005,
       epoch=10,
   )

   # Transfer with custom data directories
   TransferExperiment(
       fromname="source_dataset",
       toname="target_dataset",
       fromsize=200,
       new_size=1000,
       model="maf",
       data_dir="./transfer_data/",
       output_dir="./transfer_results/",
       apply_log=True,
       epoch=50,
   )

Choosing a Model
----------------

SyNG-BTS supports several generative models:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Model
     - Description
     - Best For
   * - ``VAE1-10``
     - VAE with 1:10 loss ratio
     - General purpose
   * - ``VAE1-20``
     - VAE with 1:20 loss ratio
     - Higher fidelity
   * - ``CVAE1-10``
     - Conditional VAE (1:10)
     - Labeled data
   * - ``CVAE1-20``
     - Conditional VAE (1:20)
     - Case studies
   * - ``GAN``
     - Standard GAN
     - Alternative to VAE
   * - ``WGANGP``
     - Wasserstein GAN-GP
     - Stable training
   * - ``maf``
     - Masked Autoregressive Flow
     - Transfer learning

See :doc:`configuration` for detailed parameter descriptions.
