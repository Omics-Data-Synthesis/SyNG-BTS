Usage Guide
===========

This guide covers installation and basic usage of SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _installation:

Installation
------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install SyNG-BTS using pip:

.. code-block:: console

   $ pip install syng-bts

From Source
~~~~~~~~~~~

For development or the latest features:

.. code-block:: console

   $ git clone https://github.com/LXQin/SyNG-BTS.git
   $ cd SyNG-BTS
   $ pip install -e .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Install documentation dependencies:

.. code-block:: console

   $ pip install syng-bts[docs]

Install development dependencies (testing, linting):

.. code-block:: console

   $ pip install syng-bts[dev]

Install all optional dependencies:

.. code-block:: console

   $ pip install syng-bts[all]

Quick Start
-----------

Basic Import
~~~~~~~~~~~~

After installation, import SyNG-BTS in your Python code:

.. code-block:: python

   from syng_bts import (
       PilotExperiment,
       ApplyExperiment,
       TransferExperiment,
       load_dataset,
       list_bundled_datasets,
   )

Load Example Data
~~~~~~~~~~~~~~~~~

SyNG-BTS includes bundled datasets for testing:

.. code-block:: python

   from syng_bts import load_dataset, list_bundled_datasets

   # See available datasets
   print(list_bundled_datasets())
   # ['SKCMPositive_4', 'BRCA', 'PRAD', 'BRCASubtypeSel', ...]

   # Load a dataset
   data = load_dataset("SKCMPositive_4")
   print(f"Dataset shape: {data.shape}")

Run a Pilot Experiment
~~~~~~~~~~~~~~~~~~~~~~

Train a VAE on example data:

.. code-block:: python

   from syng_bts import PilotExperiment

   PilotExperiment(
       dataname="SKCMPositive_4",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
       early_stop_num=30,
   )

This will:

1. Load the SKCMPositive_4 dataset
2. Train a VAE with 1:10 loss ratio
3. Generate 100 synthetic samples
4. Save results to ``./GeneratedData/``

Configure Output Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control where output files are saved:

.. code-block:: python

   from syng_bts import PilotExperiment, set_default_output_dir

   # Option 1: Set globally
   set_default_output_dir("./my_results")
   PilotExperiment(dataname="SKCMPositive_4", ...)

   # Option 2: Specify per experiment
   PilotExperiment(
       dataname="SKCMPositive_4",
       output_dir="./experiment_1",
       ...
   )

Evaluate Results
~~~~~~~~~~~~~~~~

Visualize generated data with UMAP and heatmaps:

.. code-block:: python

   from syng_bts import UMAP_eval, heatmap_eval

   # UMAP visualization
   UMAP_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

   # Heatmap comparison
   heatmap_eval(
       dataname="SKCMPositive_4",
       pilot_size=100,
       model="VAE1-10",
       new_size=100,
   )

Next Steps
----------

- See :doc:`methods` for detailed experiment examples
- See :doc:`configuration` for all available parameters
- See :doc:`api` for the complete API reference
- See :doc:`datasets` for information about bundled datasets


