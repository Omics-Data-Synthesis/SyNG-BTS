Usage Guide
===========

This guide covers installation and basic usage of SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _installation:

Installation
------------

**Requirements:** Python 3.10 or later.

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

*TODO: Not implemented yet.* Will be available in the future.
Install SyNG-BTS using pip:

.. code-block:: console

   $ pip install syng-bts  # TODO: Not implemented yet

From Source
~~~~~~~~~~~

For development or the latest features:

.. code-block:: console

   $ git clone https://github.com/Omics-Data-Synthesis/SyNG-BTS
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
       generate,
       pilot_study,
       transfer,
       list_bundled_datasets,
       resolve_data,
       SyngResult,
       PilotResult,
   )

Browse Bundled Datasets
~~~~~~~~~~~~~~~~~~~~~~~

SyNG-BTS includes bundled datasets for testing:

.. code-block:: python

   from syng_bts import list_bundled_datasets, resolve_data

   # See available datasets
   print(list_bundled_datasets())
   # ['SKCMPositive_4', 'BRCA', 'PRAD', 'BRCASubtypeSel', ...]

   # Load a bundled dataset (returns a tuple of DataFrame and optional groups)
   data, groups = resolve_data("SKCMPositive_4")
   print(f"Dataset shape: {data.shape}")

Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~

Train a generative model and produce synthetic samples:

.. code-block:: python

   from syng_bts import generate

   result = generate(
       data="SKCMPositive_4",  # bundled dataset name, CSV path, or DataFrame
       model="VAE1-10",
       new_size=500,
       batch_frac=0.1,
       learning_rate=0.0005,
   )

   # Access results in memory
   print(result.generated_data.shape)   # (500, n_features)
   print(result.loss.columns.tolist())  # ['kl', 'recons']
   print(result.summary())

   # Plot training loss (one figure per loss column)
   figs = result.plot_loss()
   figs["kl"].savefig("kl_loss.png")

   # Optionally save to disk
   result.save("./my_output/")

   # Load a previously saved result
   from syng_bts import SyngResult
   loaded = SyngResult.load("./my_output/")

Run a Pilot Study
~~~~~~~~~~~~~~~~~

Sweep over multiple pilot sizes with replicated random draws:

.. code-block:: python

   from syng_bts import pilot_study

   pilot = pilot_study(
       data="SKCMPositive_4",
       pilot_size=[50, 100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
   )

   # Access individual runs
   run = pilot.runs[(50, 1)]  # (pilot_size, draw_index)
   print(run.generated_data.head())

   # Save all runs
   pilot.save("./pilot_output/")

Use DataFrame Input
~~~~~~~~~~~~~~~~~~~

Pass your own data as a pandas DataFrame:

.. code-block:: python

   import pandas as pd
   from syng_bts import generate

   my_data = pd.read_csv("my_dataset.csv")
   result = generate(
       data=my_data,
       name="my_dataset",  # used in output filenames
       model="WGANGP",
       new_size=1000,
       epoch=50,
   )

Evaluate Results
~~~~~~~~~~~~~~~~

Visualize generated data with heatmaps and UMAP:

.. code-block:: python

   from syng_bts import generate, heatmap_eval, UMAP_eval, resolve_data

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)

   # Built-in heatmap on the result object
   fig = result.plot_heatmap()

   # Standalone evaluation comparing real and generated data
   real_data, _groups = resolve_data("SKCMPositive_4")
   heatmap_eval(real_data=real_data, generated_data=result.generated_data)
   UMAP_eval(real_data=real_data, generated_data=result.generated_data)

Next Steps
----------

- See :doc:`methods` for detailed experiment examples
- See :doc:`configuration` for all available parameters
- See :doc:`api` for the complete API reference
- See :doc:`datasets` for information about bundled datasets
- See :doc:`migration` for upgrading from v2.x


