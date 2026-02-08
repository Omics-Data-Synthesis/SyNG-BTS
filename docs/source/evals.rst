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

Examples
~~~~~~~~

**Example 1: Visualize only real data**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import load_dataset, heatmap_eval

   # Load real data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # Visualize real data only
   heatmap_eval(
       dat_real=real_data_numeric.head(50),  # Use subset for better visualization
       save=False
   )

**Example 2: Compare real and generated data**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import load_dataset, heatmap_eval

   # Load real data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # Simulate generated data (in practice, this comes from a trained model)
   generated_data = real_data_numeric.copy() + np.random.normal(0, 0.1, real_data_numeric.shape)

   # Compare with heatmap
   heatmap_eval(
       dat_real=real_data_numeric.head(50),
       dat_generated=generated_data.head(50),
       save=False
   )

.. _umap:

UMAP Visualization
------------------

Visualize real and generated data distributions using UMAP.

.. autofunction:: syng_bts.UMAP_eval
   :no-index:

Examples
~~~~~~~~

**Example 1: Visualize only real data**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import load_dataset, UMAP_eval

   # Load real data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # UMAP projection of real data only
   UMAP_eval(
       dat_generated=None,
       dat_real=real_data_numeric,
       random_state=42
   )

**Example 2: Compare real and generated data**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import load_dataset, UMAP_eval

   # Load real data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # Simulate generated data (in practice, this comes from a trained model)
   generated_data = real_data_numeric.copy() + np.random.normal(0, 0.1, real_data_numeric.shape)

   # UMAP projection comparing both datasets
   UMAP_eval(
       dat_generated=generated_data,
       dat_real=real_data_numeric,
       random_state=42,
       legend_pos="best"
   )

**Example 3: UMAP with group labels**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import load_dataset, UMAP_eval

   # Load real data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # Simulate generated data
   generated_data = real_data_numeric.copy() + np.random.normal(0, 0.1, real_data_numeric.shape)

   # Create group labels for both datasets
   groups_real = pd.Series(['Group A', 'Group B'] * (len(real_data_numeric) // 2))
   groups_generated = pd.Series(['Group A', 'Group B'] * (len(generated_data) // 2))

   # UMAP projection with group information
   UMAP_eval(
       dat_generated=generated_data,
       dat_real=real_data_numeric,
       groups_generated=groups_generated,
       groups_real=groups_real,
       random_state=42,
       legend_pos="best"
   )

.. _evaluation:

Comprehensive Evaluation
------------------------

Run comprehensive evaluation metrics on generated data.

.. autofunction:: syng_bts.evaluation
   :no-index:

Example
~~~~~~~

The ``evaluation`` function is designed for advanced use cases where you have 
pre-generated data files from experiments. It loads data from files and performs 
comprehensive evaluation.

.. code-block:: python

   from syng_bts import evaluation

   # Run evaluation with default bundled data
   # Note: This requires specific data files to be present
   evaluation(
       generated_input="my_generated_data.csv",
       real_input="my_real_data.csv",
       data_dir="./my_data/"
   )

For most use cases, use ``heatmap_eval`` and ``UMAP_eval`` directly with dataframes 
as shown in the examples above.

Evaluation Workflow
-------------------

A typical evaluation workflow after running experiments:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from syng_bts import (
       PilotExperiment,
       load_dataset,
       heatmap_eval,
       UMAP_eval,
   )

   # Step 1: Load the original data
   real_data = load_dataset("SKCMPositive_4")
   real_data_numeric = real_data.select_dtypes(include=[np.number])

   # Step 2: Load or simulate generated data
   # In practice, load from the experiment output
   # For demonstration, we simulate it here
   generated_data = real_data_numeric.copy() + np.random.normal(0, 0.1, real_data_numeric.shape)

   # Step 3: Visualize with UMAP
   UMAP_eval(
       dat_generated=generated_data,
       dat_real=real_data_numeric,
       random_state=42
   )

   # Step 4: Compare with heatmap
   heatmap_eval(
       dat_real=real_data_numeric.head(50),
       dat_generated=generated_data.head(50),
       save=False
   )

See :doc:`methods` for more information on running experiments.