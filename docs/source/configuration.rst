Configuration Reference
=======================

This page documents all configuration parameters available in SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Available Models
----------------

SyNG-BTS supports several deep generative models for data augmentation:

.. list-table:: Supported Models
   :header-rows: 1
   :widths: 20 40 40

   * - Model Code
     - Description
     - Use Case
   * - ``VAE1-10``
     - Variational Autoencoder with 1:10 reconstruction/KL loss ratio
     - General purpose, good for pilot experiments
   * - ``VAE1-20``
     - VAE with 1:20 loss ratio
     - Higher reconstruction fidelity
   * - ``CVAE1-10``
     - Conditional VAE with 1:10 loss ratio
     - When class labels are available
   * - ``CVAE1-20``
     - Conditional VAE with 1:20 loss ratio
     - Case studies with labeled data
   * - ``GAN``
     - Standard Generative Adversarial Network
     - Alternative generative approach
   * - ``WGANGP``
     - Wasserstein GAN with Gradient Penalty
     - More stable GAN training
   * - ``maf``
     - Masked Autoregressive Flow
     - Transfer learning scenarios

Common Parameters
-----------------

These parameters are shared across all experiment functions (``generate``,
``pilot_study``, ``transfer``):

Data Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``data``
     - DataFrame, str, or Path
     - Input data — a pandas DataFrame, a path to a CSV file, or the name
       of a bundled dataset (e.g. ``"SKCMPositive_4"``).
   * - ``name``
     - str or None
     - Short name for output filenames. Derived automatically from *data*
       when ``None``.
   * - ``output_dir``
     - str, Path, or None
     - If set, save results to this directory. When ``None`` (default),
       no files are written — data stays in memory.

Training Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``model``
     - str
     - The generative model to use (e.g. ``"VAE1-10"``, ``"WGANGP"``, ``"maf"``)
   * - ``batch_frac``
     - float
     - Batch size as a fraction of training data (default: 0.1)
   * - ``learning_rate``
     - float
     - Learning rate for optimizer (default: 0.0005)
   * - ``epoch``
     - int or None
     - Number of training epochs. If ``None``, uses early stopping.
   * - ``early_stop_patience``
     - int or None
     - Stop if loss does not improve for this many epochs. ``None``
       disables early stopping (requires *epoch* to be set).
   * - ``apply_log``
     - bool
     - Apply ``log2(x + 1)`` preprocessing to data (default: ``True``).
   * - ``random_seed``
     - int
     - Random seed for reproducibility (default: 123).

Generation Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``new_size``
     - int or list[int]
     - Number of synthetic samples to generate (default: 500).
   * - ``pilot_size``
     - list[int]
     - Sample sizes to evaluate (only for ``pilot_study()``).

Augmentation Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``off_aug``
     - str or None
     - Offline augmentation mode: ``"AE_head"``, ``"Gaussian_head"``, or
       ``None`` (default: ``None``).
   * - ``AE_head_num``
     - int
     - Fold multiplier for AE-head augmentation (default: 2).
   * - ``Gaussian_head_num``
     - int
     - Fold multiplier for Gaussian-head augmentation (default: 9).

Advanced Parameters (``generate`` only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``val_ratio``
     - float
     - Validation split ratio for AE family (default: 0.2).
   * - ``use_scheduler``
     - bool
     - Enable learning-rate scheduler for AE family (default: ``False``).
   * - ``step_size``
     - int
     - Scheduler step size (default: 10).
   * - ``gamma``
     - float
     - Scheduler gamma (default: 0.5).
   * - ``cap``
     - bool
     - Cap generated values to observed range (default: ``False``).

``generate()`` Parameters
-------------------------

.. code-block:: python

   from syng_bts import generate

   result = generate(
       data="SKCMPositive_4",       # Data input (required)
       name=None,                   # Output name (auto-derived)
       new_size=500,                # Samples to generate
       model="VAE1-10",             # Model specification
       apply_log=True,              # Log-transform data
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=None,                  # Epochs (None=early stopping)
       early_stop_patience=None,    # Early stopping patience
       off_aug=None,                # Offline augmentation
       AE_head_num=2,               # AE-head folds
       Gaussian_head_num=9,         # Gaussian-head folds
       use_scheduler=False,         # LR scheduler
       cap=False,                   # Cap generated values
       random_seed=123,             # Random seed
       output_dir=None,             # Output directory
   )

``pilot_study()`` Parameters
-----------------------------

.. code-block:: python

   from syng_bts import pilot_study

   result = pilot_study(
       data="SKCMPositive_4",       # Data input (required)
       pilot_size=[50, 100],        # Pilot sizes (required)
       name=None,                   # Output name (auto-derived)
       model="VAE1-10",             # Model specification
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=None,                  # Epochs (None=early stopping)
       early_stop_patience=30,      # Early stopping patience
       off_aug=None,                # Offline augmentation
       AE_head_num=2,               # AE-head folds
       Gaussian_head_num=9,         # Gaussian-head folds
       random_seed=123,             # Random seed
       output_dir=None,             # Output directory
   )

``transfer()`` Parameters
--------------------------

.. code-block:: python

   from syng_bts import transfer

   result = transfer(
       source_data="PRAD",          # Source dataset (required)
       target_data="BRCA",          # Target dataset (required)
       source_name=None,            # Source name (auto-derived)
       target_name=None,            # Target name (auto-derived)
       pilot_size=None,             # Pilot sizes (None=use generate)
       source_size=500,             # Source generation size
       new_size=500,                # Target generation size
       model="maf",                 # Model specification
       apply_log=True,              # Log-transform data
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=None,                  # Epochs (None=early stopping)
       early_stop_patience=30,      # Early stopping patience
       off_aug=None,                # Offline augmentation
       random_seed=123,             # Random seed
       output_dir=None,             # Output directory
   )

Output and Saving
-----------------

In v3.0, **no files are written by default**. Results stay in memory as
``SyngResult`` or ``PilotResult`` objects. To persist results to disk, either:

1. Pass ``output_dir`` to the experiment function, or
2. Call ``result.save(output_dir)`` on the returned object.

.. code-block:: python

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)

   # Option 1: Save later
   paths = result.save("./my_output/")
   print(paths)
   # {'generated': PosixPath('./my_output/SKCMPositive_4_VAE1-10_generated.csv'),
   #  'loss': PosixPath('./my_output/SKCMPositive_4_VAE1-10_loss.csv'), ...}

   # Option 2: Save automatically
   result = generate(
       data="SKCMPositive_4", model="VAE1-10", epoch=5,
       output_dir="./auto_output/",
   )

Bundled Datasets
----------------

SyNG-BTS includes several bundled datasets for testing and examples:

.. code-block:: python

   from syng_bts import list_bundled_datasets, resolve_data

   # List all available datasets
   print(list_bundled_datasets())
   # ['SKCMPositive_4', 'BRCA', 'PRAD', 'BRCASubtypeSel', ...]

   # Load a bundled dataset as a DataFrame
   data, groups = resolve_data("SKCMPositive_4")
   print(f"Shape: {data.shape}")

Available bundled datasets:

- **Examples**: ``SKCMPositive_4``
- **Transfer Learning**: ``BRCA``, ``PRAD``
- **BRCA Subtype**: ``BRCASubtypeSel``, ``BRCASubtypeSel_train``, ``BRCASubtypeSel_test``
- **LIHC Subtype**: ``LIHCSubtypeFamInd``, ``LIHCSubtypeFamInd_DESeq``, and more
