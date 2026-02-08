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

These parameters are shared across experiment functions:

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
     - The generative model to use (e.g., ``"VAE1-10"``, ``"WGANGP"``, ``"maf"``)
   * - ``batch_frac``
     - float
     - Batch size as a fraction of training data (default: 0.1)
   * - ``learning_rate``
     - float
     - Learning rate for optimizer (default: 0.0005)
   * - ``epoch``
     - int or None
     - Number of training epochs. If ``None``, uses early stopping
   * - ``early_stop_num``
     - int
     - Patience for early stopping (default: 30)

Data Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``dataname``
     - str
     - Name of the dataset (without .csv extension)
   * - ``data_dir``
     - str or None
     - Directory containing input data. If ``None``, uses bundled data
   * - ``output_dir``
     - str or None
     - Directory for output files. If ``None``, uses current directory

Generation Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``pilot_size``
     - list[int]
     - Sample sizes for pilot experiments
   * - ``new_size``
     - list[int]
     - Number of new samples to generate
   * - ``apply_log``
     - bool
     - Whether to apply log transformation to data

Model Architecture Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``AE_head_num``
     - int
     - Number of encoder/decoder hidden layers (default: 2)
   * - ``Gaussian_head_num``
     - int
     - Number of layers in normalizing flow (default: 9)

PilotExperiment Parameters
--------------------------

Specific to :func:`syng_bts.PilotExperiment`:

.. code-block:: python

   from syng_bts import PilotExperiment

   PilotExperiment(
       dataname="SKCMPositive_4",  # Dataset name (required)
       pilot_size=[100],            # Sample sizes to test (required)
       model="VAE1-10",             # Model type (required)
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=None,                  # Epochs (None=early stopping)
       early_stop_num=30,           # Early stopping patience
       data_dir=None,               # Input data directory
       output_dir=None,             # Output directory
       pre_model=None,              # Pre-trained model path
       off_aug=None,                # Disable augmentation
       AE_head_num=2,               # Encoder layers
       Gaussian_head_num=9,         # Flow layers
   )

ApplyExperiment Parameters
--------------------------

Specific to :func:`syng_bts.ApplyExperiment`:

.. code-block:: python

   from syng_bts import ApplyExperiment

   ApplyExperiment(
       dataname="BRCASubtype",      # Dataset name (required)
       new_size=[1000],             # Samples to generate (required)
       model="WGANGP",              # Model type (required)
       path=None,                   # DEPRECATED: use data_dir instead
       data_dir="./my_data",        # Input data directory
       output_dir="./results",      # Output directory
       apply_log=True,              # Apply log transform
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=10,                    # Training epochs
       early_stop_num=30,           # Early stopping patience
       pre_model=None,              # Pre-trained model
       save_model=None,             # Save trained model path
       off_aug=None,                # Disable augmentation
       AE_head_num=2,               # Encoder layers
       Gaussian_head_num=9,         # Flow layers
   )

TransferExperiment Parameters
-----------------------------

Specific to :func:`syng_bts.TransferExperiment`:

.. code-block:: python

   from syng_bts import TransferExperiment

   TransferExperiment(
       fromname="PRAD",             # Source dataset (required)
       toname="BRCA",               # Target dataset (required)
       fromsize=551,                # Source sample size (required)
       new_size=500,                # Samples to generate (required)
       model="maf",                 # Model type (required)
       pilot_size=None,             # Pilot sizes
       apply_log=True,              # Apply log transform
       batch_frac=0.1,              # Batch fraction
       learning_rate=0.0005,        # Learning rate
       epoch=10,                    # Training epochs
       data_dir=None,               # Input data directory
       output_dir=None,             # Output directory
       off_aug=None,                # Disable augmentation
   )

Output Directory Configuration
------------------------------

SyNG-BTS creates the following subdirectories in the output directory:

.. code-block:: text

   output_dir/
   ├── GeneratedData/    # Generated synthetic samples
   ├── Loss/             # Training loss logs
   ├── ReconsData/       # Reconstructed samples (for VAE)
   └── runs/             # TensorBoard logs

You can configure the default output directory globally:

.. code-block:: python

   from syng_bts import set_default_output_dir, get_output_dir

   # Set default output directory
   set_default_output_dir("./my_results")

   # Check current output directory
   print(get_output_dir())  # ./my_results

   # Now all experiments will use this directory
   PilotExperiment(dataname="SKCMPositive_4", ...)  # Outputs to ./my_results/

Bundled Datasets
----------------

SyNG-BTS includes several bundled datasets for testing and examples:

.. code-block:: python

   from syng_bts import list_bundled_datasets, load_dataset

   # List all available datasets
   print(list_bundled_datasets())
   # ['SKCMPositive_4', 'BRCA', 'PRAD', 'BRCASubtypeSel', ...]

   # Load a bundled dataset
   data = load_dataset("SKCMPositive_4")
   print(f"Shape: {data.shape}")

Available bundled datasets:

- **Examples**: ``SKCMPositive_4``
- **Transfer Learning**: ``BRCA``, ``PRAD``
- **BRCA Subtype**: ``BRCASubtypeSel``, ``BRCASubtypeSel_train``, ``BRCASubtypeSel_test``
- **LIHC Subtype**: ``LIHCSubtypeFamInd``, ``LIHCSubtypeFamInd_DESeq``, and more
