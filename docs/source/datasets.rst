Example Datasets
================

SyNG-BTS includes bundled datasets for testing and experimentation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The bundled datasets come from TCGA (The Cancer Genome Atlas) studies:

- **BRCA** - Breast Invasive Carcinoma
- **PRAD** - Prostate Adenocarcinoma  
- **SKCM** - Skin Cutaneous Melanoma

Loading Datasets
----------------

Use the data utility functions to access bundled datasets:

.. code-block:: python

   from syng_bts import list_bundled_datasets, load_dataset

   # List all available datasets
   datasets = list_bundled_datasets()
   print(datasets)

   # Load a specific dataset
   data = load_dataset("SKCMPositive_4")
   print(f"Shape: {data.shape}")
   print(f"Columns: {data.columns.tolist()[:5]}...")

Available Datasets
------------------

Example Datasets
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset Name
     - Description
   * - ``SKCMPositive_4``
     - SKCM miRNA-seq data with mean threshold filtering (log scale > 4)

Transfer Learning Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset Name
     - Description
   * - ``BRCA``
     - Breast Invasive Carcinoma miRNA-seq data
   * - ``PRAD``
     - Prostate Adenocarcinoma miRNA-seq data

BRCA Subtype Case Study
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset Name
     - Description
   * - ``BRCASubtypeSel``
     - BRCA with cancer subtypes (ILC, IDC), marker-filtered
   * - ``BRCASubtypeSel_train``
     - Training split of BRCASubtypeSel
   * - ``BRCASubtypeSel_test``
     - Test split of BRCASubtypeSel

LIHC Subtype Case Study
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset Name
     - Description
   * - ``LIHCSubtypeFamInd``
     - Liver Hepatocellular Carcinoma subtype data
   * - ``LIHCSubtypeFamInd_DESeq``
     - LIHC with DESeq2 normalization
   * - ``LIHCSubtypeFamInd_test74``
     - Test split (74 samples)
   * - ``LIHCSubtypeFamInd_test74_DESeq``
     - Test split with DESeq2 normalization
   * - ``LIHCSubtypeFamInd_train294``
     - Training split (294 samples)
   * - ``LIHCSubtypeFamInd_train294_DESeq``
     - Training split with DESeq2 normalization

Usage Examples
--------------

Pilot Experiment with SKCMPositive_4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from syng_bts import PilotExperiment

   # Uses bundled SKCMPositive_4 automatically
   PilotExperiment(
       dataname="SKCMPositive_4",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
       early_stop_num=30,
   )

Case Study with BRCA Subtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from syng_bts import ApplyExperiment

   ApplyExperiment(
       dataname="BRCASubtypeSel_train",
       new_size=[1000],
       model="CVAE1-20",
       apply_log=True,
       batch_frac=0.1,
       learning_rate=0.0005,
       epoch=10,
   )

Transfer Learning from PRAD to BRCA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from syng_bts import TransferExperiment

   TransferExperiment(
       fromname="PRAD",
       toname="BRCA",
       fromsize=551,
       new_size=500,
       model="maf",
       apply_log=True,
       epoch=10,
   )

Using Custom Datasets
---------------------

You can also use your own datasets:

.. code-block:: python

   from syng_bts import PilotExperiment

   # Load from custom directory
   PilotExperiment(
       dataname="my_data",  # Will load my_data.csv
       data_dir="./custom_data/",
       output_dir="./results/",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
   )

Your CSV file should have:

- Samples as rows
- Features (genes/miRNAs) as columns
- First column can be sample IDs or index

Data Source
-----------

The example datasets are derived from `TCGA <https://www.cancer.gov/tcga>`_
(The Cancer Genome Atlas) miRNA-seq data.

For more information about the data processing and marker selection,
see the research paper:

    Qin, L.-X., et al. (2025). Optimizing sample size for supervised machine
    learning with bulk transcriptomic sequencing: a learning curve approach.
    *BMC Bioinformatics*, 26, 83.
    https://doi.org/10.1186/s12859-025-06079-9