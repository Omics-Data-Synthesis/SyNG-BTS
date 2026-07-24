TCGA Datasets
=============

The TCGA loader downloads, caches, and exposes two packaged TCGA bundles
through a small Python API: 24 miRNA cohorts (``modality="mirna"``, the
default) — each containing the raw expression matrix, three
normalizations, and CVAE-synthesized counterparts — and 5 bulk RNA-seq
cohorts (``modality="rnaseq"``, see `Bulk RNA-seq cohorts`_). For a
runnable end-to-end walkthrough, see :doc:`notebooks/tcga_quickstart`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Unlike the small parquet files in :doc:`datasets`, TCGA cohorts are
downloaded on first use and cached locally. Each miRNA dataset packages
four kinds of data: the raw counts, three normalizations (``raw_norm``,
``TC``, ``DESeq``), and nine synthetic groups (three CVAE models × three
normalizations). Once cached, all subsequent access is local.

Quick Start
-----------

.. code-block:: python

   from syng_bts import list_tcga_datasets, load_tcga_dataset

   # Browse available cohorts
   list_tcga_datasets(short=True)

   # Load a cohort (downloads on first call)
   ds = load_tcga_dataset("BRCA")

   # Real expression data (DESeq-normalized by default)
   real_df, real_groups = ds.real("TC")
   real_df.shape
   real_groups.value_counts()

   # CVAE-synthesized counterpart — synth(normalization, model)
   synth_df, synth_groups = ds.synth("TC", "CVAE1_5")
   synth_df.shape

The :meth:`~syng_bts.TCGADataset.real` and :meth:`~syng_bts.TCGADataset.synth` accessors return a
``(expression, groups)`` tuple — the expression matrix as a
:class:`pandas.DataFrame` and the group labels as a
:class:`pandas.Series`. Per-slice metadata (KL weight, epochs trained,
etc.) is held by the underlying :class:`~syng_bts.Subset` objects,
reachable via ``ds.processed[norm]`` and ``ds.synthetic[norm][model]``.

Data layout
------------------

.. code-block:: text

   TCGADataset
   │   attributes: name, cancer_type, group_labels, n_filtered_samples, ...
   │
   ├── ds.raw                              → Subset
   │       unfiltered raw counts, 1881 features
   │
   ├── ds.processed[<norm>]                → Subset
   │       filtered, log2-transformed
   │       <norm> ∈ {"raw_norm", "TC", "DESeq" (default)}
   │
   └── ds.synthetic[<norm>][<model>]       → Subset
           CVAE-generated, 1000 samples
           <norm>  ∈ {"raw_norm", "TC", "DESeq" (default)}
           <model> ∈ {"CVAE1_5" (default), "CVAE1_10", "CVAE1_20"}

Each :class:`~syng_bts.Subset` has ``.expression`` (DataFrame),
``.groups`` (Series), and ``.metadata`` (dict).

Available Datasets
------------------

.. list-table:: TCGA cohorts in data-v1.0
   :header-rows: 1
   :widths: 12 53 18

   * - Code
     - Cancer name
     - Samples
   * - BLCA
     - Bladder Urothelial Carcinoma
     - 432
   * - BRCA
     - Breast Invasive Carcinoma
     - 1144
   * - COAD
     - Colon Adenocarcinoma
     - 414
   * - ESCA
     - Esophageal Carcinoma
     - 200
   * - HNSC
     - Head and Neck Squamous Cell Carcinoma
     - 569
   * - KICH
     - Kidney Chromophobe
     - 91
   * - KIRC
     - Kidney Renal Clear Cell Carcinoma
     - 613
   * - KIRP
     - Kidney Renal Papillary Cell Carcinoma
     - 295
   * - LAML
     - Acute Myeloid Leukemia
     - 187
   * - LIHC
     - Liver Hepatocellular Carcinoma
     - 352
   * - LUAD
     - Lung Adenocarcinoma
     - 557
   * - LUSC
     - Lung Squamous Cell Carcinoma
     - 516
   * - OV
     - Ovarian Serous Cystadenocarcinoma
     - 502
   * - PAAD
     - Pancreatic Adenocarcinoma
     - 183
   * - PCPG
     - Pheochromocytoma and Paraganglioma
     - 187
   * - PRAD
     - Prostate Adenocarcinoma
     - 536
   * - READ
     - Rectum Adenocarcinoma
     - 144
   * - SARC
     - Sarcoma
     - 263
   * - SKCM
     - Skin Cutaneous Melanoma
     - 449
   * - STAD
     - Stomach Adenocarcinoma
     - 490
   * - TGCT
     - Testicular Germ Cell Tumors
     - 139
   * - THCA
     - Thyroid Carcinoma
     - 573
   * - THYM
     - Thymoma
     - 123
   * - UCS
     - Uterine Carcinosarcoma
     - 57

Sample counts reflect TC-normalized real expression. Richer per-cohort
metadata is available at runtime via the :class:`~syng_bts.TCGADataset`
attributes (``ds.cancer_type``, ``ds.group_labels``, ``ds.schema_version``,
etc.) and per-slice on the underlying :class:`~syng_bts.Subset` objects
(``ds.processed[norm].metadata``, ``ds.synthetic[norm][model].metadata``).

Bulk RNA-seq cohorts
--------------------

Alongside the miRNA bundle, SyNG-BTS packages 5 bulk RNA-seq TCGA cohorts,
reachable through the same :func:`~syng_bts.list_tcga_datasets` /
:func:`~syng_bts.load_tcga_dataset` API by passing ``modality="rnaseq"``.

.. list-table:: Bulk RNA-seq cohorts in data-rnaseq-v1.0
   :header-rows: 1
   :widths: 12 40 14 12 18 14

   * - Code
     - Cancer name
     - Samples
     - Genes
     - Model family
     - Synthetic
   * - COAD
     - Colon Adenocarcinoma
     - 467
     - 723
     - CVAE
     - 1000
   * - LAML
     - Acute Myeloid Leukemia
     - 151
     - 850
     - VAE
     - 10000
   * - PAAD
     - Pancreatic Adenocarcinoma
     - 182
     - 866
     - VAE
     - 10000
   * - READ
     - Rectum Adenocarcinoma
     - 174
     - 633
     - VAE
     - 10000
   * - SKCM
     - Skin Cutaneous Melanoma
     - 361
     - 1091
     - CVAE
     - 1000

Quick start
~~~~~~~~~~~

.. code-block:: python

   from syng_bts import list_tcga_datasets, load_tcga_dataset

   list_tcga_datasets(modality="rnaseq")
   # ['COAD', 'LAML', 'PAAD', 'READ', 'SKCM']

   ds = load_tcga_dataset("SKCM", modality="rnaseq")
   real_df, real_groups = ds.real("DESeq")          # 361 x 1091, train + test
   synth_df, synth_groups = ds.synth("DESeq", "CVAE1-50")
   aug_df, _ = ds.synth("DESeq", "CVAE1-50", off_aug="AE_head")

How bulk differs from miRNA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``ds.raw`` is ``None`` — there is no raw expression subset.
- Model names are per-cohort and hyphenated, e.g. ``"CVAE1-50"``, versus
  miRNA's underscored ``"CVAE1_5"``.
- LAML, PAAD, and READ were trained with an unconditional model, so
  ``ds.synth(...)`` returns ``(expression, None)`` for those three
  cohorts — there are no synthetic group labels.
- Expression is stored as ``float32`` with sample-barcode-indexed frames,
  versus miRNA's ``float64`` with a plain ``RangeIndex``.
- Feature names are versioned Ensembl gene IDs (e.g.
  ``"ENSG00000141510.18"``) rather than miRBase miRNA names.



Working with TCGADataset
------------------------

real() and synth() return tuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~syng_bts.TCGADataset.real` returns a
``tuple[pandas.DataFrame, pandas.Series]``, and
:meth:`~syng_bts.TCGADataset.synth` returns a
``tuple[pandas.DataFrame, pandas.Series | None]`` — the expression matrix
and the aligned group labels:

- ``expression`` — :class:`pandas.DataFrame` of shape
  ``(n_samples, n_features)``. Columns are miRNA features.
- ``groups`` — :class:`pandas.Series` aligned to ``expression.index``
  with group labels (e.g. tumor vs. normal). ``None`` for
  :meth:`~syng_bts.TCGADataset.synth` on cohorts generated by an
  unconditional model (e.g. the bulk LAML, PAAD, and READ cohorts),
  which carry no synthetic group labels.

This is the most common usage pattern: pass ``expression`` to a model
and ``groups`` to a classifier.

Reaching the underlying Subset for metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For per-slice metadata (HDF5 attributes captured at dataset assembly
time — normalization method, transform, KL weight, epochs trained,
etc.), reach the underlying :class:`~syng_bts.Subset` directly:

.. code-block:: python

   real_subset = ds.processed["TC"]                    # Subset for real / TC
   synth_subset = ds.synthetic["TC"]["CVAE1_5"]        # Subset for synthetic / TC / CVAE1_5

   real_subset.expression       # same DataFrame returned by ds.real("TC")[0]
   real_subset.groups           # same Series returned by ds.real("TC")[1]
   real_subset.metadata         # dict of HDF5 attributes

Choosing a normalization
~~~~~~~~~~~~~~~~~~~~~~~~

Three normalizations are available (constants in
``syng_bts.tcga.VALID_NORMALIZATIONS``):

- ``"raw_norm"`` — raw counts after preprocessing
- ``"TC"`` — total-count normalized; common starting point for miRNA
  differential analyses
- ``"DESeq"`` *(default)* — DESeq2 size-factor normalized; preferred
  when downstream analysis assumes DESeq2 conventions

Choosing a synthetic model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Three CVAE variants (constants in ``syng_bts.tcga.VALID_MODELS``)
trade off reconstruction fidelity vs. latent-space diversity via the KL
weight:

- ``"CVAE1_5"`` *(default)* — KL weight 5; balanced
- ``"CVAE1_10"`` — KL weight 10; higher diversity, lower reconstruction
- ``"CVAE1_20"`` — KL weight 20; highest diversity

Caching and Offline Use
-----------------------

Cache location
~~~~~~~~~~~~~~

By default, datasets are cached under
``~/.cache/syng-bts/tcga/<modality>/<version>/``, where ``<modality>`` is
``"mirna"`` or ``"rnaseq"`` and ``<version>`` is the manifest version
(e.g. ``1.0``). Inspect the active cache root at runtime:

.. code-block:: python

   from syng_bts import tcga_cache_dir
   tcga_cache_dir()
   # PosixPath('/home/alice/.cache/syng-bts/tcga')

Custom cache directory
~~~~~~~~~~~~~~~~~~~~~~

Set the ``SYNG_BTS_CACHE_DIR`` environment variable to override the
cache root. Useful for shared filesystems, CI runners with restricted
homes, or when you want all SyNG-BTS caches in one place:

.. code-block:: bash

   export SYNG_BTS_CACHE_DIR=/data/shared/syng-bts-cache

First-call download semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first call to :func:`~syng_bts.load_tcga_dataset` for a given
dataset fetches the manifest, downloads the corresponding HDF5,
verifies its sha256, and atomically renames the temporary
file into the versioned cache directory. The download will retry once
on sha256 mismatch or transient network error before raising.

Force redownload and cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To redownload a single cohort (e.g. after a corrupt download or to pick
up an updated file from a custom mirror), pass ``force=True``:

.. code-block:: python

   ds = load_tcga_dataset("BRCA", force=True)

To remove the entire TCGA cache (all versions, all cohorts):

.. code-block:: python

   from syng_bts import clear_tcga_cache
   clear_tcga_cache()

Advanced
--------

Custom manifest URL
~~~~~~~~~~~~~~~~~~~

For staging or private mirrors, pass ``manifest_url=`` to
:func:`~syng_bts.load_tcga_dataset` or
:func:`~syng_bts.list_tcga_datasets`:

.. code-block:: python

   manifest = "https://my-mirror.example/tcga/manifest.json"
   ds = load_tcga_dataset("BRCA", manifest_url=manifest)

The mirror must serve a manifest with the same schema as the published
``data-v1.0`` manifest.

Dataset versioning
~~~~~~~~~~~~~~~~~~

The cache is keyed by manifest version. Releasing a new manifest (e.g.
``data-v1.1``) creates a new versioned subdirectory under
``tcga_cache_dir() / <modality>`` and does **not** invalidate or replace
existing ``data-v1.0`` files. Pinning to an older manifest with
``manifest_url=`` will still read from its versioned cache subdirectory.

Errors users may see
~~~~~~~~~~~~~~~~~~~~

- ``ValueError("Corrupt HDF5 at {path}; pass force=True to redownload.")``
  — the cached file is unreadable. Pass ``force=True`` to redownload.
- :class:`OSError` — manifest unreachable or sha256 mismatch persists
  after one retry. Check connectivity and the ``manifest_url`` argument.

See also
--------

- :doc:`methods` — using TCGA data with :func:`~syng_bts.generate` /
  :func:`~syng_bts.pilot_study`
- :doc:`evals` — visual evaluation of real vs. synthetic
- :doc:`api` — full API reference for the TCGA symbols

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   notebooks/tcga_quickstart
