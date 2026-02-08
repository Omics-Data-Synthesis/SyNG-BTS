SyNG-BTS Documentation
======================

**SyNG-BTS** (Synthesis of Next Generation Bulk Transcriptomic Sequencing) is a Python package
for data augmentation of bulk transcriptomic sequencing data using deep generative models.

.. image:: https://badge.fury.io/py/syng-bts.svg
   :target: https://badge.fury.io/py/syng-bts
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
   :alt: License: AGPL v3

Overview
--------

SyNG-BTS synthesizes transcriptomics data with realistic distributions without relying on
predefined formulas. It supports three types of deep generative models:

- **Variational Autoencoders (VAE/CVAE)** - For general data augmentation
- **Generative Adversarial Networks (GAN/WGANGP)** - Alternative generative approach
- **Flow-based Models (MAF)** - For transfer learning scenarios

These models are trained on pilot datasets and used to generate synthetic samples
for any desired sample size.

Quick Start
-----------

Install SyNG-BTS:

.. code-block:: bash

   pip install syng-bts

Run a pilot experiment:

.. code-block:: python

   from syng_bts import PilotExperiment, load_dataset

   # Load example data
   data = load_dataset("SKCMPositive_4")
   print(f"Dataset shape: {data.shape}")

   # Train VAE and generate samples
   PilotExperiment(
       dataname="SKCMPositive_4",
       pilot_size=[100],
       model="VAE1-10",
       batch_frac=0.1,
       learning_rate=0.0005,
       early_stop_num=30,
   )

For more details, see the :doc:`usage` guide.

Citation
--------

If you use SyNG-BTS in your research, please cite:

   Qin, L.-X., Qi, Y., Wang, X., & Dueren, Y. (2025). Optimizing sample size
   for supervised machine learning with bulk transcriptomic sequencing: a
   learning curve approach. *BMC Bioinformatics*, 26, 83.
   https://doi.org/10.1186/s12859-025-06079-9

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   methods
   evals
   datasets
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Links
-----

- **GitHub Repository**: https://github.com/LXQin/SyNG-BTS
- **PyPI Package**: https://pypi.org/project/syng-bts/
- **Issue Tracker**: https://github.com/LXQin/SyNG-BTS/issues

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
   