SyNG-BTS Documentation
======================

**SyNG-BTS** (Synthesis of Next Generation Bulk Transcriptomic Sequencing) is a Python package
for data augmentation of bulk transcriptomic sequencing data using deep generative models.

.. image:: https://badge.fury.io/py/syng-bts.svg
   :target: https://badge.fury.io/py/syng-bts
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

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

Generate synthetic data:

.. code-block:: python

   from syng_bts import generate

   result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)
   print(result.generated_data.shape)
   figs = result.plot_loss()

Run a pilot study:

.. code-block:: python

   from syng_bts import pilot_study

   result = pilot_study(
       data="SKCMPositive_4",
       pilot_size=[50, 100],
       model="VAE1-10",
       early_stop_patience=30,
   )
   print(result.summary())

For more details, see the :doc:`usage` guide.
For upgrading from v2.x, see the :doc:`migration` guide.

Citation
--------

If you use SyNG-BTS in your research, please cite:

   Qin, L.-X., et al. (2025). Optimizing sample size for supervised machine
   learning with bulk transcriptomic sequencing: a learning curve approach.
   *BMC Bioinformatics*, 26.
   https://doi.org/10.1093/bib/bbaf097

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage
   migration

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

- **GitHub Repository**: https://github.com/Omics-Data-Synthesis/SyNG-BTS
- **PyPI Package**: https://pypi.org/project/syng-bts/
- **Issue Tracker**: https://github.com/Omics-Data-Synthesis/SyNG-BTS

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
   