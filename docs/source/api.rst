API Reference
=============

This page documents the complete public API of SyNG-BTS.

.. contents:: Table of Contents
   :local:
   :depth: 2

Experiment Functions
--------------------

These are the main functions for running data augmentation experiments.

PilotExperiment
~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.PilotExperiment

ApplyExperiment
~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.ApplyExperiment

TransferExperiment
~~~~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.TransferExperiment

Evaluation Functions
--------------------

Functions for evaluating and visualizing generated data.

heatmap_eval
~~~~~~~~~~~~

.. autofunction:: syng_bts.heatmap_eval

UMAP_eval
~~~~~~~~~

.. autofunction:: syng_bts.UMAP_eval

evaluation
~~~~~~~~~~

.. autofunction:: syng_bts.evaluation

Data Utilities
--------------

Functions for loading and managing datasets.

load_dataset
~~~~~~~~~~~~

.. autofunction:: syng_bts.load_dataset

list_bundled_datasets
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.list_bundled_datasets

set_default_output_dir
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.set_default_output_dir

get_output_dir
~~~~~~~~~~~~~~

.. autofunction:: syng_bts.get_output_dir

Model Classes
-------------

Advanced users can access the model classes directly.

.. note::
   These classes are for advanced usage. Most users should use the
   experiment functions (``PilotExperiment``, ``ApplyExperiment``, etc.)
   which handle model creation and training automatically.

AE (Autoencoder)
~~~~~~~~~~~~~~~~

.. autoclass:: syng_bts.AE
   :members:
   :undoc-members:
   :show-inheritance:

VAE (Variational Autoencoder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: syng_bts.VAE
   :members:
   :undoc-members:
   :show-inheritance:

CVAE (Conditional VAE)
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: syng_bts.CVAE
   :members:
   :undoc-members:
   :show-inheritance:

GAN (Generative Adversarial Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: syng_bts.GAN
   :members:
   :undoc-members:
   :show-inheritance:

Package Information
-------------------

Version and metadata information.

.. py:data:: syng_bts.__version__

   The current version of SyNG-BTS.

.. py:data:: syng_bts.__author__

   The package authors.

.. py:data:: syng_bts.__license__

   The package license (AGPL-3.0).
