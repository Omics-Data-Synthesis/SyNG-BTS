API Reference
=============

This page documents the complete public API of SyNG-BTS v3.0.

.. contents:: Table of Contents
   :local:
   :depth: 2

Experiment Functions
--------------------

These are the main entry points for training generative models and
producing synthetic data. All three functions accept data as a pandas
DataFrame, a CSV file path, or a bundled dataset name, and return rich
result objects.

generate
~~~~~~~~

.. autofunction:: syng_bts.generate

pilot_study
~~~~~~~~~~~

.. autofunction:: syng_bts.pilot_study

transfer
~~~~~~~~

.. autofunction:: syng_bts.transfer

Result Objects
--------------

Experiment functions return result objects that carry generated data,
loss logs, reconstructed data, and model state as attributes.

SyngResult
~~~~~~~~~~

.. autoclass:: syng_bts.SyngResult
   :members:
   :exclude-members: __init__, generated_data, loss, reconstructed_data, model_state, metadata

PilotResult
~~~~~~~~~~~

.. autoclass:: syng_bts.PilotResult
   :members:
   :exclude-members: __init__, runs, metadata

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

resolve_data
~~~~~~~~~~~~

.. autofunction:: syng_bts.resolve_data

derive_dataname
~~~~~~~~~~~~~~~

.. autofunction:: syng_bts.derive_dataname

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
   experiment functions (``generate``, ``pilot_study``, ``transfer``)
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
