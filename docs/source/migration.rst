Migration Guide / Changelog
============================

SyNG-BTS v3.0 is a **breaking** release that replaces the file-centric API
with a Pythonic, DataFrame-friendly interface. This guide covers the key
changes and all subsequent version updates. See also
`Changes in v3.1 <v31-changes_>`_ and `Changes in v3.2 <v32-changes_>`_ below.

.. contents:: Table of Contents
   :local:
   :depth: 2

Function Renames
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - v2.x
     - v3.0
     - Returns
   * - ``PilotExperiment(...)``
     - ``pilot_study(...)``
     - ``PilotResult``
   * - ``ApplyExperiment(...)``
     - ``generate(...)``
     - ``SyngResult``
   * - ``TransferExperiment(...)``
     - ``transfer(...)``
     - ``SyngResult``

Parameter Changes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - v2.x Parameter
     - v3.0 Parameter
     - Notes
   * - ``dataname="SKCMPositive_4"``
     - ``data="SKCMPositive_4"``
     - Also accepts a DataFrame or file path
   * - ``data_dir="./input/"``
     - *(removed)*
     - Pass the full file path via ``data``
   * - ``path="./input/"``
     - *(removed)*
     - Was deprecated in v2; removed in v3
   * - ``early_stop_num=30``
     - ``early_stop_patience=30``
     - Renamed for clarity
   * - ``fromname`` / ``toname``
     - ``source_data`` / ``target_data``
     - Now accept DataFrames too
   * - ``fromsize``
     - *(removed)*
     - ``transfer()`` no longer generates from the source phase; source is pre-training only

Data Input
----------

**v2.x** — String name only, with optional directory

**v3.0** — DataFrame, file path, or bundled name:

.. code-block:: python

   # v3.0 — bundled dataset (same as before, shorter param name)
   result = generate(data="SKCMPositive_4", ...)

   # v3.0 — file path
   result = generate(data="./my_data/my_data.csv", ...)

   # v3.0 — DataFrame (new!)
   import pandas as pd
   df = pd.read_csv("./my_data/my_data.csv")
   result = generate(data=df, name="my_data", ...)

Output Handling
---------------

**v2.x** — Results always written to disk

**v3.0** — Results returned in memory; disk write is optional:

.. code-block:: python

   # v3.0 — no files written by default
   result = generate(data="SKCMPositive_4", ...)

   # Access data directly
   generated = result.generated_data         # pd.DataFrame
   loss_log  = result.loss                   # pd.DataFrame
   recons    = result.reconstructed_data     # pd.DataFrame or None
   state     = result.model_state            # dict (state_dict)

   # Optional: save to disk
   result.save("./my_output/")

   # Or pass output_dir to save automatically
   result = generate(data="SKCMPositive_4", ..., output_dir="./my_output/")

Plotting
--------

**v2.x** — Standalone functions

**v3.0** — Methods on result objects (never call ``plt.show()``):

.. code-block:: python

   # v3.0
   result = generate(data="SKCMPositive_4", ...)

   figs = result.plot_loss()        # dict[str, Figure] per loss column
   fig = result.plot_heatmap()      # heatmap of generated data

   # For pilot studies
   pilot = pilot_study(data="SKCMPositive_4", pilot_size=[50, 100], ...)
   figs = pilot.plot_loss()                   # dict[(pilot, draw)] -> dict[str, Figure]
   figs = pilot.plot_loss(aggregate=True)     # dict[str, Figure], all runs overlaid

Evaluation Functions
--------------------

The ``evaluation()`` function now accepts DataFrames or dataset names:

.. code-block:: python

   # v3.0
   figs = evaluation(
       real_data="SKCMPositive_4",
       generated_data=result.generated_data,
   )


Quick Migration Checklist
-------------------------

1. Update imports: ``PilotExperiment`` → ``pilot_study``, etc.
2. Replace ``dataname=`` with ``data=``.
3. Replace ``early_stop_num=`` with ``early_stop_patience=``.
4. Remove ``data_dir`` / ``path`` parameters; pass full paths via ``data``.
5. Capture the return value (``SyngResult`` / ``PilotResult``).
6. Access generated data via ``result.generated_data`` instead of reading CSVs.
7. Use ``result.save(output_dir)`` when you need files on disk.
8. Replace standalone plot calls with ``result.plot_loss()`` / ``result.plot_heatmap()``.
9. Update eval calls: ``dat_real`` → ``real_data``, ``random_state`` → ``random_seed``.

.. _v31-changes:

Changes in v3.1
---------------

SyNG-BTS v3.1 mainly tightens data and evaluation contracts.

- ``resolve_data()`` now returns ``(dataframe, groups_or_none)``.
  Update callers to unpack both values.
- Group labels are explicit API inputs (``groups=``, ``source_groups=``,
  ``target_groups=``); do not pass metadata columns inside feature data.
- User feature DataFrames are strict: numeric columns only; ``groups`` and
  ``samples`` columns are rejected.
- Bundled datasets are stored as Parquet (transparent via loader APIs).
- Generated/reconstructed outputs are returned in count scale when
  ``apply_log=True``.
- ``SyngResult``/``PilotResult`` include ``original_data``.
- ``transfer()`` is single-run only and returns a ``SyngResult``; pilot controls
  (``pilot_size``, ``n_draws``) and ``source_size`` are removed.
- ``evaluation()`` uses ``real_groups``/``generated_groups``;
  ``load_data()`` and ``load_dataset()`` are removed.

.. code-block:: python

   # v3.0
   data = resolve_data("SKCMPositive_4")

   # v3.1
   data, groups = resolve_data("SKCMPositive_4")

v3.1 Quick Migration Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Unpack loader returns: ``data, groups = resolve_data(...)``.
2. Pass labels via ``groups=`` (or transfer equivalents), not DataFrame columns.
3. Remove ``groups`` / ``samples`` columns from user input DataFrames.
4. Replace ``group_names=`` with ``real_groups=`` / ``generated_groups=``.
5. Replace ``load_data()`` / ``load_dataset()`` with ``resolve_data()``.

.. _v32-changes:

Changes in v3.2
---------------

SyNG-BTS v3.2 is an **internal refactor** with no breaking changes to public
API outputs or result schemas.

- :func:`~syng_bts.transfer` is now **single-run only**: the parameters
  ``pilot_size``, ``n_draws``, and ``source_size`` are removed from its
  signature. Use :func:`~syng_bts.pilot_study` for multi-draw sweeps.
- Training orchestration is centralized in an internal ``orchestrate_training()``
  helper that consolidates early-stop resolution, data augmentation, batch-size
  computation, and model dispatch — no change in training behavior.
- ``_train_model()``, ``_run_generate()``, and ``_run_pilot()`` private helpers
  are removed; all code paths now use the unified orchestrator.

v3.2 Migration Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Remove ``pilot_size``, ``n_draws``, and ``source_size`` arguments from any
   ``transfer()`` calls; these parameters no longer exist.
2. No other public API changes — all other call sites are unaffected.
