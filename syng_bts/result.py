"""
Result objects for SyNG-BTS experiment outputs.

This module defines result classes that experiment functions return
instead of writing directly to disk. Results carry generated data, loss logs,
reconstructed data, and trained model state, which are all accessible as attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


def _json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-safe equivalents."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


@dataclass
class SyngResult:
    """Result of a single SyNG-BTS model training and generation run.

    Attributes
    ----------
    generated_data : pd.DataFrame
        Synthetic samples with the original column names preserved.
    loss : pd.DataFrame
        Training loss log (columns depend on the model family).
    reconstructed_data : pd.DataFrame or None
        Reconstructions of the input data (AE/VAE/CVAE only).
    original_data : pd.DataFrame or None
        The full original input data.
    model_state : dict or None
        The ``state_dict()`` of the trained model, suitable for
        ``torch.save()`` / ``torch.load()``.
    metadata : dict
        Run parameters and summary statistics, e.g. model name,
        kl_weight, seed, epoch count, input data dimensions.
    original_groups : pd.Series or None
        Group labels for the original input data. Populated when
        groups were provided or bundled with the dataset.
    generated_groups : pd.Series or None
        Group labels for the generated data, derived from the
        label column produced during generation and mapped back
        to the original group values.
    reconstructed_groups : pd.Series or None
        Group labels for the reconstructed data (AE/VAE/CVAE only),
        derived from the label column and mapped back to original
        group values.

    Examples
    --------
    >>> result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)
    >>> result.generated_data.head()
    >>> result.save("./my_output/")
    >>> figs = result.plot_loss()  # dict[str, Figure]
    """

    generated_data: pd.DataFrame
    loss: pd.DataFrame
    reconstructed_data: pd.DataFrame | None = None
    original_data: pd.DataFrame | None = None
    model_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    original_groups: pd.Series | None = None
    generated_groups: pd.Series | None = None
    reconstructed_groups: pd.Series | None = None

    # Non-serialized lazy model cache (excluded from dataclass init)
    _cached_model: nn.Module | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Lazy model resolver
    # ------------------------------------------------------------------

    def _resolve_model(self) -> nn.Module:
        """Return the cached model, rebuilding from state if needed.

        Uses ``model_state`` and ``metadata["arch_params"]`` to rebuild
        the model via :func:`model_factory.rebuild_model`.  The rebuilt
        model is cached for subsequent calls.

        Returns
        -------
        nn.Module
            The trained model in ``eval()`` mode.

        Raises
        ------
        ValueError
            If ``model_state`` or ``metadata["arch_params"]`` is missing.
        """
        if self._cached_model is not None:
            return self._cached_model

        if self.model_state is None:
            raise ValueError(
                "Cannot resolve model: 'model_state' is None. "
                "Ensure the SyngResult was created with a model_state, "
                "or loaded from a directory that contains a .pt file."
            )

        arch_params = self.metadata.get("arch_params")
        if arch_params is None:
            raise ValueError(
                "Cannot resolve model: 'arch_params' is missing from metadata. "
                "Ensure the SyngResult was created by generate() v3.1+ or "
                "loaded from a directory with the metadata JSON."
            )

        from .model_factory import rebuild_model

        self._cached_model = rebuild_model(arch_params, self.model_state)
        return self._cached_model

    # ------------------------------------------------------------------
    # Post-training generation
    # ------------------------------------------------------------------

    def generate_new_samples(
        self,
        n: int,
        *,
        mode: str = "new",
    ) -> SyngResult:
        """Generate new synthetic samples from the trained model.

        This method reuses the same generation and post-processing path
        as :func:`generate`, applying the same inverse-log transform
        and column naming.

        Parameters
        ----------
        n : int
            Number of new samples to generate.
        mode : str
            How to incorporate the new samples:

            - ``"new"`` (default): return a **new** ``SyngResult`` whose
              ``generated_data`` contains only the newly generated samples.
              All other fields (loss, metadata, model_state, etc.) are
              copied from ``self``.
            - ``"overwrite"``: **replace** ``self.generated_data`` with the
              new samples and return ``self``.
            - ``"append"``: **append** the new samples to
              ``self.generated_data`` and return ``self``.

        Returns
        -------
        SyngResult
            The result containing the new samples (see *mode*).

        Raises
        ------
        ValueError
            If ``model_state`` is ``None``, ``arch_params`` is missing
            from metadata, or *mode* is not one of the accepted values.

        Examples
        --------
        >>> result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)
        >>> new_result = result.generate_new_samples(200)
        >>> new_result.generated_data.shape[0]
        200

        >>> # After save/load round-trip:
        >>> loaded = SyngResult.load("output/")
        >>> more = loaded.generate_new_samples(100, mode="append")
        >>> more.generated_data.shape[0]  # original + 100
        """
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}")

        if mode not in ("new", "overwrite", "append"):
            raise ValueError(
                f"mode must be 'new', 'overwrite', or 'append', got {mode!r}"
            )

        from .helper_training import TrainedModel
        from .inference import run_generation

        # Resolve the trained model (lazy rebuild from state_dict)
        model = self._resolve_model()
        arch_params = self.metadata["arch_params"]

        trained = TrainedModel(
            model=model,
            model_state=self.model_state,
            arch_params=arch_params,
            log_dict={},
            epochs_trained=self.metadata.get("epochs_trained", 0),
        )

        # Generate raw samples via the unified inference dispatcher
        gen_tensor = run_generation(trained, num_samples=n)

        # Post-processing: same as generate() in core.py
        gen_np = gen_tensor.detach().numpy()
        colnames = list(self.generated_data.columns)
        modelname = arch_params.get("modelname", "")
        gen_labels: pd.Series | None = None

        # Strip the appended label/blur-label column when the generated
        # tensor has more columns than the original feature set.
        if gen_np.shape[1] > len(colnames):
            gen_labels = pd.Series(gen_np[:, -1], name="label")
            gen_np = gen_np[:, :-1]

        gen_df = pd.DataFrame(gen_np, columns=colnames)

        if gen_df.columns.tolist() != colnames:
            raise RuntimeError(
                "Column order mismatch in generated_data. "
                "This is an internal error; please report it."
            )

        # Inverse log transform if the original run used apply_log
        apply_log = self.metadata.get("apply_log", False)
        if apply_log:
            from .helper_utils import inverse_log2

            gen_df = inverse_log2(gen_df)

        # Derive generated_groups from labels + group_mapping
        group_mapping = self.metadata.get("group_mapping")
        new_gen_groups: pd.Series | None = None
        if group_mapping is not None and gen_labels is not None:
            from .core import _labels_to_groups

            new_gen_groups = _labels_to_groups(
                gen_labels, group_mapping, modelname=modelname
            )

        def _as_series_or_none(value: Any) -> pd.Series | None:
            if value is None:
                return None
            if isinstance(value, pd.Series):
                return value.reset_index(drop=True)
            if isinstance(value, np.ndarray):
                return pd.Series(value, name="label")
            if isinstance(value, list):
                return pd.Series(value, name="label")
            return None

        # Apply mode
        if mode == "new":
            new_metadata = self.metadata.copy()
            new_metadata["generated_labels"] = gen_labels
            return SyngResult(
                generated_data=gen_df,
                loss=self.loss.copy(),
                reconstructed_data=(
                    self.reconstructed_data.copy()
                    if self.reconstructed_data is not None
                    else None
                ),
                original_data=(
                    self.original_data.copy()
                    if self.original_data is not None
                    else None
                ),
                model_state=self.model_state,
                metadata=new_metadata,
                original_groups=(
                    self.original_groups.copy()
                    if self.original_groups is not None
                    else None
                ),
                generated_groups=new_gen_groups,
                reconstructed_groups=(
                    self.reconstructed_groups.copy()
                    if self.reconstructed_groups is not None
                    else None
                ),
            )
        elif mode == "overwrite":
            self.generated_data = gen_df
            self.metadata["generated_labels"] = gen_labels
            self.generated_groups = new_gen_groups
            return self
        else:  # append
            self.generated_data = pd.concat(
                [self.generated_data, gen_df], ignore_index=True
            )

            if gen_labels is None:
                self.metadata["generated_labels"] = None
            else:
                old_labels = _as_series_or_none(self.metadata.get("generated_labels"))
                if old_labels is None:
                    self.metadata["generated_labels"] = gen_labels
                else:
                    self.metadata["generated_labels"] = pd.concat(
                        [old_labels, gen_labels], ignore_index=True
                    )

            # Append generated groups
            if new_gen_groups is None:
                self.generated_groups = None
            elif self.generated_groups is None:
                self.generated_groups = new_gen_groups
            else:
                self.generated_groups = pd.concat(
                    [self.generated_groups, new_gen_groups], ignore_index=True
                )
            return self

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        prefix: str | None = None,
    ) -> dict[str, Path]:
        """Save all non-None results to *output_dir*.

        Files are written into a single flat directory. CSVs include column
        headers. Model state is saved as a ``.pt`` file. Metadata is written
        as a human-readable JSON file.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write files into (created if it does not exist).
        prefix : str or None
            Optional filename prefix. When ``None``, uses
            ``metadata["dataname"]`` if available, otherwise ``"syng"``.

        Returns
        -------
        dict[str, Path]
            Mapping of output type (``"generated"``, ``"loss"``,
            ``"reconstructed"``, ``"model"``, ``"metadata"``) to the
            written file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = self.metadata.get("dataname", "syng")

        model_tag = self.metadata.get("model", "")
        if model_tag:
            stem = f"{prefix}_{model_tag}"
        else:
            stem = prefix

        paths: dict[str, Path] = {}

        # Generated data
        gen_path = out / f"{stem}_generated.csv"
        self.generated_data.to_csv(gen_path, index=False)
        paths["generated"] = gen_path

        # Loss log
        loss_path = out / f"{stem}_loss.csv"
        self.loss.to_csv(loss_path, index=False)
        paths["loss"] = loss_path

        # Reconstructed data
        if self.reconstructed_data is not None:
            recon_path = out / f"{stem}_reconstructed.csv"
            self.reconstructed_data.to_csv(recon_path, index=False)
            paths["reconstructed"] = recon_path

        # Original data
        if self.original_data is not None:
            orig_path = out / f"{stem}_original.csv"
            self.original_data.to_csv(orig_path, index=True)
            paths["original"] = orig_path

        # Group attributes
        if self.original_groups is not None:
            og_path = out / f"{stem}_original_groups.csv"
            self.original_groups.to_frame().to_csv(og_path, index=False)
            paths["original_groups"] = og_path

        if self.generated_groups is not None:
            gg_path = out / f"{stem}_generated_groups.csv"
            self.generated_groups.to_frame().to_csv(gg_path, index=False)
            paths["generated_groups"] = gg_path

        if self.reconstructed_groups is not None:
            rg_path = out / f"{stem}_reconstructed_groups.csv"
            self.reconstructed_groups.to_frame().to_csv(rg_path, index=False)
            paths["reconstructed_groups"] = rg_path

        # Model state dict
        if self.model_state is not None:
            model_path = out / f"{stem}_model.pt"
            torch.save(self.model_state, model_path)
            paths["model"] = model_path

        # Metadata as human-readable JSON
        if self.metadata:
            meta_path = out / f"{stem}_metadata.json"
            meta_path.write_text(
                json.dumps(self.metadata, indent=2, default=_json_serializable),
                encoding="utf-8",
            )
            paths["metadata"] = meta_path

        return paths

    def plot_loss(
        self,
        running_average_window: int = 25,
        x_axis: str = "epochs",
    ) -> dict[str, plt.Figure]:
        """Plot the training loss curve(s), one figure per loss column.

        Each returned figure shows the raw loss series (``alpha=0.4``)
        and a running-average overlay.

        Parameters
        ----------
        running_average_window : int
            Window size for the running-average overlay. Must be > 0.
            Default: 25.
        x_axis : str
            ``"epochs"`` (default) maps the x-axis to epoch space using
            ``metadata["epochs_trained"]`` (must be present and > 0).
            ``"iterations"`` numbers data points 0…N-1.

        Returns
        -------
        dict[str, matplotlib.figure.Figure]
            ``{loss_column_name: figure}`` for every column in ``self.loss``.

        Raises
        ------
        ValueError
            If *running_average_window* ≤ 0, if *x_axis* is not
            ``"iterations"`` or ``"epochs"``, if ``x_axis="epochs"``
            but ``metadata["epochs_trained"]`` is missing or ≤ 0, or if
            the window is larger than a loss series.
        """
        if running_average_window <= 0:
            raise ValueError(
                f"running_average_window must be > 0, got {running_average_window}"
            )
        if x_axis not in ("iterations", "epochs"):
            raise ValueError(f"x_axis must be 'iterations' or 'epochs', got {x_axis!r}")
        num_epochs: float | None = None
        if x_axis == "epochs":
            raw_num_epochs = self.metadata.get("epochs_trained")
            if raw_num_epochs is None or isinstance(raw_num_epochs, bool):
                raise ValueError(
                    "x_axis='epochs' requires metadata['epochs_trained'] > 0, "
                    f"got {raw_num_epochs!r}"
                )
            try:
                num_epochs = float(raw_num_epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "x_axis='epochs' requires metadata['epochs_trained'] > 0, "
                    f"got {raw_num_epochs!r}"
                ) from exc
            if num_epochs <= 0:
                raise ValueError(
                    "x_axis='epochs' requires metadata['epochs_trained'] > 0, "
                    f"got {raw_num_epochs!r}"
                )

        figures: dict[str, plt.Figure] = {}
        for col in self.loss.columns:
            values = self.loss[col].to_numpy()

            if running_average_window > len(values):
                raise ValueError(
                    f"running_average_window ({running_average_window}) is larger "
                    f"than the '{col}' series length ({len(values)})"
                )

            fig, ax = plt.subplots()

            # --- Build x-coordinates ---
            if x_axis == "epochs":
                assert num_epochs is not None
                x = np.linspace(0, num_epochs, len(values))
                ax.set_xlabel("Epochs")
            else:
                x = np.arange(len(values))
                ax.set_xlabel("Iterations")

            ax.plot(x, values, alpha=0.4, label=f"{col} (raw)")

            # --- Running average ---
            kernel = np.ones(running_average_window) / running_average_window
            smoothed = np.convolve(values, kernel, mode="valid")
            offset = running_average_window - 1
            ax.plot(
                x[offset:],
                smoothed,
                label=f"{col} (avg, w={running_average_window})",
            )

            ax.set_ylabel("Loss")
            ax.set_title(f"{col} loss")

            # --- Y-axis scaling: ignore the initial spike ---
            n = len(values)
            skip = n // 2 if n < 1001 else 1000
            if n > skip:
                later_max = float(np.max(values[skip:]))
                if later_max > 0:
                    ax.set_ylim([0, later_max * 1.5])

            ax.legend()
            fig.tight_layout()
            figures[col] = fig

        return figures

    def plot_heatmap(
        self, which: str = "generated", log_scale: bool = True
    ) -> plt.Figure:
        """Render a seaborn heatmap of generated or reconstructed data.

        Parameters
        ----------
        which : str
            ``"generated"``, ``"reconstructed"``, or ``"original"``.
        log_scale : bool
            If ``True`` (default), apply ``log2(x + 1)`` scaling to the
            data before plotting.  This compresses wide-ranging values
            and often produces more readable heatmaps.

        Returns
        -------
        matplotlib.figure.Figure
            The heatmap figure (not shown; caller decides when to display).

        Raises
        ------
        ValueError
            If *which* is ``"reconstructed"`` but no reconstructed data exists,
            or if *which* is not a recognised value.
        """
        if which == "generated":
            df = self.generated_data
        elif which == "reconstructed":
            if self.reconstructed_data is None:
                raise ValueError(
                    "No reconstructed data available in this result. "
                    "Reconstructed data is only produced by AE/VAE/CVAE models."
                )
            df = self.reconstructed_data
        elif which == "original":
            if self.original_data is None:
                raise ValueError(
                    "No original data available in this result. "
                    "Pass original_data when constructing the result."
                )
            df = self.original_data
        else:
            raise ValueError(
                f"Unknown value which={which!r}; "
                f"expected 'generated', 'reconstructed', or 'original'."
            )

        data = df.to_numpy()
        if log_scale:
            data = np.log2(data + 1)

        fig, ax = plt.subplots()
        sns.heatmap(data, cmap="YlGnBu", ax=ax)
        title = f"{which.capitalize()} data"
        if log_scale:
            title += " (log2)"
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def summary(self) -> str:
        """Return a short textual summary of this result.

        Returns
        -------
        str
            A paragraph describing the run dimensions, epoch count,
            and final loss values.
        """
        meta = self.metadata
        model = meta.get("model", "unknown")
        n_gen, n_feat = self.generated_data.shape
        epochs = meta.get("epochs_trained", "?")

        # Summarise final loss values
        final_losses = {
            col: f"{self.loss[col].iloc[-1]:.4f}" for col in self.loss.columns
        }
        loss_str = ", ".join(f"{k}={v}" for k, v in final_losses.items())

        parts = [
            f"Model: {model}",
            f"Generated data: {n_gen} samples × {n_feat} features",
            f"Epochs trained: {epochs}",
            f"Final loss: {loss_str}",
        ]
        if self.reconstructed_data is not None:
            r, c = self.reconstructed_data.shape
            parts.append(f"Reconstructed data: {r} rows × {c} cols")
        if self.original_data is not None:
            r, c = self.original_data.shape
            parts.append(f"Original data: {r} rows × {c} cols")
        if self.original_groups is not None:
            n_classes = self.original_groups.nunique()
            parts.append(f"Groups: {n_classes} classes")
        if "seed" in meta:
            parts.append(f"Random seed: {meta['seed']}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        n_gen, n_feat = self.generated_data.shape
        model = self.metadata.get("model", "?")
        has_recon = self.reconstructed_data is not None
        has_original = self.original_data is not None
        has_model = self.model_state is not None
        has_groups = self.original_groups is not None
        return (
            f"SyngResult(model={model!r}, "
            f"generated={n_gen}×{n_feat}, "
            f"loss_cols={list(self.loss.columns)}, "
            f"has_reconstructed={has_recon}, "
            f"has_original={has_original}, "
            f"has_model_state={has_model}, "
            f"has_groups={has_groups})"
        )

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: str | Path,
        prefix: str | None = None,
    ) -> SyngResult:
        """Load a previously saved ``SyngResult`` from disk.

        Parameters
        ----------
        directory : str or Path
            Directory that contains the saved files.
        prefix : str or None
            The filename stem (everything before ``_generated.csv``).
            When ``None``, auto-detected from ``*_generated.csv`` files
            in the directory; exactly one match is required.

        Returns
        -------
        SyngResult
            Reconstructed result with all available artifacts.

        Raises
        ------
        FileNotFoundError
            If the required ``*_generated.csv`` or ``*_loss.csv`` file
            is missing.
        ValueError
            If *prefix* is ``None`` and zero or more than one
            ``*_generated.csv`` file is found (ambiguous).
        """
        d = Path(directory)

        if prefix is None:
            candidates = sorted(d.glob("*_generated.csv"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"No *_generated.csv files found in {d}")
            if len(candidates) > 1:
                stems = [c.name.removesuffix("_generated.csv") for c in candidates]
                looks_like_pilot_dir = any(
                    "_pilot" in c.name and "_draw" in c.name for c in candidates
                )
                if looks_like_pilot_dir:
                    raise ValueError(
                        "Multiple generated files found and directory appears to "
                        f"contain PilotResult outputs: {stems}. "
                        "SyngResult.load() loads one run at a time; pass prefix "
                        "for a specific run stem, e.g. '<dataname>_pilot50_draw1_<model>'."
                    )
                raise ValueError(
                    f"Multiple generated files found in {d}: {stems}. "
                    "Specify 'prefix' to disambiguate."
                )
            stem = candidates[0].name.removesuffix("_generated.csv")
        else:
            stem = prefix

        # --- Required files ---
        gen_path = d / f"{stem}_generated.csv"
        if not gen_path.exists():
            raise FileNotFoundError(f"Required file not found: {gen_path}")
        generated_data = pd.read_csv(gen_path)

        loss_path = d / f"{stem}_loss.csv"
        if not loss_path.exists():
            raise FileNotFoundError(f"Required file not found: {loss_path}")
        loss = pd.read_csv(loss_path)

        # --- Optional files ---
        recon_path = d / f"{stem}_reconstructed.csv"
        reconstructed_data = pd.read_csv(recon_path) if recon_path.exists() else None

        orig_path = d / f"{stem}_original.csv"
        original_data = (
            pd.read_csv(orig_path, index_col=0) if orig_path.exists() else None
        )

        model_path = d / f"{stem}_model.pt"
        model_state = (
            torch.load(model_path, weights_only=False) if model_path.exists() else None
        )

        # --- Group sidecar files ---
        def _load_groups(suffix: str) -> pd.Series | None:
            path = d / f"{stem}_{suffix}.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
            return df.iloc[:, 0].rename("group")

        original_groups = _load_groups("original_groups")
        generated_groups = _load_groups("generated_groups")
        reconstructed_groups = _load_groups("reconstructed_groups")

        meta_path = d / f"{stem}_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            # Restore tuples that were serialised as lists
            if "input_shape" in metadata and isinstance(metadata["input_shape"], list):
                metadata["input_shape"] = tuple(metadata["input_shape"])
            # Restore group_mapping keys from JSON strings to ints
            if "group_mapping" in metadata and isinstance(
                metadata["group_mapping"], dict
            ):
                metadata["group_mapping"] = {
                    int(k): v for k, v in metadata["group_mapping"].items()
                }
        else:
            metadata = {}

        return cls(
            generated_data=generated_data,
            loss=loss,
            reconstructed_data=reconstructed_data,
            original_data=original_data,
            model_state=model_state,
            metadata=metadata,
            original_groups=original_groups,
            generated_groups=generated_groups,
            reconstructed_groups=reconstructed_groups,
        )


@dataclass
class PilotResult:
    """Result of a pilot study run across multiple pilot sizes and draws.

    Attributes
    ----------
    runs : dict[tuple[int, int], SyngResult]
        Mapping of ``(pilot_size, draw_index)`` → individual run result.
        ``draw_index`` is 1-based (1 through 5).
    original_data : pd.DataFrame or None
        The full original input data (before subsetting).
    metadata : dict
        Shared metadata across all runs (model, data dimensions, etc.).

    Examples
    --------
    >>> result = pilot_study(data="SKCMPositive_4", pilot_size=[50, 100], ...)
    >>> result.runs[(50, 1)].generated_data.head()
    >>> result.save("./pilot_output/")
    """

    runs: dict[tuple[int, int], SyngResult] = field(default_factory=dict)
    original_data: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        prefix: str | None = None,
    ) -> dict[tuple[int, int], dict[str, Path]]:
        """Save all individual run results to *output_dir*.

        Each run is saved with a filename that encodes the pilot size and
        draw index.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write files into (created if it does not exist).
        prefix : str or None
            Optional filename prefix. Falls back to
            ``metadata["dataname"]`` or ``"syng"``.

        Returns
        -------
        dict[tuple[int, int], dict[str, Path]]
            Nested mapping: ``(pilot_size, draw) → {output_type → path}``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = self.metadata.get("dataname", "syng")

        all_paths: dict[tuple[int, int], dict[str, Path]] = {}
        for (pilot_size, draw), result in sorted(self.runs.items()):
            run_prefix = f"{prefix}_pilot{pilot_size}_draw{draw}"
            all_paths[(pilot_size, draw)] = result.save(out, prefix=run_prefix)

        # Save top-level original data
        if self.original_data is not None:
            orig_path = out / f"{prefix}_original.csv"
            self.original_data.to_csv(orig_path, index=True)

        # Save top-level metadata
        if self.metadata:
            meta_path = out / f"{prefix}_pilot_metadata.json"
            meta_path.write_text(
                json.dumps(self.metadata, indent=2, default=_json_serializable),
                encoding="utf-8",
            )

        return all_paths

    def plot_loss(
        self,
        style: str = "overlay_runs",
        running_average_window: int = 25,
        x_axis: str = "epochs",
        truncate: bool = True,
    ) -> dict[tuple[int, int], dict[str, plt.Figure]] | dict[str, plt.Figure]:
        """Plot loss curves for every run in the pilot study.

        Parameters
        ----------
        style : str
            Plotting style for loss trajectories.

            - ``"per_run"`` (default): one figure per run per loss
              column, delegating to :meth:`SyngResult.plot_loss`.
            - ``"overlay_runs"``: overlay all runs on the same plot for each loss
              column. Only the running-average line is drawn per run
              (no raw trace) to keep the plot readable.
            - ``"mean_band"``: plot the mean loss trajectory across all
              runs for each loss column, with a shaded ±1 std band.
              Mean and std are computed on raw loss values; the mean
              line is then optionally smoothed with a running average.

            For all styles, y-axis scaling is applied to reduce the effect
            of large initial spikes (analogous to :meth:`SyngResult.plot_loss`).

        running_average_window : int
            Window size for the running-average overlay. Must be > 0.
            Default: 25.
        x_axis : str
            ``"epochs"`` (default) maps the x-axis to epoch space using
            each run's ``metadata["epochs_trained"]``.
            ``"iterations"`` numbers data points 0…N-1.
        truncate : bool
            Only relevant for ``style="mean_band"`` and ``style="overlay_runs"``.
            - ``True`` (default): only plot epochs/iterations common to
              **all** runs (truncate to the shortest run).
            - ``False``: plot all epochs/iterations; statistics are
              computed from whichever runs still have data at each point.

        Returns
        -------
        dict[tuple[int, int], dict[str, Figure]] or dict[str, Figure]
            ``style="per_run"``: nested dict keyed by
            ``(pilot_size, draw)`` → ``{column: Figure}``.
            ``style="overlay_runs"`` or ``style="mean_band"``: flat dict
            ``{column: Figure}``.

        Raises
        ------
        ValueError
            If *style* is not one of the accepted values, if
            *running_average_window* ≤ 0, or if *x_axis* is invalid.

        Examples
        --------
        >>> figs = pilot_result.plot_loss(style="overlay_runs")
        >>> figs = pilot_result.plot_loss(style="mean_band", truncate=False)
        """
        # --- Input validation ---
        valid_styles = ("per_run", "overlay_runs", "mean_band")
        if style not in valid_styles:
            raise ValueError(f"style must be one of {valid_styles!r}, got {style!r}")
        if running_average_window <= 0:
            raise ValueError(
                f"running_average_window must be > 0, got {running_average_window}"
            )
        if x_axis not in ("iterations", "epochs"):
            raise ValueError(f"x_axis must be 'iterations' or 'epochs', got {x_axis!r}")

        # --- style="per_run": delegate to SyngResult.plot_loss() ---
        if style == "per_run":
            return {
                key: result.plot_loss(
                    running_average_window=running_average_window,
                    x_axis=x_axis,
                )
                for key, result in sorted(self.runs.items())
            }

        # --- Shared helpers for "overlay_runs" and "mean_band" ---

        def _collect_loss_columns() -> list[str]:
            """Return the ordered union of loss column names across runs."""
            cols: list[str] = []
            seen: set[str] = set()
            for result in self.runs.values():
                for col in result.loss.columns:
                    if col not in seen:
                        cols.append(col)
                        seen.add(col)
            return cols

        def _resolve_epochs(result: SyngResult, key: tuple[int, int]) -> float:
            """Validate and return ``epochs_trained`` for a single run."""
            raw = result.metadata.get("epochs_trained")
            if raw is None or isinstance(raw, bool):
                raise ValueError(
                    f"x_axis='epochs' requires metadata['epochs_trained'] > 0 "
                    f"for run {key}, got {raw!r}"
                )
            try:
                val = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"x_axis='epochs' requires metadata['epochs_trained'] > 0 "
                    f"for run {key}, got {raw!r}"
                ) from exc
            if val <= 0:
                raise ValueError(
                    f"x_axis='epochs' requires metadata['epochs_trained'] > 0 "
                    f"for run {key}, got {raw!r}"
                )
            return val

        def _build_x(length: int, num_epochs: float | None) -> np.ndarray:
            """Build x-coordinate array."""
            if x_axis == "epochs" and num_epochs is not None:
                return np.linspace(0, num_epochs, length)
            return np.arange(length)

        def _apply_ylim_scaling(ax, values: np.ndarray) -> None:
            """Set y-axis limits to suppress the initial spike."""
            n = len(values)
            skip = n // 2 if n < 1001 else 1000
            if n > skip:
                later_max = float(np.nanmax(values[skip:]))
                if later_max > 0:
                    ax.set_ylim([0, later_max * 1.5])

        def _later_max_for_scaling(values: np.ndarray) -> float | None:
            """Return post-spike max used for y-axis scaling, if available."""
            n = len(values)
            skip = n // 2 if n < 1001 else 1000
            if n <= skip:
                return None
            later_max = float(np.nanmax(values[skip:]))
            if later_max <= 0:
                return None
            return later_max

        all_columns = _collect_loss_columns()

        # --- style="overlay_runs": overlay running-average per run ---
        if style == "overlay_runs":
            figures: dict[str, plt.Figure] = {}
            cmap = plt.colormaps["tab10"]
            kernel = np.ones(running_average_window) / running_average_window

            for col in all_columns:
                fig, ax = plt.subplots()
                colour_idx = 0
                later_max_values: list[float] = []

                eligible_runs: list[tuple[tuple[int, int], SyngResult]] = [
                    (key, result)
                    for key, result in sorted(self.runs.items())
                    if col in result.loss.columns
                ]

                if not eligible_runs:
                    ax.set_xlabel("Epochs" if x_axis == "epochs" else "Iterations")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"{col} loss (overlay_runs)")
                    fig.tight_layout()
                    figures[col] = fig
                    continue

                # If truncating, use the common prefix length across runs.
                truncate_len: int | None = None
                common_epochs: float | None = None
                if truncate:
                    truncate_len = min(
                        len(result.loss[col]) for _key, result in eligible_runs
                    )
                    if x_axis == "epochs":
                        common_epochs = min(
                            _resolve_epochs(result, key)
                            for key, result in eligible_runs
                        )

                for (ps, draw), result in eligible_runs:
                    values = result.loss[col].to_numpy()
                    if truncate_len is not None:
                        values = values[:truncate_len]

                    num_epochs: float | None = None
                    if x_axis == "epochs":
                        if truncate and common_epochs is not None:
                            num_epochs = common_epochs
                        else:
                            num_epochs = _resolve_epochs(result, (ps, draw))

                    x_full = _build_x(len(values), num_epochs)

                    colour = cmap(colour_idx % 10)
                    colour_idx += 1

                    # Only plot running-average values when possible; otherwise
                    # fall back to raw values (short series).
                    if running_average_window <= len(values):
                        smoothed = np.convolve(values, kernel, mode="valid")
                        offset = running_average_window - 1
                        ax.plot(
                            x_full[offset:],
                            smoothed,
                            alpha=0.7,
                            color=colour,
                            label=f"pilot={ps} draw={draw}",
                        )
                        later_max = _later_max_for_scaling(smoothed)
                        if later_max is not None:
                            later_max_values.append(later_max)
                    else:
                        ax.plot(
                            x_full,
                            values,
                            alpha=0.7,
                            color=colour,
                            label=f"pilot={ps} draw={draw} (raw)",
                        )
                        later_max = _later_max_for_scaling(values)
                        if later_max is not None:
                            later_max_values.append(later_max)

                ax.set_xlabel("Epochs" if x_axis == "epochs" else "Iterations")
                ax.set_ylabel("Loss")
                ax.set_title(f"{col} loss (overlay_runs)")
                ax.legend(fontsize="x-small")

                # Y-axis scaling: ignore each run's initial spike, then combine
                if later_max_values:
                    ax.set_ylim([0, max(later_max_values) * 1.5])

                fig.tight_layout()
                figures[col] = fig

            return figures

        # --- style="mean_band": mean ± std across runs ---
        figures = {}
        kernel = np.ones(running_average_window) / running_average_window

        for col in all_columns:
            # Gather raw loss arrays for this column
            arrays: list[np.ndarray] = []
            epoch_values: list[float] = []
            for (ps, draw), result in sorted(self.runs.items()):
                if col not in result.loss.columns:
                    continue
                arrays.append(result.loss[col].to_numpy())
                if x_axis == "epochs":
                    epoch_values.append(_resolve_epochs(result, (ps, draw)))

            if not arrays:
                continue

            # Determine target length and stack
            lengths = [len(a) for a in arrays]
            if truncate:
                target_len = min(lengths)
                stacked = np.array([a[:target_len] for a in arrays])
            else:
                target_len = max(lengths)
                stacked = np.full((len(arrays), target_len), np.nan)
                for i, a in enumerate(arrays):
                    stacked[i, : len(a)] = a

            mean_vals = np.nanmean(stacked, axis=0)
            std_vals = np.nanstd(stacked, axis=0)

            # Build x-coordinates
            if x_axis == "epochs":
                # Use the epoch range corresponding to the target length
                if truncate:
                    ref_epochs = min(epoch_values)
                else:
                    ref_epochs = max(epoch_values)
                x_full = np.linspace(0, ref_epochs, target_len)
            else:
                x_full = np.arange(target_len)

            fig, ax = plt.subplots()

            # Shaded std band (on raw values)
            ax.fill_between(
                x_full,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.3,
                color="tab:blue",
                label="±1 std",
            )

            # Smoothed mean line
            if running_average_window <= target_len:
                smoothed_mean = np.convolve(mean_vals, kernel, mode="valid")
                offset = running_average_window - 1
                ax.plot(
                    x_full[offset:],
                    smoothed_mean,
                    color="tab:blue",
                    label=f"mean (avg w={running_average_window})",
                )
            else:
                # Window too large for smoothing; plot raw mean
                ax.plot(
                    x_full,
                    mean_vals,
                    color="tab:blue",
                    label="mean",
                )

            ax.set_xlabel("Epochs" if x_axis == "epochs" else "Iterations")
            ax.set_ylabel("Loss")
            n_runs = stacked.shape[0]
            ax.set_title(f"{col} loss (mean_band, n={n_runs})")
            ax.legend()

            # Y-axis scaling from mean values
            _apply_ylim_scaling(ax, mean_vals)

            fig.tight_layout()
            figures[col] = fig

        return figures

    def summary(self) -> str:
        """Return an aggregate summary of all pilot runs.

        Returns
        -------
        str
            Multi-line summary with one line per run.
        """
        lines = [f"PilotResult: {len(self.runs)} runs"]
        model = self.metadata.get("model", "?")
        lines.append(f"Model: {model}")

        pilot_sizes = sorted({ps for ps, _ in self.runs})
        lines.append(f"Pilot sizes: {pilot_sizes}")

        if self.original_data is not None:
            r, c = self.original_data.shape
            lines.append(f"Original data: {r} rows × {c} cols")

        for key in sorted(self.runs):
            r = self.runs[key]
            n_gen = r.generated_data.shape[0]
            final_losses = {
                col: f"{r.loss[col].iloc[-1]:.4f}" for col in r.loss.columns
            }
            loss_str = ", ".join(f"{k}={v}" for k, v in final_losses.items())
            lines.append(
                f"  pilot={key[0]}, draw={key[1]}: "
                f"{n_gen} generated, final loss: {loss_str}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_runs = len(self.runs)
        pilot_sizes = sorted({ps for ps, _ in self.runs})
        model = self.metadata.get("model", "?")
        has_original = self.original_data is not None
        return (
            f"PilotResult(model={model!r}, n_runs={n_runs}, "
            f"pilot_sizes={pilot_sizes}, has_original={has_original})"
        )
