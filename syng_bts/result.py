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
    model_state : dict or None
        The ``state_dict()`` of the trained model, suitable for
        ``torch.save()`` / ``torch.load()``.
    metadata : dict
        Run parameters and summary statistics, e.g. model name,
        kl_weight, seed, epoch count, input data dimensions.

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
        running_average_window: int = 50,
        x_axis: str = "iterations",
    ) -> dict[str, plt.Figure]:
        """Plot the training loss curve(s), one figure per loss column.

        Each returned figure shows the raw loss series (``alpha=0.4``)
        and a running-average overlay.

        Parameters
        ----------
        running_average_window : int
            Window size for the running-average overlay. Must be > 0.
        x_axis : str
            ``"iterations"`` (default) numbers data points 0…N-1.
            ``"epochs"`` maps the x-axis to epoch space using
            ``metadata["num_epochs"]`` (must be present and > 0).

        Returns
        -------
        dict[str, matplotlib.figure.Figure]
            ``{loss_column_name: figure}`` for every column in ``self.loss``.

        Raises
        ------
        ValueError
            If *running_average_window* ≤ 0, if *x_axis* is not
            ``"iterations"`` or ``"epochs"``, if ``x_axis="epochs"``
            but ``metadata["num_epochs"]`` is missing or ≤ 0, or if
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
            raw_num_epochs = self.metadata.get("num_epochs")
            if raw_num_epochs is None or isinstance(raw_num_epochs, bool):
                raise ValueError(
                    "x_axis='epochs' requires metadata['num_epochs'] > 0, "
                    f"got {raw_num_epochs!r}"
                )
            try:
                num_epochs = float(raw_num_epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "x_axis='epochs' requires metadata['num_epochs'] > 0, "
                    f"got {raw_num_epochs!r}"
                ) from exc
            if num_epochs <= 0:
                raise ValueError(
                    "x_axis='epochs' requires metadata['num_epochs'] > 0, "
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

    def plot_heatmap(self, which: str = "generated") -> plt.Figure:
        """Render a seaborn heatmap of generated or reconstructed data.

        Parameters
        ----------
        which : str
            ``"generated"`` or ``"reconstructed"``.

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

        fig, ax = plt.subplots()
        sns.heatmap(df.to_numpy(), cmap="YlGnBu", ax=ax)
        ax.set_title(f"{which.capitalize()} data")
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
        if "seed" in meta:
            parts.append(f"Random seed: {meta['seed']}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        n_gen, n_feat = self.generated_data.shape
        model = self.metadata.get("model", "?")
        has_recon = self.reconstructed_data is not None
        has_original = self.original_data is not None
        has_model = self.model_state is not None
        return (
            f"SyngResult(model={model!r}, "
            f"generated={n_gen}×{n_feat}, "
            f"loss_cols={list(self.loss.columns)}, "
            f"has_reconstructed={has_recon}, "
            f"has_original={has_original}, "
            f"has_model_state={has_model})"
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

        meta_path = d / f"{stem}_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            # Restore tuples that were serialised as lists
            if "input_shape" in metadata and isinstance(metadata["input_shape"], list):
                metadata["input_shape"] = tuple(metadata["input_shape"])
        else:
            metadata = {}

        return cls(
            generated_data=generated_data,
            loss=loss,
            reconstructed_data=reconstructed_data,
            original_data=original_data,
            model_state=model_state,
            metadata=metadata,
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
        aggregate: bool = False,
        running_average_window: int = 50,
        x_axis: str = "iterations",
    ) -> dict[tuple[int, int], dict[str, plt.Figure]] | dict[str, plt.Figure]:
        """Plot loss curves for every run.

        Parameters
        ----------
        aggregate : bool
            When ``True``, produce one figure per loss column with all
            runs overlaid, colour-coded by ``(pilot_size, draw)``.
            When ``False`` (default), return per-run dicts of per-column
            figures.
        running_average_window : int
            Window size for the running-average overlay (per-run mode
            only). Must be > 0.
        x_axis : str
            ``"iterations"`` or ``"epochs"``. In per-run mode this is
            forwarded to each ``SyngResult.plot_loss()`` call. In
            aggregate mode it controls the x-axis for each run overlay.

        Returns
        -------
        dict[tuple[int, int], dict[str, Figure]] or dict[str, Figure]
            Per-run nested dict when ``aggregate=False``;
            ``{column: Figure}`` when ``aggregate=True``.
        """
        if not aggregate:
            return {
                key: result.plot_loss(
                    running_average_window=running_average_window,
                    x_axis=x_axis,
                )
                for key, result in sorted(self.runs.items())
            }

        # --- Aggregate mode: one figure per loss column ---
        # Collect all column names present across runs.
        all_columns: list[str] = []
        seen: set[str] = set()
        for result in self.runs.values():
            for col in result.loss.columns:
                if col not in seen:
                    all_columns.append(col)
                    seen.add(col)

        figures: dict[str, plt.Figure] = {}
        cmap = plt.colormaps["tab10"]
        for col in all_columns:
            fig, ax = plt.subplots()
            colour_idx = 0
            for (ps, draw), result in sorted(self.runs.items()):
                if col not in result.loss.columns:
                    continue
                values = result.loss[col].to_numpy()
                if x_axis == "epochs":
                    raw_num_epochs = result.metadata.get("num_epochs")
                    if raw_num_epochs is None or isinstance(raw_num_epochs, bool):
                        raise ValueError(
                            "x_axis='epochs' requires metadata['num_epochs'] > 0 "
                            f"for run {(ps, draw)}, got {raw_num_epochs!r}"
                        )
                    try:
                        num_epochs = float(raw_num_epochs)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "x_axis='epochs' requires metadata['num_epochs'] > 0 "
                            f"for run {(ps, draw)}, got {raw_num_epochs!r}"
                        ) from exc
                    if num_epochs <= 0:
                        raise ValueError(
                            "x_axis='epochs' requires metadata['num_epochs'] > 0 "
                            f"for run {(ps, draw)}, got {raw_num_epochs!r}"
                        )
                    x = np.linspace(0, num_epochs, len(values))
                else:
                    x = np.arange(len(values))
                colour = cmap(colour_idx % 10)
                colour_idx += 1
                ax.plot(
                    x,
                    alpha=0.5,
                    color=colour,
                    label=f"pilot={ps} draw={draw}",
                )
            ax.set_xlabel("Epochs" if x_axis == "epochs" else "Iterations")
            ax.set_ylabel("Loss")
            ax.set_title(f"{col} loss (aggregate)")
            ax.legend(fontsize="x-small")
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
