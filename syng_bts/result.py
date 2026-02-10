"""
Result objects for SyNG-BTS experiment outputs.

This module defines result classes that experiment functions return
instead of writing directly to disk. Results carry generated data, loss logs,
reconstructed data, and trained model state, which are all accessible as attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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
    >>> fig = result.plot_loss()
    """

    generated_data: pd.DataFrame
    loss: pd.DataFrame
    reconstructed_data: pd.DataFrame | None = None
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
        headers. Model state is saved as a ``.pt`` file.

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
            ``"reconstructed"``, ``"model"``) to the written file path.
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

        # Model state dict
        if self.model_state is not None:
            model_path = out / f"{stem}_model.pt"
            torch.save(self.model_state, model_path)
            paths["model"] = model_path

        return paths

    def plot_loss(self, averaging_iterations: int = 100) -> plt.Figure:
        """Plot the training loss curve(s).

        Parameters
        ----------
        averaging_iterations : int
            Window size for the running-average overlay.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object (not shown; caller decides when to display).
        """
        fig, ax = plt.subplots()
        for col in self.loss.columns:
            values = self.loss[col].to_numpy()
            ax.plot(values, alpha=0.4, label=col)
            if len(values) > averaging_iterations:
                kernel = np.ones(averaging_iterations) / averaging_iterations
                smoothed = np.convolve(values, kernel, mode="valid")
                ax.plot(
                    range(averaging_iterations - 1, len(values)),
                    smoothed,
                    label=f"{col} (avg)",
                )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()
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
        if "seed" in meta:
            parts.append(f"Random seed: {meta['seed']}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        n_gen, n_feat = self.generated_data.shape
        model = self.metadata.get("model", "?")
        has_recon = self.reconstructed_data is not None
        has_model = self.model_state is not None
        return (
            f"SyngResult(model={model!r}, "
            f"generated={n_gen}×{n_feat}, "
            f"loss_cols={list(self.loss.columns)}, "
            f"has_reconstructed={has_recon}, "
            f"has_model_state={has_model})"
        )


@dataclass
class PilotResult:
    """Result of a pilot study run across multiple pilot sizes and draws.

    Attributes
    ----------
    runs : dict[tuple[int, int], SyngResult]
        Mapping of ``(pilot_size, draw_index)`` → individual run result.
        ``draw_index`` is 1-based (1 through 5).
    metadata : dict
        Shared metadata across all runs (model, data dimensions, etc.).

    Examples
    --------
    >>> result = pilot_study(data="SKCMPositive_4", pilot_size=[50, 100], ...)
    >>> result.runs[(50, 1)].generated_data.head()
    >>> result.save("./pilot_output/")
    """

    runs: dict[tuple[int, int], SyngResult] = field(default_factory=dict)
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
        return all_paths

    def plot_loss(self) -> dict[tuple[int, int], plt.Figure]:
        """Plot loss curves for every run.

        Returns
        -------
        dict[tuple[int, int], Figure]
            Mapping of ``(pilot_size, draw)`` → figure.
        """
        return {key: result.plot_loss() for key, result in sorted(self.runs.items())}

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
        return (
            f"PilotResult(model={model!r}, n_runs={n_runs}, pilot_sizes={pilot_sizes})"
        )
