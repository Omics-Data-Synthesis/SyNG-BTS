from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap  # noqa: F401
from matplotlib.figure import Figure
from umap import UMAP

from .data_utils import resolve_data


def _coerce_groups(
    groups: pd.Series | np.ndarray | list | tuple | pd.Index | None,
    *,
    param_name: str,
    expected_len: int,
) -> pd.Series | None:
    """Coerce optional group labels to a length-validated Series."""
    if groups is None:
        return None

    if isinstance(groups, pd.Series):
        coerced = groups.reset_index(drop=True)
    elif isinstance(groups, (np.ndarray, list, tuple, pd.Index)):
        coerced = pd.Series(groups)
    else:
        raise TypeError(
            f"{param_name} must be a pandas Series, numpy array, list, tuple, "
            f"or None; got {type(groups).__name__}."
        )

    if len(coerced) != expected_len:
        raise ValueError(
            f"{param_name} length ({len(coerced)}) does not match the number "
            f"of rows in the corresponding dataset ({expected_len})."
        )

    return coerced.astype(str)


def heatmap_eval(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame | None = None,
    *,
    cmap: str = "YlGnBu",
) -> Figure:
    r"""Create a heatmap visualization comparing real and generated data.

    If only one dataset is provided, displays a single heatmap. If both
    real and generated data are provided, displays them side by side.

    Parameters
    ----------
    real_data : pd.DataFrame
        The original/real data.
    generated_data : pd.DataFrame or None, optional
        The generated/synthetic data. If ``None``, only *real_data* is plotted.
    cmap : str, default ``"YlGnBu"``
        Colormap passed to :func:`seaborn.heatmap`.

    Returns
    -------
    Figure
        The matplotlib Figure containing the heatmap(s).
    """
    # Select only numeric columns.
    real_data_plot = real_data.select_dtypes(include=["number"])
    generated_data_plot = (
        generated_data.select_dtypes(include=["number"])
        if generated_data is not None
        else None
    )

    if generated_data_plot is None:
        fig = plt.figure(figsize=(6, 6))
        ax = sns.heatmap(real_data_plot, cbar=True, cmap=cmap)
        ax.set_title("Real Data")
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")
    else:
        fig, axs = plt.subplots(
            ncols=2, figsize=(12, 6), gridspec_kw={"width_ratios": [0.5, 0.55]}
        )

        sns.heatmap(generated_data_plot, ax=axs[0], cbar=False, cmap=cmap)
        axs[0].set_title("Generated Data")
        axs[0].set_xlabel("Features")
        axs[0].set_ylabel("Samples")

        sns.heatmap(real_data_plot, ax=axs[1], cbar=True, cmap=cmap)
        axs[1].set_title("Real Data")
        axs[1].set_xlabel("Features")
        axs[1].set_ylabel("Samples")

    fig.tight_layout()
    return fig


def UMAP_eval(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame | None = None,
    *,
    groups_real: pd.Series | None = None,
    groups_generated: pd.Series | None = None,
    random_seed: int = 42,
    legend_pos: str = "best",
) -> Figure:
    r"""Create a UMAP visualization comparing real and generated data.

    Uses UMAP dimensionality reduction to visualize high-dimensional
    data in 2D, with optional group colouring.

    Parameters
    ----------
    real_data : pd.DataFrame
        The original/real data.
    generated_data : pd.DataFrame or None, optional
        The generated/synthetic data. If ``None``, only *real_data* is
        visualised.
    groups_real : pd.Series or None, optional
        Group labels for real samples. Used for styling.
    groups_generated : pd.Series or None, optional
        Group labels for generated samples. Used for styling.
    random_seed : int, default 42
        Random seed for UMAP reproducibility.
    legend_pos : str, default ``"best"``
        Legend position (``"best"``, ``"upper right"``, ``"lower left"``, …).

    Returns
    -------
    Figure
        The matplotlib Figure containing the UMAP scatter plot.
    """
    if generated_data is None:
        reducer = UMAP(random_state=random_seed)
        embedding = reducer.fit_transform(real_data.values)

        umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])

        fig, ax = plt.subplots(figsize=(10, 8))
        if groups_real is not None:
            umap_df["Group"] = groups_real.astype(str).values
            sns.scatterplot(
                data=umap_df,
                x="UMAP1",
                y="UMAP2",
                style="Group",
                palette="bright",
                ax=ax,
            )
            ax.legend(title="Group", loc=legend_pos)
            ax.set_title("UMAP Projection of Real Data with Groups")
        else:
            ax.scatter(umap_df["UMAP1"], umap_df["UMAP2"], alpha=0.7)
            ax.set_title("UMAP Projection of Real Data")

        return fig

    # Filter out features with zero variance in generated data
    non_zero_var_cols = generated_data.var(axis=0) != 0
    real_filtered = real_data.loc[:, non_zero_var_cols]
    gen_filtered = generated_data.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((real_filtered.values, gen_filtered.values))
    combined_labels = np.array(
        ["Real"] * real_filtered.shape[0] + ["Generated"] * gen_filtered.shape[0]
    )

    reducer = UMAP(random_state=random_seed)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df["Data Type"] = combined_labels

    fig, ax = plt.subplots(figsize=(10, 8))

    if groups_real is not None and groups_generated is not None:
        combined_groups = [
            str(g) for g in np.concatenate((groups_real, groups_generated))
        ]
        umap_df["Group"] = combined_groups
        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="Data Type",
            style="Group",
            palette="bright",
            ax=ax,
        )
        ax.legend(title="Data Type / Group", loc=legend_pos)
        ax.set_title("UMAP Projection of Real and Generated Data with Groups")
    else:
        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="Data Type",
            palette="bright",
            ax=ax,
        )
        ax.legend(title="Data Type", loc=legend_pos)
        ax.set_title("UMAP Projection of Real and Generated Data")

    return fig


def evaluation(
    real_data: pd.DataFrame | str | Path,
    generated_data: pd.DataFrame | str | Path,
    *,
    real_groups: pd.Series | np.ndarray | list | tuple | pd.Index | None = None,
    generated_groups: pd.Series | np.ndarray | list | tuple | pd.Index | None = None,
    n_samples: int | None = 200,
    apply_log: bool = True,
    random_seed: int = 42,
) -> dict[str, Figure]:
    r"""Preprocessing and visualization of generated vs real data.

    Loads and preprocesses the input data, then creates heatmap and UMAP
    visualizations comparing generated and real datasets.

    Parameters
    ----------
    real_data : pd.DataFrame, str, or Path
        The original/real dataset. Accepts a DataFrame, a file path, or
        a bundled dataset name (resolved via :func:`resolve_data`).
    generated_data : pd.DataFrame, str, or Path
        The generated/synthetic dataset. Same input types as *real_data*.
    real_groups : pd.Series, np.ndarray, list, tuple, pd.Index, or None, optional
        Group labels for the real samples.  When provided, takes
        precedence over any bundled groups resolved from *real_data*.
        Values are used as-is for plot labels (converted to ``str``).
    generated_groups : pd.Series, np.ndarray, list, tuple, pd.Index, or None, optional
        Group labels for the generated samples.  When provided, takes
        precedence over any bundled groups resolved from *generated_data*.
        Values are used as-is for plot labels (converted to ``str``).
    n_samples : int or None, default 200
        Number of samples from each end of the dataset to use for
        visualization (to keep UMAP fast).  If ``None``, all samples are
        used.
    apply_log : bool, default True
        Whether to apply ``log2(x + 1)`` transformation to both real
        and generated data before comparison.
    random_seed : int, default 42
        Random seed for UMAP reproducibility.

    Returns
    -------
    dict[str, Figure]
        ``{"heatmap": <Figure>, "umap": <Figure>}`` — the two evaluation
        figures.  Neither figure has been displayed; the caller decides
        when to call ``plt.show()`` or ``fig.savefig()``.
    """
    real_df, bundled_groups_real = resolve_data(real_data)
    gen_df, bundled_groups_gen = resolve_data(generated_data)

    # --- Resolve group labels -----------------------------------------------
    # Precedence: explicit parameter > bundled groups > None
    groups_real = _coerce_groups(
        real_groups,
        param_name="real_groups",
        expected_len=len(real_df),
    )
    groups_generated = _coerce_groups(
        generated_groups,
        param_name="generated_groups",
        expected_len=len(gen_df),
    )

    if groups_real is None and bundled_groups_real is not None:
        groups_real = bundled_groups_real.reset_index(drop=True).astype(str)
    if groups_generated is None and bundled_groups_gen is not None:
        groups_generated = bundled_groups_gen.reset_index(drop=True).astype(str)

    # --- Prepare numeric matrices -------------------------------------------
    real_numeric = real_df.select_dtypes(include=[np.number])
    gen_numeric = gen_df.iloc[:, : real_numeric.shape[1]].copy()
    gen_numeric.columns = real_numeric.columns

    # When apply_log is True, log-transform both real and generated data so
    # they are compared in the same (log2) scale.  Generated data is now
    # returned in count scale by the experiment API.
    if apply_log:
        real_numeric = np.log2(real_numeric + 1)
        gen_numeric = np.log2(gen_numeric + 1)

    # --- Sub-sample for speed -----------------------------------------------
    if n_samples is not None and n_samples < len(real_numeric):
        n = min(n_samples, len(real_numeric) // 2)
        real_idx = list(range(n)) + list(
            range(len(real_numeric) - n, len(real_numeric))
        )
    else:
        real_idx = list(range(len(real_numeric)))

    if n_samples is not None and n_samples < len(gen_numeric):
        n = min(n_samples, len(gen_numeric) // 2)
        gen_idx = list(range(n)) + list(range(len(gen_numeric) - n, len(gen_numeric)))
    else:
        gen_idx = list(range(len(gen_numeric)))

    real_sub = real_numeric.iloc[real_idx]
    gen_sub = gen_numeric.iloc[gen_idx]

    groups_real_sub = groups_real.iloc[real_idx] if groups_real is not None else None
    groups_gen_sub = (
        groups_generated.iloc[gen_idx] if groups_generated is not None else None
    )

    # --- Produce figures ----------------------------------------------------
    fig_heatmap = heatmap_eval(real_data=real_sub, generated_data=gen_sub)
    fig_umap = UMAP_eval(
        real_data=real_sub,
        generated_data=gen_sub,
        groups_real=groups_real_sub,
        groups_generated=groups_gen_sub,
        random_seed=random_seed,
    )

    return {"heatmap": fig_heatmap, "umap": fig_umap}
