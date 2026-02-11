from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap  # noqa: F401
from matplotlib.figure import Figure
from umap import UMAP

from .data_utils import load_dataset


def heatmap_eval(
    dat_real: pd.DataFrame,
    dat_generated: Optional[pd.DataFrame] = None,
    save: bool = False,
) -> Optional[Figure]:
    r"""
    Create a heatmap visualization comparing generated data and real data.

    If only one dataset is provided, displays a single heatmap. If both
    real and generated data are provided, displays them side by side.

    Parameters
    ----------
    dat_real : pd.DataFrame
        The original/real data.
    dat_generated : pd.DataFrame, optional
        The generated data. If None, only dat_real is plotted.
    save : bool, default=False
        If True, return the figure instead of displaying it.

    Returns
    -------
    Figure or None
        If save=True, returns the matplotlib Figure. Otherwise, displays
        the figure and returns None.
    """
    if dat_generated is None:
        # Only plot dat_real if dat_generated is None
        fig = plt.figure(figsize=(6, 6))
        ax = sns.heatmap(dat_real, cbar=True)
        ax.set_title("Real Data")
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")
    else:
        # Plot both dat_generated and dat_real side by side
        fig, axs = plt.subplots(
            ncols=2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[0.5, 0.55])
        )

        sns.heatmap(dat_generated, ax=axs[0], cbar=False)
        axs[0].set_title("Generated Data")
        axs[0].set_xlabel("Features")
        axs[0].set_ylabel("Samples")

        sns.heatmap(dat_real, ax=axs[1], cbar=True)
        axs[1].set_title("Real Data")
        axs[1].set_xlabel("Features")
        axs[1].set_ylabel("Samples")

    plt.tight_layout()

    if save:
        return fig
    else:
        plt.show()


def UMAP_eval(
    dat_generated: Optional[pd.DataFrame],
    dat_real: pd.DataFrame,
    groups_generated: Optional[pd.Series] = None,
    groups_real: Optional[pd.Series] = None,
    random_state: int = 42,
    legend_pos: str = "best",
) -> None:
    r"""
    Create a UMAP visualization comparing generated data and real data.

    Uses UMAP dimensionality reduction to visualize high-dimensional
    data in 2D, with optional group coloring.

    Parameters
    ----------
    dat_generated : pd.DataFrame or None
        The generated data. If None, only dat_real is visualized.
    dat_real : pd.DataFrame
        The original/real data.
    groups_generated : pd.Series or None, optional
        Group labels for generated samples. Used for coloring/styling.
    groups_real : pd.Series or None, optional
        Group labels for real samples. Used for coloring/styling.
    random_state : int, default=42
        Random seed for UMAP reproducibility.
    legend_pos : str, default="best"
        Legend position ("best", "upper right", "lower left", etc.).

    Returns
    -------
    None
        Displays the UMAP plot.
    """

    if dat_generated is None:
        # Only plot the real data
        reducer = UMAP(random_state=random_state)
        embedding = reducer.fit_transform(dat_real.values)

        umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])

        plt.figure(figsize=(10, 8))
        if groups_real is not None:
            umap_df["Group"] = groups_real.astype(
                str
            )  # Ensure groups are hashable for seaborn
            sns.scatterplot(
                data=umap_df, x="UMAP1", y="UMAP2", style="Group", palette="bright"
            )
            plt.legend(title="Group", loc=legend_pos)
            plt.title("UMAP Projection of Real Data with Groups")
        else:
            plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], alpha=0.7)
            plt.title("UMAP Projection of Real Data")

        plt.show()
        return

    # If dat_generated is provided, we process both real and generated data
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))
    combined_labels = np.array(
        ["Real"] * dat_real.shape[0] + ["Generated"] * dat_generated.shape[0]
    )

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=random_state)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df["Data Type"] = combined_labels

    plt.figure(figsize=(10, 8))

    if groups_real is not None and groups_generated is not None:
        # If group information is available, use it for coloring
        combined_groups = np.concatenate((groups_real, groups_generated))
        combined_groups = [
            str(group) for group in combined_groups
        ]  # Convert groups to string if not already
        umap_df["Group"] = combined_groups
        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="Data Type",
            style="Group",
            palette="bright",
        )
        plt.legend(title="Data Type/Group", loc="best")
        plt.title("UMAP Projection of Real and Generated Data with Groups")

    else:
        # If no group information, just plot real vs. generated data
        sns.scatterplot(
            data=umap_df, x="UMAP1", y="UMAP2", hue="Data Type", palette="bright"
        )
        plt.legend(title="Data Type", loc="best")
        plt.title("UMAP Projection of Real and Generated Data")

    plt.show()


def evaluation(
    generated_input: str = "BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv",
    real_input: str = "BRCASubtypeSel_test.csv",
    data_dir: Optional[Union[str, Path]] = None,
) -> None:
    r"""
    Preprocessing and visualization of generated vs real data.

    This method preprocesses the input data and creates visualizations
    comparing generated and real datasets using heatmaps and UMAP plots.

    Parameters
    ----------
    generated_input : str
        Filename of the generated dataset. A default example is provided.
    real_input : str
        Filename of the real original dataset. A default example is provided.
    data_dir : str, Path, or None
        Directory containing the data files. If None, tries bundled package data.

    Returns
    -------
    None
        Displays visualization plots.
    """
    # Load generated data
    generated_name = generated_input.replace(".csv", "")
    if data_dir is not None:
        generated_path = Path(data_dir) / generated_input
        if generated_path.exists():
            generated = pd.read_csv(generated_path, header=0)
        else:
            generated = load_dataset(generated_name, data_path=generated_path)
    else:
        try:
            generated = load_dataset(generated_name)
        except FileNotFoundError:
            # Legacy path fallback
            legacy_path = Path("../Case/BRCASubtype") / generated_input
            if legacy_path.exists():
                generated = pd.read_csv(legacy_path, header=0)
            else:
                raise FileNotFoundError(
                    f"Could not find generated data '{generated_input}'. "
                    f"Specify data_dir parameter."
                )

    # Load real data
    real_name = real_input.replace(".csv", "")
    if data_dir is not None:
        real_path = Path(data_dir) / real_input
        if real_path.exists():
            real = pd.read_csv(real_path, header=0)
        else:
            real = load_dataset(real_name, data_path=real_path)
    else:
        try:
            real = load_dataset(real_name)
        except FileNotFoundError:
            # Legacy path fallback
            legacy_path = Path("../Case/BRCASubtype") / real_input
            if legacy_path.exists():
                real = pd.read_csv(legacy_path, header=0)
            else:
                raise FileNotFoundError(
                    f"Could not find real data '{real_input}'. "
                    f"Specify data_dir parameter."
                )

    # Define the default group level
    level0 = real["groups"].iloc[0]
    level1 = list(set(real["groups"]) - set([level0]))

    # Get sample groups
    groups_real = pd.Series(
        np.where(real["groups"] == "Infiltrating Ductal Carcinoma", "Ductal", "Lobular")
    )

    groups_generated = pd.Series(
        np.where(generated.iloc[:, -1] == 1, "Ductal", "Lobular")
    )

    # Get pure data matrices
    real = real.select_dtypes(include=[np.number])
    real = np.log2(real + 1)
    generated = generated.iloc[:, : real.shape[1]]
    generated.columns = real.columns

    # Select samples for analysis to save running time
    real_ind = list(range(200)) + list(range(len(real) - 200, len(real)))
    generated_ind = list(range(200)) + list(range(len(generated) - 200, len(generated)))

    # Call evaluation functions
    h_subtypes = heatmap_eval(
        dat_real=real.iloc[real_ind,], dat_generated=generated.iloc[generated_ind,]
    )
    p_umap_subtypes = UMAP_eval(
        dat_real=real.iloc[real_ind,],
        dat_generated=generated.iloc[generated_ind,],
        groups_real=groups_real.iloc[real_ind],
        groups_generated=groups_generated.iloc[generated_ind],
        legend_pos="bottom",
    )


# evaluation()
