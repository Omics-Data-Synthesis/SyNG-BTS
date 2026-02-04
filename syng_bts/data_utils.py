"""
Data loading and path utilities for SyNG-BTS.

This module provides functions for loading bundled data files and managing
output directories in a cross-platform compatible way.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Try importlib.resources (Python 3.9+) with fallback to importlib_resources
try:
    from importlib.resources import files, as_file
except ImportError:
    from importlib_resources import files, as_file

# For older Python compatibility
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


# Default output directory
_DEFAULT_OUTPUT_DIR: Optional[Path] = None


def set_default_output_dir(path: Union[str, Path, None]) -> None:
    """
    Set the default output directory for all SyNG-BTS operations.

    Parameters
    ----------
    path : str, Path, or None
        The default output directory. If None, uses current working directory.
    """
    global _DEFAULT_OUTPUT_DIR
    if path is None:
        _DEFAULT_OUTPUT_DIR = None
    else:
        _DEFAULT_OUTPUT_DIR = Path(path)


def get_output_dir(output_dir: Union[str, Path, None] = None) -> Path:
    """
    Get the output directory to use for saving files.

    Parameters
    ----------
    output_dir : str, Path, or None
        Explicit output directory. If None, uses the default or current working directory.

    Returns
    -------
    Path
        The resolved output directory path.
    """
    if output_dir is not None:
        return Path(output_dir)
    if _DEFAULT_OUTPUT_DIR is not None:
        return _DEFAULT_OUTPUT_DIR
    return Path.cwd()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        The directory path to ensure exists.

    Returns
    -------
    Path
        The path as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(
    output_dir: Union[str, Path, None],
    subdir: str,
    filename: str,
    create_dir: bool = True,
) -> Path:
    """
    Get a full output path, optionally creating the directory.

    Parameters
    ----------
    output_dir : str, Path, or None
        Base output directory. If None, uses default.
    subdir : str
        Subdirectory name (e.g., "GeneratedData", "Loss").
    filename : str
        The filename.
    create_dir : bool
        Whether to create the directory if it doesn't exist.

    Returns
    -------
    Path
        The full output path.
    """
    base_dir = get_output_dir(output_dir)
    full_dir = base_dir / subdir
    if create_dir:
        ensure_dir(full_dir)
    return full_dir / filename


def load_bundled_data(subdir: str, filename: str) -> pd.DataFrame:
    """
    Load a bundled CSV data file from the package's data directory.

    Parameters
    ----------
    subdir : str
        The subdirectory within syng_bts/data/ (e.g., "examples", "transfer").
    filename : str
        The filename to load.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    try:
        # Construct the path within the data package
        data_package = files("syng_bts.data")
        # Navigate through subdirectories
        resource = data_package
        for part in subdir.split("/"):
            resource = resource.joinpath(part)
        resource = resource.joinpath(filename)

        with as_file(resource) as path:
            return pd.read_csv(path, header=0)
    except (TypeError, AttributeError, FileNotFoundError) as e:
        # Fallback: try direct file path (for development/editable installs)
        import syng_bts

        package_dir = Path(syng_bts.__file__).parent
        file_path = package_dir / "data" / subdir / filename
        if file_path.exists():
            return pd.read_csv(file_path, header=0)
        raise FileNotFoundError(
            f"Could not find bundled data file: {subdir}/{filename}"
        ) from e


def load_data(
    dataname: str,
    data_path: Union[str, Path, None] = None,
    bundled_info: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Load data from a file path or bundled package resource.

    First tries to load from the specified path. If the file doesn't exist
    and bundled_info is specified, tries to load from package resources.

    Parameters
    ----------
    dataname : str
        The data name (without .csv extension).
    data_path : str, Path, or None
        Path to the data file or directory containing the file.
        If a directory, will look for {dataname}.csv in it.
    bundled_info : tuple, optional
        Tuple of (subdir, filename) to load bundled data from if file not found.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    FileNotFoundError
        If the data cannot be found in either location.
    """
    filename = f"{dataname}.csv"

    # Determine the full file path
    if data_path is not None:
        path = Path(data_path)
        if path.is_dir():
            file_path = path / filename
        else:
            file_path = path
    else:
        file_path = Path(filename)

    # Try loading from file path first
    if file_path.exists():
        return pd.read_csv(file_path, header=0)

    # Try loading from bundled data
    if bundled_info is not None:
        try:
            subdir, bundled_filename = bundled_info
            return load_bundled_data(subdir, bundled_filename)
        except (FileNotFoundError, ModuleNotFoundError):
            pass

    raise FileNotFoundError(
        f"Could not find data '{dataname}'. "
        f"Looked in: {file_path}"
        + (f" and bundled data '{bundled_info}'" if bundled_info else "")
    )


# Map of known bundled datasets to their package locations and subdirectories
# Format: "dataset_name": ("subdir_path", "filename.csv")
BUNDLED_DATASETS = {
    # Example datasets
    "SKCMPositive_4": ("examples", "SKCMPositive_4.csv"),
    # Transfer learning datasets
    "BRCA": ("transfer", "BRCA.csv"),
    "PRAD": ("transfer", "PRAD.csv"),
    # BRCA subtype case study
    "BRCASubtypeSel": ("case/brca_subtype", "BRCASubtypeSel.csv"),
    "BRCASubtypeSel_test": ("case/brca_subtype", "BRCASubtypeSel_test.csv"),
    "BRCASubtypeSel_train": ("case/brca_subtype", "BRCASubtypeSel_train.csv"),
    # LIHC subtype case study
    "LIHCSubtypeFamInd": ("case/lihc_subtype", "LIHCSubtypeFamInd.csv"),
    "LIHCSubtypeFamInd_DESeq": ("case/lihc_subtype", "LIHCSubtypeFamInd_DESeq.csv"),
    "LIHCSubtypeFamInd_test74": ("case/lihc_subtype", "LIHCSubtypeFamInd_test74.csv"),
    "LIHCSubtypeFamInd_test74_DESeq": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_test74_DESeq.csv",
    ),
    "LIHCSubtypeFamInd_train294": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_train294.csv",
    ),
    "LIHCSubtypeFamInd_train294_DESeq": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_train294_DESeq.csv",
    ),
}


def load_dataset(
    dataname: str,
    data_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Load a dataset, checking bundled data if not found at path.

    This is a convenience function that automatically checks if the dataset
    is a known bundled dataset.

    Parameters
    ----------
    dataname : str
        The dataset name (without .csv extension).
    data_path : str, Path, or None
        Optional path to look for the data first.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Examples
    --------
    >>> from syng_bts import load_dataset
    >>> # Load bundled example data
    >>> data = load_dataset("SKCMPositive_4")
    >>> # Load from custom path
    >>> data = load_dataset("my_data", data_path="./my_data_dir/")
    """
    bundled_info = BUNDLED_DATASETS.get(dataname)
    return load_data(dataname, data_path, bundled_info)


def list_bundled_datasets() -> list:
    """
    List all available bundled datasets.

    Returns
    -------
    list
        List of dataset names that can be loaded with load_dataset().
    """
    return list(BUNDLED_DATASETS.keys())
