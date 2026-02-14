"""
Data loading and path utilities for SyNG-BTS.

This module provides functions for loading bundled data files and managing
output directories in a cross-platform compatible way.
"""

from pathlib import Path

import pandas as pd

# Try importlib.resources (Python 3.9+) with fallback to importlib_resources
try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

# Default output directory
_DEFAULT_OUTPUT_DIR: Path | None = None


def set_default_output_dir(path: str | Path | None) -> None:
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


def get_output_dir(output_dir: str | Path | None = None) -> Path:
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


def ensure_dir(path: str | Path) -> Path:
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
    output_dir: str | Path | None,
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
    Load a bundled Parquet data file from the package's data directory.

    Parameters
    ----------
    subdir : str
        The subdirectory within syng_bts/data/ (e.g., "examples", "transfer").
    filename : str
        The Parquet filename to load.

    Returns
    -------
    pd.DataFrame
        The loaded feature data (index preserved from Parquet).
    """
    try:
        data_package = files("syng_bts.data")
        resource = data_package
        for part in subdir.split("/"):
            resource = resource.joinpath(part)
        resource = resource.joinpath(filename)

        with as_file(resource) as path:
            return pd.read_parquet(path, engine="pyarrow")
    except (TypeError, AttributeError, FileNotFoundError) as e:
        import syng_bts

        package_dir = Path(syng_bts.__file__).parent
        file_path = package_dir / "data" / subdir / filename
        if file_path.exists():
            return pd.read_parquet(file_path, engine="pyarrow")
        raise FileNotFoundError(
            f"Could not find bundled data file: {subdir}/{filename}"
        ) from e


def load_data(
    dataname: str,
    data_path: str | Path | None = None,
    bundled_info: tuple | None = None,
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


# Map of known bundled datasets to their package locations and subdirectories.
# Format: "dataset_name": ("subdir_path", "features.parquet", "groups.parquet" | None)
BUNDLED_DATASETS: dict[str, tuple[str, str, str | None]] = {
    # Example datasets
    "SKCMPositive_4": ("examples", "SKCMPositive_4.parquet", None),
    # Transfer learning datasets
    "BRCA": ("transfer", "BRCA.parquet", None),
    "PRAD": ("transfer", "PRAD.parquet", None),
    # BRCA subtype case study
    "BRCASubtypeSel": (
        "case/brca_subtype",
        "BRCASubtypeSel.parquet",
        "BRCASubtypeSel_groups.parquet",
    ),
    "BRCASubtypeSel_test": (
        "case/brca_subtype",
        "BRCASubtypeSel_test.parquet",
        "BRCASubtypeSel_test_groups.parquet",
    ),
    "BRCASubtypeSel_train": (
        "case/brca_subtype",
        "BRCASubtypeSel_train.parquet",
        "BRCASubtypeSel_train_groups.parquet",
    ),
    # LIHC subtype case study
    "LIHCSubtypeFamInd": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd.parquet",
        "LIHCSubtypeFamInd_groups.parquet",
    ),
    "LIHCSubtypeFamInd_DESeq": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_DESeq.parquet",
        "LIHCSubtypeFamInd_DESeq_groups.parquet",
    ),
    "LIHCSubtypeFamInd_test74": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_test74.parquet",
        "LIHCSubtypeFamInd_test74_groups.parquet",
    ),
    "LIHCSubtypeFamInd_test74_DESeq": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_test74_DESeq.parquet",
        "LIHCSubtypeFamInd_test74_DESeq_groups.parquet",
    ),
    "LIHCSubtypeFamInd_train294": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_train294.parquet",
        "LIHCSubtypeFamInd_train294_groups.parquet",
    ),
    "LIHCSubtypeFamInd_train294_DESeq": (
        "case/lihc_subtype",
        "LIHCSubtypeFamInd_train294_DESeq.parquet",
        "LIHCSubtypeFamInd_train294_DESeq_groups.parquet",
    ),
}


def load_dataset(
    dataname: str,
    data_path: str | Path | None = None,
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
    if bundled_info is not None:
        # Extract (subdir, feature_filename) for load_data compatibility
        bundled_info = (bundled_info[0], bundled_info[1])
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


def resolve_data(data: "pd.DataFrame | str | Path") -> pd.DataFrame:
    """
    Resolve a flexible data input to a pandas DataFrame.

    Accepts a DataFrame (returned as-is), a file path (loaded via
    ``pd.read_csv``), or the name of a bundled dataset.

    Parameters
    ----------
    data : pd.DataFrame, str, or Path
        One of:

        - A ``pd.DataFrame`` — returned directly.
        - A ``str`` or ``Path`` pointing to an existing CSV file (must
          include an extension such as ``.csv``).
        - A plain name (no extension, no path separators) of a bundled
          dataset, e.g. ``"SKCMPositive_4"``.

    Returns
    -------
    pd.DataFrame
        The resolved data.

    Raises
    ------
    ValueError
        If *data* looks like a bundled-dataset name but is not found in the
        registry.  The error message lists all available bundled datasets.
    FileNotFoundError
        If *data* looks like a file path but the file does not exist.
    TypeError
        If *data* is not a DataFrame, str, or Path.

    Examples
    --------
    >>> from syng_bts.data_utils import resolve_data
    >>> df = resolve_data("SKCMPositive_4")          # bundled dataset
    >>> df = resolve_data("./my_data/custom.csv")    # file path
    >>> df = resolve_data(existing_dataframe)         # pass-through
    """
    # 1. DataFrame pass-through
    if isinstance(data, pd.DataFrame):
        return data

    # 2. Convert to string for inspection
    if isinstance(data, Path):
        data_str = str(data)
    elif isinstance(data, str):
        data_str = data
    else:
        raise TypeError(
            f"'data' must be a pd.DataFrame, str, or Path, got {type(data).__name__}"
        )

    path = Path(data_str)

    # 3. If it looks like a real file path (has path separators), try to load it
    has_separators = "/" in data_str or "\\" in data_str
    if has_separators:
        if path.exists():
            return pd.read_csv(path, header=0)
        raise FileNotFoundError(f"Data file not found: {path}")

    # 4. Treat as a bundled dataset name — strip .csv/.parquet if the user added it
    name = data_str
    if name.lower().endswith(".csv"):
        name = name[: -len(".csv")]
    elif name.lower().endswith(".parquet"):
        name = name[: -len(".parquet")]

    bundled_info = BUNDLED_DATASETS.get(name)
    if bundled_info is not None:
        subdir, filename, _groups = bundled_info
        return load_bundled_data(subdir, filename)

    # 5. Last resort: try as a local file (e.g. "myfile.csv" in cwd)
    if path.suffix and path.exists():
        return pd.read_csv(path, header=0)

    available = ", ".join(sorted(BUNDLED_DATASETS.keys()))
    raise ValueError(
        f"Unknown dataset name '{name}'. Available bundled datasets: {available}"
    )


def derive_dataname(
    data: "pd.DataFrame | str | Path",
    name: "str | None" = None,
) -> str:
    """
    Derive a short human-readable name for a dataset.

    The name is used in output filenames and metadata.  An explicit *name*
    always takes priority.

    Parameters
    ----------
    data : pd.DataFrame, str, or Path
        The original ``data`` argument the user passed.
    name : str or None
        Explicit override.  When provided, returned as-is.

    Returns
    -------
    str
        A short identifier for the dataset.
    """
    if name is not None:
        return name

    if isinstance(data, (str, Path)):
        p = Path(data)
        # Strip .csv extension if present for bundled-name lookup
        stem = p.stem if p.suffix else str(p)
        # If user passed a bare name like "SKCMPositive_4", stem == name
        return stem

    # DataFrame: try df.attrs["name"], fall back to "data"
    if isinstance(data, pd.DataFrame):
        return data.attrs.get("name", "data")

    return "data"
