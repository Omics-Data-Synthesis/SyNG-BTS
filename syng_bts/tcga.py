"""TCGA miRNA dataset loader.

Downloads, caches, and exposes 24 packaged TCGA miRNA HDF5 datasets through a
small, ergonomic Python API.

Quick start::

    from syng_bts import load_tcga_dataset, list_tcga_datasets
    list_tcga_datasets()
    ds = load_tcga_dataset("BRCA")
    real_df, real_groups = ds.real("TC")

Files are downloaded once on first access and cached under
``~/.cache/syng-bts/tcga/`` (override with the ``SYNG_BTS_CACHE_DIR`` environment
variable).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Phase 3 (Task 16) replaces this with the live URL.
_DEFAULT_MANIFEST_URL = (
    "TBD-replace-after-publish: "
    "https://github.com/Omics-Data-Synthesis/SyNG-BTS/"
    "releases/download/data-v1.0/manifest.json"
)

_CACHE_ENV_VAR = "SYNG_BTS_CACHE_DIR"
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "syng-bts"
_NETWORK_TIMEOUT_SECS = 60
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB

VALID_NORMALIZATIONS = ("raw_norm", "TC", "DESeq")
VALID_MODELS = ("CVAE1_5", "CVAE1_10", "CVAE1_20")
DEFAULT_NORMALIZATION = "TC"
DEFAULT_MODEL = "CVAE1_5"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class _NetworkError(OSError):
    """Raised when a network request fails. Carries an offline-staging hint."""


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def tcga_cache_dir() -> Path:
    """Return the active TCGA cache directory (without the version subdir).

    Honors the ``SYNG_BTS_CACHE_DIR`` environment variable if set; otherwise
    returns ``~/.cache/syng-bts/tcga``. The directory is **not** created by this
    call.
    """
    root_str = os.environ.get(_CACHE_ENV_VAR)
    root = Path(root_str) if root_str else _DEFAULT_CACHE_ROOT
    return root / "tcga"


# ---------------------------------------------------------------------------
# Manifest fetching
# ---------------------------------------------------------------------------


def _url_index_path() -> Path:
    return tcga_cache_dir() / ".url_index.json"


def _read_url_index() -> dict[str, str]:
    p = _url_index_path()
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_url_index(index: dict[str, str]) -> None:
    p = _url_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(index, f, indent=2)


def _download_bytes(url: str) -> bytes:
    """Download a small URL fully into memory. Used for the manifest only."""
    try:
        with urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT_SECS) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise _NetworkError(
            f"Failed to download {url}: {e.reason if hasattr(e, 'reason') else e}.\n"
            f"Check your network connection, or pre-stage the file under "
            f"{tcga_cache_dir()} and set {_CACHE_ENV_VAR} if needed."
        ) from e


def _parse_manifest(payload: bytes, *, source: str) -> dict:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        snippet = payload[:200].decode("utf-8", errors="replace")
        raise ValueError(
            f"Could not parse manifest from {source}: {e}. "
            f"First 200 bytes of body: {snippet!r}"
        ) from e
    if not isinstance(data, dict) or "version" not in data or "datasets" not in data:
        raise ValueError(
            f"Manifest from {source} is missing required fields "
            f"('version', 'datasets')."
        )
    return data


def _fetch_manifest(manifest_url: str | None) -> dict:
    """Return the parsed manifest dict.

    For ``manifest_url is None`` (default-URL flow): consult the URL-version
    index, return the cached manifest if present, otherwise download once and
    populate the cache. For an explicit ``manifest_url``: always download
    fresh, never cache.
    """
    if manifest_url is not None:
        return _parse_manifest(_download_bytes(manifest_url), source=manifest_url)

    url = _DEFAULT_MANIFEST_URL
    index = _read_url_index()
    cached_version = index.get(url)
    if cached_version is not None:
        cached_path = tcga_cache_dir() / cached_version / "manifest.json"
        if cached_path.exists():
            try:
                with open(cached_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # fall through to redownload

    manifest = _parse_manifest(_download_bytes(url), source=url)
    version = str(manifest["version"])
    version_dir = tcga_cache_dir() / version
    version_dir.mkdir(parents=True, exist_ok=True)
    with open(version_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    index[url] = version
    _write_url_index(index)

    return manifest


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _resolve_name(name: str, manifest: dict) -> str:
    """Resolve a user-supplied name to a full dataset name.

    Resolution order:
      1. Exact full-name match.
      2. Cancer-type prefix matching exactly one dataset.
      3. Multiple matches → ValueError listing matches.
      4. No match → ValueError listing all available datasets.
    """
    full_names = [d["dataset_name"] for d in manifest["datasets"]]

    if name in full_names:
        return name

    matches = [n for n in full_names if n.split("_", 1)[0] == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous name '{name}'. Matches: {matches}. "
            f"Pass the full name to disambiguate."
        )

    raise ValueError(
        f"Unknown TCGA dataset '{name}'. Available: {sorted(full_names)}"
    )


# ---------------------------------------------------------------------------
# Public function: list_tcga_datasets
# ---------------------------------------------------------------------------


def list_tcga_datasets(
    *,
    short: bool = False,
    manifest_url: str | None = None,
) -> list[str]:
    """List the names of all TCGA datasets available in the manifest.

    Parameters
    ----------
    short : bool, default False
        If False, returns full project names. If True, returns cancer-type
        aliases (e.g. "BRCA").
    manifest_url : str or None
        Override the default manifest URL. Useful for tests, forks, or
        pre-staged air-gapped environments.

    Returns
    -------
    list[str]
        Sorted list of dataset names (deduplicated when ``short=True``).
    """
    manifest = _fetch_manifest(manifest_url)
    full_names = [d["dataset_name"] for d in manifest["datasets"]]
    if short:
        aliases = sorted({n.split("_", 1)[0] for n in full_names})
        return aliases
    return sorted(full_names)


# ---------------------------------------------------------------------------
# HDF5 download + sha256 verification
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` via ``dest.tmp`` then atomic rename.

    Streams in 1 MiB chunks so memory use stays flat regardless of file size.
    Uses tqdm if available; otherwise prints a single info line to stderr.
    Wraps URL errors as ``_NetworkError``. Cleans up the ``.tmp`` on any
    exception.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT_SECS) as resp:
            total = int(resp.headers.get("Content-Length", "0") or 0)
            pbar = None
            if _HAS_TQDM and total > 0:
                pbar = _tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                )
            else:
                size_str = (
                    f"{total / 1e6:.1f} MB" if total > 0 else "size unknown"
                )
                print(
                    f"Downloading {dest.name} ({size_str})…",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        f.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
            finally:
                if pbar is not None:
                    pbar.close()

        tmp.replace(dest)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise _NetworkError(
            f"Failed to download {url}: "
            f"{e.reason if hasattr(e, 'reason') else e}.\n"
            f"Check your network connection, or pre-stage the file at "
            f"{dest} and set {_CACHE_ENV_VAR} if needed."
        ) from e
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _fetch_and_verify_h5(
    url: str, dest: Path, expected_sha256: str
) -> None:
    """Download to ``dest``, verify sha256, retry once on mismatch.

    On second mismatch, removes ``dest`` and raises ``ValueError``.
    """
    last_actual = None
    for _ in range(2):
        _stream_download(url, dest)
        actual = _sha256_of_file(dest)
        if actual == expected_sha256:
            return
        last_actual = actual
        dest.unlink(missing_ok=True)

    raise ValueError(
        f"Checksum mismatch for {dest.name} after retry. "
        f"Expected sha256={expected_sha256}, got {last_actual}. "
        f"The cache and the published release may be out of sync — "
        f"please file an issue at "
        f"https://github.com/Omics-Data-Synthesis/SyNG-BTS/issues."
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subset:
    """One slice of a TCGA dataset (raw, processed/X, or synthetic/X/Y).

    Attributes
    ----------
    expression : pd.DataFrame
        Expression matrix. Index is sample IDs (raw) or RangeIndex
        (processed, synthetic). Columns are feature names.
    groups : pd.Series
        Group labels indexed identically to ``expression``.
    metadata : dict[str, Any]
        Per-slice attributes (normalization_method, transform, kl_weight,
        epochs_trained, etc.).
    """

    expression: pd.DataFrame
    groups: pd.Series
    metadata: dict[str, Any]


class TCGADataset:
    """Eagerly-loaded view of one TCGA dataset bundle.

    Built once inside ``load_tcga_dataset`` from the corresponding HDF5 file.
    All 13 ``Subset`` views (raw + 3 processed + 9 synthetic) are constructed
    before the file handle closes.
    """

    def __init__(
        self,
        *,
        name: str,
        cancer_type: str,
        clinical_variable: str,
        group_labels: list[str],
        n_raw_samples: int,
        n_filtered_samples: int,
        n_raw_features: int,
        n_filtered_features: int,
        schema_version: str,
        creation_date: str,
        syng_bts_version: str,
        raw: Subset,
        processed: dict[str, Subset],
        synthetic: dict[str, dict[str, Subset]],
    ) -> None:
        self.name = name
        self.cancer_type = cancer_type
        self.clinical_variable = clinical_variable
        self.group_labels = group_labels
        self.n_raw_samples = n_raw_samples
        self.n_filtered_samples = n_filtered_samples
        self.n_raw_features = n_raw_features
        self.n_filtered_features = n_filtered_features
        self.schema_version = schema_version
        self.creation_date = creation_date
        self.syng_bts_version = syng_bts_version
        self.raw = raw
        self.processed = processed
        self.synthetic = synthetic


# ---------------------------------------------------------------------------
# HDF5 → DataFrame construction
# ---------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    """Decode bytes / numpy scalars / numpy arrays into pure-Python values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [
            v.decode() if isinstance(v, bytes) else v.item()
            if hasattr(v, "item")
            else v
            for v in value
        ]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_strings(dataset: h5py.Dataset) -> list[str]:
    raw = dataset[:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _read_attrs(group: h5py.Group) -> dict[str, Any]:
    return {k: _decode(v) for k, v in group.attrs.items()}


def _build_subset_from_group(
    group: h5py.Group,
    *,
    feature_names: list[str] | None = None,
    sample_ids: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Subset:
    """Construct a Subset from an HDF5 group containing expression + groups.

    ``feature_names`` is read from the group if not supplied (raw/processed
    case); pass it explicitly for synthetic groups, where feature_names lives
    one level up.
    """
    expr_arr = group["expression"][:]
    groups = _read_strings(group["groups"])

    if feature_names is None:
        feature_names = _read_strings(group["feature_names"])

    if sample_ids is not None:
        index = pd.Index(sample_ids, name="sample_id")
    else:
        index = pd.RangeIndex(len(groups))

    expression = pd.DataFrame(expr_arr, columns=feature_names, index=index)
    groups_series = pd.Series(groups, index=index, name="groups")

    metadata = _read_attrs(group)
    if extra_metadata:
        # Group-specific attrs win over inherited shared attrs.
        merged = {**extra_metadata, **metadata}
        metadata = merged

    return Subset(
        expression=expression, groups=groups_series, metadata=metadata
    )


def _build_dataset_from_h5(path: Path) -> TCGADataset:
    """Read an entire v1.0 TCGA HDF5 file and return a TCGADataset."""
    with h5py.File(path, "r") as f:
        attrs = {k: _decode(v) for k, v in f.attrs.items()}

        # Raw subset (with sample_ids)
        raw_grp = f["raw"]
        raw_sample_ids = (
            _read_strings(raw_grp["sample_ids"])
            if "sample_ids" in raw_grp
            else None
        )
        raw = _build_subset_from_group(raw_grp, sample_ids=raw_sample_ids)

        # Processed subsets (RangeIndex)
        processed: dict[str, Subset] = {}
        for norm in VALID_NORMALIZATIONS:
            processed[norm] = _build_subset_from_group(f[f"processed/{norm}"])

        # Synthetic subsets — feature_names lives at /synthetic/{norm}/
        synth_root = f["synthetic"]
        shared_attrs = _read_attrs(synth_root)
        synthetic: dict[str, dict[str, Subset]] = {}
        for norm in VALID_NORMALIZATIONS:
            norm_grp = synth_root[norm]
            features = _read_strings(norm_grp["feature_names"])
            synthetic[norm] = {}
            for model in VALID_MODELS:
                synthetic[norm][model] = _build_subset_from_group(
                    norm_grp[model],
                    feature_names=features,
                    extra_metadata=shared_attrs,
                )

    return TCGADataset(
        name=str(attrs["dataset_name"]),
        cancer_type=str(attrs["cancer_type"]),
        clinical_variable=str(attrs["clinical_variable"]),
        group_labels=[str(x) for x in attrs.get("group_labels", [])],
        n_raw_samples=int(attrs["n_raw_samples"]),
        n_filtered_samples=int(attrs["n_filtered_samples"]),
        n_raw_features=int(attrs["n_raw_features"]),
        n_filtered_features=int(attrs["n_filtered_features"]),
        schema_version=str(attrs["version"]),
        creation_date=str(attrs["creation_date"]),
        syng_bts_version=str(attrs["syng_bts_version"]),
        raw=raw,
        processed=processed,
        synthetic=synthetic,
    )


# ---------------------------------------------------------------------------
# Public function: load_tcga_dataset
# ---------------------------------------------------------------------------


def _dataset_url_from_manifest(manifest_url: str, file: str) -> str:
    """Resolve a dataset URL relative to the manifest URL."""
    return manifest_url.rsplit("/", 1)[0] + "/" + file


def _entry_for(manifest: dict, full_name: str) -> dict:
    for entry in manifest["datasets"]:
        if entry["dataset_name"] == full_name:
            return entry
    raise KeyError(full_name)  # pragma: no cover  (caller has already resolved)


def load_tcga_dataset(
    name: str,
    *,
    force: bool = False,
    manifest_url: str | None = None,
) -> TCGADataset:
    """Download (if needed), cache, and load a TCGA dataset bundle.

    Parameters
    ----------
    name : str
        Either a full project name (e.g.
        ``"BRCA_breast_carcinoma_estrogen_receptor_status"``) or a unique
        cancer-type prefix (e.g. ``"BRCA"``).
    force : bool, default False
        If True, redownload the HDF5 file even if cached.
    manifest_url : str or None, default None
        Override the default manifest URL.

    Returns
    -------
    TCGADataset
        Eagerly-loaded container with all 13 splits available as DataFrames.

    Raises
    ------
    ValueError
        If ``name`` does not resolve, or the downloaded file fails sha256
        verification twice.
    OSError
        On network failure (wrapped with an offline-staging hint).
    """
    manifest = _fetch_manifest(manifest_url)
    full_name = _resolve_name(name, manifest)
    entry = _entry_for(manifest, full_name)

    version = str(manifest["version"])
    version_dir = tcga_cache_dir() / version
    version_dir.mkdir(parents=True, exist_ok=True)

    cached_h5 = version_dir / entry["file"]

    if force or not cached_h5.exists():
        url = _dataset_url_from_manifest(
            manifest_url if manifest_url is not None else _DEFAULT_MANIFEST_URL,
            entry["file"],
        )
        _fetch_and_verify_h5(url, cached_h5, entry["sha256"])

    try:
        return _build_dataset_from_h5(cached_h5)
    except (OSError, KeyError) as e:
        # h5py raises OSError for malformed files and KeyError for missing
        # groups/datasets — both indicate the cached file is corrupt.
        raise ValueError(
            f"Corrupt HDF5 at {cached_h5}; pass force=True to redownload."
        ) from e
