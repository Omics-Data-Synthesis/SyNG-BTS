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

import hashlib  # noqa: F401  # used in later tasks
import json
import os
import sys  # noqa: F401  # used in later tasks
import urllib.error
import urllib.request
from dataclasses import dataclass  # noqa: F401  # used in later tasks
from pathlib import Path
from typing import Any  # noqa: F401  # used in later tasks

import h5py  # noqa: F401  # used in later tasks
import numpy as np  # noqa: F401  # used in later tasks
import pandas as pd  # noqa: F401  # used in later tasks

try:
    from tqdm import tqdm as _tqdm  # noqa: F401  # used in later tasks

    _HAS_TQDM = True
except ImportError:
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
