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
import json  # noqa: F401  # used in later tasks
import os
import sys  # noqa: F401  # used in later tasks
import urllib.error  # noqa: F401  # used in later tasks
import urllib.request  # noqa: F401  # used in later tasks
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
