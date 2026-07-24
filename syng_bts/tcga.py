"""TCGA dataset loader.

Downloads, caches, and exposes two packaged TCGA HDF5 bundles through a
small, ergonomic Python API: 24 miRNA cohorts (``modality="mirna"``, the
default) and 5 bulk RNA-seq cohorts (``modality="rnaseq"``).

Quick start::

    from syng_bts import load_tcga_dataset, list_tcga_datasets
    list_tcga_datasets()
    ds = load_tcga_dataset("BRCA")
    real_df, real_groups = ds.real("TC")

Files are downloaded once on first access and cached under
``~/.cache/syng-bts/tcga/<modality>/`` (override with the
``SYNG_BTS_CACHE_DIR`` environment variable).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
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

# URLs of the published dataset manifests, keyed by modality.
_DEFAULT_MANIFEST_URL: dict[str, str] = {
    "mirna": (
        "https://github.com/Omics-Data-Synthesis/SyNG-BTS/"
        "releases/download/data-v1.0/manifest.json"
    ),
    "rnaseq": (
        "https://github.com/Omics-Data-Synthesis/SyNG-BTS/"
        "releases/download/data-rnaseq-v1.0/manifest.json"
    ),
}

_CACHE_ENV_VAR = "SYNG_BTS_CACHE_DIR"
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "syng-bts"
_NETWORK_TIMEOUT_SECS = 60
# Wall-clock cap for one file download. Bulk RNA-seq assets reach ~550 MB, so
# 600s would have required 0.92 MB/s sustained; 3600s drops that to 0.15 MB/s.
_DOWNLOAD_DEADLINE_SECS = 3600
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB

VALID_MODALITIES = ("mirna", "rnaseq")
DEFAULT_MODALITY = "mirna"
VALID_NORMALIZATIONS = ("raw_norm", "TC", "DESeq")
VALID_MODELS = ("CVAE1_5", "CVAE1_10", "CVAE1_20")
DEFAULT_NORMALIZATION = "DESeq"
DEFAULT_MODEL = "CVAE1_5"
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "2.0")


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
    returns ``~/.cache/syng-bts/tcga``. The directory is **not** created by
    this call.

    Returns:
        The cache root for TCGA datasets. Versioned dataset files live under
        ``tcga_cache_dir() / <modality> / <manifest-version>``.

    Example:
        >>> from syng_bts import tcga_cache_dir
        >>> tcga_cache_dir()
        PosixPath('/home/alice/.cache/syng-bts/tcga')
    """
    root_str = os.environ.get(_CACHE_ENV_VAR)
    root = Path(root_str) if root_str else _DEFAULT_CACHE_ROOT
    return root / "tcga"


def _validate_modality(modality: str) -> None:
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Valid options: {VALID_MODALITIES}"
        )


def _modality_cache_dir(modality: str) -> Path:
    """Return the cache directory for one modality, without the version subdir."""
    _validate_modality(modality)
    return tcga_cache_dir() / modality


# ---------------------------------------------------------------------------
# Manifest fetching
# ---------------------------------------------------------------------------


def _url_index_path(modality: str) -> Path:
    return _modality_cache_dir(modality) / ".url_index.json"


def _read_url_index(modality: str) -> dict[str, str]:
    p = _url_index_path(modality)
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


def _write_url_index(modality: str, index: dict[str, str]) -> None:
    p = _url_index_path(modality)
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


def _check_manifest_modality(manifest: dict, modality: str, source: str) -> None:
    """Reject a manifest that declares a different modality.

    The published ``data-v1.0`` manifest predates this field, so absence is
    treated as ``"mirna"`` and never raises.
    """
    declared = manifest.get("modality")
    if declared is not None and declared != modality:
        raise ValueError(
            f"Manifest at {source} declares modality '{declared}', "
            f"but modality='{modality}' was requested. Pass "
            f"modality='{declared}' to load it."
        )


def _fetch_manifest(manifest_url: str | None, modality: str = DEFAULT_MODALITY) -> dict:
    """Return the parsed manifest dict for one modality.

    For ``manifest_url is None`` (default-URL flow): consult the URL-version
    index, return the cached manifest if present, otherwise download once and
    populate the cache. For an explicit ``manifest_url``: always download
    fresh, never cache.
    """
    _validate_modality(modality)

    if manifest_url is not None:
        manifest = _parse_manifest(_download_bytes(manifest_url), source=manifest_url)
        _check_manifest_modality(manifest, modality, manifest_url)
        return manifest

    url = _DEFAULT_MANIFEST_URL[modality]
    index = _read_url_index(modality)
    cached_version = index.get(url)
    if cached_version is not None:
        cached_path = _modality_cache_dir(modality) / cached_version / "manifest.json"
        if cached_path.exists():
            try:
                with open(cached_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # fall through to redownload

    manifest = _parse_manifest(_download_bytes(url), source=url)
    _check_manifest_modality(manifest, modality, url)
    version = str(manifest["version"])
    version_dir = _modality_cache_dir(modality) / version
    version_dir.mkdir(parents=True, exist_ok=True)
    with open(version_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    index[url] = version
    _write_url_index(modality, index)

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

    raise ValueError(f"Unknown TCGA dataset '{name}'. Available: {sorted(full_names)}")


# ---------------------------------------------------------------------------
# Public function: list_tcga_datasets
# ---------------------------------------------------------------------------


def list_tcga_datasets(
    *,
    short: bool = False,
    modality: str = DEFAULT_MODALITY,
    manifest_url: str | None = None,
) -> list[str]:
    """Return the names of all TCGA cohorts in the published manifest.

    Args:
        short: If ``True``, return short cohort codes (e.g. ``"BRCA"``).
            Otherwise return the full manifest dataset names.
        modality: Dataset modality to list. One of
            ``syng_bts.tcga.VALID_MODALITIES``. Defaults to ``"mirna"``.
        manifest_url: Override the published manifest URL. Defaults to the
            data-v1.0 release manifest.

    Returns:
        A list of dataset names. With ``short=False`` (default), names are
        the full manifest entries; with ``short=True``, the leading cohort
        code.

    Example:
        >>> from syng_bts import list_tcga_datasets
        >>> list_tcga_datasets(short=True)[:3]
        ['ACC', 'BLCA', 'BRCA']
    """
    _validate_modality(modality)
    manifest = _fetch_manifest(manifest_url, modality)
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
    Aborts with ``_NetworkError`` if the transfer exceeds
    ``_DOWNLOAD_DEADLINE_SECS`` wall-clock seconds, so a stalled connection
    fails fast instead of hanging forever. Wraps URL errors as
    ``_NetworkError``. Cleans up the ``.tmp`` on any exception.
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
                size_str = f"{total / 1e6:.1f} MB" if total > 0 else "size unknown"
                print(
                    f"Downloading {dest.name} ({size_str})…",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                start = time.monotonic()
                with open(tmp, "wb") as f:
                    while True:
                        if time.monotonic() - start > _DOWNLOAD_DEADLINE_SECS:
                            raise _NetworkError(
                                f"Download of {dest.name} aborted after "
                                f"{_DOWNLOAD_DEADLINE_SECS}s — transfer "
                                f"stalled or too slow.\n"
                                f"Check your network connection, or "
                                f"pre-stage the file at {dest} and set "
                                f"{_CACHE_ENV_VAR} if needed."
                            )
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


def _fetch_and_verify_h5(url: str, dest: Path, expected_sha256: str) -> None:
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
    """An expression subset returned by :class:`TCGADataset` accessors.

    Attributes:
        expression: A :class:`pandas.DataFrame` of shape
            ``(n_samples, n_features)``. Rows are TCGA samples (indexed by
            sample barcode); columns are miRNA features.
        groups: A :class:`pandas.Series` aligned to ``expression.index`` with
            categorical group labels (e.g. tumor vs. normal), or ``None`` for
            synthetic data generated by an unconditional model (plain VAE),
            which carries no labels.
        metadata: Dict of HDF5 attributes captured at dataset assembly time
            (e.g. version, normalization, source).
    """

    expression: pd.DataFrame
    groups: pd.Series | None
    metadata: dict[str, Any]


class TCGADataset:
    """A loaded TCGA cohort with real and synthetic accessors.

    Returned by :func:`load_tcga_dataset`. Wraps a single HDF5 file whose
    layout depends on ``modality``:

    - ``modality="mirna"`` (schema 1.0): a raw expression matrix (``raw``),
      three normalizations (``raw_norm``, ``TC``, ``DESeq``), and nine
      synthetic groups (three CVAE models × three normalizations).
    - ``modality="rnaseq"`` (schema 2.0): no raw matrix (``raw`` is
      ``None``); three normalizations, each with synthetic data nested one
      level deeper — by offline-augmentation setting, then by the cohort's
      own model family (CVAE or VAE) and KL weight.

    Use :meth:`real` to access real expression data and :meth:`synth` to
    access a synthetic counterpart.
    """

    def __init__(
        self,
        *,
        name: str,
        modality: str,
        cancer_type: str,
        clinical_variable: str,
        group_labels: list[str],
        n_raw_samples: int | None = None,
        n_filtered_samples: int | None = None,
        n_raw_features: int | None = None,
        n_filtered_features: int | None = None,
        source_label: str = "",
        n_samples: int | None = None,
        n_train: int | None = None,
        n_test: int | None = None,
        n_features: int | None = None,
        model_family: str | None = None,
        synthetic_has_groups: bool | None = None,
        schema_version: str,
        creation_date: str,
        syng_bts_version: str,
        raw: Subset | None = None,
        processed: dict[str, Subset],
        synthetic: dict[str, Any],
    ) -> None:
        self.name = name
        self.modality = modality
        self.cancer_type = cancer_type
        self.clinical_variable = clinical_variable
        self.group_labels = group_labels
        self.n_raw_samples = n_raw_samples
        self.n_filtered_samples = n_filtered_samples
        self.n_raw_features = n_raw_features
        self.n_filtered_features = n_filtered_features
        self.source_label = source_label
        self.n_samples = n_samples
        self.n_train = n_train
        self.n_test = n_test
        self.n_features = n_features
        self.model_family = model_family
        self.synthetic_has_groups = synthetic_has_groups
        self.schema_version = schema_version
        self.creation_date = creation_date
        self.syng_bts_version = syng_bts_version
        self.raw = raw
        self.processed = processed
        self.synthetic = synthetic

    # --- Convenience accessors --------------------------------------------

    _OFF_AUG_NONE = "none"

    def _offaug_key(self, off_aug: str | None) -> str:
        """Map the public ``off_aug`` value onto an HDF5 group name."""
        key = self._OFF_AUG_NONE if off_aug is None else str(off_aug)
        available = sorted(self.synthetic[DEFAULT_NORMALIZATION])
        if key not in available:
            raise ValueError(
                f"Invalid off_aug '{off_aug}' for {self.name}. "
                f"Valid options: {tuple(available)}"
            )
        return key

    def _models_by_kl(self, normalization: str, offaug_key: str) -> tuple[str, ...]:
        """Model names ordered by ascending kl_weight, not alphabetically."""
        candidates = self.synthetic[normalization][offaug_key]
        return tuple(
            sorted(
                candidates,
                key=lambda m: int(candidates[m].metadata.get("kl_weight", 0)),
            )
        )

    def _default_model(self, normalization: str, offaug_key: str) -> str:
        """Return the model with the lowest kl_weight for this cohort."""
        return self._models_by_kl(normalization, offaug_key)[0]

    def real(
        self,
        normalization: str = DEFAULT_NORMALIZATION,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(expression, groups)`` for one processed normalization.

        Args:
            normalization: One of ``"raw_norm"``, ``"TC"``, or ``"DESeq"``
                (default). See ``syng_bts.tcga.VALID_NORMALIZATIONS``.

        Returns:
            A ``(expression, groups)`` tuple where ``expression`` is a
            :class:`pandas.DataFrame` of shape ``(n_samples, n_features)`` and
            ``groups`` is a :class:`pandas.Series` of group labels aligned to
            ``expression.index``. For ``modality="rnaseq"`` cohorts, the
            returned frame covers train and test together. To access the
            per-slice metadata dict, use the underlying :class:`Subset`
            directly via ``dataset.processed[normalization]``.

        Raises:
            ValueError: If ``normalization`` is not in
                ``syng_bts.tcga.VALID_NORMALIZATIONS``.

        Example:
            >>> ds = load_tcga_dataset("BRCA")
            >>> real_df, real_groups = ds.real("TC")
            >>> real_df.shape
            (1144, 570)
        """
        if normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"Invalid normalization '{normalization}'. "
                f"Valid options: {VALID_NORMALIZATIONS}"
            )
        sub = self.processed[normalization]
        return sub.expression, sub.groups

    def synth(
        self,
        normalization: str = DEFAULT_NORMALIZATION,
        model: str | None = None,
        off_aug: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Return ``(expression, groups)`` for one synthetic configuration.

        Args:
            normalization: One of ``"raw_norm"``, ``"TC"``, or ``"DESeq"``
                (default).
            model: For ``modality="mirna"``, one of
                ``syng_bts.tcga.VALID_MODELS`` (default ``"CVAE1_5"``). For
                ``modality="rnaseq"``, one of the cohort's own models — e.g.
                ``"CVAE1-50"`` for COAD and SKCM, ``"VAE1-50"`` for LAML, PAAD
                and READ. ``None`` selects the cohort's lowest KL weight.
            off_aug: Offline-augmentation setting, ``modality="rnaseq"`` only.
                ``None`` (default) or the equivalent string ``"none"`` selects
                the un-augmented runs; ``"AE_head"`` selects the runs trained
                with ``off_aug="AE_head", AE_head_num=2``.

        Returns:
            An ``(expression, groups)`` tuple. ``groups`` is ``None`` for bulk
            cohorts generated by an unconditional model (plain VAE), which
            produce no labels. Per-slice metadata is on the underlying
            :class:`Subset`, via ``dataset.synthetic[norm][model]`` for miRNA
            and ``dataset.synthetic[norm][off_aug][model]`` for bulk.

        Raises:
            ValueError: If ``normalization``, ``model`` or ``off_aug`` is not
                valid for this dataset, or if ``off_aug`` is passed to a miRNA
                dataset.

        Example:
            >>> ds = load_tcga_dataset("SKCM", modality="rnaseq")
            >>> synth_df, synth_groups = ds.synth("DESeq", "CVAE1-50")
            >>> synth_df.shape
            (1000, 1091)
        """
        if normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"Invalid normalization '{normalization}'. "
                f"Valid options: {VALID_NORMALIZATIONS}"
            )

        if self.modality == "mirna":
            if off_aug is not None:
                raise ValueError(
                    "off_aug is only available for modality='rnaseq'; this "
                    f"dataset has modality='{self.modality}'."
                )
            model = DEFAULT_MODEL if model is None else model
            if model not in VALID_MODELS:
                raise ValueError(
                    f"Invalid model '{model}'. Valid options: {VALID_MODELS}"
                )
            sub = self.synthetic[normalization][model]
            return sub.expression, sub.groups

        offaug_key = self._offaug_key(off_aug)
        available = self.synthetic[normalization][offaug_key]
        if model is None:
            model = self._default_model(normalization, offaug_key)
        if model not in available:
            raise ValueError(
                f"Invalid model '{model}' for {self.name}. "
                f"Valid options: {self._models_by_kl(normalization, offaug_key)}"
            )
        sub = available[model]
        return sub.expression, sub.groups

    def __repr__(self) -> str:
        labels = "/".join(self.group_labels) if self.group_labels else "—"

        if self.modality == "rnaseq":
            present = self.synthetic[DEFAULT_NORMALIZATION]
            offaugs = [k for k in ("none", "AE_head") if k in present]
            models = list(self._models_by_kl(DEFAULT_NORMALIZATION, offaugs[0]))
            first = present[offaugs[0]][models[0]]
            synth_n = int(first.metadata.get("new_size", len(first.expression)))
            return (
                f"TCGADataset(name='{self.name}', modality='rnaseq',\n"
                f"  cancer_type='{self.cancer_type}'\n"
                f"  real: {self.n_samples} samples "
                f"({self.n_train} train + {self.n_test} test) × "
                f"{self.n_features} features\n"
                f"  groups: {labels}\n"
                f"  processed: {list(self.processed)}\n"
                f"  synthetic: {list(self.synthetic)} × {offaugs} × {models} "
                f"({synth_n} samples each, "
                f"groups={'yes' if self.synthetic_has_groups else 'no'})\n"
                f"  schema_version: {self.schema_version}, "
                f"created: {self.creation_date})"
            )

        try:
            synth_n = int(
                self.synthetic[DEFAULT_NORMALIZATION][DEFAULT_MODEL].metadata.get(
                    "new_size", 0
                )
            )
        except (KeyError, ValueError, TypeError):
            synth_n = 0
        synth_n_str = f"{synth_n} samples each" if synth_n else "n samples each"
        return (
            f"TCGADataset(name='{self.name}', cancer_type='{self.cancer_type}',\n"
            f"  raw: {self.n_raw_samples} samples × "
            f"{self.n_raw_features} features\n"
            f"  filtered: {self.n_filtered_samples} samples × "
            f"{self.n_filtered_features} features\n"
            f"  groups: {labels}\n"
            f"  processed: {list(self.processed)}\n"
            f"  synthetic: {list(self.synthetic)} × "
            f"{list(VALID_MODELS)} ({synth_n_str})\n"
            f"  schema_version: {self.schema_version}, "
            f"created: {self.creation_date})"
        )


# ---------------------------------------------------------------------------
# HDF5 → DataFrame construction
# ---------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    """Decode bytes / numpy scalars / numpy arrays into pure-Python values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [
            v.decode()
            if isinstance(v, bytes)
            else v.item()
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
    expect_groups: bool = True,
) -> Subset:
    """Construct a Subset from an HDF5 group containing expression + groups.

    ``feature_names`` is read from the group if not supplied (raw/processed
    case); pass it explicitly for synthetic groups, where feature_names lives
    one level up. When ``expect_groups`` is False and no ``groups`` dataset is
    present, ``Subset.groups`` is ``None`` — this is the case for synthetic
    data from unconditional (plain VAE) models.
    """
    expr_arr = group["expression"][:]

    groups: list[str] | None = None
    if "groups" in group:
        groups = _read_strings(group["groups"])
    elif expect_groups:
        raise KeyError(f"{group.name}/groups")

    if feature_names is None:
        feature_names = _read_strings(group["feature_names"])

    n_rows = len(groups) if groups is not None else expr_arr.shape[0]

    if sample_ids is not None:
        index = pd.Index(sample_ids, name="sample_id")
    else:
        index = pd.RangeIndex(n_rows)

    expression = pd.DataFrame(expr_arr, columns=feature_names, index=index)
    groups_series = (
        pd.Series(groups, index=index, name="groups") if groups is not None else None
    )

    metadata = _read_attrs(group)
    if extra_metadata:
        # Group-specific attrs win over inherited shared attrs.
        metadata = {**extra_metadata, **metadata}

    return Subset(expression=expression, groups=groups_series, metadata=metadata)


def _build_mirna(f: h5py.File, attrs: dict[str, Any]) -> TCGADataset:
    """Read a schema-1.0 (miRNA) file and return a TCGADataset."""
    # Raw subset (with sample_ids)
    raw_grp = f["raw"]
    raw_sample_ids = (
        _read_strings(raw_grp["sample_ids"]) if "sample_ids" in raw_grp else None
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
        modality="mirna",
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


def _build_rnaseq(f: h5py.File, attrs: dict[str, Any]) -> TCGADataset:
    """Read a schema-2.0 (bulk RNA-seq) file and return a TCGADataset."""
    synthetic_has_groups = bool(attrs.get("synthetic_has_groups", True))

    processed: dict[str, Subset] = {}
    for norm in VALID_NORMALIZATIONS:
        grp = f[f"processed/{norm}"]
        sample_ids = _read_strings(grp["sample_ids"])
        sub = _build_subset_from_group(grp, sample_ids=sample_ids)
        # `split` is a dataset, not an attr, so _read_attrs never sees it.
        sub.metadata["split"] = _read_strings(grp["split"])
        processed[norm] = sub

    synth_root = f["synthetic"]
    shared_attrs = _read_attrs(synth_root)

    synthetic: dict[str, dict[str, dict[str, Subset]]] = {}
    for norm in VALID_NORMALIZATIONS:
        norm_grp = synth_root[norm]
        features = _read_strings(norm_grp["feature_names"])
        synthetic[norm] = {}
        for offaug in norm_grp:
            offaug_grp = norm_grp[offaug]
            if not isinstance(offaug_grp, h5py.Group):
                continue  # skip the feature_names dataset
            offaug_attrs = {**shared_attrs, **_read_attrs(offaug_grp)}
            synthetic[norm][offaug] = {}
            for model in offaug_grp:
                synthetic[norm][offaug][model] = _build_subset_from_group(
                    offaug_grp[model],
                    feature_names=features,
                    extra_metadata=offaug_attrs,
                    expect_groups=synthetic_has_groups,
                )

    return TCGADataset(
        name=str(attrs["dataset_name"]),
        modality="rnaseq",
        cancer_type=str(attrs["cancer_type"]),
        clinical_variable=str(attrs.get("clinical_variable", "")),
        source_label=str(attrs.get("source_label", "")),
        group_labels=[str(x) for x in attrs.get("group_labels", [])],
        n_samples=int(attrs["n_samples"]),
        n_train=int(attrs["n_train"]),
        n_test=int(attrs["n_test"]),
        n_features=int(attrs["n_features"]),
        model_family=str(attrs.get("model_family", "")),
        synthetic_has_groups=synthetic_has_groups,
        schema_version=str(attrs["version"]),
        creation_date=str(attrs["creation_date"]),
        syng_bts_version=str(attrs["syng_bts_version"]),
        raw=None,
        processed=processed,
        synthetic=synthetic,
    )


def _build_dataset_from_h5(path: Path) -> TCGADataset:
    """Read a TCGA HDF5 file and return a TCGADataset, dispatching on schema."""
    with h5py.File(path, "r") as f:
        attrs = {k: _decode(v) for k, v in f.attrs.items()}
        schema_version = str(attrs.get("version", "1.0"))

        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"{path.name} declares schema version '{schema_version}', which "
                f"this version of syng-bts does not support. Supported: "
                f"{SUPPORTED_SCHEMA_VERSIONS}. Upgrade syng-bts to read it."
            )

        if schema_version == "2.0":
            return _build_rnaseq(f, attrs)
        return _build_mirna(f, attrs)


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
    modality: str = DEFAULT_MODALITY,
    manifest_url: str | None = None,
) -> TCGADataset:
    """Download (if needed) and return a TCGA cohort as a :class:`TCGADataset`.

    On first call for a given dataset, the loader fetches the manifest,
    downloads the corresponding HDF5 file, verifies its sha256, and caches
    the file under ``tcga_cache_dir() / <modality> / <manifest-version>``.
    Subsequent calls reuse the cached file.

    Args:
        name: Cohort code (e.g. ``"BRCA"``) or full dataset name from the
            manifest. Cancer-type prefixes resolve to the canonical entry
            when unambiguous.
        force: If ``True``, redownload even when a cached file exists.
        modality: Dataset modality to load. One of
            ``syng_bts.tcga.VALID_MODALITIES``. Defaults to ``"mirna"``.
        manifest_url: Override the published manifest URL (useful for
            staging mirrors). Defaults to the data-v1.0 release manifest.

    Returns:
        A :class:`TCGADataset` exposing :meth:`TCGADataset.real` and
        :meth:`TCGADataset.synth` accessors.

    Raises:
        ValueError: If ``name`` does not match any cohort in the manifest,
            if the cached HDF5 file is corrupt (pass ``force=True`` to
            redownload), or if the sha256 checksum fails twice after retry.
        OSError: If the manifest or HDF5 file cannot be downloaded due to a
            network failure.

    Example:
        >>> from syng_bts import load_tcga_dataset
        >>> ds = load_tcga_dataset("BRCA")
        >>> real_df, real_groups = ds.real()
        >>> real_df.shape
        (1144, 570)
    """
    _validate_modality(modality)
    manifest = _fetch_manifest(manifest_url, modality)
    full_name = _resolve_name(name, manifest)
    entry = _entry_for(manifest, full_name)

    declared_schema = str(manifest.get("schema_version", "1.0"))
    if declared_schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Manifest declares schema version '{declared_schema}', which this "
            f"version of syng-bts does not support. Supported: "
            f"{SUPPORTED_SCHEMA_VERSIONS}. Upgrade syng-bts to read it."
        )

    version = str(manifest["version"])
    version_dir = _modality_cache_dir(modality) / version
    version_dir.mkdir(parents=True, exist_ok=True)

    cached_h5 = version_dir / entry["file"]

    if force or not cached_h5.exists():
        url = _dataset_url_from_manifest(
            manifest_url
            if manifest_url is not None
            else _DEFAULT_MANIFEST_URL[modality],
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


# ---------------------------------------------------------------------------
# Public function: clear_tcga_cache
# ---------------------------------------------------------------------------


def clear_tcga_cache() -> None:
    """Remove the entire TCGA cache directory.

    Deletes ``tcga_cache_dir()`` recursively. The next call to
    :func:`load_tcga_dataset` will redownload from the manifest. Use this
    for full cleanup; for per-dataset redownload, prefer
    :func:`load_tcga_dataset` with ``force=True``.

    Example:
        >>> from syng_bts import clear_tcga_cache
        >>> clear_tcga_cache()
    """
    target = tcga_cache_dir()
    if target.exists():
        shutil.rmtree(target)
