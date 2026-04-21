"""Tests for syng_bts.tcga: TCGA dataset loader."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import h5py
import numpy as np
import pytest

from syng_bts import tcga
from syng_bts.tcga import tcga_cache_dir


class TestTcgaCacheDir:
    def test_default_cache_dir(self, monkeypatch):
        monkeypatch.delenv("SYNG_BTS_CACHE_DIR", raising=False)
        result = tcga_cache_dir()
        assert result == Path.home() / ".cache" / "syng-bts" / "tcga"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNG_BTS_CACHE_DIR", str(tmp_path))
        result = tcga_cache_dir()
        assert result == tmp_path / "tcga"

    def test_does_not_create_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNG_BTS_CACHE_DIR", str(tmp_path))
        result = tcga_cache_dir()
        assert not result.exists()


class TestNetworkErrorClass:
    def test_is_oserror_subclass(self):
        assert issubclass(tcga._NetworkError, OSError)


# ---------------------------------------------------------------------------
# Helpers: build a tiny but schema-valid HDF5 file and matching manifest entry
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_test_h5(
    path: Path,
    *,
    dataset_name: str,
    n_raw_samples: int = 5,
    n_filtered_samples: int = 4,
    n_raw_features: int = 10,
    n_filtered_features: int = 6,
    n_synthetic: int = 8,
    group_labels: tuple[str, ...] = ("A", "B"),
    rng_seed: int = 0,
) -> dict:
    """Create a tiny but schema-valid v1.0 HDF5 file at ``path``.

    Returns a manifest-entry-shaped dict (``dataset_name``, ``cancer_type``,
    ``clinical_variable``, ``group_labels``, ``n_raw_samples``,
    ``n_filtered_samples``, ``n_raw_features``, ``n_filtered_features``,
    ``file``, ``file_size_bytes``, ``sha256``). Tests can pass these dicts
    directly to ``make_test_manifest``.
    """
    rng = np.random.default_rng(rng_seed)
    cancer_type, clinical_variable = dataset_name.split("_", 1)

    raw_features = [f"hsa-feat-{i}" for i in range(n_raw_features)]
    proc_features = raw_features[:n_filtered_features]

    def _alternating_groups(n: int) -> list[str]:
        return [group_labels[i % len(group_labels)] for i in range(n)]

    raw_expr = rng.random((n_raw_samples, n_raw_features))
    raw_groups = _alternating_groups(n_raw_samples)
    sample_ids = [f"S{i}" for i in range(n_raw_samples)]

    proc_data = {
        norm: {
            "expression": rng.random((n_filtered_samples, n_filtered_features)),
            "groups": _alternating_groups(n_filtered_samples),
            "feature_names": list(proc_features),
        }
        for norm in ("raw_norm", "TC", "DESeq")
    }

    synth_data = {}
    for norm in ("raw_norm", "TC", "DESeq"):
        synth_data[norm] = {}
        for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
            synth_data[norm][model] = {
                "expression": rng.random((n_synthetic, n_filtered_features)),
                "groups": _alternating_groups(n_synthetic),
            }

    norm_attrs = {
        "raw_norm": {
            "normalization_method": "log2_filtered",
            "transform": "log2(x+1)",
        },
        "TC": {
            "normalization_method": "TC_CPM",
            "transform": "log2(x+1)",
        },
        "DESeq": {
            "normalization_method": "DESeq_median_of_ratios",
            "transform": "log2(x+1)",
        },
    }
    model_kl = {"CVAE1_5": 5, "CVAE1_10": 10, "CVAE1_20": 20}

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["cancer_type"] = cancer_type
        f.attrs["clinical_variable"] = clinical_variable
        f.attrs["group_labels"] = list(group_labels)
        f.attrs["version"] = "1.0"
        f.attrs["creation_date"] = "2026-04-30T00:00:00+00:00"
        f.attrs["syng_bts_version"] = "3.3.2"
        f.attrs["n_raw_samples"] = n_raw_samples
        f.attrs["n_filtered_samples"] = n_filtered_samples
        f.attrs["n_raw_features"] = n_raw_features
        f.attrs["n_filtered_features"] = n_filtered_features

        raw = f.create_group("raw")
        raw.create_dataset("expression", data=raw_expr, dtype=np.float64)
        raw.create_dataset("groups", data=raw_groups, dtype=h5py.string_dtype())
        raw.create_dataset("sample_ids", data=sample_ids, dtype=h5py.string_dtype())
        raw.create_dataset(
            "feature_names", data=raw_features, dtype=h5py.string_dtype()
        )

        proc = f.create_group("processed")
        for norm in ("raw_norm", "TC", "DESeq"):
            ng = proc.create_group(norm)
            for k, v in norm_attrs[norm].items():
                ng.attrs[k] = v
            d = proc_data[norm]
            ng.create_dataset("expression", data=d["expression"], dtype=np.float64)
            ng.create_dataset(
                "groups", data=d["groups"], dtype=h5py.string_dtype()
            )
            ng.create_dataset(
                "feature_names",
                data=d["feature_names"],
                dtype=h5py.string_dtype(),
            )

        synth = f.create_group("synthetic")
        synth.attrs["epoch"] = 3000
        synth.attrs["early_stop_patience"] = 20
        synth.attrs["batch_frac"] = 0.1
        synth.attrs["learning_rate"] = 0.0005
        synth.attrs["random_seed"] = 42
        synth.attrs["new_size"] = n_synthetic
        synth.attrs["apply_log"] = False
        for norm in ("raw_norm", "TC", "DESeq"):
            ng = synth.create_group(norm)
            ng.create_dataset(
                "feature_names",
                data=proc_features,
                dtype=h5py.string_dtype(),
            )
            for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
                mg = ng.create_group(model)
                mg.attrs["kl_weight"] = model_kl[model]
                mg.attrs["reconstruction_term_weight"] = 1
                mg.attrs["epochs_trained"] = 100
                mg.attrs["normalization"] = norm
                d = synth_data[norm][model]
                mg.create_dataset(
                    "expression", data=d["expression"], dtype=np.float64
                )
                mg.create_dataset(
                    "groups", data=d["groups"], dtype=h5py.string_dtype()
                )

    return {
        "dataset_name": dataset_name,
        "cancer_type": cancer_type,
        "clinical_variable": clinical_variable,
        "group_labels": list(group_labels),
        "n_raw_samples": n_raw_samples,
        "n_filtered_samples": n_filtered_samples,
        "n_raw_features": n_raw_features,
        "n_filtered_features": n_filtered_features,
        "file": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256_of_file(path),
    }


def make_test_manifest(*entries: dict, version: str = "1.0") -> dict:
    """Wrap a sequence of entry dicts into a manifest-shaped dict."""
    return {
        "version": version,
        "created": "2026-04-30T00:00:00+00:00",
        "syng_bts_version": "3.3.2",
        "datasets": list(entries),
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal response object compatible with `urllib.request.urlopen`."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._buf.close()
        return False


class NetworkStub:
    """Container for served bytes and a log of fetched URLs."""

    def __init__(self):
        self.served: dict[str, bytes] = {}
        self.calls: list[str] = []

    def serve(self, url: str, content: bytes) -> None:
        self.served[url] = content


@pytest.fixture
def network_stub(monkeypatch) -> NetworkStub:
    """Replace `urllib.request.urlopen` with a stub that serves from a dict."""
    stub = NetworkStub()

    def fake_urlopen(url, timeout=None):  # noqa: ARG001
        url_str = url if isinstance(url, str) else url.full_url
        stub.calls.append(url_str)
        if url_str not in stub.served:
            raise urllib.error.URLError(f"Stub: no fixture for {url_str}")
        return _FakeResponse(stub.served[url_str])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return stub


@pytest.fixture
def cache_root(monkeypatch, tmp_path) -> Path:
    """Point ``SYNG_BTS_CACHE_DIR`` at ``tmp_path`` for the test."""
    monkeypatch.setenv("SYNG_BTS_CACHE_DIR", str(tmp_path))
    return tmp_path


# A canonical fixture URL used across tests.
FIXTURE_BASE_URL = "https://fixture.test/data-v1.0"
FIXTURE_MANIFEST_URL = f"{FIXTURE_BASE_URL}/manifest.json"


def _dataset_url(file: str) -> str:
    return f"{FIXTURE_BASE_URL}/{file}"


class TestFixtureBuilder:
    def test_make_test_h5_creates_readable_file(self, tmp_path):
        path = tmp_path / "BRCA_carcinoma.h5"
        entry = make_test_h5(path, dataset_name="BRCA_carcinoma")

        assert path.exists()
        assert entry["dataset_name"] == "BRCA_carcinoma"
        assert entry["cancer_type"] == "BRCA"
        assert entry["clinical_variable"] == "carcinoma"
        assert entry["file"] == "BRCA_carcinoma.h5"
        assert entry["file_size_bytes"] > 0
        assert len(entry["sha256"]) == 64

        with h5py.File(path, "r") as f:
            assert f.attrs["dataset_name"] == "BRCA_carcinoma"
            assert f.attrs["version"] == "1.0"
            assert f["raw/expression"].shape == (5, 10)
            assert f["processed/TC/expression"].shape == (4, 6)
            assert f["synthetic/TC/CVAE1_5/expression"].shape == (8, 6)

    def test_network_stub_serves_bytes(self, network_stub):
        import urllib.request

        network_stub.serve("http://example.com/foo", b"hello")
        with urllib.request.urlopen("http://example.com/foo") as resp:
            assert resp.read() == b"hello"
        assert network_stub.calls == ["http://example.com/foo"]

    def test_network_stub_unregistered_url_raises(self, network_stub):
        import urllib.request

        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen("http://nope.test/missing")

    def test_cache_root_sets_env(self, cache_root, monkeypatch):
        import os

        assert os.environ["SYNG_BTS_CACHE_DIR"] == str(cache_root)


class TestFetchManifest:
    def test_default_url_first_call_downloads_and_caches(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        # Arrange: fixture file + manifest, served via the network stub
        h5_path = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        h5_path.parent.mkdir()
        entry = make_test_h5(h5_path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())

        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        # Act
        result = tcga._fetch_manifest(None)

        # Assert
        assert result == manifest
        assert network_stub.calls == [FIXTURE_MANIFEST_URL]
        cached_path = cache_root / "tcga" / "1.0" / "manifest.json"
        assert cached_path.exists()
        assert json.loads(cached_path.read_text()) == manifest

        index_path = cache_root / "tcga" / ".url_index.json"
        assert index_path.exists()
        assert json.loads(index_path.read_text()) == {FIXTURE_MANIFEST_URL: "1.0"}

    def test_default_url_second_call_uses_cache(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        h5_path = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        h5_path.parent.mkdir()
        entry = make_test_h5(h5_path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())

        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        tcga._fetch_manifest(None)
        n_first = len(network_stub.calls)

        result = tcga._fetch_manifest(None)
        n_second = len(network_stub.calls)

        assert result == manifest
        assert n_first == 1
        assert n_second == 1  # no extra network call

    def test_override_url_always_fresh(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        h5_path = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        h5_path.parent.mkdir()
        entry = make_test_h5(h5_path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        override_url = "https://override.test/data-v1.0/manifest.json"
        network_stub.serve(override_url, json.dumps(manifest).encode())

        # Default URL is unset (placeholder) — irrelevant when override is passed.
        tcga._fetch_manifest(override_url)
        tcga._fetch_manifest(override_url)

        assert network_stub.calls == [override_url, override_url]
        # Override does NOT write to cache.
        assert not (cache_root / "tcga" / "1.0" / "manifest.json").exists()
        assert not (cache_root / "tcga" / ".url_index.json").exists()

    def test_malformed_json_raises_value_error(
        self, monkeypatch, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)
        network_stub.serve(FIXTURE_MANIFEST_URL, b"not json {{{")

        with pytest.raises(ValueError, match="manifest"):
            tcga._fetch_manifest(None)

    def test_network_failure_raises_network_error(self, monkeypatch, cache_root):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)
        # No `network_stub` fixture in this test, so urlopen will hit the real
        # network; instead, install a stub that always fails.

        def always_fail(url, timeout=None):  # noqa: ARG001
            raise urllib.error.URLError("simulated")

        monkeypatch.setattr("urllib.request.urlopen", always_fail)

        with pytest.raises(tcga._NetworkError, match="Failed to download"):
            tcga._fetch_manifest(None)


class TestResolveName:
    @pytest.fixture
    def manifest_with_three_entries(self):
        """Manifest with one unique cancer-type and two BRCA entries."""
        return {
            "version": "1.0",
            "datasets": [
                {"dataset_name": "UCS_primary_pathology_total_pelv_lnr"},
                {"dataset_name": "BRCA_breast_carcinoma_estrogen_receptor_status"},
                {"dataset_name": "BRCA_other_subtype"},
            ],
        }

    @pytest.fixture
    def manifest_unique(self):
        return {
            "version": "1.0",
            "datasets": [
                {"dataset_name": "UCS_primary_pathology_total_pelv_lnr"},
                {"dataset_name": "BRCA_breast_carcinoma_estrogen_receptor_status"},
                {"dataset_name": "LIHC_platelet_norm_range_lower"},
            ],
        }

    def test_full_name_match(self, manifest_unique):
        assert (
            tcga._resolve_name(
                "UCS_primary_pathology_total_pelv_lnr", manifest_unique
            )
            == "UCS_primary_pathology_total_pelv_lnr"
        )

    def test_short_alias_unique(self, manifest_unique):
        assert (
            tcga._resolve_name("BRCA", manifest_unique)
            == "BRCA_breast_carcinoma_estrogen_receptor_status"
        )

    def test_short_alias_ambiguous(self, manifest_with_three_entries):
        with pytest.raises(ValueError, match="Ambiguous"):
            tcga._resolve_name("BRCA", manifest_with_three_entries)

    def test_unknown_name(self, manifest_unique):
        with pytest.raises(ValueError, match="Unknown TCGA dataset"):
            tcga._resolve_name("FAKE", manifest_unique)

    def test_unknown_lists_available(self, manifest_unique):
        with pytest.raises(ValueError) as exc_info:
            tcga._resolve_name("FAKE", manifest_unique)
        msg = str(exc_info.value)
        assert "UCS_primary_pathology_total_pelv_lnr" in msg
        assert "BRCA_breast_carcinoma_estrogen_receptor_status" in msg
