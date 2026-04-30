"""Tests for syng_bts.tcga: TCGA dataset loader."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from syng_bts import tcga
from syng_bts.tcga import (
    clear_tcga_cache,
    list_tcga_datasets,
    load_tcga_dataset,
    tcga_cache_dir,
)


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


class TestListTcgaDatasets:
    def _three_dataset_manifest(self, tmp_path):
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()
        entries = [
            make_test_h5(
                h5_dir / "UCS_primary_pathology.h5",
                dataset_name="UCS_primary_pathology",
            ),
            make_test_h5(
                h5_dir / "BRCA_carcinoma.h5",
                dataset_name="BRCA_carcinoma",
            ),
            make_test_h5(
                h5_dir / "LIHC_platelet.h5",
                dataset_name="LIHC_platelet",
            ),
        ]
        return make_test_manifest(*entries)

    def test_default_returns_full_names_sorted(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        manifest = self._three_dataset_manifest(tmp_path)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        result = list_tcga_datasets()

        assert result == [
            "BRCA_carcinoma",
            "LIHC_platelet",
            "UCS_primary_pathology",
        ]

    def test_short_returns_aliases_sorted_unique(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        manifest = self._three_dataset_manifest(tmp_path)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        result = list_tcga_datasets(short=True)

        assert result == ["BRCA", "LIHC", "UCS"]

    def test_manifest_url_override(self, network_stub, cache_root, tmp_path):
        manifest = self._three_dataset_manifest(tmp_path)
        url = "https://override.test/manifest.json"
        network_stub.serve(url, json.dumps(manifest).encode())

        result = list_tcga_datasets(manifest_url=url)

        assert result == [
            "BRCA_carcinoma",
            "LIHC_platelet",
            "UCS_primary_pathology",
        ]


class TestFetchAndVerifyH5:
    def test_successful_download(
        self, network_stub, cache_root, tmp_path
    ):
        # Build a fixture HDF5, capture its bytes and sha256
        src = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        src.parent.mkdir()
        entry = make_test_h5(src, dataset_name="BRCA_carcinoma")
        h5_bytes = src.read_bytes()
        url = _dataset_url(entry["file"])
        network_stub.serve(url, h5_bytes)

        dest = tmp_path / "out" / "BRCA_carcinoma.h5"
        dest.parent.mkdir()

        tcga._fetch_and_verify_h5(url, dest, entry["sha256"])

        assert dest.exists()
        assert dest.read_bytes() == h5_bytes
        # No leftover .tmp
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()
        assert network_stub.calls == [url]

    def test_sha256_mismatch_then_success_retry(
        self, network_stub, cache_root, tmp_path, monkeypatch
    ):
        # First response: corrupt; second response: correct.
        src = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        src.parent.mkdir()
        entry = make_test_h5(src, dataset_name="BRCA_carcinoma")
        good_bytes = src.read_bytes()
        bad_bytes = good_bytes[:-1] + b"\xff"  # one-byte corruption

        url = _dataset_url(entry["file"])
        responses = iter([bad_bytes, good_bytes])

        def fake_urlopen(u, timeout=None):  # noqa: ARG001
            assert u == url
            return _FakeResponse(next(responses))

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        dest = tmp_path / "out" / "BRCA_carcinoma.h5"
        dest.parent.mkdir()

        tcga._fetch_and_verify_h5(url, dest, entry["sha256"])

        assert dest.exists()
        assert dest.read_bytes() == good_bytes

    def test_sha256_mismatch_twice_raises(
        self, network_stub, cache_root, tmp_path
    ):
        src = tmp_path / "_fixture" / "BRCA_carcinoma.h5"
        src.parent.mkdir()
        entry = make_test_h5(src, dataset_name="BRCA_carcinoma")
        good_bytes = src.read_bytes()
        bad_bytes = good_bytes[:-1] + b"\xff"

        url = _dataset_url(entry["file"])
        network_stub.serve(url, bad_bytes)

        dest = tmp_path / "out" / "BRCA_carcinoma.h5"
        dest.parent.mkdir()

        with pytest.raises(ValueError, match="Checksum mismatch"):
            tcga._fetch_and_verify_h5(url, dest, entry["sha256"])

        # Both .h5 and .tmp should be cleaned up
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()

    def test_network_error_wrapped(self, monkeypatch, cache_root, tmp_path):
        url = _dataset_url("BRCA.h5")
        dest = tmp_path / "out" / "BRCA.h5"
        dest.parent.mkdir()

        def always_fail(u, timeout=None):  # noqa: ARG001
            raise urllib.error.URLError("simulated")

        monkeypatch.setattr("urllib.request.urlopen", always_fail)

        with pytest.raises(tcga._NetworkError, match="Failed to download"):
            tcga._fetch_and_verify_h5(url, dest, "deadbeef" * 8)


class TestBuildDatasetFromHdf5:
    @pytest.fixture
    def h5_file(self, tmp_path):
        path = tmp_path / "BRCA_carcinoma.h5"
        make_test_h5(
            path,
            dataset_name="BRCA_carcinoma",
            n_raw_samples=5,
            n_filtered_samples=4,
            n_raw_features=10,
            n_filtered_features=6,
            n_synthetic=8,
        )
        return path

    def test_returns_tcga_dataset(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert isinstance(ds, tcga.TCGADataset)

    def test_root_attributes_populated(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert ds.name == "BRCA_carcinoma"
        assert ds.cancer_type == "BRCA"
        assert ds.clinical_variable == "carcinoma"
        assert ds.group_labels == ["A", "B"]
        assert ds.n_raw_samples == 5
        assert ds.n_filtered_samples == 4
        assert ds.n_raw_features == 10
        assert ds.n_filtered_features == 6
        assert ds.schema_version == "1.0"
        assert ds.creation_date == "2026-04-30T00:00:00+00:00"
        assert ds.syng_bts_version == "3.3.2"

    def test_raw_subset_shape_and_index(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert isinstance(ds.raw, tcga.Subset)
        assert ds.raw.expression.shape == (5, 10)
        assert list(ds.raw.expression.columns) == [
            f"hsa-feat-{i}" for i in range(10)
        ]
        assert list(ds.raw.expression.index) == [f"S{i}" for i in range(5)]
        assert list(ds.raw.groups.index) == [f"S{i}" for i in range(5)]
        assert ds.raw.groups.tolist() == ["A", "B", "A", "B", "A"]

    def test_processed_subset_shape_and_rangeindex(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert set(ds.processed.keys()) == {"raw_norm", "TC", "DESeq"}
        for norm in ("raw_norm", "TC", "DESeq"):
            sub = ds.processed[norm]
            assert isinstance(sub, tcga.Subset)
            assert sub.expression.shape == (4, 6)
            assert isinstance(sub.expression.index, pd.RangeIndex)
            assert list(sub.expression.index) == list(range(4))
            assert list(sub.groups.index) == list(range(4))

    def test_processed_metadata(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert ds.processed["TC"].metadata["normalization_method"] == "TC_CPM"
        assert ds.processed["TC"].metadata["transform"] == "log2(x+1)"
        assert (
            ds.processed["DESeq"].metadata["normalization_method"]
            == "DESeq_median_of_ratios"
        )

    def test_synthetic_subset_shape_and_metadata(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        assert set(ds.synthetic.keys()) == {"raw_norm", "TC", "DESeq"}
        for norm in ("raw_norm", "TC", "DESeq"):
            assert set(ds.synthetic[norm].keys()) == {
                "CVAE1_5",
                "CVAE1_10",
                "CVAE1_20",
            }
            for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
                sub = ds.synthetic[norm][model]
                assert isinstance(sub, tcga.Subset)
                assert sub.expression.shape == (8, 6)
                assert isinstance(sub.expression.index, pd.RangeIndex)
                assert sub.metadata["normalization"] == norm
                assert sub.metadata["epochs_trained"] == 100
                assert sub.metadata["reconstruction_term_weight"] == 1
        # Per-model attrs from MODEL_PARAMS
        assert ds.synthetic["TC"]["CVAE1_5"].metadata["kl_weight"] == 5
        assert ds.synthetic["TC"]["CVAE1_10"].metadata["kl_weight"] == 10
        assert ds.synthetic["TC"]["CVAE1_20"].metadata["kl_weight"] == 20

    def test_synthetic_inherits_shared_attrs(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        meta = ds.synthetic["TC"]["CVAE1_5"].metadata
        # /synthetic root attrs (epoch, batch_frac, ...) merged into each subset
        assert meta["epoch"] == 3000
        assert meta["new_size"] == 8
        assert meta["random_seed"] == 42

    def test_subset_is_frozen(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        with pytest.raises((AttributeError, TypeError)):
            ds.raw.expression = pd.DataFrame()  # type: ignore[misc]

    def test_no_open_file_handles_after_construction(self, h5_file):
        ds = tcga._build_dataset_from_h5(h5_file)
        # Deletion of the source HDF5 should not affect the in-memory dataset
        h5_file.unlink()
        assert ds.raw.expression.shape == (5, 10)


class TestLoadTcgaDataset:
    @pytest.fixture
    def two_dataset_setup(self, tmp_path, network_stub):
        """Stage two fixture datasets (BRCA + UCS) and serve them."""
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()

        brca_path = h5_dir / "BRCA_carcinoma.h5"
        ucs_path = h5_dir / "UCS_other.h5"
        brca_entry = make_test_h5(brca_path, dataset_name="BRCA_carcinoma")
        ucs_entry = make_test_h5(ucs_path, dataset_name="UCS_other")
        manifest = make_test_manifest(brca_entry, ucs_entry)

        network_stub.serve(
            FIXTURE_MANIFEST_URL, json.dumps(manifest).encode()
        )
        network_stub.serve(_dataset_url("BRCA_carcinoma.h5"), brca_path.read_bytes())
        network_stub.serve(_dataset_url("UCS_other.h5"), ucs_path.read_bytes())

        return {"manifest": manifest, "brca_entry": brca_entry}

    def test_load_full_name(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        ds = load_tcga_dataset("BRCA_carcinoma")

        assert isinstance(ds, tcga.TCGADataset)
        assert ds.name == "BRCA_carcinoma"

        # Cached on disk
        cached = cache_root / "tcga" / "1.0" / "BRCA_carcinoma.h5"
        assert cached.exists()

    def test_load_short_alias(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        ds = load_tcga_dataset("BRCA")

        assert ds.name == "BRCA_carcinoma"

    def test_unknown_name_raises(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        with pytest.raises(ValueError, match="Unknown TCGA dataset"):
            load_tcga_dataset("FAKE")

    def test_cache_hit_no_network(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        load_tcga_dataset("BRCA_carcinoma")
        n_first = len(network_stub.calls)
        load_tcga_dataset("BRCA_carcinoma")
        n_second = len(network_stub.calls)

        # First call: manifest + h5. Second call: zero new network calls.
        assert n_first == 2
        assert n_second == n_first

    def test_force_redownload(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        load_tcga_dataset("BRCA_carcinoma")
        n_first = len(network_stub.calls)
        load_tcga_dataset("BRCA_carcinoma", force=True)
        n_second = len(network_stub.calls)

        # force=True triggers one extra h5 download (manifest still cached).
        assert n_second == n_first + 1

    def test_manifest_url_override(
        self, two_dataset_setup, network_stub, cache_root
    ):
        ds = load_tcga_dataset(
            "BRCA_carcinoma", manifest_url=FIXTURE_MANIFEST_URL
        )
        assert ds.name == "BRCA_carcinoma"

    def test_corrupt_cached_h5_raises_helpful_error(
        self, monkeypatch, two_dataset_setup, network_stub, cache_root
    ):
        """If a cached HDF5 file is unreadable, the loader wraps the h5py
        error with a hint to pass force=True."""
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        # Populate the cache normally.
        load_tcga_dataset("BRCA_carcinoma")
        cached = cache_root / "tcga" / "1.0" / "BRCA_carcinoma.h5"
        assert cached.exists()

        # Corrupt the cached file (truncate to garbage that h5py rejects).
        cached.write_bytes(b"not an hdf5 file")

        with pytest.raises(ValueError, match="Corrupt HDF5"):
            load_tcga_dataset("BRCA_carcinoma")


class TestClearTcgaCache:
    def test_no_op_when_missing(self, cache_root):
        # Cache dir does not exist yet
        assert not (cache_root / "tcga").exists()
        clear_tcga_cache()
        assert not (cache_root / "tcga").exists()

    def test_removes_everything(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        # Populate the cache
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()
        brca_path = h5_dir / "BRCA_carcinoma.h5"
        entry = make_test_h5(brca_path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        network_stub.serve(
            FIXTURE_MANIFEST_URL, json.dumps(manifest).encode()
        )
        network_stub.serve(
            _dataset_url("BRCA_carcinoma.h5"), brca_path.read_bytes()
        )
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        load_tcga_dataset("BRCA_carcinoma")
        assert (cache_root / "tcga" / "1.0" / "BRCA_carcinoma.h5").exists()
        assert (cache_root / "tcga" / "1.0" / "manifest.json").exists()
        assert (cache_root / "tcga" / ".url_index.json").exists()

        clear_tcga_cache()

        assert not (cache_root / "tcga").exists()

    def test_subsequent_load_redownloads(
        self, monkeypatch, network_stub, cache_root, tmp_path
    ):
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()
        brca_path = h5_dir / "BRCA_carcinoma.h5"
        entry = make_test_h5(brca_path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        network_stub.serve(
            FIXTURE_MANIFEST_URL, json.dumps(manifest).encode()
        )
        network_stub.serve(
            _dataset_url("BRCA_carcinoma.h5"), brca_path.read_bytes()
        )
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)

        load_tcga_dataset("BRCA_carcinoma")
        n_before_clear = len(network_stub.calls)
        clear_tcga_cache()
        load_tcga_dataset("BRCA_carcinoma")
        n_after_clear = len(network_stub.calls)

        # Both manifest and HDF5 must be redownloaded
        assert n_after_clear == n_before_clear + 2


class TestConvenienceAccessors:
    @pytest.fixture
    def loaded_dataset(self, monkeypatch, network_stub, cache_root, tmp_path):
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()
        path = h5_dir / "BRCA_carcinoma.h5"
        entry = make_test_h5(path, dataset_name="BRCA_carcinoma")
        manifest = make_test_manifest(entry)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())
        network_stub.serve(_dataset_url(entry["file"]), path.read_bytes())
        monkeypatch.setattr(tcga, "_DEFAULT_MANIFEST_URL", FIXTURE_MANIFEST_URL)
        return load_tcga_dataset("BRCA_carcinoma")

    def test_real_default_is_TC(self, loaded_dataset):
        df, groups = loaded_dataset.real()
        # Default normalization is "TC"
        assert df.equals(loaded_dataset.processed["TC"].expression)
        assert groups.equals(loaded_dataset.processed["TC"].groups)

    def test_real_explicit_norm(self, loaded_dataset):
        df, groups = loaded_dataset.real(normalization="DESeq")
        assert df.equals(loaded_dataset.processed["DESeq"].expression)
        assert groups.equals(loaded_dataset.processed["DESeq"].groups)

    def test_real_invalid_norm_raises(self, loaded_dataset):
        with pytest.raises(ValueError, match="normalization"):
            loaded_dataset.real(normalization="XYZ")

    def test_synth_default(self, loaded_dataset):
        df, groups = loaded_dataset.synth()
        # Defaults: normalization="TC", model="CVAE1_5"
        assert df.equals(loaded_dataset.synthetic["TC"]["CVAE1_5"].expression)
        assert groups.equals(loaded_dataset.synthetic["TC"]["CVAE1_5"].groups)

    def test_synth_explicit(self, loaded_dataset):
        df, groups = loaded_dataset.synth(
            normalization="DESeq", model="CVAE1_20"
        )
        assert df.equals(
            loaded_dataset.synthetic["DESeq"]["CVAE1_20"].expression
        )
        assert groups.equals(
            loaded_dataset.synthetic["DESeq"]["CVAE1_20"].groups
        )

    def test_synth_invalid_model_raises(self, loaded_dataset):
        with pytest.raises(ValueError, match="model"):
            loaded_dataset.synth(normalization="TC", model="FAKE")

    def test_repr_contains_useful_info(self, loaded_dataset):
        text = repr(loaded_dataset)
        assert "BRCA_carcinoma" in text
        assert "TCGADataset" in text
        assert "5" in text  # n_raw_samples
        assert "4" in text  # n_filtered_samples
        # Synthetic count is read dynamically from metadata, not hardcoded.
        # The fixture builds with n_synthetic=8 (see make_test_h5).
        assert "8 samples each" in text


class TestPackageExports:
    def test_imports_from_top_level(self):
        from syng_bts import (
            Subset,
            TCGADataset,
            clear_tcga_cache,
            list_tcga_datasets,
            load_tcga_dataset,
            tcga_cache_dir,
        )

        # Smoke-check identity to confirm they're the actual implementations
        assert callable(load_tcga_dataset)
        assert callable(list_tcga_datasets)
        assert callable(clear_tcga_cache)
        assert callable(tcga_cache_dir)
        assert isinstance(TCGADataset, type)
        assert isinstance(Subset, type)

    def test_in_dunder_all(self):
        import syng_bts

        for name in (
            "load_tcga_dataset",
            "list_tcga_datasets",
            "clear_tcga_cache",
            "tcga_cache_dir",
            "TCGADataset",
            "Subset",
        ):
            assert name in syng_bts.__all__, f"{name!r} missing from __all__"


@pytest.mark.real_data
class TestRealDataAllTcgaDatasets:
    """Full validation against the live published GitHub Release.

    Run with:
        pytest tests/test_tcga.py -m real_data

    Skipped while ``_DEFAULT_MANIFEST_URL`` is a placeholder.
    """

    def _maybe_skip(self):
        if tcga._DEFAULT_MANIFEST_URL.startswith("TBD-"):
            pytest.skip(
                "Default manifest URL not yet configured; "
                "see Phase 3 in the plan."
            )

    def test_manifest_lists_24_datasets(self, cache_root):
        self._maybe_skip()
        names = list_tcga_datasets()
        assert len(names) == 24
        aliases = list_tcga_datasets(short=True)
        assert len(aliases) == 24
        assert len(set(aliases)) == 24, "Cancer-type aliases must be unique"

    def test_load_every_dataset(self, cache_root):
        """Load all 24 datasets and verify every loaded TCGADataset is
        consistent with its corresponding manifest entry."""
        self._maybe_skip()

        # Fetch the manifest once via the loader so we can cross-check.
        manifest = tcga._fetch_manifest(None)
        entries_by_name = {e["dataset_name"]: e for e in manifest["datasets"]}
        names = list_tcga_datasets()
        assert set(names) == set(entries_by_name.keys())

        for name in names:
            entry = entries_by_name[name]
            ds = load_tcga_dataset(name)

            # 1. Schema invariants
            assert ds.name == name
            assert ds.n_raw_features == 1881
            assert ds.schema_version == "1.0"

            # 2. Root attrs match the manifest entry
            assert ds.cancer_type == entry["cancer_type"], name
            assert ds.clinical_variable == entry["clinical_variable"], name
            assert sorted(ds.group_labels) == sorted(entry["group_labels"]), name
            assert ds.n_raw_samples == entry["n_raw_samples"], name
            assert ds.n_filtered_samples == entry["n_filtered_samples"], name
            assert ds.n_filtered_features == entry["n_filtered_features"], name

            # 3. Cached HDF5 sha256 matches the manifest entry
            cached = tcga.tcga_cache_dir() / "1.0" / entry["file"]
            assert cached.exists(), name
            assert tcga._sha256_of_file(cached) == entry["sha256"], name

            # 4. All 13 splits present
            assert isinstance(ds.raw, tcga.Subset)
            for norm in ("raw_norm", "TC", "DESeq"):
                assert norm in ds.processed, name
                assert norm in ds.synthetic, name
                for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
                    assert model in ds.synthetic[norm], name

            # 5. Index alignment in every Subset
            assert ds.raw.expression.index.equals(ds.raw.groups.index), name
            for norm in ("raw_norm", "TC", "DESeq"):
                assert (
                    ds.processed[norm]
                    .expression.index.equals(ds.processed[norm].groups.index)
                ), name
                for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
                    sub = ds.synthetic[norm][model]
                    assert sub.expression.index.equals(sub.groups.index), name

            # 6. Expression shapes consistent with manifest counts
            assert ds.raw.expression.shape == (
                entry["n_raw_samples"],
                entry["n_raw_features"],
            ), name
            for norm in ("raw_norm", "TC", "DESeq"):
                assert ds.processed[norm].expression.shape == (
                    entry["n_filtered_samples"],
                    entry["n_filtered_features"],
                ), name

    def test_cache_hit_after_load(self, cache_root):
        self._maybe_skip()
        # Load BRCA once to populate the cache
        load_tcga_dataset("BRCA")

        # Wrap urlopen and assert no further call on the second load
        import urllib.request as _ur

        original = _ur.urlopen
        calls: list[str] = []

        def counting_urlopen(url, timeout=None):
            url_str = url if isinstance(url, str) else url.full_url
            calls.append(url_str)
            return original(url, timeout=timeout)

        try:
            _ur.urlopen = counting_urlopen  # type: ignore[assignment]
            load_tcga_dataset("BRCA")
        finally:
            _ur.urlopen = original  # type: ignore[assignment]

        assert calls == [], f"Expected zero network calls, got {calls}"

    def test_pipeline_integration_with_smallest_dataset(self, cache_root):
        self._maybe_skip()
        from syng_bts import generate

        ds = load_tcga_dataset("UCS")
        real_df, real_groups = ds.real("TC")

        # CPU-friendly model and tiny epoch budget
        result = generate(
            data=real_df,
            groups=real_groups,
            model="VAE1-10",
            new_size=10,
            batch_frac=0.1,
            learning_rate=0.0005,
            epoch=2,
            random_seed=42,
        )

        assert result is not None
        assert result.generated_data is not None
        assert len(result.generated_data) == 10
        assert list(result.generated_data.columns) == list(real_df.columns)


@pytest.mark.slow
class TestSlowIntegration:
    """Hits the live published manifest. Skipped during Phase 1.

    After Phase 3 (manifest URL wired in), run with:
        pytest tests/test_tcga.py -m slow
    """

    def test_live_manifest_and_smallest_dataset(self, cache_root):
        if tcga._DEFAULT_MANIFEST_URL.startswith("TBD-"):
            pytest.skip(
                "Default manifest URL not yet configured; "
                "see Phase 3 in the plan."
            )

        # Use the smallest dataset (UCS, ~14 MB) for the live round-trip.
        ds = load_tcga_dataset("UCS")

        assert isinstance(ds, tcga.TCGADataset)
        assert ds.cancer_type == "UCS"
        # Schema invariants
        assert ds.n_raw_features == 1881
        assert ds.schema_version == "1.0"
        assert ds.raw.expression.shape[0] == ds.n_raw_samples
        assert ds.raw.expression.shape[1] == 1881
        # Processed / synthetic shapes are consistent
        for norm in ("raw_norm", "TC", "DESeq"):
            assert (
                ds.processed[norm].expression.shape
                == (ds.n_filtered_samples, ds.n_filtered_features)
            )
            for model in ("CVAE1_5", "CVAE1_10", "CVAE1_20"):
                assert (
                    ds.synthetic[norm][model].expression.shape[1]
                    == ds.n_filtered_features
                )
