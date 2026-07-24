"""
Pytest configuration and fixtures for SyNG-BTS tests.

This module provides shared test fixtures for all test modules including:
- temp_dir: Temporary directory with automatic cleanup
- sample_data: Small transcriptomics-like DataFrame (20x50)
- sample_csv_file: Temporary CSV file for testing I/O
- small_training_config: Minimal training parameters for fast tests
- network_stub: Mock urllib for testing network operations
- cache_root: Temporary directory for TCGA cache with env override

Usage:
    pytest tests/ -v              # Run all tests
    pytest tests/ -m slow         # Run only slow tests
    pytest tests/ -m "not slow"   # Skip slow tests
    pytest tests/ --cov=syng_bts  # Run with coverage
"""

import io
import shutil
import tempfile
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    # Cleanup after test
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    # Create a small dataset similar to transcriptomics data
    # 20 samples, 50 features
    data = pd.DataFrame(
        np.random.randint(0, 11, size=(20, 50)).astype(float),
        columns=[f"gene_{i}" for i in range(50)],
    )
    return data


@pytest.fixture
def sample_csv_file(temp_dir, sample_data):
    """Create a sample CSV file for testing data loading."""
    csv_path = temp_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def small_training_config():
    """Return a minimal training configuration for fast testing.

    Configuration uses very few epochs and small batch sizes
    to allow integration tests to run quickly.
    """
    return {
        "pilot_size": [10],
        "model": "VAE1-10",
        "batch_frac": 0.5,
        "learning_rate": 0.001,
        "epoch": 2,  # Very few epochs for fast testing
        "early_stop_num": 5,
        "AE_head_num": 1,
        "Gaussian_head_num": 2,
        "random_seed": 42,
    }


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
