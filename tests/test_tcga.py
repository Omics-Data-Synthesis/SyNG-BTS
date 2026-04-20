"""Tests for syng_bts.tcga: TCGA dataset loader."""

from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401  # used in later tasks (marks)

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
