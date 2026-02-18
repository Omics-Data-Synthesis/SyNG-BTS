"""
Tests for package imports and public API.
"""

import pytest


class TestPackageImports:
    """Test that all public API can be imported."""

    def test_import_package(self):
        """Test basic package import."""
        import syng_bts

        assert syng_bts is not None

    def test_import_version(self):
        """Test version is accessible."""
        from syng_bts import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Version should follow semver pattern
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_import_author(self):
        """Test author metadata."""
        from syng_bts import __author__, __email__, __license__

        assert __author__ is not None
        assert __email__ is not None
        assert __license__ is not None

    def test_import_experiment_functions(self):
        """Test new experiment functions are importable."""
        from syng_bts import generate, pilot_study, transfer

        assert callable(generate)
        assert callable(pilot_study)
        assert callable(transfer)

    def test_import_evaluation_functions(self):
        """Test evaluation functions are importable."""
        from syng_bts import UMAP_eval, evaluation, heatmap_eval

        assert callable(heatmap_eval)
        assert callable(UMAP_eval)
        assert callable(evaluation)

    def test_import_data_utils(self):
        """Test data utility functions are importable."""
        from syng_bts import (
            list_bundled_datasets,
            resolve_data,
        )

        assert callable(list_bundled_datasets)
        assert callable(resolve_data)

    def test_import_model_classes(self):
        """Test model classes are importable."""
        from syng_bts import AE, CVAE, GAN, VAE

        assert AE is not None
        assert VAE is not None
        assert CVAE is not None
        assert GAN is not None

    def test_import_result_objects(self):
        """Test result objects are importable."""
        from syng_bts import PilotResult, SyngResult

        assert SyngResult is not None
        assert PilotResult is not None

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        import syng_bts

        expected = [
            "__version__",
            "generate",
            "pilot_study",
            "transfer",
            "heatmap_eval",
            "UMAP_eval",
            "evaluation",
            "AE",
            "VAE",
            "CVAE",
            "GAN",
            "list_bundled_datasets",
            "resolve_data",
            "SyngResult",
            "PilotResult",
        ]

        for name in expected:
            assert name in syng_bts.__all__, f"{name} not in __all__"

    def test_legacy_names_removed(self):
        """Verify that legacy names are no longer exported."""
        import syng_bts

        for name in [
            "PilotExperiment",
            "ApplyExperiment",
            "TransferExperiment",
            "Transfer",
            "load_dataset",
            "set_default_output_dir",
            "get_output_dir",
            "derive_dataname",
        ]:
            assert name not in syng_bts.__all__, f"{name} should not be in __all__"

    def test_legacy_import_raises(self):
        """Importing removed legacy names should fail."""
        with pytest.raises(ImportError):
            from syng_bts import PilotExperiment  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import ApplyExperiment  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import TransferExperiment  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import Transfer  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import load_dataset  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import set_default_output_dir  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import get_output_dir  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import derive_dataname  # noqa: F401


class TestDynamicVersion:
    """Test dynamic version reading from metadata."""

    def test_version_from_metadata(self):
        """Test version can be read from package metadata."""
        try:
            from importlib.metadata import version

            pkg_version = version("syng-bts")
            from syng_bts import __version__

            # Should match package metadata
            assert __version__ == pkg_version
        except Exception:
            # Package might not be installed, skip
            pytest.skip("Package not installed via pip")

    def test_version_fallback(self):
        """Test version fallback works."""
        from syng_bts import __version__

        # Should have a valid version string even if package not installed
        assert __version__ is not None
        assert len(__version__) > 0
