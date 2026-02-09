"""
Tests for data loading and path utility functions.
"""

import pytest
from pathlib import Path
import pandas as pd


class TestBundledDatasets:
    """Test bundled dataset loading functionality."""

    def test_list_bundled_datasets(self):
        """Test listing available bundled datasets."""
        from syng_bts import list_bundled_datasets

        datasets = list_bundled_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) > 0
        # Check expected datasets are present
        assert "SKCMPositive_4" in datasets
        assert "BRCA" in datasets
        assert "PRAD" in datasets

    def test_list_bundled_datasets_count(self):
        """Test correct number of bundled datasets."""
        from syng_bts import list_bundled_datasets

        datasets = list_bundled_datasets()
        # Should have 12 datasets total
        assert len(datasets) == 12

    def test_load_bundled_dataset_skcm(self):
        """Test loading SKCMPositive_4 bundled dataset."""
        from syng_bts import load_dataset

        data = load_dataset("SKCMPositive_4")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data.columns) > 0

    def test_load_bundled_dataset_brca(self):
        """Test loading BRCA bundled dataset."""
        from syng_bts import load_dataset

        data = load_dataset("BRCA")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_bundled_dataset_prad(self):
        """Test loading PRAD bundled dataset."""
        from syng_bts import load_dataset

        data = load_dataset("PRAD")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_bundled_dataset_brca_subtype(self):
        """Test loading BRCA subtype case study dataset."""
        from syng_bts import load_dataset

        data = load_dataset("BRCASubtypeSel")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_all_bundled_datasets(self):
        """Test that all bundled datasets can be loaded."""
        from syng_bts import list_bundled_datasets, load_dataset

        datasets = list_bundled_datasets()

        for name in datasets:
            data = load_dataset(name)
            assert isinstance(data, pd.DataFrame), f"Failed to load {name}"
            assert len(data) > 0, f"Dataset {name} is empty"

    def test_load_nonexistent_dataset_raises(self):
        """Test that loading a non-existent dataset raises an error."""
        from syng_bts import load_dataset

        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset_xyz")


class TestDataLoading:
    """Test data loading from files."""

    def test_load_dataset_from_path(self, sample_csv_file):
        """Test loading dataset from a custom path."""
        from syng_bts import load_dataset

        parent_dir = sample_csv_file.parent
        # Load using the data_path parameter
        data = load_dataset("test_data", data_path=parent_dir)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20  # Sample data has 20 rows

    def test_load_dataset_fallback_to_bundled(self):
        """Test that load_dataset falls back to bundled data."""
        from syng_bts import load_dataset

        # Should load bundled data when file doesn't exist locally
        data = load_dataset("SKCMPositive_4")

        assert isinstance(data, pd.DataFrame)


class TestOutputDirectory:
    """Test output directory management."""

    def test_get_output_dir_default(self):
        """Test default output directory is current working directory."""
        from syng_bts import get_output_dir
        from syng_bts.data_utils import set_default_output_dir

        # Reset to default
        set_default_output_dir(None)

        output_dir = get_output_dir()
        assert output_dir == Path.cwd()

    def test_set_default_output_dir(self, temp_dir):
        """Test setting default output directory."""
        from syng_bts import set_default_output_dir, get_output_dir

        set_default_output_dir(temp_dir)
        output_dir = get_output_dir()

        assert output_dir == temp_dir

        # Reset to default
        set_default_output_dir(None)

    def test_get_output_dir_with_explicit(self, temp_dir):
        """Test explicit output directory overrides default."""
        from syng_bts import set_default_output_dir, get_output_dir

        default_dir = temp_dir / "default"
        explicit_dir = temp_dir / "explicit"

        set_default_output_dir(default_dir)
        output_dir = get_output_dir(explicit_dir)

        assert output_dir == explicit_dir

        # Reset
        set_default_output_dir(None)


class TestDirectoryCreation:
    """Test output directory creation logic."""

    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test ensure_dir creates the directory."""
        from syng_bts.data_utils import ensure_dir

        new_dir = temp_dir / "new_subdir" / "nested"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_existing(self, temp_dir):
        """Test ensure_dir works with existing directory."""
        from syng_bts.data_utils import ensure_dir

        # Directory already exists
        result = ensure_dir(temp_dir)

        assert temp_dir.exists()
        assert result == temp_dir

    def test_get_output_path_creates_subdirectory(self, temp_dir):
        """Test get_output_path creates subdirectory."""
        from syng_bts.data_utils import get_output_path

        output_path = get_output_path(
            output_dir=temp_dir,
            subdir="GeneratedData",
            filename="test_output.csv",
            create_dir=True,
        )

        expected_dir = temp_dir / "GeneratedData"
        assert expected_dir.exists()
        assert output_path == expected_dir / "test_output.csv"

    def test_get_output_path_no_create(self, temp_dir):
        """Test get_output_path without creating directory."""
        from syng_bts.data_utils import get_output_path

        output_path = get_output_path(
            output_dir=temp_dir,
            subdir="NonExistent",
            filename="test.csv",
            create_dir=False,
        )

        expected_dir = temp_dir / "NonExistent"
        assert not expected_dir.exists()
        assert output_path == expected_dir / "test.csv"
