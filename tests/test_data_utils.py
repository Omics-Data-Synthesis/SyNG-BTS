"""
Tests for data loading and path utility functions.
"""

from pathlib import Path

import pandas as pd
import pytest


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

    def test_resolve_bundled_dataset_skcm(self):
        """Test loading SKCMPositive_4 bundled dataset via resolve_data."""
        from syng_bts import resolve_data

        data = resolve_data("SKCMPositive_4")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data.columns) > 0

    def test_resolve_bundled_dataset_brca(self):
        """Test loading BRCA bundled dataset."""
        from syng_bts import resolve_data

        data = resolve_data("BRCA")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_resolve_bundled_dataset_prad(self):
        """Test loading PRAD bundled dataset."""
        from syng_bts import resolve_data

        data = resolve_data("PRAD")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_resolve_bundled_dataset_brca_subtype(self):
        """Test loading BRCA subtype case study dataset."""
        from syng_bts import resolve_data

        data = resolve_data("BRCASubtypeSel")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_resolve_all_bundled_datasets(self):
        """Test that all bundled datasets can be loaded via resolve_data."""
        from syng_bts import list_bundled_datasets, resolve_data

        datasets = list_bundled_datasets()

        for name in datasets:
            data = resolve_data(name)
            assert isinstance(data, pd.DataFrame), f"Failed to load {name}"
            assert len(data) > 0, f"Dataset {name} is empty"

    def test_resolve_nonexistent_dataset_raises(self):
        """Test that resolving a non-existent dataset raises an error."""
        from syng_bts import resolve_data

        with pytest.raises((FileNotFoundError, ValueError)):
            resolve_data("nonexistent_dataset_xyz")


class TestDataLoading:
    """Test data loading from files."""

    def test_resolve_data_from_path(self, sample_csv_file):
        """Test loading dataset from a file path via resolve_data."""
        from syng_bts import resolve_data

        data = resolve_data(str(sample_csv_file))

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20  # Sample data has 20 rows

    def test_resolve_data_from_path_object(self, sample_csv_file):
        """Test resolve_data accepts Path objects."""
        from syng_bts import resolve_data

        data = resolve_data(sample_csv_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20

    def test_resolve_data_dataframe_passthrough(self, sample_data):
        """Test resolve_data passes DataFrame through unchanged."""
        from syng_bts import resolve_data

        result = resolve_data(sample_data)

        assert result is sample_data

    def test_resolve_data_fallback_to_bundled(self):
        """Test that resolve_data falls back to bundled data."""
        from syng_bts import resolve_data

        data = resolve_data("SKCMPositive_4")

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
        from syng_bts import get_output_dir, set_default_output_dir

        set_default_output_dir(temp_dir)
        output_dir = get_output_dir()

        assert output_dir == temp_dir

        # Reset to default
        set_default_output_dir(None)

    def test_get_output_dir_with_explicit(self, temp_dir):
        """Test explicit output directory overrides default."""
        from syng_bts import get_output_dir, set_default_output_dir

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


class TestDataLoadingEdgeCases:
    """Test edge cases and error handling in data loading."""

    def test_resolve_data_empty_name(self):
        """Test resolving with empty dataset name raises error."""
        from syng_bts import resolve_data

        with pytest.raises((FileNotFoundError, ValueError)):
            resolve_data("")

    def test_resolve_data_with_extension(self):
        """Test resolve_data handles name with .csv extension."""
        from syng_bts import resolve_data

        # resolve_data should strip .csv and find bundled dataset
        data = resolve_data("SKCMPositive_4.csv")
        assert isinstance(data, pd.DataFrame)

    def test_resolve_data_path_types(self, temp_dir, sample_data):
        """Test resolve_data accepts both str and Path paths."""
        from syng_bts import resolve_data

        # Save sample data
        csv_path = temp_dir / "path_test.csv"
        sample_data.to_csv(csv_path, index=False)

        # Test with string path
        data1 = resolve_data(str(csv_path))
        assert len(data1) == 20

        # Test with Path object
        data2 = resolve_data(csv_path)
        assert len(data2) == 20


class TestOutputPathEdgeCases:
    """Test edge cases in output path handling."""

    def test_ensure_dir_with_string_path(self, temp_dir):
        """Test ensure_dir works with string paths."""
        from syng_bts.data_utils import ensure_dir

        new_dir = str(temp_dir / "string_path_test")
        ensure_dir(new_dir)

        assert Path(new_dir).exists()

    def test_get_output_path_nested_subdirs(self, temp_dir):
        """Test get_output_path with deeply nested subdirectories."""
        from syng_bts.data_utils import get_output_path

        output_path = get_output_path(
            output_dir=temp_dir,
            subdir="level1/level2/level3",
            filename="deep.csv",
            create_dir=True,
        )

        expected_dir = temp_dir / "level1" / "level2" / "level3"
        assert expected_dir.exists()
        assert output_path == expected_dir / "deep.csv"

    def test_set_output_dir_with_string(self, temp_dir):
        """Test set_default_output_dir works with string paths."""
        from syng_bts import get_output_dir, set_default_output_dir

        set_default_output_dir(str(temp_dir))
        output_dir = get_output_dir()

        assert output_dir == temp_dir

        # Reset
        set_default_output_dir(None)


# ---------------------------------------------------------------------------
# resolve_data() — additional edge cases (merged from test_resolve_data.py)
# ---------------------------------------------------------------------------
class TestResolveDataEdgeCases:
    """Additional resolve_data() tests for error messages and type handling."""

    def test_case_insensitive_csv_extension(self):
        """Test that .CSV extension is also stripped."""
        from syng_bts import resolve_data

        df = resolve_data("SKCMPositive_4.CSV")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_unknown_name_raises_valueerror(self):
        """Unknown dataset name raises ValueError with descriptive message."""
        from syng_bts import resolve_data

        with pytest.raises(ValueError, match="Unknown dataset name"):
            resolve_data("totally_nonexistent_dataset")

    def test_nonexistent_path_object_raises(self, temp_dir):
        """Non-existent Path raises FileNotFoundError."""
        from syng_bts import resolve_data

        with pytest.raises(FileNotFoundError):
            resolve_data(temp_dir / "no_such_file.csv")

    def test_invalid_type_raises_typeerror(self):
        """Invalid type raises TypeError."""
        from syng_bts import resolve_data

        with pytest.raises(TypeError, match="pd.DataFrame, str, or Path"):
            resolve_data(12345)

    def test_path_with_directory_separator(self, sample_csv_file):
        """Paths with separators are treated as file paths."""
        from syng_bts import resolve_data

        df = resolve_data(str(sample_csv_file))
        assert len(df) == 20


# ---------------------------------------------------------------------------
# derive_dataname()
# ---------------------------------------------------------------------------
class TestDeriveDataname:
    """Test the derive_dataname() helper."""

    def test_explicit_name_wins(self, sample_data):
        """Explicit name parameter takes priority."""
        from syng_bts import derive_dataname

        result = derive_dataname(sample_data, name="override")
        assert result == "override"

    def test_from_file_path_string(self):
        """Derive name from a string file path."""
        from syng_bts import derive_dataname

        result = derive_dataname("/some/path/MyDataset.csv")
        assert result == "MyDataset"

    def test_from_file_path_object(self):
        """Derive name from a Path object."""
        from syng_bts import derive_dataname

        result = derive_dataname(Path("/some/path/MyDataset.csv"))
        assert result == "MyDataset"

    def test_from_bundled_name(self):
        """Derive name from a plain bundled name string."""
        from syng_bts import derive_dataname

        result = derive_dataname("SKCMPositive_4")
        assert result == "SKCMPositive_4"

    def test_from_dataframe_with_attrs(self):
        """Derive name from a DataFrame with .attrs['name']."""
        from syng_bts import derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        df.attrs["name"] = "my_dataset"
        result = derive_dataname(df)
        assert result == "my_dataset"

    def test_from_dataframe_without_attrs(self):
        """DataFrame without attrs falls back to 'data'."""
        from syng_bts import derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        result = derive_dataname(df)
        assert result == "data"

    def test_explicit_name_overrides_dataframe_attrs(self, sample_data):
        """Explicit name overrides DataFrame attrs."""
        from syng_bts import derive_dataname

        sample_data.attrs["name"] = "from_attrs"
        result = derive_dataname(sample_data, name="explicit")
        assert result == "explicit"

    def test_name_from_string_without_extension(self):
        """Plain string without extension used as-is."""
        from syng_bts import derive_dataname

        result = derive_dataname("BRCASubtypeSel_train")
        assert result == "BRCASubtypeSel_train"
