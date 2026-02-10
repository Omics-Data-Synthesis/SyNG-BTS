"""
Tests for resolve_data() and derive_dataname() functions.
"""

from pathlib import Path

import pandas as pd
import pytest


class TestResolveData:
    """Test the resolve_data() unified data input function."""

    def test_dataframe_passthrough(self, sample_data):
        """Test that a DataFrame is returned as-is."""
        from syng_bts import resolve_data

        result = resolve_data(sample_data)
        assert result is sample_data

    def test_bundled_dataset_by_name(self):
        """Test loading a bundled dataset by plain name."""
        from syng_bts import resolve_data

        df = resolve_data("SKCMPositive_4")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_bundled_dataset_strips_csv_extension(self):
        """Test that .csv extension is stripped for bundled lookup."""
        from syng_bts import resolve_data

        df = resolve_data("SKCMPositive_4.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_bundled_dataset_case_sensitive_extension(self):
        """Test that .CSV extension is also stripped."""
        from syng_bts import resolve_data

        df = resolve_data("SKCMPositive_4.CSV")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_file_path_string(self, sample_csv_file):
        """Test loading from a string file path."""
        from syng_bts import resolve_data

        df = resolve_data(str(sample_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_file_path_object(self, sample_csv_file):
        """Test loading from a Path object."""
        from syng_bts import resolve_data

        df = resolve_data(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_unknown_name_raises_valueerror(self):
        """Test that an unknown dataset name raises ValueError."""
        from syng_bts import resolve_data

        with pytest.raises(ValueError, match="Unknown dataset name"):
            resolve_data("totally_nonexistent_dataset")

    def test_unknown_name_lists_available(self):
        """Test that the error message lists available datasets."""
        from syng_bts import resolve_data

        with pytest.raises(ValueError, match="SKCMPositive_4"):
            resolve_data("nonexistent")

    def test_nonexistent_file_path_raises(self, temp_dir):
        """Test that a non-existent file path raises FileNotFoundError."""
        from syng_bts import resolve_data

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_data(str(temp_dir / "no_such_file.csv"))

    def test_nonexistent_path_object_raises(self, temp_dir):
        """Test that a non-existent Path raises FileNotFoundError."""
        from syng_bts import resolve_data

        with pytest.raises(FileNotFoundError):
            resolve_data(temp_dir / "no_such_file.csv")

    def test_invalid_type_raises_typeerror(self):
        """Test that an invalid type raises TypeError."""
        from syng_bts import resolve_data

        with pytest.raises(TypeError, match="pd.DataFrame, str, or Path"):
            resolve_data(12345)

    def test_all_bundled_datasets(self):
        """Test that resolve_data can load every bundled dataset."""
        from syng_bts import list_bundled_datasets, resolve_data

        for name in list_bundled_datasets():
            df = resolve_data(name)
            assert isinstance(df, pd.DataFrame), f"Failed for {name}"
            assert len(df) > 0, f"Empty result for {name}"

    def test_path_with_directory_separator(self, sample_csv_file):
        """Test that paths with separators are treated as file paths."""
        from syng_bts import resolve_data

        # Ensure a path with "/" is treated as a file path
        df = resolve_data(str(sample_csv_file))
        assert len(df) == 20


class TestDeriveDataname:
    """Test the derive_dataname() helper."""

    def test_explicit_name_wins(self, sample_data):
        """Test that an explicit name parameter takes priority."""
        from syng_bts import derive_dataname

        result = derive_dataname(sample_data, name="override")
        assert result == "override"

    def test_from_file_path_string(self):
        """Test deriving name from a string file path."""
        from syng_bts import derive_dataname

        result = derive_dataname("/some/path/MyDataset.csv")
        assert result == "MyDataset"

    def test_from_file_path_object(self):
        """Test deriving name from a Path object."""
        from syng_bts import derive_dataname

        result = derive_dataname(Path("/some/path/MyDataset.csv"))
        assert result == "MyDataset"

    def test_from_bundled_name(self):
        """Test deriving name from a plain bundled name string."""
        from syng_bts import derive_dataname

        result = derive_dataname("SKCMPositive_4")
        assert result == "SKCMPositive_4"

    def test_from_dataframe_with_attrs(self):
        """Test deriving name from a DataFrame with .attrs['name']."""
        from syng_bts import derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        df.attrs["name"] = "my_dataset"
        result = derive_dataname(df)
        assert result == "my_dataset"

    def test_from_dataframe_without_attrs(self):
        """Test deriving name from a DataFrame without attrs falls back."""
        from syng_bts import derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        result = derive_dataname(df)
        assert result == "data"

    def test_explicit_name_with_dataframe(self, sample_data):
        """Test explicit name overrides DataFrame attrs."""
        from syng_bts import derive_dataname

        sample_data.attrs["name"] = "from_attrs"
        result = derive_dataname(sample_data, name="explicit")
        assert result == "explicit"

    def test_name_from_string_without_extension(self):
        """Test deriving from a plain string (no extension)."""
        from syng_bts import derive_dataname

        result = derive_dataname("BRCASubtypeSel_train")
        assert result == "BRCASubtypeSel_train"
