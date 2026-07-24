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
        # Should have 13 datasets total
        assert len(datasets) == 13

    @pytest.mark.parametrize(
        "dataset_name,has_groups",
        [
            # Datasets without a groups sidecar → groups is None
            ("SKCMPositive_4", False),
            ("BRCA", False),
            ("PRAD", False),
            # Dataset with a groups sidecar → groups is a Series
            ("BRCASubtypeSel", True),
        ],
        ids=["SKCMPositive_4", "BRCA", "PRAD", "BRCASubtypeSel"],
    )
    def test_resolve_bundled_dataset(self, dataset_name, has_groups):
        """Bundled datasets load via resolve_data, with groups present iff
        the dataset ships a groups sidecar."""
        from syng_bts import resolve_data

        data, groups = resolve_data(dataset_name)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data.columns) > 0

        if has_groups:
            assert groups is not None
            assert isinstance(groups, pd.Series)
            assert len(groups) == len(data)
        else:
            assert groups is None

    def test_resolve_generated_dataset(self):
        """The CVAE-generated BRCA training set has a specific shape and the two
        expected subtype labels in its groups sidecar."""
        from syng_bts import resolve_data

        data, groups = resolve_data("BRCASubtypeSel_train_epoch285_CVAE1-20_generated")

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert len(data.columns) == 47
        assert groups is not None
        assert isinstance(groups, pd.Series)
        assert len(groups) == len(data)
        assert set(groups.unique()) == {
            "Infiltrating Lobular Carcinoma",
            "Infiltrating Ductal Carcinoma",
        }

    def test_resolve_all_bundled_datasets(self):
        """Test that all bundled datasets can be loaded via resolve_data."""
        from syng_bts import list_bundled_datasets, resolve_data

        datasets = list_bundled_datasets()

        for name in datasets:
            data, groups = resolve_data(name)
            assert isinstance(data, pd.DataFrame), f"Failed to load {name}"
            assert len(data) > 0, f"Dataset {name} is empty"
            # Groups should be Series or None
            assert groups is None or isinstance(groups, pd.Series), (
                f"Unexpected groups type for {name}: {type(groups)}"
            )

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

        data, groups = resolve_data(str(sample_csv_file))

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20  # Sample data has 20 rows
        assert groups is None  # User files never return groups

    def test_resolve_data_from_path_object(self, sample_csv_file):
        """Test resolve_data accepts Path objects."""
        from syng_bts import resolve_data

        data, groups = resolve_data(sample_csv_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20
        assert groups is None

    def test_resolve_data_dataframe_passthrough(self, sample_data):
        """Test resolve_data passes DataFrame through unchanged."""
        from syng_bts import resolve_data

        result, groups = resolve_data(sample_data)

        assert result is sample_data
        assert groups is None

    def test_resolve_data_fallback_to_bundled(self):
        """Test that resolve_data falls back to bundled data."""
        from syng_bts import resolve_data

        data, _groups = resolve_data("SKCMPositive_4")

        assert isinstance(data, pd.DataFrame)


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
        data, _groups = resolve_data("SKCMPositive_4.csv")
        assert isinstance(data, pd.DataFrame)

    def test_resolve_data_path_types(self, temp_dir, sample_data):
        """Test resolve_data accepts both str and Path paths."""
        from syng_bts import resolve_data

        # Save sample data
        csv_path = temp_dir / "path_test.csv"
        sample_data.to_csv(csv_path, index=False)

        # Test with string path
        data1, _g1 = resolve_data(str(csv_path))
        assert len(data1) == 20

        # Test with Path object
        data2, _g2 = resolve_data(csv_path)
        assert len(data2) == 20


# ---------------------------------------------------------------------------
# resolve_data() — additional edge cases (merged from test_resolve_data.py)
# ---------------------------------------------------------------------------
class TestResolveDataEdgeCases:
    """Additional resolve_data() tests for error messages and type handling."""

    def test_case_insensitive_csv_extension(self):
        """Test that .CSV extension is also stripped."""
        from syng_bts import resolve_data

        df, _groups = resolve_data("SKCMPositive_4.CSV")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.parametrize(
        "input_val,exc_type,match",
        [
            ("totally_nonexistent_dataset", ValueError, "Unknown dataset name"),
            (12345, TypeError, "pd.DataFrame, str, or Path"),
        ],
        ids=["unknown_name", "invalid_type"],
    )
    def test_raises_on_bad_input(self, input_val, exc_type, match):
        """Bad inputs raise the expected exception with a descriptive message."""
        from syng_bts import resolve_data

        with pytest.raises(exc_type, match=match):
            resolve_data(input_val)

    def test_nonexistent_path_object_raises(self, temp_dir):
        """Non-existent Path raises FileNotFoundError."""
        from syng_bts import resolve_data

        with pytest.raises(FileNotFoundError):
            resolve_data(temp_dir / "no_such_file.csv")

    def test_unsupported_file_type_raises(self, temp_dir):
        """Unsupported file extensions raise ValueError."""
        from syng_bts.data_utils import _read_user_file

        txt_file = temp_dir / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")

        with pytest.raises(ValueError, match="Unsupported file type"):
            _read_user_file(txt_file)

    def test_path_with_directory_separator(self, sample_csv_file):
        """Paths with separators are treated as file paths."""
        from syng_bts import resolve_data

        df, groups = resolve_data(str(sample_csv_file))
        assert len(df) == 20
        assert groups is None


# ---------------------------------------------------------------------------
# _derive_dataname()
# ---------------------------------------------------------------------------
class TestDeriveDataname:
    """Test the _derive_dataname() helper."""

    def test_explicit_name_wins(self, sample_data):
        """Explicit name parameter takes priority."""
        from syng_bts.data_utils import _derive_dataname

        result = _derive_dataname(sample_data, name="override")
        assert result == "override"

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("/some/path/MyDataset.csv", "MyDataset"),
            ("SKCMPositive_4", "SKCMPositive_4"),
            ("BRCASubtypeSel_train", "BRCASubtypeSel_train"),
        ],
        ids=["file_path_strips_extension", "bundled_name", "name_without_extension"],
    )
    def test_from_string(self, input_val, expected):
        """Derive name from string inputs (file paths and plain names)."""
        from syng_bts.data_utils import _derive_dataname

        assert _derive_dataname(input_val) == expected

    def test_from_file_path_object(self):
        """Derive name from a Path object."""
        from syng_bts.data_utils import _derive_dataname

        result = _derive_dataname(Path("/some/path/MyDataset.csv"))
        assert result == "MyDataset"

    def test_from_dataframe_with_attrs(self):
        """Derive name from a DataFrame with .attrs['name']."""
        from syng_bts.data_utils import _derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        df.attrs["name"] = "my_dataset"
        result = _derive_dataname(df)
        assert result == "my_dataset"

    def test_from_dataframe_without_attrs(self):
        """DataFrame without attrs falls back to 'data'."""
        from syng_bts.data_utils import _derive_dataname

        df = pd.DataFrame({"a": [1, 2]})
        result = _derive_dataname(df)
        assert result == "data"

    def test_explicit_name_overrides_dataframe_attrs(self, sample_data):
        """Explicit name overrides DataFrame attrs."""
        from syng_bts.data_utils import _derive_dataname

        sample_data.attrs["name"] = "from_attrs"
        result = _derive_dataname(sample_data, name="explicit")
        assert result == "explicit"


# ---------------------------------------------------------------------------
# tuple-return contract and strict validation
# ---------------------------------------------------------------------------
class TestResolveDataTupleReturn:
    """Verify resolve_data() returns (DataFrame, Series | None) tuples."""

    def test_bundled_no_groups_returns_none(self):
        """Bundled datasets without groups return (df, None)."""
        from syng_bts import resolve_data

        df, groups = resolve_data("SKCMPositive_4")
        assert isinstance(df, pd.DataFrame)
        assert groups is None

    def test_bundled_with_groups_returns_series(self):
        """Bundled datasets with groups return (df, Series)."""
        from syng_bts import resolve_data

        df, groups = resolve_data("BRCASubtypeSel")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(groups, pd.Series)
        assert groups.name == "groups"
        assert len(groups) == len(df)
        # groups should have exactly 2 unique values (binary)
        assert len(groups.unique()) == 2

    def test_bundled_lihc_with_groups(self):
        """LIHC grouped dataset returns groups correctly."""
        from syng_bts import resolve_data

        df, groups = resolve_data("LIHCSubtypeFamInd")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(groups, pd.Series)
        assert len(groups) == len(df)
        assert set(groups.unique()) == {"YES", "NO"}

    def test_grouped_datasets_feature_only(self):
        """Grouped datasets have no 'groups' or 'samples' columns in df."""
        from syng_bts import resolve_data

        for name in ["BRCASubtypeSel", "LIHCSubtypeFamInd"]:
            df, groups = resolve_data(name)
            assert "groups" not in df.columns, f"{name} has groups col in df"
            assert "samples" not in df.columns, f"{name} has samples col in df"
            assert groups is not None, f"{name} should have groups"

    def test_dataframe_input_returns_none_groups(self, sample_data):
        """DataFrame pass-through returns (df, None)."""
        from syng_bts import resolve_data

        df, groups = resolve_data(sample_data)
        assert df is sample_data
        assert groups is None

    def test_csv_file_returns_none_groups(self, sample_csv_file):
        """CSV file input returns (df, None)."""
        from syng_bts import resolve_data

        df, groups = resolve_data(str(sample_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert groups is None

    def test_parquet_file_returns_none_groups(self, temp_dir, sample_data):
        """Parquet file input returns (df, None)."""
        from syng_bts import resolve_data

        pq_path = temp_dir / "test_data.parquet"
        sample_data.to_parquet(pq_path, engine="pyarrow")

        df, groups = resolve_data(str(pq_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert groups is None

    def test_parquet_suffix_stripped_for_bundled_lookup(self):
        """resolve_data('SKCMPositive_4.parquet') resolves bundled."""
        from syng_bts import resolve_data

        df, groups = resolve_data("SKCMPositive_4.parquet")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestValidateFeatureData:
    """Tests for the strict data-contract validator."""

    def test_valid_data_passes(self, sample_data):
        """Valid numeric-only DataFrame passes validation."""
        from syng_bts.data_utils import _validate_feature_data

        _validate_feature_data(sample_data)  # should not raise

    @pytest.mark.parametrize(
        "col_name,col_value,match",
        [
            ("groups", 0, "metadata column"),
            ("samples", "TCGA-XX", "metadata column"),
            ("Groups", 1, "metadata column"),
            ("Samples", "X", "metadata column"),
        ],
        ids=["groups", "samples", "Groups_caps", "Samples_caps"],
    )
    def test_rejects_metadata_columns(self, sample_data, col_name, col_value, match):
        """DataFrame with metadata-like columns is rejected."""
        from syng_bts.data_utils import _validate_feature_data

        bad = sample_data.copy()
        bad[col_name] = col_value
        with pytest.raises(ValueError, match=match):
            _validate_feature_data(bad)

    def test_rejects_non_numeric_columns(self):
        """DataFrame with string columns is rejected."""
        from syng_bts.data_utils import _validate_feature_data

        bad = pd.DataFrame({"gene_1": [1.0, 2.0], "label": ["A", "B"]})
        with pytest.raises(ValueError, match="non-numeric column"):
            _validate_feature_data(bad)

    def test_rejects_duplicate_index(self, sample_data):
        """DataFrame with duplicate index values is rejected."""
        from syng_bts.data_utils import _validate_feature_data

        bad = sample_data.copy()
        bad.index = [0] * len(bad)
        with pytest.raises(ValueError, match="duplicate"):
            _validate_feature_data(bad)
