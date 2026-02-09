# SyNG-BTS Test Suite

This directory contains the pytest test suite for the SyNG-BTS package.

## Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Package marker
├── conftest.py            # Shared pytest fixtures
├── test_imports.py        # Package import and API tests
├── test_data_utils.py     # Data loading and path utility tests
├── test_helpers.py        # Helper function tests
├── test_evaluations.py    # Visualization function tests
└── test_experiments.py    # Experiment function and integration tests
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=syng_bts --cov-report=html

# Run specific test file
pytest tests/test_data_utils.py -v

# Run specific test class
pytest tests/test_data_utils.py::TestDataLoading -v

# Run specific test
pytest tests/test_data_utils.py::TestDataLoading::test_load_dataset_from_path -v
```

### Using Markers

```bash
# Skip slow integration tests
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m slow
```

### Debug Options

```bash
# Show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -v --tb=long
```

## Test Categories

### Import Tests (`test_imports.py`)
Tests that verify the public API is correctly exposed:
- Package import works
- Version string is valid
- All exported functions are callable
- `__all__` exports are complete

### Data Utils Tests (`test_data_utils.py`)
Tests for data loading and path management:
- Bundled dataset listing and loading
- Custom data path loading
- Output directory configuration
- Directory creation utilities

### Evaluation Tests (`test_evaluations.py`)
Tests for visualization functions:
- Heatmap generation
- UMAP embedding and plotting
- Evaluation pipeline

### Experiment Tests (`test_experiments.py`)
Tests for training pipeline functions:
- Integration tests with small data (marked as slow)

## Fixtures

Common fixtures available in all test modules (defined in `conftest.py`):

| Fixture | Description |
|---------|-------------|
| `temp_dir` | Temporary directory with automatic cleanup |
| `sample_data` | Small 20x50 DataFrame mimicking transcriptomics data |
| `sample_data_with_labels` | Sample data with binary class labels |
| `sample_csv_file` | Temporary CSV file for I/O testing |
| `small_training_config` | Minimal config for fast training tests |

## Adding New Tests

1. Create a new test file named `test_*.py`
2. Import pytest and required modules
3. Use fixtures from `conftest.py` as needed
4. Follow the naming convention: `test_*` for functions/methods

Example:
```python
import pytest

class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_basic(self, sample_data):
        """Test feature with basic input."""
        from syng_bts import new_feature
        
        result = new_feature(sample_data)
        assert result is not None
```