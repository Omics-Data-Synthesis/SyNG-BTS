"""
Pytest configuration and fixtures for SyNG-BTS tests.
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


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
        np.random.rand(20, 50) * 10,
        columns=[f"gene_{i}" for i in range(50)],
    )
    return data


@pytest.fixture
def sample_data_with_labels(sample_data):
    """Create sample data with class labels for conditional models."""
    data = sample_data.copy()
    # Add a label column (binary classification)
    data.insert(0, "label", np.random.choice([0, 1], size=len(data)))
    return data


@pytest.fixture
def sample_csv_file(temp_dir, sample_data):
    """Create a sample CSV file for testing data loading."""
    csv_path = temp_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def small_training_config():
    """Return a minimal training configuration for fast testing."""
    return {
        "pilot_size": [10],
        "model": "VAE1-10",
        "batch_frac": 0.5,
        "learning_rate": 0.001,
        "epoch": 2,  # Very few epochs for fast testing
        "early_stop_num": 5,
        "AE_head_num": 1,
        "Gaussian_head_num": 2,
    }
