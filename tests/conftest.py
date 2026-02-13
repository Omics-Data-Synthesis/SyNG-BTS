"""
Pytest configuration and fixtures for SyNG-BTS tests.

This module provides shared test fixtures for all test modules including:
- temp_dir: Temporary directory with automatic cleanup
- sample_data: Small transcriptomics-like DataFrame (20x50)
- sample_data_with_labels: Sample data with binary class labels
- sample_csv_file: Temporary CSV file for testing I/O
- small_training_config: Minimal training parameters for fast tests

Usage:
    pytest tests/ -v              # Run all tests
    pytest tests/ -m slow         # Run only slow tests
    pytest tests/ -m "not slow"   # Skip slow tests
    pytest tests/ --cov=syng_bts  # Run with coverage
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


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


@pytest.fixture
def sample_result(sample_data):
    """Create a minimal SyngResult for downstream testing.

    Uses sample_data as the generated DataFrame with small noise added,
    and a synthetic loss DataFrame. Useful for tests that need a
    pre-built SyngResult without running actual training.
    """
    from syng_bts import SyngResult

    rng = np.random.RandomState(42)
    gen = sample_data.copy()
    gen = gen + rng.normal(0, 0.1, size=gen.shape)

    loss = pd.DataFrame(
        {
            "kl": rng.rand(50) * 10,
            "recons": rng.rand(50) * 5,
        }
    )

    return SyngResult(
        generated_data=gen,
        loss=loss,
        metadata={"model": "VAE1-10", "dataname": "test", "seed": 42},
    )


@pytest.fixture
def sample_result_with_model(sample_data):
    """Create a SyngResult with all optional fields populated."""
    from syng_bts import SyngResult

    rng = np.random.RandomState(123)
    gen = sample_data.copy()
    recon = sample_data.copy()

    loss = pd.DataFrame(
        {
            "kl": rng.rand(50),
            "recons": rng.rand(50),
        }
    )

    model_state = {"weight": torch.randn(10, 5)}

    return SyngResult(
        generated_data=gen,
        loss=loss,
        reconstructed_data=recon,
        model_state=model_state,
        metadata={
            "model": "VAE1-10",
            "dataname": "test_full",
            "seed": 123,
            "epochs_trained": 100,
        },
    )
