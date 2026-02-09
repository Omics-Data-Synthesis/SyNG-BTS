"""
Tests for helper utility functions.

These tests cover the internal helper functions used by the
main experiment functions for data preprocessing, augmentation,
and visualization utilities.
"""

import numpy as np
import pandas as pd
import pytest
import torch


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_preprocessinglog2(self):
        """Test log2 preprocessing transforms tensor data correctly."""
        from syng_bts.helper_utils import preprocessinglog2

        # Function expects tensor input
        data = torch.randn(20, 50).abs() * 10  # Positive values
        result = preprocessinglog2(data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape
        # Log2(x+1) should be different from x
        assert not torch.allclose(result, data)

    def test_preprocessinglog2_handles_zeros(self):
        """Test log2 preprocessing handles zero values."""
        from syng_bts.helper_utils import preprocessinglog2

        # Create data with zeros
        data = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]])
        result = preprocessinglog2(data)

        assert isinstance(result, torch.Tensor)
        assert not torch.isinf(result).any()  # No inf values
        # Check log2(0+1) = 0
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0


class TestSeedSetting:
    """Test random seed utilities."""

    def test_set_all_seeds(self):
        """Test setting all random seeds produces reproducible results."""
        from syng_bts.helper_utils import set_all_seeds

        set_all_seeds(42)
        rand1 = torch.randn(5)
        np_rand1 = np.random.rand(5)

        set_all_seeds(42)
        rand2 = torch.randn(5)
        np_rand2 = np.random.rand(5)

        assert torch.allclose(rand1, rand2)
        assert np.allclose(np_rand1, np_rand2)

    def test_set_all_seeds_different_seeds(self):
        """Test different seeds produce different results."""
        from syng_bts.helper_utils import set_all_seeds

        set_all_seeds(42)
        rand1 = torch.randn(5)

        set_all_seeds(123)
        rand2 = torch.randn(5)

        assert not torch.allclose(rand1, rand2)


class TestLabelCreation:
    """Test label creation functions."""

    def test_create_labels_returns_tuple(self):
        """Test create_labels returns (labels, blurlabels) tuple."""
        from syng_bts.helper_utils import create_labels

        n_samples = 100
        result = create_labels(n_samples)

        assert isinstance(result, tuple)
        assert len(result) == 2
        labels, blurlabels = result
        assert labels.shape == (n_samples, 1)
        assert blurlabels.shape == (n_samples, 1)

    def test_create_labels_with_groups(self):
        """Test label creation with provided groups."""
        from syng_bts.helper_utils import create_labels

        groups = pd.Series(["A", "A", "B", "B", "B"])
        labels, blurlabels = create_labels(len(groups), groups=groups)

        assert labels.shape == (5, 1)
        assert blurlabels.shape == (5, 1)
        # Check binary labels
        unique = torch.unique(labels)
        assert len(unique) == 2

    def test_create_labels_mul_returns_tuple(self):
        """Test multiclass label creation returns tuple."""
        from syng_bts.helper_utils import create_labels_mul

        groups = pd.Series(["A", "A", "B", "B", "C", "C"])
        result = create_labels_mul(len(groups), groups=groups)

        assert isinstance(result, tuple)
        assert len(result) == 2
        labels, blurlabels = result
        assert labels.shape == (6, 1)


class TestDataAugmentation:
    """Test data augmentation functions."""

    def test_gaussian_aug_basic(self):
        """Test Gaussian augmentation creates more samples."""
        from syng_bts.helper_utils import Gaussian_aug

        # Create simple data
        rawdata = torch.randn(10, 50)
        rawlabels = torch.zeros(10, 1)
        multiplier = [5]  # Must be a list

        aug_data, aug_labels = Gaussian_aug(rawdata, rawlabels, multiplier)

        # Should have more samples: original + multiplier * original
        expected_size = 10 + 5 * 10  # 60
        assert len(aug_data) == expected_size
        assert len(aug_labels) == len(aug_data)

    def test_gaussian_aug_preserves_shape(self):
        """Test Gaussian augmentation preserves feature dimensions."""
        from syng_bts.helper_utils import Gaussian_aug

        rawdata = torch.randn(20, 100)
        rawlabels = torch.ones(20, 1)
        multiplier = [3]

        aug_data, aug_labels = Gaussian_aug(rawdata, rawlabels, multiplier)

        assert aug_data.shape[1] == rawdata.shape[1]  # Same features


class TestPilotDrawing:
    """Test pilot data sampling functions."""

    def test_draw_pilot_single_label(self):
        """Test draw_pilot with single label returns correct size."""
        from syng_bts.helper_utils import draw_pilot

        # Create dataset with only one label (all zeros)
        dataset = torch.randn(100, 50)
        labels = torch.zeros(100, 1)  # Single unique label
        blurlabels = labels.clone()

        pilot_data, pilot_labels, pilot_blur = draw_pilot(
            dataset, labels, blurlabels, n_pilot=20, seednum=42
        )

        # With single label, should get exactly n_pilot samples
        assert len(pilot_data) == 20
        assert len(pilot_labels) == 20

    def test_draw_pilot_two_labels(self):
        """Test draw_pilot with two labels draws from both groups."""
        from syng_bts.helper_utils import draw_pilot

        # Create dataset with two labels (balanced)
        dataset = torch.randn(100, 50)
        labels = torch.cat([torch.zeros(50, 1), torch.ones(50, 1)])
        blurlabels = labels.clone()

        pilot_data, pilot_labels, pilot_blur = draw_pilot(
            dataset, labels, blurlabels, n_pilot=20, seednum=42
        )

        # With two labels, n_pilot is per group, so total = 2 * n_pilot
        assert len(pilot_data) == 40  # 20 per group

    def test_draw_pilot_reproducible(self):
        """Test draw_pilot is reproducible with same seed."""
        from syng_bts.helper_utils import draw_pilot

        dataset = torch.randn(100, 50)
        labels = torch.zeros(100, 1)  # Single label for simpler test
        blurlabels = labels.clone()

        pilot1, _, _ = draw_pilot(dataset, labels, blurlabels, n_pilot=20, seednum=42)
        pilot2, _, _ = draw_pilot(dataset, labels, blurlabels, n_pilot=20, seednum=42)

        assert torch.allclose(pilot1, pilot2)

    def test_draw_pilot_different_seeds(self):
        """Test draw_pilot gives different results with different seeds."""
        from syng_bts.helper_utils import draw_pilot

        dataset = torch.randn(100, 50)
        labels = torch.zeros(100, 1)
        blurlabels = labels.clone()

        pilot1, _, _ = draw_pilot(dataset, labels, blurlabels, n_pilot=20, seednum=42)
        pilot2, _, _ = draw_pilot(dataset, labels, blurlabels, n_pilot=20, seednum=99)

        assert not torch.allclose(pilot1, pilot2)
