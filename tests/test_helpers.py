"""
Tests for helper utility functions and internal code quality.

Covers:
- Data preprocessing (log2 transform)
- Random seed utilities
- Label creation (binary and multiclass)
- Data augmentation (Gaussian)
- Pilot data sampling
- generate_samples() and reconstruct_samples() pure functions
- create_labels / create_labels_mul global RNG preservation
"""

import inspect
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from syng_bts.helper_models import AE, CVAE, GAN, VAE


# ---------------------------------------------------------------------------
# Fixtures for model and data
# ---------------------------------------------------------------------------
@pytest.fixture
def ae_model():
    """Create a small AE model for testing."""
    model = AE(num_features=50)
    model.eval()
    return model


@pytest.fixture
def vae_model():
    """Create a small VAE model for testing."""
    model = VAE(num_features=50)
    model.eval()
    return model


@pytest.fixture
def cvae_model():
    """Create a small CVAE model for testing (2 classes)."""
    model = CVAE(num_features=50, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def gan_model():
    """Create a small GAN model for testing."""
    model = GAN(num_features=50, latent_dim=32)
    model.eval()
    return model


@pytest.fixture
def sample_dataloader():
    """Create a DataLoader with small sample data."""
    data = torch.randn(20, 50)
    labels = torch.zeros(20, 1)
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)


@pytest.fixture
def sample_dataloader_two_class():
    """Create a DataLoader with two-class labels for CVAE."""
    data = torch.randn(20, 50)
    labels = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)])
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)


# ===========================================================================
# Preprocessing
# ===========================================================================
class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_preprocessinglog2(self):
        """Test log2 preprocessing transforms tensor data correctly."""
        from syng_bts.helper_utils import preprocessinglog2

        data = torch.randn(20, 50).abs() * 10
        result = preprocessinglog2(data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape
        assert not torch.allclose(result, data)

    def test_preprocessinglog2_handles_zeros(self):
        """Test log2 preprocessing handles zero values."""
        from syng_bts.helper_utils import preprocessinglog2

        data = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]])
        result = preprocessinglog2(data)

        assert isinstance(result, torch.Tensor)
        assert not torch.isinf(result).any()
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0


# ===========================================================================
# Seed setting
# ===========================================================================
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


# ===========================================================================
# Label creation
# ===========================================================================
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


# ===========================================================================
# Data augmentation
# ===========================================================================
class TestDataAugmentation:
    """Test data augmentation functions."""

    def test_gaussian_aug_basic(self):
        """Test Gaussian augmentation creates more samples."""
        from syng_bts.helper_utils import Gaussian_aug

        rawdata = torch.randn(10, 50)
        rawlabels = torch.zeros(10, 1)
        multiplier = [5]

        aug_data, aug_labels = Gaussian_aug(rawdata, rawlabels, multiplier)

        expected_size = 10 + 5 * 10
        assert len(aug_data) == expected_size
        assert len(aug_labels) == len(aug_data)

    def test_gaussian_aug_preserves_shape(self):
        """Test Gaussian augmentation preserves feature dimensions."""
        from syng_bts.helper_utils import Gaussian_aug

        rawdata = torch.randn(20, 100)
        rawlabels = torch.ones(20, 1)
        multiplier = [3]

        aug_data, aug_labels = Gaussian_aug(rawdata, rawlabels, multiplier)

        assert aug_data.shape[1] == rawdata.shape[1]


# ===========================================================================
# Pilot drawing
# ===========================================================================
class TestPilotDrawing:
    """Test pilot data sampling functions."""

    def test_draw_pilot_single_label(self):
        """Test draw_pilot with single label returns correct size."""
        from syng_bts.helper_utils import draw_pilot

        dataset = torch.randn(100, 50)
        labels = torch.zeros(100, 1)
        blurlabels = labels.clone()

        pilot_data, pilot_labels, pilot_blur = draw_pilot(
            dataset, labels, blurlabels, n_pilot=20, seednum=42
        )

        assert len(pilot_data) == 20
        assert len(pilot_labels) == 20

    def test_draw_pilot_two_labels(self):
        """Test draw_pilot with two labels draws from both groups."""
        from syng_bts.helper_utils import draw_pilot

        dataset = torch.randn(100, 50)
        labels = torch.cat([torch.zeros(50, 1), torch.ones(50, 1)])
        blurlabels = labels.clone()

        pilot_data, pilot_labels, pilot_blur = draw_pilot(
            dataset, labels, blurlabels, n_pilot=20, seednum=42
        )

        assert len(pilot_data) == 40  # 20 per group

    def test_draw_pilot_reproducible(self):
        """Test draw_pilot is reproducible with same seed."""
        from syng_bts.helper_utils import draw_pilot

        dataset = torch.randn(100, 50)
        labels = torch.zeros(100, 1)
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


# ===========================================================================
# generate_samples() — pure computation function
# ===========================================================================
class TestGenerateSamples:
    """Tests for the generate_samples() pure function."""

    def test_ae_returns_tensor(self, ae_model):
        """generate_samples returns a tensor for AE models."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=ae_model, modelname="AE", latent_size=64, num_images=10
        )
        assert isinstance(result, torch.Tensor)

    def test_ae_correct_shape(self, ae_model):
        """AE generates correct number of samples with correct features."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=ae_model, modelname="AE", latent_size=64, num_images=15
        )
        assert result.shape == (15, 50)

    def test_vae_returns_tensor(self, vae_model):
        """generate_samples returns a tensor for VAE models."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=vae_model, modelname="VAE", latent_size=32, num_images=10
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 50)

    def test_cvae_returns_tensor_with_label_col(self, cvae_model):
        """CVAE generates samples with an extra label column."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=cvae_model, modelname="CVAE", latent_size=32, num_images=10
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 51)  # 50 features + 1 label

    def test_gan_returns_tensor(self, gan_model):
        """GAN generates correct samples."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=gan_model, modelname="GANs", latent_size=32, num_images=10
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 50)

    def test_num_images_as_list(self, ae_model):
        """Passing num_images as single-element list works."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=ae_model, modelname="AE", latent_size=64, num_images=[10]
        )
        assert result.shape == (10, 50)

    def test_no_files_written(self, ae_model, tmp_path):
        """generate_samples writes no files to disk."""
        from syng_bts.helper_utils import generate_samples

        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            generate_samples(
                model=ae_model, modelname="AE", latent_size=64, num_images=5
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, ae_model):
        """generate_samples does not create matplotlib figures."""
        from syng_bts.helper_utils import generate_samples

        figs_before = plt.get_fignums()
        generate_samples(model=ae_model, modelname="AE", latent_size=64, num_images=5)
        figs_after = plt.get_fignums()
        assert figs_before == figs_after

    def test_capping(self, ae_model):
        """Capping with col_max and col_sd works without error."""
        from syng_bts.helper_utils import generate_samples

        col_max = torch.ones(50) * 5.0
        col_sd = torch.ones(50) * 1.0
        result = generate_samples(
            model=ae_model,
            modelname="AE",
            latent_size=64,
            num_images=10,
            col_max=col_max,
            col_sd=col_sd,
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 50)

    def test_cvae_multi_group(self, cvae_model):
        """CVAE multi-group generation with replicate factor."""
        from syng_bts.helper_utils import generate_samples

        result = generate_samples(
            model=cvae_model,
            modelname="CVAE",
            latent_size=32,
            num_images=[5, 5, 2],
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 20  # (5+5) * 2
        assert result.shape[1] == 51  # 50 features + 1 label


# ===========================================================================
# reconstruct_samples() — pure computation function
# ===========================================================================
class TestReconstructSamples:
    """Tests for the reconstruct_samples() pure function."""

    def test_ae_returns_tuple(self, ae_model, sample_dataloader):
        """reconstruct_samples returns a (data, labels) tuple for AE."""
        from syng_bts.helper_utils import reconstruct_samples

        result = reconstruct_samples(
            model=ae_model, modelname="AE", data_loader=sample_dataloader, n_features=50
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ae_data_shape(self, ae_model, sample_dataloader):
        """AE reconstruction doubles the sample count (orig + decoded)."""
        from syng_bts.helper_utils import reconstruct_samples

        data, labels = reconstruct_samples(
            model=ae_model, modelname="AE", data_loader=sample_dataloader, n_features=50
        )
        assert data.shape == (40, 50)
        assert isinstance(data, torch.Tensor)

    def test_vae_returns_correct_shapes(self, vae_model, sample_dataloader):
        """VAE reconstruction returns correct shapes."""
        from syng_bts.helper_utils import reconstruct_samples

        data, labels = reconstruct_samples(
            model=vae_model,
            modelname="VAE",
            data_loader=sample_dataloader,
            n_features=50,
        )
        assert data.shape == (40, 50)
        assert isinstance(labels, torch.Tensor)

    def test_cvae_appends_labels(self, cvae_model, sample_dataloader_two_class):
        """CVAE reconstruction appends label column to data."""
        from syng_bts.helper_utils import reconstruct_samples

        data, labels = reconstruct_samples(
            model=cvae_model,
            modelname="CVAE",
            data_loader=sample_dataloader_two_class,
            n_features=50,
        )
        assert data.shape[1] == 51
        assert data.shape[0] == 40

    def test_no_files_written(self, ae_model, sample_dataloader, tmp_path):
        """reconstruct_samples writes no files to disk."""
        from syng_bts.helper_utils import reconstruct_samples

        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            reconstruct_samples(
                model=ae_model,
                modelname="AE",
                data_loader=sample_dataloader,
                n_features=50,
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, ae_model, sample_dataloader):
        """reconstruct_samples does not create matplotlib figures."""
        from syng_bts.helper_utils import reconstruct_samples

        figs_before = plt.get_fignums()
        reconstruct_samples(
            model=ae_model, modelname="AE", data_loader=sample_dataloader, n_features=50
        )
        figs_after = plt.get_fignums()
        assert figs_before == figs_after

    def test_data_is_detached(self, ae_model, sample_dataloader):
        """Returned data tensor should be detached from computation graph."""
        from syng_bts.helper_utils import reconstruct_samples

        data, labels = reconstruct_samples(
            model=ae_model, modelname="AE", data_loader=sample_dataloader, n_features=50
        )
        assert not data.requires_grad


# ===========================================================================
# create_labels global RNG preservation
# ===========================================================================
class TestCreateLabelsNoGlobalSeed:
    """create_labels and create_labels_mul don't mutate global RNG state."""

    def test_create_labels_preserves_torch_rng(self):
        from syng_bts.helper_utils import create_labels

        groups = pd.Series(["A", "A", "B", "B", "B"])

        torch.manual_seed(42)
        before = torch.randn(5)

        torch.manual_seed(42)
        create_labels(len(groups), groups=groups)
        after = torch.randn(5)

        assert torch.allclose(before, after), (
            "create_labels mutated global torch RNG state"
        )

    def test_create_labels_preserves_numpy_rng(self):
        from syng_bts.helper_utils import create_labels

        groups = pd.Series(["A", "A", "B", "B", "B"])

        np.random.seed(42)
        before = np.random.rand(5)

        np.random.seed(42)
        create_labels(len(groups), groups=groups)
        after = np.random.rand(5)

        np.testing.assert_array_equal(before, after)

    def test_create_labels_preserves_python_rng(self):
        import random

        from syng_bts.helper_utils import create_labels

        groups = pd.Series(["A", "A", "B", "B", "B"])

        random.seed(42)
        before = [random.random() for _ in range(5)]

        random.seed(42)
        create_labels(len(groups), groups=groups)
        after = [random.random() for _ in range(5)]

        assert before == after

    def test_create_labels_deterministic(self):
        from syng_bts.helper_utils import create_labels

        groups = pd.Series(["A", "A", "B", "B", "B"])
        _, blur1 = create_labels(len(groups), groups=groups)
        _, blur2 = create_labels(len(groups), groups=groups)

        assert torch.allclose(blur1, blur2)

    def test_create_labels_mul_preserves_torch_rng(self):
        from syng_bts.helper_utils import create_labels_mul

        groups = pd.Series(["A", "A", "B", "B", "C", "C"])

        torch.manual_seed(42)
        before = torch.randn(5)

        torch.manual_seed(42)
        create_labels_mul(len(groups), groups=groups)
        after = torch.randn(5)

        assert torch.allclose(before, after), (
            "create_labels_mul mutated global torch RNG state"
        )

    def test_create_labels_mul_deterministic(self):
        from syng_bts.helper_utils import create_labels_mul

        groups = pd.Series(["A", "A", "B", "B", "C", "C"])
        _, blur1 = create_labels_mul(len(groups), groups=groups)
        _, blur2 = create_labels_mul(len(groups), groups=groups)

        assert torch.allclose(blur1, blur2)
