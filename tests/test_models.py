"""
Tests for neural network model instantiation and basic forward pass.
"""

import pytest
import torch
import pandas as pd


class TestAEModel:
    """Test Autoencoder model."""

    def test_ae_instantiation(self):
        """Test AE model instantiation."""
        from syng_bts import AE

        num_features = 100
        model = AE(num_features)

        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")

    def test_ae_forward_pass(self):
        """Test AE forward pass."""
        from syng_bts import AE

        num_features = 100
        batch_size = 16
        model = AE(num_features)

        x = torch.randn(batch_size, num_features)
        encoded, decoded = model(x)

        assert encoded.shape == (batch_size, 64)  # Latent dim is 64
        assert decoded.shape == (batch_size, num_features)

    def test_ae_different_feature_sizes(self):
        """Test AE with different input sizes."""
        from syng_bts import AE

        for num_features in [50, 200, 500, 1000]:
            model = AE(num_features)
            x = torch.randn(8, num_features)
            encoded, decoded = model(x)

            assert decoded.shape == (8, num_features)


class TestVAEModel:
    """Test Variational Autoencoder model."""

    def test_vae_instantiation(self):
        """Test VAE model instantiation."""
        from syng_bts import VAE

        num_features = 100
        model = VAE(num_features)

        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "z_mean")
        assert hasattr(model, "z_log_var")

    def test_vae_forward_pass(self):
        """Test VAE forward pass."""
        from syng_bts import VAE

        num_features = 100
        batch_size = 16
        model = VAE(num_features)

        x = torch.randn(batch_size, num_features)
        encoded, z_mean, z_log_var, decoded = model(x)

        assert encoded.shape == (batch_size, 32)  # Latent dim is 32
        assert z_mean.shape == (batch_size, 32)
        assert z_log_var.shape == (batch_size, 32)
        assert decoded.shape == (batch_size, num_features)

    def test_vae_deterministic_mode(self):
        """Test VAE deterministic (no reparameterization) mode."""
        from syng_bts import VAE

        num_features = 100
        model = VAE(num_features)

        x = torch.randn(8, num_features)

        # In deterministic mode, encoded should equal z_mean
        encoded, z_mean, z_log_var, decoded = model(x, deterministic=True)

        assert torch.allclose(encoded, z_mean)

    def test_vae_encoding_fn(self):
        """Test VAE encoding function."""
        from syng_bts import VAE

        num_features = 100
        model = VAE(num_features)

        x = torch.randn(8, num_features)
        encoded = model.encoding_fn(x)

        assert encoded.shape == (8, 32)


class TestCVAEModel:
    """Test Conditional Variational Autoencoder model."""

    def test_cvae_instantiation(self):
        """Test CVAE model instantiation."""
        from syng_bts import CVAE

        num_features = 100
        num_classes = 2
        model = CVAE(num_features, num_classes)

        assert model is not None
        assert model.num_features == num_features
        assert model.num_classes == num_classes

    def test_cvae_forward_pass(self):
        """Test CVAE forward pass with labels."""
        from syng_bts import CVAE

        num_features = 100
        num_classes = 2
        batch_size = 16
        model = CVAE(num_features, num_classes)

        x = torch.randn(batch_size, num_features)
        y = torch.randint(0, num_classes, (batch_size, 1)).float()

        encoded, z_mean, z_log_var, decoded = model(x, y)

        assert encoded.shape == (batch_size, 32)
        assert z_mean.shape == (batch_size, 32)
        assert z_log_var.shape == (batch_size, 32)
        assert decoded.shape == (batch_size, num_features)

    def test_cvae_encoding_decoding(self):
        """Test CVAE separate encoding and decoding functions."""
        from syng_bts import CVAE

        num_features = 100
        num_classes = 2
        batch_size = 8
        model = CVAE(num_features, num_classes)

        x = torch.randn(batch_size, num_features)
        y = torch.randint(0, num_classes, (batch_size, 1)).float()

        z_mean, z_log_var, encoded = model.encoding_fn(x, y)
        decoded = model.decoding_fn(encoded, y)

        assert encoded.shape == (batch_size, 32)
        assert decoded.shape == (batch_size, num_features)


class TestGANModel:
    """Test GAN model."""

    def test_gan_instantiation(self):
        """Test GAN model instantiation."""
        from syng_bts import GAN

        num_features = 100
        model = GAN(num_features)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")
        assert model.num_features == num_features

    def test_gan_generator_forward(self):
        """Test GAN generator forward pass."""
        from syng_bts import GAN

        num_features = 100
        latent_dim = 32
        batch_size = 16
        model = GAN(num_features, latent_dim=latent_dim)

        z = torch.randn(batch_size, latent_dim)
        generated = model.generator_forward(z)

        assert generated.shape == (batch_size, num_features)

    def test_gan_discriminator_forward(self):
        """Test GAN discriminator forward pass."""
        from syng_bts import GAN

        num_features = 100
        batch_size = 16
        model = GAN(num_features)

        fake_data = torch.randn(batch_size, num_features)
        logits = model.discriminator_forward(fake_data)

        assert logits.shape == (batch_size, 1)

    def test_gan_custom_latent_dim(self):
        """Test GAN with custom latent dimension."""
        from syng_bts import GAN

        num_features = 100
        latent_dim = 64
        model = GAN(num_features, latent_dim=latent_dim)

        z = torch.randn(8, latent_dim)
        generated = model.generator_forward(z)

        assert generated.shape == (8, num_features)


class TestModelsTrainable:
    """Test that models have trainable parameters."""

    def test_ae_has_parameters(self):
        """Test AE has trainable parameters."""
        from syng_bts import AE

        model = AE(100)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_vae_has_parameters(self):
        """Test VAE has trainable parameters."""
        from syng_bts import VAE

        model = VAE(100)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_cvae_has_parameters(self):
        """Test CVAE has trainable parameters."""
        from syng_bts import CVAE

        model = CVAE(100, 2)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_gan_has_parameters(self):
        """Test GAN has trainable parameters."""
        from syng_bts import GAN

        model = GAN(100)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)
