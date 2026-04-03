"""
Tests for model forward passes.

Basic forward-pass tests for AE, VAE, CVAE, GAN models
to verify correct input/output shapes and construction.
"""

import pytest
import torch

from syng_bts.helper_models import AE, CVAE, GAN, VAE


# ===========================================================================
# AE
# ===========================================================================
class TestAEForwardPass:
    """Test AE model forward passes."""

    def test_ae_forward_returns_two_tensors(self):
        """AE.forward returns (encoded, decoded)."""
        model = AE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            encoded, decoded = model(x)
        assert encoded.shape == (5, 64)  # latent dim = 64
        assert decoded.shape == (5, 50)

    def test_ae_encoder_decoder_shapes(self):
        """Encoder and decoder produce correct shapes."""
        model = AE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            encoded = model.encoder(x)
            decoded = model.decoder(encoded)
        assert encoded.shape == (5, 64)
        assert decoded.shape == (5, 50)

    @pytest.mark.parametrize("n_feat", [10, 50, 100, 500])
    def test_ae_varying_features(self, n_feat):
        """AE works with varying input dimensions."""
        model = AE(num_features=n_feat)
        model.eval()
        x = torch.randn(3, n_feat)
        with torch.no_grad():
            encoded, decoded = model(x)
        assert decoded.shape == (3, n_feat)

    def test_ae_batch_size_one(self):
        """AE works with single-sample batch."""
        model = AE(num_features=50)
        model.eval()
        x = torch.randn(1, 50)
        with torch.no_grad():
            _, decoded = model(x)
        assert decoded.shape == (1, 50)


# ===========================================================================
# VAE
# ===========================================================================
class TestVAEForwardPass:
    """Test VAE model forward passes."""

    def test_vae_forward_returns_four_tensors(self):
        """VAE.forward returns (encoded, z_mean, z_log_var, decoded)."""
        model = VAE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            encoded, z_mean, z_log_var, decoded = model(x, deterministic=True)
        assert encoded.shape == (5, 32)  # latent dim = 32
        assert z_mean.shape == (5, 32)
        assert z_log_var.shape == (5, 32)
        assert decoded.shape == (5, 50)

    def test_vae_deterministic_encoding(self):
        """Deterministic mode returns z_mean as encoded."""
        model = VAE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            encoded, z_mean, _, _ = model(x, deterministic=True)
        assert torch.allclose(encoded, z_mean)

    def test_vae_stochastic_encoding(self):
        """Stochastic mode adds noise to encoding."""
        model = VAE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        # Two forward passes with stochastic mode should differ
        enc1, _, _, _ = model(x, deterministic=False)
        enc2, _, _, _ = model(x, deterministic=False)
        # Very unlikely to be exactly equal with stochasticity
        assert not torch.allclose(enc1, enc2)

    @pytest.mark.parametrize("n_feat", [10, 50, 100])
    def test_vae_varying_features(self, n_feat):
        """VAE works with varying input dimensions."""
        model = VAE(num_features=n_feat)
        model.eval()
        x = torch.randn(3, n_feat)
        with torch.no_grad():
            _, _, _, decoded = model(x, deterministic=True)
        assert decoded.shape == (3, n_feat)

    def test_vae_encoding_fn(self):
        """encoding_fn returns a latent vector."""
        model = VAE(num_features=50)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            encoded = model.encoding_fn(x, deterministic=True)
        assert encoded.shape == (5, 32)


# ===========================================================================
# CVAE
# ===========================================================================
class TestCVAEForwardPass:
    """Test CVAE model forward passes."""

    def test_cvae_forward_returns_four_tensors(self):
        """CVAE.forward returns (encoded, z_mean, z_log_var, decoded)."""
        model = CVAE(num_features=50, num_classes=2)
        model.eval()
        x = torch.randn(5, 50)
        y = torch.randint(0, 2, (5, 1)).float()
        with torch.no_grad():
            encoded, z_mean, z_log_var, decoded = model(x, y, deterministic=True)
        assert encoded.shape == (5, 32)
        assert z_mean.shape == (5, 32)
        assert z_log_var.shape == (5, 32)
        assert decoded.shape == (5, 50)

    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_cvae_varying_classes(self, n_classes):
        """CVAE works with different numbers of classes."""
        model = CVAE(num_features=50, num_classes=n_classes)
        model.eval()
        x = torch.randn(4, 50)
        y = torch.randint(0, n_classes, (4, 1)).float()
        with torch.no_grad():
            _, _, _, decoded = model(x, y, deterministic=True)
        assert decoded.shape == (4, 50)

    def test_cvae_encoding_fn(self):
        """encoding_fn returns (z_mean, z_log_var, encoded)."""
        model = CVAE(num_features=50, num_classes=2)
        model.eval()
        x = torch.randn(5, 50)
        y = torch.ones(5, 1)
        with torch.no_grad():
            z_mean, z_log_var, encoded = model.encoding_fn(x, y, deterministic=True)
        assert z_mean.shape == (5, 32)
        assert encoded.shape == (5, 32)

    def test_cvae_decoding_fn(self):
        """decoding_fn returns decoded output."""
        model = CVAE(num_features=50, num_classes=2)
        model.eval()
        z = torch.randn(5, 32)
        y = torch.ones(5, 1)
        with torch.no_grad():
            decoded = model.decoding_fn(z, y)
        assert decoded.shape == (5, 50)


# ===========================================================================
# CVAE Wide Network
# ===========================================================================
class TestCVAEWideNetwork:
    """Test CVAE wide_network architecture option."""

    def test_standard_encoder_layer_count(self):
        """wide_network=False → 3 hidden layers (256, 128, 64)."""
        model = CVAE(num_features=50, num_classes=2, wide_network=False)
        # 3 hidden layers × 3 modules each (Linear + BN + ReLU) = 9
        assert len(model.encoder) == 9

    def test_wide_encoder_layer_count(self):
        """wide_network=True → 4 hidden layers (512, 256, 128, 64)."""
        model = CVAE(num_features=50, num_classes=2, wide_network=True)
        # 4 hidden layers × 3 modules each = 12
        assert len(model.encoder) == 12

    def test_standard_decoder_layer_count(self):
        """wide_network=False → decoder has 3 hidden + 1 output + 1 ReLU = 11."""
        model = CVAE(num_features=50, num_classes=2, wide_network=False)
        # 3 hidden × 3 (Linear+BN+ReLU) + Linear + ReLU = 11
        assert len(model.decoder) == 11

    def test_wide_decoder_layer_count(self):
        """wide_network=True → decoder has 4 hidden + 1 output + 1 ReLU = 14."""
        model = CVAE(num_features=50, num_classes=2, wide_network=True)
        # 4 hidden × 3 + Linear + ReLU = 14
        assert len(model.decoder) == 14

    def test_standard_encoder_first_linear_dim(self):
        """wide_network=False → first Linear is (num_features+1, 256)."""
        model = CVAE(num_features=50, num_classes=2, wide_network=False)
        first_linear = model.encoder[0]
        assert first_linear.in_features == 51
        assert first_linear.out_features == 256

    def test_wide_encoder_first_linear_dim(self):
        """wide_network=True → first Linear is (num_features+1, 512)."""
        model = CVAE(num_features=50, num_classes=2, wide_network=True)
        first_linear = model.encoder[0]
        assert first_linear.in_features == 51
        assert first_linear.out_features == 512

    def test_wide_network_forward_pass(self):
        """Forward pass works with wide_network=True."""
        model = CVAE(num_features=50, num_classes=2, wide_network=True)
        model.eval()
        x = torch.randn(5, 50)
        y = torch.randint(0, 2, (5, 1)).float()
        with torch.no_grad():
            encoded, z_mean, z_log_var, decoded = model(x, y, deterministic=True)
        assert encoded.shape == (5, 32)
        assert z_mean.shape == (5, 32)
        assert z_log_var.shape == (5, 32)
        assert decoded.shape == (5, 50)

    def test_wide_network_attribute(self):
        """wide_network attribute is stored on the model."""
        model_std = CVAE(num_features=50, num_classes=2, wide_network=False)
        model_wide = CVAE(num_features=50, num_classes=2, wide_network=True)
        assert model_std.wide_network is False
        assert model_wide.wide_network is True

    def test_default_is_standard(self):
        """Default (no wide_network arg) matches wide_network=False."""
        model_default = CVAE(num_features=50, num_classes=2)
        model_std = CVAE(num_features=50, num_classes=2, wide_network=False)
        assert len(model_default.encoder) == len(model_std.encoder)
        assert len(model_default.decoder) == len(model_std.decoder)


# ===========================================================================
# GAN
# ===========================================================================
class TestGANForwardPass:
    """Test GAN model forward passes."""

    def test_gan_generator(self):
        """Generator produces correct output shape."""
        model = GAN(num_features=50, latent_dim=32)
        model.eval()
        z = torch.randn(5, 32)
        with torch.no_grad():
            generated = model.generator(z)
        assert generated.shape == (5, 50)

    def test_gan_discriminator(self):
        """Discriminator produces single logit per sample."""
        model = GAN(num_features=50, latent_dim=32)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            disc_out = model.discriminator(x)
        assert disc_out.shape == (5, 1)

    def test_gan_generator_forward(self):
        """generator_forward method works."""
        model = GAN(num_features=50, latent_dim=32)
        model.eval()
        z = torch.randn(5, 32)
        with torch.no_grad():
            generated = model.generator_forward(z)
        assert generated.shape == (5, 50)

    def test_gan_discriminator_forward(self):
        """discriminator_forward method works."""
        model = GAN(num_features=50, latent_dim=32)
        model.eval()
        x = torch.randn(5, 50)
        with torch.no_grad():
            logits = model.discriminator_forward(x)
        assert logits.shape == (5, 1)

    @pytest.mark.parametrize("n_feat", [10, 50, 100])
    def test_gan_varying_features(self, n_feat):
        """GAN works with varying feature dimensions."""
        model = GAN(num_features=n_feat, latent_dim=32)
        model.eval()
        z = torch.randn(3, 32)
        with torch.no_grad():
            generated = model.generator(z)
        assert generated.shape == (3, n_feat)

    @pytest.mark.parametrize("latent_dim", [16, 32, 64])
    def test_gan_varying_latent_dim(self, latent_dim):
        """GAN works with varying latent dimensions."""
        model = GAN(num_features=50, latent_dim=latent_dim)
        model.eval()
        z = torch.randn(3, latent_dim)
        with torch.no_grad():
            generated = model.generator(z)
        assert generated.shape == (3, 50)
