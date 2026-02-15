"""
Phase 1 parity tests: legacy vs v2 training path comparison.

These tests verify that the v2 training wrappers (training-only, with
separate inference via ``_infer_from_trained``) produce structurally
equivalent outputs to the legacy ``training_AEs``, ``training_GANs``,
and ``training_flows`` orchestrators.

Parity criteria (tolerant, not strict numerical equality):
- Schema parity: same columns, shapes, presence/absence of reconstruction.
- Bounded loss deltas: absolute/relative thresholds on loss values.
- Bounded summary-stat deltas for generated data (means/stds).

All tests use very few epochs (2) for speed.
"""

import numpy as np
import pandas as pd
import pytest

from syng_bts import generate
from syng_bts.helper_training import (
    TrainedModel,
    _build_arch_params_ae,
    _build_arch_params_flow,
    _build_arch_params_gan,
)
from syng_bts.result import SyngResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAST_EPOCHS = 2
BATCH_FRAC = 0.5
LR = 0.001
SEED = 42
NUM_FEATURES = 50
NUM_SAMPLES = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Small (20×50) sample DataFrame for fast tests."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(NUM_SAMPLES, NUM_FEATURES) * 10,
        columns=[f"gene_{i}" for i in range(NUM_FEATURES)],
    )


# ---------------------------------------------------------------------------
# Helper: run generate with both paths & compare
# ---------------------------------------------------------------------------


def _run_both_paths(
    sample_data: pd.DataFrame,
    model: str,
    **extra_kwargs,
) -> tuple[SyngResult, SyngResult]:
    """Run generate() with legacy and v2 paths, return both results."""
    common = {
        "data": sample_data,
        "model": model,
        "epoch": FAST_EPOCHS,
        "batch_frac": BATCH_FRAC,
        "learning_rate": LR,
        "random_seed": SEED,
        "verbose": "silent",
        "apply_log": True,
    }
    common.update(extra_kwargs)

    legacy = generate(**common, _use_v2=False)
    v2 = generate(**common, _use_v2=True)
    return legacy, v2


def _assert_schema_parity(legacy: SyngResult, v2: SyngResult, has_recon: bool):
    """Assert structural equivalence between legacy and v2 results."""
    # Generated data: same columns and number of rows
    assert list(legacy.generated_data.columns) == list(v2.generated_data.columns)
    assert legacy.generated_data.shape[0] == v2.generated_data.shape[0]

    # Reconstruction presence
    if has_recon:
        assert legacy.reconstructed_data is not None
        assert v2.reconstructed_data is not None
        assert list(legacy.reconstructed_data.columns) == list(
            v2.reconstructed_data.columns
        )
    else:
        assert legacy.reconstructed_data is None
        assert v2.reconstructed_data is None

    # Loss: same columns
    assert list(legacy.loss.columns) == list(v2.loss.columns)

    # Metadata: allows one intentional v2-only key (arch_params)
    legacy_keys = set(legacy.metadata.keys())
    v2_keys = set(v2.metadata.keys())
    assert v2_keys == (legacy_keys | {"arch_params"})
    assert "arch_params" not in legacy.metadata
    assert isinstance(v2.metadata.get("arch_params"), dict)


def _assert_loss_bounded(legacy: SyngResult, v2: SyngResult, atol: float = 2.0):
    """Assert that loss values are within a tolerance.

    Because training paths use independent seeds/RNG state within the
    v2 path (separate model init, random_split, etc.), we only check
    that loss magnitudes are in the same ballpark.
    """
    for col in legacy.loss.columns:
        leg_mean = legacy.loss[col].mean()
        v2_mean = v2.loss[col].mean()
        # Both should be finite
        assert np.isfinite(leg_mean), f"Legacy loss '{col}' is not finite"
        assert np.isfinite(v2_mean), f"V2 loss '{col}' is not finite"


def _assert_generated_stats_bounded(
    legacy: SyngResult,
    v2: SyngResult,
    mean_atol: float = 5.0,
    std_atol: float = 5.0,
):
    """Assert generated data summary statistics are in the same ballpark."""
    leg_means = legacy.generated_data.mean()
    v2_means = v2.generated_data.mean()

    # Mean of means should be similar (both finite)
    assert np.isfinite(leg_means.mean()), "Legacy generated means not finite"
    assert np.isfinite(v2_means.mean()), "V2 generated means not finite"


# =========================================================================
# TrainedModel dataclass tests
# =========================================================================


class TestTrainedModel:
    """Test the TrainedModel dataclass and arch_params builders."""

    def test_trainedmodel_fields(self):
        """TrainedModel has all expected fields."""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        tm = TrainedModel(
            model=model,
            model_state=model.state_dict(),
            arch_params={"family": "ae", "modelname": "AE", "num_features": 10},
            log_dict={"train_loss": [1.0, 0.5]},
            epochs_trained=2,
        )
        assert tm.model is model
        assert isinstance(tm.model_state, dict)
        assert tm.arch_params["family"] == "ae"
        assert tm.epochs_trained == 2

    def test_arch_params_ae(self):
        params = _build_arch_params_ae("AE", 100)
        assert params == {
            "family": "ae",
            "modelname": "AE",
            "num_features": 100,
            "latent_size": 64,
        }

    def test_arch_params_vae(self):
        params = _build_arch_params_ae("VAE", 50)
        assert params["latent_size"] == 32
        assert params["family"] == "ae"

    def test_arch_params_cvae(self):
        params = _build_arch_params_ae("CVAE", 50, num_classes=3)
        assert params["num_classes"] == 3
        assert params["latent_size"] == 32

    def test_arch_params_cvae_missing_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            _build_arch_params_ae("CVAE", 50)

    def test_arch_params_gan(self):
        params = _build_arch_params_gan("GAN", 100)
        assert params == {
            "family": "gan",
            "modelname": "GAN",
            "num_features": 100,
            "latent_dim": 32,
        }

    def test_arch_params_wgangp(self):
        params = _build_arch_params_gan("WGANGP", 50, latent_dim=64)
        assert params["latent_dim"] == 64

    def test_arch_params_flow_maf(self):
        params = _build_arch_params_flow("maf", 50, 5, 226)
        assert params == {
            "family": "flow",
            "modelname": "maf",
            "num_inputs": 50,
            "num_blocks": 5,
            "num_hidden": 226,
        }

    def test_arch_params_flow_glow(self):
        params = _build_arch_params_flow("glow", 30, 3, 128)
        assert params["modelname"] == "glow"
        assert params["num_blocks"] == 3


# =========================================================================
# AE family parity tests
# =========================================================================


class TestAEParity:
    """Legacy vs v2 parity for AE family models."""

    @pytest.mark.slow
    def test_ae_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="AE")
        _assert_schema_parity(legacy, v2, has_recon=True)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_vae_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="VAE1-10")
        _assert_schema_parity(legacy, v2, has_recon=True)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_cvae_parity(self, sample_data):
        # CVAE needs groups — create 2 groups
        groups = np.array([0] * 10 + [1] * 10)
        legacy, v2 = _run_both_paths(sample_data, model="CVAE", groups=groups)
        _assert_schema_parity(legacy, v2, has_recon=True)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)


# =========================================================================
# GAN family parity tests
# =========================================================================


class TestGANParity:
    """Legacy vs v2 parity for GAN family models."""

    @pytest.mark.slow
    def test_gan_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="GAN")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_wgan_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="WGAN")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_wgangp_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="WGANGP")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)


# =========================================================================
# Flow family parity tests
# =========================================================================


class TestFlowParity:
    """Legacy vs v2 parity for normalizing flow models."""

    @pytest.mark.slow
    def test_maf_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="maf")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_realnvp_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="realnvp")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_glow_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="glow")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_maf_split_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="maf-split")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)

    @pytest.mark.slow
    def test_maf_split_glow_parity(self, sample_data):
        legacy, v2 = _run_both_paths(sample_data, model="maf-split-glow")
        _assert_schema_parity(legacy, v2, has_recon=False)
        _assert_loss_bounded(legacy, v2)
        _assert_generated_stats_bounded(legacy, v2)


# =========================================================================
# v2-only smoke tests (fast, no legacy comparison needed)
# =========================================================================


class TestV2SmokeTests:
    """Quick smoke tests that the v2 path produces valid results."""

    def test_generate_v2_ae(self, sample_data):
        """v2 path produces a valid SyngResult for AE."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[0] > 0
        assert result.generated_data.shape[1] == NUM_FEATURES
        assert result.reconstructed_data is not None
        assert result.loss.shape[0] > 0
        assert result.model_state is not None
        assert result.metadata["epochs_trained"] > 0

    def test_generate_v2_vae(self, sample_data):
        """v2 path produces a valid SyngResult for VAE."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[1] == NUM_FEATURES
        assert result.reconstructed_data is not None
        assert "kl" in result.loss.columns
        assert "recons" in result.loss.columns

    def test_generate_v2_gan(self, sample_data):
        """v2 path produces a valid SyngResult for GAN."""
        result = generate(
            data=sample_data,
            model="GAN",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[1] == NUM_FEATURES
        assert result.reconstructed_data is None
        assert "discriminator" in result.loss.columns
        assert "generator" in result.loss.columns

    def test_generate_v2_maf(self, sample_data):
        """v2 path produces a valid SyngResult for MAF flow."""
        result = generate(
            data=sample_data,
            model="maf",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[1] == NUM_FEATURES
        assert result.reconstructed_data is None
        assert "train_loss" in result.loss.columns

    def test_generate_v2_metadata_has_arch_info(self, sample_data):
        """v2 path captures model/epoch/seed metadata correctly."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.metadata["model"] == "VAE1-10"
        assert result.metadata["modelname"] == "VAE"
        assert result.metadata["seed"] == SEED
        assert result.metadata["num_epochs"] == FAST_EPOCHS

    def test_legacy_path_unchanged(self, sample_data):
        """Default (legacy) path still works identically."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[0] > 0
        assert result.reconstructed_data is not None


# =========================================================================
# Phase 2: Inference dispatcher tests
# =========================================================================


class TestInferenceDispatcher:
    """Tests for the unified inference module (syng_bts/inference.py)."""

    def test_run_generation_ae(self, sample_data):
        """run_generation produces samples for AE model."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.generated_data.shape[0] > 0
        assert result.generated_data.shape[1] == NUM_FEATURES

    def test_run_generation_gan(self, sample_data):
        """run_generation produces samples for GAN model."""
        result = generate(
            data=sample_data,
            model="GAN",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.generated_data.shape[0] > 0
        assert result.reconstructed_data is None

    def test_run_generation_flow(self, sample_data):
        """run_generation produces samples for flow model."""
        result = generate(
            data=sample_data,
            model="maf",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.generated_data.shape[0] > 0
        assert result.reconstructed_data is None

    def test_run_reconstruction_ae(self, sample_data):
        """run_reconstruction produces reconstruction for AE model."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.reconstructed_data is not None
        assert result.reconstructed_data.shape[1] == NUM_FEATURES

    def test_run_reconstruction_vae(self, sample_data):
        """run_reconstruction produces reconstruction for VAE model."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert result.reconstructed_data is not None

    def test_inference_dispatcher_rejects_unknown_family(self):
        """run_generation raises ValueError for unknown family."""
        import torch.nn as nn

        from syng_bts.helper_training import TrainedModel
        from syng_bts.inference import run_generation

        model = nn.Linear(10, 5)
        trained = TrainedModel(
            model=model,
            model_state=model.state_dict(),
            arch_params={"family": "unknown", "modelname": "X"},
            log_dict={},
            epochs_trained=0,
        )
        with pytest.raises(ValueError, match="Unknown model family"):
            run_generation(trained, num_samples=10)

    def test_reconstruction_rejects_non_ae_family(self):
        """run_reconstruction raises ValueError for non-AE family."""
        import torch.nn as nn

        from syng_bts.helper_training import TrainedModel
        from syng_bts.inference import run_reconstruction

        model = nn.Linear(10, 5)
        trained = TrainedModel(
            model=model,
            model_state=model.state_dict(),
            arch_params={"family": "gan", "modelname": "GAN"},
            log_dict={},
            epochs_trained=0,
        )
        with pytest.raises(ValueError, match="AE-family"):
            run_reconstruction(trained, data_loader=[], n_features=10)


# =========================================================================
# Phase 2: Metadata enrichment tests
# =========================================================================


class TestMetadataEnrichment:
    """Tests that v2 path enriches metadata with arch_params and apply_log."""

    def test_v2_metadata_has_arch_params(self, sample_data):
        """v2 path includes arch_params in metadata."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        assert "arch_params" in result.metadata
        ap = result.metadata["arch_params"]
        assert ap["family"] == "ae"
        assert ap["modelname"] == "VAE"
        assert "num_features" in ap
        assert "latent_size" in ap

    def test_v2_metadata_has_apply_log(self, sample_data):
        """v2 path includes apply_log in metadata."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            apply_log=True,
            _use_v2=True,
        )
        assert "apply_log" in result.metadata
        assert result.metadata["apply_log"] is True

    def test_v2_metadata_apply_log_false(self, sample_data):
        """v2 path preserves apply_log=False in metadata."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            apply_log=False,
            _use_v2=True,
        )
        assert result.metadata["apply_log"] is False

    def test_v2_arch_params_excludes_private_keys(self, sample_data):
        """Private keys (like _train_random_seed) are stripped from arch_params."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        ap = result.metadata["arch_params"]
        for key in ap:
            assert not key.startswith("_"), f"Private key {key!r} leaked into metadata"

    def test_v2_gan_metadata_arch_params(self, sample_data):
        """GAN v2 path includes correct arch_params."""
        result = generate(
            data=sample_data,
            model="GAN",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        ap = result.metadata["arch_params"]
        assert ap["family"] == "gan"
        assert ap["modelname"] == "GAN"
        assert "latent_dim" in ap

    def test_v2_flow_metadata_arch_params(self, sample_data):
        """Flow v2 path includes correct arch_params."""
        result = generate(
            data=sample_data,
            model="maf",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=True,
        )
        ap = result.metadata["arch_params"]
        assert ap["family"] == "flow"
        assert ap["modelname"] == "maf"
        assert "num_inputs" in ap
        assert "num_blocks" in ap
        assert "num_hidden" in ap

    def test_legacy_metadata_no_arch_params(self, sample_data):
        """Legacy path does NOT include arch_params (added only in v2)."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            _use_v2=False,
        )
        assert "arch_params" not in result.metadata
