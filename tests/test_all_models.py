"""
Validation tests for all model families.

These tests verify that the training wrappers (training-only, with
separate inference via the unified dispatcher) produce correct
results for all model families and variants.

Coverage:
  - Smoke tests for each model family.
  - Inference dispatcher and metadata enrichment tests.
  - Model reconstruction parity: reloading ``state_dict`` into a
    freshly constructed model yields equivalent inference outputs.
  - Metadata schema and required-key assertions.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from syng_bts import generate
from syng_bts.helper_models import AE, CVAE, GAN, VAE
from syng_bts.helper_training import (
    TrainedModel,
    _build_arch_params_ae,
    _build_arch_params_flow,
    _build_arch_params_gan,
)
from syng_bts.model_factory import rebuild_model
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
        np.random.randint(0, 11, size=(NUM_SAMPLES, NUM_FEATURES)).astype(float),
        columns=[f"gene_{i}" for i in range(NUM_FEATURES)],
    )


# ---------------------------------------------------------------------------
# Helper: assert metadata schema
# ---------------------------------------------------------------------------


def _assert_metadata_schema(result: SyngResult, *, model_str: str):
    """Assert metadata has all required keys and correct types.

    Parameters
    ----------
    result : SyngResult
    model_str : str
        The model specification string (for validation).
    """
    meta = result.metadata
    required_keys = {
        "model",
        "modelname",
        "dataname",
        "num_epochs",
        "epochs_trained",
        "seed",
        "kl_weight",
        "reconstruction_term_weight",
        "input_shape",
        "early_stop",
        "early_stop_patience",
        "apply_log",
        "arch_params",
    }
    for key in required_keys:
        assert key in meta, f"Missing required metadata key: {key!r}"

    assert meta["model"] == model_str
    assert isinstance(meta["epochs_trained"], int)
    assert meta["epochs_trained"] > 0
    assert isinstance(meta["input_shape"], (tuple, list))
    assert len(meta["input_shape"]) == 2
    assert isinstance(meta["apply_log"], bool)

    ap = meta["arch_params"]
    assert isinstance(ap, dict)
    assert "family" in ap
    assert "modelname" in ap
    # No private keys in public metadata
    for key in ap:
        assert not key.startswith("_"), (
            f"Private key {key!r} leaked into metadata arch_params"
        )


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
# Inference dispatcher tests
# =========================================================================


class TestInferenceDispatcher:
    """Tests for the unified inference module (syng_bts/inference.py)."""

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
# Metadata enrichment tests
# =========================================================================


class TestMetadataEnrichment:
    """Tests that metadata records the apply_log value faithfully."""

    @pytest.mark.parametrize("apply_log", [True, False])
    def test_metadata_apply_log_value(self, sample_data, apply_log):
        """metadata['apply_log'] round-trips the requested value."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
            apply_log=apply_log,
        )
        assert result.metadata["apply_log"] is apply_log


# =========================================================================
# Model reconstruction parity (state_dict reload)
# =========================================================================


class TestModelReconstructionParity:
    """Verify original trained model vs rebuilt-model inference parity.

    These tests validate that a model rebuilt from ``arch_params`` and
    ``model_state`` produces parity-equivalent outputs to the originally
    trained model object from ``TrainedModel.model``.
    """

    @staticmethod
    def _train_test_model(sample_data, model: str, *, groups=None) -> TrainedModel:
        """Train a model and return TrainedModel for direct parity checks."""
        from syng_bts.core import _parse_model_spec, orchestrate_training
        from syng_bts.helper_utils import create_labels, preprocessinglog2

        oridata = torch.from_numpy(sample_data.to_numpy().copy()).to(torch.float32)
        oridata = preprocessinglog2(oridata)
        n_samples = oridata.shape[0]

        orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)
        modelname, recon_weight, kl_weight = _parse_model_spec(model)

        trained, _ctx = orchestrate_training(
            rawdata=oridata,
            rawlabels=orilabels,
            oriblurlabels=oriblurlabels,
            modelname=modelname,
            kl_weight=kl_weight,
            reconstruction_term_weight=recon_weight,
            batch_frac=BATCH_FRAC,
            random_seed=SEED,
            epoch=FAST_EPOCHS,
            early_stop_patience=None,
            learning_rate=LR,
            verbose=0,
        )
        return trained

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["AE", "VAE1-10"])
    def test_ae_family_original_vs_rebuilt_generation_parity(self, sample_data, model):
        """AE/VAE: original trained model and rebuilt model generate identically."""
        from syng_bts.inference import run_generation

        trained_original = self._train_test_model(sample_data, model)
        ap = trained_original.arch_params
        rebuilt_model = rebuild_model(ap, trained_original.model_state)
        trained_rebuilt = TrainedModel(
            model=rebuilt_model,
            model_state=trained_original.model_state,
            arch_params=ap,
            log_dict={},
            epochs_trained=0,
        )

        torch.manual_seed(99)
        gen_original = run_generation(trained_original, num_samples=50)
        torch.manual_seed(99)
        gen_rebuilt = run_generation(trained_rebuilt, num_samples=50)

        assert gen_original.shape == gen_rebuilt.shape
        assert torch.allclose(gen_original, gen_rebuilt)
        assert torch.isfinite(gen_original).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["GAN", "WGAN", "WGANGP"])
    def test_gan_family_original_vs_rebuilt_generation_parity(self, sample_data, model):
        """GAN variants: original trained model and rebuilt model generate identically."""
        from syng_bts.inference import run_generation

        trained_original = self._train_test_model(sample_data, model)
        ap = trained_original.arch_params
        rebuilt_model = rebuild_model(ap, trained_original.model_state)
        trained_rebuilt = TrainedModel(
            model=rebuilt_model,
            model_state=trained_original.model_state,
            arch_params=ap,
            log_dict={},
            epochs_trained=0,
        )

        torch.manual_seed(99)
        gen_original = run_generation(trained_original, num_samples=50)
        torch.manual_seed(99)
        gen_rebuilt = run_generation(trained_rebuilt, num_samples=50)

        assert gen_original.shape == gen_rebuilt.shape
        assert torch.allclose(gen_original, gen_rebuilt)
        assert torch.isfinite(gen_original).all()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model", ["maf", "realnvp", "glow", "maf-split", "maf-split-glow"]
    )
    def test_flow_family_original_vs_rebuilt_generation_parity(
        self, sample_data, model
    ):
        """Flow variants: original trained model and rebuilt model generate identically."""
        from syng_bts.inference import run_generation

        if model == "glow":
            pytest.xfail(
                "glow reconstruction from state_dict is not parity-stable yet: "
                "LUInvertibleMM stores critical tensors (e.g., P/sign_S) outside "
                "the serialized state"
            )

        trained_original = self._train_test_model(sample_data, model)
        ap = trained_original.arch_params
        rebuilt_model = rebuild_model(ap, trained_original.model_state)
        trained_rebuilt = TrainedModel(
            model=rebuilt_model,
            model_state=trained_original.model_state,
            arch_params=ap,
            log_dict={},
            epochs_trained=0,
        )

        torch.manual_seed(99)
        gen_original = run_generation(trained_original, num_samples=50)
        torch.manual_seed(99)
        gen_rebuilt = run_generation(trained_rebuilt, num_samples=50)

        assert gen_original.shape == gen_rebuilt.shape
        assert torch.allclose(gen_original, gen_rebuilt)
        assert torch.isfinite(gen_original).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["AE", "VAE1-10"])
    def test_ae_family_original_vs_rebuilt_reconstruction_parity(
        self, sample_data, model
    ):
        """AE/VAE: original trained model and rebuilt model reconstruct identically."""
        from torch.utils.data import DataLoader, TensorDataset

        from syng_bts.helper_utils import preprocessinglog2
        from syng_bts.inference import run_reconstruction

        trained_original = self._train_test_model(sample_data, model)
        ap = trained_original.arch_params
        rebuilt_model = rebuild_model(ap, trained_original.model_state)
        trained_rebuilt = TrainedModel(
            model=rebuilt_model,
            model_state=trained_original.model_state,
            arch_params=ap,
            log_dict={},
            epochs_trained=0,
        )

        # Build a small deterministic data loader
        oridata = torch.from_numpy(sample_data.to_numpy().copy()).to(torch.float32)
        oridata = preprocessinglog2(oridata)
        labels = torch.zeros(oridata.shape[0], 1)
        loader = DataLoader(
            TensorDataset(oridata, labels),
            batch_size=oridata.shape[0],
            shuffle=False,
            drop_last=False,
        )

        torch.manual_seed(1234)
        recon_original, _ = run_reconstruction(
            trained_original, loader, oridata.shape[1]
        )
        torch.manual_seed(1234)
        recon_rebuilt, _ = run_reconstruction(trained_rebuilt, loader, oridata.shape[1])

        assert recon_original.shape == recon_rebuilt.shape
        assert torch.allclose(recon_original, recon_rebuilt)
        assert torch.isfinite(recon_original).all()


# =========================================================================
# Metadata required keys and schema validation
# =========================================================================


class TestMetadataSchemaValidation:
    """Comprehensive metadata schema checks for all model families.

    Ensures required keys are present, types are correct, and
    arch_params schema matches the model family.
    """

    def test_ae_metadata_schema(self, sample_data):
        """AE metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="AE")
        ap = result.metadata["arch_params"]
        assert ap["family"] == "ae"
        assert ap["modelname"] == "AE"
        assert "num_features" in ap
        assert "latent_size" in ap
        assert ap["latent_size"] == 64  # AE uses 64

    def test_vae_metadata_schema(self, sample_data):
        """VAE metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="VAE1-10")
        ap = result.metadata["arch_params"]
        assert ap["latent_size"] == 32  # VAE uses 32

    def test_gan_metadata_schema(self, sample_data):
        """GAN metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="GAN",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="GAN")
        ap = result.metadata["arch_params"]
        assert ap["family"] == "gan"
        assert "latent_dim" in ap
        assert ap["latent_dim"] == 32

    def test_wgangp_metadata_schema(self, sample_data):
        """WGANGP metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="WGANGP",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="WGANGP")

    def test_maf_metadata_schema(self, sample_data):
        """MAF metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="maf",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="maf")
        ap = result.metadata["arch_params"]
        assert ap["family"] == "flow"
        assert "num_inputs" in ap
        assert "num_blocks" in ap
        assert "num_hidden" in ap

    def test_glow_metadata_schema(self, sample_data):
        """Glow metadata has correct schema."""
        result = generate(
            data=sample_data,
            model="glow",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )
        _assert_metadata_schema(result, model_str="glow")


# =========================================================================
# Model factory tests
# =========================================================================


class TestModelFactory:
    """Tests for syng_bts/model_factory.py rebuild_model()."""

    @pytest.mark.parametrize(
        "model_cls, ctor_kwargs, arch",
        [
            (
                AE,
                {},
                {
                    "family": "ae",
                    "modelname": "AE",
                    "num_features": NUM_FEATURES,
                    "latent_size": 64,
                },
            ),
            (
                VAE,
                {},
                {
                    "family": "ae",
                    "modelname": "VAE",
                    "num_features": NUM_FEATURES,
                    "latent_size": 32,
                },
            ),
            (
                CVAE,
                {"num_classes": 2},
                {
                    "family": "ae",
                    "modelname": "CVAE",
                    "num_features": NUM_FEATURES,
                    "latent_size": 32,
                    "num_classes": 2,
                },
            ),
            (
                GAN,
                {"latent_dim": 32},
                {
                    "family": "gan",
                    "modelname": "GAN",
                    "num_features": NUM_FEATURES,
                    "latent_dim": 32,
                },
            ),
        ],
    )
    def test_rebuild_construct(self, model_cls, ctor_kwargs, arch):
        """rebuild_model reconstructs each model family in eval mode."""
        model = model_cls(NUM_FEATURES, **ctor_kwargs)
        rebuilt = rebuild_model(arch, model.state_dict())
        assert isinstance(rebuilt, model_cls)
        assert not rebuilt.training

    def test_rebuild_maf(self):
        """rebuild_model reconstructs a MAF flow model."""
        from syng_bts.model_factory import rebuild_model

        # Train a tiny flow to get valid state_dict
        trained = TestModelReconstructionParity._train_test_model(
            pd.DataFrame(
                np.random.RandomState(42).rand(NUM_SAMPLES, NUM_FEATURES) * 10,
                columns=[f"gene_{i}" for i in range(NUM_FEATURES)],
            ),
            "maf",
        )
        rebuilt = rebuild_model(trained.arch_params, trained.model_state)
        assert not rebuilt.training

    def test_rebuild_unknown_family_raises(self):
        """rebuild_model raises ValueError for unknown family."""
        from syng_bts.model_factory import rebuild_model

        with pytest.raises(ValueError, match="Unknown model family"):
            rebuild_model({"family": "rnn", "modelname": "X"}, {})

    def test_rebuild_unknown_ae_model_raises(self):
        """rebuild_model raises ValueError for unknown AE model."""
        from syng_bts.model_factory import rebuild_model

        with pytest.raises(ValueError, match="Unknown AE-family model"):
            rebuild_model(
                {"family": "ae", "modelname": "XVAE", "num_features": 10},
                {},
            )

    def test_rebuild_unknown_flow_model_raises(self):
        """rebuild_model raises ValueError for unknown flow model."""
        from syng_bts.model_factory import rebuild_model

        with pytest.raises(ValueError, match="Unknown flow model"):
            rebuild_model(
                {
                    "family": "flow",
                    "modelname": "superflow",
                    "num_inputs": 10,
                    "num_blocks": 1,
                    "num_hidden": 32,
                },
                {},
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("model_str", ["AE", "GAN", "maf"])
    def test_factory_vs_syngresult_cached_model_parity(self, sample_data, model_str):
        """Factory output matches SyngResult cached-model output under fixed seed."""
        from syng_bts import generate
        from syng_bts.helper_training import TrainedModel
        from syng_bts.inference import run_generation
        from syng_bts.model_factory import rebuild_model

        result = generate(
            data=sample_data,
            model=model_str,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=SEED,
            verbose="silent",
        )

        cached_model = result._resolve_model()
        arch_params = result.metadata["arch_params"]
        rebuilt_model = rebuild_model(arch_params, result.model_state)

        trained_cached = TrainedModel(
            model=cached_model,
            model_state=result.model_state,
            arch_params=arch_params,
            log_dict={},
            epochs_trained=result.metadata.get("epochs_trained", 0),
        )
        trained_rebuilt = TrainedModel(
            model=rebuilt_model,
            model_state=result.model_state,
            arch_params=arch_params,
            log_dict={},
            epochs_trained=result.metadata.get("epochs_trained", 0),
        )

        torch.manual_seed(101)
        gen_cached = run_generation(trained_cached, num_samples=60)
        torch.manual_seed(101)
        gen_rebuilt = run_generation(trained_rebuilt, num_samples=60)

        assert gen_cached.shape == gen_rebuilt.shape
        assert torch.allclose(gen_cached, gen_rebuilt)
