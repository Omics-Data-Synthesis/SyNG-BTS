"""
Tests for training orchestrator functions.

Covers:
- TrainedModel dataclass structure
- training_AEs() returns TrainedModel for AE, VAE, CVAE
- training_GANs() returns TrainedModel with correct fields
- training_flows() returns TrainedModel with correct fields
- training_iter() returns (data, labels) tuple
- train_GAN returns (log_dict, best_model) consistently
- No files written to disk during training
- No matplotlib figures created during training
- Old I/O parameters are removed from function signatures
"""

import inspect
import os

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from syng_bts.helper_training import (
    TrainedModel,
    training_AEs,
    training_flows,
    training_GANs,
    training_iter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FEATURES = 50
NUM_SAMPLES = 20
FAST_EPOCHS = 2
BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _use_agg_backend():
    """Force matplotlib to use non-interactive backend for all tests."""
    backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    yield
    matplotlib.use(backend)


@pytest.fixture
def raw_data():
    """Small raw data tensor for training."""
    torch.manual_seed(42)
    return torch.randn(NUM_SAMPLES, NUM_FEATURES).abs()


@pytest.fixture
def raw_labels():
    """Labels tensor (single column, two classes)."""
    labels = torch.zeros(NUM_SAMPLES, 1)
    labels[NUM_SAMPLES // 2 :] = 1.0
    return labels


@pytest.fixture
def raw_labels_single_class():
    """Labels tensor (single class, for AE/VAE)."""
    return torch.zeros(NUM_SAMPLES, 1)


# ===========================================================================
# TrainedModel
# ===========================================================================
class TestTrainedModel:
    """Verify TrainedModel dataclass structure."""

    def test_is_dataclass(self):
        from dataclasses import fields as dc_fields

        import torch.nn as nn

        model = nn.Linear(1, 1)
        t = TrainedModel(
            model=model,
            model_state=model.state_dict(),
            arch_params={"family": "ae"},
            log_dict={},
            epochs_trained=1,
        )
        names = {f.name for f in dc_fields(t)}
        assert names == {
            "model",
            "model_state",
            "arch_params",
            "log_dict",
            "epochs_trained",
        }


# ===========================================================================
# training_AEs
# ===========================================================================
class TestTrainingAEs:
    """Tests for the training_AEs() function."""

    def test_returns_trained_model_vae(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result, TrainedModel)

    def test_model_state_is_dict_vae(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result.model_state, dict)
        assert len(result.model_state) > 0

    def test_arch_params_vae(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert result.arch_params["family"] == "ae"
        assert result.arch_params["modelname"] == "VAE"
        assert "latent_size" in result.arch_params

    def test_log_dict_has_loss_keys_vae(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert "val_reconstruction_loss_per_batch" in result.log_dict
        assert "val_kl_loss_per_batch" in result.log_dict

    def test_returns_trained_model_ae(self, raw_data, raw_labels_single_class):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels_single_class,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="AE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result, TrainedModel)
        assert result.arch_params["modelname"] == "AE"

    def test_returns_trained_model_cvae(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="CVAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result, TrainedModel)
        assert result.arch_params["modelname"] == "CVAE"

    def test_epochs_trained(self, raw_data, raw_labels):
        result = training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert result.epochs_trained == FAST_EPOCHS

    def test_no_files_written(self, raw_data, raw_labels, tmp_path):
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            training_AEs(
                rawdata=raw_data,
                rawlabels=raw_labels,
                batch_size=BATCH_SIZE,
                random_seed=42,
                modelname="VAE",
                num_epochs=FAST_EPOCHS,
                learning_rate=0.001,
                early_stop=False,
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, raw_data, raw_labels):
        plt.close("all")
        figs_before = plt.get_fignums()
        training_AEs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="VAE",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        figs_after = plt.get_fignums()
        assert figs_before == figs_after


# ===========================================================================
# training_GANs
# ===========================================================================
class TestTrainingGANs:
    """Tests for the training_GANs() function."""

    def test_returns_trained_model(self, raw_data, raw_labels):
        result = training_GANs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="GAN",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result, TrainedModel)

    def test_arch_params(self, raw_data, raw_labels):
        result = training_GANs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="GAN",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert result.arch_params["family"] == "gan"
        assert result.arch_params["modelname"] == "GAN"
        assert "latent_dim" in result.arch_params

    def test_log_dict_has_gan_keys(self, raw_data, raw_labels):
        result = training_GANs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="GAN",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert "train_discriminator_loss_per_batch" in result.log_dict
        assert "train_generator_loss_per_batch" in result.log_dict

    def test_model_state_is_dict(self, raw_data, raw_labels):
        result = training_GANs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="GAN",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        assert isinstance(result.model_state, dict)
        assert len(result.model_state) > 0

    def test_no_files_written(self, raw_data, raw_labels, tmp_path):
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            training_GANs(
                rawdata=raw_data,
                rawlabels=raw_labels,
                batch_size=BATCH_SIZE,
                random_seed=42,
                modelname="GAN",
                num_epochs=FAST_EPOCHS,
                learning_rate=0.001,
                early_stop=False,
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, raw_data, raw_labels):
        plt.close("all")
        figs_before = plt.get_fignums()
        training_GANs(
            rawdata=raw_data,
            rawlabels=raw_labels,
            batch_size=BATCH_SIZE,
            random_seed=42,
            modelname="GAN",
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            early_stop=False,
        )
        figs_after = plt.get_fignums()
        assert figs_before == figs_after


# ===========================================================================
# training_flows
# ===========================================================================
class TestTrainingFlows:
    """Tests for the training_flows() function."""

    def test_returns_trained_model(self, raw_data):
        result = training_flows(
            rawdata=raw_data,
            batch_frac=0.5,
            valid_batch_frac=0.3,
            random_seed=42,
            modelname="maf",
            num_blocks=2,
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            num_hidden=32,
            early_stop=False,
            early_stop_num=30,
        )
        assert isinstance(result, TrainedModel)

    def test_arch_params(self, raw_data):
        result = training_flows(
            rawdata=raw_data,
            batch_frac=0.5,
            valid_batch_frac=0.3,
            random_seed=42,
            modelname="maf",
            num_blocks=2,
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            num_hidden=32,
            early_stop=False,
            early_stop_num=30,
        )
        assert result.arch_params["family"] == "flow"
        assert result.arch_params["modelname"] == "maf"
        assert "num_hidden" in result.arch_params

    def test_log_dict_has_train_loss(self, raw_data):
        result = training_flows(
            rawdata=raw_data,
            batch_frac=0.5,
            valid_batch_frac=0.3,
            random_seed=42,
            modelname="maf",
            num_blocks=2,
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            num_hidden=32,
            early_stop=False,
            early_stop_num=30,
        )
        assert "train_loss_per_epoch" in result.log_dict
        assert len(result.log_dict["train_loss_per_epoch"]) == FAST_EPOCHS

    def test_no_files_written(self, raw_data, tmp_path):
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            training_flows(
                rawdata=raw_data,
                batch_frac=0.5,
                valid_batch_frac=0.3,
                random_seed=42,
                modelname="maf",
                num_blocks=2,
                num_epochs=FAST_EPOCHS,
                learning_rate=0.001,
                num_hidden=32,
                early_stop=False,
                early_stop_num=30,
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, raw_data):
        plt.close("all")
        figs_before = plt.get_fignums()
        training_flows(
            rawdata=raw_data,
            batch_frac=0.5,
            valid_batch_frac=0.3,
            random_seed=42,
            modelname="maf",
            num_blocks=2,
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            num_hidden=32,
            early_stop=False,
            early_stop_num=30,
        )
        figs_after = plt.get_fignums()
        assert figs_before == figs_after

    def test_tensorboard_opt_in(self, raw_data, tmp_path):
        """When tensorboard_dir is set, TensorBoard files are created."""
        tb_dir = tmp_path / "tb_logs"
        training_flows(
            rawdata=raw_data,
            batch_frac=0.5,
            valid_batch_frac=0.3,
            random_seed=42,
            modelname="maf",
            num_blocks=2,
            num_epochs=FAST_EPOCHS,
            learning_rate=0.001,
            num_hidden=32,
            early_stop=False,
            early_stop_num=30,
            tensorboard_dir=str(tb_dir),
        )
        assert tb_dir.exists()
        assert len(list(tb_dir.iterdir())) > 0


# ===========================================================================
# training_iter
# ===========================================================================
class TestTrainingIter:
    """Tests for the training_iter() function."""

    def test_returns_tuple(self, raw_data, raw_labels_single_class):
        result = training_iter(
            iter_times=1,
            rawdata=raw_data,
            rawlabels=raw_labels_single_class,
            random_seed=42,
            modelname="AE",
            num_epochs=FAST_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_data_shape_doubles(self, raw_data, raw_labels_single_class):
        feed_data, feed_labels = training_iter(
            iter_times=1,
            rawdata=raw_data,
            rawlabels=raw_labels_single_class,
            random_seed=42,
            modelname="AE",
            num_epochs=FAST_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
        )
        assert feed_data.shape[0] == NUM_SAMPLES * 2
        assert feed_data.shape[1] == NUM_FEATURES

    def test_labels_match_data(self, raw_data, raw_labels_single_class):
        feed_data, feed_labels = training_iter(
            iter_times=1,
            rawdata=raw_data,
            rawlabels=raw_labels_single_class,
            random_seed=42,
            modelname="AE",
            num_epochs=FAST_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
        )
        assert feed_labels.shape[0] == feed_data.shape[0]

    def test_no_files_written(self, raw_data, raw_labels_single_class, tmp_path):
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            training_iter(
                iter_times=1,
                rawdata=raw_data,
                rawlabels=raw_labels_single_class,
                random_seed=42,
                modelname="AE",
                num_epochs=FAST_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=0.001,
            )
            assert len(list(tmp_path.iterdir())) == 0
        finally:
            os.chdir(orig_cwd)

    def test_no_figures_created(self, raw_data, raw_labels_single_class):
        plt.close("all")
        figs_before = plt.get_fignums()
        training_iter(
            iter_times=1,
            rawdata=raw_data,
            rawlabels=raw_labels_single_class,
            random_seed=42,
            modelname="AE",
            num_epochs=FAST_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
        )
        figs_after = plt.get_fignums()
        assert figs_before == figs_after


# ===========================================================================
# Removed parameters from training signatures
# ===========================================================================
class TestRemovedParams:
    """Verify old I/O parameters are removed from training orchestrators."""

    def test_no_savepathextend_param(self):
        sig = inspect.signature(training_iter)
        assert "savepathextend" not in sig.parameters
        assert "saveextend" not in sig.parameters
        assert "plot" not in sig.parameters

    def test_removed_params_from_training_aes(self):
        sig = inspect.signature(training_AEs)
        for param in (
            "savepath",
            "savepathnew",
            "save_recons",
            "save_new",
            "plot",
            "colnames",
            "new_size",
        ):
            assert param not in sig.parameters, f"{param} should be removed"

    def test_removed_params_from_training_gans(self):
        sig = inspect.signature(training_GANs)
        for param in ("savepathnew", "save_new", "plot", "new_size"):
            assert param not in sig.parameters, f"{param} should be removed"


# ===========================================================================
# train_GAN return type fix
# ===========================================================================
class TestTrainGANReturnType:
    """Verify train_GAN returns (log_dict, best_model) like WGAN/WGANGP."""

    def test_train_gan_returns_tuple(self, raw_data, raw_labels):
        from syng_bts import helper_train as ht
        from syng_bts.helper_models import GAN

        model = GAN(num_features=NUM_FEATURES, latent_dim=32)
        optim_gen = torch.optim.Adam(model.generator.parameters(), lr=0.001)
        optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=0.001)

        data = torch.utils.data.TensorDataset(raw_data, raw_labels)
        loader = torch.utils.data.DataLoader(
            data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )

        result = ht.train_GAN(
            num_epochs=FAST_EPOCHS,
            model=model,
            optimizer_gen=optim_gen,
            optimizer_discr=optim_discr,
            latent_dim=32,
            train_loader=loader,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        log_dict, best_model = result
        assert isinstance(log_dict, dict)
        assert hasattr(best_model, "state_dict")
