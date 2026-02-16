"""
Tests for experiment functions including integration training tests.

Tests cover:
- generate() returns SyngResult with correct structure
- pilot_study() returns PilotResult with correct runs dict
- transfer() pre-trains, fine-tunes, and returns appropriate result type
- Internal helpers: _build_loss_df, _parse_model_spec, _compute_new_size
- Column names preservation, output_dir persistence
- Legacy names are removed
"""

import inspect

import pandas as pd
import pytest
import torch

from syng_bts import generate, pilot_study, transfer
from syng_bts.data_utils import resolve_data
from syng_bts.experiments import (
    _build_loss_df,
    _compute_new_size,
    _parse_model_spec,
    _resolve_early_stopping_config,
)
from syng_bts.helper_training import TrainedModel
from syng_bts.result import PilotResult, SyngResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FEATURES = 50
FAST_EPOCHS = 2
BATCH_FRAC = 0.5
LR = 0.001


# =========================================================================
# Import tests
# =========================================================================
class TestExperimentImports:
    """Test experiment functions are importable and legacy names are removed."""

    def test_generate_import(self):
        from syng_bts import generate

        assert generate is not None
        assert callable(generate)

    def test_pilot_study_import(self):
        from syng_bts import pilot_study

        assert pilot_study is not None
        assert callable(pilot_study)

    def test_transfer_import(self):
        from syng_bts import transfer

        assert transfer is not None
        assert callable(transfer)

    def test_legacy_names_not_importable(self):
        """Legacy function names should not be importable."""
        with pytest.raises(ImportError):
            from syng_bts import PilotExperiment  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import ApplyExperiment  # noqa: F401

        with pytest.raises(ImportError):
            from syng_bts import TransferExperiment  # noqa: F401

    def test_helper_training_module_exists(self):
        from syng_bts import helper_training

        assert helper_training is not None

    def test_helper_utils_module_exists(self):
        from syng_bts import helper_utils

        assert helper_utils is not None

    def test_helper_models_module_exists(self):
        from syng_bts import helper_models

        assert helper_models is not None

    def test_generate_signature_removes_save_model(self):
        sig = inspect.signature(generate)
        assert "save_model" not in sig.parameters

    def test_generate_signature_removes_pre_model(self):
        """generate() no longer exposes a pre_model parameter (Phase 2)."""
        sig = inspect.signature(generate)
        assert "pre_model" not in sig.parameters

    def test_pilot_study_signature_removes_pre_model(self):
        """pilot_study() no longer exposes a pre_model parameter (Phase 2)."""
        sig = inspect.signature(pilot_study)
        assert "pre_model" not in sig.parameters


# =========================================================================
# _parse_model_spec
# =========================================================================
class TestParseModelSpec:
    """Unit tests for _parse_model_spec."""

    def test_vae_with_kl(self):
        assert _parse_model_spec("VAE1-10") == ("VAE", 10)

    def test_ae_with_kl(self):
        assert _parse_model_spec("AE1-1") == ("AE", 1)

    def test_cvae_with_kl(self):
        assert _parse_model_spec("CVAE1-20") == ("CVAE", 20)

    def test_plain_model(self):
        assert _parse_model_spec("GAN") == ("GAN", 1)

    def test_flow_model(self):
        assert _parse_model_spec("maf") == ("maf", 1)


# =========================================================================
# _build_loss_df
# =========================================================================
class TestBuildLossDf:
    """Unit tests for _build_loss_df."""

    def test_ae_family(self):
        log = {
            "val_kl_loss_per_batch": [1.0, 0.5],
            "val_reconstruction_loss_per_batch": [2.0, 1.5],
        }
        df = _build_loss_df(log, "VAE")
        assert list(df.columns) == ["kl", "recons"]
        assert len(df) == 2

    def test_ae_plain(self):
        """AE models log train/val total loss, not kl/recons split."""
        log = {
            "train_loss_per_batch": [100.0, 90.0],
            "val_loss_per_batch": [110.0, 95.0],
        }
        df = _build_loss_df(log, "AE")
        assert list(df.columns) == ["train_loss", "val_loss"]
        assert len(df) == 2

    def test_cvae_family_train_fallback(self):
        log = {
            "train_kl_loss_per_batch": [1.0],
            "train_reconstruction_loss_per_batch": [2.0],
        }
        df = _build_loss_df(log, "CVAE")
        assert list(df.columns) == ["kl", "recons"]
        assert len(df) == 1

    def test_gan_family(self):
        log = {
            "train_discriminator_loss_per_batch": [3.0, 2.5],
            "train_generator_loss_per_batch": [1.0, 0.8],
        }
        df = _build_loss_df(log, "GAN")
        assert list(df.columns) == ["discriminator", "generator"]

    def test_flow(self):
        log = {"train_loss_per_epoch": [10.0, 9.0, 8.0]}
        df = _build_loss_df(log, "maf")
        assert list(df.columns) == ["train_loss"]
        assert len(df) == 3

    def test_wgan_family(self):
        log = {
            "train_discriminator_loss_per_batch": [3.0],
            "train_generator_loss_per_batch": [1.0],
        }
        df = _build_loss_df(log, "WGAN")
        assert list(df.columns) == ["discriminator", "generator"]

    def test_wgangp_family(self):
        log = {
            "train_discriminator_loss_per_batch": [3.0],
            "train_generator_loss_per_batch": [1.0],
        }
        df = _build_loss_df(log, "WGANGP")
        assert list(df.columns) == ["discriminator", "generator"]

    def test_realnvp_flow(self):
        log = {"train_loss_per_epoch": [5.0, 4.0]}
        df = _build_loss_df(log, "realnvp")
        assert list(df.columns) == ["train_loss"]

    def test_glow_flow(self):
        log = {"train_loss_per_epoch": [5.0]}
        df = _build_loss_df(log, "glow")
        assert list(df.columns) == ["train_loss"]


# =========================================================================
# Result schema verification
# =========================================================================
class TestResultSchema:
    """Verify result structure contracts documented in result_structure.md.

    Each test trains a model for 2 epochs and checks that the returned
    SyngResult has the correct loss columns, generated shape, and
    reconstruction presence/absence for that model family.
    """

    def test_ae_schema(self, sample_data):
        """AE: loss=[train_loss, val_loss], has reconstruction, columns match."""
        result = generate(
            data=sample_data,
            model="AE",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        # Loss columns
        assert list(result.loss.columns) == ["train_loss", "val_loss"]
        assert len(result.loss) > 0
        # Generated shape and column names
        assert result.generated_data.shape == (5, NUM_FEATURES)
        assert list(result.generated_data.columns) == list(sample_data.columns)
        # Reconstruction present
        assert result.reconstructed_data is not None
        assert result.reconstructed_data.shape[1] == NUM_FEATURES
        # Metadata
        assert result.metadata["modelname"] == "AE"

    def test_vae_schema(self, sample_data):
        """VAE: loss=[kl, recons], has reconstruction, columns match."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert list(result.loss.columns) == ["kl", "recons"]
        assert len(result.loss) > 0
        assert result.generated_data.shape == (5, NUM_FEATURES)
        assert list(result.generated_data.columns) == list(sample_data.columns)
        assert result.reconstructed_data is not None
        assert result.reconstructed_data.shape[1] == NUM_FEATURES
        assert result.metadata["modelname"] == "VAE"
        assert result.metadata["kl_weight"] == 10

    def test_gan_schema(self, sample_data):
        """GAN: loss=[discriminator, generator], no reconstruction."""
        result = generate(
            data=sample_data,
            model="GAN",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert list(result.loss.columns) == ["discriminator", "generator"]
        assert len(result.loss) > 0
        assert result.generated_data.shape == (5, NUM_FEATURES)
        assert list(result.generated_data.columns) == list(sample_data.columns)
        assert result.reconstructed_data is None
        assert result.metadata["modelname"] == "GAN"

    def test_flow_schema(self, sample_data):
        """Flow (maf): loss=[train_loss], no reconstruction."""
        result = generate(
            data=sample_data,
            model="maf",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert list(result.loss.columns) == ["train_loss"]
        assert len(result.loss) > 0
        assert result.generated_data.shape == (5, NUM_FEATURES)
        assert list(result.generated_data.columns) == list(sample_data.columns)
        assert result.reconstructed_data is None
        assert result.metadata["modelname"] == "maf"

    def test_metadata_keys_present(self, sample_data):
        """All standard metadata keys populated for a basic generate() call."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        expected_keys = {
            "model",
            "modelname",
            "dataname",
            "num_epochs",
            "epochs_trained",
            "seed",
            "kl_weight",
            "input_shape",
        }
        assert expected_keys.issubset(result.metadata.keys()), (
            f"Missing metadata keys: {expected_keys - result.metadata.keys()}"
        )
        assert result.metadata["input_shape"] == (20, NUM_FEATURES)

    def test_model_state_present(self, sample_data):
        """model_state is a non-empty dict after training."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result.model_state, dict)
        assert len(result.model_state) > 0

    def test_column_order_preservation(self, sample_data):
        """Column order is preserved across all DataFrames (original, generated, reconstructed).

        This test verifies that column order is maintained throughout the entire
        data pipeline, from input through tensor operations to final DataFrames.
        """
        # Use AE model which produces reconstruction, and apply_log to test inverse transform
        result = generate(
            data=sample_data,
            model="AE",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            apply_log=True,  # Test inverse transform path
        )

        # Get expected column order from input
        expected_cols = list(sample_data.columns)

        # Verify original_data preserves column order
        assert result.original_data is not None
        assert list(result.original_data.columns) == expected_cols, (
            "original_data column order differs from input"
        )

        # Verify generated_data preserves column order
        assert list(result.generated_data.columns) == expected_cols, (
            "generated_data column order differs from input"
        )

        # Verify reconstructed_data preserves column order
        assert result.reconstructed_data is not None
        assert list(result.reconstructed_data.columns) == expected_cols, (
            "reconstructed_data column order differs from input"
        )

        # Verify all three match each other
        assert list(result.generated_data.columns) == list(result.original_data.columns)
        assert list(result.reconstructed_data.columns) == list(
            result.original_data.columns
        )

    def test_pilot_result_schema(self, sample_data):
        """PilotResult.runs has correct keys and each value is a SyngResult."""
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, PilotResult)
        assert len(result.runs) == 5
        for (psize, draw), run in result.runs.items():
            assert psize == 10
            assert 1 <= draw <= 5
            assert isinstance(run, SyngResult)
            assert list(run.loss.columns) == ["kl", "recons"]
            assert list(run.generated_data.columns) == list(sample_data.columns)


# =========================================================================
# Early stopping logic
# =========================================================================
class TestResolveEarlyStoppingConfig:
    """Unit tests for _resolve_early_stopping_config() helper function.

    Directly tests the four-way interaction between epoch and
    early_stop_patience parameters without training models.
    """

    def test_both_epoch_and_patience(self):
        """epoch + early_stop_patience → early_stop=True, num_epochs=epoch."""
        num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
            epoch=500,
            early_stop_patience=2,
            default_max_epochs=1000,
            default_patience=30,
        )
        assert num_epochs == 500
        assert early_stop is True
        assert early_stop_num == 2

    def test_epoch_only(self):
        """epoch only → early_stop=False, num_epochs=epoch."""
        num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
            epoch=500,
            early_stop_patience=None,
            default_max_epochs=1000,
            default_patience=30,
        )
        assert num_epochs == 500
        assert early_stop is False
        assert early_stop_num == 30  # Default patience still returned

    def test_patience_only(self):
        """early_stop_patience only → early_stop=True, num_epochs=default_max."""
        num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
            epoch=None,
            early_stop_patience=2,
            default_max_epochs=1000,
            default_patience=30,
        )
        assert num_epochs == 1000
        assert early_stop is True
        assert early_stop_num == 2

    def test_neither_epoch_nor_patience(self):
        """Neither → early_stop=True, num_epochs=default_max, patience=default."""
        num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
            epoch=None,
            early_stop_patience=None,
            default_max_epochs=1000,
            default_patience=30,
        )
        assert num_epochs == 1000
        assert early_stop is True
        assert early_stop_num == 30

    def test_custom_defaults(self):
        """Helper respects custom default values."""
        num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
            epoch=None,
            early_stop_patience=None,
            default_max_epochs=500,
            default_patience=15,
        )
        assert num_epochs == 500
        assert early_stop is True
        assert early_stop_num == 15

    def test_epoch_zero_raises(self):
        """epoch must be a positive integer when provided."""
        with pytest.raises(ValueError, match="epoch must be a positive integer"):
            _resolve_early_stopping_config(
                epoch=0,
                early_stop_patience=5,
                default_max_epochs=1000,
                default_patience=30,
            )

    def test_negative_epoch_raises(self):
        with pytest.raises(ValueError, match="epoch must be a positive integer"):
            _resolve_early_stopping_config(
                epoch=-1,
                early_stop_patience=None,
                default_max_epochs=1000,
                default_patience=30,
            )

    def test_non_positive_patience_raises(self):
        with pytest.raises(
            ValueError, match="early_stop_patience must be a positive integer"
        ):
            _resolve_early_stopping_config(
                epoch=None,
                early_stop_patience=0,
                default_max_epochs=1000,
                default_patience=30,
            )

    def test_non_integer_epoch_raises(self):
        with pytest.raises(ValueError, match="epoch must be a positive integer"):
            _resolve_early_stopping_config(
                epoch=1.5,
                early_stop_patience=None,
                default_max_epochs=1000,
                default_patience=30,
            )


# =========================================================================
# _compute_new_size
# =========================================================================
class TestComputeNewSize:
    """Unit tests for _compute_new_size."""

    def test_single_group(self):
        labels = torch.zeros(20)
        assert _compute_new_size(labels, 20, 500) == 500

    def test_already_list(self):
        labels = torch.zeros(20)
        assert _compute_new_size(labels, 20, [100, 200]) == [100, 200]

    def test_unbalanced_two_groups(self):
        labels = torch.tensor([0.0] * 12 + [1.0] * 8)
        result = _compute_new_size(labels, 20, 500, repli=5)
        assert result == [12, 8, 5]

    def test_balanced_two_groups(self):
        labels = torch.tensor([0.0] * 10 + [1.0] * 10)
        assert _compute_new_size(labels, 20, 500) == 500


# =========================================================================
# generate()
# =========================================================================
class TestGenerate:
    """Integration tests for generate() — fast config (2 epochs)."""

    def test_returns_syng_result(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)

    def test_generated_columns_match_input(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert list(result.generated_data.columns) == list(sample_data.columns)

    def test_generated_row_count(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=15,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert len(result.generated_data) == 15

    def test_loss_is_dataframe(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result.loss, pd.DataFrame)
        assert len(result.loss) > 0

    def test_no_files_without_output_dir(self, sample_data, temp_dir):
        """generate() without output_dir should not write files."""
        before = set(temp_dir.rglob("*"))
        generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        after = set(temp_dir.rglob("*"))
        assert before == after

    def test_saves_with_output_dir(self, sample_data, temp_dir):
        """generate() with output_dir should persist files."""
        generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            output_dir=str(temp_dir),
        )
        files = list(temp_dir.rglob("*.csv"))
        assert len(files) > 0, "Expected CSV files in output_dir"

    def test_from_csv_path(self, sample_csv_file):
        result = generate(
            data=sample_csv_file,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert result.generated_data.shape[1] == NUM_FEATURES

    def test_from_bundled_dataset(self):
        """generate() accepts a bundled dataset name."""
        result = generate(
            data="SKCMPositive_4",
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert len(result.generated_data) == 10

    def test_metadata_populated(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert result.metadata["modelname"] == "VAE"
        assert result.metadata["kl_weight"] == 10

    def test_epochs_trained_metadata_within_bounds(self, sample_data):
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=6,
            early_stop_patience=1,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert result.metadata["epochs_trained"] > 0
        assert result.metadata["epochs_trained"] <= result.metadata["num_epochs"]

    def test_epochs_trained_uses_training_output_value(self, sample_data, monkeypatch):
        import torch.nn as nn

        import syng_bts.experiments as exp

        fake_model = nn.Linear(NUM_FEATURES, NUM_FEATURES)

        def fake_train_model(*args, **kwargs):
            return TrainedModel(
                model=fake_model,
                model_state=fake_model.state_dict(),
                arch_params={
                    "family": "ae",
                    "modelname": "AE",
                    "num_features": NUM_FEATURES,
                    "latent_size": 64,
                },
                log_dict={
                    "train_loss_per_batch": [1.0],
                    "val_loss_per_batch": [1.0],
                },
                epochs_trained=2,
            )

        def fake_infer(trained, *, new_size, rawdata, rawlabels, batch_size, cap=False):
            gen = torch.zeros((5, NUM_FEATURES), dtype=torch.float32)
            recon = torch.zeros((10, NUM_FEATURES), dtype=torch.float32)
            return gen, recon

        monkeypatch.setattr(exp, "_train_model", fake_train_model)
        monkeypatch.setattr(exp, "_infer_from_trained", fake_infer)

        result = generate(
            data=sample_data,
            model="AE",
            new_size=5,
            epoch=8,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )

        assert result.metadata["num_epochs"] == 8
        assert result.metadata["epochs_trained"] == 2

    def test_gan_model(self, sample_data):
        """generate() works with GAN model."""
        result = generate(
            data=sample_data,
            model="GAN",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert len(result.generated_data) == 10

    def test_early_stop_default(self, sample_data):
        """epoch=None enables early stopping with patience."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=5,
            epoch=None,
            early_stop_patience=2,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)

    def test_unsupported_model_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unsupported model"):
            generate(
                data=sample_data,
                model="UNKNOWN",
                new_size=5,
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )

    def test_generate_rejects_metadata_columns(self, sample_data):
        """generate() rejects user DataFrames with metadata-like columns."""
        bad = sample_data.copy()
        bad["groups"] = 0

        with pytest.raises(ValueError, match="metadata column"):
            generate(
                data=bad,
                model="VAE1-10",
                new_size=5,
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )

    def test_generate_rejects_non_numeric_columns(self, sample_data):
        """generate() rejects user DataFrames with non-numeric columns."""
        bad = sample_data.copy()
        bad["sample_name"] = "A"

        with pytest.raises(ValueError, match="non-numeric"):
            generate(
                data=bad,
                model="VAE1-10",
                new_size=5,
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )

    def test_generate_rejects_non_binary_groups(self, sample_data):
        """generate() enforces binary-group scope in v3.1."""
        groups = pd.Series(["A", "B", "C", "A", "B"] * 4)

        with pytest.raises(ValueError, match="supports only binary groups"):
            generate(
                data=sample_data,
                groups=groups,
                model="VAE1-10",
                new_size=5,
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )

    def test_generate_with_dataframe_input(self):
        """generate() works with a DataFrame loaded from bundled dataset."""
        from syng_bts import resolve_data

        df, _groups = resolve_data("SKCMPositive_4")
        result = generate(
            data=df,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)
        assert isinstance(result.generated_data, pd.DataFrame)
        assert len(result.generated_data) == 10
        # Column count should match (possibly minus metadata columns like 'samples')
        assert result.generated_data.shape[1] > 0

    def test_generated_data_is_dataframe_with_shape(self, sample_data):
        """generated_data is a DataFrame with correct shape and column names."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=8,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result.generated_data, pd.DataFrame)
        assert result.generated_data.shape == (8, NUM_FEATURES)
        assert list(result.generated_data.columns) == list(sample_data.columns)

    def test_save_writes_to_disk(self, sample_data, temp_dir):
        """SyngResult.save() writes correct files to output_dir."""
        result = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        paths = result.save(temp_dir)
        assert "generated" in paths
        assert paths["generated"].exists()
        # Roundtrip: saved CSV should match generated_data
        df_back = pd.read_csv(paths["generated"])
        pd.testing.assert_frame_equal(
            df_back, result.generated_data, atol=1e-6, check_dtype=False
        )

    def test_cvae_generated_data_feature_only(self, sample_data):
        """CVAE generated_data is feature-only; labels in metadata."""
        groups = pd.Series([0] * 10 + [1] * 10)
        result = generate(
            data=sample_data,
            model="CVAE1-10",
            new_size=10,
            groups=groups,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        # generated_data should match original columns (no label)
        assert list(result.generated_data.columns) == list(sample_data.columns)
        assert "label" not in result.generated_data.columns
        # Labels stored in metadata
        assert result.metadata["generated_labels"] is not None
        assert len(result.metadata["generated_labels"]) == len(result.generated_data)
        # Non-CVAE models have None labels
        result2 = generate(
            data=sample_data,
            model="VAE1-10",
            new_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert result2.metadata["generated_labels"] is None


# =========================================================================
# pilot_study()
# =========================================================================
class TestPilotStudy:
    """Integration tests for pilot_study()."""

    def test_returns_pilot_result(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, PilotResult)

    def test_pilot_study_rejects_metadata_columns(self, sample_data):
        """pilot_study() rejects user DataFrames with metadata-like columns."""
        bad = sample_data.copy()
        bad["samples"] = "TCGA-XX"

        with pytest.raises(ValueError, match="metadata column"):
            pilot_study(
                data=bad,
                pilot_size=[10],
                model="VAE1-10",
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )

    def test_runs_count(self, sample_data):
        """5 draws per pilot size."""
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert len(result.runs) == 5  # 1 pilot × 5 draws

    def test_multiple_pilot_sizes(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[8, 12],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert len(result.runs) == 10  # 2 pilots × 5 draws

    def test_run_keys_are_tuples(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        for key in result.runs:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert key[0] == 10
            assert key[1] in range(1, 6)

    def test_each_run_is_syng_result(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        for run in result.runs.values():
            assert isinstance(run, SyngResult)

    def test_generated_columns_match_input(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        for run in result.runs.values():
            assert list(run.generated_data.columns) == list(sample_data.columns)

    def test_cvae_generated_data_excludes_label_column(self, sample_data):
        """CVAE generated_data is feature-only; labels in metadata."""
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="CVAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        for run in result.runs.values():
            # generated_data should match original columns (no label)
            assert list(run.generated_data.columns) == list(sample_data.columns)
            assert "label" not in run.generated_data.columns
            # Labels stored in metadata
            assert "generated_labels" in run.metadata
            assert run.metadata["generated_labels"] is not None
            assert len(run.metadata["generated_labels"]) == len(run.generated_data)

    def test_pilot_run_metadata_includes_epochs_trained(self, sample_data):
        result = pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        for run in result.runs.values():
            assert "epochs_trained" in run.metadata
            assert run.metadata["epochs_trained"] > 0
            assert run.metadata["epochs_trained"] <= run.metadata["num_epochs"]

    def test_saves_with_output_dir(self, sample_data, temp_dir):
        pilot_study(
            data=sample_data,
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            output_dir=str(temp_dir),
        )
        files = list(temp_dir.rglob("*.csv"))
        assert len(files) > 0

    def test_pilot_study_with_bundled_data(self):
        """pilot_study() works with a bundled dataset."""
        result = pilot_study(
            data="SKCMPositive_4",
            pilot_size=[10],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, PilotResult)
        assert len(result.runs) == 5

    def test_explicit_groups_override_bundled_groups(self):
        """Explicit groups take precedence over bundled groups."""
        grouped_df, _ = resolve_data("BRCASubtypeSel")
        groups_all_zero = pd.Series(["class0"] * len(grouped_df))

        result = pilot_study(
            data="BRCASubtypeSel",
            groups=groups_all_zero,
            pilot_size=[8],
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )

        for run in result.runs.values():
            assert run.metadata["pilot_size"] == 8

    def test_pilot_study_rejects_non_binary_groups(self, sample_data):
        """pilot_study() enforces binary-group scope in v3.1."""
        groups = pd.Series(["A", "B", "C", "A", "B"] * 4)

        with pytest.raises(ValueError, match="supports only binary groups"):
            pilot_study(
                data=sample_data,
                groups=groups,
                pilot_size=[10],
                model="VAE1-10",
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
            )


# =========================================================================
# transfer()
# =========================================================================
class TestTransfer:
    """Integration tests for transfer()."""

    def test_returns_syng_result(self, sample_data):
        """transfer() without pilot_size returns SyngResult."""
        result = transfer(
            source_data=sample_data,
            target_data=sample_data,
            model="VAE1-10",
            new_size=10,
            source_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)

    def test_returns_pilot_result(self, sample_data):
        """transfer() with pilot_size returns PilotResult."""
        result = transfer(
            source_data=sample_data,
            target_data=sample_data,
            pilot_size=[10],
            source_size=10,
            model="VAE1-10",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, PilotResult)

    def test_transfer_without_output_dir_returns_result(self, sample_data):
        """transfer() without output_dir still returns a valid result."""
        result = transfer(
            source_data=sample_data,
            target_data=sample_data,
            model="VAE1-10",
            new_size=5,
            source_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)

    def test_saves_with_output_dir(self, sample_data, temp_dir):
        transfer(
            source_data=sample_data,
            target_data=sample_data,
            model="VAE1-10",
            new_size=5,
            source_size=5,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            output_dir=str(temp_dir),
        )
        # Should have Transfer subdir and output files
        assert (temp_dir / "Transfer").exists()

    def test_transfer_with_bundled_data(self):
        """transfer() works with bundled dataset names."""
        result = transfer(
            source_data="BRCA",
            target_data="PRAD",
            model="VAE1-10",
            new_size=10,
            source_size=10,
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
        )
        assert isinstance(result, SyngResult)

    def test_transfer_forwards_groups_and_apply_log(self, sample_data, monkeypatch):
        """transfer() forwards new group and apply_log args to downstream calls."""
        import syng_bts.experiments as exp

        observed: dict[str, object] = {}
        source_calls: list[dict[str, object]] = []

        def fake_transfer_target_generate(**kwargs):
            source_calls.append(kwargs)
            observed["source_groups"] = kwargs.get("target_groups")
            return SyngResult(
                generated_data=pd.DataFrame([[0.0]]),
                loss=pd.DataFrame({"train_loss": [1.0]}),
                metadata={"model": "AE", "dataname": "x", "seed": 1},
                model_state={"fake": True},
            )

        def fake_transfer_target_pilot(**kwargs):
            observed["target_groups_pilot"] = kwargs.get("target_groups")
            observed["pilot_apply_log"] = kwargs.get("apply_log")
            return PilotResult(runs={})

        monkeypatch.setattr(
            exp, "_transfer_target_generate", fake_transfer_target_generate
        )
        monkeypatch.setattr(exp, "_transfer_target_pilot", fake_transfer_target_pilot)

        source_groups = pd.Series(["S0"] * len(sample_data))
        target_groups = pd.Series(["T0"] * len(sample_data))

        result = transfer(
            source_data=sample_data,
            target_data=sample_data,
            source_groups=source_groups,
            target_groups=target_groups,
            pilot_size=[10],
            apply_log=False,
            model="AE",
            epoch=1,
        )

        assert isinstance(result, PilotResult)
        assert observed["source_groups"] is source_groups
        assert observed["target_groups_pilot"] is target_groups
        assert observed["pilot_apply_log"] is False
        assert len(source_calls) == 1

    def test_transfer_does_not_call_public_generate_or_pilot_study(
        self, sample_data, monkeypatch
    ):
        """transfer() uses internal orchestration helpers, not public APIs."""
        import syng_bts.experiments as exp

        def fail_generate(*args, **kwargs):
            raise AssertionError("transfer() should not call public generate()")

        def fail_pilot_study(*args, **kwargs):
            raise AssertionError("transfer() should not call public pilot_study()")

        def fake_transfer_target_generate(**kwargs):
            return SyngResult(
                generated_data=pd.DataFrame([[0.0]]),
                loss=pd.DataFrame({"train_loss": [1.0]}),
                metadata={"model": "AE", "dataname": "x", "seed": 1},
                model_state={"fake": True},
            )

        monkeypatch.setattr(exp, "generate", fail_generate)
        monkeypatch.setattr(exp, "pilot_study", fail_pilot_study)
        monkeypatch.setattr(
            exp, "_transfer_target_generate", fake_transfer_target_generate
        )

        result = transfer(
            source_data=sample_data,
            target_data=sample_data,
            model="AE",
            source_size=5,
            new_size=5,
            epoch=1,
        )

        assert isinstance(result, SyngResult)


# =========================================================================
# Slow integration tests
# =========================================================================
@pytest.mark.slow
class TestSlowTraining:
    """Integration tests for training with larger data or more epochs.

    These tests are marked as slow because they use more resources.
    Run with: pytest -m slow
    """

    def test_generate_vae_training(
        self, temp_dir, sample_csv_file, small_training_config
    ):
        """Test generate() trains VAE and returns correct result."""
        config = small_training_config

        result = generate(
            data=sample_csv_file,
            apply_log=False,
            new_size=10,
            model="VAE1-10",
            batch_frac=config["batch_frac"],
            learning_rate=config["learning_rate"],
            epoch=config["epoch"],
            random_seed=config["random_seed"],
            output_dir=temp_dir,
        )

        assert isinstance(result, SyngResult)
        assert isinstance(result.generated_data, pd.DataFrame)
        assert len(result.generated_data) == 10
        # Verify output files were created
        assert temp_dir.exists()

    def test_generate_creates_output(self, temp_dir, sample_csv_file):
        """Test that generate() creates output when output_dir is set."""
        output_dir = temp_dir / "test_output"

        result = generate(
            data=sample_csv_file,
            apply_log=False,
            new_size=5,
            model="VAE1-10",
            batch_frac=0.5,
            learning_rate=0.001,
            epoch=2,
            random_seed=42,
            output_dir=output_dir,
        )

        assert isinstance(result, SyngResult)
        assert output_dir.exists()
        assert output_dir.is_dir()
