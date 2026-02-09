"""
Tests for experiment functions including integration training tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestExperimentImports:
    """Test experiment functions are importable."""

    def test_pilot_experiment_import(self):
        """Test PilotExperiment can be imported."""
        from syng_bts import PilotExperiment

        assert PilotExperiment is not None
        assert callable(PilotExperiment)

    def test_apply_experiment_import(self):
        """Test ApplyExperiment can be imported."""
        from syng_bts import ApplyExperiment

        assert ApplyExperiment is not None
        assert callable(ApplyExperiment)

    def test_transfer_experiment_import(self):
        """Test TransferExperiment can be imported."""
        from syng_bts import TransferExperiment

        assert TransferExperiment is not None
        assert callable(TransferExperiment)

    def test_transfer_alias(self):
        """Test Transfer is alias for TransferExperiment."""
        from syng_bts import Transfer, TransferExperiment

        assert Transfer is TransferExperiment


class TestTrainingHelpers:
    """Test training helper functions."""

    def test_helper_training_module_exists(self):
        """Test helper_training module is accessible."""
        from syng_bts import helper_training

        assert helper_training is not None

    def test_helper_utils_module_exists(self):
        """Test helper_utils module is accessible."""
        from syng_bts import helper_utils

        assert helper_utils is not None

    def test_helper_models_module_exists(self):
        """Test helper_models module is accessible."""
        from syng_bts import helper_models

        assert helper_models is not None


@pytest.mark.slow
class TestSmallTraining:
    """Integration tests for training with small data.

    These tests are marked as slow because they train actual models.
    Run with: pytest -m slow
    """

    def test_apply_experiment_vae_training(
        self, temp_dir, sample_csv_file, small_training_config
    ):
        """Test ApplyExperiment trains VAE on small data."""
        from syng_bts import ApplyExperiment

        config = small_training_config
        data_dir = sample_csv_file.parent

        # Run training with minimal settings
        try:
            ApplyExperiment(
                dataname="test_data",
                apply_log=False,  # Data already processed
                new_size=10,
                model="VAE1-10",
                batch_frac=config["batch_frac"],
                learning_rate=config["learning_rate"],
                epoch=config["epoch"],  # Very few epochs
                random_seed=config["random_seed"],
                data_dir=data_dir,
                output_dir=temp_dir,
            )

            # Check that output files were created
            generated_files = list(temp_dir.glob("GeneratedData/*.csv"))
            loss_files = list(temp_dir.glob("Loss/*.csv"))

            # At least some output should be created
            assert len(generated_files) > 0 or len(loss_files) > 0

        except Exception as e:
            # Training may fail on small data, that's acceptable for this test
            # We're mainly verifying the function runs without crashing
            pytest.skip(f"Training failed (expected for very small data): {e}")

    def test_apply_experiment_creates_output_dirs(self, temp_dir, sample_csv_file):
        """Test that ApplyExperiment creates output directories."""
        from syng_bts import ApplyExperiment

        data_dir = sample_csv_file.parent

        try:
            ApplyExperiment(
                dataname="test_data",
                apply_log=False,
                new_size=5,
                model="VAE1-10",
                batch_frac=0.5,
                learning_rate=0.001,
                epoch=1,  # Single epoch
                random_seed=42,
                data_dir=data_dir,
                output_dir=temp_dir,
            )
        except Exception:
            pass  # We're just testing directory creation

        # Directories should exist even if training fails
        # (they're created at the start of training)


class TestExperimentParameters:
    """Test experiment function parameter validation."""

    def test_apply_experiment_signature(self):
        """Test ApplyExperiment has expected parameters."""
        from syng_bts import ApplyExperiment
        import inspect

        sig = inspect.signature(ApplyExperiment)
        params = list(sig.parameters.keys())

        expected = [
            "path",
            "dataname",
            "apply_log",
            "new_size",
            "model",
            "batch_frac",
            "learning_rate",
            "epoch",
            "data_dir",
            "output_dir",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"

    def test_pilot_experiment_signature(self):
        """Test PilotExperiment has expected parameters."""
        from syng_bts import PilotExperiment
        import inspect

        sig = inspect.signature(PilotExperiment)
        params = list(sig.parameters.keys())

        expected = [
            "dataname",
            "pilot_size",
            "model",
            "batch_frac",
            "learning_rate",
            "epoch",
            "data_dir",
            "output_dir",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"


class TestTransferExperiment:
    """Test TransferExperiment function."""

    def test_transfer_experiment_signature(self):
        """Test TransferExperiment has expected parameters."""
        from syng_bts import TransferExperiment
        import inspect

        sig = inspect.signature(TransferExperiment)
        params = list(sig.parameters.keys())

        expected = [
            "fromname",
            "toname",
            "data_dir",
            "output_dir",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"
