"""
Tests for experiment functions including integration training tests.
"""

import pandas as pd
import pytest

from syng_bts import generate, pilot_study, transfer
from syng_bts.result import SyngResult


class TestExperimentImports:
    """Test experiment functions are importable."""

    def test_generate_import(self):
        """Test generate can be imported."""
        from syng_bts import generate

        assert generate is not None
        assert callable(generate)

    def test_pilot_study_import(self):
        """Test pilot_study can be imported."""

        assert pilot_study is not None
        assert callable(pilot_study)

    def test_transfer_import(self):
        """Test transfer can be imported."""

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

    def test_generate_vae_training(
        self, temp_dir, sample_csv_file, small_training_config
    ):
        """Test generate() trains VAE on small data."""
        config = small_training_config
        data_path = sample_csv_file

        result = generate(
            data=data_path,
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
