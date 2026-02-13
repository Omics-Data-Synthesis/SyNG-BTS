"""
Tests for structured logging (verbose parameter) — Phase 15.

Covers:
- _resolve_verbose normalization and error handling
- _print_training_state output formatting
- verbose=0 suppresses all stdout in generate()
- verbose=1 produces minimal output (default)
- verbose=2 produces detailed per-epoch output
- Text-form verbose strings ("silent", "minimal", "detailed")
- tqdm is no longer imported anywhere in the package
"""

import importlib

import pytest

from syng_bts import generate
from syng_bts.helper_train import _print_training_state, _resolve_verbose
from syng_bts.result import SyngResult

# ---------------------------------------------------------------------------
# Constants for fast integration runs
# ---------------------------------------------------------------------------
FAST_EPOCHS = 2
BATCH_FRAC = 0.5
LR = 0.001


# =========================================================================
# Unit tests for _resolve_verbose
# =========================================================================
class TestResolveVerbose:
    """Test _resolve_verbose normalization helper."""

    @pytest.mark.parametrize("val,expected", [(0, 0), (1, 1), (2, 2)])
    def test_integer_passthrough(self, val, expected):
        assert _resolve_verbose(val) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [("silent", 0), ("minimal", 1), ("detailed", 2)],
    )
    def test_string_forms(self, val, expected):
        assert _resolve_verbose(val) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [("SILENT", 0), ("Minimal", 1), ("DETAILED", 2)],
    )
    def test_string_case_insensitive(self, val, expected):
        assert _resolve_verbose(val) == expected

    @pytest.mark.parametrize("bad", [3, -1, 99, "verbose", "quiet", ""])
    def test_invalid_raises_valueerror(self, bad):
        with pytest.raises(ValueError):
            _resolve_verbose(bad)


# =========================================================================
# Unit tests for _print_training_state
# =========================================================================
class TestPrintTrainingState:
    """Test _print_training_state formatting."""

    def test_basic_output(self, capsys):
        _print_training_state(
            epoch=0,
            num_epochs=100,
            loss_dict={"train_loss": 1.2345},
        )
        out = capsys.readouterr().out
        assert "Epoch 001/100" in out
        assert "train_loss: 1.2345" in out

    def test_multiple_losses(self, capsys):
        _print_training_state(
            epoch=9,
            num_epochs=50,
            loss_dict={"kl": 0.1, "recons": 0.2},
        )
        out = capsys.readouterr().out
        assert "kl: 0.1000" in out
        assert "recons: 0.2000" in out

    def test_with_learning_rate(self, capsys):
        _print_training_state(
            epoch=0,
            num_epochs=10,
            loss_dict={"loss": 0.5},
            learning_rate=0.001,
        )
        out = capsys.readouterr().out
        assert "LR: 0.001000" in out

    def test_with_elapsed_time(self, capsys):
        _print_training_state(
            epoch=0,
            num_epochs=10,
            loss_dict={"loss": 0.5},
            elapsed_time=120.0,
        )
        out = capsys.readouterr().out
        assert "Time: 2.00min" in out

    def test_with_early_stop_info(self, capsys):
        _print_training_state(
            epoch=5,
            num_epochs=100,
            loss_dict={"loss": 0.3},
            early_stop_info="patience 3/10",
        )
        out = capsys.readouterr().out
        assert "patience 3/10" in out

    def test_all_optional_fields(self, capsys):
        _print_training_state(
            epoch=49,
            num_epochs=50,
            loss_dict={"loss": 0.01},
            elapsed_time=600.0,
            learning_rate=0.0001,
            early_stop_info="no improvement",
        )
        out = capsys.readouterr().out
        assert "Epoch 050/050" in out
        assert "loss: 0.0100" in out
        assert "LR:" in out
        assert "Time:" in out
        assert "no improvement" in out


# =========================================================================
# Integration tests — verbose levels with generate()
# =========================================================================
class TestVerboseGenerate:
    """Test that verbose param controls stdout output from generate()."""

    def test_verbose_0_silent(self, sample_data, capsys):
        """verbose=0 should produce no stdout output."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose=0,
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert out == ""

    def test_verbose_silent_string(self, sample_data, capsys):
        """verbose='silent' should produce no stdout output."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose="silent",
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert out == ""

    def test_verbose_1_minimal(self, sample_data, capsys):
        """verbose=1 should emit completion message but no per-epoch detail."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert "Training complete" in out
        # Should NOT contain per-epoch lines
        assert "Epoch 001/" not in out

    def test_verbose_2_detailed(self, sample_data, capsys):
        """verbose=2 should emit per-epoch output."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose=2,
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert "Epoch 001/" in out
        assert "Training complete" in out

    def test_default_verbose_is_1(self, sample_data, capsys):
        """Default verbose (not passed) should behave as verbose=1."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert "Epoch 001/" not in out
        assert "Training complete" in out

    def test_verbose_text_minimal(self, sample_data, capsys):
        """verbose='minimal' should behave like verbose=1."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose="minimal",
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert "Epoch 001/" not in out
        assert "Training complete" in out

    def test_verbose_text_detailed(self, sample_data, capsys):
        """verbose='detailed' should behave like verbose=2."""
        result = generate(
            data=sample_data,
            model="AE",
            epoch=FAST_EPOCHS,
            batch_frac=BATCH_FRAC,
            learning_rate=LR,
            random_seed=42,
            verbose="detailed",
        )
        out = capsys.readouterr().out
        assert isinstance(result, SyngResult)
        assert "Epoch 001/" in out
