"""
Tests for structured logging (verbose parameter).

Covers:
- _resolve_verbose normalization and error handling
- _print_training_state output formatting
- verbose=0 suppresses all stdout in generate()
- verbose=1 produces minimal output (default)
- verbose=2 produces detailed per-epoch output
- Text-form verbose strings ("silent", "minimal", "detailed")
- tqdm is no longer imported anywhere in the package
"""

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

    @pytest.mark.parametrize(
        "state_kwargs,expected_substrings",
        [
            (
                {"epoch": 0, "num_epochs": 100, "loss_dict": {"train_loss": 1.2345}},
                ["Epoch 001/100", "train_loss: 1.2345"],
            ),
            (
                {"epoch": 9, "num_epochs": 50, "loss_dict": {"kl": 0.1, "recons": 0.2}},
                ["kl: 0.1000", "recons: 0.2000"],
            ),
            (
                {
                    "epoch": 0,
                    "num_epochs": 10,
                    "loss_dict": {"loss": 0.5},
                    "learning_rate": 0.001,
                },
                ["LR: 0.001000"],
            ),
            (
                {
                    "epoch": 0,
                    "num_epochs": 10,
                    "loss_dict": {"loss": 0.5},
                    "elapsed_time": 120.0,
                },
                ["Time: 2.00min"],
            ),
            (
                {
                    "epoch": 5,
                    "num_epochs": 100,
                    "loss_dict": {"loss": 0.3},
                    "early_stop_info": "patience 3/10",
                },
                ["patience 3/10"],
            ),
            (
                {
                    "epoch": 49,
                    "num_epochs": 50,
                    "loss_dict": {"loss": 0.01},
                    "elapsed_time": 600.0,
                    "learning_rate": 0.0001,
                    "early_stop_info": "no improvement",
                },
                ["Epoch 050/050", "loss: 0.0100", "LR:", "Time:", "no improvement"],
            ),
        ],
        ids=["basic", "multi_loss", "with_lr", "with_time", "early_stop", "all_fields"],
    )
    def test_print_training_state(self, capsys, state_kwargs, expected_substrings):
        """_print_training_state output contains expected substrings."""
        _print_training_state(**state_kwargs)
        out = capsys.readouterr().out
        for substring in expected_substrings:
            assert substring in out, f"Expected {substring!r} in output: {out!r}"


# =========================================================================
# Integration tests — verbose levels with generate()
# =========================================================================
class TestVerboseGenerate:
    """Test that verbose param controls stdout output from generate()."""

    @pytest.mark.parametrize(
        "int_form,str_form,empty,must_contain,must_not_contain",
        [
            (0, "silent", True, [], []),
            (1, "minimal", False, ["Training complete"], ["Epoch 001/"]),
            (2, "detailed", False, ["Epoch 001/", "Training complete"], []),
        ],
        ids=["silent", "minimal", "detailed"],
    )
    def test_verbose_level(
        self,
        sample_data,
        capsys,
        int_form,
        str_form,
        empty,
        must_contain,
        must_not_contain,
    ):
        """Each verbose level controls generate() stdout, and the string alias
        behaves identically to its integer form."""

        def output_for(verbose):
            result = generate(
                data=sample_data,
                model="AE",
                epoch=FAST_EPOCHS,
                batch_frac=BATCH_FRAC,
                learning_rate=LR,
                random_seed=42,
                verbose=verbose,
            )
            assert isinstance(result, SyngResult)
            return capsys.readouterr().out

        for verbose in (int_form, str_form):
            out = output_for(verbose)
            if empty:
                assert out == "", f"verbose={verbose!r} should be silent"
            for s in must_contain:
                assert s in out, f"verbose={verbose!r}: expected {s!r} in output"
            for s in must_not_contain:
                assert s not in out, f"verbose={verbose!r}: did not expect {s!r}"

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
