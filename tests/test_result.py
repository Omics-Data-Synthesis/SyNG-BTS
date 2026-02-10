"""
Tests for SyngResult and PilotResult result objects.
"""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_loss():
    """Create a small loss DataFrame."""
    return pd.DataFrame(
        {
            "kl": np.random.rand(50) * 10,
            "recons": np.random.rand(50) * 5,
        }
    )


@pytest.fixture
def sample_generated(sample_data):
    """Create a generated-data DataFrame with column names and small deterministic noise.

    Adds Gaussian noise (mean=0, sd=0.1) to numeric columns to better simulate
    generated data as shown in the README example. Uses a fixed RNG seed for
    reproducibility.
    """
    rng = np.random.RandomState(42)
    gen = sample_data.copy()
    numeric_cols = gen.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        noise = rng.normal(0, 0.1, size=(len(gen), len(numeric_cols)))
        gen.loc[:, numeric_cols] = gen.loc[:, numeric_cols].values + noise
    return gen


@pytest.fixture
def sample_result(sample_generated, sample_loss):
    """Create a minimal SyngResult for testing."""
    from syng_bts import SyngResult

    return SyngResult(
        generated_data=sample_generated,
        loss=sample_loss,
        metadata={"model": "VAE1-10", "dataname": "test", "seed": 42},
    )


@pytest.fixture
def sample_result_full(sample_generated, sample_loss):
    """Create a SyngResult with all optional fields filled."""
    from syng_bts import SyngResult

    model_state = {"weight": torch.randn(10, 5)}
    recon = sample_generated.copy()

    return SyngResult(
        generated_data=sample_generated,
        loss=sample_loss,
        reconstructed_data=recon,
        model_state=model_state,
        metadata={
            "model": "VAE1-10",
            "dataname": "test_full",
            "seed": 123,
            "epochs_trained": 100,
        },
    )


class TestSyngResult:
    """Test SyngResult dataclass."""

    def test_construction_minimal(self, sample_generated, sample_loss):
        """Test creating a SyngResult with required fields only."""
        from syng_bts import SyngResult

        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
        )
        assert isinstance(result.generated_data, pd.DataFrame)
        assert isinstance(result.loss, pd.DataFrame)
        assert result.reconstructed_data is None
        assert result.model_state is None
        assert result.metadata == {}

    def test_construction_full(self, sample_result_full):
        """Test creating a SyngResult with all fields."""
        result = sample_result_full
        assert result.reconstructed_data is not None
        assert result.model_state is not None
        assert result.metadata["model"] == "VAE1-10"

    def test_repr(self, sample_result):
        """Test __repr__ is informative."""
        r = repr(sample_result)
        assert "SyngResult" in r
        assert "VAE1-10" in r
        assert "generated=" in r
        assert "loss_cols=" in r

    def test_repr_full(self, sample_result_full):
        """Test __repr__ shows optional fields."""
        r = repr(sample_result_full)
        assert "has_reconstructed=True" in r
        assert "has_model_state=True" in r

    def test_summary(self, sample_result):
        """Test summary returns a string with key info."""
        s = sample_result.summary()
        assert "VAE1-10" in s
        assert "samples" in s
        assert "features" in s

    def test_summary_full(self, sample_result_full):
        """Test summary with reconstructed data and seed."""
        s = sample_result_full.summary()
        assert "Reconstructed" in s
        assert "seed" in s.lower() or "Seed" in s

    def test_plot_loss(self, sample_result):
        """Test plot_loss returns a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig = sample_result.plot_loss()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_loss_short_series(self, sample_generated):
        """Test plot_loss with fewer data points than averaging window."""
        from syng_bts import SyngResult

        short_loss = pd.DataFrame({"loss": [1.0, 0.9, 0.8]})
        result = SyngResult(
            generated_data=sample_generated,
            loss=short_loss,
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = result.plot_loss(averaging_iterations=100)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_minimal(self, sample_result, temp_dir):
        """Test save writes generated and loss CSVs."""
        paths = sample_result.save(temp_dir)

        assert "generated" in paths
        assert "loss" in paths
        assert paths["generated"].exists()
        assert paths["loss"].exists()
        # Should not have reconstructed or model files
        assert "reconstructed" not in paths
        assert "model" not in paths

    def test_save_creates_directory(self, sample_result, temp_dir):
        """Test save creates the output directory if it doesn't exist."""
        out = temp_dir / "nested" / "output"
        paths = sample_result.save(out)
        assert out.exists()
        assert paths["generated"].exists()

    def test_save_full(self, sample_result_full, temp_dir):
        """Test save with all optional fields."""
        paths = sample_result_full.save(temp_dir)

        assert "generated" in paths
        assert "loss" in paths
        assert "reconstructed" in paths
        assert "model" in paths

        for p in paths.values():
            assert p.exists()

    def test_save_with_custom_prefix(self, sample_result, temp_dir):
        """Test save with a custom prefix."""
        paths = sample_result.save(temp_dir, prefix="custom")
        for p in paths.values():
            assert "custom" in p.name

    def test_save_uses_dataname_from_metadata(self, sample_result, temp_dir):
        """Test save infers prefix from metadata['dataname']."""
        paths = sample_result.save(temp_dir)
        # metadata has dataname="test"
        assert paths["generated"].name.startswith("test_")

    def test_saved_csv_has_headers(self, sample_result, temp_dir):
        """Test that saved CSVs include column headers."""
        paths = sample_result.save(temp_dir)
        # Read back generated data
        df = pd.read_csv(paths["generated"])
        assert list(df.columns) == list(sample_result.generated_data.columns)

    def test_saved_csv_roundtrip(self, sample_result, temp_dir):
        """Test CSV roundtrip preserves data."""
        paths = sample_result.save(temp_dir)
        df_back = pd.read_csv(paths["generated"])
        pd.testing.assert_frame_equal(
            df_back,
            sample_result.generated_data,
            atol=1e-6,
        )

    def test_saved_model_state_loadable(self, sample_result_full, temp_dir):
        """Test saved model state can be loaded back."""
        paths = sample_result_full.save(temp_dir)
        loaded = torch.load(paths["model"], weights_only=False)
        assert isinstance(loaded, dict)


class TestPilotResult:
    """Test PilotResult container."""

    @pytest.fixture
    def pilot_result(self, sample_generated, sample_loss):
        """Create a PilotResult with 2 pilot sizes × 2 draws."""
        from syng_bts import PilotResult, SyngResult

        runs = {}
        for ps in [10, 20]:
            for draw in [1, 2]:
                runs[(ps, draw)] = SyngResult(
                    generated_data=sample_generated.copy(),
                    loss=sample_loss.copy(),
                    metadata={
                        "model": "VAE1-10",
                        "dataname": "test",
                        "pilot_size": ps,
                        "draw": draw,
                    },
                )
        return PilotResult(
            runs=runs,
            metadata={"model": "VAE1-10", "dataname": "test"},
        )

    def test_construction(self, pilot_result):
        """Test PilotResult stores runs correctly."""
        assert len(pilot_result.runs) == 4
        assert (10, 1) in pilot_result.runs
        assert (20, 2) in pilot_result.runs

    def test_repr(self, pilot_result):
        """Test __repr__ shows key info."""
        r = repr(pilot_result)
        assert "PilotResult" in r
        assert "n_runs=4" in r
        assert "10" in r
        assert "20" in r

    def test_summary(self, pilot_result):
        """Test summary includes all runs."""
        s = pilot_result.summary()
        assert "PilotResult" in s
        assert "4 runs" in s
        assert "pilot=10" in s
        assert "pilot=20" in s

    def test_save(self, pilot_result, temp_dir):
        """Test save writes files for each run."""
        all_paths = pilot_result.save(temp_dir)
        assert len(all_paths) == 4
        for _key, paths in all_paths.items():
            assert "generated" in paths
            assert paths["generated"].exists()

    def test_save_filenames_encode_run(self, pilot_result, temp_dir):
        """Test saved filenames contain pilot size and draw index."""
        all_paths = pilot_result.save(temp_dir)
        gen_path = all_paths[(10, 1)]["generated"]
        assert "pilot10" in gen_path.name
        assert "draw1" in gen_path.name

    def test_plot_loss(self, pilot_result):
        """Test plot_loss returns a dict of figures."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = pilot_result.plot_loss()
        assert len(figs) == 4
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
