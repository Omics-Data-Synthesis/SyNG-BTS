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

    def test_plot_loss_separate_figures(self, sample_result):
        """Test plot_loss returns a dict of figures, one per loss column."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = sample_result.plot_loss()
        assert isinstance(figs, dict)
        # sample_result has loss columns "kl" and "recons"
        assert set(figs.keys()) == set(sample_result.loss.columns)
        for name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"Expected Figure for {name}"
            plt.close(fig)

    def test_plot_loss_x_axis_iterations(self, sample_result):
        """Test plot_loss with x_axis='iterations' uses Iterations label."""
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        figs = sample_result.plot_loss(x_axis="iterations")
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == "Iterations"
            plt.close(fig)

    def test_plot_loss_x_axis_epochs(self, sample_generated):
        """Test plot_loss with x_axis='epochs' maps x to epoch space."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from syng_bts import SyngResult

        loss = pd.DataFrame({"kl": np.random.rand(500), "recons": np.random.rand(500)})
        result = SyngResult(
            generated_data=sample_generated,
            loss=loss,
            metadata={"model": "VAE1-10", "num_epochs": 10},
        )
        figs = result.plot_loss(x_axis="epochs")
        assert set(figs.keys()) == {"kl", "recons"}
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == "Epochs"
            plt.close(fig)

    def test_plot_loss_invalid_x_axis(self, sample_result):
        """Test plot_loss raises ValueError for unsupported x_axis."""
        with pytest.raises(ValueError, match="x_axis must be"):
            sample_result.plot_loss(x_axis="batches")

    def test_plot_loss_missing_num_epochs_for_epochs_axis(self, sample_result):
        """Test plot_loss raises ValueError when x_axis='epochs' but num_epochs missing."""
        with pytest.raises(ValueError, match="num_epochs"):
            sample_result.plot_loss(x_axis="epochs")

    def test_plot_loss_non_numeric_num_epochs_for_epochs_axis(self, sample_generated):
        """Test plot_loss raises ValueError when num_epochs is non-numeric."""
        from syng_bts import SyngResult

        loss = pd.DataFrame({"loss": np.random.rand(20)})
        result = SyngResult(
            generated_data=sample_generated,
            loss=loss,
            metadata={"num_epochs": "ten"},
        )
        with pytest.raises(ValueError, match="num_epochs"):
            result.plot_loss(x_axis="epochs")

    def test_plot_loss_invalid_running_average_window(self, sample_result):
        """Test plot_loss raises ValueError when running_average_window <= 0."""
        with pytest.raises(ValueError, match="running_average_window must be > 0"):
            sample_result.plot_loss(running_average_window=0)
        with pytest.raises(ValueError, match="running_average_window must be > 0"):
            sample_result.plot_loss(running_average_window=-5)

    def test_plot_loss_window_larger_than_series(self, sample_generated):
        """Test plot_loss raises ValueError when window exceeds series length."""
        from syng_bts import SyngResult

        short_loss = pd.DataFrame({"loss": [1.0, 0.9, 0.8]})
        result = SyngResult(
            generated_data=sample_generated,
            loss=short_loss,
        )
        with pytest.raises(ValueError, match="larger than"):
            result.plot_loss(running_average_window=100)

    def test_plot_loss_ylim_scaling(self, sample_generated):
        """Test plot_loss applies y-axis scaling to ignore the initial spike."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from syng_bts import SyngResult

        # Create a loss series with a huge initial spike
        vals = np.concatenate([np.array([1000.0, 800.0]), np.random.rand(200) * 5])
        loss = pd.DataFrame({"loss": vals})
        result = SyngResult(
            generated_data=sample_generated,
            loss=loss,
            metadata={"model": "AE"},
        )
        figs = result.plot_loss(running_average_window=10)
        fig = figs["loss"]
        ax = fig.get_axes()[0]
        ylim = ax.get_ylim()
        # The upper ylim should be much less than the spike (1000)
        assert ylim[1] < 100, (
            f"y-axis upper limit {ylim[1]} too high; spike not ignored"
        )
        plt.close(fig)

    def test_plot_heatmap_generated(self, sample_result):
        """Test plot_heatmap returns a heatmap figure for generated data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_result.plot_heatmap("generated")
        assert isinstance(fig, plt.Figure)
        # Title should mention "Generated"
        ax = fig.get_axes()[0]
        assert "Generated" in ax.get_title()
        plt.close(fig)

    def test_plot_heatmap_reconstructed(self, sample_result_full):
        """Test plot_heatmap returns a heatmap figure for reconstructed data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_result_full.plot_heatmap("reconstructed")
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()
        # seaborn heatmaps produce axes; just check it returned a figure
        assert len(ax) >= 1
        plt.close(fig)

    def test_plot_heatmap_reconstructed_missing(self, sample_result):
        """Test plot_heatmap raises ValueError when reconstructed data is None."""
        with pytest.raises(ValueError, match="No reconstructed data"):
            sample_result.plot_heatmap("reconstructed")

    def test_plot_heatmap_invalid_which(self, sample_result):
        """Test plot_heatmap raises ValueError for unknown 'which' value."""
        with pytest.raises(ValueError, match="Unknown value"):
            sample_result.plot_heatmap("invalid")

    def test_plot_heatmap_default(self, sample_result):
        """Test plot_heatmap defaults to 'generated'."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_result.plot_heatmap()
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
                        "num_epochs": 10,
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

    def test_plot_loss_separate(self, pilot_result):
        """Test plot_loss returns nested dict of per-column figures per run."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = pilot_result.plot_loss()
        assert isinstance(figs, dict)
        assert len(figs) == 4  # 2 pilot sizes × 2 draws
        for key, col_figs in figs.items():
            assert isinstance(col_figs, dict), f"Expected dict for run {key}"
            # Each run should have per-column figures
            for col, fig in col_figs.items():
                assert isinstance(fig, plt.Figure), f"Expected Figure for {key}/{col}"
                plt.close(fig)

    def test_plot_loss_aggregate_per_column(self, pilot_result):
        """Test plot_loss(aggregate=True) returns one figure per loss column."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = pilot_result.plot_loss(aggregate=True)
        assert isinstance(figs, dict)
        # loss columns are "kl" and "recons"
        assert set(figs.keys()) == {"kl", "recons"}
        for _col, fig in figs.items():
            assert isinstance(fig, plt.Figure)
            # Should have at least one axis
            assert len(fig.get_axes()) >= 1
            plt.close(fig)

    def test_plot_loss_aggregate_x_axis_epochs(self, pilot_result):
        """Test aggregate plot_loss supports x_axis='epochs'."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = pilot_result.plot_loss(aggregate=True, x_axis="epochs")
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == "Epochs"
            plt.close(fig)


class TestSaveLoadRoundtrip:
    """Tests for SyngResult and PilotResult save/load round-trip."""

    def test_save_writes_metadata_json(self, sample_result_full, temp_dir):
        """Test save produces a metadata JSON file."""
        paths = sample_result_full.save(temp_dir)
        assert "metadata" in paths
        assert paths["metadata"].exists()
        assert paths["metadata"].suffix == ".json"

    def test_metadata_json_format(self, sample_result_full, temp_dir):
        """Test metadata JSON is valid JSON with expected keys."""
        import json

        paths = sample_result_full.save(temp_dir)
        content = paths["metadata"].read_text(encoding="utf-8")
        meta = json.loads(content)
        assert isinstance(meta, dict)
        assert meta["model"] == "VAE1-10"
        assert meta["dataname"] == "test_full"
        assert meta["seed"] == 123

    def test_metadata_json_tuple_serialization(
        self, sample_generated, sample_loss, temp_dir
    ):
        """Test tuples in metadata are serialised as lists in JSON."""
        import json

        from syng_bts import SyngResult

        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"model": "AE", "input_shape": (20, 50)},
        )
        paths = result.save(temp_dir)
        content = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert content["input_shape"] == [20, 50]

    def test_save_no_metadata_when_empty(self, sample_generated, sample_loss, temp_dir):
        """Test save skips metadata JSON when metadata is empty dict."""
        from syng_bts import SyngResult

        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={},
        )
        paths = result.save(temp_dir)
        assert "metadata" not in paths

    def test_load_roundtrip(self, sample_result_full, temp_dir):
        """Test save → load round-trip preserves all fields."""
        from syng_bts import SyngResult

        sample_result_full.save(temp_dir)
        loaded = SyngResult.load(temp_dir)

        pd.testing.assert_frame_equal(
            loaded.generated_data,
            sample_result_full.generated_data,
            atol=1e-6,
        )
        pd.testing.assert_frame_equal(
            loaded.loss,
            sample_result_full.loss,
            atol=1e-6,
        )
        assert loaded.reconstructed_data is not None
        pd.testing.assert_frame_equal(
            loaded.reconstructed_data,
            sample_result_full.reconstructed_data,
            atol=1e-6,
        )
        assert loaded.model_state is not None
        assert loaded.metadata["model"] == sample_result_full.metadata["model"]
        assert loaded.metadata["dataname"] == sample_result_full.metadata["dataname"]

        assert loaded.metadata["seed"] == sample_result_full.metadata["seed"]
        assert (
            loaded.metadata["epochs_trained"]
            == sample_result_full.metadata["epochs_trained"]
        )

    def test_load_auto_prefix(self, sample_result, temp_dir):
        """Test load auto-detects prefix from a single generated CSV."""
        from syng_bts import SyngResult

        sample_result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert loaded.generated_data.shape == sample_result.generated_data.shape

    def test_load_with_explicit_prefix(self, sample_result_full, temp_dir):
        """Test load with an explicit prefix."""
        from syng_bts import SyngResult

        paths = sample_result_full.save(temp_dir)
        # Derive the stem from the generated file name
        stem = paths["generated"].name.removesuffix("_generated.csv")
        loaded = SyngResult.load(temp_dir, prefix=stem)
        assert loaded.metadata["model"] == "VAE1-10"

    def test_load_missing_optional_files(self, sample_generated, sample_loss, temp_dir):
        """Test load works when optional files are absent."""
        from syng_bts import SyngResult

        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "minimal"},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert loaded.reconstructed_data is None
        assert loaded.model_state is None

    def test_load_auto_prefix_ambiguous(self, sample_generated, sample_loss, temp_dir):
        """Test load raises ValueError when multiple generated files exist."""
        from syng_bts import SyngResult

        r1 = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "data_a", "model": "AE"},
        )
        r2 = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "data_b", "model": "AE"},
        )
        r1.save(temp_dir)
        r2.save(temp_dir)
        with pytest.raises(ValueError, match="Multiple generated files"):
            SyngResult.load(temp_dir)

    def test_load_pilot_output_directory_requires_prefix(
        self, sample_generated, sample_loss, temp_dir
    ):
        """Test loading a pilot-output directory requires explicit run prefix."""
        from syng_bts import PilotResult, SyngResult

        runs = {
            (10, 1): SyngResult(
                generated_data=sample_generated.copy(),
                loss=sample_loss.copy(),
                metadata={"model": "VAE1-10", "dataname": "pilot_ds"},
            ),
            (10, 2): SyngResult(
                generated_data=sample_generated.copy(),
                loss=sample_loss.copy(),
                metadata={"model": "VAE1-10", "dataname": "pilot_ds"},
            ),
        }
        PilotResult(runs=runs, metadata={"dataname": "pilot_ds"}).save(temp_dir)

        with pytest.raises(ValueError, match="PilotResult outputs"):
            SyngResult.load(temp_dir)

    def test_load_missing_generated_raises(self, temp_dir):
        """Test load raises FileNotFoundError when no generated CSV exists."""
        from syng_bts import SyngResult

        with pytest.raises(FileNotFoundError, match="generated"):
            SyngResult.load(temp_dir)

    def test_load_missing_loss_raises(self, sample_generated, temp_dir):
        """Test load raises FileNotFoundError when loss CSV is missing."""
        from syng_bts import SyngResult

        # Write only a generated CSV (no loss)
        gen_path = temp_dir / "test_AE_generated.csv"
        sample_generated.to_csv(gen_path, index=False)
        with pytest.raises(FileNotFoundError, match="loss"):
            SyngResult.load(temp_dir)

    def test_load_input_shape_restored_as_tuple(
        self, sample_generated, sample_loss, temp_dir
    ):
        """Test that input_shape is restored as a tuple after round-trip."""
        from syng_bts import SyngResult

        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test", "input_shape": (20, 50)},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert isinstance(loaded.metadata["input_shape"], tuple)
        assert loaded.metadata["input_shape"] == (20, 50)
