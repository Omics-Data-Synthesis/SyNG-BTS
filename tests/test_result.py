"""
Tests for SyngResult and PilotResult result objects.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from syng_bts.result import PilotResult, SyngResult


@pytest.fixture
def plt():
    """Provide pyplot on the Agg backend; close all figures on teardown.

    Collapses the repeated ``import matplotlib; matplotlib.use("Agg"); import
    matplotlib.pyplot as plt`` boilerplate that every plotting test needs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


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
    return SyngResult(
        generated_data=sample_generated,
        loss=sample_loss,
        metadata={
            "model": "VAE1-10",
            "dataname": "test",
            "seed": 42,
            "epochs_trained": 50,
        },
    )


@pytest.fixture
def sample_result_full(sample_generated, sample_loss):
    """Create a SyngResult with all optional fields filled."""
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

    def test_plot_loss_separate_figures(self, sample_result, plt):
        """Test plot_loss returns a dict of figures, one per loss column."""
        figs = sample_result.plot_loss()
        assert isinstance(figs, dict)
        # sample_result has loss columns "kl" and "recons"
        assert set(figs.keys()) == set(sample_result.loss.columns)
        for name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"Expected Figure for {name}"
            plt.close(fig)

    def test_plot_loss_x_axis_iterations(self, sample_result, plt):
        """Test plot_loss with x_axis='iterations' uses Iterations label."""
        figs = sample_result.plot_loss(x_axis="iterations")
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == "Iterations"
            plt.close(fig)

    def test_plot_loss_x_axis_epochs(self, sample_generated, plt):
        """Test plot_loss with x_axis='epochs' maps x to epoch space."""
        loss = pd.DataFrame({"kl": np.random.rand(500), "recons": np.random.rand(500)})
        result = SyngResult(
            generated_data=sample_generated,
            loss=loss,
            metadata={"model": "VAE1-10", "epochs_trained": 10},
        )
        figs = result.plot_loss(x_axis="epochs")
        assert set(figs.keys()) == {"kl", "recons"}
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == "Epochs"
            plt.close(fig)

    # Invalid-input cluster (sample_result cases)

    @pytest.mark.parametrize(
        "kwargs,exc_type,match",
        [
            ({"x_axis": "batches"}, ValueError, "x_axis must be"),
            (
                {"running_average_window": 0, "x_axis": "iterations"},
                ValueError,
                "running_average_window must be > 0",
            ),
            (
                {"running_average_window": -5, "x_axis": "iterations"},
                ValueError,
                "running_average_window must be > 0",
            ),
        ],
        ids=["invalid_x_axis", "window_zero", "window_negative"],
    )
    def test_plot_loss_invalid_args(self, sample_result, kwargs, exc_type, match):
        """Test plot_loss raises for invalid arguments (sample_result cases)."""
        with pytest.raises(exc_type, match=match):
            sample_result.plot_loss(**kwargs)

    @pytest.mark.parametrize(
        "loss,metadata,kwargs,match",
        [
            # x_axis='epochs' but epochs_trained missing from metadata
            (
                None,
                {"model": "VAE1-10"},
                {"x_axis": "epochs"},
                "epochs_trained",
            ),
            # epochs_trained present but non-numeric
            (
                pd.DataFrame({"loss": np.random.rand(20)}),
                {"epochs_trained": "ten"},
                {"x_axis": "epochs"},
                "epochs_trained",
            ),
            # running_average_window larger than the loss series
            (
                pd.DataFrame({"loss": [1.0, 0.9, 0.8]}),
                {"epochs_trained": 3},
                {"running_average_window": 100},
                "larger than",
            ),
        ],
        ids=["missing_epochs", "non_numeric_epochs", "window_larger_than_series"],
    )
    def test_plot_loss_invalid_metadata_raises(
        self, sample_generated, sample_loss, loss, metadata, kwargs, match
    ):
        """Test plot_loss raises for metadata/series-dependent invalid configs."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss if loss is None else loss,
            metadata=metadata,
        )
        with pytest.raises(ValueError, match=match):
            result.plot_loss(**kwargs)

    def test_plot_loss_ylim_scaling(self, sample_generated, plt):
        """Test plot_loss applies y-axis scaling to ignore the initial spike."""
        # Create a loss series with a huge initial spike
        vals = np.concatenate([np.array([1000.0, 800.0]), np.random.rand(200) * 5])
        loss = pd.DataFrame({"loss": vals})
        result = SyngResult(
            generated_data=sample_generated,
            loss=loss,
            metadata={"model": "AE", "epochs_trained": 100},
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

    # plot_heatmap valid which-variants

    @pytest.mark.parametrize(
        "which,fixture_name",
        [
            ("generated", "sample_result"),
            ("reconstructed", "sample_result_full"),
            (None, "sample_result"),
        ],
        ids=["generated", "reconstructed", "default"],
    )
    def test_plot_heatmap_which(self, request, which, fixture_name, plt):
        """Test plot_heatmap for valid 'which' values."""
        result = request.getfixturevalue(fixture_name)
        if which is None:
            fig = result.plot_heatmap()
        else:
            fig = result.plot_heatmap(which)
        assert isinstance(fig, plt.Figure)
        if which == "generated":
            ax = fig.get_axes()[0]
            assert "Generated" in ax.get_title()
        elif which == "reconstructed":
            assert len(fig.get_axes()) >= 1
        plt.close(fig)

    # plot_heatmap invalid / missing raises

    @pytest.mark.parametrize(
        "which,fixture_name,match",
        [
            ("invalid", "sample_result", "Unknown value"),
            ("reconstructed", "sample_result", "No reconstructed data"),
        ],
        ids=["invalid_which", "reconstructed_missing"],
    )
    def test_plot_heatmap_raises(self, request, which, fixture_name, match):
        """Test plot_heatmap raises ValueError for invalid or missing data."""
        result = request.getfixturevalue(fixture_name)
        with pytest.raises(ValueError, match=match):
            result.plot_heatmap(which)

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


class TestOriginalData:
    """Test original_data field on SyngResult and PilotResult."""

    def test_syng_result_original_data_default_none(
        self, sample_generated, sample_loss
    ):
        """SyngResult.original_data defaults to None."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={},
        )
        assert result.original_data is None

    def test_syng_result_with_original_data(
        self, sample_data, sample_generated, sample_loss
    ):
        """SyngResult stores original_data when provided."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={},
        )
        assert result.original_data is not None
        pd.testing.assert_frame_equal(result.original_data, sample_data)

    def test_save_writes_original_csv(
        self, sample_data, sample_generated, sample_loss, temp_dir
    ):
        """save() writes an _original.csv when original_data is present."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={"dataname": "test"},
        )
        paths = result.save(temp_dir)
        assert "original" in paths
        assert paths["original"].exists()

    def test_save_no_original_csv_when_none(
        self, sample_generated, sample_loss, temp_dir
    ):
        """save() does not write _original.csv when original_data is None."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test"},
        )
        paths = result.save(temp_dir)
        assert "original" not in paths

    def test_load_restores_original_data(
        self, sample_data, sample_generated, sample_loss, temp_dir
    ):
        """load() restores original_data from _original.csv."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={"dataname": "test"},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)

        assert loaded.original_data is not None
        pd.testing.assert_frame_equal(loaded.original_data, sample_data, atol=1e-6)

    def test_load_no_original_returns_none(
        self, sample_generated, sample_loss, temp_dir
    ):
        """load() sets original_data=None when no _original.csv exists."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test"},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert loaded.original_data is None

    def test_summary_includes_original_shape(
        self, sample_data, sample_generated, sample_loss
    ):
        """summary() mentions original-data shape when present."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={},
        )
        s = result.summary()
        assert "original" in s.lower()

    def test_repr_shows_has_original(self, sample_data, sample_generated, sample_loss):
        """__repr__ includes has_original=True when original_data is present."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={},
        )
        assert "has_original=True" in repr(result)

    def test_plot_heatmap_original(
        self, sample_data, sample_generated, sample_loss, plt
    ):
        """plot_heatmap(which='original') works."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            original_data=sample_data,
            metadata={},
        )
        fig = result.plot_heatmap(which="original")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_heatmap_original_missing_raises(self, sample_generated, sample_loss):
        """plot_heatmap(which='original') raises when no original_data."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={},
        )
        with pytest.raises(ValueError, match="[Oo]riginal"):
            result.plot_heatmap(which="original")

    def test_pilot_result_original_data(
        self, sample_data, sample_generated, sample_loss
    ):
        """PilotResult stores top-level original_data."""
        runs = {
            (10, 1): SyngResult(
                generated_data=sample_generated,
                loss=sample_loss,
                metadata={},
            )
        }
        pr = PilotResult(runs=runs, metadata={}, original_data=sample_data)
        assert pr.original_data is not None
        pd.testing.assert_frame_equal(pr.original_data, sample_data)

    def test_pilot_result_save_writes_original(
        self, sample_data, sample_generated, sample_loss, temp_dir
    ):
        """PilotResult.save() writes top-level _original.csv."""
        runs = {
            (10, 1): SyngResult(
                generated_data=sample_generated,
                loss=sample_loss,
                metadata={"dataname": "test", "model": "VAE1-10"},
            )
        }
        pr = PilotResult(
            runs=runs,
            metadata={"dataname": "test", "model": "VAE1-10"},
            original_data=sample_data,
        )
        pr.save(temp_dir)
        # Check that the top-level original CSV was written
        original_files = list(temp_dir.glob("*_original.csv"))
        assert len(original_files) >= 1


class TestPilotResult:
    """Test PilotResult container."""

    def test_no_load_method(self):
        """PilotResult intentionally does not provide a load() method."""
        assert not hasattr(PilotResult, "load")

    @pytest.fixture
    def pilot_result(self, sample_generated, sample_loss):
        """Create a PilotResult with 2 pilot sizes × 2 draws."""
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
                        "epochs_trained": 10,
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

    def test_plot_loss_per_run(self, pilot_result, plt):
        """Test plot_loss(style='per_run') returns nested per-run figures."""
        figs = pilot_result.plot_loss(style="per_run")
        assert isinstance(figs, dict)
        assert len(figs) == 4  # 2 pilot sizes × 2 draws
        for key, col_figs in figs.items():
            assert isinstance(col_figs, dict), f"Expected dict for run {key}"
            for col, fig in col_figs.items():
                assert isinstance(fig, plt.Figure), f"Expected Figure for {key}/{col}"
                plt.close(fig)

    def test_plot_loss_overlay_runs_is_default(self, pilot_result, plt):
        """Test that default style is 'overlay_runs'."""
        figs = pilot_result.plot_loss()
        assert isinstance(figs, dict)
        # Default should now be overlay_runs, which returns flattened dict
        assert set(figs.keys()) == {"kl", "recons"}
        for fig in figs.values():
            plt.close(fig)

    def test_plot_loss_overlay_runs(self, pilot_result, plt):
        """Test plot_loss(style='overlay_runs') overlays all runs per loss column."""
        figs = pilot_result.plot_loss(style="overlay_runs")
        assert isinstance(figs, dict)
        assert set(figs.keys()) == {"kl", "recons"}
        for col, fig in figs.items():
            assert isinstance(fig, plt.Figure)
            ax = fig.get_axes()[0]
            # Should have 4 lines (one per run)
            lines = ax.get_lines()
            assert len(lines) == 4, f"Expected 4 lines for {col}, got {len(lines)}"
            plt.close(fig)

    # overlay/mean_band x_axis variants

    @pytest.mark.parametrize(
        "style,x_axis,expected_xlabel",
        [
            ("overlay_runs", "epochs", "Epochs"),
            ("overlay_runs", "iterations", "Iterations"),
            ("mean_band", "epochs", "Epochs"),
        ],
        ids=[
            "overlay_runs_epochs",
            "overlay_runs_iterations",
            "mean_band_epochs",
        ],
    )
    def test_plot_loss_x_axis_variants(
        self, pilot_result, style, x_axis, expected_xlabel, plt
    ):
        """Test overlay_runs and mean_band plot_loss x_axis label variants."""
        figs = pilot_result.plot_loss(style=style, x_axis=x_axis)
        for fig in figs.values():
            ax = fig.get_axes()[0]
            assert ax.get_xlabel() == expected_xlabel
            plt.close(fig)

    def test_plot_loss_mean_band(self, pilot_result, plt):
        """Test plot_loss(style='mean_band') produces mean ± std plot."""
        from matplotlib.collections import PolyCollection

        figs = pilot_result.plot_loss(style="mean_band")
        assert isinstance(figs, dict)
        assert set(figs.keys()) == {"kl", "recons"}
        for col, fig in figs.items():
            assert isinstance(fig, plt.Figure)
            ax = fig.get_axes()[0]
            # Should have at least one line (smoothed mean)
            assert len(ax.get_lines()) >= 1, f"No mean line for {col}"
            # Should have a shaded band (PolyCollection from fill_between)
            polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
            assert len(polys) >= 1, f"No shaded std band for {col}"
            plt.close(fig)

    def _mean_band_max_x(self, sample_generated, truncate):
        """Build a 2-run PilotResult (lengths 100 and 50), plot a mean_band, and
        return the band's largest x-vertex (the run length it extends to)."""
        from matplotlib.collections import PolyCollection

        runs = {}
        for draw, length in [(1, 100), (2, 50)]:
            runs[(10, draw)] = SyngResult(
                generated_data=sample_generated.copy(),
                loss=pd.DataFrame({"loss": np.random.rand(length)}),
                metadata={"model": "AE", "epochs_trained": length, "dataname": "t"},
            )
        pr = PilotResult(runs=runs, metadata={"model": "AE"})
        figs = pr.plot_loss(
            style="mean_band",
            truncate=truncate,
            running_average_window=5,
            x_axis="iterations",
        )
        ax = figs["loss"].get_axes()[0]
        for child in ax.get_children():
            if isinstance(child, PolyCollection):
                paths = child.get_paths()
                return paths[0].vertices[:, 0].max() if paths else None
        return None

    def test_plot_loss_mean_band_truncate_true(self, sample_generated, plt):
        """truncate=True: band extends only to the shortest run length (50)."""
        max_x = self._mean_band_max_x(sample_generated, truncate=True)
        assert max_x is not None
        assert max_x <= 50

    def test_plot_loss_mean_band_truncate_false(self, sample_generated, plt):
        """truncate=False: band extends to the longest run length (~99)."""
        max_x = self._mean_band_max_x(sample_generated, truncate=False)
        assert max_x is not None
        assert max_x >= 90

    # PilotResult invalid-argument cluster

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"style": "fancy"}, "style must be one of"),
            ({"running_average_window": 0}, "running_average_window must be > 0"),
            ({"x_axis": "batches"}, "x_axis must be"),
        ],
        ids=["invalid_style", "invalid_window", "invalid_x_axis"],
    )
    def test_plot_loss_invalid_args(self, pilot_result, kwargs, match):
        """Test PilotResult.plot_loss raises ValueError for invalid arguments."""
        with pytest.raises(ValueError, match=match):
            pilot_result.plot_loss(**kwargs)

    # ylim_scaling for multi-run styles

    @pytest.mark.parametrize(
        "style,fig_key",
        [
            ("overlay_runs", "loss"),
            ("mean_band", "loss"),
        ],
        ids=["overlay_runs", "mean_band"],
    )
    def test_plot_loss_multirun_ylim_scaling(
        self, sample_generated, style, fig_key, plt
    ):
        """Test overlay_runs and mean_band styles apply y-axis spike suppression."""
        vals1 = np.concatenate([np.array([1000.0, 800.0]), np.random.rand(200) * 5])
        vals2 = np.concatenate([np.array([900.0, 700.0]), np.random.rand(200) * 5])

        runs = {
            (10, 1): SyngResult(
                generated_data=sample_generated.copy(),
                loss=pd.DataFrame({"loss": vals1}),
                metadata={"model": "AE", "epochs_trained": 100},
            ),
            (10, 2): SyngResult(
                generated_data=sample_generated.copy(),
                loss=pd.DataFrame({"loss": vals2}),
                metadata={"model": "AE", "epochs_trained": 100},
            ),
        }
        pr = PilotResult(runs=runs, metadata={"model": "AE"})

        figs = pr.plot_loss(style=style, running_average_window=10)
        fig = figs[fig_key]
        ylim = fig.get_axes()[0].get_ylim()
        assert ylim[1] < 100, (
            f"y-axis upper limit {ylim[1]} too high; spike not ignored"
        )
        plt.close(fig)

    def test_plot_loss_individual_ylim_scaling(self, sample_generated, plt):
        """Test style='per_run' applies y-axis spike suppression via SyngResult."""
        vals = np.concatenate([np.array([1000.0, 800.0]), np.random.rand(200) * 5])
        runs = {
            (10, 1): SyngResult(
                generated_data=sample_generated.copy(),
                loss=pd.DataFrame({"loss": vals}),
                metadata={"model": "AE", "epochs_trained": 100},
            )
        }
        pr = PilotResult(runs=runs, metadata={"model": "AE"})

        figs = pr.plot_loss(style="per_run", running_average_window=10)
        fig = figs[(10, 1)]["loss"]
        ylim = fig.get_axes()[0].get_ylim()
        assert ylim[1] < 100, (
            f"y-axis upper limit {ylim[1]} too high; spike not ignored"
        )
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
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={},
        )
        paths = result.save(temp_dir)
        assert "metadata" not in paths

    def test_load_roundtrip(self, sample_result_full, temp_dir):
        """Test save → load round-trip preserves all fields."""
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
        sample_result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert loaded.generated_data.shape == sample_result.generated_data.shape

    def test_load_with_explicit_prefix(self, sample_result_full, temp_dir):
        """Test load with an explicit prefix."""
        paths = sample_result_full.save(temp_dir)
        # Derive the stem from the generated file name
        stem = paths["generated"].name.removesuffix("_generated.csv")
        loaded = SyngResult.load(temp_dir, prefix=stem)
        assert loaded.metadata["model"] == "VAE1-10"

    def test_load_missing_optional_files(self, sample_generated, sample_loss, temp_dir):
        """Test load works when optional files are absent."""
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
        with pytest.raises(FileNotFoundError, match="generated"):
            SyngResult.load(temp_dir)

    def test_load_missing_loss_raises(self, sample_generated, temp_dir):
        """Test load raises FileNotFoundError when loss CSV is missing."""
        # Write only a generated CSV (no loss)
        gen_path = temp_dir / "test_AE_generated.csv"
        sample_generated.to_csv(gen_path, index=False)
        with pytest.raises(FileNotFoundError, match="loss"):
            SyngResult.load(temp_dir)

    def test_load_input_shape_restored_as_tuple(
        self, sample_generated, sample_loss, temp_dir
    ):
        """Test that input_shape is restored as a tuple after round-trip."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test", "input_shape": (20, 50)},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert isinstance(loaded.metadata["input_shape"], tuple)
        assert loaded.metadata["input_shape"] == (20, 50)


class TestGenerateNewSamples:
    """Tests for SyngResult.generate_new_samples()."""

    @pytest.fixture
    def trained_result(self, sample_data):
        """Create a real SyngResult from generate() for testing."""
        from syng_bts import generate

        return generate(
            data=sample_data,
            model="AE",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
        )

    @pytest.fixture
    def trained_result_vae(self, sample_data):
        """Create a VAE SyngResult for testing."""
        from syng_bts import generate

        return generate(
            data=sample_data,
            model="VAE1-10",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
        )

    @pytest.fixture
    def trained_result_gan(self, sample_data):
        """Create a GAN SyngResult for testing."""
        from syng_bts import generate

        return generate(
            data=sample_data,
            model="GAN",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
        )

    @pytest.fixture
    def trained_result_maf(self, sample_data):
        """Create a MAF SyngResult for testing."""
        from syng_bts import generate

        return generate(
            data=sample_data,
            model="maf",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
        )

    @pytest.fixture
    def trained_result_cvae(self, sample_data):
        """Create a CVAE SyngResult for testing."""
        from syng_bts import generate

        groups = np.array([0, 1] * (len(sample_data) // 2))
        return generate(
            data=sample_data,
            groups=groups,
            model="CVAE",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
        )

    # --- Mode semantics ---

    def test_mode_new_returns_new_result(self, trained_result):
        """mode='new' returns a distinct SyngResult."""
        original_gen = trained_result.generated_data.copy()
        new_result = trained_result.generate_new_samples(100, mode="new")

        assert isinstance(new_result, SyngResult)
        assert new_result is not trained_result
        assert new_result.generated_data.shape == (100, original_gen.shape[1])
        # Original unchanged
        pd.testing.assert_frame_equal(trained_result.generated_data, original_gen)
        # Metadata and loss copied
        assert new_result.metadata == trained_result.metadata
        assert new_result.model_state is trained_result.model_state

    def test_mode_overwrite_mutates_self(self, trained_result):
        """mode='overwrite' replaces generated_data on self."""
        original_shape = trained_result.generated_data.shape
        result = trained_result.generate_new_samples(200, mode="overwrite")

        assert result is trained_result
        assert result.generated_data.shape == (200, original_shape[1])

    def test_mode_append_concatenates(self, trained_result):
        """mode='append' appends new rows to generated_data."""
        original_n = trained_result.generated_data.shape[0]
        n_new = 150
        result = trained_result.generate_new_samples(n_new, mode="append")

        assert result is trained_result
        assert result.generated_data.shape[0] == original_n + n_new

    def test_mode_default_is_new(self, trained_result):
        """Default mode is 'new'."""
        new_result = trained_result.generate_new_samples(50)
        assert new_result is not trained_result
        assert new_result.generated_data.shape[0] == 50

    def test_invalid_mode_raises(self, trained_result):
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            trained_result.generate_new_samples(10, mode="replace")

    @pytest.mark.parametrize("invalid_n", [0, -1, True, 1.5])
    def test_invalid_n_raises(self, trained_result, invalid_n):
        """n must be a positive integer."""
        with pytest.raises(ValueError, match="positive integer"):
            trained_result.generate_new_samples(invalid_n)

    # --- Column names preservation ---

    def test_column_names_preserved(self, trained_result):
        """Generated data has same column names as original."""
        new_result = trained_result.generate_new_samples(50)
        assert list(new_result.generated_data.columns) == list(
            trained_result.generated_data.columns
        )

    # --- apply_log handling ---

    def test_apply_log_applied(self, sample_data):
        """When apply_log=True, inverse log is applied to new samples."""
        from syng_bts import generate

        result = generate(
            data=sample_data,
            model="AE",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
            apply_log=True,
        )
        assert result.metadata["apply_log"] is True
        new_result = result.generate_new_samples(50)
        # Values should be in count space (generally >= 0 for ReLU outputs)
        assert new_result.generated_data.shape == (50, sample_data.shape[1])

    def test_apply_log_false(self, sample_data):
        """When apply_log=False, no inverse transform applied."""
        from syng_bts import generate

        result = generate(
            data=sample_data,
            model="AE",
            epoch=2,
            batch_frac=0.5,
            learning_rate=0.001,
            random_seed=42,
            verbose="silent",
            apply_log=False,
        )
        assert result.metadata["apply_log"] is False
        new_result = result.generate_new_samples(50)
        assert new_result.generated_data.shape == (50, sample_data.shape[1])

    # per-model generate_new

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "trained_result",
            "trained_result_vae",
            "trained_result_gan",
            "trained_result_maf",
        ],
        ids=["ae", "vae", "gan", "maf"],
    )
    def test_generate_new_model_families(self, request, fixture_name):
        """generate_new_samples works for AE, VAE, GAN, and MAF models."""
        result = request.getfixturevalue(fixture_name)
        new_result = result.generate_new_samples(100)
        assert new_result.generated_data.shape[0] == 100
        assert new_result.generated_data.shape[1] == result.generated_data.shape[1]
        if fixture_name != "trained_result_maf":
            assert np.isfinite(new_result.generated_data.to_numpy()).all()

    # save/load then generate pair

    @pytest.mark.parametrize(
        "fixture_name,n",
        [
            ("trained_result", 100),
            ("trained_result_vae", 80),
        ],
        ids=["ae", "vae"],
    )
    def test_save_load_then_generate(self, request, fixture_name, n, temp_dir):
        """generate_new_samples works after save/load round-trip."""
        trained = request.getfixturevalue(fixture_name)
        trained.save(temp_dir)
        loaded = SyngResult.load(temp_dir)

        new_result = loaded.generate_new_samples(n)
        assert new_result.generated_data.shape[0] == n
        assert list(new_result.generated_data.columns) == list(
            trained.generated_data.columns
        )

    # --- Error cases ---

    def test_missing_model_state_raises(self, sample_data):
        """Raises ValueError when model_state is None."""
        result = SyngResult(
            generated_data=sample_data,
            loss=pd.DataFrame({"loss": [1.0]}),
            model_state=None,
            metadata={"arch_params": {"family": "ae", "modelname": "AE"}},
        )
        with pytest.raises(ValueError, match="model_state"):
            result.generate_new_samples(10)

    def test_missing_arch_params_raises(self, sample_data):
        """Raises ValueError when arch_params is missing from metadata."""
        result = SyngResult(
            generated_data=sample_data,
            loss=pd.DataFrame({"loss": [1.0]}),
            model_state={"some": "state"},
            metadata={"model": "AE"},
        )
        with pytest.raises(ValueError, match="arch_params"):
            result.generate_new_samples(10)

    # --- Cached model reuse ---

    def test_cached_model_reused(self, trained_result):
        """Second call reuses the lazy-cached model."""
        _ = trained_result.generate_new_samples(10)
        assert trained_result._cached_model is not None
        cached = trained_result._cached_model
        _ = trained_result.generate_new_samples(10)
        assert trained_result._cached_model is cached

    # --- New result has correct reconstructed/original fields ---

    def test_mode_new_copies_optional_fields(self, trained_result):
        """mode='new' copies reconstructed_data and original_data."""
        new_result = trained_result.generate_new_samples(50, mode="new")
        if trained_result.reconstructed_data is not None:
            assert new_result.reconstructed_data is not None
            pd.testing.assert_frame_equal(
                new_result.reconstructed_data, trained_result.reconstructed_data
            )
        if trained_result.original_data is not None:
            assert new_result.original_data is not None
            pd.testing.assert_frame_equal(
                new_result.original_data, trained_result.original_data
            )

    # CVAE label modes

    @pytest.mark.parametrize(
        "mode,n",
        [
            ("new", 40),
            ("overwrite", 30),
        ],
        ids=["new_mode", "overwrite_mode"],
    )
    def test_cvae_generated_labels_mode(self, trained_result_cvae, mode, n):
        """CVAE new/overwrite mode updates generated_labels metadata correctly."""
        result = trained_result_cvae.generate_new_samples(n, mode=mode)
        labels = result.metadata.get("generated_labels")
        assert isinstance(labels, pd.Series)
        assert len(labels) == result.generated_data.shape[0]

    def test_cvae_generated_labels_append_mode(self, trained_result_cvae):
        """CVAE append mode appends generated_labels consistently."""
        first = trained_result_cvae.generate_new_samples(20, mode="append")
        labels_1 = first.metadata.get("generated_labels")
        assert isinstance(labels_1, pd.Series)
        assert len(labels_1) == first.generated_data.shape[0]

        second = first.generate_new_samples(15, mode="append")
        labels_2 = second.metadata.get("generated_labels")
        assert isinstance(labels_2, pd.Series)
        assert len(labels_2) == second.generated_data.shape[0]


# =========================================================================
# Group attributes tests
# =========================================================================


class TestGroupAttributes:
    """Test group fields on SyngResult: original_groups, generated_groups, etc."""

    @pytest.fixture
    def sample_groups(self):
        """Binary group Series matching sample_data (20 rows)."""
        return pd.Series(["A"] * 10 + ["B"] * 10, name="group")

    @pytest.fixture
    def sample_result_with_groups(self, sample_generated, sample_loss, sample_groups):
        """SyngResult populated with all three group attributes."""
        return SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={
                "model": "VAE1-10",
                "dataname": "test_groups",
                "seed": 42,
                "epochs_trained": 50,
                "group_mapping": {0: "A", 1: "B"},
            },
            original_groups=sample_groups.copy(),
            generated_groups=sample_groups.copy(),
            reconstructed_groups=sample_groups.copy(),
        )

    def test_default_groups_are_none(self, sample_generated, sample_loss):
        """SyngResult group attributes default to None."""
        result = SyngResult(generated_data=sample_generated, loss=sample_loss)
        assert result.original_groups is None
        assert result.generated_groups is None
        assert result.reconstructed_groups is None

    def test_groups_set_on_construction(self, sample_result_with_groups):
        """Group attributes are populated when provided."""
        result = sample_result_with_groups
        assert result.original_groups is not None
        assert result.generated_groups is not None
        assert result.reconstructed_groups is not None
        assert len(result.original_groups) == 20
        assert set(result.original_groups.unique()) == {"A", "B"}

    def test_repr_shows_has_groups_true(self, sample_result_with_groups):
        """__repr__ reports has_groups=True when groups are present."""
        assert "has_groups=True" in repr(sample_result_with_groups)

    def test_repr_shows_has_groups_false(self, sample_generated, sample_loss):
        """__repr__ reports has_groups=False when groups are absent."""
        result = SyngResult(
            generated_data=sample_generated, loss=sample_loss, metadata={}
        )
        assert "has_groups=False" in repr(result)

    def test_summary_includes_group_info(self, sample_result_with_groups):
        """summary() reports group info and class count when groups are present."""
        s = sample_result_with_groups.summary()
        assert "Groups" in s
        assert "2 classes" in s

    def test_summary_no_group_info_when_none(self, sample_generated, sample_loss):
        """summary() omits group info when groups are absent."""
        result = SyngResult(
            generated_data=sample_generated, loss=sample_loss, metadata={}
        )
        assert "Groups" not in result.summary()

    def test_save_writes_group_csvs(self, sample_result_with_groups, temp_dir):
        """save() writes group sidecar CSVs."""
        paths = sample_result_with_groups.save(temp_dir)
        assert "original_groups" in paths
        assert "generated_groups" in paths
        assert "reconstructed_groups" in paths
        for key in ("original_groups", "generated_groups", "reconstructed_groups"):
            assert paths[key].exists()

    def test_save_no_group_csvs_when_none(
        self, sample_generated, sample_loss, temp_dir
    ):
        """save() does not write group CSVs when groups are None."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test"},
        )
        paths = result.save(temp_dir)
        assert "original_groups" not in paths
        assert "generated_groups" not in paths
        assert "reconstructed_groups" not in paths

    def test_load_restores_groups(self, sample_result_with_groups, temp_dir):
        """load() restores group attributes from sidecar CSVs."""
        sample_result_with_groups.save(temp_dir)
        loaded = SyngResult.load(temp_dir)

        assert loaded.original_groups is not None
        pd.testing.assert_series_equal(
            loaded.original_groups,
            sample_result_with_groups.original_groups,
            check_names=False,
        )
        assert loaded.generated_groups is not None
        assert loaded.reconstructed_groups is not None

    def test_load_no_groups_returns_none(self, sample_generated, sample_loss, temp_dir):
        """load() sets groups to None when no sidecar CSVs exist."""
        result = SyngResult(
            generated_data=sample_generated,
            loss=sample_loss,
            metadata={"dataname": "test"},
        )
        result.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        assert loaded.original_groups is None
        assert loaded.generated_groups is None
        assert loaded.reconstructed_groups is None

    def test_group_mapping_persisted_in_metadata(
        self, sample_result_with_groups, temp_dir
    ):
        """group_mapping is serialised to metadata JSON and restored on load."""
        sample_result_with_groups.save(temp_dir)
        loaded = SyngResult.load(temp_dir)
        mapping = loaded.metadata.get("group_mapping")
        assert mapping is not None
        assert mapping == {0: "A", 1: "B"}

    def test_save_load_roundtrip_group_values(
        self, sample_result_with_groups, temp_dir
    ):
        """Group values are preserved exactly through save/load."""
        sample_result_with_groups.save(temp_dir)
        loaded = SyngResult.load(temp_dir)

        orig_vals = sample_result_with_groups.original_groups.tolist()
        loaded_vals = loaded.original_groups.tolist()
        assert orig_vals == loaded_vals
