"""
Tests for evaluation and visualization functions.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")  # Use non-interactive backend for testing


# ---------------------------------------------------------------------------
# heatmap_eval
# ---------------------------------------------------------------------------


class TestHeatmapEval:
    """Test heatmap evaluation function."""

    def test_heatmap_eval_import(self):
        """Test heatmap_eval can be imported."""
        from syng_bts import heatmap_eval

        assert heatmap_eval is not None
        assert callable(heatmap_eval)

    def test_heatmap_eval_single_dataset(self, sample_data):
        """Test heatmap with single dataset returns a Figure."""
        from syng_bts import heatmap_eval

        fig = heatmap_eval(sample_data)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_eval_two_datasets(self, sample_data):
        """Test heatmap comparing two datasets returns a Figure."""
        from syng_bts import heatmap_eval

        dat_generated = sample_data * 1.1
        fig = heatmap_eval(sample_data, generated_data=dat_generated)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_eval_always_returns_figure(self, sample_data):
        """heatmap_eval always returns a Figure (no save parameter)."""
        from syng_bts import heatmap_eval

        fig = heatmap_eval(sample_data)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_no_plt_show(self, sample_data, monkeypatch):
        """heatmap_eval never calls plt.show()."""
        from syng_bts import heatmap_eval

        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
        fig = heatmap_eval(sample_data)
        assert show_called == []
        plt.close(fig)

    def test_heatmap_custom_cmap(self, sample_data):
        """Custom cmap parameter is accepted."""
        from syng_bts import heatmap_eval

        fig = heatmap_eval(sample_data, cmap="viridis")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_real_first_param_name(self, sample_data):
        """First positional param is real_data (not dat_real)."""
        from syng_bts import heatmap_eval

        fig = heatmap_eval(real_data=sample_data)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# UMAP_eval
# ---------------------------------------------------------------------------


class TestUMAPEval:
    """Test UMAP evaluation function."""

    def test_umap_eval_import(self):
        """Test UMAP_eval can be imported."""
        from syng_bts import UMAP_eval

        assert UMAP_eval is not None
        assert callable(UMAP_eval)

    def test_umap_eval_single_dataset_returns_figure(self, sample_data):
        """UMAP with only real data returns a Figure."""
        from syng_bts import UMAP_eval

        fig = UMAP_eval(sample_data)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_umap_eval_with_groups(self, sample_data):
        """UMAP with group labels returns a Figure."""
        from syng_bts import UMAP_eval

        groups = pd.Series(["A"] * 10 + ["B"] * 10)
        fig = UMAP_eval(sample_data, groups_real=groups)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_umap_eval_two_datasets(self, sample_data):
        """UMAP comparing two datasets returns a Figure."""
        from syng_bts import UMAP_eval

        dat_generated = sample_data * 1.1 + 0.01  # avoid zero variance
        fig = UMAP_eval(sample_data, generated_data=dat_generated)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_umap_eval_two_datasets_with_groups(self, sample_data):
        """UMAP with both datasets and groups returns a Figure."""
        from syng_bts import UMAP_eval

        dat_generated = sample_data * 1.1 + 0.01
        groups_r = pd.Series(["A"] * 10 + ["B"] * 10)
        groups_g = pd.Series(["A"] * 10 + ["B"] * 10)
        fig = UMAP_eval(
            sample_data,
            generated_data=dat_generated,
            groups_real=groups_r,
            groups_generated=groups_g,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_umap_eval_random_seed(self, sample_data):
        """Custom random_seed parameter accepted."""
        from syng_bts import UMAP_eval

        fig = UMAP_eval(sample_data, random_seed=123)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_umap_no_plt_show(self, sample_data, monkeypatch):
        """UMAP_eval never calls plt.show()."""
        from syng_bts import UMAP_eval

        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
        fig = UMAP_eval(sample_data)
        assert show_called == []
        plt.close(fig)

    def test_umap_real_first_param(self, sample_data):
        """First positional param is real_data, second is generated_data."""
        from syng_bts import UMAP_eval

        gen = sample_data * 1.1 + 0.01
        fig = UMAP_eval(real_data=sample_data, generated_data=gen)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


class TestEvaluationFunction:
    """Test the main evaluation() function."""

    def test_evaluation_import(self):
        """Test evaluation function can be imported."""
        from syng_bts import evaluation

        assert evaluation is not None
        assert callable(evaluation)

    def test_evaluation_returns_dict_of_figures(self, sample_data):
        """evaluation() returns a dict with 'heatmap' and 'umap' Figures."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(real_data=sample_data, generated_data=gen, apply_log=False)
        assert isinstance(figs, dict)
        assert "heatmap" in figs
        assert "umap" in figs
        assert isinstance(figs["heatmap"], Figure)
        assert isinstance(figs["umap"], Figure)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_no_plt_show(self, sample_data, monkeypatch):
        """evaluation() never calls plt.show()."""
        from syng_bts import evaluation

        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=False)
        assert show_called == []
        for f in figs.values():
            plt.close(f)

    def test_evaluation_with_apply_log(self, sample_data):
        """apply_log=True applies log2(x+1) to real data."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=True)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_accepts_dataframes(self, sample_data):
        """evaluation() accepts DataFrames directly."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=False)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_accepts_file_path(self, temp_dir, sample_data):
        """evaluation() accepts file paths via resolve_data."""
        from syng_bts import evaluation

        real_path = temp_dir / "real.csv"
        gen_path = temp_dir / "gen.csv"
        sample_data.to_csv(real_path, index=False)
        (sample_data * 1.1 + 0.01).to_csv(gen_path, index=False)

        figs = evaluation(str(real_path), str(gen_path), apply_log=False)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_n_samples_none(self, sample_data):
        """n_samples=None uses all samples."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=False, n_samples=None)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_n_samples_small(self, sample_data):
        """Small n_samples still works (sub-samples from each end)."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=False, n_samples=5)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_with_explicit_groups(self, sample_data):
        """evaluation() accepts explicit real_groups and generated_groups."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        real_groups = pd.Series(["TypeA"] * 10 + ["TypeB"] * 10)
        gen_groups = pd.Series(["TypeA"] * 10 + ["TypeB"] * 10)

        figs = evaluation(
            sample_data,
            gen,
            real_groups=real_groups,
            generated_groups=gen_groups,
            apply_log=False,
        )
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_explicit_groups_override_bundled(self):
        """Explicit real_groups take precedence over bundled groups."""
        from syng_bts import evaluation
        from syng_bts.data_utils import resolve_data

        real_df, bundled_groups = resolve_data("BRCASubtypeSel_test")
        assert bundled_groups is not None  # bundled groups exist

        gen = real_df * 1.1 + 0.01
        custom_groups = pd.Series(
            ["CustomA"] * (len(real_df) // 2)
            + ["CustomB"] * (len(real_df) - len(real_df) // 2)
        )

        figs = evaluation(
            real_df,
            gen,
            real_groups=custom_groups,
            apply_log=False,
        )
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_bundled_groups_used_as_fallback(self):
        """Bundled groups are used when no explicit groups are passed."""
        from syng_bts import evaluation
        from syng_bts.data_utils import resolve_data

        real_df, bundled_groups = resolve_data("BRCASubtypeSel_test")
        assert bundled_groups is not None

        gen = real_df * 1.1 + 0.01

        # No explicit groups — bundled should be used
        figs = evaluation(real_df, gen, apply_log=False)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_generated_bundled_groups_used_as_fallback(self, monkeypatch):
        """Generated bundled groups are used when generated_groups is omitted."""
        from syng_bts import evaluation
        from syng_bts.data_utils import resolve_data

        real_df, real_bundled_groups = resolve_data("BRCASubtypeSel_test")
        gen_df, gen_bundled_groups = resolve_data("BRCASubtypeSel_test")
        assert real_bundled_groups is not None
        assert gen_bundled_groups is not None

        captured: dict[str, pd.Series | None] = {}

        def _capture_umap(*args, **kwargs):
            captured["groups_real"] = kwargs.get("groups_real")
            captured["groups_generated"] = kwargs.get("groups_generated")
            return plt.figure()

        monkeypatch.setattr("syng_bts.evaluations.UMAP_eval", _capture_umap)

        figs = evaluation(
            "BRCASubtypeSel_test",
            "BRCASubtypeSel_test",
            apply_log=False,
        )
        assert isinstance(figs, dict)
        assert captured["groups_real"] is not None
        assert captured["groups_generated"] is not None
        assert len(captured["groups_real"]) == len(real_df)
        assert len(captured["groups_generated"]) == len(gen_df)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_no_groups(self, sample_data):
        """evaluation() works without any group information."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        figs = evaluation(sample_data, gen, apply_log=False)
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_accepts_list_and_ndarray_groups(self, sample_data):
        """evaluation() accepts list/ndarray group labels."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        real_groups = ["A"] * 10 + ["B"] * 10
        gen_groups = np.array(["A"] * 10 + ["B"] * 10)

        figs = evaluation(
            sample_data,
            gen,
            real_groups=real_groups,
            generated_groups=gen_groups,
            apply_log=False,
        )
        assert isinstance(figs, dict)
        for f in figs.values():
            plt.close(f)

    def test_evaluation_real_groups_length_mismatch_raises(self, sample_data):
        """real_groups length mismatch raises ValueError."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        bad_real_groups = pd.Series(["A"] * (len(sample_data) - 1))

        with pytest.raises(ValueError, match="real_groups length"):
            evaluation(
                sample_data,
                gen,
                real_groups=bad_real_groups,
                apply_log=False,
            )

    def test_evaluation_generated_groups_length_mismatch_raises(self, sample_data):
        """generated_groups length mismatch raises ValueError."""
        from syng_bts import evaluation

        gen = sample_data * 1.1 + 0.01
        bad_gen_groups = pd.Series(["A"] * (len(gen) - 1))

        with pytest.raises(ValueError, match="generated_groups length"):
            evaluation(
                sample_data,
                gen,
                generated_groups=bad_gen_groups,
                apply_log=False,
            )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestEvaluationFunctionsExist:
    """Test evaluation functions are accessible from main package."""

    def test_all_evaluation_functions_exported(self):
        """Test all evaluation functions are in package exports."""
        from syng_bts import __all__

        expected = ["heatmap_eval", "UMAP_eval", "evaluation"]
        for func_name in expected:
            assert func_name in __all__

    def test_evaluation_module_has_functions(self):
        """Test evaluations module has expected functions."""
        from syng_bts import evaluations

        assert hasattr(evaluations, "heatmap_eval")
        assert hasattr(evaluations, "UMAP_eval")
        assert hasattr(evaluations, "evaluation")

    def test_old_save_param_removed(self):
        """heatmap_eval no longer accepts 'save' parameter."""
        import inspect

        from syng_bts import heatmap_eval

        sig = inspect.signature(heatmap_eval)
        assert "save" not in sig.parameters

    def test_old_dat_real_param_renamed(self):
        """Parameters renamed from dat_real → real_data, etc."""
        import inspect

        from syng_bts import UMAP_eval, heatmap_eval

        hm_sig = inspect.signature(heatmap_eval)
        assert "real_data" in hm_sig.parameters
        assert "dat_real" not in hm_sig.parameters

        umap_sig = inspect.signature(UMAP_eval)
        assert "real_data" in umap_sig.parameters
        assert "dat_generated" not in umap_sig.parameters
        assert "dat_real" not in umap_sig.parameters

    def test_old_evaluation_params_removed(self):
        """evaluation() no longer accepts old generated_input/real_input/data_dir/group_names."""
        import inspect

        from syng_bts import evaluation

        sig = inspect.signature(evaluation)
        assert "generated_input" not in sig.parameters
        assert "real_input" not in sig.parameters
        assert "data_dir" not in sig.parameters
        assert "group_names" not in sig.parameters
