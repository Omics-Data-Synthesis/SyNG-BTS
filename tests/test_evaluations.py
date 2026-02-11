"""
Tests for evaluation and visualization functions.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing


class TestHeatmapEval:
    """Test heatmap evaluation function."""

    def test_heatmap_eval_import(self):
        """Test heatmap_eval can be imported."""
        from syng_bts import heatmap_eval

        assert heatmap_eval is not None
        assert callable(heatmap_eval)

    def test_heatmap_eval_single_dataset(self, sample_data):
        """Test heatmap with single dataset."""
        from syng_bts import heatmap_eval

        # Should not raise
        fig = heatmap_eval(sample_data, save=True)

        assert fig is not None

    def test_heatmap_eval_two_datasets(self, sample_data):
        """Test heatmap comparing two datasets."""
        from syng_bts import heatmap_eval

        # Create a second dataset
        dat_generated = sample_data * 1.1

        fig = heatmap_eval(sample_data, dat_generated=dat_generated, save=True)

        assert fig is not None

    def test_heatmap_eval_returns_figure(self, sample_data):
        """Test that save=True returns matplotlib figure."""
        import matplotlib.pyplot as plt

        from syng_bts import heatmap_eval

        fig = heatmap_eval(sample_data, save=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestUMAPEval:
    """Test UMAP evaluation function."""

    def test_umap_eval_import(self):
        """Test UMAP_eval can be imported."""
        from syng_bts import UMAP_eval

        assert UMAP_eval is not None
        assert callable(UMAP_eval)

    def test_umap_eval_single_dataset(self, sample_data):
        """Test UMAP with single dataset (generated=None)."""
        import matplotlib.pyplot as plt

        from syng_bts import UMAP_eval

        # Run with only real data
        UMAP_eval(dat_generated=None, dat_real=sample_data)

        # Close figure to avoid display
        plt.close("all")

    def test_umap_eval_with_groups(self, sample_data):
        """Test UMAP with group labels."""
        import matplotlib.pyplot as plt

        from syng_bts import UMAP_eval

        # Create groups
        groups = pd.Series(["A"] * 10 + ["B"] * 10)

        UMAP_eval(dat_generated=None, dat_real=sample_data, groups_real=groups)

        plt.close("all")

    def test_umap_eval_two_datasets(self, sample_data):
        """Test UMAP comparing two datasets."""
        import matplotlib.pyplot as plt

        from syng_bts import UMAP_eval

        dat_generated = sample_data * 1.1

        UMAP_eval(dat_generated=dat_generated, dat_real=sample_data)

        plt.close("all")

    def test_umap_eval_random_state(self, sample_data):
        """Test UMAP with specific random state for reproducibility."""
        import matplotlib.pyplot as plt

        from syng_bts import UMAP_eval

        UMAP_eval(dat_generated=None, dat_real=sample_data, random_state=123)

        plt.close("all")


class TestEvaluationFunction:
    """Test the main evaluation function."""

    def test_evaluation_import(self):
        """Test evaluation function can be imported."""
        from syng_bts import evaluation

        assert evaluation is not None
        assert callable(evaluation)

    def test_evaluation_with_bundled_data(self):
        """Test evaluation with bundled BRCA data."""
        import matplotlib.pyplot as plt

        from syng_bts import evaluation

        # Uses default bundled BRCASubtype case study data
        try:
            evaluation(
                generated_input="BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv",
                real_input="BRCASubtypeSel_test.csv",
            )
        except FileNotFoundError:
            # Expected if the specific generated file doesn't exist
            pytest.skip("Bundled generated data not available")
        finally:
            plt.close("all")

    def test_evaluation_file_not_found(self, temp_dir):
        """Test evaluation raises FileNotFoundError for missing files."""
        from syng_bts import evaluation

        with pytest.raises(FileNotFoundError):
            evaluation(
                generated_input="nonexistent_generated.csv",
                real_input="nonexistent_real.csv",
                data_dir=temp_dir,
            )


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
