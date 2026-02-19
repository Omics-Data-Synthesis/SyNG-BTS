"""Tests for syng_bts.synthesize — Phase 1 (legacy API baseline).

These tests verify the ported SyntheSize functions ``eval_classifier`` and
``vis_classifier`` using bundled BRCA datasets and small synthetic data.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from syng_bts.data_utils import resolve_data
from syng_bts.synthesize import eval_classifier, vis_classifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def brca_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load BRCASubtypeSel_test bundled dataset."""
    df, groups = resolve_data("BRCASubtypeSel_test")
    assert groups is not None
    return df, groups


@pytest.fixture
def brca_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load BRCASubtypeSel_train bundled dataset."""
    df, groups = resolve_data("BRCASubtypeSel_train")
    assert groups is not None
    return df, groups


@pytest.fixture
def small_synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a small synthetic dataset for fast unit tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    data = pd.DataFrame(
        np.random.rand(n_samples, n_features) * 10,
        columns=[f"gene_{i}" for i in range(n_features)],
    )
    groups = np.array(["A"] * 50 + ["B"] * 50)
    return data, groups


# ---------------------------------------------------------------------------
# eval_classifier tests
# ---------------------------------------------------------------------------


class TestEvalClassifier:
    """Tests for eval_classifier (legacy API)."""

    def test_small_synthetic_single_method(self, small_synthetic_data):
        """Run a single classifier on synthetic data — fast smoke test."""
        data, groups = small_synthetic_data
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[50],
            n_draw=1,
            log=True,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "total_size",
            "draw",
            "method",
            "f1_score",
            "accuracy",
            "auc",
        }
        assert len(result) == 1
        assert result["method"].iloc[0] == "LOGIS"

    def test_small_synthetic_all_methods(self, small_synthetic_data):
        """Run all classifiers on synthetic data — verifies all methods."""
        data, groups = small_synthetic_data
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[60],
            n_draw=1,
            log=True,
        )
        assert isinstance(result, pd.DataFrame)
        expected_methods = {"LOGIS", "SVM", "KNN", "RF", "XGB"}
        assert set(result["method"]) == expected_methods
        assert len(result) == 5  # 1 sample size × 1 draw × 5 methods

    def test_multiple_sizes_and_draws(self, small_synthetic_data):
        """Multiple candidate sizes and draws produce expected row count."""
        data, groups = small_synthetic_data
        n_candidate = [40, 60]
        n_draw = 2
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=n_candidate,
            n_draw=n_draw,
            log=True,
            methods=["LOGIS", "RF"],
        )
        expected_rows = len(n_candidate) * n_draw * 2  # 2 methods
        assert len(result) == expected_rows

    def test_log_transform_applied(self, small_synthetic_data):
        """Setting log=False should apply log2(x+1) internally."""
        data, groups = small_synthetic_data
        # Should not raise
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[50],
            n_draw=1,
            log=False,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_invalid_method_raises(self, small_synthetic_data):
        """Unknown classifier name should raise ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="Unknown classifier"):
            eval_classifier(
                whole_generated=data,
                whole_groups=groups,
                n_candidate=[50],
                n_draw=1,
                methods=["INVALID_METHOD"],
            )

    def test_metrics_in_valid_range(self, small_synthetic_data):
        """All returned metrics should be in [0, 1]."""
        data, groups = small_synthetic_data
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[60],
            n_draw=2,
            log=True,
            methods=["LOGIS"],
        )
        for col in ["f1_score", "accuracy", "auc"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"
            assert (result[col] <= 1).all(), f"{col} exceeds 1"

    @pytest.mark.slow
    def test_brca_test_data_baseline(self, brca_test_data):
        """Baseline test with real BRCA test data and a single classifier."""
        data, groups = brca_test_data
        # Use small sample sizes and 1 draw for speed
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[50, 100],
            n_draw=1,
            log=False,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 sizes × 1 draw × 1 method
        # BRCA subtypes should be distinguishable — expect AUC > 0.5
        assert (result["auc"] > 0.5).all()

    @pytest.mark.slow
    def test_brca_train_data_baseline(self, brca_train_data):
        """Baseline test with real BRCA train data and a single classifier."""
        data, groups = brca_train_data
        result = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[100],
            n_draw=1,
            log=False,
            methods=["RF"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["auc"].iloc[0] > 0.5

    @pytest.mark.slow
    def test_brca_eval_and_vis_integration(self, brca_test_data, brca_train_data):
        """Run BRCA eval on real+generated datasets and visualize both curves."""
        real_data, real_groups = brca_test_data
        generated_data, generated_groups = brca_train_data

        # >= 3 sample sizes needed for 3-parameter curve fitting in vis_classifier
        sample_sizes = [40, 80, 120]

        metric_real = eval_classifier(
            whole_generated=real_data,
            whole_groups=real_groups,
            n_candidate=sample_sizes,
            n_draw=1,
            log=False,
            methods=["LOGIS"],
        )
        metric_generated = eval_classifier(
            whole_generated=generated_data,
            whole_groups=generated_groups,
            n_candidate=sample_sizes,
            n_draw=1,
            log=False,
            methods=["LOGIS"],
        )

        fig = vis_classifier(
            metric_real=metric_real,
            n_target=150,
            metric_generated=metric_generated,
            metric_name="f1_score",
            save=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# vis_classifier tests
# ---------------------------------------------------------------------------


class TestVisClassifier:
    """Tests for vis_classifier (legacy API)."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self):
        """Use non-interactive matplotlib backend for tests."""
        backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        yield
        matplotlib.use(backend)

    def test_returns_figure_when_save(self, small_synthetic_data):
        """vis_classifier with save=True should return a Figure."""
        data, groups = small_synthetic_data
        metrics = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[40, 60, 80],
            n_draw=2,
            log=True,
            methods=["LOGIS"],
        )
        fig = vis_classifier(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
            save=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_none_when_not_save(self, small_synthetic_data):
        """vis_classifier with save=False should return None."""
        data, groups = small_synthetic_data
        metrics = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[40, 60, 80],
            n_draw=2,
            log=True,
            methods=["LOGIS"],
        )
        result = vis_classifier(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
            save=False,
        )
        assert result is None
        plt.close("all")

    def test_two_panel_with_generated(self, small_synthetic_data):
        """Providing metric_generated should produce a two-column figure."""
        data, groups = small_synthetic_data
        metrics = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[40, 60, 80],
            n_draw=2,
            log=True,
            methods=["LOGIS"],
        )
        fig = vis_classifier(
            metric_real=metrics,
            n_target=100,
            metric_generated=metrics,  # same data, just testing layout
            metric_name="f1_score",
            save=True,
        )
        assert isinstance(fig, plt.Figure)
        # Should have 2 axes (1 method × 2 panels)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_multiple_methods_panels(self, small_synthetic_data):
        """Multiple methods should produce one row per method."""
        data, groups = small_synthetic_data
        metrics = eval_classifier(
            whole_generated=data,
            whole_groups=groups,
            n_candidate=[40, 60, 80],
            n_draw=2,
            log=True,
            methods=["LOGIS", "RF"],
        )
        fig = vis_classifier(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
            save=True,
        )
        assert isinstance(fig, plt.Figure)
        # 2 methods × 1 column = 2 axes
        assert len(fig.axes) == 2
        plt.close(fig)
