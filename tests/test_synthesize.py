"""Tests for syng_bts.synthesize (Redesigned API).

These tests verify the public functions ``evaluate_sample_sizes`` and
``plot_sample_sizes`` using both DataFrame and SyngResult inputs, bundled
BRCA datasets, method aliases, validation, and ``apply_log`` semantics.
"""

from __future__ import annotations

import inspect

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from syng_bts.data_utils import resolve_data
from syng_bts.result import SyngResult
from syng_bts.synthesize import evaluate_sample_sizes, plot_sample_sizes

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
def brca_generated_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load bundled generated BRCA dataset."""
    df, groups = resolve_data("BRCASubtypeSel_train_epoch285_CVAE1-20_generated")
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


@pytest.fixture
def small_syng_result(small_synthetic_data) -> SyngResult:
    """Create a minimal SyngResult with generated data and groups."""
    data, groups = small_synthetic_data
    return SyngResult(
        generated_data=data,
        loss=pd.DataFrame({"loss": [1.0, 0.5]}),
        original_data=data.copy(),
        reconstructed_data=data.copy(),
        metadata={"model": "VAE", "seed": 42},
        original_groups=pd.Series(groups, name="group"),
        generated_groups=pd.Series(groups, name="group"),
        reconstructed_groups=pd.Series(groups, name="group"),
    )


@pytest.fixture
def small_syng_result_no_groups(small_synthetic_data) -> SyngResult:
    """SyngResult without group attributes."""
    data, _groups = small_synthetic_data
    return SyngResult(
        generated_data=data,
        loss=pd.DataFrame({"loss": [1.0, 0.5]}),
        metadata={"model": "VAE", "seed": 42},
    )


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — DataFrame path
# ---------------------------------------------------------------------------


class TestEvaluateSampleSizesDataFrame:
    """Tests for evaluate_sample_sizes using DataFrame inputs."""

    def test_single_method(self, small_synthetic_data):
        """Run a single classifier on synthetic data — smoke test."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"total_size", "draw", "method", "f1_score", "accuracy", "auc"}
        assert set(result.columns) == expected_cols
        assert len(result) == 1
        assert result["method"].iloc[0] == "LOGIS"

    def test_all_methods(self, small_synthetic_data):
        """Run all classifiers — verifies all 5 methods execute."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[60],
            groups=groups,
            n_draws=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result["method"]) == {"LOGIS", "SVM", "KNN", "RF", "XGB"}
        assert len(result) == 5

    def test_multiple_sizes_and_draws(self, small_synthetic_data):
        """Multiple candidate sizes and draws produce expected row count."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60],
            groups=groups,
            n_draws=2,
            methods=["LOGIS", "RF"],
        )
        expected_rows = 2 * 2 * 2  # 2 sizes × 2 draws × 2 methods
        assert len(result) == expected_rows

    def test_metrics_in_valid_range(self, small_synthetic_data):
        """All returned metrics should be in [0, 1]."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[60],
            groups=groups,
            n_draws=2,
            methods=["LOGIS"],
        )
        for col in ["f1_score", "accuracy", "auc"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"
            assert (result[col] <= 1).all(), f"{col} exceeds 1"

    def test_apply_log_transform(self, small_synthetic_data):
        """Setting apply_log=True should apply log2(x+1) internally."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            apply_log=True,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_apply_log_true_is_default(self, small_synthetic_data):
        """Default apply_log=True; users can explicitly opt out."""
        assert (
            inspect.signature(evaluate_sample_sizes)
            .parameters["apply_log"]
            .default
            is True
        )

        data, groups = small_synthetic_data
        # Explicit opt-out path remains supported.
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            apply_log=False,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — SyngResult path
# ---------------------------------------------------------------------------


class TestEvaluateSampleSizesSyngResult:
    """Tests for evaluate_sample_sizes using SyngResult inputs."""

    def test_which_generated(self, small_syng_result):
        """SyngResult with which='generated' uses generated_data + groups."""
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            n_draws=1,
            which="generated",
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_which_original(self, small_syng_result):
        """SyngResult with which='original' uses original_data + groups."""
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            n_draws=1,
            which="original",
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_which_reconstructed(self, small_syng_result):
        """SyngResult with which='reconstructed' uses reconstructed_data."""
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            n_draws=1,
            which="reconstructed",
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_which_default_is_generated(self, small_syng_result):
        """Default which='generated' when not specified."""
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            n_draws=1,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_explicit_groups_override(self, small_syng_result):
        """Explicit groups override auto-resolved SyngResult groups."""
        custom_groups = np.array(["X"] * 50 + ["Y"] * 50)
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            groups=custom_groups,
            n_draws=1,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_no_groups_raises(self, small_syng_result_no_groups):
        """SyngResult without groups and no explicit groups raises."""
        with pytest.raises(ValueError, match="no generated_groups"):
            evaluate_sample_sizes(
                data=small_syng_result_no_groups,
                sample_sizes=[50],
                n_draws=1,
                methods=["LOGIS"],
            )

    def test_which_invalid_raises(self, small_syng_result):
        """Invalid which value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid 'which'"):
            evaluate_sample_sizes(
                data=small_syng_result,
                sample_sizes=[50],
                which="invalid",
                n_draws=1,
                methods=["LOGIS"],
            )

    def test_which_original_no_data_raises(self, small_synthetic_data):
        """SyngResult without original_data and which='original' raises."""
        data, _groups = small_synthetic_data
        sr = SyngResult(
            generated_data=data,
            loss=pd.DataFrame({"loss": [1.0]}),
            metadata={},
        )
        with pytest.raises(ValueError, match="no original_data"):
            evaluate_sample_sizes(
                data=sr,
                sample_sizes=[50],
                which="original",
                n_draws=1,
            )

    def test_which_reconstructed_no_data_raises(self, small_synthetic_data):
        """SyngResult without reconstructed_data raises for that selector."""
        data, _groups = small_synthetic_data
        sr = SyngResult(
            generated_data=data,
            loss=pd.DataFrame({"loss": [1.0]}),
            metadata={},
        )
        with pytest.raises(ValueError, match="no reconstructed_data"):
            evaluate_sample_sizes(
                data=sr,
                sample_sizes=[50],
                which="reconstructed",
                n_draws=1,
            )


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — Validation
# ---------------------------------------------------------------------------


class TestEvaluateValidation:
    """Tests for input validation in evaluate_sample_sizes."""

    def test_invalid_data_type_raises(self):
        """Non-DataFrame/SyngResult raises TypeError."""
        with pytest.raises(TypeError, match="pd.DataFrame or SyngResult"):
            evaluate_sample_sizes(
                data="not_a_dataframe",  # type: ignore[arg-type]
                sample_sizes=[50],
                groups=["A", "B"],
            )

    def test_dataframe_missing_groups_raises(self, small_synthetic_data):
        """DataFrame without groups raises ValueError."""
        data, _groups = small_synthetic_data
        with pytest.raises(ValueError, match="'groups' is required"):
            evaluate_sample_sizes(data=data, sample_sizes=[50])

    def test_empty_sample_sizes_raises(self, small_synthetic_data):
        """Empty sample_sizes raises ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_sample_sizes(data=data, sample_sizes=[], groups=groups)

    def test_negative_sample_size_raises(self, small_synthetic_data):
        """Non-positive sample size raises ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive integers"):
            evaluate_sample_sizes(data=data, sample_sizes=[-10], groups=groups)

    def test_zero_sample_size_raises(self, small_synthetic_data):
        """Zero sample size raises ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive integers"):
            evaluate_sample_sizes(data=data, sample_sizes=[0], groups=groups)

    def test_sample_size_exceeds_rows_raises(self, small_synthetic_data):
        """Sample size larger than data raises ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="exceeds available rows"):
            evaluate_sample_sizes(
                data=data, sample_sizes=[200], groups=groups, n_draws=1
            )

    def test_invalid_n_draws_raises(self, small_synthetic_data):
        """Non-positive n_draws raises ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive integer"):
            evaluate_sample_sizes(
                data=data, sample_sizes=[50], groups=groups, n_draws=0
            )

    def test_invalid_method_raises(self, small_synthetic_data):
        """Unknown classifier name should raise ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="Unknown classifier method"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["INVALID_METHOD"],
            )

    def test_empty_dataframe_raises(self):
        """Empty data should raise ValueError."""
        data = pd.DataFrame(columns=["a", "b"])
        groups = np.array([])
        with pytest.raises(ValueError, match="at least 1 row and 1 column"):
            evaluate_sample_sizes(data=data, sample_sizes=[10], groups=groups)

    def test_non_numeric_columns_raise(self):
        """Non-numeric feature columns should raise ValueError."""
        data = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
        groups = np.array(["A", "B"])
        with pytest.raises(ValueError, match="only numeric columns"):
            evaluate_sample_sizes(data=data, sample_sizes=[2], groups=groups)

    def test_groups_length_mismatch_raises(self, small_synthetic_data):
        """Groups length must match data rows."""
        data, _groups = small_synthetic_data
        short_groups = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_sample_sizes(data=data, sample_sizes=[50], groups=short_groups)

    def test_single_class_raises(self, small_synthetic_data):
        """At least two classes are required for classifier evaluation."""
        data, _groups = small_synthetic_data
        one_group = np.array(["A"] * len(data))
        with pytest.raises(ValueError, match="At least two unique groups"):
            evaluate_sample_sizes(data=data, sample_sizes=[50], groups=one_group)

    def test_sample_size_too_small_for_stratified_cv_raises(self, small_synthetic_data):
        """Sample sizes too small for 5-fold stratified CV should fail early."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="too small for 5-fold stratified CV"):
            evaluate_sample_sizes(data=data, sample_sizes=[8], groups=groups)


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — Method aliases
# ---------------------------------------------------------------------------


class TestMethodAliases:
    """Tests for method name resolution and aliases."""

    def test_canonical_names(self, small_synthetic_data):
        """Canonical method names are accepted."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["LOGIS", "SVM", "KNN", "RF", "XGB"],
        )
        assert set(result["method"]) == {"LOGIS", "SVM", "KNN", "RF", "XGB"}

    def test_alias_logistic(self, small_synthetic_data):
        """'LOGISTIC' alias resolves to 'LOGIS'."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["LOGISTIC"],
        )
        assert result["method"].iloc[0] == "LOGIS"

    def test_alias_lr(self, small_synthetic_data):
        """'LR' alias resolves to 'LOGIS'."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["LR"],
        )
        assert result["method"].iloc[0] == "LOGIS"

    def test_alias_random_forest(self, small_synthetic_data):
        """'RANDOM_FOREST' alias resolves to 'RF'."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["RANDOM_FOREST"],
        )
        assert result["method"].iloc[0] == "RF"

    def test_alias_xgboost(self, small_synthetic_data):
        """'XGBOOST' alias resolves to 'XGB'."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["XGBOOST"],
        )
        assert result["method"].iloc[0] == "XGB"

    def test_case_insensitive(self, small_synthetic_data):
        """Method names are case-insensitive."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=["logis", "rf"],
        )
        assert set(result["method"]) == {"LOGIS", "RF"}


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — BRCA baselines
# ---------------------------------------------------------------------------


class TestEvaluateBRCABaselines:
    """Baseline tests using bundled BRCA datasets."""

    @pytest.mark.slow
    def test_brca_test_data(self, brca_test_data):
        """Baseline with real BRCA test data — single classifier."""
        data, groups = brca_test_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50, 100],
            groups=groups,
            n_draws=1,
            apply_log=True,
            methods=["LOGIS"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert (result["auc"] > 0.5).all()

    @pytest.mark.slow
    def test_brca_generated_data(self, brca_generated_data):
        """Baseline with bundled generated BRCA data — single classifier."""
        data, groups = brca_generated_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[100],
            groups=groups,
            n_draws=1,
            apply_log=True,
            methods=["RF"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["auc"].iloc[0] > 0.5

    @pytest.mark.slow
    def test_brca_eval_and_plot_integration(self, brca_test_data, brca_generated_data):
        """Full eval + plot pipeline on both BRCA datasets."""
        real_data, real_groups = brca_test_data
        gen_data, gen_groups = brca_generated_data

        sample_sizes = [40, 80, 120]

        metric_real = evaluate_sample_sizes(
            data=real_data,
            sample_sizes=sample_sizes,
            groups=real_groups,
            n_draws=1,
            apply_log=True,
            methods=["LOGIS"],
        )
        metric_gen = evaluate_sample_sizes(
            data=gen_data,
            sample_sizes=sample_sizes,
            groups=gen_groups,
            n_draws=1,
            apply_log=True,
            methods=["LOGIS"],
        )

        fig = plot_sample_sizes(
            metric_real=metric_real,
            n_target=150,
            metric_generated=metric_gen,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_sample_sizes
# ---------------------------------------------------------------------------


class TestPlotSampleSizes:
    """Tests for plot_sample_sizes (redesigned API)."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self):
        """Use non-interactive matplotlib backend for tests."""
        backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        yield
        matplotlib.use(backend)

    def test_always_returns_figure(self, small_synthetic_data):
        """plot_sample_sizes always returns a Figure."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=2,
            methods=["LOGIS"],
        )
        fig = plot_sample_sizes(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_panel_with_generated(self, small_synthetic_data):
        """Providing metric_generated produces a two-column figure."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=2,
            methods=["LOGIS"],
        )
        fig = plot_sample_sizes(
            metric_real=metrics,
            n_target=100,
            metric_generated=metrics,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # 1 method × 2 panels
        plt.close(fig)

    def test_multiple_methods_panels(self, small_synthetic_data):
        """Multiple methods produce one row per method."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=2,
            methods=["LOGIS", "RF"],
        )
        fig = plot_sample_sizes(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # 2 methods × 1 column
        plt.close(fig)

    def test_multi_method_two_panel(self, small_synthetic_data):
        """Multiple methods with generated data produce method×2 panels."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=2,
            methods=["LOGIS", "RF"],
        )
        fig = plot_sample_sizes(
            metric_real=metrics,
            n_target=100,
            metric_generated=metrics,
            metric_name="accuracy",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2 methods × 2 columns
        plt.close(fig)

    def test_no_plt_show_called(self, small_synthetic_data, monkeypatch):
        """Verify plt.show() is never called."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
        )
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
        fig = plot_sample_sizes(
            metric_real=metrics,
            n_target=100,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        assert len(show_called) == 0
        plt.close(fig)

    def test_invalid_metric_name_raises(self, small_synthetic_data):
        """Invalid metric_name should raise ValueError."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
        )
        with pytest.raises(ValueError, match="Invalid metric_name"):
            plot_sample_sizes(metric_real=metrics, n_target=100, metric_name="bad")

    def test_missing_required_columns_raises(self):
        """metric_real missing required columns should raise ValueError."""
        bad_metrics = pd.DataFrame(
            {"total_size": [40], "method": ["LOGIS"], "f1_score": [0.9]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            plot_sample_sizes(metric_real=bad_metrics, n_target=100)

    def test_missing_generated_method_raises(self, small_synthetic_data):
        """metric_generated must include every method present in metric_real."""
        data, groups = small_synthetic_data
        metric_real = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=1,
            methods=["LOGIS", "RF"],
        )
        metric_generated = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
        )
        with pytest.raises(ValueError, match="Missing method"):
            plot_sample_sizes(
                metric_real=metric_real,
                n_target=100,
                metric_generated=metric_generated,
            )


# ---------------------------------------------------------------------------
# Top-level import tests
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    """Verify public API exports in syng_bts.__init__."""

    def test_evaluate_sample_sizes_importable(self):
        """evaluate_sample_sizes is importable from top-level package."""
        from syng_bts import evaluate_sample_sizes as fn

        assert callable(fn)

    def test_plot_sample_sizes_importable(self):
        """plot_sample_sizes is importable from top-level package."""
        from syng_bts import plot_sample_sizes as fn

        assert callable(fn)

    def test_in_all(self):
        """Both functions are listed in __all__."""
        import syng_bts

        assert "evaluate_sample_sizes" in syng_bts.__all__
        assert "plot_sample_sizes" in syng_bts.__all__
