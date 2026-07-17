"""Tests for syng_bts.synthesize (Redesigned API).

These tests verify the public functions ``evaluate_sample_sizes`` and
``plot_sample_sizes`` using both DataFrame and SyngResult inputs, bundled
BRCA datasets, method aliases, validation, and ``apply_log`` semantics.
"""

from __future__ import annotations

import inspect
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import syng_bts.synthesize as synthesize
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
# Inverse power-law uncertainty
# ---------------------------------------------------------------------------


class TestPowerLawUncertainty:
    """Regression tests for delta-method confidence-band calculations."""

    def test_parameter_gradient_matches_finite_differences(self):
        """The analytic gradient differentiates with respect to a, b, and c."""
        x = 10.0
        params = np.array([0.1, 0.2, -0.5])
        epsilon = 1e-6
        finite_difference = np.empty(3)

        for index in range(3):
            offset = np.zeros(3)
            offset[index] = epsilon
            finite_difference[index] = (
                synthesize._power_law(x, *(params + offset))
                - synthesize._power_law(x, *(params - offset))
            ) / (2 * epsilon)

        analytic = synthesize._power_law_gradient(x, *params)

        np.testing.assert_allclose(analytic, finite_difference, rtol=1e-7, atol=1e-9)

    def test_prediction_variance_matches_hand_calculation(self):
        """Parameter covariance is propagated as J Sigma J-transpose."""
        params = np.array([0.1, 0.2, -0.5])
        covariance = np.array(
            [
                [0.04, 0.002, -0.003],
                [0.002, 0.01, 0.001],
                [-0.003, 0.001, 0.09],
            ]
        )

        variance = synthesize._power_law_prediction_variance(10.0, params, covariance)

        assert variance == pytest.approx(0.043391928179533926)

    def test_curve_fit_uses_r_weights_in_sample_size_order(self, monkeypatch):
        """Curve fitting uses the R row weights after sorting by sample size."""
        captured: dict[str, np.ndarray] = {}

        def capture_curve_fit(_func, x, y, **kwargs):
            captured["x"] = np.asarray(x)
            captured["y"] = np.asarray(y)
            captured["sigma"] = kwargs.get("sigma")
            return np.array([0.1, 0.2, -0.5]), np.eye(3) * 0.01

        monkeypatch.setattr(synthesize, "curve_fit", capture_curve_fit)
        metrics = pd.DataFrame(
            {
                "n": [30, 10, 20],
                "f1_score": [0.8, 0.6, 0.7],
            }
        )

        synthesize._fit_curve(metrics, "f1_score", plot=False)

        np.testing.assert_array_equal(captured["x"], [10, 20, 30])
        np.testing.assert_array_equal(captured["y"], [0.6, 0.7, 0.8])
        weights = np.arange(1, 4) / 3
        np.testing.assert_allclose(captured["sigma"], 1 / np.sqrt(weights))

    def test_curve_fit_warns_when_optimizer_fails(self, monkeypatch):
        """A failed optimizer emits a clear warning instead of failing silently."""

        def fail_curve_fit(*_args, **_kwargs):
            raise RuntimeError("did not converge")

        monkeypatch.setattr(synthesize, "curve_fit", fail_curve_fit)
        metrics = pd.DataFrame({"n": [10, 20, 30], "f1_score": [0.6, 0.7, 0.8]})

        with pytest.warns(RuntimeWarning, match="Curve fit failed"):
            synthesize._fit_curve(metrics, "f1_score", plot=False)

    def test_curve_fit_warns_for_non_finite_covariance(self, monkeypatch):
        """A fitted curve with unusable covariance warns and omits its band."""

        def non_finite_covariance(*_args, **_kwargs):
            return np.array([0.1, 0.2, -0.5]), np.full((3, 3), np.inf)

        monkeypatch.setattr(synthesize, "curve_fit", non_finite_covariance)
        metrics = pd.DataFrame({"n": [10, 20, 30], "f1_score": [0.6, 0.7, 0.8]})

        with pytest.warns(RuntimeWarning, match="covariance"):
            synthesize._fit_curve(metrics, "f1_score", plot=False)

    def test_optimize_warning_retains_fitted_curve_without_band(self, monkeypatch):
        """Finite fit parameters survive a covariance OptimizeWarning."""

        def warning_with_finite_fit(*_args, **_kwargs):
            warnings.warn(
                "covariance unavailable", synthesize.OptimizeWarning, stacklevel=2
            )
            return np.array([0.1, 0.2, -0.5]), np.full((3, 3), np.inf)

        monkeypatch.setattr(synthesize, "curve_fit", warning_with_finite_fit)
        metrics = pd.DataFrame({"n": [10, 20, 30], "f1_score": [0.6, 0.7, 0.8]})

        with pytest.warns(RuntimeWarning, match="covariance"):
            ax = synthesize._fit_curve(metrics, "f1_score", plot=True)

        assert ax is not None
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        plt.close(ax.figure)


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
            inspect.signature(evaluate_sample_sizes).parameters["apply_log"].default
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

    def test_float32_input_no_scaling_warning(self, brca_generated_data):
        """float32 feature matrices should not emit sklearn scaling warnings."""
        data, groups = brca_generated_data

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = evaluate_sample_sizes(
                data=data.astype("float32"),
                sample_sizes=[100],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                apply_log=True,
                verbose="silent",
            )

        assert isinstance(result, pd.DataFrame)
        assert len(caught) == 0


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — Preprocessing
# ---------------------------------------------------------------------------


class TestEvaluationPreprocessing:
    """Regression tests for training-fitted feature standardization."""

    def test_evaluation_uses_training_fitted_standardization(self, monkeypatch):
        """Evaluation values use training means and standard deviations."""
        data = pd.DataFrame(
            {
                "shifted": [
                    *range(5),
                    *range(100, 105),
                    *range(10, 15),
                    *range(110, 115),
                ],
                "constant_in_training": [7] * 5 + [99] * 5 + [7] * 5 + [99] * 5,
            }
        )
        groups = np.array(["A"] * 10 + ["B"] * 10)
        train_index = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        test_index = np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])
        captured: dict[str, np.ndarray] = {}

        class SingleSplit:
            def __init__(self, **_kwargs):
                pass

            def split(self, _data, _labels):
                yield train_index, test_index

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            captured["train_data"] = train_data.copy()
            captured["test_data"] = test_data.copy()
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setattr(synthesize, "StratifiedKFold", SingleSplit)
        monkeypatch.setattr(
            synthesize.np.random,
            "choice",
            lambda values, size, replace: values[:size],
        )
        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "LOGIS", capture_classifier)

        evaluate_sample_sizes(
            data=data,
            sample_sizes=[20],
            groups=groups,
            n_draws=1,
            apply_log=False,
            methods=["LOGIS"],
            verbose=0,
        )

        raw_train = data.iloc[train_index, 0].to_numpy()
        raw_test = data.iloc[test_index, 0].to_numpy()
        expected_test = (raw_test - raw_train.mean()) / raw_train.std()

        np.testing.assert_allclose(captured["train_data"][:, 0].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(captured["train_data"][:, 0].std(), 1.0)
        np.testing.assert_allclose(captured["test_data"][:, 0], expected_test)
        assert not np.isclose(captured["test_data"][:, 0].mean(), 0.0)
        np.testing.assert_array_equal(captured["train_data"][:, 1], 7)
        np.testing.assert_array_equal(captured["test_data"][:, 1], 99)


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — External evaluation set
# ---------------------------------------------------------------------------


class TestExternalEvaluation:
    """Tests for evaluation against a fixed user-supplied test set."""

    @pytest.mark.parametrize("missing", ["test_data", "test_groups"])
    def test_external_arguments_must_be_paired(self, small_synthetic_data, missing):
        """Supplying only one external argument raises a clear error."""
        data, groups = small_synthetic_data
        external_kwargs = {
            "test_data": data.iloc[:10].copy(),
            "test_groups": groups[:10],
        }
        external_kwargs.pop(missing)

        with pytest.raises(ValueError, match="must be provided together"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[20],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=0,
                **external_kwargs,
            )

    def test_external_feature_columns_must_match(self, small_synthetic_data):
        """Candidate and external datasets must expose the same features."""
        data, groups = small_synthetic_data
        test_data = data.iloc[:10].rename(columns={"gene_0": "different_gene"})

        with pytest.raises(ValueError, match="same feature columns"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[20],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=0,
                test_data=test_data,
                test_groups=groups[:10],
            )

    def test_internal_mode_retains_five_fold_evaluation(
        self, small_synthetic_data, monkeypatch
    ):
        """Omitting an external set preserves the existing five classifier calls."""
        data, groups = small_synthetic_data
        calls: list[int] = []

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            calls.append(len(test_data))
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "LOGIS", capture_classifier)

        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[20],
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
            verbose=0,
        )

        assert len(calls) == 5
        assert len(result) == 1

    def test_external_mode_scores_fixed_rows_once(self, monkeypatch):
        """The full candidate subset trains once and scores external rows."""
        candidate = pd.DataFrame(
            {
                "gene_a": np.arange(1, 21, dtype=float),
                "gene_b": np.arange(21, 41, dtype=float),
            }
        )
        groups = np.array(["A"] * 10 + ["B"] * 10)
        external = pd.DataFrame(
            {
                "gene_a": np.arange(101, 107, dtype=float),
                "gene_b": np.arange(201, 207, dtype=float),
            }
        )
        external_input = external.loc[:, ["gene_b", "gene_a"]].copy()
        external_before = external_input.copy()
        test_groups = np.array(["A", "B"] * 3)
        calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            calls.append(
                (
                    train_data.copy(),
                    train_labels.copy(),
                    test_data.copy(),
                    test_labels.copy(),
                )
            )
            return {"f1": 0.2, "accuracy": 0.3, "auc": 0.4}

        monkeypatch.setattr(
            synthesize.np.random,
            "choice",
            lambda values, size, replace: values[:size],
        )
        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "LOGIS", capture_classifier)

        result = evaluate_sample_sizes(
            data=candidate,
            sample_sizes=[10],
            groups=groups,
            n_draws=1,
            apply_log=True,
            methods=["LOGIS"],
            verbose=0,
            test_data=external_input,
            test_groups=test_groups,
        )

        assert len(calls) == 1
        train_data, train_labels, evaluation_data, evaluation_labels = calls[0]
        selected_indices = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        logged_train = np.log2(candidate.iloc[selected_indices] + 1).to_numpy()
        logged_external = np.log2(external + 1).to_numpy()
        expected_external = (
            logged_external - logged_train.mean(axis=0)
        ) / logged_train.std(axis=0)

        assert train_data.shape == (10, 2)
        np.testing.assert_array_equal(train_labels, [0] * 5 + [1] * 5)
        np.testing.assert_allclose(evaluation_data, expected_external)
        np.testing.assert_array_equal(evaluation_labels, [0, 1] * 3)
        assert result.iloc[0]["f1_score"] == pytest.approx(0.2)
        assert result.iloc[0]["accuracy"] == pytest.approx(0.3)
        assert result.iloc[0]["auc"] == pytest.approx(0.4)
        pd.testing.assert_frame_equal(external_input, external_before)

    def test_same_external_set_can_score_real_and_generated_calls(self, monkeypatch):
        """One empirical test set can be reused across independent evaluations."""
        real = pd.DataFrame(
            {
                "gene_a": np.arange(1, 21, dtype=float),
                "gene_b": np.arange(21, 41, dtype=float),
            }
        )
        generated = real * 2
        groups = np.array(["A"] * 10 + ["B"] * 10)
        external = real.iloc[:6].copy()
        external_before = external.copy()
        test_groups = np.array(["A", "B", "A", "B", "A", "B"])
        evaluation_sizes: list[int] = []

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            evaluation_sizes.append(len(test_data))
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setattr(
            synthesize.np.random,
            "choice",
            lambda values, size, replace: values[:size],
        )
        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "LOGIS", capture_classifier)

        results = [
            evaluate_sample_sizes(
                data=candidate,
                sample_sizes=[10],
                groups=groups,
                n_draws=1,
                apply_log=False,
                methods=["LOGIS"],
                verbose=0,
                test_data=external,
                test_groups=test_groups,
            )
            for candidate in (real, generated)
        ]

        assert evaluation_sizes == [len(external), len(external)]
        assert all(len(result) == 1 for result in results)
        pd.testing.assert_frame_equal(external, external_before)


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — Reproducibility
# ---------------------------------------------------------------------------


class TestRandomState:
    """Tests for reproducible evaluation."""

    def test_stochastic_classifier_helpers_accept_random_state(self):
        """Classifier helpers expose the seed propagated by the public API."""
        for helper in (
            synthesize._logis,
            synthesize._svm,
            synthesize._rf,
            synthesize._xgb,
        ):
            assert "random_state" in inspect.signature(helper).parameters

    def test_random_state_reproduces_sampling_and_reaches_classifier(
        self, small_synthetic_data, monkeypatch
    ):
        """One random_state reproduces subsets and seeds classifier fitting."""
        data, groups = small_synthetic_data
        captured_data: list[np.ndarray] = []
        captured_states: list[int | None] = []

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            captured_data.append(train_data.copy())
            captured_states.append(random_state)
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "RF", capture_classifier)

        for _ in range(2):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[20],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                apply_log=False,
                verbose=0,
                test_data=data.iloc[:10],
                test_groups=groups[:5].tolist() + groups[-5:].tolist(),
                random_state=17,
            )

        np.testing.assert_array_equal(captured_data[0], captured_data[1])
        assert captured_states == [17, 17]

    def test_random_state_controls_outer_cv(self, small_synthetic_data, monkeypatch):
        """The public random_state is passed to shuffled outer CV."""
        data, groups = small_synthetic_data
        seen_states: list[int | None] = []
        original_stratified_kfold = synthesize.StratifiedKFold

        def capture_stratified_kfold(*args, **kwargs):
            seen_states.append(kwargs.get("random_state"))
            return original_stratified_kfold(*args, **kwargs)

        def capture_classifier(
            train_data,
            train_labels,
            test_data,
            test_labels,
            random_state=None,
        ):
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setattr(synthesize, "StratifiedKFold", capture_stratified_kfold)
        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "RF", capture_classifier)

        evaluate_sample_sizes(
            data=data,
            sample_sizes=[20],
            groups=groups,
            n_draws=1,
            methods=["RF"],
            verbose=0,
            random_state=23,
        )

        assert seen_states == [23]


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — SyngResult path
# ---------------------------------------------------------------------------


class TestEvaluateSampleSizesSyngResult:
    """Tests for evaluate_sample_sizes using SyngResult inputs."""

    @pytest.mark.parametrize("which", ["generated", "original", "reconstructed", None])
    def test_which_selectors(self, small_syng_result, which):
        """Each 'which' selector (and the default) produces a result row."""
        kwargs = {} if which is None else {"which": which}
        result = evaluate_sample_sizes(
            data=small_syng_result,
            sample_sizes=[50],
            n_draws=1,
            methods=["LOGIS"],
            **kwargs,
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

    @pytest.mark.parametrize("bad_size", [-10, 0])
    def test_non_positive_sample_size_raises(self, small_synthetic_data, bad_size):
        """Non-positive sample sizes raise ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive integers"):
            evaluate_sample_sizes(data=data, sample_sizes=[bad_size], groups=groups)

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

    def test_nested_logistic_cv_infeasibility_raises_before_fitting(self, monkeypatch):
        """Outer folds must leave five members per class for inner LOGIS CV."""
        data = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4))
        groups = np.array(["A"] * 5 + ["B"] * 5)

        def should_not_fit(*_args, **_kwargs):
            pytest.fail("classifier fitting should not be reached")

        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "LOGIS", should_not_fit)

        with pytest.raises(ValueError, match="inner 5-fold"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[10],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                apply_log=False,
                verbose=0,
            )

    def test_external_mode_does_not_apply_outer_cv_minimum(self, monkeypatch):
        """A non-CV external RF evaluation can use fewer than five rows per class."""
        data = pd.DataFrame(np.arange(16, dtype=float).reshape(4, 4))
        groups = np.array(["A", "A", "B", "B"])

        def capture_classifier(*_args, **_kwargs):
            return {"f1": 0.5, "accuracy": 0.5, "auc": 0.5}

        monkeypatch.setitem(synthesize._CLASSIFIER_MAP, "RF", capture_classifier)

        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[4],
            groups=groups,
            n_draws=1,
            methods=["RF"],
            apply_log=False,
            verbose=0,
            test_data=data.copy(),
            test_groups=groups.copy(),
        )

        assert len(result) == 1

    def test_external_knn_requires_five_training_rows(self):
        """KNN feasibility is checked directly in external-evaluation mode."""
        data = pd.DataFrame(np.arange(16, dtype=float).reshape(4, 4))
        groups = np.array(["A", "A", "B", "B"])

        with pytest.raises(ValueError, match="KNN"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[4],
                groups=groups,
                n_draws=1,
                methods=["KNN"],
                apply_log=False,
                verbose=0,
                test_data=data.copy(),
                test_groups=groups.copy(),
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_candidate_values_raise(self, small_synthetic_data, bad_value):
        """Candidate feature data must contain only finite values."""
        data, groups = small_synthetic_data
        data = data.copy()
        data.iloc[0, 0] = bad_value

        with pytest.raises(ValueError, match="finite"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                apply_log=False,
                verbose=0,
            )

    def test_non_finite_external_values_raise(self, small_synthetic_data):
        """External feature data receives the same finite-value validation."""
        data, groups = small_synthetic_data
        test_data = data.iloc[:10].copy()
        test_data.iloc[0, 0] = np.nan
        test_groups = np.array(["A"] * 5 + ["B"] * 5)

        with pytest.raises(ValueError, match="test_data.*finite"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                verbose=0,
                test_data=test_data,
                test_groups=test_groups,
            )

    def test_invalid_log_input_raises_before_transform(self, small_synthetic_data):
        """Values outside the log2(x + 1) domain fail before transformation."""
        data, groups = small_synthetic_data
        data = data.copy()
        data.iloc[0, 0] = -1

        with pytest.raises(ValueError, match=r"log2\(x \+ 1\)"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                apply_log=True,
                verbose=0,
            )

    def test_missing_candidate_labels_raise(self, small_synthetic_data):
        """Missing candidate labels are rejected rather than becoming a class."""
        data, groups = small_synthetic_data
        groups = groups.astype(object)
        groups[0] = None

        with pytest.raises(ValueError, match="groups.*missing"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                verbose=0,
            )

    def test_missing_external_labels_raise(self, small_synthetic_data):
        """Missing external labels are rejected before string conversion."""
        data, groups = small_synthetic_data
        test_groups = np.array(["A"] * 5 + ["B"] * 4 + [None], dtype=object)

        with pytest.raises(ValueError, match="test_groups.*missing"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                verbose=0,
                test_data=data.iloc[:10].copy(),
                test_groups=test_groups,
            )

    def test_external_labels_must_cover_candidate_classes(self, small_synthetic_data):
        """Metric evaluation requires every candidate class in the external set."""
        data, groups = small_synthetic_data

        with pytest.raises(ValueError, match="test_groups.*all classes"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["RF"],
                verbose=0,
                test_data=data.iloc[:10].copy(),
                test_groups=np.array(["A"] * 10),
            )


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

    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("LOGISTIC", "LOGIS"),
            ("LR", "LOGIS"),
            ("RANDOM_FOREST", "RF"),
            ("XGBOOST", "XGB"),
            ("logis", "LOGIS"),
            ("rf", "RF"),
        ],
    )
    def test_method_alias_resolution(self, small_synthetic_data, alias, canonical):
        """Aliases and mixed-case names resolve to canonical method names."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[50],
            groups=groups,
            n_draws=1,
            methods=[alias],
        )
        assert result["method"].iloc[0] == canonical


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

    def test_n_target_is_removed_from_plotting_api(self):
        """The unused n_target argument is absent from public and private APIs."""
        assert "n_target" not in inspect.signature(plot_sample_sizes).parameters
        assert "n_target" not in inspect.signature(synthesize._fit_curve).parameters

    def test_plot_uses_candidate_size_label_and_does_not_hide_low_values(
        self, monkeypatch
    ):
        """Curve panels identify candidate size and allow low values to remain visible."""

        def stable_curve_fit(*_args, **_kwargs):
            return np.array([0.6, 0.2, -0.5]), np.eye(3) * 0.001

        monkeypatch.setattr(synthesize, "curve_fit", stable_curve_fit)
        metrics = pd.DataFrame({"n": [10, 20, 30], "f1_score": [0.2, 0.25, 0.3]})

        ax = synthesize._fit_curve(metrics, "f1_score", plot=True)

        assert ax is not None
        assert ax.get_xlabel() == "Candidate subset size"
        assert ax.get_ylim()[0] < 0.2
        plt.close(ax.figure)

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
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.parametrize(
        "methods, with_generated, expected_axes",
        [
            (["LOGIS"], True, 2),
            (["LOGIS", "RF"], False, 2),
            (["LOGIS", "RF"], True, 4),
        ],
    )
    def test_panel_geometry(
        self, small_synthetic_data, methods, with_generated, expected_axes
    ):
        """Panel count = n_methods rows × (2 if generated else 1) columns."""
        data, groups = small_synthetic_data
        metrics = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60, 80],
            groups=groups,
            n_draws=2,
            methods=methods,
        )
        fig = plot_sample_sizes(
            metric_real=metrics,
            metric_generated=metrics if with_generated else None,
            metric_name="f1_score",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == expected_axes
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
            plot_sample_sizes(metric_real=metrics, metric_name="bad")

    def test_missing_required_columns_raises(self):
        """metric_real missing required columns should raise ValueError."""
        bad_metrics = pd.DataFrame(
            {"total_size": [40], "method": ["LOGIS"], "f1_score": [0.9]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            plot_sample_sizes(metric_real=bad_metrics)

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
                metric_generated=metric_generated,
            )


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — Verbose parameter
# ---------------------------------------------------------------------------


class TestVerboseEvaluate:
    """Tests for the verbose parameter of evaluate_sample_sizes."""

    @pytest.mark.parametrize(
        "int_form,str_form,empty,requires_bar,must_contain,must_not_contain",
        [
            (0, "silent", True, False, [], []),
            (1, "minimal", False, True, ["Progress", "size=1/1"], ["F1:"]),
            (2, "detailed", False, False, ["F1:", "Acc:", "AUC:"], []),
        ],
        ids=["silent", "minimal", "detailed"],
    )
    def test_verbose_level(
        self,
        small_synthetic_data,
        capsys,
        int_form,
        str_form,
        empty,
        requires_bar,
        must_contain,
        must_not_contain,
    ):
        """Each verbose level controls evaluate_sample_sizes() stdout, and the
        string alias behaves identically to its integer form."""
        data, groups = small_synthetic_data

        def output_for(verbose):
            result = evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=verbose,
            )
            assert isinstance(result, pd.DataFrame)
            return capsys.readouterr().out

        for verbose in (int_form, str_form):
            out = output_for(verbose)
            if empty:
                assert out == "", f"verbose={verbose!r} should be silent"
            if requires_bar:
                assert "\u2588" in out or "\u2591" in out, (
                    f"verbose={verbose!r}: expected block-bar characters"
                )
            for s in must_contain:
                assert s in out, f"verbose={verbose!r}: expected {s!r} in output"
            for s in must_not_contain:
                assert s not in out, f"verbose={verbose!r}: did not expect {s!r}"

    def test_verbose_invalid_raises(self, small_synthetic_data):
        """Invalid verbose values should raise ValueError."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=3,
            )
        with pytest.raises(ValueError):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[50],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose="loud",
            )

    def test_minimal_multiple_sizes_draws(self, small_synthetic_data, capsys):
        """Minimal mode tracks overall progress across sizes/draws/methods."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=[40, 60],
            groups=groups,
            n_draws=2,
            methods=["LOGIS", "RF"],
            verbose=1,
        )
        out = capsys.readouterr().out
        assert isinstance(result, pd.DataFrame)
        assert "Progress" in out
        assert "8/8" in out
        assert "size=1/2" in out
        assert "size=2/2" in out


# ---------------------------------------------------------------------------
# evaluate_sample_sizes — sample_sizes input types
# ---------------------------------------------------------------------------


class TestSampleSizesInput:
    """Tests for expanded sample_sizes input types."""

    @pytest.mark.parametrize(
        "sizes",
        [
            np.array([40, 60]),
            np.arange(40, 80, 20),
            pd.Series([40, 60]),
            np.array([33, 66]),
        ],
    )
    def test_sample_sizes_array_like(self, small_synthetic_data, sizes):
        """Array-like sample_sizes (ndarray / arange / Series) are accepted."""
        data, groups = small_synthetic_data
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=sizes,
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
            verbose=0,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result["total_size"]) == {int(s) for s in sizes}

    def test_single_int_equidistant(self):
        """Verify equidistant pattern for single int input."""
        data = pd.DataFrame(np.random.rand(90, 5), columns=[f"g{i}" for i in range(5)])
        groups = np.array(["A"] * 45 + ["B"] * 45)
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=3,
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
            verbose=0,
        )
        sizes = sorted(result["total_size"].unique())
        assert sizes == [30, 60, 90]

    def test_single_int_1(self, small_synthetic_data):
        """sample_sizes=1 should produce [n_rows]."""
        data, groups = small_synthetic_data  # 100 rows
        result = evaluate_sample_sizes(
            data=data,
            sample_sizes=1,
            groups=groups,
            n_draws=1,
            methods=["LOGIS"],
            verbose=0,
        )
        assert list(result["total_size"].unique()) == [100]

    def test_single_int_grid_count_cannot_exceed_rows(self, small_synthetic_data):
        """A scalar grid cannot request more unique sizes than available rows."""
        data, groups = small_synthetic_data

        with pytest.raises(ValueError, match="cannot exceed.*rows"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=101,
                groups=groups,
                n_draws=1,
                methods=["RF"],
                verbose=0,
            )

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_single_int_non_positive_raises(self, small_synthetic_data, bad_size):
        """A non-positive single int sample size is rejected."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=bad_size,
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=0,
            )

    def test_float_list_raises(self, small_synthetic_data):
        """Float sample sizes should be rejected."""
        data, groups = small_synthetic_data
        with pytest.raises(ValueError, match="positive integers"):
            evaluate_sample_sizes(
                data=data,
                sample_sizes=[40.5, 60.5],
                groups=groups,
                n_draws=1,
                methods=["LOGIS"],
                verbose=0,
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
