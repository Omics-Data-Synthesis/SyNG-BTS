"""Numerical fitting, failure semantics, serialization and consumer agreement."""

import json
import warnings
from decimal import Decimal, localcontext
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit
from scipy.stats import norm

import syng_bts.synthesize as synthesize
from syng_bts import LearningCurveFit, fit_sample_sizes, plot_sample_sizes


@pytest.fixture
def metrics():
    ns = np.array([20, 35, 60, 100, 180, 300, 500, 800])
    means = (
        0.94
        - 1.7 * ns**-0.5
        + np.array([0.04, -0.025, 0.012, -0.009, 0.007, -0.005, 0.004, -0.003])
    )
    rows = []
    for n, mean in zip(ns, means, strict=True):
        for draw, offset in enumerate([-0.01, 0.01]):
            rows.append(
                {
                    "total_size": n,
                    "draw": draw,
                    "method": "RF",
                    "f1_score": mean + offset,
                }
            )
    return pd.DataFrame(rows)


def stub_optimizer(monkeypatch, params=None, covariance=None):
    if params is None:
        params = np.array([0.1, 0.2, -0.5])
    if covariance is None:
        covariance = np.array(
            [[0.04, 0.002, -0.003], [0.002, 0.01, 0.001], [-0.003, 0.001, 0.09]]
        )
    monkeypatch.setattr(
        synthesize, "curve_fit", lambda *args, **kwargs: (params, covariance)
    )
    return params, covariance


def decimal_uncertainty(observed, params, xs):
    """Independent 70-digit normal-equation reference; standard library only."""
    with localcontext() as context:
        context.prec = 70
        a, b, c = [Decimal.from_float(float(value)) for value in params]
        ns, ys, weights = [
            [Decimal.from_float(float(value)) for value in observed[column]]
            for column in ["n", "observed_mean", "weight"]
        ]

        def gradient(n):
            power = (c * n.ln()).exp()
            return [-Decimal(1), -power, -b * power * n.ln()]

        jacobian = [gradient(n) for n in ns]
        augmented = [
            [
                sum(w * g[i] * g[j] for w, g in zip(weights, jacobian, strict=True))
                for j in range(3)
            ]
            + [Decimal(i == j) for j in range(3)]
            for i in range(3)
        ]
        # Pivoted Gauss-Jordan inversion, independently of the production QR.
        for j in range(3):
            pivot = max(range(j, 3), key=lambda i: abs(augmented[i][j]))
            augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
            divisor = augmented[j][j]
            augmented[j] = [value / divisor for value in augmented[j]]
            for i in range(3):
                if i != j:
                    multiplier = augmented[i][j]
                    augmented[i] = [
                        x - multiplier * y
                        for x, y in zip(augmented[i], augmented[j], strict=True)
                    ]
        scale = sum(
            w * (y - (1 - a - b * (c * n.ln()).exp())) ** 2
            for n, y, w in zip(ns, ys, weights, strict=True)
        ) / (len(ns) - 3)
        covariance = [[v * scale for v in row[3:]] for row in augmented]
        variance = []
        for x in xs:
            g = gradient(Decimal.from_float(float(x)))
            variance.append(
                sum(g[i] * covariance[i][j] * g[j] for i in range(3) for j in range(3))
            )
        return np.array(covariance, dtype=float), np.array(variance, dtype=float)


@pytest.mark.parametrize(
    "c,tolerance", [(-0.5, 1e-10), (-0.001, 1e-8), (-0.0001, 5e-7)]
)
def test_qr_uncertainty_matches_high_precision_on_arbitrary_grids(
    monkeypatch, c, tolerance
):
    ns = np.array([20, 35, 60, 100, 180, 300, 500, 800])
    params = np.array([0.2 - 0.05 / abs(c), 0.05 / abs(c), c])
    means = synthesize._power_law(ns, *params) + np.array(
        [0.004, -0.0025, 0.0012, -0.0009, 0.0007, -0.0005, 0.0004, -0.0003]
    )
    metrics = pd.DataFrame(
        {"total_size": ns, "draw": 0, "method": "RF", "f1_score": means}
    )
    stub_optimizer(monkeypatch, params=params)
    fit = fit_sample_sizes(metrics)["RF"]
    assert fit.ci_ok
    xs = np.r_[ns, 1, 10, 40.5, 40.5, 1600, 8000]
    covariance, variance = decimal_uncertainty(fit.observed, params, xs)
    predictions = fit.predict(xs)
    np.testing.assert_allclose(fit.covariance, covariance, rtol=tolerance)
    half_width = predictions.ci_high - predictions.predicted
    np.testing.assert_allclose(
        half_width**2 / norm.ppf(0.975) ** 2, variance, rtol=tolerance
    )
    pd.testing.assert_frame_equal(fit.predict(ns), fit.predictions)
    assert predictions.iloc[-3].equals(predictions.iloc[-4])

    # Existing exported fields suffice to reproduce uncertainty without an optimizer.
    payload = json.loads(json.dumps(fit.to_dict(), allow_nan=False))
    restored = LearningCurveFit(
        payload["method"],
        payload["metric_name"],
        pd.DataFrame(payload["observed"]),
        pd.DataFrame(payload["predictions"]),
        np.array(payload["parameters"]),
        np.array(payload["covariance"]),
        payload["fit_status"],
        payload["interval_status"],
        payload["message"],
    )
    pd.testing.assert_frame_equal(restored.predict(xs), predictions)

    # Paired row/weight permutation and uniform weight scaling preserve the estimator.
    for observed in [
        fit.observed.iloc[::-1],
        fit.observed.assign(weight=fit.observed.weight * 1e4),
    ]:
        actual_covariance, actual_variance = synthesize._curve_uncertainty(
            observed, params, xs
        )
        np.testing.assert_allclose(actual_covariance, covariance, rtol=tolerance)
        np.testing.assert_allclose(actual_variance, variance, rtol=tolerance)


@pytest.mark.parametrize(
    "params,sizes",
    [
        ([0.1, 0.0, -0.5], [20, 35, 60, 100, 180, 300, 500, 800]),
        ([0.1, 1.0, 0.0], [20, 35, 60, 100, 180, 300, 500, 800]),
        ([0.1, 1.0, -1e-9], [20, 35, 60, 100, 180, 300, 500, 800]),
        ([0.1, 1.0, -0.5], list(range(100000000, 100000008))),
    ],
)
def test_rank_deficiency_retains_curve_without_truncating_directions(
    metrics, monkeypatch, params, sizes
):
    stub_optimizer(monkeypatch, params=np.array(params))
    table = metrics.assign(total_size=np.repeat(sizes, 2))
    fit = fit_sample_sizes(table)["RF"]
    assert fit.fit_ok and not fit.ci_ok
    assert "Jacobian" in fit.message
    assert fit.predictions.ci_low.isna().all()
    assert fit.predict([25, 100]).ci_low.isna().all()
    json.dumps(fit.to_dict(), allow_nan=False)


def test_finite_optimizer_covariance_does_not_set_analytic_intervals(
    metrics, monkeypatch
):
    stub_optimizer(monkeypatch, covariance=np.eye(3))
    first = fit_sample_sizes(metrics)["RF"]
    stub_optimizer(monkeypatch, covariance=np.eye(3) * 1e12)
    second = fit_sample_sizes(metrics)["RF"]
    np.testing.assert_array_equal(first.covariance, second.covariance)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_weighted_fit_matches_independent_reference(metrics):
    result = fit_sample_sizes(metrics)["RF"]
    assert isinstance(result, LearningCurveFit)
    assert result.fit_ok and result.ci_ok
    observed = result.observed
    weights = np.arange(1, 9) / 8

    def model(n, a, b, c):
        return 1 - a - b * n**c

    weighted, _ = curve_fit(
        model,
        observed.n,
        observed.observed_mean,
        p0=[0, 1, -0.5],
        sigma=1 / np.sqrt(weights),
        maxfev=50000,
        absolute_sigma=False,
    )
    unweighted, _ = curve_fit(
        model, observed.n, observed.observed_mean, p0=[0, 1, -0.5], maxfev=50000
    )
    np.testing.assert_allclose(result.parameters, weighted)
    covariance, variance = decimal_uncertainty(result.observed, weighted, observed.n)
    np.testing.assert_allclose(result.covariance, covariance, rtol=1e-10)
    np.testing.assert_allclose(
        (result.predictions.ci_high - result.predictions.predicted) ** 2,
        norm.ppf(0.975) ** 2 * variance,
        rtol=1e-10,
    )
    np.testing.assert_allclose(observed.weight, weights)
    weighted_loss = np.sum(
        weights * (model(observed.n, *weighted) - observed.observed_mean) ** 2
    )
    unweighted_loss = np.sum(
        weights * (model(observed.n, *unweighted) - observed.observed_mean) ** 2
    )
    assert weighted_loss < unweighted_loss
    assert not np.allclose(weighted, unweighted, rtol=0.01)


def test_aggregation_order_counts_and_input_preserved(metrics):
    extra = metrics.iloc[[0]].assign(f1_score=0.5)
    table = pd.concat([metrics, extra], ignore_index=True).sample(
        frac=1, random_state=9
    )
    before = table.copy(deep=True)
    fit = fit_sample_sizes(table)["RF"]
    expected = table.groupby("total_size").f1_score.agg(["mean", "std", "size"])
    np.testing.assert_allclose(fit.observed.observed_mean, expected["mean"])
    np.testing.assert_allclose(fit.observed.observed_std, expected["std"])
    np.testing.assert_array_equal(fit.observed.n_draws, expected["size"])
    ordered = fit_sample_sizes(table.sort_values("total_size"))["RF"]
    np.testing.assert_allclose(fit.parameters, ordered.parameters, rtol=1e-5)
    pd.testing.assert_frame_equal(before, table)


def test_classifier_order_and_metric_selection(metrics):
    table = pd.concat([metrics.assign(method="SVM"), metrics], ignore_index=True)
    table["auc"] = table.f1_score + 0.03
    table["accuracy"] = table.f1_score + 0.02
    for metric in ["auc", "accuracy", "f1_score"]:
        fits = fit_sample_sizes(table, metric)
        assert list(fits) == ["SVM", "RF"]
        np.testing.assert_allclose(
            fits["RF"].observed.observed_mean,
            table[table.method == "RF"].groupby("total_size")[metric].mean(),
        )


def test_parameter_intervals_and_new_grid_do_not_refit(metrics, monkeypatch):
    params, _ = stub_optimizer(monkeypatch)
    fit = fit_sample_sizes(metrics)["RF"]

    def unexpected(*args, **kwargs):
        pytest.fail("prediction must not refit")

    monkeypatch.setattr(synthesize, "curve_fit", unexpected)
    xs = np.array([40.5, 10.0, 40.5, 100.0])
    prediction = fit.predict(xs)
    np.testing.assert_array_equal(prediction.n, xs)
    expected = 1 - params[0] - params[1] * xs ** params[2]
    covariance, variance = decimal_uncertainty(fit.observed, params, xs)
    np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-10)
    np.testing.assert_allclose(prediction.predicted, expected)
    np.testing.assert_allclose(
        prediction.ci_low, expected - norm.ppf(0.975) * np.sqrt(variance)
    )
    np.testing.assert_allclose(
        prediction.ci_high, expected + norm.ppf(0.975) * np.sqrt(variance)
    )
    assert list(prediction.columns) == ["n", "predicted", "ci_low", "ci_high"]


@pytest.mark.parametrize("count,status", [(1, "failed"), (2, "failed"), (3, "ok")])
def test_few_sizes(metrics, count, status):
    fit = fit_sample_sizes(metrics.iloc[: count * 2])["RF"]
    assert fit.fit_status == status
    assert not fit.ci_ok
    assert fit.interval_status == "unavailable"
    assert ("four" if count == 3 else "three") in fit.message
    assert fit.predictions.ci_low.isna().all()


def test_optimizer_failure_preserves_observations_and_serializes(metrics, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(synthesize, "curve_fit", fail)
    with warnings.catch_warnings(record=True) as caught:
        fit = fit_sample_sizes(metrics)["RF"]
    assert not caught
    assert fit.fit_status == "failed"
    assert fit.message == "no convergence"
    assert len(fit.observed) == 8
    assert fit.predictions.predicted.isna().all()
    assert json.loads(json.dumps(fit.to_dict(), allow_nan=False))["parameters"] is None
    with pytest.warns(RuntimeWarning, match="Curve fit failed"):
        fig = plot_sample_sizes(metrics)
    assert len(fig.axes[0].lines) == 0
    assert len(fig.axes[0].collections) == 1
    plt.close(fig)


@pytest.mark.parametrize(
    "covariance",
    [
        np.full((3, 3), np.inf),
        np.full((3, 3), np.nan),
    ],
)
def test_unusable_covariance_retains_curve(metrics, monkeypatch, covariance):
    stub_optimizer(monkeypatch, covariance=covariance)
    fit = fit_sample_sizes(metrics)["RF"]
    assert fit.fit_ok and not fit.ci_ok
    assert fit.interval_status == "unavailable"
    assert np.isfinite(fit.predictions.predicted).all()
    assert fit.predictions.ci_low.isna().all()
    json.dumps(fit.to_dict(), allow_nan=False)
    with pytest.warns(RuntimeWarning, match="covariance"):
        fig = plot_sample_sizes(metrics)
    assert len(fig.axes[0].lines) == 1
    assert len(fig.axes[0].collections) == 1
    plt.close(fig)


def test_covariance_warning_with_finite_matrix_is_not_silently_accepted(
    metrics, monkeypatch
):
    def warn(*args, **kwargs):
        warnings.warn("unreliable covariance", synthesize.OptimizeWarning, stacklevel=2)
        return np.array([0.1, 0.2, -0.5]), np.eye(3)

    monkeypatch.setattr(synthesize, "curve_fit", warn)
    fit = fit_sample_sizes(metrics)["RF"]
    assert fit.fit_ok
    assert fit.interval_status == "unavailable"
    assert fit.message == "unreliable covariance"


@pytest.mark.parametrize(
    "params,message",
    [
        (np.array([np.nan, 1, -0.5]), "non-finite parameters"),
        (np.array([0, 1, 500]), "non-finite predictions"),
    ],
)
def test_invalid_optimizer_values(metrics, monkeypatch, params, message):
    stub_optimizer(monkeypatch, params=params)
    fit = fit_sample_sizes(metrics)["RF"]
    assert fit.fit_status == "failed"
    assert message in fit.message
    assert not fit.fit_ok and not fit.ci_ok
    assert fit.predictions.predicted.isna().all()
    json.dumps(fit.to_dict(), allow_nan=False)


@pytest.mark.parametrize("variance", [-0.01, np.inf, np.nan])
def test_unusable_propagated_variance_is_not_clipped(metrics, monkeypatch, variance):
    stub_optimizer(monkeypatch)
    monkeypatch.setattr(
        synthesize,
        "_curve_uncertainty",
        lambda *args: (np.eye(3), np.full(len(args[2]), variance)),
    )
    fit = fit_sample_sizes(metrics)["RF"]
    assert fit.fit_ok
    assert fit.interval_status == "unavailable"
    assert fit.predictions.ci_low.isna().all()


def test_prediction_overflow_does_not_change_observed_status(metrics, monkeypatch):
    stub_optimizer(
        monkeypatch, params=np.array([0.1, 0.2, 2.0]), covariance=np.eye(3) * 1e-4
    )
    fit = fit_sample_sizes(metrics)["RF"]
    stored = fit.predictions.copy()
    with pytest.raises(ValueError, match="non-finite predictions"):
        fit.predict([1e200])
    pd.testing.assert_frame_equal(fit.predictions, stored)
    assert fit.fit_ok and fit.ci_ok


def test_serialization_and_csv_round_trip(metrics):
    one_draw = metrics[metrics.draw == 0]
    fit = fit_sample_sizes(one_draw)["RF"]
    payload = json.loads(json.dumps(fit.to_dict(), allow_nan=False))
    assert payload["schema_version"] == 1
    assert payload["parameter_names"] == ["a", "b", "c"]
    assert all(row["observed_std"] is None for row in payload["observed"])
    np.testing.assert_allclose(payload["covariance"], fit.covariance)
    pd.testing.assert_frame_equal(pd.DataFrame(payload["predictions"]), fit.predictions)
    restored = pd.read_csv(StringIO(metrics.to_csv(index=False)))
    fit_restored = fit_sample_sizes(restored)["RF"]
    np.testing.assert_allclose(
        fit_restored.predictions.predicted,
        fit_sample_sizes(metrics)["RF"].predictions.predicted,
        rtol=1e-6,
    )


def test_plot_and_public_api_agree(metrics, monkeypatch):
    fit = fit_sample_sizes(metrics)["RF"]
    calls = []
    original = synthesize.fit_sample_sizes

    def capture(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(synthesize, "fit_sample_sizes", capture)
    fig = plot_sample_sizes(metrics, metrics, y_limits=None)
    assert len(calls) == 2
    for ax in fig.axes:
        np.testing.assert_allclose(ax.lines[0].get_ydata(), fit.predictions.predicted)
        np.testing.assert_allclose(
            ax.collections[0].get_offsets()[:, 1], fit.observed.observed_mean
        )
        band = ax.collections[1].get_paths()[0].vertices
        for row in fit.predictions.itertuples():
            ys = band[band[:, 0] == row.n, 1]
            assert ys.min() == pytest.approx(row.ci_low)
            assert ys.max() == pytest.approx(row.ci_high)
    plt.close(fig)


@pytest.mark.parametrize(
    "column,value",
    [
        ("total_size", 0),
        ("total_size", -2),
        ("total_size", 1.5),
        ("total_size", np.inf),
        ("f1_score", np.nan),
        ("f1_score", np.inf),
        ("method", None),
        ("method", ""),
        ("draw", None),
    ],
)
def test_invalid_inputs_raise_before_fitting(metrics, column, value):
    table = metrics.copy()
    table[column] = value
    with pytest.raises(ValueError):
        fit_sample_sizes(table)


@pytest.mark.parametrize(
    "grid", [[], [0], [-1], [np.nan], [np.inf], [[10]], [True], ["10"], 10]
)
def test_invalid_prediction_grid(metrics, grid):
    fit = fit_sample_sizes(metrics)["RF"]
    with pytest.raises(ValueError):
        fit.predict(grid)


def test_invalid_table_schema(metrics):
    with pytest.raises(TypeError, match="DataFrame"):
        fit_sample_sizes([])
    with pytest.raises(ValueError, match="non-empty"):
        fit_sample_sizes(metrics.iloc[:0])
    with pytest.raises(ValueError, match="missing"):
        fit_sample_sizes(metrics.drop(columns="draw"))
    with pytest.raises(ValueError, match="metric_name"):
        fit_sample_sizes(metrics, "unsupported")
    with pytest.raises(ValueError, match="unique column"):
        fit_sample_sizes(pd.concat([metrics, metrics[["f1_score"]]], axis=1))


def test_synthetic_evaluation_to_fit_export_and_plot(tmp_path):
    rng = np.random.default_rng(17)
    groups = np.repeat([0, 1], 30)
    features = pd.DataFrame(rng.normal(size=(60, 4)) + groups[:, None] * 0.7)
    metrics = synthesize.evaluate_sample_sizes(
        features,
        [20, 30, 40, 50, 60],
        groups=groups,
        n_draws=2,
        methods=["KNN"],
        apply_log=False,
        random_seed=17,
        verbose=0,
    )
    assert len(metrics) == 10
    fit = fit_sample_sizes(metrics, "accuracy")["KNN"]
    assert len(fit.observed) == 5
    assert fit.observed.n_draws.eq(2).all()
    assert fit.fit_ok
    export = tmp_path / "fit.json"
    export.write_text(json.dumps(fit.to_dict(), allow_nan=False))
    assert json.loads(export.read_text())["fit_status"] == "ok"
    fig = plot_sample_sizes(metrics, metric_name="accuracy")
    np.testing.assert_allclose(
        fig.axes[0].lines[0].get_ydata(), fit.predictions.predicted
    )
    fig.savefig(tmp_path / "learning_curve.png")
    assert (tmp_path / "learning_curve.png").stat().st_size > 0
    plt.close(fig)


def test_invalid_plot_input_does_not_leave_a_figure(metrics):
    figures_before = plt.get_fignums()
    with pytest.raises(ValueError, match="finite"):
        plot_sample_sizes(metrics.assign(f1_score=np.nan))
    with pytest.raises(ValueError, match="Missing method"):
        plot_sample_sizes(metrics, metrics.assign(method="KNN"))
    assert plt.get_fignums() == figures_before


def test_stored_predictions_and_export_do_not_recalculate(metrics, monkeypatch):
    fit = fit_sample_sizes(metrics)["RF"]
    stored = fit.predictions.copy()

    def unexpected(*args, **kwargs):
        pytest.fail("stored predictions and export must not recalculate")

    monkeypatch.setattr(synthesize, "_predict_values", unexpected)
    monkeypatch.setattr(synthesize, "_confidence_limits", unexpected)
    for _ in range(2):
        pd.testing.assert_frame_equal(fit.predictions, stored)
        payload = fit.to_dict()
        pd.testing.assert_frame_equal(pd.DataFrame(payload["predictions"]), stored)
        assert "message" in payload
        assert set(payload["predictions"][0]) == {"n", "predicted", "ci_low", "ci_high"}


def test_bad_new_grid_intervals_raise_without_mutating_fit(metrics, monkeypatch):
    fit = fit_sample_sizes(metrics)["RF"]
    stored = fit.predictions.copy()
    monkeypatch.setattr(
        synthesize,
        "_curve_uncertainty",
        lambda *args: (np.eye(3), np.full(len(args[2]), -1.0)),
    )
    with pytest.raises(ValueError, match="negative variance"):
        fit.predict([25, 50])
    assert fit.fit_status == fit.interval_status == "ok"
    assert fit.message == ""
    pd.testing.assert_frame_equal(fit.predictions, stored)


def test_predict_keeps_known_unavailable_fit_or_band(metrics):
    for count in [2, 3]:
        fit = fit_sample_sizes(metrics.iloc[: count * 2])["RF"]
        prediction = fit.predict([25, 50])
        assert prediction.ci_low.isna().all()
        assert prediction.predicted.isna().all() == (count == 2)
