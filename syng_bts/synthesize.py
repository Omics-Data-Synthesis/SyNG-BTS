"""SyntheSize integration — sample-size evaluation via classifier learning curves.

This module provides classifier-based evaluation of synthetic data across
candidate sample sizes, using either stratified cross-validation or a fixed
external evaluation set, plus inverse power-law curve fitting.

Public API
----------
- :func:`evaluate_sample_sizes` — Evaluate classifiers across candidate sample
  sizes using stratified cross-validation or a fixed external evaluation set.
- :func:`fit_sample_sizes` — Return reusable numerical learning-curve results.
- :class:`LearningCurveFit` — Parameters, covariance, predictions and fit status.
- :func:`plot_sample_sizes` — Visualize IPLF learning curves from evaluation
  metrics.

References
----------
- SyntheSize (R): https://github.com/LXQin/SyntheSize
- SyntheSize (Python): https://github.com/LXQin/SyntheSize_py
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import DMatrix
from xgboost import train as xgb_train

from .helper_train import VerbosityLevel, _resolve_verbose

if TYPE_CHECKING:
    from .result import SyngResult

# ---------------------------------------------------------------------------
# Private classifier helpers
# ---------------------------------------------------------------------------


def _logis(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    random_state: int | None = None,
) -> dict[str, float]:
    """Ridge (L2-penalised) logistic regression classifier."""
    model_kwargs: dict[str, object] = {
        "Cs": 10,
        "cv": 5,
        "solver": "liblinear",
        "scoring": "accuracy",
        "random_state": random_state,
        "max_iter": 1000,
    }

    lr_params = inspect.signature(LogisticRegressionCV).parameters
    if "l1_ratios" in lr_params:
        model_kwargs["l1_ratios"] = (0,)
    elif "penalty" in lr_params:
        model_kwargs["penalty"] = "l2"

    if "use_legacy_attributes" in lr_params:
        model_kwargs["use_legacy_attributes"] = False

    model = LogisticRegressionCV(**model_kwargs)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(
            test_labels, predictions_proba, multi_class="ovo", average="macro"
        )

    return {
        "f1": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "auc": auc,
    }


def _svm(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    random_state: int | None = None,
) -> dict[str, float]:
    """Support Vector Machine classifier."""
    model = SVC(probability=True, random_state=random_state)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(
            test_labels, predictions_proba, multi_class="ovo", average="macro"
        )

    return {
        "f1": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "auc": auc,
    }


def _knn(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    random_state: int | None = None,
) -> dict[str, float]:
    """K-Nearest Neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(
            test_labels, predictions_proba, multi_class="ovo", average="macro"
        )

    return {
        "f1": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "auc": auc,
    }


def _rf(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    random_state: int | None = None,
) -> dict[str, float]:
    """Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(
            test_labels, predictions_proba, multi_class="ovo", average="macro"
        )

    return {
        "f1": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "auc": auc,
    }


def _xgb(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    random_state: int | None = None,
) -> dict[str, float]:
    """XGBoost classifier."""
    num_class = len(np.unique(train_labels))
    dtrain = DMatrix(train_data, label=train_labels)
    dtest = DMatrix(test_data, label=test_labels)

    if num_class == 2:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
        }
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
        }
    if random_state is not None:
        params["seed"] = random_state

    bst = xgb_train(params, dtrain, num_boost_round=10)
    predictions_proba = bst.predict(dtest)

    if predictions_proba.ndim == 1:
        predictions = (predictions_proba > 0.5).astype(int)
        auc = roc_auc_score(test_labels, predictions_proba)
    else:
        predictions = np.argmax(predictions_proba, axis=1)
        auc = roc_auc_score(
            test_labels, predictions_proba, multi_class="ovo", average="macro"
        )

    return {
        "f1": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "auc": auc,
    }


# Map canonical method names to private classifier callables
_CLASSIFIER_MAP: dict[
    str,
    Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None],
        dict[str, float],
    ],
] = {
    "LOGIS": _logis,
    "SVM": _svm,
    "KNN": _knn,
    "RF": _rf,
    "XGB": _xgb,
}

# Common aliases (case-insensitive lookup via upper())
_METHOD_ALIASES: dict[str, str] = {
    "LOGIS": "LOGIS",
    "LOGISTIC": "LOGIS",
    "LR": "LOGIS",
    "SVM": "SVM",
    "KNN": "KNN",
    "RF": "RF",
    "RANDOM_FOREST": "RF",
    "XGB": "XGB",
    "XGBOOST": "XGB",
}


def _print_eval_progress(
    step: int,
    total_steps: int,
    size_index: int,
    n_sizes: int,
    n: int,
    draw: int,
    method: str,
) -> None:
    """Print a single ``\\r``-overwritten progress line (MINIMAL verbosity).

    Format::

        Progress |████░░░░░░░░░░░░░░░░| 3/10 size=1/3 (n=50), draw=1, method=RF
    """
    pct = step / total_steps
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    print(
        f"\rProgress |{bar}| {step}/{total_steps} "
        f"size={size_index + 1}/{n_sizes} (n={n}), "
        f"draw={draw}, method={method}",
        end="",
        flush=True,
    )


def _resolve_methods(methods: list[str] | None) -> list[str]:
    """Resolve and validate classifier method names, accepting aliases."""
    if methods is None:
        return ["LOGIS", "SVM", "KNN", "RF", "XGB"]
    resolved: list[str] = []
    for m in methods:
        canonical = _METHOD_ALIASES.get(m.upper())
        if canonical is None:
            raise ValueError(
                f"Unknown classifier method: {m!r}. "
                f"Valid options: {sorted(set(_METHOD_ALIASES.values()))}"
            )
        resolved.append(canonical)
    return resolved


def _resolve_data_and_groups(
    data: pd.DataFrame | SyngResult,
    groups: np.ndarray | pd.Series | list | None,
    which: str,
) -> tuple[pd.DataFrame, np.ndarray | pd.Series]:
    """Resolve data and groups from a DataFrame or SyngResult.

    Parameters
    ----------
    data : pd.DataFrame or SyngResult
        Input data source.
    groups : array-like or None
        Explicit group labels. Required when *data* is a DataFrame.
        When provided alongside a SyngResult, overrides auto-resolved groups.
    which : str
        Selector for SyngResult fields: ``"generated"``, ``"original"``,
        or ``"reconstructed"``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | pd.Series]
        Resolved (features, group_labels) pair.
    """
    from .result import SyngResult

    if isinstance(data, SyngResult):
        valid_which = ("generated", "original", "reconstructed")
        if which not in valid_which:
            raise ValueError(
                f"Invalid 'which' value: {which!r}. Must be one of {valid_which}."
            )
        if which == "generated":
            resolved_data = data.generated_data
            resolved_groups = data.generated_groups
        elif which == "original":
            if data.original_data is None:
                raise ValueError("SyngResult has no original_data.")
            resolved_data = data.original_data
            resolved_groups = data.original_groups
        else:  # reconstructed
            if data.reconstructed_data is None:
                raise ValueError("SyngResult has no reconstructed_data.")
            resolved_data = data.reconstructed_data
            resolved_groups = data.reconstructed_groups

        # Allow explicit groups to override auto-resolved groups
        if groups is not None:
            resolved_groups = groups

        if resolved_groups is None:
            raise ValueError(
                f"SyngResult has no {which}_groups and no explicit 'groups' provided."
            )
        return resolved_data, resolved_groups

    if isinstance(data, pd.DataFrame):
        if groups is None:
            raise ValueError("'groups' is required when 'data' is a DataFrame.")
        return data, groups

    raise TypeError(
        f"'data' must be a pd.DataFrame or SyngResult, got {type(data).__name__}"
    )


def _allocate_stratified_counts(
    total_size: int,
    group_counts: dict[str, int],
) -> dict[str, int]:
    """Allocate per-group sample counts with largest-remainder rounding.

    Produces integer counts that sum to *total_size* and do not exceed each
    group's available count.
    """
    total_available = sum(group_counts.values())
    if total_size > total_available:
        raise ValueError(
            f"Requested sample size {total_size} exceeds available rows "
            f"({total_available})."
        )

    groups = list(group_counts.keys())
    raw = {
        group: (total_size * group_counts[group] / total_available) for group in groups
    }
    allocated = {
        group: min(int(np.floor(raw[group])), group_counts[group]) for group in groups
    }

    remaining = total_size - sum(allocated.values())
    if remaining > 0:
        order = sorted(
            groups,
            key=lambda group: raw[group] - allocated[group],
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for group in order:
                if allocated[group] < group_counts[group]:
                    allocated[group] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

    if sum(allocated.values()) != total_size:
        raise ValueError(
            "Could not allocate stratified sample counts that sum to the "
            f"requested size {total_size}."
        )

    return allocated


# ---------------------------------------------------------------------------
# Curve fitting helpers
# ---------------------------------------------------------------------------


def _power_law(x: float, a: float, b: float, c: float) -> float:
    """Inverse power-law function: ``(1 - a) - b * x^c``."""
    return (1 - a) - (b * (x**c))


def _power_law_gradient(
    x: float | np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """Gradient of :func:`_power_law` with respect to ``(a, b, c)``."""
    x_power_c = x**c
    return np.stack([-np.ones_like(x), -x_power_c, -b * x_power_c * np.log(x)], axis=-1)


def _curve_uncertainty(
    observed: pd.DataFrame,
    params: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic delta covariance and variances without dense propagation.

    For A = sqrt(W) J D and A[:, pivot] = Q R, solve R.T z = (g D)[pivot].
    Then Var(f(x)) = s² ||z||². This retains every parameter direction; the
    usual floating-point rank test rejects an unresolved Jacobian altogether.
    It is not a statistical identifiability or forward-accuracy guarantee.
    """
    ns = observed["n"].to_numpy()
    weights = observed["weight"].to_numpy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weighted = _power_law_gradient(ns, *params) * np.sqrt(weights)[:, None]
        scales = np.linalg.norm(weighted, axis=0)
        if not np.isfinite(scales).all() or (scales == 0).any():
            raise ValueError("analytic Jacobian has a zero or non-finite column")
        _, r, pivot = qr(weighted / scales, mode="economic", pivoting=True)
        singular_values = np.linalg.svd(r, compute_uv=False)
        tolerance = np.finfo(float).eps * max(weighted.shape) * singular_values[0]
        if singular_values[-1] <= tolerance:
            raise ValueError("analytic Jacobian is numerically rank deficient")
        residual = observed["observed_mean"].to_numpy() - _predict_values(ns, params)
        residual_variance = np.sum(weights * residual**2) / (len(ns) - 3)
        # Undo the column permutation/scaling when exporting parameter covariance.
        factor = np.empty((3, 3))
        factor[pivot] = solve_triangular(r, np.eye(3))
        factor *= np.sqrt(residual_variance) / scales[:, None]
        covariance = factor @ factor.T
        gradient = _power_law_gradient(xs, *params) / scales
        if not np.isfinite(gradient).all():
            raise ValueError("non-finite parameter gradient on the requested grid")
        projected = solve_triangular(r.T, gradient[:, pivot].T, lower=True)
        variance = residual_variance * np.sum(projected**2, axis=0)
    if not np.isfinite(covariance).all():
        raise ValueError("analytic parameter covariance is non-finite")
    return covariance, variance


@dataclass
class LearningCurveFit:
    """Numerical inverse power-law result for one classifier and metric.

    Obtain instances from :func:`fit_sample_sizes`. ``observed`` has columns
    ``n``, ``observed_mean``, ``observed_std`` (sample SD; NaN for one draw),
    ``n_draws`` and ``weight``. ``predictions`` stores ``n``, ``predicted``,
    ``ci_low`` and ``ci_high`` at observed sizes, calculated once during fitting.
    Parameters and covariance use order ``a, b, c``; missing values are None.
    Covariance uses the analytic Jacobian; if its calculation is unavailable,
    optimizer covariance is retained for inspection. Dense covariance rounding
    can lose information in ill-conditioned fits, so intervals use QR factors.

    ``fit_status`` is ``"ok"`` or ``"failed"``; ``interval_status`` is ``"ok"``
    or ``"unavailable"``. ``message`` explains a failed fit or omitted band.
    Treat result attributes as read-only. Intervals are approximate pointwise
    95% fitted-mean intervals, not prediction intervals.
    """

    method: str
    metric_name: str
    observed: pd.DataFrame
    predictions: pd.DataFrame
    parameters: np.ndarray | None = None
    covariance: np.ndarray | None = None
    fit_status: str = "failed"
    interval_status: str = "unavailable"
    message: str = ""

    @property
    def fit_ok(self) -> bool:
        """Whether parameters and fitted values at observed sizes are usable."""
        return self.fit_status == "ok"

    @property
    def ci_ok(self) -> bool:
        """Whether intervals at observed sizes are usable."""
        return self.interval_status == "ok"

    def predict(
        self, sample_sizes: list[float] | np.ndarray | pd.Series
    ) -> pd.DataFrame:
        """Evaluate the stored fit on another grid without refitting.

        Return ``n``, ``predicted``, ``ci_low`` and ``ci_high``, preserving grid
        order. Sizes must be positive finite numbers; fractional and repeated
        sizes are allowed. Missing fits/bands remain NaN. If an otherwise usable
        fit or band cannot be evaluated on this grid, raise ValueError without
        changing stored results. Extrapolation is allowed but unvalidated.
        """
        xs = _numeric_vector(sample_sizes, "sample_sizes")
        if (xs <= 0).any():
            raise ValueError("sample_sizes must be positive.")
        table = _empty_predictions(xs)
        if self.fit_ok:
            table["predicted"] = _predict_values(xs, self.parameters)
            if self.ci_ok:
                _, variance = _curve_uncertainty(self.observed, self.parameters, xs)
                table["ci_low"], table["ci_high"] = _confidence_limits(
                    table["predicted"].to_numpy(), variance
                )
        return table

    def to_dict(self) -> dict:
        """Return a JSON-compatible export, replacing non-finite values by null.

        Includes schema/method metadata, observations, parameters, covariance,
        predictions and statuses. No files are written. Use
        ``json.dumps(fit.to_dict(), allow_nan=False)`` for strict JSON.
        """

        def finite_or_none(value: object) -> object:
            if isinstance(value, dict):
                return {key: finite_or_none(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_or_none(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        return finite_or_none(
            {
                "schema_version": 1,
                "model": "1 - a - b * n**c",
                "weighting": "sorted_size_rank",
                "absolute_sigma": False,
                "confidence_level": 0.95,
                "interval_method": "parameter_delta_normal",
                "method": self.method,
                "metric_name": self.metric_name,
                "parameter_names": ["a", "b", "c"],
                "parameters": None
                if self.parameters is None
                else self.parameters.tolist(),
                "covariance": None
                if self.covariance is None
                else self.covariance.tolist(),
                "fit_status": self.fit_status,
                "interval_status": self.interval_status,
                "message": self.message,
                "observed": self.observed.to_dict(orient="records"),
                "predictions": self.predictions.to_dict(orient="records"),
            }
        )


def _numeric_vector(values: object, name: str) -> np.ndarray:
    """Validate finite real numeric data without coercing strings or booleans."""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a non-empty one-dimensional numeric array.")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _empty_predictions(xs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"n": xs, "predicted": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    )


def _predict_values(xs: np.ndarray, params: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predicted = _power_law(xs, *params)
    if not np.isfinite(predicted).all():
        raise ValueError("non-finite predictions on the requested grid")
    return predicted


def _confidence_limits(
    predicted: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if not np.isfinite(variance).all() or (variance < 0).any():
            raise ValueError(
                "parameter covariance gives non-finite or negative variance"
            )
        half_width = norm.ppf(0.975) * np.sqrt(variance)
        low, high = predicted - half_width, predicted + half_width
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("non-finite confidence limits on the requested grid")
    return low, high


def _fit_observed(
    observed: pd.DataFrame, method: str, metric_name: str
) -> LearningCurveFit:
    """Fit once, retaining observations or a usable curve if later steps fail."""
    xs = observed["n"].to_numpy()
    result = LearningCurveFit(method, metric_name, observed, _empty_predictions(xs))
    try:
        if len(observed) < 3:
            raise ValueError("at least three distinct sample sizes are required")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params, covariance = curve_fit(
                    _power_law,
                    xs,
                    observed["observed_mean"],
                    p0=[0, 1, -0.5],
                    sigma=1 / np.sqrt(observed["weight"]),
                    absolute_sigma=False,
                    maxfev=50000,
                )
        if not np.isfinite(params).all():
            raise ValueError("optimizer returned non-finite parameters")
        result.parameters, result.covariance = params, covariance
        result.predictions["predicted"] = _predict_values(xs, params)
        result.fit_status = "ok"
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        result.message = str(exc)
        return result

    try:
        if len(observed) == 3:
            raise ValueError(
                "at least four sizes are required to estimate parameter covariance"
            )
        covariance_warnings = "; ".join(
            str(w.message) for w in caught if issubclass(w.category, OptimizeWarning)
        )
        if covariance_warnings:
            raise ValueError(covariance_warnings)
        if not np.isfinite(covariance).all():
            raise ValueError("optimizer returned non-finite covariance")
        result.covariance, variance = _curve_uncertainty(observed, params, xs)
        result.predictions["ci_low"], result.predictions["ci_high"] = (
            _confidence_limits(result.predictions["predicted"].to_numpy(), variance)
        )
        result.interval_status = "ok"
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.message = str(exc)
    return result


def fit_sample_sizes(
    metrics: pd.DataFrame,
    metric_name: str = "f1_score",
) -> dict[str, LearningCurveFit]:
    """Fit reusable weighted learning curves from evaluation rows.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Non-empty output of :func:`evaluate_sample_sizes`, with columns
        ``total_size``, ``draw``, ``method`` and the selected metric. Sizes must
        be positive finite integers and metric values finite real numbers.
        Method labels must be non-empty strings and draw labels non-missing.
        Every row contributes equally to its size/method mean, including
        repeated draw labels. No rows are silently dropped. Inputs are unchanged.
    metric_name : str, default "f1_score"
        One of ``"f1_score"``, ``"accuracy"`` or ``"auc"``.

    Returns
    -------
    dict[str, LearningCurveFit]
        Results in first-seen classifier order. Each curve aggregates rows by
        size (mean, sample SD and row count), sorts distinct sizes and assigns
        weights ``1/m, ..., 1``. The fit minimizes weighted squared residuals
        with an unconstrained inverse power law. Covariance is residual-scaled,
        not estimated from the observed draw SDs.

    Raises
    ------
    ValueError
        For empty, malformed or non-finite input. Numerical fit/interval
        failures instead return observations and explicit result statuses.
    """
    if metric_name not in {"f1_score", "accuracy", "auc"}:
        raise ValueError(f"Invalid metric_name {metric_name!r}.")
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame.")
    required = {"total_size", "draw", "method", metric_name}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing required columns: {sorted(missing)}")
    if not metrics.columns.is_unique:
        raise ValueError("metrics must have unique column names.")
    if metrics.empty:
        raise ValueError("metrics must be non-empty.")
    sizes = _numeric_vector(metrics["total_size"], "total_size")
    if (sizes <= 0).any() or (sizes != np.floor(sizes)).any():
        raise ValueError("total_size must contain positive integers.")
    values = _numeric_vector(metrics[metric_name], metric_name)
    if (
        not metrics["method"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("method must contain non-empty strings.")
    if metrics["draw"].isna().any():
        raise ValueError("draw must not contain missing labels.")
    # Use validated float64 values without modifying the caller's frame.
    table = pd.DataFrame(
        {"n": sizes, "method": metrics["method"].to_numpy(), "value": values}
    )
    fits = {}
    for method, rows in table.groupby("method", sort=False, observed=True):
        observed = (
            rows.groupby("n", sort=True)
            .agg(
                observed_mean=("value", "mean"),
                observed_std=("value", "std"),
                n_draws=("value", "size"),
            )
            .reset_index()
        )
        if not np.isfinite(observed["observed_mean"]).all():
            raise ValueError("aggregation produced non-finite metric means.")
        observed["weight"] = np.arange(1, len(observed) + 1) / len(observed)
        fits[method] = _fit_observed(observed, method, metric_name)
    return fits


def _warn_fit(result: LearningCurveFit, annotation: str) -> None:
    context = f" for {annotation}" if annotation else ""
    if not result.fit_ok:
        warnings.warn(
            f"Curve fit failed{context}: {result.message}",
            RuntimeWarning,
            stacklevel=3,
        )
    elif not result.ci_ok:
        warnings.warn(
            f"Curve fit covariance is unusable{context}: {result.message}; "
            "the confidence band is omitted.",
            RuntimeWarning,
            stacklevel=3,
        )


def _plot_fit(result: LearningCurveFit, ax: plt.Axes, annotation: str) -> plt.Axes:
    _warn_fit(result, annotation)
    observed, predictions = result.observed, result.predictions
    ax.scatter(
        observed["n"], observed["observed_mean"], label="Actual Data", color="red"
    )
    if result.fit_ok:
        ax.plot(
            predictions["n"],
            predictions["predicted"],
            label="Fitted",
            color="blue",
            linestyle="--",
        )
        if result.ci_ok:
            ax.fill_between(
                predictions["n"],
                predictions["ci_low"],
                predictions["ci_high"],
                color="blue",
                alpha=0.2,
                label="95% CI",
            )
    ax.set_xlabel("Candidate subset size")
    ax.legend(loc="best")
    ax.set_title(annotation)
    return ax


def _fit_curve(
    acc_table: pd.DataFrame,
    metric_name: str,
    plot: bool = True,
    ax: plt.Axes | None = None,
    annotation: str = "",
) -> plt.Axes | None:
    """Compatibility wrapper for the former private plotting helper."""
    metrics = acc_table.rename(columns={"n": "total_size"}).assign(
        method="curve", draw=0
    )
    result = fit_sample_sizes(metrics, metric_name)["curve"]
    if not plot:
        _warn_fit(result, annotation)
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    return _plot_fit(result, ax, annotation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_sample_sizes(
    data: pd.DataFrame | SyngResult,
    sample_sizes: list[int] | np.ndarray | pd.Series | int,
    groups: np.ndarray | pd.Series | list | None = None,
    which: str = "generated",
    n_draws: int = 5,
    apply_log: bool = True,
    methods: list[str] | None = None,
    verbose: int | str = "minimal",
    test_data: pd.DataFrame | None = None,
    test_groups: np.ndarray | pd.Series | list | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    r"""Evaluate classifiers across candidate sample sizes.

    For each classifier and candidate sample size, performs *n_draws* rounds
    of stratified sampling proportional to the input class distribution. When
    no external test set is supplied, metrics are averaged over 5-fold
    stratified cross-validation. When *test_data* and *test_groups* are
    supplied, each classifier is trained on the complete candidate subset and
    evaluated once on those fixed external rows.

    The returned ``total_size`` is the candidate subset size. Internal
    cross-validation trains each fold on about 80% of that subset; external
    evaluation trains on the complete subset.

    Parameters
    ----------
    data : pd.DataFrame or SyngResult
        The dataset to evaluate. When a :class:`~syng_bts.result.SyngResult`
        is provided, the *which* parameter selects the data attribute and
        groups are auto-resolved from the corresponding ``*_groups`` field.
    sample_sizes : list[int], np.ndarray, pd.Series, or int
        Candidate sample sizes to evaluate.  Accepts a list, numpy array,
        or pandas Series of positive integers.  When a **single int** is
        provided it is interpreted as the *number* of equidistant sizes to
        create — the maximum equals the number of data rows.  For example,
        ``sample_sizes=3`` with 15-row data produces ``[5, 10, 15]``.
        The grid count cannot exceed the number of data rows.
    groups : array-like or None
        Class labels corresponding to the rows of *data*. **Required**
        when *data* is a ``pd.DataFrame``. When provided alongside a
        ``SyngResult``, overrides the auto-resolved groups.
    which : str, default ``"generated"``
        Selector when *data* is a ``SyngResult``:
        ``"generated"``, ``"original"``, or ``"reconstructed"``.
    n_draws : int, default 5
        Number of resampling repetitions for each sample size.
    apply_log : bool, default True
        When ``True``, a ``log2(x + 1)`` transform is applied to the candidate
        and external data before evaluation.
    methods : list[str] or None
        Classifier names to evaluate. Accepts canonical names
        (``'LOGIS'``, ``'SVM'``, ``'KNN'``, ``'RF'``, ``'XGB'``) and
        common aliases (``'LOGISTIC'``, ``'LR'``, ``'RANDOM_FOREST'``,
        ``'XGBOOST'``). Defaults to all five classifiers.
    verbose : int or str, default "minimal"
        Controls output verbosity.  Accepts ``0`` / ``"silent"`` (no
        output), ``1`` / ``"minimal"`` (one dynamic overall progress bar
        across all sample sizes, draws, and methods), or ``2`` /
        ``"detailed"`` (per-draw/method metric
        lines).
    test_data : pd.DataFrame or None
        Fixed external evaluation data. Must have the same feature columns as
        *data*. When supplied, *test_groups* is also required. External rows
        are transformed using preprocessing fitted on each candidate subset.
    test_groups : array-like or None
        Class labels corresponding to the rows of *test_data*. Must be supplied
        together with *test_data* and use labels present in *groups*.
    random_seed : int or None
        Seed for candidate sampling, shuffled cross-validation, and stochastic
        classifiers.

    Returns
    -------
    pd.DataFrame
        Columns: ``total_size``, ``draw``, ``method``, ``f1_score``,
        ``accuracy``, ``auc``.

    Raises
    ------
    TypeError
        If *data* is not a ``pd.DataFrame`` or ``SyngResult``, or supplied
        *test_data* is not a ``pd.DataFrame``.
    ValueError
        If *groups* is missing when required, *which* is invalid,
        *methods* contains unknown names, *sample_sizes* is empty or
        contains non-positive values, or any sample size exceeds the
        number of available rows. Also raised when only one external argument
        is supplied or the external rows, labels, or feature columns are
        incompatible, or when numerical values are invalid.

    Examples
    --------
    Using a DataFrame:

    >>> df = pd.read_csv("mydata.csv")
    >>> groups = df.pop("group")
    >>> result = evaluate_sample_sizes(df, sample_sizes=[50, 100], groups=groups)

    Using a SyngResult:

    >>> from syng_bts import generate
    >>> sr = generate(data="BRCASubtypeSel_test", model="CVAE1-20", epoch=10)
    >>> result = evaluate_sample_sizes(sr, sample_sizes=[50], which="generated")

    Evaluating candidate data on a fixed empirical test set:

    >>> result = evaluate_sample_sizes(
    ...     df,
    ...     sample_sizes=[50, 100],
    ...     groups=groups,
    ...     test_data=empirical_test,
    ...     test_groups=empirical_test_groups,
    ... )
    """
    # --- Resolve verbose level ---
    verbose_level = _resolve_verbose(verbose)

    # --- Resolve evaluation mode ---
    if (test_data is None) != (test_groups is None):
        raise ValueError(
            "'test_data' and 'test_groups' must be provided together or both omitted."
        )
    external_mode = test_data is not None

    # --- Resolve data and groups ---
    resolved_data, resolved_groups = _resolve_data_and_groups(data, groups, which)

    # --- Validate data shape/content ---
    if resolved_data.shape[0] == 0 or resolved_data.shape[1] == 0:
        raise ValueError("'data' must have at least 1 row and 1 column.")
    non_numeric_cols = [
        col
        for col in resolved_data.columns
        if not pd.api.types.is_numeric_dtype(resolved_data[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            "'data' must contain only numeric columns; non-numeric columns: "
            f"{non_numeric_cols}"
        )
    data_values = resolved_data.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(data_values).all():
        raise ValueError("'data' must contain only finite values.")
    if apply_log and (data_values <= -1).any():
        raise ValueError("'data' values must be greater than -1 for log2(x + 1).")

    resolved_test_data: pd.DataFrame | None = None
    test_group_arr: np.ndarray | None = None
    if external_mode:
        if not isinstance(test_data, pd.DataFrame):
            raise TypeError(
                f"'test_data' must be a pd.DataFrame, got {type(test_data).__name__}"
            )
        if test_data.shape[0] == 0 or test_data.shape[1] == 0:
            raise ValueError("'test_data' must have at least 1 row and 1 column.")

        non_numeric_test_cols = [
            col
            for col in test_data.columns
            if not pd.api.types.is_numeric_dtype(test_data[col])
        ]
        if non_numeric_test_cols:
            raise ValueError(
                "'test_data' must contain only numeric columns; non-numeric "
                f"columns: {non_numeric_test_cols}"
            )
        test_values = test_data.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(test_values).all():
            raise ValueError("'test_data' must contain only finite values.")
        if apply_log and (test_values <= -1).any():
            raise ValueError(
                "'test_data' values must be greater than -1 for log2(x + 1)."
            )

        missing_test_cols = resolved_data.columns.difference(test_data.columns).tolist()
        extra_test_cols = test_data.columns.difference(resolved_data.columns).tolist()
        if (
            missing_test_cols
            or extra_test_cols
            or len(test_data.columns) != len(resolved_data.columns)
        ):
            raise ValueError(
                "'test_data' must have the same feature columns as 'data'; "
                f"missing={missing_test_cols}, unexpected={extra_test_cols}."
            )
        resolved_test_data = test_data.loc[:, resolved_data.columns].copy()

        test_group_arr = np.asarray(test_groups)
        if test_group_arr.ndim != 1:
            raise ValueError("'test_groups' must be one-dimensional.")
        if len(test_group_arr) != len(resolved_test_data):
            raise ValueError(
                "Length mismatch: 'test_groups' must have one label per "
                f"test-data row (test_groups={len(test_group_arr)}, "
                f"rows={len(resolved_test_data)})."
            )
        if len(test_group_arr) == 0:
            raise ValueError("'test_groups' must be non-empty.")
        if pd.isna(test_group_arr).any():
            raise ValueError("'test_groups' must not contain missing labels.")

    group_arr = np.asarray(resolved_groups)
    if group_arr.ndim != 1:
        raise ValueError("'groups' must be one-dimensional.")
    if len(group_arr) != len(resolved_data):
        raise ValueError(
            "Length mismatch: 'groups' must have one label per data row "
            f"(groups={len(group_arr)}, rows={len(resolved_data)})."
        )
    if len(group_arr) == 0:
        raise ValueError("'groups' must be non-empty.")
    if pd.isna(group_arr).any():
        raise ValueError("'groups' must not contain missing labels.")
    unique_labels = np.unique(group_arr.astype(str))
    if len(unique_labels) < 2:
        raise ValueError("At least two unique groups are required for evaluation.")

    # --- Resolve and validate methods ---
    resolved_methods = _resolve_methods(methods)

    # --- Validate random seed ---
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ValueError(
                f"'random_seed' must be an integer or None, got {random_seed!r}."
            )
        random_seed = int(random_seed)

    # --- Normalise sample_sizes to list[int] ---
    n_rows = len(resolved_data)

    if isinstance(sample_sizes, (np.ndarray, pd.Series)):
        sample_sizes = sample_sizes.tolist()  # type: ignore[assignment]

    if isinstance(sample_sizes, (int, np.integer)) and not isinstance(
        sample_sizes, bool
    ):
        k = int(sample_sizes)
        if k <= 0:
            raise ValueError(f"'sample_sizes' as int must be positive, got {k}.")
        if k > n_rows:
            raise ValueError(
                "'sample_sizes' grid count cannot exceed the number of data rows "
                f"({n_rows}), got {k}."
            )
        sample_sizes = np.round(np.linspace(n_rows / k, n_rows, k)).astype(int).tolist()
        if len(set(sample_sizes)) != k or any(size <= 0 for size in sample_sizes):
            raise ValueError(
                "'sample_sizes' scalar grid must contain positive, unique sizes."
            )

    if not sample_sizes:
        raise ValueError("'sample_sizes' must be a non-empty list of integers.")
    normalized_sample_sizes: list[int] = []
    for s in sample_sizes:
        if isinstance(s, bool) or not isinstance(s, Integral) or int(s) <= 0:
            raise ValueError(f"All sample sizes must be positive integers, got {s!r}.")
        normalized_sample_sizes.append(int(s))

    for s in normalized_sample_sizes:
        if s > n_rows:
            raise ValueError(f"Sample size {s} exceeds available rows ({n_rows}).")

    # --- Validate n_draws ---
    if not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError(f"'n_draws' must be a positive integer, got {n_draws!r}.")

    n_splits = 5

    # --- Apply log transform if requested ---
    if apply_log:
        resolved_data = np.log2(resolved_data + 1)
        if resolved_test_data is not None:
            resolved_test_data = np.log2(resolved_test_data + 1)

    # Ensure float64 before sklearn scaling to avoid float32 numerical-warning
    # spam on high-range expression data.
    if (resolved_data.dtypes == np.float32).any():
        resolved_data = resolved_data.astype(np.float64)

    # Encode groups as integer labels
    group_arr = np.array([str(item) for item in group_arr])
    unique_groups = np.unique(group_arr)
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    labels = np.array([group_dict[g] for g in group_arr])

    external_data: np.ndarray | None = None
    external_labels: np.ndarray | None = None
    if test_group_arr is not None:
        test_group_arr = np.array([str(item) for item in test_group_arr])
        unknown_test_groups = sorted(set(test_group_arr) - set(group_dict))
        if unknown_test_groups:
            raise ValueError(
                "'test_groups' contains labels not present in 'groups': "
                f"{unknown_test_groups}."
            )
        missing_test_groups = sorted(set(group_dict) - set(test_group_arr))
        if missing_test_groups:
            raise ValueError(
                "'test_groups' must include all classes present in 'groups'; "
                f"missing={missing_test_groups}."
            )
        external_labels = np.array([group_dict[g] for g in test_group_arr])
        assert resolved_test_data is not None
        external_data = resolved_test_data.to_numpy(dtype=np.float64, copy=True)

    # Compute class proportions and per-group indices
    group_counts = {g: int(np.sum(group_arr == g)) for g in unique_groups}
    group_indices_dict = {g: np.where(group_arr == g)[0] for g in unique_groups}

    # Feasibility checks per requested sample size and evaluation mode
    for s in normalized_sample_sizes:
        counts = _allocate_stratified_counts(s, group_counts)
        empty_groups = [group for group, count in counts.items() if count < 1]
        if empty_groups:
            raise ValueError(
                "Sample size yields no candidate rows for one or more classes: "
                f"n={s}, groups={empty_groups}."
            )

        if external_mode:
            if "LOGIS" in resolved_methods:
                too_small_logis = [
                    group for group, count in counts.items() if count < n_splits
                ]
                if too_small_logis:
                    raise ValueError(
                        "Sample size yields too few samples per class for the "
                        "LOGIS inner 5-fold CV. "
                        f"n={s}, groups={too_small_logis}."
                    )
            if "KNN" in resolved_methods and s < 5:
                raise ValueError(
                    f"KNN requires at least 5 candidate training rows, got n={s}."
                )
            continue

        if s < n_splits * len(unique_groups):
            raise ValueError(
                "Sample size is too small for 5-fold stratified CV across all "
                f"classes: n={s}, classes={len(unique_groups)}, minimum="
                f"{n_splits * len(unique_groups)}."
            )
        too_small_outer = [group for group, count in counts.items() if count < n_splits]
        if too_small_outer:
            raise ValueError(
                "Sample size yields too few samples per class for 5-fold "
                "stratified CV. Increase sample size or reduce class imbalance. "
                f"n={s}, groups={too_small_outer}."
            )
        if "LOGIS" in resolved_methods:
            too_small_inner = [
                group
                for group, count in counts.items()
                if count - int(np.ceil(count / n_splits)) < n_splits
            ]
            if too_small_inner:
                raise ValueError(
                    "Sample size leaves too few samples per class for the LOGIS "
                    "inner 5-fold CV after the outer split. "
                    f"n={s}, groups={too_small_inner}."
                )

    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    results: list[dict] = []
    total_steps_overall = len(normalized_sample_sizes) * n_draws * len(resolved_methods)
    overall_step_counter = 0

    for n_index, n in enumerate(normalized_sample_sizes):
        if verbose_level >= VerbosityLevel.DETAILED:
            print(
                f"\nRunning sample size index "
                f"{n_index + 1}/{len(normalized_sample_sizes)} (n = {n})\n"
            )
        for draw in range(n_draws):
            # Stratified subsample
            indices: list[int] = []
            allocation = _allocate_stratified_counts(n, group_counts)
            for g in unique_groups:
                n_g = allocation[g]
                if rng is None:
                    selected = np.random.choice(
                        group_indices_dict[g], n_g, replace=False
                    )
                else:
                    selected = rng.choice(group_indices_dict[g], n_g, replace=False)
                indices.extend(selected)
            idx = np.array(indices)

            dat_candidate = resolved_data.iloc[idx].values
            labels_candidate = labels[idx]

            # Accumulate per-fold metrics per classifier
            metrics: dict[str, dict[str, list]] = {
                method: {"f1": [], "accuracy": [], "auc": []}
                for method in resolved_methods
            }

            if external_mode:
                split_indices = [(np.arange(len(dat_candidate)), None)]
            else:
                skf = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_seed,
                )
                split_indices = skf.split(dat_candidate, labels_candidate)

            for train_index, test_index in split_indices:
                train_data = dat_candidate[train_index].astype(np.float64, copy=True)
                train_labels = labels_candidate[train_index]

                if test_index is None:
                    assert external_data is not None
                    assert external_labels is not None
                    evaluation_data = external_data.copy()
                    evaluation_labels = external_labels
                else:
                    evaluation_data = dat_candidate[test_index].astype(
                        np.float64, copy=True
                    )
                    evaluation_labels = labels_candidate[test_index]

                # Fit preprocessing on training data and reuse it for evaluation
                non_zero_std = train_data.std(axis=0) != 0
                scaler = StandardScaler()
                train_data[:, non_zero_std] = scaler.fit_transform(
                    train_data[:, non_zero_std]
                )
                evaluation_data[:, non_zero_std] = scaler.transform(
                    evaluation_data[:, non_zero_std]
                )

                for method in resolved_methods:
                    clf_func = _CLASSIFIER_MAP[method]
                    res = clf_func(
                        train_data,
                        train_labels,
                        evaluation_data,
                        evaluation_labels,
                        random_seed,
                    )
                    metrics[method]["f1"].append(res["f1"])
                    metrics[method]["accuracy"].append(res["accuracy"])
                    metrics[method]["auc"].append(res["auc"])

            for method in resolved_methods:
                mean_f1 = float(np.mean(metrics[method]["f1"]))
                mean_acc = float(np.mean(metrics[method]["accuracy"]))
                mean_auc = float(np.mean(metrics[method]["auc"]))
                overall_step_counter += 1
                if verbose_level == VerbosityLevel.MINIMAL:
                    _print_eval_progress(
                        step=overall_step_counter,
                        total_steps=total_steps_overall,
                        size_index=n_index,
                        n_sizes=len(normalized_sample_sizes),
                        n=n,
                        draw=draw,
                        method=method,
                    )
                elif verbose_level >= VerbosityLevel.DETAILED:
                    print(
                        f"[n={n}, draw={draw}, method={method}] "
                        f"F1: {mean_f1:.4f}, Acc: {mean_acc:.4f}, "
                        f"AUC: {mean_auc:.4f}"
                    )
                results.append(
                    {
                        "total_size": n,
                        "draw": draw,
                        "method": method,
                        "f1_score": mean_f1,
                        "accuracy": mean_acc,
                        "auc": mean_auc,
                    }
                )
    if verbose_level == VerbosityLevel.MINIMAL:
        print()  # move past final \r line

    return pd.DataFrame(results)


def plot_sample_sizes(
    metric_real: pd.DataFrame,
    metric_generated: pd.DataFrame | None = None,
    metric_name: str = "f1_score",
    y_limits: tuple[float, float] | None = (0.4, 1),
) -> plt.Figure:
    r"""Visualize IPLF learning curves fitted from evaluation metrics.

    Fits weighted inverse power-law curves to the evaluation metrics produced by
    :func:`evaluate_sample_sizes` and plots observed values, fitted curves,
    and approximate pointwise 95% confidence intervals for the fitted mean
    curves. These bands are not prediction intervals. Three distinct sample
    sizes are sufficient to fit the curve, but at least four fitted points are
    required to estimate parameter covariance and display a confidence band.

    The returned figure is never displayed automatically — call
    ``fig.savefig(...)`` or ``plt.show()`` explicitly to display or save.

    Parameters
    ----------
    metric_real : pd.DataFrame
        Metrics from :func:`evaluate_sample_sizes` on real data.
    metric_generated : pd.DataFrame or None
        Metrics from :func:`evaluate_sample_sizes` on generated data.
        When provided, a second column of panels is added.
    metric_name : str, default ``"f1_score"``
        Metric to visualize (``"f1_score"``, ``"accuracy"``, or ``"auc"``).
    y_limits : tuple of float or None, default ``(0.4, 1)``
        Limits applied to the y-axis of every panel. Set to ``None`` to use
        Matplotlib's automatic scaling.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the learning-curve panels.

    Examples
    --------
    >>> metrics = evaluate_sample_sizes(df, [50, 100, 150, 200], groups=g)
    >>> fig = plot_sample_sizes(metrics)
    >>> fig.savefig("learning_curves.png")
    """
    valid_metric_names = {"f1_score", "accuracy", "auc"}
    if metric_name not in valid_metric_names:
        raise ValueError(
            f"Invalid metric_name {metric_name!r}. "
            f"Valid options: {sorted(valid_metric_names)}"
        )

    required_cols = {"total_size", "draw", "method", metric_name}
    missing_real = required_cols - set(metric_real.columns)
    if missing_real:
        raise ValueError(
            f"metric_real is missing required columns: {sorted(missing_real)}"
        )
    if metric_real.empty:
        raise ValueError("metric_real must be non-empty.")

    if metric_generated is not None:
        missing_generated = required_cols - set(metric_generated.columns)
        if missing_generated:
            raise ValueError(
                "metric_generated is missing required columns: "
                f"{sorted(missing_generated)}"
            )

    methods = metric_real["method"].unique()
    num_methods = len(methods)
    if metric_generated is not None:
        for method in methods:
            if method not in set(metric_generated["method"]):
                raise ValueError(
                    "metric_generated must include rows for every method in "
                    f"metric_real. Missing method: {method!r}."
                )

    real_fits = fit_sample_sizes(metric_real, metric_name)
    generated_fits = (
        fit_sample_sizes(metric_generated, metric_name)
        if metric_generated is not None
        else {}
    )

    cols = 2 if metric_generated is not None else 1
    fig, axs = plt.subplots(num_methods, cols, figsize=(15, 5 * num_methods))

    # Normalise axes array for uniform indexing
    if num_methods == 1 and cols == 1:
        axs = np.array([[axs]])
    elif num_methods == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = axs.reshape(-1, 1)

    for i, method in enumerate(methods):
        _plot_fit(real_fits[method], axs[i, 0], f"{method}: Real ({metric_name})")
        if y_limits is not None:
            axs[i, 0].set_ylim(y_limits)
        if metric_generated is not None:
            _plot_fit(
                generated_fits[method],
                axs[i, 1],
                f"{method}: Generated ({metric_name})",
            )
            if y_limits is not None:
                axs[i, 1].set_ylim(y_limits)

    fig.tight_layout()
    return fig
