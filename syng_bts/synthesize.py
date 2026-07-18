"""SyntheSize integration — sample-size evaluation via classifier learning curves.

This module provides classifier-based evaluation of synthetic data across
candidate sample sizes, using either stratified cross-validation or a fixed
external evaluation set, plus inverse power-law curve fitting.

Public API
----------
- :func:`evaluate_sample_sizes` — Evaluate classifiers across candidate sample
  sizes using stratified cross-validation or a fixed external evaluation set.
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
from numbers import Integral
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _power_law_gradient(x: float, a: float, b: float, c: float) -> np.ndarray:
    """Gradient of :func:`_power_law` with respect to ``(a, b, c)``."""
    x_power_c = x**c
    return np.array([-1.0, -x_power_c, -b * x_power_c * np.log(x)])


def _power_law_prediction_variance(
    x: float,
    params: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Propagate parameter covariance to fitted-curve variance at ``x``."""
    gradient = _power_law_gradient(x, *params)
    return float(gradient @ covariance @ gradient.T)


def _fit_curve(
    acc_table: pd.DataFrame,
    metric_name: str,
    plot: bool = True,
    ax: plt.Axes | None = None,
    annotation: str = "",
) -> plt.Axes | None:
    """Fit a weighted inverse power-law curve to evaluation metrics.

    After sorting by candidate size, applies the R implementation's increasing
    row weights ``1/m, 2/m, ..., m/m`` for *m* curve points.

    Parameters
    ----------
    acc_table : pd.DataFrame
        Must contain columns ``"n"`` and *metric_name*.
    metric_name : str
        Column in *acc_table* to fit against.
    plot : bool
        Whether to create a plot.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when ``None``.
    annotation : str
        Subplot title.

    Returns
    -------
    matplotlib Axes or None
    """
    acc_table = acc_table.sort_values("n").copy()
    initial_params = [0, 1, -0.5]
    max_iterations = 50000
    fit_ok = False
    ci_ok = False
    warning_context = f" for {annotation}" if annotation else ""

    try:
        if acc_table["n"].nunique() < 3:
            raise ValueError("at least three distinct sample sizes are required")
        weights = np.arange(1, len(acc_table) + 1) / len(acc_table)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", OptimizeWarning)
            popt, pcov = curve_fit(
                _power_law,
                acc_table["n"],
                acc_table[metric_name],
                p0=initial_params,
                sigma=1 / np.sqrt(weights),
                maxfev=max_iterations,
            )

        if not np.isfinite(popt).all():
            raise ValueError("optimizer returned non-finite parameters")

        acc_table["predicted"] = _power_law(acc_table["n"], *popt)
        if not np.isfinite(acc_table["predicted"]).all():
            raise ValueError("optimizer returned non-finite fitted values")
        fit_ok = True

        optimizer_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, OptimizeWarning)
        ]
        if optimizer_warnings:
            warning_messages = "; ".join(
                str(warning.message) for warning in optimizer_warnings
            )
            warnings.warn(
                f"Curve fit covariance warning{warning_context}: "
                f"{warning_messages}; the confidence band is omitted.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not np.isfinite(pcov).all():
            warnings.warn(
                f"Curve fit covariance is non-finite{warning_context}; "
                "the confidence band is omitted.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # Pointwise confidence intervals for the fitted mean curve via delta method
            pred_var = np.array(
                [
                    _power_law_prediction_variance(float(x), popt, pcov)
                    for x in acc_table["n"]
                ]
            )
            if not np.isfinite(pred_var).all() or (pred_var < 0).any():
                warnings.warn(
                    f"Curve fit covariance is unusable{warning_context}; "
                    "the confidence band is omitted.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                pred_std = np.sqrt(pred_var)
                t = norm.ppf(0.975)
                acc_table["ci_low"] = acc_table["predicted"] - t * pred_std
                acc_table["ci_high"] = acc_table["predicted"] + t * pred_std
                ci_ok = True
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Curve fit failed{warning_context}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            acc_table["n"],
            acc_table[metric_name],
            label="Actual Data",
            color="red",
        )
        if fit_ok:
            ax.plot(
                acc_table["n"],
                acc_table["predicted"],
                label="Fitted",
                color="blue",
                linestyle="--",
            )
            if ci_ok:
                ax.fill_between(
                    acc_table["n"],
                    acc_table["ci_low"],
                    acc_table["ci_high"],
                    color="blue",
                    alpha=0.2,
                    label="95% CI",
                )
        ax.set_xlabel("Candidate subset size")
        ax.legend(loc="best")
        ax.set_title(annotation)
        return ax

    return None


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

    cols = 2 if metric_generated is not None else 1
    fig, axs = plt.subplots(num_methods, cols, figsize=(15, 5 * num_methods))

    # Normalise axes array for uniform indexing
    if num_methods == 1 and cols == 1:
        axs = np.array([[axs]])
    elif num_methods == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = axs.reshape(-1, 1)

    def _mean_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        return (
            df.groupby(["total_size", "method"])
            .agg({metric: "mean"})
            .reset_index()
            .rename(columns={"total_size": "n"})
        )

    for i, method in enumerate(methods):
        df_real = metric_real[metric_real["method"] == method]
        mean_real = _mean_metrics(df_real, metric_name)

        _fit_curve(
            mean_real,
            metric_name,
            plot=True,
            ax=axs[i, 0],
            annotation=f"{method}: Real ({metric_name})",
        )
        if y_limits is not None:
            axs[i, 0].set_ylim(y_limits)

        if metric_generated is not None:
            df_gen = metric_generated[metric_generated["method"] == method]
            if df_gen.empty:
                raise ValueError(
                    "metric_generated must include rows for every method in "
                    f"metric_real. Missing method: {method!r}."
                )
            mean_gen = _mean_metrics(df_gen, metric_name)
            _fit_curve(
                mean_gen,
                metric_name,
                plot=True,
                ax=axs[i, 1],
                annotation=f"{method}: Generated ({metric_name})",
            )
            if y_limits is not None:
                axs[i, 1].set_ylim(y_limits)

    fig.tight_layout()
    return fig
