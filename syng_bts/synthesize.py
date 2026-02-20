"""SyntheSize integration — sample-size evaluation via classifier learning curves.

This module provides classifier-based evaluation of synthetic data across
candidate sample sizes, using stratified cross-validation and inverse
power-law curve fitting.

Public API
----------
- :func:`evaluate_sample_sizes` — Evaluate classifiers across candidate sample
  sizes using stratified cross-validation.
- :func:`plot_sample_sizes` — Visualize IPLF learning curves from evaluation
  metrics.

References
----------
- SyntheSize (R): https://github.com/LXQin/SyntheSize
- SyntheSize (Python): https://github.com/LXQin/SyntheSize_py
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, curve_fit
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from xgboost import DMatrix
from xgboost import train as xgb_train

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
) -> dict[str, float]:
    """Ridge (L2-penalised) logistic regression classifier."""
    model_kwargs: dict[str, object] = {
        "Cs": 10,
        "cv": 5,
        "solver": "liblinear",
        "scoring": "accuracy",
        "random_state": 0,
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
) -> dict[str, float]:
    """Support Vector Machine classifier."""
    model = SVC(probability=True)
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
) -> dict[str, float]:
    """Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100)
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
) -> dict[str, float]:
    """XGBoost classifier."""
    num_class = len(np.unique(train_labels))
    dtrain = DMatrix(train_data, label=train_labels)
    dtest = DMatrix(test_data, label=test_labels)

    if num_class == 2:
        params = {"objective": "binary:logistic", "eval_metric": "auc"}
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
        }

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
    str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]
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


def _fit_curve(
    acc_table: pd.DataFrame,
    metric_name: str,
    n_target: int | list | None = None,
    plot: bool = True,
    ax: plt.Axes | None = None,
    annotation: str = "",
) -> plt.Axes | None:
    """Fit an inverse power-law curve to evaluation metrics.

    Parameters
    ----------
    acc_table : pd.DataFrame
        Must contain columns ``"n"`` and *metric_name*.
    metric_name : str
        Column in *acc_table* to fit against.
    n_target : int, list, or None
        Unused in this implementation (reserved for future extrapolation).
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
    acc_table = acc_table.copy()
    initial_params = [0, 1, -0.5]
    max_iterations = 50000
    fit_ok = False

    try:
        popt, pcov = curve_fit(
            _power_law,
            acc_table["n"],
            acc_table[metric_name],
            p0=initial_params,
            maxfev=max_iterations,
        )

        acc_table["predicted"] = _power_law(acc_table["n"], *popt)

        # Confidence intervals via delta method
        epsilon = np.sqrt(np.finfo(float).eps)
        jacobian = np.empty((len(acc_table["n"]), len(popt)))
        for i, x in enumerate(acc_table["n"]):
            jacobian[i] = approx_fprime(
                [x], lambda x_: _power_law(x_[0], *popt), epsilon
            )
        pred_var = np.sum((jacobian @ pcov) * jacobian, axis=1)
        pred_std = np.sqrt(pred_var)
        t = norm.ppf(0.975)
        acc_table["ci_low"] = acc_table["predicted"] - t * pred_std
        acc_table["ci_high"] = acc_table["predicted"] + t * pred_std
        fit_ok = True
    except (RuntimeError, ValueError):
        fit_ok = False

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
            ax.fill_between(
                acc_table["n"],
                acc_table["ci_low"],
                acc_table["ci_high"],
                color="blue",
                alpha=0.2,
                label="95% CI",
            )
        ax.set_xlabel("Sample Size")
        ax.legend(loc="best")
        ax.set_title(annotation)
        ax.set_ylim(0.4, 1)
        return ax

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_sample_sizes(
    data: pd.DataFrame | SyngResult,
    sample_sizes: list[int],
    groups: np.ndarray | pd.Series | list | None = None,
    which: str = "generated",
    n_draws: int = 5,
    apply_log: bool = True,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    r"""Evaluate classifiers across candidate sample sizes.

    For each classifier and each candidate sample size, performs *n_draws*
    rounds of stratified sampling (proportional to class distribution),
    applies 5-fold cross-validation, and averages metrics across folds.

    Parameters
    ----------
    data : pd.DataFrame or SyngResult
        The dataset to evaluate. When a :class:`~syng_bts.result.SyngResult`
        is provided, the *which* parameter selects the data attribute and
        groups are auto-resolved from the corresponding ``*_groups`` field.
    sample_sizes : list[int]
        Candidate sample sizes to evaluate.
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
        When ``True``, a ``log2(x + 1)`` transform is applied to the data
        before evaluation.
    methods : list[str] or None
        Classifier names to evaluate. Accepts canonical names
        (``'LOGIS'``, ``'SVM'``, ``'KNN'``, ``'RF'``, ``'XGB'``) and
        common aliases (``'LOGISTIC'``, ``'LR'``, ``'RANDOM_FOREST'``,
        ``'XGBOOST'``). Defaults to all five classifiers.

    Returns
    -------
    pd.DataFrame
        Columns: ``total_size``, ``draw``, ``method``, ``f1_score``,
        ``accuracy``, ``auc``.

    Raises
    ------
    TypeError
        If *data* is not a ``pd.DataFrame`` or ``SyngResult``.
    ValueError
        If *groups* is missing when required, *which* is invalid,
        *methods* contains unknown names, *sample_sizes* is empty or
        contains non-positive values, or any sample size exceeds the
        number of available rows.

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
    """
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
    unique_labels = np.unique(group_arr.astype(str))
    if len(unique_labels) < 2:
        raise ValueError("At least two unique groups are required for evaluation.")

    # --- Resolve and validate methods ---
    resolved_methods = _resolve_methods(methods)

    # --- Validate sample_sizes ---
    if not sample_sizes:
        raise ValueError("'sample_sizes' must be a non-empty list of integers.")
    normalized_sample_sizes: list[int] = []
    for s in sample_sizes:
        if isinstance(s, bool) or not isinstance(s, Integral) or int(s) <= 0:
            raise ValueError(f"All sample sizes must be positive integers, got {s!r}.")
        normalized_sample_sizes.append(int(s))

    n_rows = len(resolved_data)
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

    # Encode groups as integer labels
    group_arr = np.array([str(item) for item in group_arr])
    unique_groups = np.unique(group_arr)
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    labels = np.array([group_dict[g] for g in group_arr])

    # Compute class proportions and per-group indices
    group_counts = {g: int(np.sum(group_arr == g)) for g in unique_groups}
    group_indices_dict = {g: np.where(group_arr == g)[0] for g in unique_groups}

    # Feasibility checks per requested sample size for stratified 5-fold CV
    for s in normalized_sample_sizes:
        if s < n_splits * len(unique_groups):
            raise ValueError(
                "Sample size is too small for 5-fold stratified CV across all "
                f"classes: n={s}, classes={len(unique_groups)}, minimum="
                f"{n_splits * len(unique_groups)}."
            )
        counts = _allocate_stratified_counts(s, group_counts)
        too_small_groups = [group for group, c in counts.items() if c < n_splits]
        if too_small_groups:
            raise ValueError(
                "Sample size yields too few samples per class for 5-fold "
                "stratified CV. Increase sample size or reduce class imbalance. "
                f"n={s}, groups={too_small_groups}."
            )

    results: list[dict] = []

    for n_index, n in enumerate(normalized_sample_sizes):
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
                selected = np.random.choice(group_indices_dict[g], n_g, replace=False)
                indices.extend(selected)
            idx = np.array(indices)

            dat_candidate = resolved_data.iloc[idx].values
            labels_candidate = labels[idx]

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Accumulate per-fold metrics per classifier
            metrics: dict[str, dict[str, list]] = {
                method: {"f1": [], "accuracy": [], "auc": []}
                for method in resolved_methods
            }

            for train_index, test_index in skf.split(dat_candidate, labels_candidate):
                train_data = dat_candidate[train_index]
                test_data = dat_candidate[test_index]
                train_labels = labels_candidate[train_index]
                test_labels = labels_candidate[test_index]

                # Scale non-zero-variance features
                non_zero_std = train_data.std(axis=0) != 0
                train_data[:, non_zero_std] = scale(train_data[:, non_zero_std])
                test_data[:, non_zero_std] = scale(test_data[:, non_zero_std])

                for method in resolved_methods:
                    clf_func = _CLASSIFIER_MAP[method]
                    res = clf_func(train_data, train_labels, test_data, test_labels)
                    metrics[method]["f1"].append(res["f1"])
                    metrics[method]["accuracy"].append(res["accuracy"])
                    metrics[method]["auc"].append(res["auc"])

            for method in resolved_methods:
                mean_f1 = float(np.mean(metrics[method]["f1"]))
                mean_acc = float(np.mean(metrics[method]["accuracy"]))
                mean_auc = float(np.mean(metrics[method]["auc"]))
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

    return pd.DataFrame(results)


def plot_sample_sizes(
    metric_real: pd.DataFrame,
    n_target: int | list,
    metric_generated: pd.DataFrame | None = None,
    metric_name: str = "f1_score",
) -> plt.Figure:
    r"""Visualize IPLF learning curves fitted from evaluation metrics.

    Fits inverse power-law curves to the evaluation metrics produced by
    :func:`evaluate_sample_sizes` and plots observed values, fitted curves,
    and 95% confidence intervals.

    The returned figure is never displayed automatically — call
    ``fig.savefig(...)`` or ``plt.show()`` explicitly to display or save.

    Parameters
    ----------
    metric_real : pd.DataFrame
        Metrics from :func:`evaluate_sample_sizes` on real data.
    n_target : int or list
        Target sample sizes for extrapolation reference.
    metric_generated : pd.DataFrame or None
        Metrics from :func:`evaluate_sample_sizes` on generated data.
        When provided, a second column of panels is added.
    metric_name : str, default ``"f1_score"``
        Metric to visualize (``"f1_score"``, ``"accuracy"``, or ``"auc"``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the learning-curve panels.

    Examples
    --------
    >>> metrics = evaluate_sample_sizes(df, [50, 100, 200], groups=g)
    >>> fig = plot_sample_sizes(metrics, n_target=300)
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
            n_target=n_target,
            plot=True,
            ax=axs[i, 0],
            annotation=f"{method}: Real ({metric_name})",
        )

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
                n_target=n_target,
                plot=True,
                ax=axs[i, 1],
                annotation=f"{method}: Generated ({metric_name})",
            )

    fig.tight_layout()
    return fig
