"""SyntheSize integration — sample-size evaluation via classifier learning curves.

This module ports the core classifier evaluation and visualization logic from
the SyntheSize project into SyNG-BTS. Currently exposes the legacy-compatible
functions ``eval_classifier`` and ``vis_classifier``; later will
introduce the redesigned public API (``evaluate_sample_sizes``,
``plot_sample_sizes``).

Only the minimum required helpers are included — ``heatmap_eval``,
``UMAP_eval``, and runtime package installers are **not** ported.

References
----------
- SyntheSize (R): https://github.com/LXQin/SyntheSize
- SyntheSize (Python): https://github.com/LXQin/SyntheSize_py
"""

from __future__ import annotations

from collections.abc import Callable

# Try to import matplotlib for visualization (should always be available)
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
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        solver="liblinear",
        scoring="accuracy",
        random_state=0,
        max_iter=1000,
    )
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


# Map public method names to private classifier callables
_CLASSIFIER_MAP: dict[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]
] = {
    "LOGIS": _logis,
    "SVM": _svm,
    "KNN": _knn,
    "RF": _rf,
    "XGB": _xgb,
}


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
    except RuntimeError:
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
# Public (legacy) API
# ---------------------------------------------------------------------------


def eval_classifier(
    whole_generated: pd.DataFrame,
    whole_groups: np.ndarray | pd.Series | list,
    n_candidate: list[int],
    n_draw: int = 5,
    log: bool = True,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    r"""Evaluate classifiers across candidate sample sizes.

    For each classifier and each candidate sample size, performs *n_draw*
    rounds of stratified sampling (proportional to class distribution),
    applies 5-fold cross-validation, and averages metrics across folds.

    This is the legacy-compatible entry point.  A redesigned
    public API will be introduced later.

    Parameters
    ----------
    whole_generated : pd.DataFrame
        The dataset to sample from (features only, numeric columns).
    whole_groups : array-like
        Class labels corresponding to the rows of *whole_generated*.
    n_candidate : list[int]
        Candidate sample sizes to evaluate.
    n_draw : int, default 5
        Number of resampling repetitions for each sample size.
    log : bool, default True
        Whether the input data is **already** log-transformed.  When
        ``False``, a ``log2(x + 1)`` transform is applied internally.
    methods : list[str] or None
        Classifier names to evaluate.  Defaults to
        ``['LOGIS', 'SVM', 'KNN', 'RF', 'XGB']``.

    Returns
    -------
    pd.DataFrame
        Columns: ``total_size``, ``draw``, ``method``, ``f1_score``,
        ``accuracy``, ``auc``.
    """
    if methods is None:
        methods = ["LOGIS", "SVM", "KNN", "RF", "XGB"]

    # Validate requested methods
    unknown = set(methods) - set(_CLASSIFIER_MAP)
    if unknown:
        raise ValueError(
            f"Unknown classifier method(s): {unknown}. "
            f"Valid options: {list(_CLASSIFIER_MAP)}"
        )

    # Optional log transform
    if not log:
        whole_generated = np.log2(whole_generated + 1)

    # Encode groups as integer labels
    whole_groups = np.array([str(item) for item in whole_groups])
    unique_groups = np.unique(whole_groups)
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    whole_labels = np.array([group_dict[g] for g in whole_groups])

    # Compute class proportions and per-group indices
    group_counts = {g: int(np.sum(whole_groups == g)) for g in unique_groups}
    total = sum(group_counts.values())
    group_proportions = {g: group_counts[g] / total for g in unique_groups}
    group_indices_dict = {g: np.where(whole_groups == g)[0] for g in unique_groups}

    results: list[dict] = []

    for n_index, n in enumerate(n_candidate):
        print(
            f"\nRunning sample size index {n_index + 1}/{len(n_candidate)} (n = {n})\n"
        )
        for draw in range(n_draw):
            # Stratified subsample
            indices: list[int] = []
            for g in unique_groups:
                n_g = int(round(n * group_proportions[g]))
                selected = np.random.choice(group_indices_dict[g], n_g, replace=False)
                indices.extend(selected)
            idx = np.array(indices)

            dat_candidate = whole_generated.iloc[idx].values
            labels_candidate = whole_labels[idx]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Accumulate per-fold metrics per classifier
            metrics: dict[str, dict[str, list]] = {
                method: {"f1": [], "accuracy": [], "auc": []} for method in methods
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

                for method in methods:
                    clf_func = _CLASSIFIER_MAP[method]
                    res = clf_func(train_data, train_labels, test_data, test_labels)
                    metrics[method]["f1"].append(res["f1"])
                    metrics[method]["accuracy"].append(res["accuracy"])
                    metrics[method]["auc"].append(res["auc"])

            for method in methods:
                mean_f1 = float(np.mean(metrics[method]["f1"]))
                mean_acc = float(np.mean(metrics[method]["accuracy"]))
                mean_auc = float(np.mean(metrics[method]["auc"]))
                print(
                    f"[n={n}, draw={draw}, method={method}] "
                    f"F1: {mean_f1:.4f}, Acc: {mean_acc:.4f}, AUC: {mean_auc:.4f}"
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


def vis_classifier(
    metric_real: pd.DataFrame,
    n_target: int | list,
    metric_generated: pd.DataFrame | None = None,
    metric_name: str = "f1_score",
    save: bool = False,
) -> plt.Figure | None:
    r"""Visualize IPLF learning curves fitted from evaluation metrics.

    Parameters
    ----------
    metric_real : pd.DataFrame
        Metrics from :func:`eval_classifier` on real data.
    n_target : int or list
        Target sample sizes for extrapolation reference.
    metric_generated : pd.DataFrame or None
        Metrics from :func:`eval_classifier` on generated data.
    metric_name : str, default ``"f1_score"``
        Metric to visualize (``"f1_score"``, ``"accuracy"``, or ``"auc"``).
    save : bool, default False
        When ``True``, the figure object is returned instead of ``None``.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure when *save* is ``True``, otherwise ``None``.
    """
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
            mean_gen = _mean_metrics(df_gen, metric_name)
            _fit_curve(
                mean_gen,
                metric_name,
                n_target=n_target,
                plot=True,
                ax=axs[i, 1],
                annotation=f"{method}: Generated ({metric_name})",
            )

    plt.tight_layout()

    if save:
        return fig
    return None
