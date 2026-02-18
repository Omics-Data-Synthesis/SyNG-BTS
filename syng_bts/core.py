"""Core experiment functions for SyNG-BTS.

Public API
----------
- ``generate()`` — train a model and produce synthetic samples.
- ``pilot_study()`` — sweep over pilot sizes with replicated draws.
- ``transfer()`` — pre-train on source, fine-tune on target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_utils import _derive_dataname, _validate_feature_data, resolve_data
from .helper_train import VerbosityLevel, _resolve_verbose
from .helper_training import (
    TrainedModel,
    training_AEs,
    training_flows,
    training_GANs,
    training_iter,
)
from .helper_utils import (
    Gaussian_aug,
    create_labels,
    draw_pilot,
    inverse_log2,
    preprocessinglog2,
    set_all_seeds,
)
from .inference import run_generation, run_reconstruction
from .result import PilotResult, SyngResult

# =========================================================================
# Private helpers
# =========================================================================


@dataclass
class PreparedData:
    """Container for pre-processed data shared across public API functions.

    Holds everything produced by :func:`_prepare_data` so that
    ``generate()``, ``pilot_study()``, and ``transfer()`` can
    consume a single, validated object instead of duplicating the
    resolve → validate → convert → label pipeline.
    """

    df: pd.DataFrame
    """Original DataFrame (unmodified)."""

    colnames: list[str]
    """Column names from the original DataFrame."""

    oridata: torch.Tensor
    """Numeric data as a float32 tensor, optionally log-transformed."""

    n_samples: int
    """Number of samples (rows) in *oridata*."""

    orilabels: torch.Tensor
    """Label tensor (one-hot or single-column)."""

    oriblurlabels: torch.Tensor
    """Blurred-label tensor."""

    dataname: str
    """Short dataset name for metadata / filenames."""

    effective_groups: np.ndarray | None
    """Resolved group array (explicit > bundled > ``None``)."""

    apply_log: bool
    """Whether ``log2(x + 1)`` was applied to *oridata*."""


@dataclass
class TrainingContext:
    """Sidecar training context returned by :func:`orchestrate_training`.

    Contains non-architectural training runtime information needed for
    reconstruction parity and metadata assembly.  This avoids leaking
    private ``_train_*`` keys into ``TrainedModel.arch_params``.
    """

    random_seed: int
    """Random seed used for the training split."""

    val_ratio: float
    """Validation split ratio used during training."""

    batch_size: int
    """Computed batch size (from ``batch_frac``)."""

    num_epochs: int
    """Resolved maximum epoch count."""

    early_stop: bool
    """Whether early stopping was enabled."""

    early_stop_num: int
    """Early stopping patience value."""

    rawdata: torch.Tensor
    """Training data after blur-label appending and augmentation."""

    rawlabels: torch.Tensor
    """Training labels after augmentation."""


def _prepare_data(
    *,
    data: pd.DataFrame | str | Path,
    name: str | None,
    groups: pd.Series | np.ndarray | None,
    apply_log: bool,
) -> PreparedData:
    """Shared data-preparation pipeline for public API functions.

    Resolves the input data, validates it, derives the dataset name,
    converts to a float32 tensor (with optional log2 transform),
    resolves groups, and creates labels.

    Parameters
    ----------
    data : DataFrame, str, or Path
        Input data — a pandas DataFrame, a path to a CSV file, or the
        name of a bundled dataset.
    name : str or None
        Short name override.  Derived automatically when ``None``.
    groups : pd.Series, np.ndarray, or None
        Optional binary group labels.
    apply_log : bool
        Apply ``log2(x + 1)`` preprocessing.

    Returns
    -------
    PreparedData
    """
    df, bundled_groups = resolve_data(data)
    _validate_feature_data(df)

    # Light sanity checks when user requests automatic log2 preprocessing.
    # - If negatives are present, that's invalid for log2 and we raise.
    # - If data contains non-integer values, warn the user because the
    #   input may already be transformed (double-logging risk).
    if apply_log:
        arr = df.to_numpy()
        if (arr < 0).any():
            raise ValueError("Input contains negative values; cannot apply log2.")
        # Treat non-integer values as suspicious (non-fatal).
        if not (arr.shape[0] == 0 or np.allclose(arr, np.round(arr))):
            import warnings

            warnings.warn(
                "apply_log=True but input contains non-integer values — the data may already "
                "be log-transformed. To avoid double-logging, pass apply_log=False.",
                UserWarning,
                stacklevel=2,
            )

    dataname = _derive_dataname(data, name)

    colnames = list(df.columns)
    oridata = torch.from_numpy(df.to_numpy().copy()).to(torch.float32)

    if apply_log:
        oridata = preprocessinglog2(oridata)

    n_samples = oridata.shape[0]

    effective_groups = _resolve_effective_groups(
        groups,
        bundled_groups,
        n_samples=n_samples,
        param_name="groups",
    )

    orilabels, oriblurlabels = create_labels(
        n_samples=n_samples,
        groups=effective_groups,
    )

    return PreparedData(
        df=df,
        colnames=colnames,
        oridata=oridata,
        n_samples=n_samples,
        orilabels=orilabels,
        oriblurlabels=oriblurlabels,
        dataname=dataname,
        effective_groups=effective_groups,
        apply_log=apply_log,
    )


def _parse_model_spec(model: str) -> tuple[str, int]:
    """Parse a model string like ``'VAE1-10'`` into (modelname, kl_weight).

    Returns
    -------
    tuple[str, int]
        ``(modelname, kl_weight)`` — e.g. ``("VAE", 10)``.
    """
    parts = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)
    if len(parts) > 1:
        return parts[1], int(parts[4])
    return model, 1


def _build_loss_df(log_dict: dict, modelname: str) -> pd.DataFrame:
    """Convert a raw log_dict from a training helper into a tidy DataFrame.

    Parameters
    ----------
    log_dict : dict
        Raw loss series as returned by ``TrainedModel.log_dict``.
    modelname : str
        The short model name (``"AE"``, ``"VAE"``, ``"CVAE"``, ``"GAN"``,
        ``"WGAN"``, ``"WGANGP"``, ``"maf"``, ``"glow"``, ``"realnvp"``, etc.).

    Returns
    -------
    pd.DataFrame
    """
    if modelname == "AE":
        # AE logs total loss only (no KL/reconstruction split)
        return pd.DataFrame(
            {
                "train_loss": log_dict.get("train_loss_per_batch", []),
                "val_loss": log_dict.get("val_loss_per_batch", []),
            }
        )
    if modelname in ("VAE", "CVAE"):
        return pd.DataFrame(
            {
                "kl": log_dict.get(
                    "val_kl_loss_per_batch",
                    log_dict.get("train_kl_loss_per_batch", []),
                ),
                "recons": log_dict.get(
                    "val_reconstruction_loss_per_batch",
                    log_dict.get("train_reconstruction_loss_per_batch", []),
                ),
            }
        )
    if "GAN" in modelname:
        return pd.DataFrame(
            {
                "discriminator": log_dict["train_discriminator_loss_per_batch"],
                "generator": log_dict["train_generator_loss_per_batch"],
            }
        )
    # Flows — per-epoch loss
    return pd.DataFrame({"train_loss": log_dict["train_loss_per_epoch"]})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def orchestrate_training(
    *,
    rawdata: torch.Tensor,
    rawlabels: torch.Tensor,
    oriblurlabels: torch.Tensor,
    modelname: str,
    kl_weight: int = 1,
    batch_frac: float = 0.1,
    random_seed: int = 123,
    epoch: int | None = None,
    early_stop_patience: int | None = None,
    learning_rate: float = 0.0005,
    val_ratio: float = 0.2,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    model_state: dict | None = None,
    cap: bool = False,
    loss_fn: str = "MSE",
    use_scheduler: bool = False,
    step_size: int = 10,
    gamma: float = 0.5,
    verbose: int = 1,
) -> tuple[TrainedModel, TrainingContext]:
    """Centralized training orchestrator.

    Resolves early-stopping configuration, appends blur-labels for
    non-CVAE multi-group data, applies offline augmentation, computes
    batch size, and dispatches to the appropriate model-family
    training wrapper.

    Model parsing is **external** — the caller passes ``modelname``
    and ``kl_weight`` (see :func:`_parse_model_spec`).

    Parameters
    ----------
    rawdata : torch.Tensor
        Input data tensor (pre-blur-label, pre-augmentation).
    rawlabels : torch.Tensor
        Label tensor.
    oriblurlabels : torch.Tensor
        Blurred-label tensor for two-group training.
    modelname : str
        Short model name (``"AE"``, ``"VAE"``, ``"GAN"``, etc.).
    kl_weight : int
        KL divergence weight (VAE/CVAE only).
    batch_frac : float
        Batch size as a fraction of sample count.
    random_seed : int
        Random seed for reproducibility.
    epoch : int or None
        Fixed epoch count, or ``None`` for early stopping.
    early_stop_patience : int or None
        Early stopping patience, or ``None``.
    learning_rate : float
        Optimizer learning rate.
    val_ratio : float
        Validation split ratio (AE family only).
    off_aug : str or None
        Offline augmentation: ``"AE_head"``, ``"Gaussian_head"``,
        or ``None``.
    AE_head_num : int
        Fold multiplier for AE-head augmentation.
    Gaussian_head_num : int
        Fold multiplier for Gaussian-head augmentation.
    model_state : dict or None
        Pre-trained model state for transfer learning.
    cap : bool
        Cap generated values (AE family training).
    loss_fn : str
        Loss function name (AE family).
    use_scheduler : bool
        Enable learning-rate scheduler (AE family).
    step_size : int
        Scheduler step size.
    gamma : float
        Scheduler gamma.
    verbose : int
        Verbosity level.

    Returns
    -------
    tuple[TrainedModel, TrainingContext]
        The trained model and a sidecar context with runtime info
        needed for inference and metadata assembly.
    """
    # --- 1. Resolve early-stopping config --------------------------------
    num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        default_max_epochs=1000,
        default_patience=30,
    )

    # --- 2. Append blur-labels for non-CVAE two-group data ---------------
    if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
        rawdata = torch.cat((rawdata, oriblurlabels), dim=1)

    # --- 3. Offline augmentation -----------------------------------------
    if off_aug == "Gaussian_head":
        rawdata, rawlabels = Gaussian_aug(
            rawdata, rawlabels, multiplier=[Gaussian_head_num]
        )
    elif off_aug == "AE_head":
        # TODO Change hardcoded training config for AE head augmentation
        # to be more flexible
        feed_data, feed_labels = training_iter(
            iter_times=AE_head_num,
            rawdata=rawdata,
            rawlabels=rawlabels,
            random_seed=random_seed,
            modelname="AE",
            num_epochs=1000,
            batch_size=round(rawdata.shape[0] * 0.1),
            learning_rate=0.0005,
            early_stop=False,
            early_stop_num=30,
            kl_weight=1,
            loss_fn="MSE",
            replace=True,
            verbose=verbose,
        )
        rawdata = feed_data
        rawlabels = feed_labels

    # --- 4. Compute batch size -------------------------------------------
    batch_size = max(1, round(rawdata.shape[0] * batch_frac))

    # --- 5. Dispatch to model-family training ----------------------------
    if "GAN" in modelname:
        trained = training_GANs(
            rawdata=rawdata,
            rawlabels=rawlabels,
            batch_size=batch_size,
            random_seed=random_seed,
            modelname=modelname,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            model_state=model_state,
            verbose=verbose,
        )
    elif "AE" in modelname:
        trained = training_AEs(
            rawdata=rawdata,
            rawlabels=rawlabels,
            batch_size=batch_size,
            random_seed=random_seed,
            modelname=modelname,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            kl_weight=kl_weight,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            model_state=model_state,
            cap=cap,
            loss_fn=loss_fn,
            use_scheduler=use_scheduler,
            step_size=step_size,
            gamma=gamma,
            verbose=verbose,
        )
    elif modelname in ("maf", "realnvp", "glow", "maf-split", "maf-split-glow"):
        trained = training_flows(
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            num_hidden=226,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            model_state=model_state,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported model: {modelname!r}")

    ctx = TrainingContext(
        random_seed=random_seed,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stop=early_stop,
        early_stop_num=early_stop_num,
        rawdata=rawdata,
        rawlabels=rawlabels,
    )

    return trained, ctx


def _infer_from_trained(
    trained: TrainedModel,
    *,
    new_size: int | list[int],
    ctx: TrainingContext,
    cap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run generation and reconstruction via the unified inference dispatcher.

    This delegates to :func:`inference.run_generation` and
    :func:`inference.run_reconstruction`, keeping post-training
    inference cleanly separated from training orchestrators.

    Parameters
    ----------
    trained : TrainedModel
        Output from the training dispatch layer.
    new_size : int or list[int]
        Number of synthetic samples to generate.
    ctx : TrainingContext
        Sidecar context from :func:`orchestrate_training` containing
        the training data, batch size, and split parameters needed
        for reconstruction parity.
    cap : bool
        Cap generated values to observed range.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        ``(generated_data, reconstructed_data)``
    """
    from torch.utils.data import DataLoader, TensorDataset, random_split

    family = trained.arch_params["family"]

    # --- Determine capping values ---
    if cap:
        col_max, _ = torch.max(ctx.rawdata, dim=0)
        col_sd = torch.std(ctx.rawdata, dim=0, unbiased=True)
    else:
        col_max = None
        col_sd = None

    # --- Generation via unified dispatcher ---
    generated = run_generation(
        trained,
        num_samples=new_size,
        col_max=col_max,
        col_sd=col_sd,
    )

    # --- Reconstruction via unified dispatcher (AE family only) ---
    reconstructed = None
    if family == "ae":
        num_features = trained.arch_params["num_features"]
        data = TensorDataset(ctx.rawdata, ctx.rawlabels)

        # Reproduce legacy reconstruction path as closely as possible:
        # training_AEs reconstructs over the train split DataLoader
        # (shuffle=True, drop_last=True).
        set_all_seeds(ctx.random_seed)
        val_size = int(len(data) * ctx.val_ratio)
        train_size = len(data) - val_size
        train_dataset, _val_dataset = random_split(data, [train_size, val_size])
        recon_loader = DataLoader(
            train_dataset,
            batch_size=ctx.batch_size,
            shuffle=True,
            drop_last=True,
        )

        reconstructed, _ = run_reconstruction(
            trained,
            data_loader=recon_loader,
            n_features=num_features,
        )

    return generated, reconstructed


def _compute_new_size(
    orilabels: torch.Tensor,
    n_samples: int,
    new_size: int | list[int],
    repli: int = 5,
) -> int | list[int]:
    """Compute the generation size, honouring group balance.

    For a simple (single-group) dataset the returned value is just
    *new_size* as-is.  For a two-group dataset where the groups are
    unbalanced and *new_size* is not already a list, returns
    ``[n_class_0, n_class_1, repli]``.
    """
    if isinstance(new_size, list):
        return new_size
    if (len(torch.unique(orilabels)) > 1) and (
        int(sum(orilabels == 0)) != int(sum(orilabels == 1))
    ):
        return [int(sum(orilabels == 0)), int(sum(orilabels == 1)), repli]
    return new_size


def _assemble_result(
    *,
    gen_data: torch.Tensor,
    recon_data: torch.Tensor | None,
    trained: TrainedModel,
    colnames: list[str],
    modelname: str,
    model: str,
    dataname: str,
    n_samples: int,
    num_epochs: int,
    random_seed: int,
    kl_weight: int,
    early_stop: bool,
    early_stop_num: int,
    apply_log: bool,
    original_data: pd.DataFrame,
    extra_metadata: dict | None = None,
) -> SyngResult:
    """Assemble a :class:`SyngResult` from training/inference outputs.

    Centralises CVAE label stripping, DataFrame construction, inverse
    log transform, column-order validation, loss assembly, metadata
    assembly, and ``SyngResult`` construction.

    Parameters
    ----------
    gen_data : torch.Tensor
        Raw generated data tensor.
    recon_data : torch.Tensor or None
        Raw reconstructed data tensor (AE-family only).
    trained : TrainedModel
        Training output from the dispatch layer.
    colnames : list[str]
        Original column names.
    modelname : str
        Short model name (e.g. ``"VAE"``, ``"CVAE"``).
    model : str
        Full model specification string (e.g. ``"VAE1-10"``).
    dataname : str
        Dataset name for metadata.
    n_samples : int
        Number of original samples (for ``input_shape``).
    num_epochs : int
        Maximum epoch count configured.
    random_seed : int
        Random seed used.
    kl_weight : int
        KL weight used.
    early_stop : bool
        Whether early stopping was enabled.
    early_stop_num : int
        Early stopping patience value.
    apply_log : bool
        Whether ``log2(x + 1)`` preprocessing was applied.
    original_data : pd.DataFrame
        Original data subset to attach to the result.
    extra_metadata : dict or None
        Additional metadata entries (e.g. ``pilot_size``, ``draw``,
        ``pilot_indices``).

    Returns
    -------
    SyngResult
    """
    gen_np = gen_data.detach().numpy()

    # For CVAE, strip the appended label column — it is the conditioning
    # input, not a generated feature.  Store it in metadata instead.
    gen_labels = None
    if modelname == "CVAE" and gen_np.shape[1] > len(colnames):
        gen_labels = pd.Series(gen_np[:, -1], name="label")
        gen_np = gen_np[:, :-1]

    gen_df = pd.DataFrame(gen_np, columns=list(colnames))

    recon_df = None
    recon_labels = None
    if recon_data is not None:
        recon_np = recon_data.detach().numpy()

        if modelname == "CVAE" and recon_np.shape[1] > len(colnames):
            recon_labels = pd.Series(recon_np[:, -1], name="label")
            recon_np = recon_np[:, :-1]

        recon_df = pd.DataFrame(recon_np, columns=list(colnames))

    # Inverse log transform to count scale
    if apply_log:
        gen_df = inverse_log2(gen_df)
        if recon_df is not None:
            recon_df = inverse_log2(recon_df)

    # Validate column order consistency
    if not gen_df.columns.tolist() == colnames:
        raise RuntimeError(
            "Column order mismatch in generated_data. "
            "This is an internal error; please report it."
        )
    if recon_df is not None and not recon_df.columns.tolist() == colnames:
        raise RuntimeError(
            "Column order mismatch in reconstructed_data. "
            "This is an internal error; please report it."
        )

    loss_df = _build_loss_df(trained.log_dict, modelname)

    metadata: dict = {
        "model": model,
        "modelname": modelname,
        "dataname": dataname,
        "num_epochs": num_epochs,
        "epochs_trained": trained.epochs_trained,
        "seed": random_seed,
        "kl_weight": kl_weight,
        "input_shape": (n_samples, len(colnames)),
        "early_stop": early_stop,
        "early_stop_patience": early_stop_num,
        "generated_labels": gen_labels,
        "reconstructed_labels": recon_labels,
        "apply_log": apply_log,
        "arch_params": {
            k: v for k, v in trained.arch_params.items() if not k.startswith("_")
        },
    }

    if extra_metadata is not None:
        overlapping = set(extra_metadata).intersection(metadata)
        if overlapping:
            overlap_str = ", ".join(sorted(overlapping))
            raise ValueError(
                f"extra_metadata contains reserved metadata keys: {overlap_str}"
            )
        metadata.update(extra_metadata)

    return SyngResult(
        generated_data=gen_df,
        loss=loss_df,
        reconstructed_data=recon_df,
        original_data=original_data,
        model_state=trained.model_state,
        metadata=metadata,
    )


def _resolve_early_stopping_config(
    epoch: int | None,
    early_stop_patience: int | None,
    default_max_epochs: int = 1000,
    default_patience: int = 30,
) -> tuple[int, bool, int]:
    """Resolve epoch count and early stopping configuration.

    Parameters
    ----------
    epoch : int or None
        User-specified fixed epoch count.
    early_stop_patience : int or None
        User-specified early stopping patience.
    default_max_epochs : int
        Default maximum epochs when early stopping is enabled.
    default_patience : int
        Default patience when early stopping is enabled but patience not specified.

    Returns
    -------
    tuple[int, bool, int]
        ``(num_epochs, early_stop, early_stop_num)`` where:
        - ``num_epochs`` is the maximum epoch count to run
        - ``early_stop`` is whether early stopping is enabled
        - ``early_stop_num`` is the patience value to use
    """
    if epoch is not None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(f"epoch must be a positive integer or None, got {epoch!r}")

    if early_stop_patience is not None:
        if (
            isinstance(early_stop_patience, bool)
            or not isinstance(early_stop_patience, int)
            or early_stop_patience <= 0
        ):
            raise ValueError(
                "early_stop_patience must be a positive integer or None, "
                f"got {early_stop_patience!r}"
            )

    if epoch is not None and early_stop_patience is not None:
        # Both provided: run up to `epoch` epochs with early stopping
        return epoch, True, early_stop_patience
    elif epoch is not None:
        # Only epoch: run exactly that many epochs, no early stopping
        return epoch, False, default_patience
    elif early_stop_patience is not None:
        # Only patience: early stop with default max epochs
        return default_max_epochs, True, early_stop_patience
    else:
        # Neither: default early stopping with default patience
        return default_max_epochs, True, default_patience


def _validate_n_draws(n_draws: int, *, param_name: str = "n_draws") -> int:
    """Validate replicated draw count parameters.

    Parameters
    ----------
    n_draws : int
        Number of replicated random draws.
    param_name : str
        Parameter name used in error messages.

    Returns
    -------
    int
        Validated draw count.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws <= 0:
        raise ValueError(f"'{param_name}' must be a positive integer, got {n_draws!r}")
    return n_draws


def _coerce_groups_array(
    groups: pd.Series | np.ndarray,
    *,
    n_samples: int,
    param_name: str,
) -> np.ndarray:
    """Normalize user/bundled groups to a 1D numpy array."""
    if isinstance(groups, pd.Series):
        arr = groups.to_numpy()
    elif isinstance(groups, np.ndarray):
        arr = groups
    else:
        raise TypeError(
            f"'{param_name}' must be a pandas Series, numpy ndarray, or None, "
            f"got {type(groups).__name__}"
        )

    if arr.ndim != 1:
        raise ValueError(
            f"'{param_name}' must be one-dimensional, got shape {arr.shape}"
        )

    if arr.shape[0] != n_samples:
        raise ValueError(
            f"'{param_name}' length ({arr.shape[0]}) must match number of "
            f"samples ({n_samples})"
        )

    return arr


def _validate_binary_groups(groups: np.ndarray, *, param_name: str) -> None:
    """Enforce v3.1 binary-group scope (<= 2 distinct classes)."""
    unique = pd.Series(groups).dropna().unique()
    if len(unique) > 2:
        raise ValueError(
            f"'{param_name}' has {len(unique)} classes, but SyNG-BTS supports only "
            "binary groups (at most 2 classes)."
        )


def _resolve_effective_groups(
    explicit_groups: pd.Series | np.ndarray | None,
    bundled_groups: pd.Series | None,
    *,
    n_samples: int,
    param_name: str,
) -> np.ndarray | None:
    """Resolve groups using explicit precedence: argument > bundled."""
    candidate = explicit_groups if explicit_groups is not None else bundled_groups
    if candidate is None:
        return None

    group_array = _coerce_groups_array(
        candidate,
        n_samples=n_samples,
        param_name=param_name,
    )
    _validate_binary_groups(group_array, param_name=param_name)
    return group_array


# =========================================================================
# Public API
# =========================================================================


def generate(
    data: pd.DataFrame | str | Path,
    *,
    name: str | None = None,
    groups: pd.Series | np.ndarray | None = None,
    new_size: int | list[int] = 500,
    model: str = "VAE1-10",
    apply_log: bool = True,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    val_ratio: float = 0.2,
    early_stop_patience: int | None = None,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    use_scheduler: bool = False,
    step_size: int = 10,
    gamma: float = 0.5,
    cap: bool = False,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
    verbose: int | str = "minimal",
) -> SyngResult:
    """Train a deep generative model and generate synthetic data.

    This is the primary entry point for training a single model and
    generating synthetic samples.  It replaces the legacy
    ``ApplyExperiment`` function.

    Parameters
    ----------
    data : DataFrame, str, or Path
        Input data — a pandas DataFrame, a path to a CSV file, or the
        name of a bundled dataset (e.g. ``"SKCMPositive_4"``).
    name : str or None
        Short name for output filenames.  Derived automatically when
        ``None``.
    groups : pd.Series, np.ndarray, or None
        Optional binary group labels. When provided, these labels take
        precedence over bundled dataset groups.
    new_size : int or list[int]
        Number of synthetic samples to generate.
    model : str
        Model specification, e.g. ``"VAE1-10"`` (parsed into model type
        and kl_weight).
    apply_log : bool
        Apply ``log2(x + 1)`` preprocessing.
    batch_frac : float
        Batch size as a fraction of sample count.
    learning_rate : float
        Optimizer learning rate.
    epoch : int or None
        Fixed epoch count, or ``None`` for early stopping.

        The interaction between *epoch* and *early_stop_patience*:

        ===========  =======================  ==================================================
        ``epoch``    ``early_stop_patience``  Behaviour
        ===========  =======================  ==================================================
        ``None``     ``None``                 Early stopping ON, patience=30, max 1 000 epochs
        ``None``     ``30``                   Early stopping ON, patience=30, max 1 000 epochs
        ``500``      ``None``                 Early stopping OFF, run exactly 500 epochs
        ``500``      ``30``                   Early stopping ON, patience=30, max 500 epochs
        ===========  =======================  ==================================================
    val_ratio : float
        Validation split ratio (AE family only).
    early_stop_patience : int or None
        Stop if loss does not improve for this many epochs.  When ``None``
        and ``epoch`` is also ``None``, defaults to ``30``.
    off_aug : str or None
        Offline augmentation: ``"AE_head"``, ``"Gaussian_head"``, or
        ``None``.
    AE_head_num : int
        Fold multiplier for AE-head augmentation.
    Gaussian_head_num : int
        Fold multiplier for Gaussian-head augmentation.
    use_scheduler : bool
        Enable learning-rate scheduler (AE family).
    step_size : int
        Scheduler step size.
    gamma : float
        Scheduler gamma.
    cap : bool
        Cap generated values to observed range.
    random_seed : int
        Random seed for reproducibility.
    output_dir : str, Path, or None
        If set, automatically save results to this directory.
    verbose : int or str
        Verbosity level for training output.

        - ``"silent"`` or ``0``  — no output during training.
        - ``"minimal"`` or ``1`` (default) — print only training
          summaries and early-stopping messages.
        - ``"detailed"`` or ``2`` — print per-epoch progress
          (epoch number, loss values, elapsed time, learning rate).

    Returns
    -------
    SyngResult
        Rich result object containing generated data, loss log,
        reconstructed data (AE/VAE/CVAE), model state, and metadata.
    """
    # --- 0. Resolve verbose level ----------------------------------------
    verbose_level = _resolve_verbose(verbose)

    # --- 1. Prepare data (resolve, validate, convert, label) -------------
    prep = _prepare_data(data=data, name=name, groups=groups, apply_log=apply_log)

    # --- 2. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 3. Compute new_size (group-balanced if needed) ------------------
    effective_new_size = _compute_new_size(prep.orilabels, prep.n_samples, new_size)

    # --- 4. Train (orchestrate: early-stop, blur-label, aug, dispatch) ---
    trained, ctx = orchestrate_training(
        rawdata=prep.oridata,
        rawlabels=prep.orilabels,
        oriblurlabels=prep.oriblurlabels,
        modelname=modelname,
        kl_weight=kl_weight,
        batch_frac=batch_frac,
        random_seed=random_seed,
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        off_aug=off_aug,
        AE_head_num=AE_head_num,
        Gaussian_head_num=Gaussian_head_num,
        cap=cap,
        loss_fn="MSE",
        use_scheduler=use_scheduler,
        step_size=step_size,
        gamma=gamma,
        verbose=verbose_level,
    )

    # --- 5. Infer --------------------------------------------------------
    gen_data, recon_data = _infer_from_trained(
        trained,
        new_size=effective_new_size,
        ctx=ctx,
        cap=cap,
    )

    # --- 6. Assemble SyngResult ------------------------------------------
    result = _assemble_result(
        gen_data=gen_data,
        recon_data=recon_data,
        trained=trained,
        colnames=prep.colnames,
        modelname=modelname,
        model=model,
        dataname=prep.dataname,
        n_samples=prep.n_samples,
        num_epochs=ctx.num_epochs,
        random_seed=random_seed,
        kl_weight=kl_weight,
        early_stop=ctx.early_stop,
        early_stop_num=ctx.early_stop_num,
        apply_log=prep.apply_log,
        original_data=prep.df.copy(),
    )

    if output_dir is not None:
        result.save(output_dir)

    return result


def pilot_study(
    data: pd.DataFrame | str | Path,
    pilot_size: list[int],
    *,
    name: str | None = None,
    groups: pd.Series | np.ndarray | None = None,
    n_draws: int = 5,
    model: str = "VAE1-10",
    apply_log: bool = True,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    early_stop_patience: int | None = None,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
    verbose: int | str = "minimal",
) -> PilotResult:
    """Sweep over pilot sizes with replicated random draws.

    For each pilot size, *n_draws* random sub-samples are drawn from
    the original data.  A model is trained on each sub-sample and
    synthetic data equal to *n_draws* times the sub-sample size is
    generated.

    This replaces the legacy ``PilotExperiment`` function.

    Parameters
    ----------
    data : DataFrame, str, or Path
        Input data.
    pilot_size : list[int]
        List of pilot sizes to evaluate.
    name : str or None
        Short name for output filenames.
    groups : pd.Series, np.ndarray, or None
        Optional binary group labels. When provided, these labels take
        precedence over bundled dataset groups.
    n_draws : int
        Number of replicated random draws per pilot size (default: 5).
        Must be a positive integer.
    model : str
        Model specification (e.g. ``"VAE1-10"``).
    apply_log : bool
        Apply ``log2(x + 1)`` preprocessing.
    batch_frac : float
        Batch size as a fraction of sample count.
    learning_rate : float
        Optimizer learning rate.
    epoch : int or None
        Fixed epoch count or ``None`` for early stopping.  See
        :func:`generate` for the full interaction table.
    early_stop_patience : int or None
        Stop if loss does not improve for this many epochs.  When ``None``
        and ``epoch`` is also ``None``, defaults to ``30``.  See
        :func:`generate` for the full interaction table.
    off_aug : str or None
        Offline augmentation mode.
    AE_head_num : int
        Fold multiplier for AE-head augmentation.
    Gaussian_head_num : int
        Fold multiplier for Gaussian-head augmentation.
    random_seed : int
        Base random seed for reproducibility.
    output_dir : str, Path, or None
        If set, automatically save results to this directory.
    verbose : int or str
        Verbosity level — see :func:`generate` for details.

    Returns
    -------
    PilotResult
        Wrapper containing one ``SyngResult`` per (pilot_size, draw).
    """
    n_draws = _validate_n_draws(n_draws, param_name="n_draws")

    # --- 0. Resolve verbose level ----------------------------------------
    verbose_level = _resolve_verbose(verbose)

    # --- 1. Prepare data (resolve, validate, convert, label) -------------
    prep = _prepare_data(data=data, name=name, groups=groups, apply_log=apply_log)

    # --- 2. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 3. Pilot loop ---------------------------------------------------
    # new_size = n_draws × pilot (per group if unbalanced)
    repli = n_draws

    runs: dict[tuple[int, int], SyngResult] = {}
    last_ctx: TrainingContext | None = None

    # Calculate total number of runs for progress logging
    total_runs = len(pilot_size) * n_draws
    current_run = 0

    for n_pilot in pilot_size:
        for rand_pilot in range(1, n_draws + 1):
            current_run += 1

            # Log progress before training (if verbosity >= MINIMAL)
            if verbose_level >= VerbosityLevel.MINIMAL:
                print(
                    f"[Pilot size {n_pilot}] Draw {rand_pilot}/{n_draws} "
                    f"(training no. {current_run}/{total_runs})"
                )

            # Draw pilot sub-sample
            rawdata, rawlabels, rawblurlabels, pilot_indices = draw_pilot(
                dataset=prep.oridata,
                labels=prep.orilabels,
                blurlabels=prep.oriblurlabels,
                n_pilot=n_pilot,
                seednum=rand_pilot,
            )

            # new_size for this pilot (group-balanced if needed)
            effective_new_size = _compute_new_size(
                prep.orilabels, prep.n_samples, repli * n_pilot, repli=repli
            )

            # Train (orchestrate: early-stop, blur-label, aug, dispatch)
            trained, ctx = orchestrate_training(
                rawdata=rawdata,
                rawlabels=rawlabels,
                oriblurlabels=rawblurlabels,
                modelname=modelname,
                kl_weight=kl_weight,
                batch_frac=batch_frac,
                random_seed=random_seed,
                epoch=epoch,
                early_stop_patience=early_stop_patience,
                learning_rate=learning_rate,
                off_aug=off_aug,
                AE_head_num=AE_head_num,
                Gaussian_head_num=Gaussian_head_num,
                verbose=verbose_level,
            )
            last_ctx = ctx

            gen_data, recon_data = _infer_from_trained(
                trained,
                new_size=effective_new_size,
                ctx=ctx,
            )

            # -- Assemble SyngResult for this run -------------------------
            pilot_original = prep.df.iloc[pilot_indices.numpy()].copy()

            runs[(n_pilot, rand_pilot)] = _assemble_result(
                gen_data=gen_data,
                recon_data=recon_data,
                trained=trained,
                colnames=prep.colnames,
                modelname=modelname,
                model=model,
                dataname=prep.dataname,
                n_samples=prep.n_samples,
                num_epochs=ctx.num_epochs,
                random_seed=random_seed,
                kl_weight=kl_weight,
                early_stop=ctx.early_stop,
                early_stop_num=ctx.early_stop_num,
                apply_log=prep.apply_log,
                original_data=pilot_original,
                extra_metadata={
                    "pilot_size": n_pilot,
                    "draw": rand_pilot,
                    "pilot_indices": pilot_indices.tolist(),
                },
            )

            # Newline after each run for readability
            if verbose_level >= VerbosityLevel.MINIMAL:
                print()

    # Resolve num_epochs for PilotResult metadata — use last training
    # context if available, otherwise resolve defaults directly.
    if last_ctx is not None:
        resolved_num_epochs = last_ctx.num_epochs
    else:
        resolved_num_epochs, _, _ = _resolve_early_stopping_config(
            epoch=epoch,
            early_stop_patience=early_stop_patience,
        )

    # --- 4. Assemble PilotResult -----------------------------------------
    pilot_result = PilotResult(
        runs=runs,
        original_data=prep.df.copy(),
        metadata={
            "model": model,
            "modelname": modelname,
            "dataname": prep.dataname,
            "pilot_sizes": pilot_size,
            "num_epochs": resolved_num_epochs,
            "seed": random_seed,
        },
    )

    if output_dir is not None:
        pilot_result.save(output_dir)

    return pilot_result


def transfer(
    source_data: pd.DataFrame | str | Path,
    target_data: pd.DataFrame | str | Path,
    *,
    source_name: str | None = None,
    target_name: str | None = None,
    source_groups: pd.Series | np.ndarray | None = None,
    target_groups: pd.Series | np.ndarray | None = None,
    new_size: int | list[int] = 500,
    model: str = "VAE1-10",
    apply_log: bool = True,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    early_stop_patience: int | None = None,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
    verbose: int | str = "minimal",
) -> SyngResult:
    """Train on source data, then fine-tune and generate on target data.

    The model is first trained on *source_data* and its learned state
    is kept in-memory, then fine-tuned on *target_data*.  This is a
    single-run operation returning a :class:`SyngResult`.

    This replaces the legacy ``TransferExperiment`` function.

    Parameters
    ----------
    source_data : DataFrame, str, or Path
        Pre-training dataset.
    target_data : DataFrame, str, or Path
        Fine-tuning / target dataset.
    source_name : str or None
        Short name for the source dataset.
    target_name : str or None
        Short name for the target dataset.
    source_groups : pd.Series, np.ndarray, or None
        Optional binary groups for the source dataset.
    target_groups : pd.Series, np.ndarray, or None
        Optional binary groups for the target dataset.
    new_size : int or list[int]
        Number of samples to generate from the fine-tuned model.
    model : str
        Model specification.
    apply_log : bool
        Apply log2 preprocessing.
    batch_frac : float
        Batch fraction.
    learning_rate : float
        Learning rate.
    epoch : int or None
        Fixed epoch count, or ``None`` for early stopping.  See
        :func:`generate` for the full interaction table.
    early_stop_patience : int or None
        Stop if loss does not improve for this many epochs.  When ``None``
        and ``epoch`` is also ``None``, defaults to ``30``.  See
        :func:`generate` for the full interaction table.
    off_aug : str or None
        Offline augmentation mode.
    AE_head_num : int
        Fold multiplier for AE-head augmentation.
    Gaussian_head_num : int
        Fold multiplier for Gaussian-head augmentation.
    random_seed : int
        Random seed.
    output_dir : str, Path, or None
        If set, save results here.
    verbose : int or str
        Verbosity level — see :func:`generate` for details.

    Returns
    -------
    SyngResult
        Result from the fine-tuned target-phase model.
    """
    # --- 0. Resolve verbose level ----------------------------------------
    verbose_level = _resolve_verbose(verbose)

    # --- 1. Prepare source and target data -------------------------------
    source_prep = _prepare_data(
        data=source_data,
        name=source_name,
        groups=source_groups,
        apply_log=apply_log,
    )
    target_prep = _prepare_data(
        data=target_data,
        name=target_name,
        groups=target_groups,
        apply_log=apply_log,
    )

    # --- 2. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 3. Pre-train on source ------------------------------------------
    source_trained, _source_ctx = orchestrate_training(
        rawdata=source_prep.oridata,
        rawlabels=source_prep.orilabels,
        oriblurlabels=source_prep.oriblurlabels,
        modelname=modelname,
        kl_weight=kl_weight,
        batch_frac=batch_frac,
        random_seed=random_seed,
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        learning_rate=learning_rate,
        off_aug=off_aug,
        AE_head_num=AE_head_num,
        Gaussian_head_num=Gaussian_head_num,
        verbose=verbose_level,
    )
    source_model_state = source_trained.model_state

    # --- 4. Fine-tune on target ------------------------------------------
    effective_new_size = _compute_new_size(
        target_prep.orilabels,
        target_prep.n_samples,
        new_size,
    )

    target_trained, target_ctx = orchestrate_training(
        rawdata=target_prep.oridata,
        rawlabels=target_prep.orilabels,
        oriblurlabels=target_prep.oriblurlabels,
        modelname=modelname,
        kl_weight=kl_weight,
        batch_frac=batch_frac,
        random_seed=random_seed,
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        learning_rate=learning_rate,
        off_aug=off_aug,
        AE_head_num=AE_head_num,
        Gaussian_head_num=Gaussian_head_num,
        model_state=source_model_state,
        verbose=verbose_level,
    )

    # --- 5. Infer --------------------------------------------------------
    gen_data, recon_data = _infer_from_trained(
        target_trained,
        new_size=effective_new_size,
        ctx=target_ctx,
    )

    # --- 6. Assemble SyngResult ------------------------------------------
    result = _assemble_result(
        gen_data=gen_data,
        recon_data=recon_data,
        trained=target_trained,
        colnames=target_prep.colnames,
        modelname=modelname,
        model=model,
        dataname=target_prep.dataname,
        n_samples=target_prep.n_samples,
        num_epochs=target_ctx.num_epochs,
        random_seed=random_seed,
        kl_weight=kl_weight,
        early_stop=target_ctx.early_stop,
        early_stop_num=target_ctx.early_stop_num,
        apply_log=target_prep.apply_log,
        original_data=target_prep.df.copy(),
    )

    if output_dir is not None:
        result.save(output_dir)

    return result
