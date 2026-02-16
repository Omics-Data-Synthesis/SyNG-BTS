"""Experiment functions for SyNG-BTS.

Public API
----------
- ``generate()`` — train a model and produce synthetic samples.
- ``pilot_study()`` — sweep over pilot sizes with replicated draws.
- ``transfer()`` — pre-train on source, fine-tune on target.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_utils import _validate_feature_data, derive_dataname, resolve_data
from .helper_train import _resolve_verbose
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
# Training dispatch
# ---------------------------------------------------------------------------


def _train_model(
    rawdata: torch.Tensor,
    rawlabels: torch.Tensor,
    *,
    modelname: str,
    batch_size: int,
    random_seed: int,
    num_epochs: int,
    learning_rate: float,
    kl_weight: int = 1,
    val_ratio: float = 0.2,
    early_stop: bool = True,
    early_stop_num: int = 30,
    pre_model: str | None = None,
    cap: bool = False,
    loss_fn: str = "MSE",
    use_scheduler: bool = False,
    step_size: int = 10,
    gamma: float = 0.5,
    batch_frac: float = 0.1,
    verbose=1,
) -> TrainedModel:
    """Dispatch to the appropriate training wrapper.

    Returns
    -------
    TrainedModel
        Training-only output (no generation/reconstruction).
    """
    if "GAN" in modelname:
        return training_GANs(
            rawdata=rawdata,
            rawlabels=rawlabels,
            batch_size=batch_size,
            random_seed=random_seed,
            modelname=modelname,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            pre_model=pre_model,
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
            pre_model=pre_model,
            cap=cap,
            loss_fn=loss_fn,
            use_scheduler=use_scheduler,
            step_size=step_size,
            gamma=gamma,
            verbose=verbose,
        )
        # Persist split controls for reconstruction path
        trained.arch_params["_train_random_seed"] = random_seed
        trained.arch_params["_train_val_ratio"] = val_ratio
        return trained
    elif modelname in ("maf", "realnvp", "glow", "maf-split", "maf-split-glow"):
        return training_flows(
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
            pre_model=pre_model,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported model: {modelname!r}")


def _infer_from_trained(
    trained: TrainedModel,
    *,
    new_size: int | list[int],
    rawdata: torch.Tensor,
    rawlabels: torch.Tensor,
    batch_size: int,
    cap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run generation and reconstruction via the unified inference dispatcher.

    This delegates to :func:`inference.run_generation` and
    :func:`inference.run_reconstruction`, keeping post-training
    inference cleanly separated from training orchestrators.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        ``(generated_data, reconstructed_data)``
    """
    from torch.utils.data import DataLoader, TensorDataset, random_split

    family = trained.arch_params["family"]

    # --- Determine capping values ---
    if cap:
        col_max, _ = torch.max(rawdata, dim=0)
        col_sd = torch.std(rawdata, dim=0, unbiased=True)
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
        data = TensorDataset(rawdata, rawlabels)

        # Reproduce legacy reconstruction path as closely as possible:
        # training_AEs reconstructs over the train split DataLoader
        # (shuffle=True, drop_last=True).
        split_seed = trained.arch_params.get("_train_random_seed")
        split_val_ratio = trained.arch_params.get("_train_val_ratio")

        if (split_seed is not None) and (split_val_ratio is not None):
            set_all_seeds(int(split_seed))
            val_size = int(len(data) * float(split_val_ratio))
            train_size = len(data) - val_size
            train_dataset, _val_dataset = random_split(data, [train_size, val_size])
            recon_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
        else:
            recon_loader = DataLoader(
                data,
                batch_size=batch_size,
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
    pre_model: str | None = None,
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
    pre_model : str or None
        Path to a pre-trained model for transfer learning.
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

    # --- 1. Resolve data -------------------------------------------------
    df, bundled_groups = resolve_data(data)
    _validate_feature_data(df)
    dataname = derive_dataname(data, name)

    # --- 2. Extract numeric data, capture column names -------------------
    data_pd = df
    colnames = list(data_pd.columns)
    oridata = torch.from_numpy(data_pd.to_numpy().copy()).to(torch.float32)

    if apply_log:
        oridata = preprocessinglog2(oridata)

    n_samples = oridata.shape[0]
    effective_groups = _resolve_effective_groups(
        groups,
        bundled_groups,
        n_samples=n_samples,
        param_name="groups",
    )

    # --- 3. Labels -------------------------------------------------------
    orilabels, oriblurlabels = create_labels(
        n_samples=n_samples,
        groups=effective_groups,
    )

    # --- 4. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 5. Epoch / early-stopping logic ---------------------------------
    num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        default_max_epochs=1000,
        default_patience=30,
    )

    # --- 6. Prepare raw data & labels ------------------------------------
    rawdata = oridata
    rawlabels = orilabels

    # For training two groups without CVAE, append blurred labels
    if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
        rawdata = torch.cat((rawdata, oriblurlabels), dim=1)

    # --- 7. Compute new_size (group-balanced if needed) ------------------
    effective_new_size = _compute_new_size(orilabels, n_samples, new_size)

    # --- 8. Offline augmentation -----------------------------------------
    if off_aug == "Gaussian_head":
        rawdata, rawlabels = Gaussian_aug(
            rawdata, rawlabels, multiplier=[Gaussian_head_num]
        )
    elif off_aug == "AE_head":
        # TODO Change hardcoded training config for AE head augmentation to be more flexible
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
            verbose=verbose_level,
        )
        rawdata = feed_data
        rawlabels = feed_labels

    # --- 9. Train --------------------------------------------------------
    batch_size = max(1, round(rawdata.shape[0] * batch_frac))

    trained = _train_model(
        rawdata=rawdata,
        rawlabels=rawlabels,
        modelname=modelname,
        batch_size=batch_size,
        random_seed=random_seed,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        val_ratio=val_ratio,
        early_stop=early_stop,
        early_stop_num=early_stop_num,
        pre_model=pre_model,
        cap=cap,
        loss_fn="MSE",
        use_scheduler=use_scheduler,
        step_size=step_size,
        gamma=gamma,
        batch_frac=batch_frac,
        verbose=verbose_level,
    )
    gen_data, recon_data = _infer_from_trained(
        trained,
        new_size=effective_new_size,
        rawdata=rawdata,
        rawlabels=rawlabels,
        batch_size=batch_size,
        cap=cap,
    )

    # --- 10. Assemble SyngResult -----------------------------------------
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

    # --- 10b. Inverse log transform to count scale -----------------------
    if apply_log:
        gen_df = inverse_log2(gen_df)
        if recon_df is not None:
            recon_df = inverse_log2(recon_df)

    # --- 10c. Validate column order consistency ---------------------------
    # Defensive check: ensure all DataFrames have the same column order
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

    metadata = {
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

    result = SyngResult(
        generated_data=gen_df,
        loss=loss_df,
        reconstructed_data=recon_df,
        original_data=data_pd.copy(),
        model_state=trained.model_state,
        metadata=metadata,
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
    model: str = "VAE1-10",
    apply_log: bool = True,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    early_stop_patience: int | None = None,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    pre_model: str | None = None,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
    verbose: int | str = "minimal",
) -> PilotResult:
    """Sweep over pilot sizes with replicated random draws.

    For each pilot size, five random sub-samples are drawn from the
    original data.  A model is trained on each sub-sample and synthetic
    data equal to five times the sub-sample size is generated.

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
    pre_model : str or None
        Path to a pre-trained model for transfer learning.
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
    # --- 0. Resolve verbose level ----------------------------------------
    verbose_level = _resolve_verbose(verbose)

    # --- 1. Resolve data -------------------------------------------------
    df, bundled_groups = resolve_data(data)
    _validate_feature_data(df)
    dataname = derive_dataname(data, name)

    # --- 2. Extract numeric data, capture column names -------------------
    data_pd = df
    colnames = list(data_pd.columns)
    oridata = torch.from_numpy(data_pd.to_numpy().copy()).to(torch.float32)
    if apply_log:
        oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]

    effective_groups = _resolve_effective_groups(
        groups,
        bundled_groups,
        n_samples=n_samples,
        param_name="groups",
    )

    # --- 3. Labels -------------------------------------------------------
    orilabels, oriblurlabels = create_labels(
        n_samples=n_samples,
        groups=effective_groups,
    )

    # --- 4. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 5. Epoch / early-stopping logic ---------------------------------
    num_epochs, early_stop, early_stop_num = _resolve_early_stopping_config(
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        default_max_epochs=1000,
        default_patience=30,
    )

    # --- 6. Pilot loop ---------------------------------------------------
    # new_size = 5× pilot (per group if unbalanced)
    repli = 5

    runs: dict[tuple[int, int], SyngResult] = {}

    for n_pilot in pilot_size:
        for rand_pilot in range(1, 6):
            # Draw pilot sub-sample
            rawdata, rawlabels, rawblurlabels, pilot_indices = draw_pilot(
                dataset=oridata,
                labels=orilabels,
                blurlabels=oriblurlabels,
                n_pilot=n_pilot,
                seednum=rand_pilot,
            )

            # For two groups without CVAE, append blurred labels
            if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
                rawdata = torch.cat((rawdata, rawblurlabels), dim=1)

            # new_size for this pilot (group-balanced if needed)
            if (len(torch.unique(orilabels)) > 1) and (
                int(sum(orilabels == 0)) != int(sum(orilabels == 1))
            ):
                effective_new_size: int | list[int] = [
                    int(sum(orilabels == 0)),
                    int(sum(orilabels == 1)),
                    repli,
                ]
            else:
                effective_new_size = repli * n_pilot

            # Gaussian augmentation
            if off_aug == "Gaussian_head":
                rawdata, rawlabels = Gaussian_aug(
                    rawdata, rawlabels, multiplier=[Gaussian_head_num]
                )

            # AE head augmentation
            if off_aug == "AE_head":
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
                    verbose=verbose_level,
                )
                rawdata = feed_data
                rawlabels = feed_labels

            batch_size = max(1, round(rawdata.shape[0] * batch_frac))

            # Train and infer
            trained = _train_model(
                rawdata=rawdata,
                rawlabels=rawlabels,
                modelname=modelname,
                batch_size=batch_size,
                random_seed=random_seed,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                kl_weight=kl_weight,
                early_stop=early_stop,
                early_stop_num=early_stop_num,
                pre_model=pre_model,
                batch_frac=batch_frac,
                verbose=verbose_level,
            )
            gen_data, recon_data = _infer_from_trained(
                trained,
                new_size=effective_new_size,
                rawdata=rawdata,
                rawlabels=rawlabels,
                batch_size=batch_size,
            )

            # -- Assemble SyngResult for this run -------------------------
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

            # -- Inverse log transform to count scale ---------------------
            if apply_log:
                gen_df = inverse_log2(gen_df)
                if recon_df is not None:
                    recon_df = inverse_log2(recon_df)

            # -- Validate column order consistency -------------------------
            # Defensive check: ensure all DataFrames have the same column order
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

            # Per-draw original data subset
            pilot_original = data_pd.iloc[pilot_indices.numpy()].copy()

            loss_df = _build_loss_df(trained.log_dict, modelname)

            run_metadata = {
                "model": model,
                "modelname": modelname,
                "dataname": dataname,
                "num_epochs": num_epochs,
                "epochs_trained": trained.epochs_trained,
                "seed": random_seed,
                "kl_weight": kl_weight,
                "pilot_size": n_pilot,
                "draw": rand_pilot,
                "pilot_indices": pilot_indices.tolist(),
                "input_shape": (n_samples, len(colnames)),
                "early_stop": early_stop,
                "early_stop_patience": early_stop_num,
                "generated_labels": gen_labels,
                "reconstructed_labels": recon_labels,
                "apply_log": apply_log,
                "arch_params": {
                    k: v
                    for k, v in trained.arch_params.items()
                    if not k.startswith("_")
                },
            }

            runs[(n_pilot, rand_pilot)] = SyngResult(
                generated_data=gen_df,
                loss=loss_df,
                reconstructed_data=recon_df,
                original_data=pilot_original,
                model_state=trained.model_state,
                metadata=run_metadata,
            )

    # --- 7. Assemble PilotResult -----------------------------------------
    pilot_result = PilotResult(
        runs=runs,
        original_data=data_pd.copy(),
        metadata={
            "model": model,
            "modelname": modelname,
            "dataname": dataname,
            "pilot_sizes": pilot_size,
            "num_epochs": num_epochs,
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
    pilot_size: list[int] | None = None,
    source_size: int = 500,
    new_size: int = 500,
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
) -> SyngResult | PilotResult:
    """Train on source data, then fine-tune and generate on target data.

    The model is first trained on *source_data* and its state is saved.
    Then the saved state is loaded as a pre-trained model and fine-tuned
    on *target_data*.

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
    pilot_size : list[int] or None
        If set, uses ``pilot_study`` for the target phase.  Otherwise
        uses ``generate``.
    source_size : int
        Number of samples to generate during pre-training.
    new_size : int
        Number of samples to generate in ``generate`` mode.
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
    SyngResult or PilotResult
        ``SyngResult`` when ``pilot_size`` is ``None``, otherwise
        ``PilotResult``.
    """
    import tempfile

    fromname = derive_dataname(source_data, source_name)
    toname = derive_dataname(target_data, target_name)

    # We need a temp file to pass the model state between the two phases.
    # If output_dir is set, use a Transfer subdir; otherwise use a temp dir.
    if output_dir is not None:
        transfer_dir = Path(output_dir) / "Transfer"
        transfer_dir.mkdir(parents=True, exist_ok=True)
        save_model_path = str(transfer_dir / f"{toname}_from{fromname}_{model}.pt")
        _cleanup_transfer = False
    else:
        _transfer_tmpdir = tempfile.mkdtemp()
        save_model_path = str(
            Path(_transfer_tmpdir) / f"{toname}_from{fromname}_{model}.pt"
        )
        _cleanup_transfer = True

    # --- 1. pre-train on source ------------------------------------------
    _source_result = generate(
        data=source_data,
        name=fromname,
        groups=source_groups,
        new_size=[source_size],
        model=model,
        apply_log=apply_log,
        batch_frac=batch_frac,
        learning_rate=learning_rate,
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        off_aug=off_aug,
        AE_head_num=AE_head_num,
        Gaussian_head_num=Gaussian_head_num,
        pre_model=None,
        random_seed=random_seed,
        output_dir=(str(Path(output_dir) / "Transfer") if output_dir else None),
        verbose=verbose,
    )

    # Persist source model state to disk for the target phase to load
    # via pre_model (will be replaced with in-memory handoff in a future
    # refactor phase).
    torch.save(_source_result.model_state, save_model_path)

    # --- 2. fine-tune on target ------------------------------------------
    if pilot_size is not None:
        result = pilot_study(
            data=target_data,
            pilot_size=pilot_size,
            name=toname,
            groups=target_groups,
            model=model,
            apply_log=apply_log,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_patience=early_stop_patience,
            off_aug=off_aug,
            AE_head_num=AE_head_num,
            Gaussian_head_num=Gaussian_head_num,
            pre_model=save_model_path,
            random_seed=random_seed,
            output_dir=output_dir,
            verbose=verbose,
        )
    else:
        result = generate(
            data=target_data,
            name=toname,
            groups=target_groups,
            new_size=new_size,
            model=model,
            apply_log=apply_log,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_patience=early_stop_patience,
            off_aug=off_aug,
            AE_head_num=AE_head_num,
            Gaussian_head_num=Gaussian_head_num,
            pre_model=save_model_path,
            random_seed=random_seed,
            output_dir=output_dir,
            verbose=verbose,
        )

    # Cleanup temp dir if we created one
    if _cleanup_transfer:
        import shutil

        shutil.rmtree(_transfer_tmpdir, ignore_errors=True)

    return result
