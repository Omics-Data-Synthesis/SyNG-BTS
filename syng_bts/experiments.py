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

from .data_utils import derive_dataname, resolve_data
from .helper_training import (
    training_AEs,
    training_flows,
    training_GANs,
    training_iter,
)
from .helper_utils import (
    Gaussian_aug,
    create_labels,
    draw_pilot,
    preprocessinglog2,
)
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
        Raw loss series as returned by ``TrainOutput.log_dict``.
    modelname : str
        The short model name (``"AE"``, ``"VAE"``, ``"CVAE"``, ``"GAN"``,
        ``"WGAN"``, ``"WGANGP"``, ``"maf"``, ``"glow"``, ``"realnvp"``, etc.).

    Returns
    -------
    pd.DataFrame
    """
    if "AE" in modelname:
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


# =========================================================================
# New public API
# =========================================================================


def generate(
    data: pd.DataFrame | str | Path,
    *,
    name: str | None = None,
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
    save_model: str | None = None,
    use_scheduler: bool = False,
    step_size: int = 10,
    gamma: float = 0.5,
    cap: bool = False,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
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
        Optimiser learning rate.
    epoch : int or None
        Fixed epoch count, or ``None`` for early stopping.
    val_ratio : float
        Validation split ratio (AE family only).
    early_stop_patience : int or None
        Stop if loss does not improve for this many epochs.  ``None``
        disables early stopping (requires *epoch* to be set).
    off_aug : str or None
        Offline augmentation: ``"AE_head"``, ``"Gaussian_head"``, or
        ``None``.
    AE_head_num : int
        Fold multiplier for AE-head augmentation.
    Gaussian_head_num : int
        Fold multiplier for Gaussian-head augmentation.
    pre_model : str or None
        Path to a pre-trained model for transfer learning.
    save_model : str or None
        Path to save the trained model state.
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

    Returns
    -------
    SyngResult
        Rich result object containing generated data, loss log,
        reconstructed data (AE/VAE/CVAE), model state, and metadata.
    """
    # --- 1. Resolve data -------------------------------------------------
    df = resolve_data(data)
    dataname = derive_dataname(data, name)

    # --- 2. Extract numeric data, capture column names -------------------
    data_pd = df.select_dtypes(include=np.number)
    if "groups" in data_pd.columns:
        data_pd = data_pd.drop(columns=["groups"])
    colnames = list(data_pd.columns)
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)

    if apply_log:
        oridata = preprocessinglog2(oridata)

    n_samples = oridata.shape[0]

    # --- 3. Labels -------------------------------------------------------
    groups = df["groups"] if "groups" in df.columns else None
    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    # --- 4. Parse model spec ---------------------------------------------
    modelname, kl_weight = _parse_model_spec(model)

    # --- 5. Epoch / early-stopping logic ---------------------------------
    if epoch is not None:
        num_epochs = epoch
        early_stop = False
    elif early_stop_patience is not None:
        num_epochs = 1000
        early_stop = True
    else:
        # Default: early stopping with patience 30
        early_stop_patience = 30
        num_epochs = 1000
        early_stop = True

    early_stop_num = early_stop_patience if early_stop_patience is not None else 30

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
        )
        rawdata = feed_data
        rawlabels = feed_labels

    # --- 9. Train --------------------------------------------------------
    batch_size = max(1, round(rawdata.shape[0] * batch_frac))

    if "GAN" in modelname:
        train_out = training_GANs(
            rawdata=rawdata,
            rawlabels=rawlabels,
            batch_size=batch_size,
            random_seed=random_seed,
            modelname=modelname,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            new_size=effective_new_size,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            pre_model=pre_model,
            save_model=save_model,
        )
    elif "AE" in modelname:
        train_out = training_AEs(
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
            save_model=save_model,
            cap=cap,
            loss_fn="MSE",
            new_size=effective_new_size,
            use_scheduler=use_scheduler,
            step_size=step_size,
            gamma=gamma,
        )
    elif modelname in ("maf", "realnvp", "glow", "maf-split", "maf-split-glow"):
        train_out = training_flows(
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            new_size=effective_new_size,
            num_hidden=226,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            pre_model=pre_model,
            save_model=save_model,
        )
    else:
        raise ValueError(f"Unsupported model: {model!r}")

    # --- 10. Assemble SyngResult -----------------------------------------
    gen_np = train_out.generated_data.detach().numpy()

    # For CVAE, add label column name
    gen_colnames = list(colnames)
    if modelname == "CVAE" and gen_np.shape[1] > len(colnames):
        gen_colnames = gen_colnames + ["label"]

    gen_df = pd.DataFrame(gen_np, columns=gen_colnames[: gen_np.shape[1]])

    recon_df = None
    if train_out.reconstructed_data is not None:
        recon_np = train_out.reconstructed_data.detach().numpy()

        # For CVAE, add label column name
        recon_colnames = list(colnames)
        if modelname == "CVAE" and recon_np.shape[1] > len(colnames):
            recon_colnames = recon_colnames + ["label"]

        recon_df = pd.DataFrame(recon_np, columns=recon_colnames[: recon_np.shape[1]])

    loss_df = _build_loss_df(train_out.log_dict, modelname)

    metadata = {
        "model": model,
        "modelname": modelname,
        "dataname": dataname,
        "num_epochs": num_epochs,
        "epochs_trained": num_epochs,
        "seed": random_seed,
        "kl_weight": kl_weight,
        "input_shape": (n_samples, len(colnames)),
    }

    result = SyngResult(
        generated_data=gen_df,
        loss=loss_df,
        reconstructed_data=recon_df,
        model_state=train_out.model_state,
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
    model: str = "VAE1-10",
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    early_stop_patience: int = 30,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    pre_model: str | None = None,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
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
    model : str
        Model specification (e.g. ``"VAE1-10"``).
    batch_frac : float
        Batch size as a fraction of sample count.
    learning_rate : float
        Optimiser learning rate.
    epoch : int or None
        Fixed epoch count, or ``None`` for early stopping.
    early_stop_patience : int
        Early-stopping patience (ignored when *epoch* is set).
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

    Returns
    -------
    PilotResult
        Wrapper containing one ``SyngResult`` per (pilot_size, draw).
    """
    # --- 1. Resolve data -------------------------------------------------
    df = resolve_data(data)
    dataname = derive_dataname(data, name)

    data_pd = df.select_dtypes(include=np.number)
    if "groups" in data_pd.columns:
        data_pd = data_pd.drop(columns=["groups"])
    colnames = list(data_pd.columns)
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
    oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]

    groups = df["groups"] if "groups" in df.columns else None
    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    modelname, kl_weight = _parse_model_spec(model)

    # Epoch / early-stopping
    if epoch is not None:
        num_epochs = epoch
        early_stop = False
    else:
        num_epochs = 1000
        early_stop = True

    # new_size = 5× pilot (per group if unbalanced)
    repli = 5

    runs: dict[tuple[int, int], SyngResult] = {}

    for n_pilot in pilot_size:
        for rand_pilot in range(1, 6):
            # Draw pilot sub-sample
            rawdata, rawlabels, rawblurlabels = draw_pilot(
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
                )
                rawdata = feed_data
                rawlabels = feed_labels

            batch_size = max(1, round(rawdata.shape[0] * batch_frac))

            # Train
            if "GAN" in modelname:
                train_out = training_GANs(
                    rawdata=rawdata,
                    rawlabels=rawlabels,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    new_size=effective_new_size,
                    early_stop=early_stop,
                    early_stop_num=early_stop_patience,
                    pre_model=pre_model,
                    save_model=None,
                )
            elif "AE" in modelname:
                train_out = training_AEs(
                    rawdata=rawdata,
                    rawlabels=rawlabels,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    kl_weight=kl_weight,
                    early_stop=early_stop,
                    early_stop_num=early_stop_patience,
                    pre_model=pre_model,
                    save_model=None,
                    loss_fn="MSE",
                    new_size=effective_new_size,
                )
            elif modelname in (
                "maf",
                "realnvp",
                "glow",
                "maf-split",
                "maf-split-glow",
            ):
                train_out = training_flows(
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    new_size=effective_new_size,
                    num_hidden=226,
                    early_stop=early_stop,
                    early_stop_num=early_stop_patience,
                    pre_model=pre_model,
                    save_model=None,
                )
            else:
                raise ValueError(f"Unsupported model: {model!r}")

            # Assemble SyngResult for this run
            gen_np = train_out.generated_data.detach().numpy()
            gen_df = pd.DataFrame(gen_np, columns=colnames[: gen_np.shape[1]])

            recon_df = None
            if train_out.reconstructed_data is not None:
                recon_np = train_out.reconstructed_data.detach().numpy()
                recon_df = pd.DataFrame(recon_np, columns=colnames[: recon_np.shape[1]])

            loss_df = _build_loss_df(train_out.log_dict, modelname)

            run_metadata = {
                "model": model,
                "modelname": modelname,
                "dataname": dataname,
                "num_epochs": num_epochs,
                "seed": random_seed,
                "kl_weight": kl_weight,
                "pilot_size": n_pilot,
                "draw": rand_pilot,
                "input_shape": (n_samples, len(colnames)),
            }

            runs[(n_pilot, rand_pilot)] = SyngResult(
                generated_data=gen_df,
                loss=loss_df,
                reconstructed_data=recon_df,
                model_state=train_out.model_state,
                metadata=run_metadata,
            )

    pilot_result = PilotResult(
        runs=runs,
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
    pilot_size: list[int] | None = None,
    source_size: int = 500,
    new_size: int = 500,
    model: str = "VAE1-10",
    apply_log: bool = True,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: int | None = None,
    early_stop_patience: int = 30,
    off_aug: str | None = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    random_seed: int = 123,
    output_dir: str | Path | None = None,
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
        Fixed epoch count or ``None`` for early stopping.
    early_stop_patience : int
        Early-stopping patience.
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

    # --- Phase 1: pre-train on source ------------------------------------
    _source_result = generate(
        data=source_data,
        name=fromname,
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
        save_model=save_model_path,
        random_seed=random_seed,
        output_dir=(str(Path(output_dir) / "Transfer") if output_dir else None),
    )

    # --- Phase 2: fine-tune on target ------------------------------------
    if pilot_size is not None:
        result = pilot_study(
            data=target_data,
            pilot_size=pilot_size,
            name=toname,
            model=model,
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
        )
    else:
        result = generate(
            data=target_data,
            name=toname,
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
        )

    # Cleanup temp dir if we created one
    if _cleanup_transfer:
        import shutil

        shutil.rmtree(_transfer_tmpdir, ignore_errors=True)

    return result
