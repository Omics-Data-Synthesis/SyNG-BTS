# %%

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, random_split

from . import helper_train as ht
from .helper_models import AE, CVAE, GAN, VAE
from .helper_train import VerbosityLevel
from .helper_utils import (
    reconstruct_samples,
    set_all_seeds,
)

# ---------------------------------------------------------------------------
# Training-only return type (no generation/reconstruction)
# ---------------------------------------------------------------------------


@dataclass
class TrainedModel:
    """Training-only output from training orchestrators.

    This is the clean training→inference boundary contract.  It carries
    everything needed to reconstruct and use the model without coupling
    training to generation/reconstruction.

    Attributes
    ----------
    model : nn.Module
        The trained model in ``eval()`` mode.
    model_state : dict[str, Any]
        The ``state_dict()`` snapshot of the trained model.
    arch_params : dict[str, Any]
        Architecture parameters sufficient to rebuild the model from
        scratch.  Schema varies by family — see ``_build_arch_params_*``.
    log_dict : dict[str, list]
        Raw loss series (keys depend on model family).
    epochs_trained : int
        Actual number of epochs executed (may be < configured maximum
        when early stopping triggers).
    """

    model: nn.Module
    model_state: dict[str, Any]
    arch_params: dict[str, Any]
    log_dict: dict[str, list]
    epochs_trained: int


# ---------------------------------------------------------------------------
# arch_params builders
# ---------------------------------------------------------------------------


def _build_arch_params_ae(
    modelname: str,
    num_features: int,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Build ``arch_params`` for AE / VAE / CVAE.

    Parameters
    ----------
    modelname : str
        ``"AE"``, ``"VAE"``, or ``"CVAE"``.
    num_features : int
        Number of input features.
    num_classes : int or None
        Number of classes (required for CVAE, ignored otherwise).

    Returns
    -------
    dict[str, Any]
    """
    latent_size = 64 if modelname == "AE" else 32
    params: dict[str, Any] = {
        "family": "ae",
        "modelname": modelname,
        "num_features": num_features,
        "latent_size": latent_size,
    }
    if modelname == "CVAE":
        if num_classes is None:
            raise ValueError("num_classes is required for CVAE arch_params")
        params["num_classes"] = num_classes
    return params


def _build_arch_params_gan(
    modelname: str,
    num_features: int,
    latent_dim: int = 32,
) -> dict[str, Any]:
    """Build ``arch_params`` for GAN / WGAN / WGANGP.

    Parameters
    ----------
    modelname : str
        ``"GAN"``, ``"WGAN"``, or ``"WGANGP"``.
    num_features : int
        Number of input features.
    latent_dim : int
        Latent dimension size.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "family": "gan",
        "modelname": modelname,
        "num_features": num_features,
        "latent_dim": latent_dim,
    }


def _build_arch_params_flow(
    modelname: str,
    num_inputs: int,
    num_blocks: int,
    num_hidden: int,
) -> dict[str, Any]:
    """Build ``arch_params`` for normalizing flow models.

    Parameters
    ----------
    modelname : str
        ``"maf"``, ``"realnvp"``, ``"glow"``, ``"maf-split"``, or
        ``"maf-split-glow"``.
    num_inputs : int
        Number of input features.
    num_blocks : int
        Number of flow blocks.
    num_hidden : int
        Hidden dimension size per block.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "family": "flow",
        "modelname": modelname,
        "num_inputs": num_inputs,
        "num_blocks": num_blocks,
        "num_hidden": num_hidden,
    }


def training_iter(
    iter_times,  # how many times to iterate
    rawdata,  # pilot data
    rawlabels,  # pilot labels
    random_seed,
    modelname,  # choose from AE, VAE
    num_epochs,  # maximum number of training epochs if early stopping is not triggered
    batch_size,  # batch size
    learning_rate,  # learning rate
    early_stop=False,  # whether to use early stopping rule
    early_stop_num=30,  # training stops if loss does not improve for early_stop_num epochs
    kl_weight=1,  # only take effect for training VAE
    loss_fn="MSE",  # choose WMSE only if you know the weight, MSE by default
    replace=False,  # whether to replace the failure features in each reconstruction
    verbose=VerbosityLevel.MINIMAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively reconstruct data to augment a small pilot sample.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(feed_data, feed_labels)`` — the augmented data and labels.
    """
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata, rawlabels)

    if modelname == "AE":
        model = AE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        feed_data = rawdata
        feed_set = data
        for _i in range(iter_times):
            feed_loader = DataLoader(
                feed_set, batch_size=batch_size, shuffle=True, drop_last=True
            )
            # Use train_loader as val_loader (training_iter is an
            # augmentation utility, not concerned with proper validation).
            log_dict, best_model = ht.train_AE(
                num_epochs=num_epochs,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=feed_loader,
                val_loader=feed_loader,
                early_stop=early_stop,
                early_stop_num=early_stop_num,
                skip_epoch_stats=True,
                logging_interval=50,
                verbose=verbose,
            )
            final_model = best_model if early_stop else model
            feed_data_gen, feed_labels = reconstruct_samples(
                model=final_model,
                modelname="AE",
                data_loader=feed_loader,
                n_features=num_features,
            )
            # add labels to the generated data
            if feed_labels.dim() == 1:
                feed_labels = feed_labels.unsqueeze(1).float()
            feed_labels = torch.cat(
                (feed_labels, feed_labels), dim=0
            )  # repeat the labels for the generated data
            if verbose == VerbosityLevel.DETAILED:
                print(f"Iter data shape: {feed_data_gen.shape}")
            if replace:
                new_sample_range = range(
                    int(feed_data_gen.shape[0] / 2), feed_data_gen.shape[0]
                )
                num_failures = 0
                half_n = len(new_sample_range)  # half of the new samples
                for i_feature in range(feed_data_gen.shape[1]):
                    if (torch.std(feed_data_gen[new_sample_range, i_feature]) == 0) & (
                        torch.mean(feed_data_gen[new_sample_range, i_feature]) == 0
                    ):
                        # only replace the second half with original data
                        feed_data_gen[new_sample_range, i_feature] = feed_data[
                            :half_n, i_feature
                        ]
                        num_failures += 1
                if verbose == VerbosityLevel.DETAILED:
                    print(f"Replaced {num_failures} zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)

    elif modelname == "VAE":
        model = VAE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        feed_data = rawdata
        feed_set = data
        for _i in range(iter_times):
            batch_size = round(feed_data.shape[0] * 0.1)
            feed_loader = DataLoader(
                feed_set, batch_size=batch_size, shuffle=True, drop_last=True
            )
            log_dict, best_model = ht.train_VAE(
                num_epochs=num_epochs,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=feed_loader,
                val_loader=feed_loader,
                early_stop=early_stop,
                early_stop_num=early_stop_num,
                skip_epoch_stats=True,
                reconstruction_term_weight=1,
                kl_weight=kl_weight,
                logging_interval=50,
                verbose=verbose,
            )
            final_model = best_model if early_stop else model
            feed_data_gen, feed_labels = reconstruct_samples(
                model=final_model,
                modelname="VAE",
                data_loader=feed_loader,
                n_features=num_features,
            )
            # add labels to the generated data
            if feed_labels.dim() == 1:
                feed_labels = feed_labels.unsqueeze(1).float()
            feed_labels = torch.cat(
                (feed_labels, feed_labels), dim=0
            )  # repeat the labels for the generated data
            if verbose == VerbosityLevel.DETAILED:
                print(f"Iter data shape: {feed_data_gen.shape}")
            if replace:
                new_sample_range = range(
                    int(feed_data_gen.shape[0] / 2), feed_data_gen.shape[0]
                )
                num_failures = 0
                half_n = len(new_sample_range)  # half of the new samples
                for i_feature in range(feed_data_gen.shape[1]):
                    if (torch.std(feed_data_gen[new_sample_range, i_feature]) == 0) & (
                        torch.mean(feed_data_gen[new_sample_range, i_feature]) == 0
                    ):
                        # only replace the second half with original data
                        feed_data_gen[new_sample_range, i_feature] = feed_data[
                            :half_n, i_feature
                        ]
                        num_failures += 1
                if verbose == VerbosityLevel.DETAILED:
                    print(f"Replaced {num_failures} zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)
    else:
        raise ValueError(f"modelname '{modelname}' not supported by training_iter.")

    return feed_data, feed_labels


# =========================================================================
# Training wrappers — training only, no generation/reconstruction
# =========================================================================


def training_AEs(
    rawdata,
    rawlabels,
    batch_size,
    random_seed,
    modelname,
    num_epochs,
    learning_rate,
    val_ratio=0.2,
    pre_model=None,
    kl_weight=1,
    early_stop=True,
    early_stop_num=30,
    cap=False,
    loss_fn="MSE",
    use_scheduler=False,
    step_size=10,
    gamma=0.5,
    verbose=VerbosityLevel.MINIMAL,
) -> TrainedModel:
    """Train an AE, VAE, or CVAE and return training artifacts only.

    Does **not** perform generation or reconstruction — those are
    deferred to the inference dispatcher.

    Returns
    -------
    TrainedModel
        Training-only output with model, state_dict, arch_params,
        log_dict, and epochs_trained.
    """
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    labels_squeezed = rawlabels.squeeze(1).long()
    num_classes = len(torch.unique(labels_squeezed))
    data = TensorDataset(rawdata, rawlabels)

    if modelname == "CVAE":
        model = CVAE(num_features, num_classes)
    elif modelname == "VAE":
        model = VAE(num_features)
    elif modelname == "AE":
        model = AE(num_features)
    else:
        raise ValueError("modelname is not supported by training_AEs function.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False, drop_last=False
    )

    # transfer learning
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))

    # Train the model
    if modelname == "CVAE":
        log_dict, best_model = ht.train_CVAE(
            num_epochs=num_epochs,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            skip_epoch_stats=True,
            reconstruction_term_weight=1,
            kl_weight=kl_weight,
            logging_interval=50,
            scheduler=scheduler,
            verbose=verbose,
        )
    elif modelname == "VAE":
        log_dict, best_model = ht.train_VAE(
            num_epochs=num_epochs,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            skip_epoch_stats=True,
            reconstruction_term_weight=1,
            kl_weight=kl_weight,
            logging_interval=50,
            scheduler=scheduler,
            verbose=verbose,
        )
    else:
        log_dict, best_model = ht.train_AE(
            num_epochs=num_epochs,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            skip_epoch_stats=True,
            logging_interval=50,
            verbose=verbose,
        )

    final_model = best_model if early_stop else model
    final_model.eval()

    if modelname == "AE":
        epoch_log_key = "train_loss_per_batch"
    else:
        epoch_log_key = "train_combined_loss_per_batch"
    batches_per_epoch = len(train_loader)
    if batches_per_epoch > 0:
        epochs_trained = len(log_dict.get(epoch_log_key, [])) // batches_per_epoch
    else:
        epochs_trained = 0

    arch_params = _build_arch_params_ae(
        modelname=modelname,
        num_features=num_features,
        num_classes=num_classes if modelname == "CVAE" else None,
    )

    return TrainedModel(
        model=final_model,
        model_state=final_model.state_dict(),
        arch_params=arch_params,
        log_dict=log_dict,
        epochs_trained=epochs_trained,
    )


def training_GANs(
    rawdata,
    rawlabels,
    batch_size,
    random_seed,
    modelname,
    num_epochs,
    learning_rate,
    pre_model=None,
    early_stop=True,
    early_stop_num=30,
    verbose=VerbosityLevel.MINIMAL,
) -> TrainedModel:
    """Train a GAN/WGAN/WGANGP and return training artifacts only.

    Does **not** perform sample generation.

    Returns
    -------
    TrainedModel
        Training-only output with model, state_dict, arch_params,
        log_dict, and epochs_trained.
    """
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata, rawlabels)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    latent_dim = 32

    model = GAN(num_features=num_features, latent_dim=latent_dim)

    optim_gen = torch.optim.Adam(
        model.generator.parameters(), betas=(0.5, 0.999), lr=learning_rate
    )
    optim_discr = torch.optim.Adam(
        model.discriminator.parameters(), betas=(0.5, 0.999), lr=learning_rate
    )
    # transfer learning
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))

    if modelname == "GAN":
        log_dict, best_model = ht.train_GAN(
            num_epochs=num_epochs,
            model=model,
            optimizer_gen=optim_gen,
            optimizer_discr=optim_discr,
            latent_dim=latent_dim,
            train_loader=train_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            logging_interval=100,
            verbose=verbose,
        )
    elif modelname == "WGAN":
        log_dict, best_model = ht.train_WGAN(
            num_epochs=num_epochs,
            model=model,
            optimizer_gen=optim_gen,
            optimizer_discr=optim_discr,
            latent_dim=latent_dim,
            train_loader=train_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            logging_interval=100,
            verbose=verbose,
        )
    elif modelname == "WGANGP":
        log_dict, best_model = ht.train_WGANGP(
            num_epochs=num_epochs,
            model=model,
            optimizer_gen=optim_gen,
            optimizer_discr=optim_discr,
            latent_dim=latent_dim,
            train_loader=train_loader,
            early_stop=early_stop,
            early_stop_num=early_stop_num,
            discr_iter_per_generator_iter=5,
            logging_interval=100,
            gradient_penalty=True,
            gradient_penalty_weight=10,
            verbose=verbose,
        )
    else:
        raise ValueError(f"modelname '{modelname}' not supported by training_GANs.")

    final_model = best_model if early_stop else model
    final_model.eval()
    batches_per_epoch = len(train_loader)
    if batches_per_epoch > 0:
        epochs_trained = (
            len(log_dict.get("train_generator_loss_per_batch", [])) // batches_per_epoch
        )
    else:
        epochs_trained = 0

    arch_params = _build_arch_params_gan(
        modelname=modelname,
        num_features=num_features,
        latent_dim=latent_dim,
    )

    return TrainedModel(
        model=final_model,
        model_state=final_model.state_dict(),
        arch_params=arch_params,
        log_dict=log_dict,
        epochs_trained=epochs_trained,
    )


def training_flows(
    rawdata,
    batch_frac,
    valid_batch_frac,
    random_seed,
    modelname,
    num_blocks,
    num_epochs,
    learning_rate,
    num_hidden,
    early_stop,
    early_stop_num,
    pre_model,
    tensorboard_dir: str | None = None,
    verbose=VerbosityLevel.MINIMAL,
) -> TrainedModel:
    """Train a normalizing flow model and return training artifacts only.

    Does **not** perform sample generation.

    Parameters
    ----------
    tensorboard_dir : str | None
        If set, create a TensorBoard ``SummaryWriter`` logging to this
        directory.  When ``None`` (default), TensorBoard logging is skipped.

    Returns
    -------
    TrainedModel
        Training-only output with model, state_dict, arch_params,
        log_dict, and epochs_trained.
    """
    set_all_seeds(random_seed)
    device = torch.device("cpu")
    num_inputs = rawdata.shape[1]
    num_samples = rawdata.shape[0]

    train_dataset = TensorDataset(rawdata)
    train_batch_size = round(batch_frac * num_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True
    )

    act = "tanh"

    modules = []
    if modelname == "glow":
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()
        for _ in range(num_blocks):
            modules += [
                ht.BatchNormFlow(num_inputs),
                ht.LUInvertibleMM(num_inputs),
                ht.CouplingLayer(
                    num_inputs,
                    num_hidden,
                    mask,
                    num_cond_inputs=None,
                    s_act="tanh",
                    t_act="relu",
                ),
            ]
            mask = 1 - mask
    elif modelname == "realnvp":
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()
        for _ in range(num_blocks):
            modules += [
                ht.CouplingLayer(
                    num_inputs,
                    num_hidden,
                    mask,
                    num_cond_inputs=None,
                    s_act="tanh",
                    t_act="relu",
                ),
                ht.BatchNormFlow(num_inputs),
            ]
            mask = 1 - mask
    elif modelname == "maf":
        for _ in range(num_blocks):
            modules += [
                ht.MADE(num_inputs, num_hidden, num_cond_inputs=None, act=act),
                ht.BatchNormFlow(num_inputs),
                ht.Reverse(num_inputs),
            ]
    elif modelname == "maf-split":
        for _ in range(num_blocks):
            modules += [
                ht.MADESplit(
                    num_inputs,
                    num_hidden,
                    num_cond_inputs=None,
                    s_act="tanh",
                    t_act="relu",
                ),
                ht.BatchNormFlow(num_inputs),
                ht.Reverse(num_inputs),
            ]
    elif modelname == "maf-split-glow":
        for _ in range(num_blocks):
            modules += [
                ht.MADESplit(
                    num_inputs,
                    num_hidden,
                    num_cond_inputs=None,
                    s_act="tanh",
                    t_act="relu",
                ),
                ht.BatchNormFlow(num_inputs),
                ht.InvertibleMM(num_inputs),
            ]

    model = ht.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.fill_(0)

    # transfer learning
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # Optional TensorBoard logging
    writer = None
    if tensorboard_dir is not None:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(log_dir=tensorboard_dir)

    global_step = 0
    train_loss_per_epoch: list[float] = []

    def train_one_epoch(epoch, global_step):
        model.train()
        train_loss = 0

        for _batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            data = data.to(device)
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalar("training/loss", loss.item(), global_step)
            global_step += 1

        for module in model.modules():
            if isinstance(module, ht.BatchNormFlow):
                module.momentum = 0

            with torch.no_grad():
                model(train_loader.dataset.tensors[0].to(data.device))

        for module in model.modules():
            if isinstance(module, ht.BatchNormFlow):
                module.momentum = 1

        return global_step, train_loss / len(train_loader.dataset)

    best_train_loss = float("inf")
    best_train_epoch = 0
    best_model = model
    progress_line_active = False

    if verbose >= VerbosityLevel.MINIMAL:
        msg = (
            f"Starting training: {num_epochs} epochs, model={modelname}, "
            f"num_blocks={num_blocks}, lr={learning_rate}"
        )
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()

    for epoch in range(num_epochs):
        global_step, train_loss = train_one_epoch(epoch, global_step)
        train_loss_per_epoch.append(train_loss)

        if early_stop:
            if (
                (epoch - best_train_epoch >= early_stop_num)
                or (math.isnan(train_loss))
                or (math.isinf(train_loss))
            ):
                if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                    print()
                    progress_line_active = False
                if verbose >= VerbosityLevel.MINIMAL:
                    print(
                        f"Early stopping at epoch {best_train_epoch + 1} "
                        f"(best log-likelihood: {-best_train_loss:.4f})"
                    )
                break

        if (
            (train_loss < best_train_loss)
            and not (math.isnan(train_loss))
            and not (math.isinf(train_loss))
        ):
            best_train_epoch = epoch
            best_train_loss = train_loss
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            ht._print_progress(
                epoch,
                num_epochs,
                {"neg_loglik": train_loss},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            ht._print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={"train_loss": train_loss},
                elapsed_time=time.time() - start_time,
            )

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    if writer is not None:
        writer.close()

    best_model.eval()

    log_dict = {"train_loss_per_epoch": train_loss_per_epoch}
    epochs_trained = len(train_loss_per_epoch)

    arch_params = _build_arch_params_flow(
        modelname=modelname,
        num_inputs=num_inputs,
        num_blocks=num_blocks,
        num_hidden=num_hidden,
    )

    return TrainedModel(
        model=best_model,
        model_state=best_model.state_dict(),
        arch_params=arch_params,
        log_dict=log_dict,
        epochs_trained=epochs_trained,
    )
