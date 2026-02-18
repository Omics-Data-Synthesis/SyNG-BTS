from __future__ import annotations

import copy
import math
import time
from enum import IntEnum

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Verbosity helpers
# ---------------------------------------------------------------------------


class VerbosityLevel(IntEnum):
    """Verbosity levels for training output.

    Attributes
    ----------
    SILENT : int
        No output during training (value: 0).
    MINIMAL : int
        Print only completion summary (value: 1).
    DETAILED : int
        Print per-epoch progress (value: 2).
    """

    SILENT = 0
    MINIMAL = 1
    DETAILED = 2


_VERBOSE_MAP: dict[str, VerbosityLevel] = {
    "silent": VerbosityLevel.SILENT,
    "minimal": VerbosityLevel.MINIMAL,
    "detailed": VerbosityLevel.DETAILED,
}


def _resolve_verbose(verbose: int | str) -> VerbosityLevel:
    """Normalise a verbose argument to VerbosityLevel enum.

    Accepts ``0``, ``1``, ``2`` or the strings ``"silent"``,
    ``"minimal"``, ``"detailed"``.

    Returns
    -------
    VerbosityLevel
        The normalised verbosity level.
    """
    if isinstance(verbose, str):
        key = verbose.lower()
        if key not in _VERBOSE_MAP:
            raise ValueError(
                f"verbose must be 0, 1, 2 or one of {list(_VERBOSE_MAP)}, "
                f"got {verbose!r}"
            )
        return _VERBOSE_MAP[key]
    if verbose not in (0, 1, 2):
        raise ValueError(f"verbose must be 0, 1, or 2, got {verbose!r}")
    return VerbosityLevel(verbose)


def _print_progress(
    epoch: int,
    num_epochs: int,
    metrics: dict[str, float],
) -> None:
    """Print a ``\\r``-overwritten single-line progress bar (MINIMAL verbosity).

    The line is of the form::

        Epoch  3/100 |███░░░░░░░░░░░░░░░░░| loss: 0.1234

    It is **not** terminated with a newline so the next call overwrites it.
    Call ``print()`` once training ends to move past the progress line.
    """
    pct = (epoch + 1) / num_epochs
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(
        f"\rEpoch {epoch + 1:>{len(str(num_epochs))}}/{num_epochs} "
        f"|{bar}| {metrics_str}",
        end="",
        flush=True,
    )


def _print_training_state(
    epoch: int,
    num_epochs: int,
    loss_dict: dict[str, float],
    elapsed_time: float | None = None,
    learning_rate: float | None = None,
    early_stop_info: str | None = None,
) -> None:
    """Print epoch-level training state for ``verbose=2`` output.

    Parameters
    ----------
    epoch : int
        Current epoch number (0-indexed).
    num_epochs : int
        Total number of epochs.
    loss_dict : dict[str, float]
        Loss values (e.g. ``{"train_loss": 0.45, "val_loss": 0.50}``).
    elapsed_time : float or None
        Elapsed time in seconds, optional.
    learning_rate : float or None
        Current learning rate, optional.
    early_stop_info : str or None
        String with early stopping information, optional.
    """
    loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
    msg = f"Epoch {epoch + 1:03d}/{num_epochs:03d}: {loss_str}"

    if learning_rate is not None:
        msg += f" | LR: {learning_rate:.6f}"
    if elapsed_time is not None:
        msg += f" | Time: {elapsed_time / 60:.2f}min"
    if early_stop_info is not None:
        msg += f" | {early_stop_info}"

    print(msg)


# %%


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn):
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, _ in data_loader:
            logits = model(features)
            loss = loss_fn(logits, features, reduction="sum")
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
    return curr_loss


def WMSE(output, target, reduction="none"):
    weights = 1 / (torch.mean(target, 0) + 1)
    loss = weights * (output - target) ** 2
    return loss


# %%
def train_AE(
    num_epochs,
    model,
    optimizer,
    train_loader,
    val_loader,
    early_stop,
    early_stop_num,
    loss_fn="MSE",
    skip_epoch_stats=False,
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_loss_per_batch": [],
        "train_combined_loss_per_epoch": [],
        "val_loss_per_batch": [],
        "val_combined_loss_per_epoch": [],
    }

    loss_fn_name = loss_fn
    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE

    if verbose >= VerbosityLevel.MINIMAL:
        msg = f"Starting training: {num_epochs} epochs, loss={loss_fn_name}"
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_model = model
    progress_line_active = False

    val_features, _ = next(iter(val_loader))

    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        for _batch_idx, (features, _) in enumerate(train_loader):
            model.train()
            # FORWARD AND BACK PROP
            encoded, decoded = model(features)

            batchsize = features.shape[0]
            pixelwise = loss_fn(decoded, features, reduction="none")
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
            pixelwise = pixelwise.mean()  # average over batch dimension

            train_loss = pixelwise
            epoch_train_loss.append(train_loss.item())
            optimizer.zero_grad()

            train_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict["train_loss_per_batch"].append(pixelwise.item())

            # VALIDATION
            model.eval()
            with torch.no_grad():
                _, decoded_val = model(val_features)
                val_pixelwise = loss_fn(decoded_val, val_features, reduction="none")
                val_pixelwise = val_pixelwise.view(val_features.size(0), -1).sum(axis=1)
                val_pixelwise = val_pixelwise.mean()

                val_loss = val_pixelwise

            epoch_val_loss.append(val_loss.item())

            # LOGGING
            log_dict["val_loss_per_batch"].append(val_pixelwise.item())

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn
                )
                log_dict["train_combined_loss_per_epoch"].append(train_loss.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"train_loss": train_loss, "val_loss": val_loss},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={"train_loss": train_loss, "val_loss": val_loss},
                elapsed_time=time.time() - start_time,
            )

        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                print()  # newline after progress bar
                progress_line_active = False
            if verbose >= VerbosityLevel.MINIMAL:
                print(
                    f"Early stopping at epoch {best_epoch + 1} "
                    f"(best val_loss: {best_loss:.4f})"
                )
            break

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


# %%


def train_VAE(
    num_epochs,
    model,
    optimizer,
    train_loader,
    val_loader,
    early_stop,
    early_stop_num,
    loss_fn="MSE",
    skip_epoch_stats=False,
    reconstruction_term_weight=1,
    kl_weight=1,
    scheduler=None,
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_combined_loss_per_batch": [],
        "train_combined_loss_per_epoch": [],
        "train_reconstruction_loss_per_batch": [],
        "train_kl_loss_per_batch": [],
        "val_combined_loss_per_batch": [],
        "val_reconstruction_loss_per_batch": [],
        "val_kl_loss_per_batch": [],
        "latent_statistics_per_batch": [],
    }

    loss_fn_name = loss_fn
    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE

    if verbose >= VerbosityLevel.MINIMAL:
        msg = f"Starting training: {num_epochs} epochs, loss={loss_fn_name}, kl_weight={kl_weight}"
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_model = model
    progress_line_active = False

    val_features, _ = next(iter(val_loader))

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0][
            "lr"
        ]  # Check Learning Rate (with scheduler)

        epoch_train_loss = []
        epoch_val_loss = []
        for _batch_idx, (features, _) in enumerate(train_loader):
            model.train()
            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)

            # total loss = reconstruction loss + KL divergence
            # kl_divergence = (0.5 * (z_mean**2 +
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(
                1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
            )  # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()  # average over batch dimension

            pixelwise = loss_fn(decoded, features, reduction="none")
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
            pixelwise = pixelwise.mean()  # average over batch dimension

            train_loss = reconstruction_term_weight * pixelwise + kl_weight * kl_div
            epoch_train_loss.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict["train_combined_loss_per_batch"].append(train_loss.item())
            log_dict["train_reconstruction_loss_per_batch"].append(pixelwise.item())
            log_dict["train_kl_loss_per_batch"].append(kl_div.item())

            # VALIDATION
            model.eval()
            with torch.no_grad():
                _, z_mean_val, z_log_var_val, decoded_val = model(
                    val_features, deterministic=True
                )
                val_pixelwise = loss_fn(decoded_val, val_features, reduction="none")
                val_pixelwise = val_pixelwise.view(val_features.size(0), -1).sum(axis=1)
                val_pixelwise = val_pixelwise.mean()

                kl_div_val = -0.5 * torch.sum(
                    1 + z_log_var_val - z_mean_val**2 - torch.exp(z_log_var_val), axis=1
                )
                kl_div_val = kl_div_val.mean()

                val_loss = (
                    reconstruction_term_weight * val_pixelwise + kl_weight * kl_div_val
                )

            epoch_val_loss.append(val_loss.item())

            # LOGGING
            log_dict["val_combined_loss_per_batch"].append(val_loss.item())
            log_dict["val_reconstruction_loss_per_batch"].append(val_pixelwise.item())
            log_dict["val_kl_loss_per_batch"].append(kl_div_val.item())

        if scheduler is not None:
            scheduler.step()

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn
                )
                log_dict["train_combined_loss_per_epoch"].append(train_loss.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"kl": kl_div.item(), "recons": pixelwise.item(), "val": val_loss},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={"kl": kl_div.item(), "recons": pixelwise.item()},
                elapsed_time=time.time() - start_time,
                learning_rate=current_lr,
            )

        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                print()  # newline after progress bar
                progress_line_active = False
            if verbose >= VerbosityLevel.MINIMAL:
                print(
                    f"Early stopping at epoch {best_epoch + 1} "
                    f"(best val_loss: {best_loss:.4f})"
                )
            break

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


# %%


def train_CVAE(
    num_epochs,
    model,
    optimizer,
    train_loader,
    val_loader,
    early_stop,
    early_stop_num,
    loss_fn="MSE",
    skip_epoch_stats=False,
    reconstruction_term_weight=1,
    kl_weight=1,
    scheduler=None,
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_combined_loss_per_batch": [],
        "train_combined_loss_per_epoch": [],
        "train_reconstruction_loss_per_batch": [],
        "train_kl_loss_per_batch": [],
        "val_combined_loss_per_batch": [],
        "val_reconstruction_loss_per_batch": [],
        "val_kl_loss_per_batch": [],
        "latent_statistics_per_batch": [],
    }

    loss_fn_name = loss_fn
    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE

    if verbose >= VerbosityLevel.MINIMAL:
        msg = f"Starting training: {num_epochs} epochs, loss={loss_fn_name}, kl_weight={kl_weight}"
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_model = model
    progress_line_active = False

    val_features, val_labels = next(iter(val_loader))
    if val_labels.dim() == 1:
        val_labels = val_labels.unsqueeze(1)
    val_labels = val_labels.float()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0][
            "lr"
        ]  # Check Learning Rate (with scheduler)

        epoch_train_loss = []
        epoch_val_loss = []
        for _batch_idx, (features, lab) in enumerate(train_loader):
            model.train()
            if lab.dim() == 1:  # label (dim = 1)
                lab = lab.unsqueeze(1)
            lab = lab.float()

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features, lab)

            # total loss = reconstruction loss + KL divergence
            # kl_divergence = (0.5 * (z_mean**2 +
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(
                1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
            )  # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()  # average over batch dimension

            pixelwise = loss_fn(decoded, features, reduction="none")
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
            pixelwise = pixelwise.mean()  # average over batch dimension

            train_loss = reconstruction_term_weight * pixelwise + kl_weight * kl_div
            epoch_train_loss.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict["train_combined_loss_per_batch"].append(train_loss.item())
            log_dict["train_reconstruction_loss_per_batch"].append(pixelwise.item())
            log_dict["train_kl_loss_per_batch"].append(kl_div.item())

            # VALIDATION
            model.eval()
            with torch.no_grad():
                _, z_mean_val, z_log_var_val, decoded_val = model(
                    val_features, val_labels, deterministic=True
                )
                val_pixelwise = loss_fn(decoded_val, val_features, reduction="none")
                val_pixelwise = val_pixelwise.view(val_features.size(0), -1).sum(axis=1)
                val_pixelwise = val_pixelwise.mean()

                kl_div_val = -0.5 * torch.sum(
                    1 + z_log_var_val - z_mean_val**2 - torch.exp(z_log_var_val), axis=1
                )
                kl_div_val = kl_div_val.mean()

                val_loss = (
                    reconstruction_term_weight * val_pixelwise + kl_weight * kl_div_val
                )

            epoch_val_loss.append(val_loss.item())

            # LOGGING
            log_dict["val_combined_loss_per_batch"].append(val_loss.item())
            log_dict["val_reconstruction_loss_per_batch"].append(val_pixelwise.item())
            log_dict["val_kl_loss_per_batch"].append(kl_div_val.item())

        if scheduler is not None:
            scheduler.step()

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn
                )
                log_dict["train_combined_loss_per_epoch"].append(train_loss.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"kl": kl_div.item(), "recons": pixelwise.item(), "val": val_loss},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={"kl": kl_div.item(), "recons": pixelwise.item()},
                elapsed_time=time.time() - start_time,
                learning_rate=current_lr,
            )

        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                print()  # newline after progress bar
                progress_line_active = False
            if verbose >= VerbosityLevel.MINIMAL:
                print(
                    f"Early stopping at epoch {best_epoch + 1} "
                    f"(best val_loss: {best_loss:.4f})"
                )
            break

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


# %%
def train_GAN(
    num_epochs,
    model,
    optimizer_gen,
    optimizer_discr,
    latent_dim,
    train_loader,
    early_stop=None,
    early_stop_num=None,  # loss for GAN are not meaningful, so early stopping rule is not applied.
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_generator_loss_per_batch": [],
        "train_discriminator_loss_per_batch": [],
        "train_discriminator_real_acc_per_batch": [],
        "train_discriminator_fake_acc_per_batch": [],
    }

    loss_fn = F.binary_cross_entropy_with_logits

    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Starting training: {num_epochs} epochs, latent_dim={latent_dim}")

    start_time = time.time()
    best_model = model
    for epoch in range(num_epochs):
        model.train()
        for _batch_idx, (features, _) in enumerate(train_loader):
            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size)  # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size)  # fake label = 0
            flipped_fake_labels = real_labels  # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(
                -1
            )  # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5 * (real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------
            log_dict["train_generator_loss_per_batch"].append(gener_loss.item())
            log_dict["train_discriminator_loss_per_batch"].append(discr_loss.item())

            predicted_labels_real = torch.where(
                discr_pred_real.detach() > 0.0, 1.0, 0.0
            )
            predicted_labels_fake = torch.where(
                discr_pred_fake.detach() > 0.0, 1.0, 0.0
            )
            acc_real = (predicted_labels_real == real_labels).float().mean() * 100.0
            acc_fake = (predicted_labels_fake == fake_labels).float().mean() * 100.0
            log_dict["train_discriminator_real_acc_per_batch"].append(acc_real.item())
            log_dict["train_discriminator_fake_acc_per_batch"].append(acc_fake.item())

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"gen": gener_loss.item(), "disc": discr_loss.item()},
            )
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={
                    "generator": gener_loss.item(),
                    "discriminator": discr_loss.item(),
                },
                elapsed_time=time.time() - start_time,
            )

    if verbose == VerbosityLevel.MINIMAL:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


# %%
def train_WGAN(
    num_epochs,
    model,
    optimizer_gen,
    optimizer_discr,
    latent_dim,
    train_loader,
    early_stop,
    early_stop_num,
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_generator_loss_per_batch": [],
        "train_discriminator_loss_per_batch": [],
        "train_discriminator_real_acc_per_batch": [],
        "train_discriminator_fake_acc_per_batch": [],
    }

    # if loss == 'regular':
    #     loss_fn = F.binary_cross_entropy_with_logits
    # elif loss == 'wasserstein':
    #     def loss_fn(y_pred, y_true):
    #         return -torch.mean(y_pred * y_true)
    # else:
    #     raise ValueError('This loss is not supported.')
    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    if verbose >= VerbosityLevel.MINIMAL:
        msg = f"Starting training: {num_epochs} epochs, latent_dim={latent_dim}"
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_model = model
    progress_line_active = False
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        for _batch_idx, (features, _) in enumerate(train_loader):
            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size)  # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)
            fake_images = model.generator_forward(noise)

            # if loss == 'regular':
            #     fake_labels = torch.zeros(batch_size) # fake label = 0
            # elif loss == 'wasserstein':
            #     fake_labels = -real_labels # fake label = -1
            fake_labels = -real_labels  # fake label = -1
            flipped_fake_labels = real_labels  # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(
                -1
            )  # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5 * (real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # if loss == 'wasserstein':
            #     for p in model.discriminator.parameters():
            #         p.data.clamp_(-0.01, 0.01)
            for p in model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------
            log_dict["train_generator_loss_per_batch"].append(gener_loss.item())
            log_dict["train_discriminator_loss_per_batch"].append(discr_loss.item())

            epoch_loss.append(discr_loss.item())

            predicted_labels_real = torch.where(
                discr_pred_real.detach() > 0.0, 1.0, 0.0
            )
            predicted_labels_fake = torch.where(
                discr_pred_fake.detach() > 0.0, 1.0, 0.0
            )
            acc_real = (predicted_labels_real == real_labels).float().mean() * 100.0
            acc_fake = (predicted_labels_fake == fake_labels).float().mean() * 100.0
            log_dict["train_discriminator_real_acc_per_batch"].append(acc_real.item())
            log_dict["train_discriminator_fake_acc_per_batch"].append(acc_fake.item())

        train_loss = sum(epoch_loss) / len(epoch_loss)
        if (abs(train_loss) < abs(best_loss)) & (epoch >= 10):
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"gen": gener_loss.item(), "disc": discr_loss.item()},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={
                    "generator": gener_loss.item(),
                    "discriminator": discr_loss.item(),
                },
                elapsed_time=time.time() - start_time,
            )

        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                print()  # newline after progress bar
                progress_line_active = False
            if verbose >= VerbosityLevel.MINIMAL:
                print(
                    f"Early stopping at epoch {best_epoch + 1} "
                    f"(best loss: {best_loss:.4f})"
                )
            break

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


def train_WGANGP(
    num_epochs,
    model,
    optimizer_gen,
    optimizer_discr,
    latent_dim,
    train_loader,
    early_stop,
    early_stop_num,
    discr_iter_per_generator_iter=5,
    gradient_penalty=True,
    gradient_penalty_weight=10,
    verbose=VerbosityLevel.MINIMAL,
):

    log_dict = {
        "train_generator_loss_per_batch": [],
        "train_discriminator_loss_per_batch": [],
        "train_discriminator_real_acc_per_batch": [],
        "train_discriminator_fake_acc_per_batch": [],
    }

    if gradient_penalty:
        log_dict["train_gradient_penalty_loss_per_batch"] = []

    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    if verbose >= VerbosityLevel.MINIMAL:
        msg = f"Starting training: {num_epochs} epochs, latent_dim={latent_dim}"
        if gradient_penalty:
            msg += f", gradient_penalty_weight={gradient_penalty_weight}"
        msg += f", discr_iter_per_gen={discr_iter_per_generator_iter}"
        if early_stop:
            msg += f", early_stop={early_stop_num}"
        print(msg)

    start_time = time.time()

    skip_generator = 1
    best_loss = float("inf")
    best_epoch = 0
    best_model = model
    progress_line_active = False
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()

        for _batch_idx, (features, _) in enumerate(train_loader):
            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size)  # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  # format NCHW
            fake_images = model.generator_forward(noise)

            fake_labels = -real_labels  # fake label = -1
            flipped_fake_labels = real_labels  # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(
                -1
            )  # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5 * (real_loss + fake_loss)

            ###################################################
            # gradient penalty
            if gradient_penalty:
                alpha = torch.rand(batch_size, 1)

                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True

                discr_out = model.discriminator_forward(interpolated)

                grad_values = torch.ones(discr_out.size())
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + epsilon)

                gp_penalty_term = (
                    (gradients_norm - 1) ** 2
                ).mean() * gradient_penalty_weight
                discr_loss += gp_penalty_term

                log_dict["train_gradient_penalty_loss_per_batch"].append(
                    gp_penalty_term.item()
                )
            #######################################################

            discr_loss.backward()

            optimizer_discr.step()

            # Use weight clipping (standard Wasserstein GAN)
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            if skip_generator <= discr_iter_per_generator_iter:
                # --------------------------
                # Train Generator
                # --------------------------

                optimizer_gen.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                optimizer_gen.step()

                skip_generator += 1

            else:
                skip_generator = 1
                gener_loss = torch.tensor(0.0)

            # --------------------------
            # Logging
            # --------------------------
            epoch_loss.append(discr_loss.item())
            log_dict["train_generator_loss_per_batch"].append(gener_loss.item())
            log_dict["train_discriminator_loss_per_batch"].append(discr_loss.item())

            predicted_labels_real = torch.where(
                discr_pred_real.detach() > 0.0, 1.0, 0.0
            )
            predicted_labels_fake = torch.where(
                discr_pred_fake.detach() > 0.0, 1.0, 0.0
            )
            acc_real = (predicted_labels_real == real_labels).float().mean() * 100.0
            acc_fake = (predicted_labels_fake == fake_labels).float().mean() * 100.0
            log_dict["train_discriminator_real_acc_per_batch"].append(acc_real.item())
            log_dict["train_discriminator_fake_acc_per_batch"].append(acc_fake.item())

        train_loss = sum(epoch_loss) / len(epoch_loss)
        if (abs(train_loss) < abs(best_loss)) & (epoch >= 10):
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if verbose == VerbosityLevel.MINIMAL:
            _print_progress(
                epoch,
                num_epochs,
                {"gen": gener_loss.item(), "disc": discr_loss.item()},
            )
            progress_line_active = True
        elif verbose == VerbosityLevel.DETAILED:
            _print_training_state(
                epoch=epoch,
                num_epochs=num_epochs,
                loss_dict={
                    "generator": gener_loss.item(),
                    "discriminator": discr_loss.item(),
                },
                elapsed_time=time.time() - start_time,
            )

        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            if verbose == VerbosityLevel.MINIMAL and progress_line_active:
                print()  # newline after progress bar
                progress_line_active = False
            if verbose >= VerbosityLevel.MINIMAL:
                print(
                    f"Early stopping at epoch {best_epoch + 1} "
                    f"(best loss: {best_loss:.4f})"
                )
            break

    if verbose == VerbosityLevel.MINIMAL and progress_line_active:
        print()  # newline after progress bar
    total_time = (time.time() - start_time) / 60
    if verbose >= VerbosityLevel.MINIMAL:
        print(f"Training complete: {total_time:.2f}min")

    return log_dict, best_model


##%
## Flow models start here.....
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == "input":
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, mask, cond_in_features=None, bias=True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)

        self.register_buffer("mask", mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class MADESplit(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        s_act="tanh",
        t_act="relu",
        pre_exp_tanh=False,
    ):
        super().__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs, mask_type="output")

        act_func = activations[s_act]
        self.s_joiner = MaskedLinear(
            num_inputs, num_hidden, input_mask, num_cond_inputs
        )

        self.s_trunk = nn.Sequential(
            act_func(),
            MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            MaskedLinear(num_hidden, num_inputs, output_mask),
        )

        act_func = activations[t_act]
        self.t_joiner = MaskedLinear(
            num_inputs, num_hidden, input_mask, num_cond_inputs
        )

        self.t_trunk = nn.Sequential(
            act_func(),
            MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            MaskedLinear(num_hidden, num_inputs, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)

            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)

            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class MADE(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        act="relu",
        pre_exp_tanh=False,
    ):
        super().__init__()

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type="output"
        )

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.trunk = nn.Sequential(
            act_func(),
            MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True
            )
        else:
            return torch.log(inputs / (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                -1, keepdim=True
            )


class Logit(Sigmoid):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return super().forward(inputs, "inverse")
        else:
            return super().forward(inputs, "direct")


class BatchNormFlow(nn.Module):
    """An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super().__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)


class ActNorm(nn.Module):
    """An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if not self.initialized:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == "direct":
            return (inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                -1, keepdim=True
            ).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(-self.weight) + self.bias, -self.weight.sum(
                -1, keepdim=True
            ).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs @ self.W, torch.slogdet(self.W)[-1].unsqueeze(0).unsqueeze(
                0
            ).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(self.W)[-1].unsqueeze(
                0
            ).unsqueeze(0).repeat(inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == "direct":
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                inputs.size(0), 1
            )
        else:
            return inputs @ torch.inverse(W), -self.log_S.sum().unsqueeze(0).unsqueeze(
                0
            ).repeat(inputs.size(0), 1)


class Shuffle(nn.Module):
    """An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.register_buffer("perm", torch.randperm(num_inputs))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )


class CouplingLayer(nn.Module):
    """An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        mask,
        num_cond_inputs=None,
        s_act="tanh",
        t_act="relu",
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == "direct":
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None, save=None, save_step=None):
        u, log_jacob = self(inputs, cond_inputs)

        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True
        )

        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples
