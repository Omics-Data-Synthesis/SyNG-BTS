"""Unified inference dispatcher for SyNG-BTS.

All post-training generation and reconstruction is routed
through this module, cleanly separating inference from training
orchestrators.

Public functions
----------------
- ``run_generation`` — generate synthetic samples from any trained model.
- ``run_reconstruction`` — reconstruct input data through AE-family models.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .helper_training import TrainedModel
from .helper_utils import generate_samples, reconstruct_samples


def run_generation(
    trained: TrainedModel,
    num_samples: int | list[int],
    *,
    col_max: torch.Tensor | None = None,
    col_sd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate synthetic samples from a trained model.

    This is the single dispatch point for all model families (AE, VAE,
    CVAE, GAN, WGAN, WGANGP, and all flow variants).

    Parameters
    ----------
    trained : TrainedModel
        Output from a training wrapper.
    num_samples : int or list[int]
        Number of samples to generate.  For CVAE with multiple groups,
        pass ``[n_group_0, n_group_1, ..., replicate_factor]``.
    col_max : torch.Tensor or None
        Per-feature maximum values for capping (optional).
    col_sd : torch.Tensor or None
        Per-feature standard deviations for capping (optional).

    Returns
    -------
    torch.Tensor
        Generated samples as a 2-D tensor ``[n_samples, n_features]``.

    Raises
    ------
    ValueError
        If the model family in ``trained.arch_params`` is not recognised.
    """
    family = trained.arch_params["family"]
    modelname = trained.arch_params["modelname"]
    model = trained.model

    # Determine latent size based on model family
    if family == "ae":
        latent_size = trained.arch_params["latent_size"]
    elif family == "gan":
        latent_size = trained.arch_params["latent_dim"]
    elif family == "flow":
        latent_size = trained.arch_params["num_hidden"]
    else:
        raise ValueError(f"Unknown model family: {family!r}")

    # GANs use "GANs" as modelname for the generation function
    gen_modelname = "GANs" if family == "gan" else modelname

    return generate_samples(
        model=model,
        modelname=gen_modelname,
        latent_size=latent_size,
        num_images=num_samples,
        col_max=col_max,
        col_sd=col_sd,
    )


def run_reconstruction(
    trained: TrainedModel,
    data_loader: DataLoader,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct samples through a trained autoencoder.

    Only valid for AE-family models (AE, VAE, CVAE).  Passes all
    batches in *data_loader* through the model and returns originals
    concatenated with reconstructions.

    Parameters
    ----------
    trained : TrainedModel
        Output from a trained AE-family wrapper.
    data_loader : DataLoader
        DataLoader yielding ``(features, labels)`` tuples.
    n_features : int
        Number of input features (columns) in the data.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(data, labels)`` where *data* is the vertical concatenation
        ``[originals; reconstructions]`` and *labels* is the label
        tensor.

    Raises
    ------
    ValueError
        If the model family is not ``"ae"``.
    """
    family = trained.arch_params["family"]
    if family != "ae":
        raise ValueError(
            f"Reconstruction is only supported for AE-family models, "
            f"got family={family!r}"
        )

    modelname = trained.arch_params["modelname"]
    model = trained.model

    return reconstruct_samples(
        model=model,
        modelname=modelname,
        data_loader=data_loader,
        n_features=n_features,
    )
