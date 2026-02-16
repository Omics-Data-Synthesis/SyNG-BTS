"""Centralized model reconstruction from ``arch_params`` and ``model_state``.

This module provides a single entry point — :func:`rebuild_model` — that
reconstructs a trained model from its architecture parameters and saved
state dict.  It supports all model families:

- **AE family**: AE, VAE, CVAE
- **GAN family**: GAN, WGAN, WGANGP
- **Flow family**: maf, realnvp, glow, maf-split, maf-split-glow
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from . import helper_train as ht
from .helper_models import AE, CVAE, GAN, VAE


def rebuild_model(
    arch_params: dict[str, Any],
    model_state: dict[str, Any],
) -> nn.Module:
    """Reconstruct a trained model from architecture params and state dict.

    Parameters
    ----------
    arch_params : dict[str, Any]
        Architecture parameters as stored in ``SyngResult.metadata["arch_params"]``
        or ``TrainedModel.arch_params``.  Must contain a ``"family"`` key.
    model_state : dict[str, Any]
        The model's ``state_dict()``.

    Returns
    -------
    nn.Module
        The reconstructed model in ``eval()`` mode with weights loaded.

    Raises
    ------
    ValueError
        If the family or modelname is not recognised.
    """
    family = arch_params["family"]
    if family == "ae":
        return _rebuild_ae(arch_params, model_state)
    elif family == "gan":
        return _rebuild_gan(arch_params, model_state)
    elif family == "flow":
        return _rebuild_flow(arch_params, model_state)
    else:
        raise ValueError(f"Unknown model family: {family!r}")


# ---------------------------------------------------------------------------
# AE family
# ---------------------------------------------------------------------------


def _rebuild_ae(
    arch_params: dict[str, Any],
    model_state: dict[str, Any],
) -> nn.Module:
    """Rebuild an AE / VAE / CVAE model."""
    modelname = arch_params["modelname"]
    num_features = arch_params["num_features"]

    if modelname == "AE":
        model = AE(num_features)
    elif modelname == "VAE":
        model = VAE(num_features)
    elif modelname == "CVAE":
        num_classes = arch_params["num_classes"]
        model = CVAE(num_features, num_classes)
    else:
        raise ValueError(f"Unknown AE-family model: {modelname!r}")

    model.load_state_dict(model_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GAN family
# ---------------------------------------------------------------------------


def _rebuild_gan(
    arch_params: dict[str, Any],
    model_state: dict[str, Any],
) -> nn.Module:
    """Rebuild a GAN / WGAN / WGANGP model."""
    num_features = arch_params["num_features"]
    latent_dim = arch_params["latent_dim"]

    model = GAN(num_features=num_features, latent_dim=latent_dim)
    model.load_state_dict(model_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Flow family
# ---------------------------------------------------------------------------


def _rebuild_flow(
    arch_params: dict[str, Any],
    model_state: dict[str, Any],
) -> nn.Module:
    """Rebuild a normalizing flow model.

    This mirrors the assembly logic in ``training_flows()`` from
    ``helper_training.py``, constructing the same module sequence
    from ``arch_params`` and then loading the saved state dict.
    """
    modelname = arch_params["modelname"]
    num_inputs = arch_params["num_inputs"]
    num_blocks = arch_params["num_blocks"]
    num_hidden = arch_params["num_hidden"]

    device = torch.device("cpu")
    modules: list[nn.Module] = []

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
                ht.MADE(num_inputs, num_hidden, num_cond_inputs=None, act="tanh"),
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
    else:
        raise ValueError(f"Unknown flow model: {modelname!r}")

    model = ht.FlowSequential(*modules)

    # Apply the same orthogonal initialisation as training_flows() so the
    # layer structure is compatible before loading state_dict.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.fill_(0)

    model.load_state_dict(model_state)
    model.num_inputs = num_inputs
    model.eval()
    return model
