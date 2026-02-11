import os
import random

import numpy as np
import torch


def preprocessinglog2(dataset):
    # log2 pre-processing of count data
    return torch.log2(dataset + 1)


def set_all_seeds(seed):
    # set random seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_labels(n_samples, groups=None):
    # create binary labels and blurry labels for training two-group data
    # Use a local generator so we don't mutate global RNG state.
    _rng = torch.Generator().manual_seed(10)
    if groups is None:
        labels = torch.zeros([n_samples, 1])
        blurlabels = labels
    else:
        base = groups[0]
        labels = torch.zeros([n_samples, 1]).to(torch.float32)
        labels[groups != base, 0] = 1
        blurlabels = torch.zeros([n_samples, 1]).to(torch.float32)
        blurlabels[groups != base, 0] = (10 - 9) * torch.rand(
            sum(groups != base), generator=_rng
        ) + 9
        blurlabels[groups == base, 0] = (1 - 0) * torch.rand(
            sum(groups == base), generator=_rng
        ) + 0
    return labels, blurlabels


def create_labels_mul(n_samples, groups=None):
    # Use a local generator so we don't mutate global RNG state.
    _rng = torch.Generator().manual_seed(10)

    if groups is None:
        labels = torch.zeros([n_samples, 1], dtype=torch.float32)
        blurlabels = labels.clone()
        return labels, blurlabels

    groups_cat = groups.astype("category")
    codes = groups_cat.cat.codes
    group_tensor = torch.from_numpy(codes.copy().values)
    labels = group_tensor.float().unsqueeze(1)
    blurlabels = labels + torch.rand(labels.shape, generator=_rng)
    return labels, blurlabels


def draw_pilot(dataset, labels, blurlabels, n_pilot, seednum):
    # draw pilot datasets
    set_all_seeds(
        seednum
    )  # each draw has its own seednum, so guaranteed that 25 replicated sets are not the same
    n_samples = dataset.shape[0]
    if torch.unique(labels).shape[0] == 1:
        shuffled_indices = torch.randperm(n_samples)
        pilot_indices = shuffled_indices[-n_pilot:]
        rawdata = dataset[pilot_indices, :]
        rawlabels = labels[pilot_indices, :]
        rawblurlabels = blurlabels[pilot_indices, :]
    else:
        base = labels[0, :]
        n_pilot_1 = n_pilot
        n_pilot_2 = n_pilot
        n_samples_1 = sum(labels[:, 0] == base)
        n_samples_2 = sum(labels[:, 0] != base)
        dataset_1 = dataset[labels[:, 0] == base, :]
        dataset_2 = dataset[labels[:, 0] != base, :]
        labels_1 = labels[labels[:, 0] == base, :]
        labels_2 = labels[labels[:, 0] != base, :]
        blurlabels_1 = blurlabels[labels[:, 0] == base, :]
        blurlabels_2 = blurlabels[labels[:, 0] != base, :]
        shuffled_indices_1 = torch.randperm(n_samples_1)
        pilot_indices_1 = shuffled_indices_1[-n_pilot_1:]
        rawdata_1 = dataset_1[pilot_indices_1, :]
        rawlabels_1 = labels_1[pilot_indices_1, :]
        rawblurlabels_1 = blurlabels_1[pilot_indices_1, :]
        shuffled_indices_2 = torch.randperm(n_samples_2)
        pilot_indices_2 = shuffled_indices_2[-n_pilot_2:]
        rawdata_2 = dataset_2[pilot_indices_2, :]
        rawlabels_2 = labels_2[pilot_indices_2, :]
        rawblurlabels_2 = blurlabels_2[pilot_indices_2, :]
        rawdata = torch.cat((rawdata_1, rawdata_2), dim=0)
        rawlabels = torch.cat((rawlabels_1, rawlabels_2), dim=0)
        rawblurlabels = torch.cat((rawblurlabels_1, rawblurlabels_2), dim=0)
    return rawdata, rawlabels, rawblurlabels


def Gaussian_aug(rawdata, rawlabels, multiplier):
    # Gaussian augmentation
    # This function performs offline augmentation by adding gaussian noise to the
    # log2 counts, rawdata is the data generated from draw_pilot(), so does rawlabels,
    # multiplier specifies the number of samples for each kind of label, must be a list if
    # unique labels > 1. This function generates rawdata and rawlabels again but with
    # gaussian augmented data with size multiplier*n_rawdata

    oriraw = rawdata
    orirawlabels = rawlabels
    for all_mult in multiplier:
        for mult in list(range(all_mult)):
            rawdata = torch.cat(
                (
                    rawdata,
                    oriraw
                    + torch.normal(
                        mean=0, std=1, size=(oriraw.shape[0], oriraw.shape[1])
                    ),
                ),
                dim=0,
            )
            rawlabels = torch.cat((rawlabels, orirawlabels), dim=0)

    return rawdata, rawlabels


def reconstruct_samples(
    model,
    modelname: str,
    data_loader,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct samples through a trained autoencoder.

    Passes all batches in *data_loader* through *model* and concatenates
    the originals with their reconstructions.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder (AE, VAE, or CVAE).
    modelname : str
        Model type identifier (``"AE"``, ``"VAE"``, or ``"CVAE"``).
    data_loader : DataLoader
        DataLoader yielding ``(features, labels)`` tuples.
    n_features : int
        Number of input features (columns) in the data.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(data, labels)`` where *data* is the vertical concatenation
        ``[originals; reconstructions]`` with shape
        ``[2 * n_samples, n_features(+1 for CVAE)]`` and *labels* is
        the label tensor.
    """
    orig_all = torch.zeros([1, n_features])
    decoded_all = torch.zeros([1, n_features])
    labels = torch.zeros(0, dtype=torch.long)

    for batch_idx, (features, lab) in enumerate(data_loader):
        # Compatible with two types of labels:
        # - (N, C) one-hot -> use argmax
        # - (N, 1) single column/real number 0/1 -> directly squeeze to (N,)
        if isinstance(lab, torch.Tensor):
            if lab.dim() == 2:
                if lab.size(1) > 1:
                    labels_batch = torch.argmax(lab, dim=1)
                else:
                    labels_batch = lab.squeeze(1).long()
            else:
                labels_batch = lab.long()
        else:
            labels_batch = torch.as_tensor(lab).long()
        labels = torch.cat((labels, labels_batch), dim=0)

        with torch.no_grad():
            if modelname == "CVAE":
                # Ensure labels_batch is 2D for concatenation
                labels_for_model = (
                    labels_batch.unsqueeze(1)
                    if labels_batch.dim() == 1
                    else labels_batch
                )
                encoded, z_mean, z_log_var, decoded_images = model(
                    features, labels_for_model
                )
            elif modelname == "VAE":
                encoded, z_mean, z_log_var, decoded_images = model(features)
            else:
                encoded, decoded_images = model(features)

        orig_all = torch.cat((orig_all, features), dim=0)
        decoded_all = torch.cat((decoded_all, decoded_images), dim=0)

    orig_all = orig_all[1:]
    decoded_all = decoded_all[1:]

    if modelname == "CVAE":
        labels = labels.unsqueeze(1).float()
        orig_all = torch.cat((orig_all, labels), dim=1)
        decoded_all = torch.cat((decoded_all, labels), dim=1)

    return torch.cat((orig_all, decoded_all), dim=0).detach(), labels


def generate_samples(
    model,
    modelname: str,
    latent_size: int,
    num_images: int | list[int],
    col_max: torch.Tensor | None = None,
    col_sd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate synthetic samples from a trained generative model.

    Parameters
    ----------
    model : nn.Module
        Trained generative model (AE, VAE, CVAE, GAN, or flow).
    modelname : str
        Model type identifier (``"AE"``, ``"VAE"``, ``"CVAE"``,
        ``"GANs"``, ``"glow"``, ``"realnvp"``, ``"maf"``).
    latent_size : int
        Dimensionality of the latent space.
    num_images : int or list[int]
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
    """
    with torch.no_grad():
        if isinstance(num_images, int) or len(num_images) == 1:
            if not isinstance(num_images, int):
                num_images = num_images[0]
            rand_features = torch.randn(num_images, latent_size)
            if modelname == "CVAE":
                num_classes = model.num_classes
                base = num_images // num_classes
                rem = num_images % num_classes
                counts = [base] * num_classes
                for i in range(rem):
                    counts[i] += 1
                labels_list = []
                for class_id, n_c in enumerate(counts):
                    ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                    labels_list.append(ids)
                one_group_labels = torch.cat(labels_list)
                labels = one_group_labels.unsqueeze(1)

                rand_features = torch.cat((rand_features, labels), dim=1)
                new_images = model.decoder(rand_features)
                new_images = torch.cat((new_images, labels), dim=1)
            elif modelname == "AE":
                new_images = model.decoder(rand_features)
            elif modelname == "VAE":
                new_images = model.decoder(rand_features)
            elif modelname == "GANs":
                new_images = model.generator(rand_features)
            elif modelname in ("glow", "realnvp", "maf"):
                new_images = model.sample(num_images)
        else:
            # Multi-group: num_images = [n_for_0, ..., n_for_(K-1), replicate]
            counts = num_images[:-1]
            repli = num_images[-1]
            num_images_repe = sum(counts)
            total = num_images_repe * repli
            rand_features = torch.randn(total, latent_size)
            if modelname == "CVAE":
                labels_list = []
                for class_id, n_c in enumerate(counts):
                    ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                    labels_list.append(ids)
                one_group_labels = torch.cat(labels_list)
                labels = one_group_labels.repeat(repli).unsqueeze(1)

                rand_features = torch.cat((rand_features, labels), dim=1)
                new_images = model.decoder(rand_features)
                new_images = torch.cat((new_images, labels), dim=1)
            elif modelname == "AE":
                new_images = model.decoder(rand_features)
            elif modelname == "VAE":
                new_images = model.decoder(rand_features)
            elif modelname == "GANs":
                new_images = model.generator(rand_features)
            elif modelname in ("glow", "realnvp", "maf"):
                new_images = model.sample(total)

        # Cap: threshold outliers to col_max + noise
        if (col_max is not None) and (col_sd is not None):
            device = new_images.device
            col_max = col_max.to(device)
            col_sd = col_sd.to(device)

            # Exclude CVAE label column from capping
            feat_cols = new_images.size(1) - (1 if modelname == "CVAE" else 0)
            x = new_images[:, :feat_cols]

            # threshold = col_max + col_sd
            thr = col_max[:feat_cols] + col_sd[:feat_cols]
            noise = torch.normal(
                mean=torch.zeros_like(col_sd[:feat_cols]),
                std=0.1 * col_sd[:feat_cols],
            )
            # cap value = col_max + small noise
            cap_val = col_max[:feat_cols] + noise

            mask = x > thr
            x = torch.where(mask, cap_val.unsqueeze(0).expand_as(x), x)
            new_images = torch.cat((x, new_images[:, feat_cols:]), dim=1)

    return new_images
