# -*- coding: utf-8 -*-

# %% Import libraries
import torch
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from .helper_utils import *
from .helper_training import *
from .data_utils import load_dataset, get_output_path, ensure_dir
import re

sns.set()


# %% Define pilot experiments functions
def PilotExperiment(
    dataname: str,
    pilot_size: List[int],
    model: str,
    batch_frac: float,
    learning_rate: float,
    epoch: Optional[int] = None,
    early_stop_num: int = 30,
    off_aug: Optional[str] = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    pre_model: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    r"""
    This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP with several pilot sizes given data, model, batch_size, learning_rate, epoch, off_aug and pre_model.
    For each pilot size, there will be 5 random draws from the original dataset.
    For each draw, the pilot data is served as the input to the model training, and the generated data has sample size equal to 5 times the original sample size.

    Parameters
    ----------
    dataname : string
        pure data name without .csv. Eg: SKCMPositive_4
    pilot_size : list
        a list including potential pilot sizes
    model : string
        name of the model to be trained
    batch_frac : float
        batch fraction
    learning_rate : float
        learning rate
    epoch : int
                            choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
    early_stop_num : int
        if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == “None”
    off_aug : string (AE_head or Gaussian_head or None)
        choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    AE_head_num : int
        how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
    Gaussian_head_num : int
        how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
    pre_model : string
        transfer learning input model. If pre_model == None, no transfer learning
    data_dir : str, Path, or None
        Directory to read input data from. If None, will attempt to load the dataset from the package's bundled data or from the current working directory.
    output_dir : str, Path, or None
        Directory to write output files (reconstructed data, generated samples, loss logs, etc.). If None, the current working directory is used.

    """
    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path.cwd()

    # Read in data
    data_path = None
    if data_dir is not None:
        data_path = Path(data_dir) / f"{dataname}.csv"

    try:
        # Try loading from specified path or bundled data
        df = load_dataset(dataname, data_path=data_path)
        print(f"1. Read data: {dataname}")
    except FileNotFoundError:
        # Fallback to legacy path for backward compatibility
        legacy_path = Path("../RealData") / f"{dataname}.csv"
        if legacy_path.exists():
            df = pd.read_csv(legacy_path, header=0)
            print(f"1. Read data, path is {legacy_path}")
        else:
            raise FileNotFoundError(
                f"Could not find dataset '{dataname}'. "
                f"Specify data_dir or ensure the file exists."
            )

    dat_pd = df
    data_pd = dat_pd.select_dtypes(include=np.number)
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
    colnames = data_pd.columns

    # log2 transformation
    oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]

    # get group information if there is or is not
    if "groups" in dat_pd.columns:
        groups = dat_pd["groups"]
    else:
        groups = None

    # create 0-1 labels, this function use the first element in groups as 0.
    # also create blurlabels.
    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    # get model name and kl_weight if modelname is some autoencoder
    if len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))

    # decide batch fraction in file name
    model = "batch" + str(batch_frac).replace(".", "") + "_" + model

    # decide epochs
    if epoch is not None:
        num_epochs = epoch
        early_stop = False
        epoch_info = str(epoch)
        model = "epoch" + epoch_info + "_" + model
    else:
        num_epochs = 1000
        early_stop = True
        epoch_info = "early_stop"
        model = "epochES_" + model

    # decide offline augmentation
    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Determine the training parameters are epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learing rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
    )

    random_seed = 123
    repli = 5

    if (len(torch.unique(orilabels)) > 1) & (
        int(sum(orilabels == 0)) != int(sum(orilabels == 1))
    ):
        new_size = [int(sum(orilabels == 0)), int(sum(orilabels == 1)), repli]
    else:
        new_size = [repli * n_samples]

    if pre_model is not None:
        model = model + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    print("4. Pilot experiments start ... ")
    for n_pilot in pilot_size:
        for rand_pilot in [1, 2, 3, 4, 5]:
            print(
                "Training for data="
                + dataname
                + ", model="
                + model
                + ", pilot size="
                + str(n_pilot)
                + ", for "
                + str(rand_pilot)
                + "-th draw"
            )

            # get pilot_size real samples as seeds for DGM. For two cancers, the first n_pilot are from group 0, the second n_pilot are from group 1
            rawdata, rawlabels, rawblurlabels = draw_pilot(
                dataset=oridata,
                labels=orilabels,
                blurlabels=oriblurlabels,
                n_pilot=n_pilot,
                seednum=rand_pilot,
            )

            # for training of two cancers without CVAE, we use blurlabels as an additional feature to train
            if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
                rawdata = torch.cat((rawdata, rawblurlabels), dim=1)

            # Build output file names
            base_name = f"{dataname}_{model}_{n_pilot}_Draw{rand_pilot}.csv"
            savepath = str(get_output_path(output_dir, "ReconsData", base_name))
            savepathnew = str(get_output_path(output_dir, "GeneratedData", base_name))
            losspath = str(get_output_path(output_dir, "Loss", base_name))

            # whether or not add Gaussian_head augmentation
            if Gaussian_head:
                rawdata, rawlabels = Gaussian_aug(
                    rawdata, rawlabels, multiplier=[Gaussian_head_num]
                )
                # Update paths for Gaussian head augmentation
                gauss_base_name = (
                    f"{dataname}_Gaussianhead_{model}_{n_pilot}_Draw{rand_pilot}.csv"
                )
                savepath = str(
                    get_output_path(output_dir, "ReconsData", gauss_base_name)
                )
                savepathnew = str(
                    get_output_path(output_dir, "GeneratedData", gauss_base_name)
                )
                losspath = str(get_output_path(output_dir, "Loss", gauss_base_name))
                print("Gaussian head is added.")

            # if AE_head = True, for each pilot size, 2 iterative AE reconstruction will be conducted first
            # resulting in n_pilot * 4 samples, and the extended samples will be input to the model specified by modelname
            if AE_head:
                # Update paths for AE head augmentation
                ae_base_name = (
                    f"{dataname}_AEhead_{model}_{n_pilot}_Draw{rand_pilot}.csv"
                )
                savepath = str(get_output_path(output_dir, "ReconsData", ae_base_name))
                savepathnew = str(
                    get_output_path(output_dir, "GeneratedData", ae_base_name)
                )
                savepathextend = str(
                    get_output_path(output_dir, "ExtendData", ae_base_name)
                )
                losspath = str(get_output_path(output_dir, "Loss", ae_base_name))
                print("AE reconstruction head is added, reconstruction starting ...")
                feed_data, feed_labels = training_iter(
                    iter_times=AE_head_num,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
                    savepathextend=savepathextend,  # save path of the extended dataset
                    rawdata=rawdata,  # pilot data
                    rawlabels=rawlabels,  # pilot labels
                    random_seed=random_seed,
                    modelname="AE",  # choose from AE, VAE
                    num_epochs=1000,  # maximum number of epochs if early stop is not triggered, default value for AEhead is 1000
                    batch_size=round(
                        rawdata.shape[0] * 0.1
                    ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
                    learning_rate=0.0005,  # learning rate, default value for AEhead is 0.0005
                    early_stop=False,  # AEhead by default does not utilize early stopping rule
                    early_stop_num=30,  # won't take effect since early_stop == False
                    kl_weight=1,  # only take effect if model name is VAE, default value is 1
                    loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
                    replace=True,  # whether to replace the failure features in each reconstruction
                    saveextend=False,  # whether to save the extended dataset, if true, savepathextend must be provided
                    plot=False,
                )  # whether or not plot the heatmap of extended data

                rawdata = feed_data
                rawlabels = feed_labels
                print("Reconstruction finish, AE head is added.")
            # Training
            if "GAN" in modelname:
                log_dict = training_GANs(
                    savepathnew=savepathnew,  # path to save newly generated samples
                    rawdata=rawdata,  # raw data matrix with samples in row, features in column
                    rawlabels=rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
                    batch_size=round(
                        rawdata.shape[0] * batch_frac
                    ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
                    random_seed=random_seed,
                    modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
                    num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
                    learning_rate=learning_rate,
                    new_size=new_size,  # how many new samples you want to generate
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,  # save model for transfer learning, specify the path if want to save model
                    save_new=True,  # whether to save the newly generated samples
                    plot=False,
                )  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

                print("GAN model training for one pilot size one draw finished.")

                log_pd = pd.DataFrame(
                    {
                        "discriminator": log_dict["train_discriminator_loss_per_batch"],
                        "generator": log_dict["train_generator_loss_per_batch"],
                    }
                )
                # Directory is already created by get_output_path
                log_pd.to_csv(Path(losspath), index=False)

            elif "AE" in modelname:
                log_dict = training_AEs(
                    savepath=savepath,  # path to save reconstructed samples
                    savepathnew=savepathnew,  # path to save newly generated samples
                    rawdata=rawdata,  # raw data tensor with samples in row, features in column
                    rawlabels=rawlabels,  # abels for each sample, n_samples * 1, will not be used in AE or VAE
                    colnames=colnames,
                    batch_size=round(rawdata.shape[0] * batch_frac),  # batch size
                    random_seed=random_seed,
                    modelname=modelname,  # choose from "VAE", "AE"
                    num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
                    learning_rate=learning_rate,
                    kl_weight=kl_weight,  # only take effect if model name is VAE, default value is
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,  # save model for transfer learning, specify the path if want to save model
                    loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
                    save_recons=False,  # whether save reconstructed data, if True, savepath must be provided
                    new_size=new_size,  # how many new samples you want to generate
                    save_new=True,  # whether save new samples, if True, savepathnew must be provided
                    plot=False,
                )  # whether plot reconstructed samples' heatmap

                print("VAEs model training for one pilot size one draw finished.")
                log_pd = pd.DataFrame(
                    {
                        "kl": log_dict["train_kl_loss_per_batch"],
                        "recons": log_dict["train_reconstruction_loss_per_batch"],
                    }
                )
                # Directory is already created by get_output_path
                log_pd.to_csv(Path(losspath), index=False)
            elif "maf" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )
            elif "realnvp" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )

            elif "glow" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )

            else:
                print("wait for other models")


# %% Define application of experiment
def ApplyExperiment(
    path: Optional[Union[str, Path]] = None,
    dataname: str = "",
    apply_log: bool = True,
    new_size: Union[int, List[int]] = 500,
    model: str = "VAE1-10",
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    epoch: Optional[int] = None,
    val_ratio: float = 0.2,
    early_stop_num: Optional[int] = None,
    off_aug: Optional[str] = None,
    AE_head_num: int = 2,
    Gaussian_head_num: int = 9,
    pre_model: Optional[str] = None,
    save_model: Optional[str] = None,
    use_scheduler: bool = False,
    step_size: int = 10,
    gamma: float = 0.5,
    cap: bool = False,
    random_seed: int = 123,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    r"""
    Train deep generative models and generate new samples.

    This function trains VAE, CVAE, GAN, WGAN, WGANGP, MAF, GLOW, or RealNVP
    given data, model parameters, and generates new samples of specified size.

    Parameters
    ----------
    path : str or None, default=None
        DEPRECATED: Use data_dir and output_dir instead.
        Legacy path for reading real data and saving new data.
    dataname : str
        Pure data name without .csv extension. E.g., "BRCASubtypeSel_train"
    apply_log : bool, default=True
        Whether to apply log2 transformation before training.
    new_size : int or list of int
        Number of generated samples. For CVAE, group sample size is new_size/2.
    model : str, default="VAE1-10"
        Name of the model to train.
    batch_frac : float, default=0.1
        Batch fraction (proportion of data per batch).
    learning_rate : float, default=0.0005
        Learning rate for training.
    epoch : int or None
        Number of epochs, or None for early stopping.
    val_ratio : float, default=0.2
        Ratio of validation set.
    early_stop_num : int or None
        Stop training if loss doesn't improve for this many epochs.
    off_aug : str or None
        Offline augmentation: "AE_head", "Gaussian_head", or None.
    AE_head_num : int, default=2
        Fold multiplier for AE head augmentation.
    Gaussian_head_num : int, default=9
        Fold multiplier for Gaussian head augmentation.
    pre_model : str or None
        Path to pre-trained model for transfer learning.
    save_model : str or None
        Path to save the trained model.
    use_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    step_size : int, default=10
        Step size for scheduler.
    gamma : float, default=0.5
        Gamma for scheduler.
    cap : bool, default=False
        Whether to cap new samples.
    random_seed : int, default=123
        Random seed for reproducibility.
    data_dir : str, Path, or None
        Directory to read input data from. If None, will attempt to load the dataset from the package's bundled data or from the current working directory.
    output_dir : str, Path, or None
        Directory to write output files (reconstructed data, generated samples, loss logs, etc.). If None, the current working directory is used.
    """
    # Handle path parameter for backward compatibility
    if output_dir is None and path is not None:
        output_dir = Path(path)
    elif output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path.cwd()

    if data_dir is None and path is not None:
        data_dir = Path(path)

    # Read in data
    if data_dir is not None:
        read_path = Path(data_dir) / f"{dataname}.csv"
    else:
        read_path = Path(f"{dataname}.csv")

    try:
        if read_path.exists():
            df = pd.read_csv(read_path, header=0)
        else:
            # Try loading from bundled data
            df = load_dataset(dataname, data_path=read_path)
    except FileNotFoundError:
        # Fallback for bundled datasets
        df = load_dataset(dataname)

    print(f"1. Read data: {dataname}")

    dat_pd = df
    data_pd = dat_pd.select_dtypes(include=np.number)
    if "groups" in data_pd.columns:
        data_pd = data_pd.drop(columns=["groups"])
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
    colnames = data_pd.columns
    if apply_log:
        oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]
    if "groups" in dat_pd.columns:
        groups = dat_pd["groups"]
    else:
        groups = None

    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    # get model name and kl_weight if modelname is some autoencoder
    if len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))

    rawdata = oridata
    rawlabels = orilabels

    # decide batch fraction in file name
    model = "batch" + str(batch_frac).replace(".", "") + "_" + model

    # decide epoch
    num_epochs = epoch
    if early_stop_num is not None:
        early_stop = True
        epoch_info = "early_stop"
        model = "epochES_" + model
    else:
        early_stop = False
        epoch_info = str(epoch)
        model = "epoch" + epoch_info + "_" + model

    # decide offline augmentation
    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Determine the training parameters are epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learing rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
    )

    if pre_model is not None:
        model = model + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    # hyperparameters
    # random_seed = 123

    # Build output paths using output_dir
    savepath = str(output_dir / f"{dataname}_{model}_recons.csv")
    savepathnew = str(output_dir / f"{dataname}_{model}_generated.csv")
    losspath = str(output_dir / f"{dataname}_{model}_loss.csv")
    ensure_dir(output_dir)

    if Gaussian_head:
        rawdata, rawlabels = Gaussian_aug(
            rawdata, rawlabels, multiplier=[Gaussian_head_num]
        )
        savepath = str(output_dir / f"{dataname}_Gaussianhead_{model}_recons.csv")
        savepathnew = str(output_dir / f"{dataname}_Gaussianhead_{model}_generated.csv")
        losspath = str(output_dir / f"{dataname}_Gaussianhead_{model}_loss.csv")
        print("Gaussian head is added.")

    if AE_head:
        savepathextend = str(output_dir / f"{dataname}_AEhead_{model}_extend.csv")
        savepath = str(output_dir / f"{dataname}_AEhead_{model}_recons.csv")
        savepathnew = str(output_dir / f"{dataname}_AEhead_{model}_generated.csv")
        losspath = str(output_dir / f"{dataname}_AEhead_{model}_loss.csv")
        print("AE reconstruction head is added, reconstruction starting ...")
        feed_data, feed_labels = training_iter(
            iter_times=AE_head_num,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
            savepathextend=savepathextend,  # save path of the extended dataset
            rawdata=rawdata,  # pilot data
            rawlabels=rawlabels,  # pilot labels
            random_seed=random_seed,
            modelname="AE",  # choose from AE, VAE
            num_epochs=1000,  # maximum number of epochs if early stop is not triggered, default value for AEhead is 1000
            batch_size=round(
                rawdata.shape[0] * 0.1
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            learning_rate=0.0005,  # learning rate, default value for AEhead is 0.0005
            early_stop=False,  # AEhead by default does not utilize early stopping rule
            early_stop_num=30,  # won't take effect since early_stop == False
            kl_weight=1,  # only take effect if model name is VAE, default value is 2
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            replace=True,  # whether to replace the failure features in each reconstruction
            saveextend=False,  # whether to save the extended dataset, if true, savepathextend must be provided
            plot=False,
        )  # whether or not plot the heatmap of extended data

        rawdata = feed_data
        rawlabels = feed_labels
        print("AEhead added.")

    print("3. Training starts ......")
    # Training
    if "GAN" in modelname:
        log_dict = training_GANs(
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data matrix with samples in row, features in column
            rawlabels=rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
            batch_size=round(
                rawdata.shape[0] * batch_frac
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            random_seed=random_seed,
            modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            new_size=new_size,  # how many new samples you want to generate
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            save_new=True,  # whether to save the newly generated samples
            plot=False,
        )  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

        print("GAN model training finished.")

        log_pd = pd.DataFrame(
            {
                "discriminator": log_dict["train_discriminator_loss_per_batch"],
                "generator": log_dict["train_generator_loss_per_batch"],
            }
        )
        # Directory is already ensured to exist
        log_pd.to_csv(Path(losspath), index=False)

    elif "AE" in modelname:
        log_dict = training_AEs(
            savepath=savepath,  # path to save reconstructed samples
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data tensor with samples in row, features in column
            rawlabels=rawlabels,  # abels for each sample, n_samples * 1, will not be used in AE or VAE
            colnames=colnames,  # colnames saved
            batch_size=round(rawdata.shape[0] * batch_frac),  # batch size
            random_seed=random_seed,
            modelname=modelname,  # choose from "VAE", "AE"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            val_ratio=val_ratio,  # validation set ratio
            kl_weight=kl_weight,  # only take effect if model name is VAE, default value is
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            cap=cap,  # whether capping the new samples
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            save_recons=False,  # whether save reconstructed data, if True, savepath must be provided
            new_size=new_size,  # how many new samples you want to generate
            save_new=True,  # whether save new samples, if True, savepathnew must be provided
            plot=False,
            use_scheduler=use_scheduler,
            step_size=step_size,
            gamma=gamma,
        )  # whether plot reconstructed samples' heatmap

        print("VAEs model training finished.")
        log_pd = pd.DataFrame(
            {
                "kl": log_dict["val_kl_loss_per_batch"],
                "recons": log_dict["val_reconstruction_loss_per_batch"],
            }
        )
        # Directory is already ensured to exist
        log_pd.to_csv(Path(losspath), index=False)
    elif "maf" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )
    elif "realnvp" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )

    elif "glow" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )

    else:
        print("wait for other models")


# %% Define transfer learing
def TransferExperiment(
    pilot_size: Optional[List[int]] = None,
    fromname: str = "",
    toname: str = "",
    fromsize: int = 500,
    model: str = "VAE1-10",
    new_size: int = 500,
    apply_log: bool = True,
    epoch: Optional[int] = None,
    batch_frac: float = 0.1,
    learning_rate: float = 0.0005,
    off_aug: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run transfer learning using deep generative models.

    This function trains VAE, CVAE, GAN, WGAN, WGANGP, MAF, GLOW, or RealNVP
    using transfer learning. The model is first trained on the pre-training
    dataset, then fine-tuned on the target dataset.

    Parameters
    ----------
    pilot_size : list of int or None
        If None, uses ApplyExperiment for fine-tuning and new_size takes effect.
        Otherwise, uses PilotExperiment with the specified pilot sizes.
    fromname : str
        Name of the pre-training dataset (without .csv extension).
    toname : str
        Name of the fine-tuning dataset (without .csv extension).
    fromsize : int, default=500
        Number of samples to generate when pre-training.
    new_size : int, default=500
        Sample size for generated samples in ApplyExperiment mode.
    apply_log : bool, default=True
        Whether to apply log2 transformation before training.
    model : str, default="VAE1-10"
        Name of the model to train.
    batch_frac : float, default=0.1
        Batch fraction.
    learning_rate : float, default=0.0005
        Learning rate.
    epoch : int or None
        Number of epochs, or None for early stopping.
    off_aug : str or None
        Offline augmentation: "AE_head", "Gaussian_head", or None.
    data_dir : str, Path, or None
        Directory to read input data from. If None, will attempt to load the dataset from the package's bundled data or from the current working directory.
    output_dir : str, Path, or None
        Directory to write output files (reconstructed data, generated samples, loss logs, etc.). If None, the current working directory is used.
    """
    # Set up directories
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path.cwd()

    if data_dir is not None:
        data_dir = Path(data_dir)

    # Create transfer subdirectory for models
    transfer_dir = output_dir / "Transfer"
    ensure_dir(transfer_dir)

    save_model_path = str(transfer_dir / f"{toname}_from{fromname}_{model}.pt")

    ApplyExperiment(
        dataname=fromname,
        apply_log=apply_log,
        new_size=[fromsize],
        model=model,
        batch_frac=batch_frac,
        learning_rate=learning_rate,
        epoch=epoch,
        early_stop_num=30,
        off_aug=off_aug,
        AE_head_num=2,
        Gaussian_head_num=9,
        pre_model=None,
        save_model=save_model_path,
        data_dir=data_dir,
        output_dir=transfer_dir,
    )

    # training toname using pre-model
    pre_model_path = save_model_path
    if pilot_size is not None:
        PilotExperiment(
            dataname=toname,
            pilot_size=pilot_size,
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            pre_model=pre_model_path,
            epoch=epoch,
            off_aug=off_aug,
            early_stop_num=30,
            AE_head_num=2,
            Gaussian_head_num=9,
            data_dir=data_dir,
            output_dir=output_dir,
        )
    else:
        ApplyExperiment(
            dataname=toname,
            apply_log=apply_log,
            new_size=[new_size],
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_num=30,
            off_aug=off_aug,
            AE_head_num=2,
            Gaussian_head_num=9,
            pre_model=pre_model_path,
            save_model=None,
            data_dir=data_dir,
            output_dir=transfer_dir,
        )
