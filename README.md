# SyNG-BTS: Synthesis of Next Generation Bulk Transcriptomic Sequencing

<!-- [![PyPI version](https://badge.fury.io/py/syng-bts.svg)](https://badge.fury.io/py/syng-bts) -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- [![Documentation Status](https://readthedocs.org/projects/syng-bts/badge/?version=latest)](https://syng-bts.readthedocs.io/en/latest/?badge=latest) -->

**SyNG-BTS** is a Python package for data augmentation of bulk transcriptomic sequencing data using deep generative models. 
It synthesizes realistic transcriptomic data without relying on predefined formulas, enabling researchers to augment small pilot datasets for more robust machine learning analyses. 
SyNG-BTS supports three generative model families: variational autoencoders (VAE), generative adversarial networks (GAN), and flow-based models. 
These models are trained on a pilot dataset and can synthesize additional samples at any desired scale.

<p align="center">
  <img src="./figs/workflow.png" width="800" alt="SyNG-BTS Workflow" />
</p>

## Features

- **Multiple Generative Models**: Variational Auto-Encoder (VAE), Generative Adversarial Network (GAN), and flow-based models
- **Flexible Experimentation**: Run pilot experiments, apply to case studies, or perform transfer learning
- **Built-in Evaluation**: UMAP visualization and heatmap evaluation tools
- **Bundled Datasets**: Example datasets included for immediate experimentation
- **Easy Integration**: Simple API for data augmentation workflows

## Installation

### From PyPI (Recommended)

*TODO: This is not yet possible — the package has not been uploaded to PyPI yet. Use the "From Source" installation below until a PyPI release is available.*

```bash
# available after PyPI release
pip install syng-bts
```

### From Source

```bash
git clone https://github.com/LXQin/SyNG-BTS.git
cd SyNG-BTS
pip install -e .
```

### Optional Dependencies

For documentation building:
```bash
pip install syng-bts[docs]
```

For development (testing, linting):
```bash
pip install syng-bts[dev]
```

## Quick Start

### Basic Usage

```python
from syng_bts import PilotExperiment, load_dataset

# Load example data
data = load_dataset("SKCMPositive_4")
print(f"Dataset shape: {data.shape}")

# Run a pilot experiment with VAE
PilotExperiment(
  dataname="SKCMPositive_4",   # dataset name (without .csv)
  pilot_size=[100],            # list of pilot sizes to draw from original data
  model="VAE1-10",             # model name (autoencoder with kl-weight encoded)
  batch_frac=0.1,              # batch fraction (proportion per batch)
  learning_rate=0.0005,        # learning rate
  epoch=None,                  # None => use early stopping; otherwise specify int
  early_stop_num=30,           # stop if loss doesn't improve for this many epochs
  off_aug=None,                # offline augmentation: 'AE_head', 'Gaussian_head', or None
  AE_head_num=2,               # folds for AE_head augmentation (if used)
  Gaussian_head_num=9,         # folds for Gaussian_head augmentation (if used)
  pre_model=None               # path to pre-trained model for transfer learning (optional)
)
```

### Apply to Case Study

```python
from syng_bts import ApplyExperiment

# Apply model to generate new samples from a case study dataset
ApplyExperiment(
    dataname="BRCASubtypeSel",     # dataset name (without .csv)
    apply_log=True,                # apply log2 transform before training
    new_size=[1000],               # number of generated samples (int or list)
    model="WGANGP",                # model name (e.g., VAE1-10, WGANGP, maf)
    batch_frac=0.1,                # batch fraction (proportion per batch)
    learning_rate=0.0005,          # learning rate
    epoch=10,                      # number of epochs (None => use early stopping)
    early_stop_num=30,             # stop if loss doesn't improve for this many epochs
    off_aug=None,                  # offline augmentation: 'AE_head', 'Gaussian_head', or None
    AE_head_num=2,                 # folds for AE_head augmentation (if used)
    Gaussian_head_num=9,           # folds for Gaussian_head augmentation (if used)
    pre_model=None,                # path to pre-trained model for transfer learning
    save_model=None,               # path to save the trained model (optional)
)
```

### Transfer Learning

```python
from syng_bts import TransferExperiment

# Transfer learning from one dataset to another
TransferExperiment(
  pilot_size=None,          # list of pilot sizes; if None, uses ApplyExperiment for fine-tuning
  fromname="PRAD",          # pre-training dataset name (without .csv)
  toname="BRCA",            # target dataset name (without .csv)
  fromsize=551,             # number of samples to generate for pre-training
  new_size=500,             # sample size for generated samples during fine-tuning/ApplyExperiment
  apply_log=True,           # apply log2 transform before training
  model="maf",              # model name (e.g., VAE1-10, WGANGP, maf)
  epoch=10,                 # number of epochs (None => use early stopping)
  batch_frac=0.1,           # batch fraction (proportion per batch)
  learning_rate=0.0005,     # learning rate
  off_aug=None,             # offline augmentation: 'AE_head', 'Gaussian_head', or None
)
```

### Evaluate Generated Data

```python
import pandas as pd
import numpy as np
from syng_bts import load_dataset, UMAP_eval, heatmap_eval

# Load real data
real_data = load_dataset("SKCMPositive_4")
real_data_numeric = real_data.select_dtypes(include=[np.number])

# Simulate generated data (in practice, this comes from a trained model)
generated_data = real_data_numeric.copy() + np.random.normal(0, 0.1, real_data_numeric.shape)

# Example 1: Heatmap visualization comparing generated vs real data
heatmap_eval(
    dat_real=real_data_numeric,        # Original data (pd.DataFrame)
    dat_generated=generated_data,      # Generated data (pd.DataFrame, optional)
    save=False                         # If True, returns figure instead of displaying
)

# Example 2: UMAP projection with optional group labels
groups_real = pd.Series(['Group A', 'Group B'] * (len(real_data_numeric) // 2))
groups_generated = pd.Series(['Group A', 'Group B'] * (len(generated_data) // 2))

UMAP_eval(
    dat_generated=generated_data,      # Generated data (pd.DataFrame or None)
    dat_real=real_data_numeric,        # Original data (pd.DataFrame)
    groups_generated=groups_generated, # Group labels for generated (pd.Series or None)
    groups_real=groups_real,           # Group labels for real (pd.Series or None)
    random_state=42,                   # Random seed for reproducibility
    legend_pos="best"                  # Legend position ("best", "upper right", etc.)
)
```

### List Available Datasets

```python
from syng_bts import list_bundled_datasets

# Show all bundled datasets
datasets = list_bundled_datasets()
print(datasets)
# ['SKCMPositive_4', 'BRCA', 'PRAD', 'BRCASubtypeSel', ...]
```

## Available Models

| Model | Description |
|-------|-------------|
| `VAE1-10` | Variational Auto-Encoder with 1:10 loss ratio |
| `CVAE1-10` | Conditional VAE with 1:10 loss ratio |
| `GAN` | Standard Generative Adversarial Network |
| `WGANGP` | Wasserstein GAN with Gradient Penalty |
| `maf` | Masked Autoregressive Flow |

## Dependencies

SyNG-BTS requires Python 3.8+ and the following packages:

- torch (>=1.3.1)
- pandas (>=1.0.5)
- numpy (>=1.19.1)
- scipy (>=1.4.1)
- matplotlib (>=2.2.3)
- seaborn (>=0.9.0)
- tqdm (>=4.26.0)
- tensorboardX (>=2.5.0)
- umap-learn (>=0.5.6)

## Documentation

Full documentation is available at [syng-bts.readthedocs.io](https://syng-bts.readthedocs.io/).

- [Installation Guide](https://syng-bts.readthedocs.io/en/latest/usage.html)
- [API Reference](https://syng-bts.readthedocs.io/en/latest/methods.html)
- [Examples & Tutorials](https://syng-bts.readthedocs.io/en/latest/evals.html)

## Development

### Quick Setup

```bash
git clone https://github.com/LXQin/SyNG-BTS.git
cd SyNG-BTS
make init-dev  # Install package + dev dependencies
```

### Makefile Commands

The project includes a Makefile for common development tasks:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install package in editable mode |
| `make install-dev` | Install development dependencies |
| `make init-dev` | Full dev setup (install + dev deps in venv) |
| `make test` | Run tests with pytest |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |
| `make check` | Run lint + tests |
| `make docs` | Build documentation |
| `make clean` | Remove build artifacts |

## Citation

If you use SyNG-BTS in your research, please cite:

> Qi Y, Wang X, Qin LX. Optimizing sample size for supervised machine learning with bulk transcriptomic sequencing: a learning curve approach. Brief Bioinform. 2025 Mar 4;26(2):bbaf097. doi: 10.1093/bib/bbaf097. PMID: 40072846; PMCID: PMC11899567.

**BibTeX:**
```bibtex
@article{qin2025optimizing,
  title = {Optimizing sample size for supervised machine learning with bulk transcriptomic sequencing: a learning curve approach},
  author = {Qi, Yunhui and Wang, Xinyi and Qin, Li-Xuan},
  journal = {Brief Bioinformatics},
  year = {2025},
  volume = {26},
  number = {2},
  pages = {bbaf097},
  doi = {10.1093/bib/bbaf097},
  url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC11899567/}
}
```

## License

SyNG-BTS is licensed under the [GNU Affero General Public License v3.0](LICENSE).

## Acknowledgments

This package was developed at Memorial Sloan Kettering Cancer Center. We thank Sebastian Raschka for the [STAT 453 course materials](https://sebastianraschka.com/teaching/stat453-ss2021/) that provided foundational concepts for the deep generative models.

