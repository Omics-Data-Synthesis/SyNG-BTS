# SyNG-BTS: Synthesis of Next Generation Bulk Transcriptomic Sequencing

<!-- [![PyPI version](https://badge.fury.io/py/syng-bts.svg)](https://badge.fury.io/py/syng-bts) -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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

- **Multiple Generative Models**: VAE, CVAE, GAN, WGANGP, and flow-based models (MAF)
- **Unified API**: Run pilot experiments, generate synthetic data, and perform transfer learning
- **DataFrame-First API**: Accept pandas DataFrames, CSV file paths, or bundled dataset names
- **Rich Result Objects**: `SyngResult` / `PilotResult` with built-in plotting and export
- **In-Memory Pipeline**: No disk I/O by default — results stay in memory until you choose to save
- **Built-in Evaluation**: Heatmap and UMAP visualization functions
- **Bundled Datasets**: Example TCGA datasets included for immediate experimentation

## Installation

### From PyPI (Recommended)

*TODO: This is not yet possible — the package has not been uploaded to PyPI yet. Use the "From Source" installation below until a PyPI release is available.*

```bash
# available after PyPI release
pip install syng-bts
```

### From Source

```bash
git clone https://github.com/Omics-Data-Synthesis/SyNG-BTS
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

### Generate Synthetic Data

```python
from syng_bts import generate

# Train a VAE on bundled data and generate 500 synthetic samples
result = generate(
    data="SKCMPositive_4",   # bundled dataset name, CSV path, or DataFrame
    model="VAE1-10",         # model specification (type + kl_weight)
    new_size=500,            # number of synthetic samples
    batch_frac=0.1,          # batch fraction
    learning_rate=0.0005,    # learning rate
)

# Access results in memory
print(result.generated_data.shape)   # (500, n_features)
print(result.loss.columns.tolist())  # ['kl', 'recons']
print(result.summary())

# Plot training loss (one figure per loss column)
figs = result.plot_loss()  # dict[str, Figure]

# Optionally save to disk
result.save("./my_output/")

# Load a previously saved result
from syng_bts import SyngResult
loaded = SyngResult.load("./my_output/")
```

### Run a Pilot Study

```python
from syng_bts import pilot_study

# Sweep over multiple pilot sizes (5 random draws each)
pilot = pilot_study(
    data="SKCMPositive_4",
    pilot_size=[50, 100],
    model="VAE1-10",
    batch_frac=0.1,
    learning_rate=0.0005,
)

# Access individual runs
run = pilot.runs[(50, 1)]  # (pilot_size, draw_index)
print(run.generated_data.head())

# Aggregate loss plot (one figure per loss column, all runs overlaid)
figs = pilot.plot_loss(aggregate=True)  # dict[str, Figure]
```

### Transfer Learning

```python
from syng_bts import transfer

# Pre-train on PRAD, fine-tune and generate on BRCA
result = transfer(
    source_data="PRAD",
    target_data="BRCA",
    source_size=551,
    new_size=500,
    model="maf",
    apply_log=True,
    epoch=10,
)

print(result.generated_data.shape)
result.save("./transfer_output/")
```

### Use DataFrame Input

```python
import pandas as pd
from syng_bts import generate

my_data = pd.read_csv("my_dataset.csv")
result = generate(
    data=my_data,
    name="my_dataset",     # used in output filenames
    model="WGANGP",
    new_size=1000,
    epoch=50,
)
```

### Evaluate Generated Data

```python
from syng_bts import generate, resolve_data, heatmap_eval, UMAP_eval

result = generate(data="SKCMPositive_4", model="VAE1-10", epoch=5)
real_data, _groups = resolve_data("SKCMPositive_4")

# Built-in heatmap on result object
fig = result.plot_heatmap()

# Standalone evaluation comparing real vs generated
fig_heatmap = heatmap_eval(real_data=real_data.head(50), generated_data=result.generated_data.head(50))
fig_umap = UMAP_eval(real_data=real_data, generated_data=result.generated_data, random_seed=42)
```

### List Available Datasets

```python
from syng_bts import list_bundled_datasets

print(list_bundled_datasets())
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

SyNG-BTS requires Python 3.10+ and the following packages:

- torch (>=2.0.0)
- pandas (>=1.5.0)
- numpy (>=1.23.0)
- scipy (>=1.9.0)
- matplotlib (>=3.6.0)
- seaborn (>=0.12.0)
- tensorboardX (>=2.6.0)
- umap-learn (>=0.5.6)
- pyarrow (>=14.0.0)

## Documentation

Full documentation is available at [syng-bts.readthedocs.io](https://syng-bts.readthedocs.io/).

- [Installation Guide](https://syng-bts.readthedocs.io/en/latest/usage.html)
- [API Reference](https://syng-bts.readthedocs.io/en/latest/methods.html)
- [Examples & Tutorials](https://syng-bts.readthedocs.io/en/latest/evals.html)

## Development

### Quick Setup

```bash
git clone https://github.com/Omics-Data-Synthesis/SyNG-BTS
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

