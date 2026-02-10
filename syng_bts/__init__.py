"""
SyNG-BTS: Synthesis of Next Generation Bulk Transcriptomic Sequencing

A data augmentation tool for synthesizing transcriptomics data with realistic
distributions using deep generative models (VAE, GAN, Flow-based models).

Example usage:
    from syng_bts import PilotExperiment, ApplyExperiment

    # Run a pilot experiment
    PilotExperiment(
        dataname="SKCMPositive_4",
        pilot_size=[100],
        model="VAE1-10",
        batch_frac=0.1,
        learning_rate=0.0005,
        epoch=None,
        early_stop_num=30
    )

For more information, see:
- Documentation: https://syng-bts.readthedocs.io/
- GitHub: https://github.com/LXQin/SyNG-BTS
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11899567/
"""

# Dynamic version from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("syng-bts")
    except PackageNotFoundError:
        # Package is not installed (running from source)
        __version__ = "2.5.0"
except ImportError:
    # Python < 3.8 fallback
    __version__ = "2.5.0"

__author__ = "Li-Xuan Qin, Yunhui Qi, Xinyi Wang, Yannick Dueren"
__email__ = "qinl@mskcc.org"
__license__ = "AGPL-3.0"

# Import main experiment functions
from .experiments import (
    PilotExperiment,
    ApplyExperiment,
    TransferExperiment,
)

# Create alias for backward compatibility
Transfer = TransferExperiment

# Import evaluation functions
from .evaluations import (
    heatmap_eval,
    UMAP_eval,
    evaluation,
)

# Import models (for advanced users who want to use models directly)
from .helper_models import (
    AE,
    VAE,
    CVAE,
    GAN,
)

# Import data utilities
from .data_utils import (
    load_dataset,
    list_bundled_datasets,
    set_default_output_dir,
    get_output_dir,
    resolve_data,
    derive_dataname,
)

# Import result objects
from .result import SyngResult, PilotResult

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Main experiment functions
    "PilotExperiment",
    "ApplyExperiment",
    "TransferExperiment",
    "Transfer",  # Alias for TransferExperiment
    # Evaluation functions
    "heatmap_eval",
    "UMAP_eval",
    "evaluation",
    # Model classes (for advanced usage)
    "AE",
    "VAE",
    "CVAE",
    "GAN",
    # Data utilities
    "load_dataset",
    "list_bundled_datasets",
    "set_default_output_dir",
    "get_output_dir",
    "resolve_data",
    "derive_dataname",
    # Result objects
    "SyngResult",
    "PilotResult",
]
