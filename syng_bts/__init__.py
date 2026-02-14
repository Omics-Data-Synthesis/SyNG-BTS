"""
SyNG-BTS: Synthesis of Next Generation Bulk Transcriptomic Sequencing

A data augmentation tool for synthesizing transcriptomics data with realistic
distributions using deep generative models (VAE, GAN, Flow-based models).

Example usage:
    import pandas as pd
    from syng_bts import generate, pilot_study

    # Generate synthetic data
    result = generate(
        data="SKCMPositive_4",
        model="VAE1-10",
        new_size=500,
        batch_frac=0.1,
        learning_rate=0.0005,
    )
    result.generated  # pd.DataFrame of synthetic samples

    # Run a pilot study
    pilot = pilot_study(
        data="SKCMPositive_4",
        pilot_size=[50, 100],
        model="VAE1-10",
        batch_frac=0.1,
        learning_rate=0.0005,
    )

For more information, see:
- Documentation: https://syng-bts.readthedocs.io/
- GitHub: https://github.com/Omics-Data-Synthesis/SyNG-BTS
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11899567/
"""

# Dynamic version from package metadata (set in pyproject.toml)
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("syng-bts")
except PackageNotFoundError:
    # Package is not installed (running from source)
    __version__ = "3.1.0"

__author__ = "Li-Xuan Qin, Yunhui Qi, Xinyi Wang, Yannick Dueren"
__email__ = "qinl@mskcc.org"
__license__ = "AGPL-3.0"

# Import main experiment functions
# Import evaluation functions
# Import data utilities
from .data_utils import (
    derive_dataname,
    get_output_dir,
    list_bundled_datasets,
    resolve_data,
    set_default_output_dir,
)
from .evaluations import (
    UMAP_eval,
    evaluation,
    heatmap_eval,
)
from .experiments import (
    generate,
    pilot_study,
    transfer,
)

# Import models (for advanced users who want to use models directly)
from .helper_models import (
    AE,
    CVAE,
    GAN,
    VAE,
)

# Import result objects
from .result import PilotResult, SyngResult

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Experiment functions
    "generate",
    "pilot_study",
    "transfer",
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
    "list_bundled_datasets",
    "set_default_output_dir",
    "get_output_dir",
    "resolve_data",
    "derive_dataname",
    # Result objects
    "SyngResult",
    "PilotResult",
]
