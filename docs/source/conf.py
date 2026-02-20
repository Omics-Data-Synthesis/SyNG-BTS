# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the package root to path for autodoc
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "SyNG-BTS"
copyright = "2024-2026, Li-Xuan Qin, Yunhui Qi, Xinyi Wang, Yannick Dueren"
author = "Li-Xuan Qin, Yunhui Qi, Xinyi Wang, Yannick Dueren"

# Get version from package
try:
    from syng_bts import __version__

    version = __version__
    release = __version__
except ImportError:
    version = "3.3.0"
    release = "3.3.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "nbsphinx",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Mock heavy dependencies for ReadTheDocs builds
# This prevents build failures when these libraries aren't available
autodoc_mock_imports = [
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "sklearn",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "sklearn.svm",
    "umap",
    "tensorboardX",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "xgboost",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autosummary settings
autosummary_generate = False  # We use manual autodoc directives

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix of source filenames
source_suffix = ".rst"

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "collapse_navigation": False,
}

# html_static_path = ["_static"]  # Uncomment if you add custom static files

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = "footnote"

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = "never"  # Don't execute notebooks during build
