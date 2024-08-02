# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FOURIER-Net'
copyright = '2024, AERS + NRM'
author = 'AERS + NRM'
release = '20240614'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys 

sys.path.insert(0,os.path.abspath(".."))
sys.path.insert(0,os.path.abspath("../utils"))
sys.path.insert(0,os.path.abspath("../utils/models"))
sys.path.insert(0,os.path.abspath("../utils/models/complexCNNs"))
sys.path.insert(0,os.path.abspath("../utils/Trainers"))
sys.path.insert(0,os.path.abspath("../utils/myDatasets"))

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'  # Include both class-level and __init__ docstrings

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'haiku'
html_static_path = ['_static']
