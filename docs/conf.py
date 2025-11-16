# Configuration file for the Sphinx documentation builder

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Jericho Mk II'
copyright = '2025, Lancaster University'
author = 'Josh Wiggs, Chris Arridge, George Greenyer'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    # 'breathe',  # For C++/CUDA documentation (requires doxygen XML)
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# -- Breathe configuration ---------------------------------------------------
# Uncomment when Doxygen XML is available
# breathe_projects = {"jericho_mkII": "../build/xml"}
# breathe_default_project = "jericho_mkII"

# -- Math configuration ------------------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
