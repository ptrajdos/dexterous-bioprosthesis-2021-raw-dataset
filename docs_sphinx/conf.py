"""Sphinx configuration for Dexterous Bioprosthesis 2021 Raw Datasets."""

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))

from dexterous_bioprosthesis_2021_raw_datasets import __version__

# -- Project information ------------------------------------------------------
project = 'Dexterous Bioprosthesis 2021 Raw Datasets'
author = 'Pawel Trajdos'
copyright = '2024, Pawel Trajdos'
version = __version__
release = __version__

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx_pyreverse',
    'myst_parser',
    'sphinxcontrib.mermaid',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'index.html']

# -- Options for autodoc ------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

# -- Options for Napoleon (Google/NumPy docstrings) ---------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output --------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['uml_zoom.css']

# -- Options for Mermaid ------------------------------------------------------
mermaid_version = 'latest'
mermaid_init_js = 'mermaid.initialize({startOnLoad:true});'
mermaid_cmd = 'mmdc'
mermaid_output_format = 'png'
mermaid_params = ['-w', '2000', '--scale', '2']

# -- Options for LaTeX output -------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{pdflscape}
\usepackage{grffile}
\usepackage{adjustbox}
\usepackage[htt]{hyphenat}
\sloppy
''',
}

latex_documents = [
    ('index', 'DexterousBioprosthesis2021RawDatasets.tex',
     project, author, 'manual'),
]

# -- Options for Graphviz -----------------------------------------------------
graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir='LR', size='"12.0, 8.0"')
