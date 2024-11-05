# -- Project information -----------------------------------------------------

project = 'JAMS'
copyright = '2019, Joseph Barker'
author = 'Joseph Barker'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax'
]

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

