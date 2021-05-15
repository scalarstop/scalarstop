# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "ScalarStop"
copyright = "2021, Neocrym Records Inc."
author = "Neocrym Records Inc."


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

autodoc_inherit_docstrings = True
add_module_names = False

# AutoAPI configuration
autoapi_type = "python"
autoapi_dirs = ["../scalarstop"]
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"

# This is a mapping to Sphinx documentation sites elsewhere on the Internet.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "./_intersphinx-mappings/python.inv"),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "./_intersphinx-mappings/tensorflow.inv",
    ),
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/dev",
        "./_intersphinx-mappings/pandas.inv",
    ),
    "sqlalchemy": (
        "https://docs.sqlalchemy.org/en/14/",
        "./_intersphinx-mappings/sqlalchemy.inv",
    ),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates" "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "neocrym_sphinx_theme"
html_show_sphinx = False
html_copy_source = False
html_show_source = False
html_theme_options = dict(
    light_logo="global/images/scalarstop/v1/1x/scalarstop-wordmark-color-black-on-transparent--1x.png",
    dark_logo="global/images/scalarstop/v1/1x/scalarstop-wordmark-color-white-on-transparent--1x.png",
    sidebar_hide_name=True,
)
html_css_files = ["css/custom.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

pygments_style = "colorful"
pygments_dark_style = "fruity"
