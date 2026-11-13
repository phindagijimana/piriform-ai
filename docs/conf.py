"""Sphinx configuration for the piriform-ai documentation."""
from __future__ import annotations

project = "piriform-ai"
author = "Philbert Ndagijimana"
copyright = "2026, Philbert Ndagijimana"
release = "0.1"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.todo",
]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "piriform-ai"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

graphviz_output_format = "svg"
numfig = True
math_numfig = True
math_eqref_format = "Eq. {number}"
pygments_style = "sphinx"
