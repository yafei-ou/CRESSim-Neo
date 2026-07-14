import os
from pathlib import Path


project = "CRESSim-Neo"
author = "CRESSim-Neo contributors"

extensions = [
    "breathe",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = []

default_doxygen_xml = Path(__file__).resolve().parents[2] / "build/docs/doxygen/xml"
breathe_projects = {
    "CRESSim-Neo": os.environ.get(
        "CRESSIM_NEO_DOXYGEN_XML", str(default_doxygen_xml)
    ),
}
breathe_default_project = "CRESSim-Neo"

html_theme = "pydata_sphinx_theme"
html_static_path = []
