import os
import sys
from pathlib import Path

from docutils import nodes
from sphinx import addnodes


project = "CRESSim-Neo"
author = "CRESSim-Neo contributors"
html_title = "CRESSim-Neo Documentation"

# CI sets this to ``main`` for the development site or to the release tag for
# an immutable release build.  Keeping it independent from the compiled
# package version makes it possible to build docs from a tag before a package
# release is published.
docs_version = os.environ.get("CRESSIM_NEO_DOCS_VERSION", "main")
docs_base_url = os.environ.get(
    "CRESSIM_NEO_DOCS_BASE_URL", "https://yafei-ou.github.io/CRESSim-Neo"
).rstrip("/")
release = docs_version
version = docs_version

extensions = [
    "breathe",
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = ["colon_fence"]

python_package_dir = os.environ.get("CRESSIM_NEO_PYTHON_PACKAGE_DIR")
if not python_package_dir:
    raise RuntimeError(
        "CRESSIM_NEO_PYTHON_PACKAGE_DIR must name the directory containing "
        "the built cressim_neo package"
    )
sys.path.insert(0, python_package_dir)

root_doc = "index"
exclude_patterns = []

default_doxygen_xml = Path(__file__).resolve().parents[2] / "build/docs/doxygen/xml"
breathe_projects = {
    "CRESSim-Neo": os.environ.get(
        "CRESSIM_NEO_DOXYGEN_XML", str(default_doxygen_xml)
    ),
}
breathe_default_project = "CRESSim-Neo"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "CRESSim-Neo Documentation",
    },
    "switcher": {
        # This file is published at the Pages-site root, rather than copied
        # into each version directory.  Consequently, old static releases
        # automatically learn about newer releases.
        "json_url": f"{docs_base_url}/versions.json",
        "version_match": docs_version,
    },
    # A developer's local build has no Pages manifest to fetch.  The browser
    # still loads it in deployed docs, while disabling the build-time probe
    # keeps ``sphinx-build --fail-on-warning`` usable locally and in release CI.
    "check_switcher": False,
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "index": [],
    },
}
html_static_path = []
templates_path = ["_templates"]
html_sidebars = {
    "**": ["components/sidebar-nav-bs.html"],
    "index": [],
}


def _unlink_python_none_references(app, doctree):
    """Keep Python ``None`` annotations from linking to pybind11 enum members."""
    for node in list(doctree.findall(addnodes.pending_xref)):
        if node.get("refdomain") == "py" and node.get("reftarget") == "None":
            node.replace_self(nodes.literal(text=node.astext()))


def setup(app):
    app.connect("doctree-read", _unlink_python_none_references)
