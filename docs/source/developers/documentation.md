# Building the Documentation Site

The documentation site uses Sphinx with MyST Markdown and reStructuredText.
Doxygen supplies the C++ API XML through Breathe, and Sphinx imports the built
Python package to generate the Python reference.

On Arch Linux, install the documented toolchain:

```bash
sudo pacman -S doxygen python-sphinx python-breathe python-myst-parser \
  python-sphinx-design python-pydata-sphinx-theme
```

Build the site through its dedicated API profile:

```bash
scripts/build_docs.sh
```

The script configures and builds the `linux-docs-api` preset, then writes the
site to `build/docs-site/html/index.html`. It uses a separate build directory,
so it does not alter a native development build.

For manual automation, configure the API package first, then point the
standalone documentation project at its package directory:

```bash
cmake --preset linux-docs-api
cmake --build --preset linux-docs-api --target \
  cressim_neo_python cressim_neo_python_package_files
cmake -S docs -B build/docs-site \
  -DCRESSIM_NEO_PYTHON_PACKAGE_DIR="$PWD/build/linux-docs-api/bin"
cmake --build build/docs-site --target docs-html
```

`docs-html` runs Doxygen before Sphinx. Generated output stays below `build/`
and must not be committed. The repository-facing details are maintained in
`docs/README.md`.
