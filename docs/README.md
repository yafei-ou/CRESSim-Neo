# Documentation development

The documentation website is built with Sphinx. Written pages may use either
reStructuredText (`.rst`) or MyST Markdown (`.md`). The C++ API reference is
parsed by Doxygen and rendered as part of the Sphinx website by Breathe.

## Arch Linux prerequisites

Install the documentation toolchain from the official repositories:

```bash
sudo pacman -S doxygen python-sphinx python-breathe python-myst-parser \
  python-pydata-sphinx-theme
```

Verify the installation:

```bash
doxygen --version
sphinx-build --version
python -c "import breathe, myst_parser; print('Breathe and MyST available')"
```

Graphviz is not currently required. It may be added later if the documentation
enables generated inheritance or collaboration diagrams.

## Build the website

Configure the standalone documentation build from the repository root. Using
`docs/` as the CMake source keeps documentation setup independent from engine
dependencies and does not compile the engine, examples, or tests:

```bash
cmake -S docs -B build/docs-site \
  -DCRESSIM_NEO_PYTHON_PACKAGE_DIR="$PWD/build/linux-debug/bin"
```

The package directory must contain a `cressim_neo` package built for the active
system Python. Sphinx imports it to obtain native extension signatures and
docstrings.

Build the complete HTML website:

```bash
cmake --build build/docs-site --target docs-html
```

The `docs-html` target first runs the `docs-doxygen` target. Doxygen reads only
the public headers under `include/` and writes XML inside the selected CMake
build directory. Sphinx and Breathe then consume that XML. Doxygen does not
produce a separate HTML website. The generated C++ reference is organized by
the `common`, `engine`, `physics`, `gpu`, `graphics`, and `viewer` namespaces.
The Python API reference is generated in the same site from the compiled
package.

The generated site starts at `build/docs-site/html/index.html`. All generated
documentation remains under the ignored `build/` directory and must not be
committed.

To serve the generated website locally:

```bash
python -m http.server --directory build/docs-site/html 8000
```

Then open <http://localhost:8000/>.
