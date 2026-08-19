# Documentation development

The documentation website is built with Sphinx. Written pages may use either
reStructuredText (`.rst`) or MyST Markdown (`.md`). The C++ API reference is
parsed by Doxygen and rendered as part of the Sphinx website by Breathe.

## Arch Linux prerequisites

Install the documentation toolchain from the official repositories:

```bash
sudo pacman -S doxygen python-sphinx python-breathe python-myst-parser \
  python-sphinx-design python-pydata-sphinx-theme
```

Verify the installation:

```bash
doxygen --version
sphinx-build --version
python -c "import breathe, myst_parser, sphinx_design; print('Documentation extensions available')"
```

Graphviz is not currently required. It may be added later if the documentation
enables generated inheritance or collaboration diagrams.

## Guided build

Build the site with its dedicated Release Python API profile:

```bash
scripts/build_docs.sh
```

`build_docs.sh` owns `build/linux-docs-api`, which enables Python and viewer bindings
while disabling examples, tests, CUDA interop, and ultrasound. It builds that
package before configuring `build/docs-site`. This does not change any native
development build directory.

The generated site starts at `build/docs-site/html/index.html`. To serve it
locally:

```bash
python -m http.server --directory build/docs-site/html 8000
```

## Manual CMake workflow

Use these commands when automating the documentation build or changing its
profile. First configure and build the dedicated API package:

```bash
cmake -S . -B build/linux-docs-api -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_STATIC=OFF \
  -DCRESSIM_NEO_BUILD_PYTHON=ON \
  -DCRESSIM_NEO_BUILD_VIEWER=ON \
  -DCRESSIM_NEO_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DCRESSIM_NEO_ENABLE_CLANG_TIDY=OFF \
  -DCRESSIM_NEO_ENABLE_CUDA_INTEROP=OFF \
  -DCRESSIM_NEO_ENABLE_ULTRASOUND=OFF
cmake --build build/linux-docs-api --target \
  cressim_neo_python cressim_neo_python_package_files
```

Then configure the standalone documentation project. Using `docs/` as the
CMake source keeps site generation independent from engine build targets:

```bash
cmake -S docs -B build/docs-site \
  -DCRESSIM_NEO_PYTHON_PACKAGE_DIR="$PWD/build/linux-docs-api/bin"
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

All generated documentation remains under the ignored `build/` directory and
must not be committed.
