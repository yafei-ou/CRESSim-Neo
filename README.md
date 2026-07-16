# CRESSim-Neo

CRESSim-Neo is a C++17 simulation engine with optional Python bindings.  The
engine uses the in-tree Diligent dependency; a normal CMake build produces the
native libraries, and the Python build adds the `cressim_neo` extension module.

This document describes the current developer workflow.  Documentation-site
instructions are in [`docs/README.md`](docs/README.md).

## Prerequisites

- CMake 3.18 or newer
- Clang and Clang++ with C++17 support (the top-level CMake project currently
  selects Clang explicitly)
- A C++ build tool; Ninja is recommended
- Python 3.10 or newer, plus its development headers, when building Python
  bindings
- A Vulkan-capable graphics driver/runtime for running Vulkan-backed programs

The repository checkout must include the `extern/DiligentEngine` dependency
tree.  For Python-enabled CMake builds, CMake first looks for a pybind11 CMake
package on its configured search paths (for example, one supplied through
`CMAKE_PREFIX_PATH`).  Otherwise it uses `extern/pybind11` when present, or
fetches pybind11 there during the first configure.

On Linux, create and activate a virtual environment before working with the
Python bindings:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## CMake development build

Configure a Release build with the Python bindings enabled:

```bash
cmake -S . -B build/linux-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRESSIM_NEO_BUILD_PYTHON=ON
cmake --build build/linux-release --parallel
```

This build directory is persistent and incremental.  Re-run the second command
after changing source files; CMake/Ninja rebuild only the affected targets. To
limit parallelism:

```bash
cmake --build build/linux-release --parallel 4
```

For a smaller headless build, disable the default viewer and examples:

```bash
cmake -S . -B build/linux-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRESSIM_NEO_BUILD_PYTHON=ON \
  -DCRESSIM_NEO_BUILD_VIEWER=OFF \
  -DCRESSIM_NEO_BUILD_EXAMPLES=OFF
```

Useful optional CMake switches are:

- `-DBUILD_TESTING=ON` — build the test tree.
- `-DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON` — run clang-tidy as part of C++ builds.
- `-DCRESSIM_NEO_ENABLE_CUDA_INTEROP=ON` — require and enable CUDAToolkit.
- `-DCRESSIM_NEO_ENABLE_ULTRASOUND=ON` — enable CRESSim-Ultrasound.  This also
  requires CUDA interop and a working CUDA compiler.
- `-DENGINE_STATIC=ON` — build static engine libraries.  The installable C++
  SDK described below is currently provided by the default shared-library
  build.

## Installing from a CMake build

The project defines two install components:

- `CXXSDK` installs the shared C++ libraries, public headers (including the
  Diligent headers exposed by the public API), shaders, and the CMake package.
- `Python` installs the `cressim_neo` module, its native runtime libraries,
  Python sources, and shaders.

Install both components to a local prefix:

```bash
cmake --install build/linux-release --prefix "$HOME/.local"
```

Or install only one component:

```bash
cmake --install build/linux-release --component CXXSDK --prefix "$HOME/.local"
cmake --install build/linux-release --component Python --prefix "$HOME/.local"
```

The C++ package is found from that prefix by consumers that configure with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
```

Their `CMakeLists.txt` can then link the engine as follows:

```cmake
find_package(CRESSimNeo CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE CRESSimNeo::engine)
```

The Python component is installed into the target interpreter's `site-packages`
location below the selected prefix.  For routine Python use, prefer installing
a wheel into a virtual environment, as shown below.

## Python package development and wheels

`pyproject.toml` uses scikit-build-core to drive the same CMake project.  Its
wheel configuration builds only the Python component, with viewer, examples,
tests, CUDA interop, and ultrasound disabled.  This makes the initial public
wheel baseline independent of a CUDA toolchain.

Build a wheel:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

Install and verify the resulting wheel:

```bash
python -m pip install dist/cressim_neo-*.whl
python -c "import cressim_neo; print(cressim_neo.__file__)"
```

For an editable installation:

```bash
python -m pip install -e .
```

Wheel and editable installation builds compile the native C++ engine and can
therefore take several minutes on a clean build.  Scikit-build-core currently
uses a temporary build directory for these package builds, so use the persistent
CMake workflow above for frequent C++ iteration.  Limit package-build
parallelism when needed:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 python -m pip wheel --no-deps --wheel-dir dist .
```

The current package requires NumPy.  PyTorch and Gymnasium are optional:

```bash
python -m pip install 'cressim-neo[torch,gymnasium]'
```

## Documentation

Build the developer documentation site after building a Python package:

```bash
scripts/configure_builds.sh
```

See [`docs/README.md`](docs/README.md) for the full documentation prerequisites
and standalone build commands.
