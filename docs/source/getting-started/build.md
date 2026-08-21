# Build CRESSim-Neo

CRESSim-Neo is a C++17 simulation engine with optional Python bindings. A
normal CMake build produces the native libraries, and the Python build adds the
`cressim_neo` extension module.

## Prerequisites

- CMake 3.18 or newer
- Clang and Clang++ with C++17 support; the top-level CMake project selects
  Clang explicitly
- A C++ build tool; Ninja is recommended
- Python 3.10 or newer, plus its development headers, when building Python
  bindings
- A Vulkan-capable graphics driver and runtime for Vulkan-backed programs

On Ubuntu or Debian, install system dependencies with `apt`:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  clang \
  cmake \
  ninja-build \
  python3-dev \
  python3-venv \
  python3-pip \
  git \
  libvulkan-dev \
  vulkan-tools
```

By default, configuration downloads the pinned DXC runtime release with a
SHA-256 check. CMake therefore needs network access the first time it populates
its build-directory cache.

Initialize the repository submodules before configuring a build:

```bash
git submodule update --init --recursive
```

In particular, Python-enabled CMake builds require the pinned
`extern/pybind11` submodule. The current CMake configuration does not fall back
to an installed or downloaded copy of pybind11.

For PyTorch CUDA interoperability, use the same Python interpreter environment
for PyTorch and CRESSim-Neo, and build CRESSim-Neo with a CUDA toolkit compatible
with that PyTorch installation. A virtual environment is optional; it does not
by itself select or unify the CUDA runtime libraries used by the two projects.
The current CUDA/Torch workflow is validated on Arch Linux with system Python;
other distributions should be treated as unvalidated until tested.

## Native development

### Guided setup

The guided setup script currently supports Linux only. On other platforms, use
the {ref}`manual-cmake-configuration` instructions below.

For the normal shared-SDK workflow, configure a Release or Debug build with:

```bash
scripts/configure_builds.sh
```

The helper selects supported project features and safely preserves the generator
of existing build directories. New directories can use Ninja or Unix Makefiles.
It prints the commands for the configured build. Select Python bindings when
the same CMake build should produce the Python component.

Build and install with standard CMake commands. For example, for a Release
build with all three install components enabled:

```bash
cmake --build build/linux-release --parallel
cmake --install build/linux-release --component CXXSDK
cmake --install build/linux-release --component Examples
cmake --install build/linux-release --component Python
```

The project defines three install components:

- `CXXSDK` installs the shared C++ libraries, public headers, standard models,
  environment maps, shaders, their authoring sources, and the CMake package.
  Assets are installed under `share/cressim-neo/assets`.
- `Examples` installs the enabled standalone C++ example executables. Install
  `CXXSDK` alongside it so their default asset paths resolve.
- `Python` installs the `cressim_neo` module, its native runtime libraries,
  Python sources, shaders, models, and environment maps into the configured
  CMake prefix's Python `site-packages` location. It is not a `pip` install.

Pass `--prefix "$HOME/.local"` to an install command when a custom prefix is
needed. Static C++ SDK installation is not currently a supported public
workflow.

All runtime assets resolve through one root: set `CRESSIM_NEO_ASSET_DIR` to use
a custom asset directory; otherwise C++ resolves the installed asset tree and
Python resolves its package-local `cressim_neo/assets` directory. Explicit
shader or model paths still take precedence where supported.

(manual-cmake-configuration)=
### Manual CMake configuration

Use this equivalent workflow when scripting, automating, or choosing CMake
options directly. Configure a Release build with Python bindings enabled:

```bash
cmake -S . -B build/linux-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRESSIM_NEO_BUILD_PYTHON=ON
cmake --build build/linux-release --parallel
```

Add `-G Ninja` to the configure command to use Ninja.

For a smaller headless build, disable the default viewer and examples:

```bash
cmake -S . -B build/linux-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRESSIM_NEO_BUILD_PYTHON=ON \
  -DCRESSIM_NEO_BUILD_VIEWER=OFF \
  -DCRESSIM_NEO_BUILD_EXAMPLES=OFF
```

Useful optional CMake switches are:

- `-DBUILD_TESTING=ON` — build the test tree.
- `-DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON` — run clang-tidy as part of C++ builds.
- `-DCRESSIM_NEO_ENABLE_CUDA_INTEROP=ON` — require and enable CUDAToolkit.
- `-DCRESSIM_NEO_ENABLE_ULTRASOUND=ON` — enable CRESSim-Ultrasound. This also
  requires CUDA interop and a working CUDA compiler.
- `-DCRESSIM_NEO_DXC_PROVIDER=SYSTEM` — let Diligent use DXC discovered from
  local SDK paths instead of downloading the pinned runtime. This is intended
  only for local development.
- `-DCRESSIM_NEO_DXC_PROVIDER=OFF` — do not provision DXC. Vulkan falls back
  from DXC-only features and the Windows D3D12 shader path is unsupported.

### Consuming an installed C++ SDK

After installing `CXXSDK` to a prefix, point a consumer project at it:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
```

Its `CMakeLists.txt` can link the engine as follows:

```cmake
find_package(CRESSimNeo CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE CRESSimNeo::engine)
```

## Python package development and wheels

`pyproject.toml` uses scikit-build-core to drive the same CMake project. Its
wheel configuration builds only the Python component, with viewer, examples,
tests, CUDA interoperability, and ultrasound disabled. This makes the standard
wheel independent of a CUDA toolchain.

Build a wheel:

```bash
scripts/build_wheel.sh
```

Or manually build with:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 python3 -m pip wheel --no-deps --wheel-dir dist .
```

To build a local wheel with interactive viewer bindings, use
`scripts/build_wheel.sh --viewer`. Do not publish that artifact beside a
headless wheel with the same package version and platform tag.

Install and verify the resulting wheel with the interpreter that will run the
application:

```bash
python3 -m pip install dist/cressim_neo-*.whl
python3 -c "import cressim_neo; print(cressim_neo.__file__)"
```

For an editable installation:

```bash
python3 -m pip install -e .
```

The current package requires NumPy. PyTorch and Gymnasium are optional:

```bash
python3 -m pip install 'cressim-neo[torch,gymnasium]'
```

Continue with {doc}`first-scene` after installation.
