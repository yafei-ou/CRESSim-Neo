# Build CRESSim-Neo

CRESSim-Neo is a C++17 simulation engine with optional Python bindings.
Third-party C++ dependencies (including Diligent Engine, pybind11, and GLFW)
are managed as in-tree git submodules and built automatically by CMake. A
normal CMake build produces the native libraries; enabling Python adds the
`cressim_neo` extension module.

## Prerequisites

- CMake 3.23 or newer to use the supported CMake presets. Manual configuration
  supports CMake 3.18 or newer.
- Python 3.10 or newer. The enabled Diligent Vulkan/SPIR-V toolchain needs a
  Python interpreter even when CRESSim-Neo's Python bindings are disabled.
- Linux: Clang/Clang++ with C++17 support and Ninja.
- Windows: Visual Studio 2022 with the **Desktop development with C++**
  workload and a Windows SDK.
- macOS: Xcode Command Line Tools, Ninja, and the LunarG Vulkan SDK with
  MoltenVK. Set `VULKAN_SDK` before configuration. Apple Silicon is the
  regularly supported source-build tier; Intel and universal builds are not
  regularly validated.
- Linux viewer builds: development packages for Xcursor, Xext, Xi, Xinerama,
  and XRandR.
- Python development headers when building Python bindings.
- A Vulkan-capable graphics driver/runtime for Vulkan-backed programs.

On Ubuntu or Debian, install the standard Linux dependencies with:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  clang \
  clang++ \
  cmake \
  ninja-build \
  python3-dev \
  python3-venv \
  python3-pip \
  git \
  libvulkan-dev \
  vulkan-tools \
  libxcursor-dev \
  libxext-dev \
  libxi-dev \
  libxinerama-dev \
  libxrandr-dev
```

Linux and Windows use a pinned DXC runtime by default; CMake downloads it and
verifies its SHA-256 the first time it populates a build-directory cache.
In-tree dependencies—such as Diligent Engine (`extern/DiligentEngine`),
GLFW (`extern/glfw`), and pybind11 (`extern/pybind11`)—are tracked as git
submodules, require no manual user handling or system-level installation, and
are compiled automatically. Simply initialize them when checking out the repository:

```bash
git submodule update --init --recursive
```

Keep the intended virtual or Conda environment active when building Python so
CMake and pip use the same interpreter. macOS uses the Vulkan SDK's MoltenVK
and disables DXC. It is a source-build-only, best-effort tier: CUDA interop,
Ultrasound, and macOS release wheels are not provided.

## Native development

`CMakePresets.json` is the canonical native configuration interface. Release
and debug presets build the shared SDK, viewer, and examples; Python, tests,
CUDA interop, and Ultrasound are off by default.

### Linux

```bash
cmake --preset linux-release
cmake --build --preset linux-release --parallel
```

Use `linux-debug` for interactive debugging. `linux-ci` is a headless test
profile:

```bash
cmake --preset linux-ci
cmake --build --preset linux-ci --parallel
ctest --preset linux-ci
```

### macOS

```bash
export VULKAN_SDK=/path/to/vulkansdk-macos
cmake --preset macos-release
cmake --build --preset macos-release --parallel
cmake --install build/macos-release --component CXXSDK
cmake --install build/macos-release --component Examples
```

Use `macos-debug` for development and `macos-ci` for a lean, headless local
CTest profile.

### Windows

```powershell
cmake --preset windows-vs2022-release
cmake --build --preset windows-vs2022-release --parallel
cmake --install build/windows-vs2022-release --config Release --component CXXSDK
cmake --install build/windows-vs2022-release --config Release --component Examples
```

Use `windows-vs2022-debug` for debugging. The `windows-vs2022-ci` preset is a
headless local verification profile; run it with its matching build and test
presets. Visual Studio is multi-config, so use `--config` for installation.

### Components and customization

The install components are:

- `CXXSDK`: shared C++ libraries, public headers, standard assets, shaders,
  and the CMake package. Assets are installed below
  `share/cressim-neo/assets`.
- `Examples`: enabled standalone C++ executables. Install `CXXSDK` with it so
  their default asset paths resolve.
- `Python`: the `cressim_neo` module, native runtime libraries, package files,
  shaders, models, and environment maps.

Set a prefix while configuring (or use the same `--prefix` for every install
component):

```bash
cmake --preset linux-release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
```

Without an override, native installations go below the build directory, for
example `build/linux-release/install`. Static C++ SDK installation is not a
supported public workflow. Set `CRESSIM_NEO_ASSET_DIR` to use a custom asset
tree; otherwise C++ uses installed assets and Python uses package-local assets.

Pass `-D` options after a preset to customize it. For example:

```bash
cmake --preset linux-release -DCRESSIM_NEO_BUILD_PYTHON=ON
cmake --build --preset linux-release --parallel
```

For a headless build, add `-DCRESSIM_NEO_BUILD_VIEWER=OFF` and
`-DCRESSIM_NEO_BUILD_EXAMPLES=OFF`. Other useful switches are:

- `-DBUILD_TESTING=ON`
- `-DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON`
- `-DCRESSIM_NEO_ENABLE_CUDA_INTEROP=ON`
- `-DCRESSIM_NEO_ENABLE_ULTRASOUND=ON` (also requires CUDA interop and a CUDA
  compiler)
- `-DCRESSIM_NEO_CUDA_RUNTIME_PROVIDER=SYSTEM` for a locally built Python CUDA
  extension (`AUTO` is the local-development default; `MANAGED` is for
  distributed CUDA-wheel lanes)
- `-DCRESSIM_NEO_DXC_PROVIDER=SYSTEM` to use SDK-provided DXC, or `OFF` to omit
  DXC. Without DXC, DXC-only Vulkan features fall back and Windows D3D12 is
  unsupported.

### Consuming an installed C++ SDK

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
```

```cmake
find_package(CRESSimNeo CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE CRESSimNeo::engine)
```

## Python package development and wheels

`pyproject.toml` uses scikit-build-core and builds the Python component and
interactive viewer, with examples, tests, CUDA interop, and Ultrasound off.
Build a local wheel with:

```bash
scripts/build_local_wheel.sh
python -m pip install dist/cressim_neo-*.whl
python -c "import cressim_neo; print(cressim_neo.__file__)"
```

On Windows, activate the intended environment and run
`./scripts/build_local_wheel.ps1`. On macOS, build from a recursive checkout
with `VULKAN_SDK` set and DXC disabled:

```bash
python -m pip wheel --no-deps --wheel-dir dist \
  -C cmake.define.CRESSIM_NEO_DXC_PROVIDER=OFF .
```

For editable development, run `python -m pip install -e .`. Package builds can
take several minutes; use the preset-based native build for frequent C++
iteration. The local wheel requires NumPy. Install Gymnasium separately when
its environment wrappers are needed:

```bash
python -m pip install gymnasium
```

Local wheels do not include CUDA interop or Ultrasound. Install PyTorch
separately when needed, and use the release-wheel workflow for CUDA-enabled
Linux or Windows distributions.

## Distribution and packaging strategy

The long-term distribution plan addresses target environments across platforms:

- **Stable Linux / Windows releases:** Distributed as prebuilt native binaries
  (C++ SDK, viewer, standalone examples) with a pinned CUDA-runtime prerequisite
  matching the build toolchain.
- **Python on mainstream platforms:** Pinned CUDA-specific wheels (`cu126`, `cu130`,
  `cu132`) with pinned NVIDIA pip runtime dependencies, installed in isolated
  virtual or Conda environments alongside matching PyTorch wheels.
- **Rolling distributions:** Build from source in an environment with compatible
  system CUDA and PyTorch packages. Configure
  `CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=SYSTEM` and verify compatibility with the
  installed PyTorch runtime.

:::{note}
**PyTorch CUDA runtime compatibility:**
Torch interoperability (DLPack and CUDA external memory exchange) strictly requires
that CRESSim-Neo links and interacts with the exact same CUDA runtime as PyTorch.
On stable Linux distributions and Windows, system Python packages and system CUDA
installations frequently conflict with PyTorch's bundled CUDA runtime wheels.
Performing a system-Python install with system CUDA on stable Linux/Windows is
therefore unsupported for Torch-interoperable workflows; users should rely on
isolated virtual environments with matching wheels. For system-CUDA source builds,
use compatible CUDA and PyTorch packages.
:::
