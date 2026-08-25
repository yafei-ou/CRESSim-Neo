# CRESSim-Neo

CRESSim-Neo is a C++17 simulation engine with optional Python bindings.
Third-party C++ dependencies (including Diligent Engine, pybind11, and GLFW)
are tracked as in-tree git submodules; a normal CMake build compiles them
automatically. CMake produces the native libraries, and the Python build adds
the `cressim_neo` extension module.

This document describes the current developer workflow.  Documentation-site
instructions are in [`docs/README.md`](docs/README.md).

## Prerequisites

- CMake 3.23 or newer for the supported CMake presets. The project itself can
  still be configured manually with CMake 3.18 or newer.
- Python 3.10 or newer. The enabled Diligent Vulkan/SPIR-V toolchain requires
  a Python interpreter even when CRESSim-Neo's optional Python bindings are
  disabled.
- Linux: Clang and Clang++ with C++17 support, plus Ninja.
- Windows: Visual Studio 2022 with the Desktop development with C++ workload
  and a Windows SDK. The supported Windows presets select the MSVC toolchain.
- macOS: Xcode Command Line Tools, Ninja, and the LunarG Vulkan SDK with
  MoltenVK. Set `VULKAN_SDK` to that SDK before configuring. The supported
  source-build tier is Apple Silicon; Intel and universal builds are not
  regularly validated.
- Linux viewer builds also require the X11 development packages for Xcursor,
  Xext, Xi, Xinerama, and XRandR
- Python development headers when building Python bindings
- A Vulkan-capable graphics driver/runtime for running Vulkan-backed programs

On **Ubuntu / Debian**, install system dependencies via `apt`:

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
  vulkan-tools
```

By default, Linux and Windows configuration downloads a pinned DXC runtime release with a
SHA-256 check. The managed Linux wheel lane uses DXC `v1.8.2505.1`, the newest
official Linux binary compatible with its `manylinux_2_34` ABI floor. The same
DXC release is used on Windows. The selected runtime is copied beside build
products and installed with the C++ SDK and Python package. CMake needs network
access the first time it populates this build-directory cache.

In-tree dependencies—including Diligent Engine (`extern/DiligentEngine`),
GLFW (`extern/glfw`), and pybind11 (`extern/pybind11`)—are tracked as git
submodules and compiled automatically by CMake. They require no manual user
handling or separate system-package installation beyond checking them out:

```bash
git submodule update --init --recursive
```

Python must be available on `PATH` for every configuration. When building
Python bindings or wheels, activate the intended virtual environment or Conda
environment first so CMake and pip use the same interpreter.

macOS uses MoltenVK through the Vulkan SDK and deliberately disables DXC. This
is a source-build-only, best-effort tier: no macOS release wheels, CUDA
interop, or Ultrasound support are provided. Diligent uses its non-DXC shader
compiler fallback, so DXC-only shader features are unavailable.

## Native development

### Preset-based setup

`CMakePresets.json` is the canonical native configuration interface. The
release and debug presets enable the shared SDK, viewer, and examples; Python,
tests, CUDA interop, and Ultrasound are disabled by default.

On Linux:

```bash
cmake --preset linux-release
cmake --build --preset linux-release --parallel
```

Use `linux-debug` or `linux-ci` for the other Linux presets:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug --parallel
```

On macOS, install Xcode Command Line Tools and the LunarG Vulkan SDK first,
then set `VULKAN_SDK` in the shell that configures and runs CRESSim-Neo. The
SDK supplies the Vulkan loader and MoltenVK. The macOS presets disable DXC and
use Diligent's fallback shader compiler. Use the Apple Silicon presets:

```bash
export VULKAN_SDK=/path/to/vulkansdk-macos
cmake --preset macos-release
cmake --build --preset macos-release --parallel
cmake --install build/macos-release --component CXXSDK
cmake --install build/macos-release --component Examples
```

Use `macos-debug` for interactive development. `macos-ci` is a lean headless
CTest profile for local verification; it is not a hosted CI commitment.

On Windows, use the Visual Studio 2022 presets. `CMAKE_BUILD_TYPE` does not
select a configuration for Visual Studio, so use the matching build preset and
pass `--config` to installation:

```powershell
cmake --preset windows-vs2022-release
cmake --build --preset windows-vs2022-release --parallel
cmake --install build/windows-vs2022-release --config Release --component CXXSDK
cmake --install build/windows-vs2022-release --config Release --component Examples
```

Use `windows-vs2022-debug` for an interactive debug build. The CI preset is a
headless test profile, useful for local verification:

```powershell
cmake --preset windows-vs2022-ci
cmake --build --preset windows-vs2022-ci --parallel
ctest --preset windows-vs2022-ci
```

The project defines three install components:

- `CXXSDK` installs the shared C++ libraries, public headers (including the
  Diligent headers exposed by the public API), standard models, environment
  maps, shaders, their authoring sources, and the CMake package. Assets are
  installed under `share/cressim-neo/assets`.
- `Examples` installs the enabled standalone C++ example executables. Install
  `CXXSDK` alongside it so their default asset paths resolve.
- `Python` installs the `cressim_neo` module, its native runtime libraries,
  Python sources, shaders, models, and environment maps.

Set a persistent custom prefix while configuring, or override the prefix for
one installation. Use the same prefix for every component:

```bash
cmake --preset linux-release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
```

```bash
cmake --preset macos-release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
```

```powershell
cmake --preset windows-vs2022-release -DCMAKE_INSTALL_PREFIX='C:\CRESSim-Neo'
# Or: cmake --install ... --prefix 'C:\CRESSim-Neo'
```

Without an override, each native build installs below its build directory, for
example `build/windows-vs2022-release/install`. Static C++ SDK installation is
not currently a supported public workflow.

All runtime assets resolve through one root: set `CRESSIM_NEO_ASSET_DIR` to use
a custom asset directory; otherwise C++ resolves the installed asset tree and
Python resolves its package-local `cressim_neo/assets` directory. Explicit
shader or model paths still take precedence where supported.

### Custom preset configuration

Pass `-D` options after a preset to customize it. For example, configure a
Linux release build with Python bindings:

```bash
cmake --preset linux-release -DCRESSIM_NEO_BUILD_PYTHON=ON
cmake --build --preset linux-release --parallel
```

For a smaller headless build, disable the default viewer and examples:

```bash
cmake --preset linux-release \
  -DCRESSIM_NEO_BUILD_VIEWER=OFF \
  -DCRESSIM_NEO_BUILD_EXAMPLES=OFF
```

The equivalent Windows invocation is:

```powershell
cmake --preset windows-vs2022-release `
  -DCRESSIM_NEO_BUILD_PYTHON=ON `
  -DCRESSIM_NEO_BUILD_VIEWER=OFF `
  -DCRESSIM_NEO_BUILD_EXAMPLES=OFF
```

Useful optional CMake switches are:

- `-DBUILD_TESTING=ON` — build the test tree.
- `-DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON` — run clang-tidy as part of C++ builds.
- `-DCRESSIM_NEO_ENABLE_CUDA_INTEROP=ON` — require and enable CUDAToolkit.
- `-DCRESSIM_NEO_ENABLE_ULTRASOUND=ON` — enable CRESSim-Ultrasound.  This also
  requires CUDA interop and a working CUDA compiler.
- `-DCRESSIM_NEO_CUDA_RUNTIME_PROVIDER=SYSTEM` — use the system CUDA runtime
  when a locally built Python CUDA extension is imported. `AUTO` is the
  local-development default; `MANAGED` is reserved for distributed CUDA wheel
  lanes.
- `-DCRESSIM_NEO_DXC_PROVIDER=SYSTEM` — use DXC found through the local SDKs
  instead of the pinned runtime. This is intended only for local development.
- `-DCRESSIM_NEO_DXC_PROVIDER=OFF` — do not provision DXC. Vulkan falls back
  from DXC-only features and the Windows D3D12 shader path is unsupported.

### Consuming an installed C++ SDK

After installing `CXXSDK` to a prefix, point a consumer project at it:

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

`pyproject.toml` uses scikit-build-core to drive the same CMake project. Its
wheel configuration builds the Python component and interactive viewer, while
examples, tests, CUDA interop, and ultrasound remain disabled. This makes the
initial public wheel baseline independent of a CUDA toolchain.

Build a wheel:

```bash
scripts/build_local_wheel.sh
```

On Windows, after activating the intended environment:

```powershell
conda activate cressim_neo
.\scripts\build_local_wheel.ps1
```

On macOS, build from a recursive checkout and keep the Vulkan SDK available.
This creates a local wheel only; it is not a release artifact:

```bash
python -m pip wheel --no-deps --wheel-dir dist \
  -C cmake.define.CRESSIM_NEO_DXC_PROVIDER=OFF .
python -m pip install dist/cressim_neo-*.whl
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

The local wheel has no CUDA interop or Ultrasound dependency and requires
NumPy. Gymnasium is optional:

```bash
python -m pip install 'cressim-neo[gymnasium]'
```

The local wheel workflow does not require PyTorch. Install PyTorch separately
in the active environment when using integrations that need it. Local wheels
do not include CUDA interop or Ultrasound; use the release-wheel workflow for
those CUDA-enabled distributions on Linux or Windows.

## Distribution and packaging strategy

The long-term distribution roadmap is structured by platform and target environment:

- **Stable Linux / Windows releases:** Prebuilt native binaries (C++ SDK, viewer,
  and standalone executables) with a pinned CUDA-runtime prerequisite matching the
  toolchain release.
- **Python on mainstream platforms:** CUDA-specific wheels (such as `cu126`,
  `cu130`, `cu132`) with pinned NVIDIA pip runtime dependencies, installed into
  isolated virtual environments or Conda environments alongside matching PyTorch wheels.
- **Arch Linux / rolling distributions:** Source-based `PKGBUILD` that compiles
  against the distribution's system CUDA and official `python-pytorch-cuda` package.
  Because rolling distributions maintain a synchronized system Python, system CUDA,
  and system PyTorch stack, no upstream prebuilt CUDA wheel is required.

> [!NOTE]
> **Why system-Python installs on stable Linux / Windows are not supported for Torch workflows:**
> Zero-copy Torch interoperability (via DLPack and CUDA external memory) strictly
> requires that CRESSim-Neo brings in and links the exact same CUDA runtime as
> PyTorch does. On stable Linux distributions and Windows, system Python packages
> and system-installed CUDA toolchains frequently mismatch PyTorch's bundled/pinned
> CUDA runtime packages (such as `nvidia-cuda-runtime-cu12` or `torch/lib`).
> Doing a system-Python install on stable Linux/Windows therefore risks runtime
> conflicts and symbol mismatches. Use isolated virtual environments with matching
> wheels on stable platforms, or system packages via `PKGBUILD` on rolling distributions.

## Release wheels

Docker-based release lanes and their clean-environment tests are documented in
[the wheel build guide](packaging/README.md). The active lanes are `base`
(without CUDA interop or Ultrasound), `cu126` (CUDA 12.6 with Ultrasound),
`cu130` (CUDA 13.0 with Ultrasound), and `cu132` (CUDA 13.2 with Ultrasound).

## Documentation

Build the developer documentation site with its dedicated documentation API
profile:

```bash
scripts/build_docs.sh
```

See [`docs/README.md`](docs/README.md) for the full documentation prerequisites
and standalone build commands.

## Third-party compliance

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) is the tracked notice shipped
with the C++ SDK and Python package. It contains the reviewed license and notice
text for third-party code, assets, headers, and bundled binaries distributed by
the project.

Use [ScanCode Toolkit](https://scancode-toolkit.readthedocs.io/) to refresh the
source inventory. A full review includes the Diligent submodule:

```bash
scripts/scan_third_party.sh --include-diligent
scripts/summarize_third_party_scan.sh
```

The scan writes `build/compliance/scancode.json`. The summary command writes a
compact review dashboard at `build/compliance/third_party_review.md` and grouped
evidence, including source paths and line ranges, at
`build/compliance/third_party_evidence.json`. ScanCode findings are candidates
for review; they do not by themselves identify material that is distributed.
Use `--include-license-text` with the scan script when reviewing license text in
a specific candidate area.

Record the decision for each component in
`compliance/third_party_review.json` as `pending`, `include`, or `exclude`.
Included components declare canonical `notice_files` or exact `notice_sources`.
The review registry retains decisions across scans; generated scan reports and
evidence are separate build artifacts.

Generate the tracked notice after updating the review registry. Pass the build
directory for a release that bundles DXC so the generator can read the pinned
archive notice files declared in `compliance/third_party_artifacts.json`:

```bash
python3 scripts/generate_third_party_notice.py \
  --build-dir build/linux-release \
  --output THIRD_PARTY_NOTICES.md
```

Review and commit the updated notice with the corresponding registry changes.
The generated notice excludes internal review notes by default; use
`--include-notes` only for a review draft. CUDA is supplied by the user and is
not included in the distributed notice unless a future package bundles CUDA
components.
