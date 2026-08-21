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
- Linux viewer builds also require the X11 development packages for Xcursor,
  Xext, Xi, Xinerama, and XRandR
- Python 3.10 or newer, plus its development headers, when building Python
  bindings
- A Vulkan-capable graphics driver/runtime for running Vulkan-backed programs

By default, configuration downloads a pinned DXC runtime release with a
SHA-256 check. The managed Linux wheel lane uses DXC `v1.8.2505.1`, the newest
official Linux binary compatible with its `manylinux_2_34` ABI floor. The same
DXC release is used on Windows. The selected runtime is copied beside build
products and installed with the C++ SDK and Python package. CMake needs network
access the first time it populates this build-directory cache.

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

## Native development

### Guided setup

For the normal shared-SDK workflow, configure a Release or Debug build with:

```bash
scripts/configure_builds.sh
```

The helper selects supported project features and safely preserves the generator
of existing build directories. New directories can use Ninja or Unix Makefiles.
It prints the commands for the configured build.

Build and install with standard CMake commands. For example, for a Release
build with both components enabled:

```bash
cmake --build build/linux-release --parallel
cmake --install build/linux-release --component CXXSDK
cmake --install build/linux-release --component Examples
cmake --install build/linux-release --component Python
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

Pass `--prefix "$HOME/.local"` to either install command when a custom prefix
is needed. Static C++ SDK installation is not currently a supported public
workflow.

All runtime assets resolve through one root: set `CRESSIM_NEO_ASSET_DIR` to use
a custom asset directory; otherwise C++ resolves the installed asset tree and
Python resolves its package-local `cressim_neo/assets` directory. Explicit
shader or model paths still take precedence where supported.

### Manual CMake configuration

Use this equivalent workflow when scripting, automating, or choosing CMake
options directly. Configure a Release build with Python bindings enabled:

```bash
cmake -S . -B build/linux-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRESSIM_NEO_BUILD_PYTHON=ON
cmake --build build/linux-release --parallel
```

This build directory is persistent and incremental. Re-run the build command
after changing source files; CMake rebuilds only affected targets. To limit
parallelism:

```bash
cmake --build build/linux-release --parallel 4
```

For a new build directory, add `-G Ninja` to the configure command to use Ninja.

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
