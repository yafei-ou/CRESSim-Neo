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

By default, configuration downloads the pinned DXC runtime release
(`v1.9.2602.24`) with a SHA-256 check. This is the runtime used by released
Linux and Windows artifacts; it is copied beside build products and installed
with the C++ SDK and Python package. CMake needs network access the first time
it populates this build-directory cache.

The repository checkout must include the `extern/DiligentEngine` dependency
tree.  For Python-enabled CMake builds, CMake first looks for a pybind11 CMake
package on its configured search paths (for example, one supplied through
`CMAKE_PREFIX_PATH`).  Otherwise it uses `extern/pybind11` when present, or
fetches pybind11 there during the first configure.

### Third-party compliance inventory

Generate a full source-level candidate inventory, including initialized
submodules and fetched dependency sources, with [ScanCode Toolkit](https://scancode-toolkit.readthedocs.io/):

```bash
scripts/scan_third_party.sh
```

The script writes pretty-printed `build/compliance/scancode.json` with ScanCode
license, copyright, email, and URL findings, omitting files with no findings.
It detects licenses without copying every matched license passage into the report. Use
`--include-license-text` only for a targeted rescan of shortlisted third-party
paths. This report identifies candidates, not the shipped notice. Review it
against each supported C++ SDK and Python install profile, then retain the
required notices for code, headers, assets, and binaries actually distributed.
The initial scan excludes the large DiligentEngine submodule; add
`--include-diligent` when preparing the complete inventory.
Create or refresh `build/compliance/third_party_review.md`, a compact dashboard,
and `build/compliance/third_party_evidence.json`, the grouped source paths and
line ranges behind it:

```bash
scripts/summarize_third_party_scan.sh
```

Record each review decision in
`compliance/third_party_review.json` as `pending`, `include`, or `exclude`; the
file is preserved and extended when later scans discover new components.
Generated ScanCode evidence stays out of that tracked decision file: the raw
report remains `scancode.json`, while the regrouped evidence remains
`third_party_evidence.json`.

After review, generate a draft notice from canonical upstream notice files
declared in each included component's `notice_files` array. For an upstream
component whose complete notice appears only in a source header, a reviewed
`notice_sources` entry may instead declare its `path`, `start_line`, and
`end_line`; the generator emits that exact range verbatim:

```bash
scripts/generate_third_party_notice.py
```

The draft is written to `build/compliance/THIRD_PARTY_NOTICES.draft.md`. The
generator refuses to run when an included component has no canonical notice
file, so ScanCode matches cannot accidentally substitute for required license
text.
CUDA is a user-provided build and runtime prerequisite and is not bundled by
the current install rules, so it is not part of the shipped third-party notice.

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
cmake --install build/linux-release --component Python
```

The project defines two install components:

- `CXXSDK` installs the shared C++ libraries, public headers (including the
  Diligent headers exposed by the public API), shaders, and the CMake package.
- `Python` installs the `cressim_neo` module, its native runtime libraries,
  Python sources, and shaders.

Pass `--prefix "$HOME/.local"` to either install command when a custom prefix
is needed. Static C++ SDK installation is not currently a supported public
workflow.

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

`pyproject.toml` uses scikit-build-core to drive the same CMake project.  Its
wheel configuration builds only the Python component, with viewer, examples,
tests, CUDA interop, and ultrasound disabled.  This makes the initial public
wheel baseline independent of a CUDA toolchain.

Build a wheel:

```bash
scripts/build_wheel.sh
```

To build a local wheel with the interactive viewer bindings, use
`scripts/build_wheel.sh --viewer`. Do not publish that artifact beside a
headless wheel with the same package version and platform tag.

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

Build the developer documentation site with its dedicated documentation API
profile:

```bash
scripts/build_docs.sh
```

See [`docs/README.md`](docs/README.md) for the full documentation prerequisites
and standalone build commands.
