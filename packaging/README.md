# Python wheel builds and packaging

There are two distinct workflows: a host-local developer wheel and reproducible
release wheels. Do not use a local wheel as a release artifact.

## Distribution and packaging strategy

The long-term packaging architecture is structured around target platforms and their runtime ecosystems:

- **Stable Linux / Windows releases:** Prebuilt native binaries (C++ SDK, viewer, and executables)
  with a pinned CUDA-runtime prerequisite matching the toolchain release.
- **Python on mainstream platforms:** CUDA-specific wheels (e.g. `cu126`, `cu130`, `cu132`)
  with pinned NVIDIA pip runtime dependencies, installed in isolated virtual or Conda
  environments alongside matching PyTorch wheels.
- **Arch Linux / rolling distributions:** Source-based `PKGBUILD` that compiles against
  the distribution's system CUDA and official `python-pytorch-cuda` package. Because rolling
  distributions keep system Python, system CUDA, and system PyTorch in sync, no upstream
  prebuilt CUDA wheel is needed.

### Torch interop and CUDA runtime compatibility

Zero-copy Torch interoperability (via DLPack and CUDA external memory) strictly requires that
CRESSim-Neo brings in and links the exact same CUDA runtime version as PyTorch does.

On stable Linux distributions and Windows, system Python environments and system CUDA
installations frequently conflict with PyTorch's bundled/pinned CUDA runtime wheels (such as
`nvidia-cuda-runtime-cu12` or `torch/lib`). Attempting a system-Python install linked against
system CUDA on stable Linux/Windows creates runtime mismatches and symbol collisions.

Therefore, system-Python installs on stable Linux/Windows are not supported for Torch workflows.
Mainstream platforms should always use isolated virtual environments with matching CUDA wheels,
while rolling distributions should build against system CUDA and system PyTorch via `PKGBUILD`.

## Local developer wheel

Use the local script when iterating on Python bindings or when a wheel is needed
for the current host:

```bash
scripts/build_local_wheel.sh
python -m pip install --force-reinstall dist/cressim_neo-*.whl
python -c "import cressim_neo; print(cressim_neo.Runtime().get_info())"
```

For repeated C++ work, prefer a persistent native preset build such as
`cmake --preset linux-release`; it avoids rebuilding the engine for every
wheel.

On macOS, local source builds are supported through the `macos-*` presets and
the LunarG Vulkan SDK/MoltenVK. Set `VULKAN_SDK` before building and pass
`-C cmake.define.CRESSIM_NEO_DXC_PROVIDER=OFF` when invoking pip directly.
macOS wheels are not release artifacts and must not be published.

## Linux release lanes

Release wheels are built in controlled manylinux containers and written to
`dist/<lane>/`. The active lanes are:

| Lane | CUDA interop | Ultrasound | Builder | Runtime test image |
| --- | --- | --- | --- | --- |
| `base` | No | No | `../docker/manylinux-base/Dockerfile` | `../docker/ubuntu-base/Dockerfile` |
| `cu126` | CUDA 12.6 | Yes | `../docker/manylinux-cu126/Dockerfile` | `../docker/ubuntu-cu126/Dockerfile` |
| `cu130` | CUDA 13.0 | Yes | `../docker/manylinux-cu130/Dockerfile` | `../docker/ubuntu-cu130/Dockerfile` |
| `cu132` | CUDA 13.2 | Yes | `../docker/manylinux-cu132/Dockerfile` | `../docker/ubuntu-cu132/Dockerfile` |

Each CUDA lane pins its CUDA runtime bundle in wheel metadata. Its
`requirements.lock` separately pins the matching official PyTorch wheel used
by the CUDA runtime tests.

### Linux compatibility floor

Release wheels target `manylinux_2_34_x86_64`: glibc 2.34 or newer, including
Ubuntu 22.04+. This floor is currently required by the bundled DXC binary.
The upstream manylinux 2.34 image is marked alpha and warns that shared
libraries installed with `dnf` may target x86-64-v2. CRESSim wheels do not
bundle `dnf`-installed shared libraries; CUDA libraries are external package
dependencies, while the wheel bundles only CRESSim components and DXC. We
should revisit a stable `manylinux_2_28` baseline when DXC can be built or
obtained with that older ABI floor.

Install the host tools once:

```bash
python -m pip install cibuildwheel auditwheel
```

Build all configured CPython ABIs for a lane:

```bash
scripts/build_release_wheels.sh --lane base
scripts/build_release_wheels.sh --lane cu126
scripts/build_release_wheels.sh --lane cu130
scripts/build_release_wheels.sh --lane cu132
```

Each build removes the previous `dist/<lane>/` directory first, so a release
artifact directory contains only wheels from its current invocation.

For a quick iteration, narrow cibuildwheel's ABI selection:

```bash
CIBW_BUILD='cp312-*' scripts/build_release_wheels.sh --lane base
CIBW_BUILD='cp312-*' scripts/build_release_wheels.sh --lane cu126
CIBW_BUILD='cp312-*' scripts/build_release_wheels.sh --lane cu130
CIBW_BUILD='cp312-*' scripts/build_release_wheels.sh --lane cu132
```

After every release build, run the manual verification below. It checks the
manylinux tag and build-path hygiene for every lane, rejects CUDA dependencies
from `base`, and ensures every CUDA lane requests its external CUDART soname
without bundling NVIDIA runtime libraries.

## Release-wheel verification

Run this after building and again before attaching copied artifacts to a GitHub
Release:

```bash
for wheel in dist/base/*.whl; do
  scripts/verify_release_wheel.sh --lane base "$wheel"
done

for wheel in dist/cu126/*.whl; do
  scripts/verify_release_wheel.sh --lane cu126 "$wheel"
done

for wheel in dist/cu130/*.whl; do
  scripts/verify_release_wheel.sh --lane cu130 "$wheel"
done

for wheel in dist/cu132/*.whl; do
  scripts/verify_release_wheel.sh --lane cu132 "$wheel"
done
```

### Base lane test

The base test runs in clean Ubuntu 22.04 with Mesa Vulkan and no CUDA Toolkit,
NVIDIA driver, or NVIDIA Python packages:

```bash
docker build -f docker/ubuntu-base/Dockerfile -t cressim-neo/ubuntu-base:test .
docker run --rm -v "$PWD:/workspace" -w /workspace cressim-neo/ubuntu-base:test \
  bash -lc '
    micromamba create -y -n base -f packaging/base/environment.yml
    micromamba run -n base python -m pip install --force-reinstall dist/base/*cp312*.whl
    micromamba run -n base python -m pip check
    micromamba run -n base python scripts/test_base_wheel.py
  '
```

### CUDA lane tests

Each CUDA test requires an NVIDIA GPU host configured with NVIDIA Container
Toolkit. It validates the managed runtime loader, CUFFT/CURAND, Ultrasound,
Vulkan/CUDA external-memory interop, and a PyTorch DLPack round trip. Replace
`cu126` below with `cu130` or `cu132` to test that lane:

```bash
docker build -f docker/ubuntu-cu126/Dockerfile -t cressim-neo/ubuntu-cu126:test .
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace" -w /workspace \
  cressim-neo/ubuntu-cu126:test \
  bash -lc '
    micromamba create -y -n ci -f packaging/cuda126/environment.yml
    micromamba run -n ci python -m pip install --force-reinstall dist/cu126/*cp312*.whl
    micromamba run -n ci python -m pip check
    micromamba run -n ci python scripts/test_cuda_interop.py --lane cu126
  '
```

## Windows release lanes

Build Windows wheels directly on a Windows host; Docker is not used. Run the
release helper from an x64 MSVC developer environment with `cibuildwheel`
installed in the active Python environment:

```powershell
python -m pip install --upgrade cibuildwheel
.\scripts\build_release_wheels.ps1 -Lane base
.\scripts\build_release_wheels.ps1 -Lane cu126
```

The helper writes wheels to `dist/windows/<lane>/`, selects the corresponding
packaging configuration, and runs the Windows wheel verifier automatically.
Release build isolation installs `pybind11-stubgen` and generated
`cressim_neo/__init__.pyi` stubs are required in every release wheel.
CUDA lanes require the matching NVIDIA CUDA Toolkit installed side by side:
CUDA 12.6 for `cu126`, CUDA 13.0 for `cu130`, and CUDA 13.2 for `cu132`.
It sets the CUDA compiler and toolkit root for the isolated wheel builds; do
not configure CMake separately.

CUDA runtime DLLs are not bundled. Before installing a CUDA CRESSim wheel,
install the matching CUDA PyTorch wheel in the same environment. Pip installs
PyTorch CUDA lanes from the PyTorch index; an activated Conda environment with
the matching `pytorch-cuda` runtime is also supported. The CRESSim loader uses
PyTorch's `torch\lib` directory or Conda's `Library\bin` directory before
loading its native module.

### Windows wheel verification

The release helper runs this verification automatically. Run it again before
publishing copied artifacts:

```powershell
Get-ChildItem dist\windows\base\*.whl | ForEach-Object {
  .\scripts\verify_release_wheel.ps1 -Lane base -Wheel $_.FullName
}

Get-ChildItem dist\windows\cu126\*.whl | ForEach-Object {
  .\scripts\verify_release_wheel.ps1 -Lane cu126 -Wheel $_.FullName
}
```

The verifier requires a Visual Studio developer PowerShell because it uses
`dumpbin` to inspect every `.pyd` and `.dll` in the wheel. It rejects bundled
CUDA runtime DLLs, rejects CUDA dependencies from the base lane, and requires
the CUDA lane's expected CUDART DLL.

### Windows runtime tests

Test each wheel in a fresh virtual environment on a Windows machine with an
NVIDIA GPU and a driver compatible with the selected CUDA lane. For `cu126`:

```powershell
python -m venv .venv-cressim-cu126-test
.\.venv-cressim-cu126-test\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install --force-reinstall dist\windows\cu126\*.whl
python -m pip check
python scripts\test_cuda_interop.py --lane cu126
```

For the base lane, omit PyTorch and run `scripts\test_base_wheel.py` after
installing a wheel from `dist\windows\base\`. Use the matching PyTorch index
and lane name for `cu130` or `cu132`.
