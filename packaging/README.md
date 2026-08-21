# Python wheel builds

There are two distinct workflows: a host-local developer wheel and reproducible
Docker release wheels. Do not use a local wheel as a release artifact.

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

## Docker release lanes

Release wheels are built in controlled manylinux containers and written to
`dist/<lane>/`. The active lanes are:

| Lane | CUDA interop | Ultrasound | Builder | Runtime test image |
| --- | --- | --- | --- | --- |
| `base` | No | No | `../docker/manylinux-base/Dockerfile` | `../docker/ubuntu-base/Dockerfile` |
| `cu126` | CUDA 12.6 | Yes | `../docker/manylinux-cu126/Dockerfile` | `../docker/ubuntu-cu126/Dockerfile` |
| `cu130` | CUDA 13.0 | Yes | `../docker/manylinux-cu130/Dockerfile` | `../docker/ubuntu-cu130/Dockerfile` |
| `cu132` | CUDA 13.2 | Yes | `../docker/manylinux-cu132/Dockerfile` | `../docker/ubuntu-cu132/Dockerfile` |

Each CUDA lane pins the official PyTorch wheel and its CUDA runtime bundle in
its wheel metadata and `requirements.lock`.

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
