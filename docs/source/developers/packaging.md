# Packaging and Release Wheels

Local developer wheels and release wheels are different artifacts. Build local
wheels with `scripts/build_local_wheel.sh` (or
`scripts/build_local_wheel.ps1` on Windows); do not publish them as releases.
The local workflow is covered in {doc}`../getting-started/build`.

## Distribution strategy

The distribution method depends on both the platform and how CRESSim-Neo is used:

- **Rolling Linux distributions:** Build from source against the system CUDA and
  PyTorch packages. Configure `CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=SYSTEM` and verify
  that the resulting runtime is compatible with the installed PyTorch package.
- **Stable Linux and Windows --- Python and PyTorch workflows:** Use CUDA-specific
  wheels (for example, `cu126`, `cu130`, or `cu132`) in an isolated virtual or Conda
  environment. Install the CRESSim-Neo and PyTorch wheels from the same named CUDA
  lane. On Linux, the CRESSim-Neo wheel declares the corresponding NVIDIA runtime
  packages; on Windows, install the matching PyTorch wheel first, because the
  CRESSim-Neo loader uses its `torch\lib` directory (or Conda's `Library\bin`).
- **Stable Linux and Windows --- native applications and C++ SDK:** Use prebuilt
  native binaries for the C++ SDK, viewer, and standalone examples. These releases
  require the CUDA runtime version specified for the toolchain release.

### Torch interop and CUDA runtime compatibility

Torch interoperability uses DLPack and CUDA external-memory exchange, so CRESSim-Neo
and PyTorch must run with a compatible CUDA runtime and driver stack in the same
process. On stable Linux distributions and Windows, system Python installations can
load CUDA libraries from the system toolchain that differ from the NVIDIA CUDA runtime
libraries used by PyTorch wheels. This can cause runtime-library conflicts or
missing-symbol errors.

For Python/Torch workflows on stable platforms, install CRESSim-Neo and PyTorch in the
same isolated virtual or Conda environment, using the same supported CUDA lane. For
source builds using system CUDA, use compatible system CUDA and PyTorch packages.

## Linux release lanes

Linux releases are built in controlled manylinux containers with
`cibuildwheel`. The supported lanes are `base` (no CUDA interop or Ultrasound),
`cu126`, `cu130`, and `cu132` (matching CUDA interop and Ultrasound builds).
They target `manylinux_2_34_x86_64` and are written under `dist/<lane>/`.

Install the host tools, then build one lane at a time:

```bash
python -m pip install cibuildwheel auditwheel
scripts/build_release_wheels.sh --lane base
scripts/build_release_wheels.sh --lane cu126
```

Use `CIBW_BUILD='cp312-*'` before the script for a narrow local ABI iteration.
The build script clears the selected lane's output directory before building.
Verify every resulting wheel before publishing:

```bash
for wheel in dist/base/*.whl; do
  scripts/verify_release_wheel.sh --lane base "$wheel"
done
```

Base wheels are tested in a clean Ubuntu 22.04/Mesa container. CUDA lanes also
need an NVIDIA GPU host and NVIDIA Container Toolkit for the interop and
Ultrasound runtime test.

## Windows release lanes

Build Windows releases on an x64 Visual Studio developer environment, not in
Docker. Install `cibuildwheel` in the active Python environment, then run:

```powershell
.\scripts\build_release_wheels.ps1 -Lane base
.\scripts\build_release_wheels.ps1 -Lane cu126
```

CUDA lanes require the matching locally installed CUDA Toolkit: 12.6, 13.0, or
13.2. The helper verifies the generated wheels automatically; run the verifier
again before publishing copied artifacts. Install the matching CUDA PyTorch
wheel before testing a CUDA CRESSim-Neo wheel—CUDA runtime DLLs are not bundled.

The full commands, lane images, runtime tests, and verification requirements
are maintained in `packaging/README.md`.
