# Packaging and Release Wheels

Local developer wheels and release wheels are different artifacts. Build local
wheels with `scripts/build_local_wheel.sh` (or
`scripts/build_local_wheel.ps1` on Windows); do not publish them as releases.
The local workflow is covered in {doc}`../getting-started/build`.

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
