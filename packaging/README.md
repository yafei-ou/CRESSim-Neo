# Packaging and release workflow

The default wheel and managed CUDA wheel differ only in their CUDA-runtime
contract.

| Artifact | Builder | CUDA runtime | Distribution |
| --- | --- | --- | --- |
| Default wheel | `../scripts/build_local_wheel.sh` | Disabled | Local developer wheel |
| `cu126` wheel | `../scripts/build_cuda126_wheels.sh` | Managed NVIDIA Python packages | GitHub Release asset |

## Local development

Use `../scripts/configure_builds.sh` for persistent native CMake build trees.
It enables the viewer by default and is the fastest workflow for repeated C++
iteration. Use `../scripts/build_local_wheel.sh` only when a local wheel artifact is
needed; it builds the root package configuration and is not a portable CUDA
release artifact.

For a quick CUDA wheel iteration, build one CPython ABI from the repository
root:

```bash
CIBW_BUILD='cp312-*' scripts/build_cuda126_wheels.sh
scripts/verify_cuda126_wheel.sh dist/cu126/*cp312*.whl
```

Use the unrestricted builder only for release candidates; it builds CPython
3.10 through 3.13.

## CUDA 12.6 release candidate

The authoritative CUDA lane configuration is `cuda126/release.toml`. Before
publishing, run from the repository root:

```bash
rm -rf dist/cu126
scripts/build_cuda126_wheels.sh
for wheel in dist/cu126/*.whl; do
  scripts/verify_cuda126_wheel.sh "$wheel"
done
```

The verifier confirms that CUDA runtime libraries remain external, no
build-machine paths are present, and the wheel is compatible with
`manylinux_2_34_x86_64`.

On an NVIDIA GPU host configured with NVIDIA Container Toolkit, validate a
fresh Ubuntu environment. The test image selects NVIDIA's EGL Vulkan ICD for
headless containers; this does not alter normal user installations.

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
    micromamba run -n ci python scripts/test_cuda126_interop.py
  '
```

The GPU smoke test validates the managed runtime loader, CUFFT/CURAND,
Ultrasound initialization, Vulkan/CUDA external-memory interop, and a DLPack
round trip with PyTorch.
