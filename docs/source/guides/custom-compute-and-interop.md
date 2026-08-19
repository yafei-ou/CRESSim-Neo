# Custom compute and interop

Custom HLSL compute passes extend the C++ runtime without a host round trip.
They can read exposed simulation buffers and render targets, and write
engine-owned shared buffers.

1. After `uploadWorld()`, inspect the available custom-compute resources.
2. Create shared buffers for application-owned inputs or outputs.
3. Create and dispatch a custom compute pass.
4. Use prepared layout mappings to interpret packed rigid, joint, particle, or
   constraint data.

CUDA interop is optional. When available, a shared buffer can be exported to
CUDA or Python/PyTorch through DLPack; coordinate access with the runtime’s
explicit CUDA synchronization methods. See the C++ custom-compute examples and
the Python direct-runtime examples for language-specific syntax.
