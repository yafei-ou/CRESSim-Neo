# GPU Integration and Custom Compute

## Platform Support and CUDA Interoperability

The runtime currently targets the Vulkan and D3D12 backends provided by Diligent
Engine. Vulkan is the default cross-platform path and is used on all supported
builds, while D3D12 is available on Windows builds. On macOS, Vulkan support is
provided through MoltenVK.

CUDA interoperability can be optionally enabled at build time. When available,
the engine allocates exportable GPU buffers for shared structured data, exposing
device pointers for downstream CUDA/PyTorch use. Synchronization between the
graphics/compute backend and CUDA is handled with external timeline
semaphores/fences. If exportable allocation is unavailable, the same APIs fall
back to engine-only GPU buffers without interop.

## Resource Exposure and PyTorch Interoperability

Simulation data is exposed through prepared layout mappings, internal GPU
resources, and user-allocated shared buffers. The layout mappings provide the
information needed to interpret packed GPU buffers, such as IDs, environment
indices, and per-object particle offsets and counts. These mappings correspond
to the batched scene representation described in {doc}`batched-environments`.

| Resource | Exposed information |
| --- | --- |
| Prepared rigid layout mapping | Rigid-body and collider indexing |
| Prepared joint layout mapping | Joint indexing and body associations |
| Prepared particle layout mapping | Particle ownership and object ranges |
| Prepared constraint layout mapping | Constraint associations and routed-cable layout |
| Internal simulation buffers | Rigid, joint, particle, and related GPU state |
| Render targets | RGB, depth, segmentation, and ultrasound outputs |
| User-allocated shared buffers | Custom-compute inputs and outputs, with DLPack-exportable data |

Internal simulation buffers and render targets can be read in custom compute
passes. User-allocated shared buffers can be bound in custom compute passes and,
when CUDA interop is enabled, exported through DLPack. Using the layout mappings,
users can interpret simulation state and set up custom compute passes that read
internal simulation buffers and render targets, such as RGB outputs, and write
task-specific results into user-allocated shared buffers. Python-side code can
then construct `torch.Tensor` views over those shared buffers.

## Custom GPU Task Computation

Custom GPU task computation is supported through user-defined HLSL compute
passes. These passes are dispatched through `executeCustomComputePass()` and can
be inserted at any point in the staged frame loop, binding internal simulation
buffers, shader-readable render targets, and user-allocated shared buffers. See
{doc}`runtime-lifecycle` for the frame stages and {doc}`rendering-and-sensors`
for sensor outputs.

For robot learning environments, this allows actions, observations, rewards,
reset masks, and termination flags to be produced and consumed on GPU. Custom
GPU compute is supported independently of CUDA interop. When CUDA interop is
disabled, the same custom-compute path remains available, but user-allocated
buffers are not exportable through DLPack.
