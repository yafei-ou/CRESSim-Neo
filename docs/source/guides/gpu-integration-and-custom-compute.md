# Custom GPU Compute and Interoperability

CRESSim-Neo exposes GPU-resident simulation and sensor data to custom compute
passes. Custom compute uses HLSL and is available with the standard runtime;
CUDA interoperability is needed only when data must also be shared with CUDA or
PyTorch. See {doc}`../getting-started/build` for enabling CUDA interoperability.

## Custom GPU Compute

A custom pass is an HLSL compute shader compiled by the runtime for the active
GPU backend. A {cpp:struct}`CustomComputePassDesc <cressim::neo::engine::CustomComputePassDesc>`
(`CustomComputePassDesc` in Python) supplies its source either as the
`shaderSource` string or as a `shaderPath` relative to `shaderDirectory`. The
same descriptor specifies the entry point, HLSL thread-group size, resource
bindings, optional constant buffer, and dispatch dimensions. The
`structured_buffer_compat.hlsli` include provides the portable structured-buffer
declarations used by the built-in shaders and the examples.

This minimal shader writes four constant-buffer values to a user-owned output
buffer. The HLSL variable name must match the binding's `shaderVariableName`.

```hlsl
#include "structured_buffer_compat.hlsli"

cbuffer TaskConstants
{
    float4 values;
};

CRESSIM_RW_STRUCTURED_BUFFER(float, g_Output);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u) return;
    CRESSIM_SB_STORE(g_Output, 0u, values.x);
    CRESSIM_SB_STORE(g_Output, 1u, values.y);
    CRESSIM_SB_STORE(g_Output, 2u, values.z);
    CRESSIM_SB_STORE(g_Output, 3u, values.w);
}
```

::::{tab-set}

:::{tab-item} C++

```cpp
SharedBufferDesc outputDesc{};
outputDesc.debugName = "Task.Output";
outputDesc.elementStrideBytes = sizeof(float);
outputDesc.elementCount = 4;
outputDesc.access = SharedBufferAccess::ReadWrite;
outputDesc.bindFlags = SharedBufferBindFlags::ShaderResource |
                      SharedBufferBindFlags::UnorderedAccess;
const auto output = runtime.createSharedBuffer(outputDesc);

CustomComputePassDesc passDesc{};
passDesc.debugName = "Task.Constants";
passDesc.shaderSource = kTaskShader; // the HLSL source shown above
passDesc.threadGroupSizeX = 1;
CustomComputeResourceBindingDesc binding{};
binding.shaderVariableName = "g_Output";
binding.sharedBufferHandle = output;
binding.access = CustomComputeResourceAccess::ReadWrite;
passDesc.resourceBindings = {binding};
passDesc.constantBufferVariableName = "TaskConstants";
passDesc.constantBufferSizeBytes = sizeof(float) * 4;
const std::array<float, 4> constants{1.f, 2.f, 3.f, 4.f};
passDesc.constantData.resize(sizeof(constants));
std::memcpy(passDesc.constantData.data(), constants.data(), sizeof(constants));
passDesc.dispatch.mode = CustomComputeDispatchMode::ExplicitGroupCount;
passDesc.dispatch.groupCountX = 1;

// After authoring the scene, make its GPU resources available to the pass.
runtime.prepare();
runtime.uploadWorld();
const auto pass = runtime.createCustomComputePass(passDesc);
runtime.executeCustomComputePass(pass);
```

:::

:::{tab-item} Python

```python
import struct

output_desc = neo.SharedBufferDesc()
output_desc.debug_name = "Task.Output"
output_desc.element_stride_bytes = 4
output_desc.element_count = 4
output_desc.access = neo.SharedBufferAccess.ReadWrite
output_desc.bind_flags = (neo.SharedBufferBindFlags.ShaderResource |
                          neo.SharedBufferBindFlags.UnorderedAccess)
output = runtime.create_shared_buffer(output_desc)

pass_desc = neo.CustomComputePassDesc()
pass_desc.debug_name = "Task.Constants"
pass_desc.shader_source = task_shader  # the HLSL source shown above
pass_desc.thread_group_size_x = 1
binding = neo.CustomComputeResourceBindingDesc()
binding.shader_variable_name = "g_Output"
binding.shared_buffer_handle = output
binding.access = neo.CustomComputeResourceAccess.ReadWrite
pass_desc.resource_bindings = [binding]
pass_desc.constant_buffer_variable_name = "TaskConstants"
pass_desc.constant_buffer_size_bytes = 16
pass_desc.constant_data = list(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
pass_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
pass_desc.dispatch.group_count_x = 1

# After authoring the scene, make its GPU resources available to the pass.
runtime.prepare()
runtime.upload_world()
task_pass = runtime.create_custom_compute_pass(pass_desc)
runtime.execute_custom_compute_pass(task_pass)
```

:::

::::

Use `ExplicitGroupCount` when the work extent is known. With
`ResourceElementCount`, the runtime derives the X group count from
`countResourceKey`; the HLSL thread-group size still determines how many
elements each group covers. {download}`custom_compute_constant_buffer_torch.py
<../../../examples/python/custom_compute_constant_buffer_torch.py>` and
{download}`custom_compute_rigid_lateral_shift.py
<../../../examples/python/custom_compute_rigid_lateral_shift.py>` provide
complete runnable examples.

## Runtime Resources

Call `Runtime::listCustomComputeResources` (`Runtime.list_custom_compute_resources`
in Python) after the world has been prepared and uploaded. Each returned
`CustomComputeResourceDesc` gives the exact key, resource kind, permitted
access, element count and stride, and a `bindingGeneration`. The list is the
authoritative inventory for the current scene: resources that are not allocated
for its authored features do not appear.

| Resource group | Read/write keys | Read-only keys |
| --- | --- | --- |
| Rigid bodies | `rigid.positions`, `rigid.orientations`, `rigid.linear_velocities`, `rigid.angular_velocities`, `rigid.kinematic_target_positions`, `rigid.kinematic_target_orientations`, `rigid.kinematic_target_flags` | `rigid.inverse_inertia_local`, `rigid.body_types`, `rigid.proxy_particle_contact_materials`, `rigid.body_collider_offsets`, `rigid.body_collider_counts`, `rigid.body_collider_ranges`, `rigid.body_collider_indices` |
| Colliders | — | `collider.owner_body_indices`, `collider.broad_phase`, `collider.geometry`, `collider.materials`, `collider.shape_types`, `collider.enabled_flags` |
| Rigid constraints | `constraint.rigid_particle_attachments`, `constraint.strand_rigid_attachments`, `constraint.rigid_distance_constraints`, `constraint.routed_cable_descriptors` | `constraint.routed_cable_route_points`, `constraint.routed_cable_debug_segments` |
| Joints | `joint.ball`, `joint.spherical`, `joint.hinge`, `joint.slider` | `joint.hinge_runtime`, `joint.slider_runtime` |
| Entity poses | `entity.positions`, `entity.orientations`, `entity.scales` | — |
| Particle state | `particle.positions_inv_mass`, `particle.previous_positions`, `particle.velocities` | `particle.radii`, `particle.environment_indices`, `particle.kinds`, `particle.owner_types`, `particle.owner_indices`, `particle.strand_ids`, `particle.strand_roles`, `particle.owning_soft_body_indices`, `particle.material_indices`, `particle.fluid_material_indices`, `particle.phases`, `particle.collision_layers`, `particle.collision_masks` |
| Particle materials and adjacency | — | `particle.fluid_visuals`, `particle.contact_materials`, `particle.fluid_materials`, `particle.adjacency_offsets`, `particle.adjacency_counts`, `particle.adjacency_indices` |
| Soft-body topology and render data | — | `soft.edges`, `soft.bends`, `soft.tets`, `soft.render_positions`, `soft.render_normals`, `soft.world_aabbs` |
| Strand data | — | `strand.segments`, `strand.joints`, `strand.distance_constraints`, `strand.segment_states`, `strand.segment_joint_ranges`, `strand.segment_incident_joints` |
| Suturing data | — | `suturing.pairs`, `suturing.particle_refs`, `suturing.insertion_states`, `suturing.path_headers`, `suturing.path_nodes` |

The access mode in the returned descriptor is enforced when a pass is created.
For example, `rigid.positions` can be read or written, while many ownership,
material, and topology buffers are read-only. Use the prepared rigid, joint,
particle, and constraint mappings from {doc}`batched-environments` to interpret
IDs, environment indices, and packed ranges.

An engine-resource binding names both the HLSL variable and registry key. The
rigid lateral-shift example reads current rigid poses and writes kinematic
targets this way:

::::{tab-set}

:::{tab-item} C++

```cpp
CustomComputeResourceBindingDesc positionBinding{};
positionBinding.shaderVariableName = "g_RigidBodyPositionsInvMass";
positionBinding.resourceKey = "rigid.positions";
positionBinding.access = CustomComputeResourceAccess::ReadOnly;

CustomComputeResourceBindingDesc targetBinding{};
targetBinding.shaderVariableName = "g_RigidBodyKinematicTargetPositions";
targetBinding.resourceKey = "rigid.kinematic_target_positions";
targetBinding.access = CustomComputeResourceAccess::ReadWrite;
```

:::

:::{tab-item} Python

```python
position_binding = neo.CustomComputeResourceBindingDesc()
position_binding.shader_variable_name = "g_RigidBodyPositionsInvMass"
position_binding.resource_key = "rigid.positions"
position_binding.access = neo.CustomComputeResourceAccess.ReadOnly

target_binding = neo.CustomComputeResourceBindingDesc()
target_binding.shader_variable_name = "g_RigidBodyKinematicTargetPositions"
target_binding.resource_key = "rigid.kinematic_target_positions"
target_binding.access = neo.CustomComputeResourceAccess.ReadWrite
```

:::

::::

Every binding supplies exactly one source: `resourceKey`, `sharedBufferHandle`,
or `renderTargetBinding`. Engine resources are buffers; camera and ultrasound
images are bound as render-target textures.

## Render Targets in Compute

Use shader-readable explicit render targets to consume camera products in HLSL.
Bind a color or depth plane with `renderTargetBinding` and
`renderTargetTexturePlane`; a layered target is presented to HLSL as an array
texture. `render_target_torch_custom_compute.py` reads RGB and segmentation
layers into shared observation buffers after visual sensors have run. See
{download}`render_target_torch_custom_compute.py
<../../../examples/python/render_target_torch_custom_compute.py>`.

::::{tab-set}

:::{tab-item} C++

```cpp
CustomComputeResourceBindingDesc colorBinding{};
colorBinding.shaderVariableName = "g_ColorTarget";
colorBinding.renderTargetBinding.target = colorTarget;
colorBinding.renderTargetBinding.firstLayer = 0;
colorBinding.renderTargetBinding.layerCount = environmentCount;
colorBinding.renderTargetTexturePlane = gpu::GpuRenderTargetTexturePlane::Color;
colorBinding.access = CustomComputeResourceAccess::ReadOnly;
```

:::

:::{tab-item} Python

```python
color_binding = neo.CustomComputeResourceBindingDesc()
color_binding.shader_variable_name = "g_ColorTarget"
color_binding.render_target_binding = neo.GpuRenderTargetBinding()
color_binding.render_target_binding.target = color_target
color_binding.render_target_binding.first_layer = 0
color_binding.render_target_binding.layer_count = env_count
color_binding.render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
color_binding.access = neo.CustomComputeResourceAccess.ReadOnly
```

:::

::::

The HLSL declaration must match the target's texture shape and element type,
for example `Texture2DArray<float4> g_ColorTarget` for a layered color target.
The camera-output setup is described in {doc}`rendering-and-sensors`.

## Shared Buffers and PyTorch

`SharedBufferDesc` creates an engine-owned structured buffer for custom pass
inputs and outputs. Shader-resource and unordered-access flags determine
whether HLSL can read, write, or both. CUDA-enabled builds can export an
eligible shared buffer through DLPack, which Python can turn into a zero-copy
PyTorch tensor.

```python
tensor_desc = neo.SharedBufferTensorDesc()
tensor_desc.shape = [env_count, observation_size]
tensor_desc.dtype_code = neo.SharedBufferTensorDTypeCode.Float
tensor_desc.dtype_bits = 32
tensor = torch.utils.dlpack.from_dlpack(
    runtime.shared_buffer_to_dlpack(observation_buffer, tensor_desc)
)
```

```{note}
C++ integrations can obtain the lower-level raw CUDA view with
{cpp:func}`Runtime::tryGetSharedBufferCudaView <cressim::neo::engine::Runtime::tryGetSharedBufferCudaView>`.
The public runtime API does not provide a C++ DLPack convenience wrapper.
```

Before an engine pass reads values written by CUDA or PyTorch, call
`syncSharedBufferFromCuda`. After an engine pass writes a buffer for CUDA or
PyTorch, call `syncSharedBufferToCuda`. These operations require a buffer
successfully exported to and imported by CUDA; custom HLSL compute itself does
not require CUDA interoperability. The Python examples
{download}`custom_compute_constant_buffer_torch.py
<../../../examples/python/custom_compute_constant_buffer_torch.py>` and
{download}`render_target_torch_custom_compute.py
<../../../examples/python/render_target_torch_custom_compute.py>` show the
complete flow.

| C++ | Python |
| --- | --- |
| {cpp:func}`Runtime::createSharedBuffer <cressim::neo::engine::Runtime::createSharedBuffer>` · {cpp:func}`Runtime::syncSharedBufferToCuda <cressim::neo::engine::Runtime::syncSharedBufferToCuda>` · {cpp:func}`Runtime::syncSharedBufferFromCuda <cressim::neo::engine::Runtime::syncSharedBufferFromCuda>` | {py:meth}`Runtime.create_shared_buffer <cressim_neo.Runtime.create_shared_buffer>` · {py:meth}`Runtime.sync_shared_buffer_to_cuda <cressim_neo.Runtime.sync_shared_buffer_to_cuda>` · {py:meth}`Runtime.sync_shared_buffer_from_cuda <cressim_neo.Runtime.sync_shared_buffer_from_cuda>` |

## Lifecycle and Rebinding

After scene authoring, call `prepare()` and then `uploadWorld()` before calling
`listCustomComputeResources()` or `createCustomComputePass()`. The uploaded
world provides the internal GPU resources that a pass can bind. Place
`executeCustomComputePass()` in the stage where its inputs are current: before
physics for actions or resets, after physics for state-derived observations and
rewards, and after visual sensors for camera products. See
{doc}`runtime-lifecycle` for the full staged sequence.

Resource layouts can change when authoring changes require another
`prepare()`/`uploadWorld()` cycle. Compare `bindingGeneration` from
`listCustomComputeResources()` and recreate affected passes after a generation
changes; this prevents a pass from using stale GPU bindings. Destroy custom
passes and shared-buffer handles when their owning task or runtime is shut down.

| C++ | Python |
| --- | --- |
| {cpp:func}`Runtime::listCustomComputeResources <cressim::neo::engine::Runtime::listCustomComputeResources>` · {cpp:func}`Runtime::createCustomComputePass <cressim::neo::engine::Runtime::createCustomComputePass>` · {cpp:func}`Runtime::executeCustomComputePass <cressim::neo::engine::Runtime::executeCustomComputePass>` | {py:meth}`Runtime.list_custom_compute_resources <cressim_neo.Runtime.list_custom_compute_resources>` · {py:meth}`Runtime.create_custom_compute_pass <cressim_neo.Runtime.create_custom_compute_pass>` · {py:meth}`Runtime.execute_custom_compute_pass <cressim_neo.Runtime.execute_custom_compute_pass>` |
