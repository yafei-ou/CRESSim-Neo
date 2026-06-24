import struct

import cressim_neo as neo

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float, g_ActionShift);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RigidBodyKinematicTargetFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    uint bodyCount = 0u;
    uint stride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, stride);
    if (idx >= bodyCount)
    {
        return;
    }

    const float shift = CRESSIM_SB_LOAD(g_ActionShift, idx);
    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, idx);
    const float4 sourceOrientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, idx));

    float4 targetPosition = sourcePosition;
    targetPosition.x += shift;

    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetPositions, idx, targetPosition);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetOrientations, idx, sourceOrientation);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetFlags, idx, kKinematicTargetEnabled);
}
"""

POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observation);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Reward);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    uint bodyCount = 0u;
    uint stride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, stride);
    if (idx >= bodyCount)
    {
        return;
    }

    const float4 position = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, idx);
    CRESSIM_SB_STORE(g_Observation, idx, position.x);
    if (idx == 0u)
    {
        CRESSIM_SB_STORE(g_Reward, 0u, position.x + 0.1f);
    }
}
"""


def make_tensor(runtime: neo.Runtime, handle: neo.SharedBufferHandle, shape: list[int]) -> "torch.Tensor":
    desc = neo.SharedBufferTensorDesc()
    desc.shape = shape
    desc.dtype_code = neo.SharedBufferTensorDTypeCode.Float
    desc.dtype_bits = 32
    desc.dtype_lanes = 1
    return torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, desc))


def author_scene(runtime: neo.Runtime) -> int:
    world = runtime.world()
    entity = world.create_entity()

    transform = neo.TransformComponent()
    transform.world_transform.position = neo.Float3(0.0, 0.5, 0.0)
    world.set_transform(entity, transform)

    body = neo.RigidBodyComponent()
    body.body_type = neo.RigidBodyType.Kinematic
    body.inverse_mass = 1.0
    body.simulated = True
    world.set_rigid_body(entity, body)

    collider = neo.ColliderComponent()
    collider.shape_type = neo.ColliderShapeType.Box
    collider.shape_params = neo.Float4(0.25, 0.25, 0.25, 0.0)
    world.add_collider(entity, collider)
    return entity


def create_buffer(runtime: neo.Runtime, name: str, count: int) -> neo.SharedBufferHandle:
    desc = neo.SharedBufferDesc()
    desc.debug_name = name
    desc.element_stride_bytes = 4
    desc.element_count = count
    desc.access = neo.SharedBufferAccess.ReadWrite
    desc.bind_flags = neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
    handle = runtime.create_shared_buffer(desc)
    if not handle.is_valid():
        raise RuntimeError(f"Failed to create shared buffer: {name}")
    return handle


def create_pre_physics_pass(runtime: neo.Runtime, action_buffer: neo.SharedBufferHandle) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonTorchSharedBufferPrePhysics"
    desc.shader_source = PRE_PHYSICS_SHADER
    desc.thread_group_size_x = 64
    desc.resource_bindings = [
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
    ]

    desc.resource_bindings[0].shader_variable_name = "g_ActionShift"
    desc.resource_bindings[0].shared_buffer_handle = action_buffer
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_RigidBodyPositionsInvMass"
    desc.resource_bindings[1].resource_key = "rigid.positions"
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[2].shader_variable_name = "g_RigidBodyOrientations"
    desc.resource_bindings[2].resource_key = "rigid.orientations"
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[3].shader_variable_name = "g_RigidBodyKinematicTargetPositions"
    desc.resource_bindings[3].resource_key = "rigid.kinematic_target_positions"
    desc.resource_bindings[3].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[4].shader_variable_name = "g_RigidBodyKinematicTargetOrientations"
    desc.resource_bindings[4].resource_key = "rigid.kinematic_target_orientations"
    desc.resource_bindings[4].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[5].shader_variable_name = "g_RigidBodyKinematicTargetFlags"
    desc.resource_bindings[5].resource_key = "rigid.kinematic_target_flags"
    desc.resource_bindings[5].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ResourceElementCount
    desc.dispatch.count_resource_key = "rigid.positions"
    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create pre-physics custom compute pass.")
    return handle


def create_post_physics_pass(
    runtime: neo.Runtime,
    observation_buffer: neo.SharedBufferHandle,
    reward_buffer: neo.SharedBufferHandle,
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonTorchSharedBufferPostPhysics"
    desc.shader_source = POST_PHYSICS_SHADER
    desc.thread_group_size_x = 64
    desc.resource_bindings = [
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
    ]

    desc.resource_bindings[0].shader_variable_name = "g_RigidBodyPositionsInvMass"
    desc.resource_bindings[0].resource_key = "rigid.positions"
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_Observation"
    desc.resource_bindings[1].shared_buffer_handle = observation_buffer
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[2].shader_variable_name = "g_Reward"
    desc.resource_bindings[2].shared_buffer_handle = reward_buffer
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ResourceElementCount
    desc.dispatch.count_resource_key = "rigid.positions"
    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create post-physics custom compute pass.")
    return handle


def main() -> int:
    config = neo.RuntimeConfig()
    config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
    config.gpu_device_desc.enable_validation = False

    runtime = neo.Runtime()
    if not runtime.initialize(config):
        raise RuntimeError("Failed to initialize runtime.")

    author_scene(runtime)
    runtime.prepare()
    if not runtime.upload_world():
        raise RuntimeError("Failed to upload prepared world state.")

    body_count = 1
    action_buffer = create_buffer(runtime, "PythonTorchActions", body_count)
    observation_buffer = create_buffer(runtime, "PythonTorchObservations", body_count)
    reward_buffer = create_buffer(runtime, "PythonTorchReward", 1)

    action_tensor = make_tensor(runtime, action_buffer, [body_count])
    observation_tensor = make_tensor(runtime, observation_buffer, [body_count])
    reward_tensor = make_tensor(runtime, reward_buffer, [1])

    pre_physics = create_pre_physics_pass(runtime, action_buffer)
    post_physics = create_post_physics_pass(runtime, observation_buffer, reward_buffer)

    frame = neo.FrameContext()
    frame.delta_seconds = 1.0 / 60.0
    frame.frame_index = 0
    frame.time_seconds = 0.0

    action_tensor.fill_(0.35)

    try:
        if not runtime.sync_shared_buffer_from_cuda(action_buffer):
            raise RuntimeError("Failed to synchronize action buffer from CUDA to Vulkan.")
        if not runtime.execute_custom_compute_pass(pre_physics):
            raise RuntimeError("Failed to execute pre-physics pass.")
        if not runtime.step_physics(frame):
            raise RuntimeError("Physics step failed.")
        if not runtime.execute_custom_compute_pass(post_physics):
            raise RuntimeError("Failed to execute post-physics pass.")
        if not runtime.sync_shared_buffer_to_cuda(observation_buffer):
            raise RuntimeError("Failed to synchronize observation buffer from Vulkan to CUDA.")
        if not runtime.sync_shared_buffer_to_cuda(reward_buffer):
            raise RuntimeError("Failed to synchronize reward buffer from Vulkan to CUDA.")
        torch.cuda.synchronize()
        print("action:", action_tensor.cpu())
        print("observation:", observation_tensor.cpu())
        print("reward:", reward_tensor.cpu())
    finally:
        runtime.destroy_custom_compute_pass(pre_physics)
        runtime.destroy_custom_compute_pass(post_physics)
        runtime.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
