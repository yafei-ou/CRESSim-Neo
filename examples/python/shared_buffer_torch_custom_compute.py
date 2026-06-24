import cressim_neo as neo
import time

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


DONE_THRESHOLD = 0.8
ENV_COUNT = 4
STEP_COUNT = 4


PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float, g_ActionShift);
CRESSIM_STRUCTURED_BUFFER(uint, g_ControlledBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RigidBodyKinematicTargetFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint envStride = 0u;
    g_ActionShift.GetDimensions(envCount, envStride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint bodyIndex = CRESSIM_SB_LOAD(g_ControlledBodyIndices, envIndex);
    uint bodyCount = 0u;
    uint stride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, stride);
    if (bodyIndex >= bodyCount)
    {
        return;
    }

    const float shift = CRESSIM_SB_LOAD(g_ActionShift, envIndex);
    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, bodyIndex);
    const float4 sourceOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, bodyIndex));

    float4 targetPosition = sourcePosition;
    targetPosition.x += shift;

    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetPositions, bodyIndex, targetPosition);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetOrientations, bodyIndex, sourceOrientation);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetFlags, bodyIndex, kKinematicTargetEnabled);
}
"""


POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

static const float kDoneThreshold = 0.8f;

CRESSIM_STRUCTURED_BUFFER(uint, g_ControlledBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observation);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Reward);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Done);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint envStride = 0u;
    g_Observation.GetDimensions(envCount, envStride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint bodyIndex = CRESSIM_SB_LOAD(g_ControlledBodyIndices, envIndex);
    uint bodyCount = 0u;
    uint stride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, stride);
    if (bodyIndex >= bodyCount)
    {
        return;
    }

    const float x = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, bodyIndex).x;
    CRESSIM_SB_STORE(g_Observation, envIndex, x);
    CRESSIM_SB_STORE(g_Reward, envIndex, x);
    CRESSIM_SB_STORE(g_Done, envIndex, x >= kDoneThreshold ? 1u : 0u);
}
"""


def make_tensor(
    runtime: neo.Runtime,
    handle: neo.SharedBufferHandle,
    shape: list[int],
    dtype_code: neo.SharedBufferTensorDTypeCode,
) -> "torch.Tensor":
    desc = neo.SharedBufferTensorDesc()
    desc.shape = shape
    desc.dtype_code = dtype_code
    desc.dtype_bits = 32
    desc.dtype_lanes = 1
    return torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, desc))


def create_buffer(
    runtime: neo.Runtime,
    name: str,
    count: int,
    access: neo.SharedBufferAccess = neo.SharedBufferAccess.ReadWrite,
    bind_flags: neo.SharedBufferBindFlags = (
        neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
    ),
) -> neo.SharedBufferHandle:
    desc = neo.SharedBufferDesc()
    desc.debug_name = name
    desc.element_stride_bytes = 4
    desc.element_count = count
    desc.access = access
    desc.bind_flags = bind_flags
    handle = runtime.create_shared_buffer(desc)
    if not handle.is_valid():
        raise RuntimeError(f"Failed to create shared buffer: {name}")
    return handle


def author_env(world: neo.World, env_index: int, z_offset: float) -> int:
    ground_entity = world.create_entity(env_index)
    ground_transform = neo.TransformComponent()
    ground_transform.world_transform.position = neo.Float3(0.0, 0.0, z_offset)
    world.set_transform(ground_entity, ground_transform)

    ground_body = neo.RigidBodyComponent()
    ground_body.body_type = neo.RigidBodyType.Static
    ground_body.simulated = True
    world.set_rigid_body(ground_entity, ground_body)

    ground_collider = neo.ColliderComponent()
    ground_collider.shape_type = neo.ColliderShapeType.Box
    ground_collider.shape_params = neo.Float4(0.75, 0.1, 0.75, 0.0)
    world.add_collider(ground_entity, ground_collider)

    box_entity = world.create_entity(env_index)
    box_transform = neo.TransformComponent()
    box_transform.world_transform.position = neo.Float3(0.0, 0.5, z_offset)
    world.set_transform(box_entity, box_transform)

    box_body = neo.RigidBodyComponent()
    box_body.body_type = neo.RigidBodyType.Kinematic
    box_body.inverse_mass = 1.0
    box_body.simulated = True
    world.set_rigid_body(box_entity, box_body)

    box_collider = neo.ColliderComponent()
    box_collider.shape_type = neo.ColliderShapeType.Box
    box_collider.shape_params = neo.Float4(0.25, 0.25, 0.25, 0.0)
    world.add_collider(box_entity, box_collider)
    return box_entity


def create_pre_physics_pass(
    runtime: neo.Runtime,
    action_buffer: neo.SharedBufferHandle,
    controlled_body_buffer: neo.SharedBufferHandle,
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonTorchSharedBufferPrePhysicsMultiEnv"
    desc.shader_source = PRE_PHYSICS_SHADER
    desc.thread_group_size_x = 64
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(7)]

    desc.resource_bindings[0].shader_variable_name = "g_ActionShift"
    desc.resource_bindings[0].shared_buffer_handle = action_buffer
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_ControlledBodyIndices"
    desc.resource_bindings[1].shared_buffer_handle = controlled_body_buffer
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[2].shader_variable_name = "g_RigidBodyPositionsInvMass"
    desc.resource_bindings[2].resource_key = "rigid.positions"
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[3].shader_variable_name = "g_RigidBodyOrientations"
    desc.resource_bindings[3].resource_key = "rigid.orientations"
    desc.resource_bindings[3].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[4].shader_variable_name = "g_RigidBodyKinematicTargetPositions"
    desc.resource_bindings[4].resource_key = "rigid.kinematic_target_positions"
    desc.resource_bindings[4].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[5].shader_variable_name = "g_RigidBodyKinematicTargetOrientations"
    desc.resource_bindings[5].resource_key = "rigid.kinematic_target_orientations"
    desc.resource_bindings[5].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[6].shader_variable_name = "g_RigidBodyKinematicTargetFlags"
    desc.resource_bindings[6].resource_key = "rigid.kinematic_target_flags"
    desc.resource_bindings[6].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
    desc.dispatch.group_count_x = (ENV_COUNT + 63) // 64

    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create pre-physics custom compute pass.")
    return handle


def create_post_physics_pass(
    runtime: neo.Runtime,
    controlled_body_buffer: neo.SharedBufferHandle,
    observation_buffer: neo.SharedBufferHandle,
    reward_buffer: neo.SharedBufferHandle,
    done_buffer: neo.SharedBufferHandle,
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonTorchSharedBufferPostPhysicsMultiEnv"
    desc.shader_source = POST_PHYSICS_SHADER
    desc.thread_group_size_x = 64
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(5)]

    desc.resource_bindings[0].shader_variable_name = "g_ControlledBodyIndices"
    desc.resource_bindings[0].shared_buffer_handle = controlled_body_buffer
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_RigidBodyPositionsInvMass"
    desc.resource_bindings[1].resource_key = "rigid.positions"
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[2].shader_variable_name = "g_Observation"
    desc.resource_bindings[2].shared_buffer_handle = observation_buffer
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[3].shader_variable_name = "g_Reward"
    desc.resource_bindings[3].shared_buffer_handle = reward_buffer
    desc.resource_bindings[3].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[4].shader_variable_name = "g_Done"
    desc.resource_bindings[4].shared_buffer_handle = done_buffer
    desc.resource_bindings[4].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
    desc.dispatch.group_count_x = (ENV_COUNT + 63) // 64

    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create post-physics custom compute pass.")
    return handle


def main() -> int:
    config = neo.RuntimeConfig()
    config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
    config.gpu_device_desc.enable_validation = False
    config.scene_layout.env_count = ENV_COUNT

    runtime = neo.Runtime()
    if not runtime.initialize(config):
        raise RuntimeError("Failed to initialize runtime.")

    box_entities: list[int] = []
    for env_index in range(ENV_COUNT):
        box_entities.append(author_env(runtime.world(), env_index, env_index * 2.0))

    runtime.prepare()
    mapping = runtime.get_prepared_rigid_layout_mapping()
    if mapping.rigid_body_count == 0:
        raise RuntimeError("Rigid layout mapping is empty.")

    controlled_body_buffer = create_buffer(runtime, "PythonTorchControlledBodies", ENV_COUNT)
    action_buffer = create_buffer(runtime, "PythonTorchActions", ENV_COUNT)
    observation_buffer = create_buffer(runtime, "PythonTorchObservations", ENV_COUNT)
    reward_buffer = create_buffer(runtime, "PythonTorchRewards", ENV_COUNT)
    done_buffer = create_buffer(runtime, "PythonTorchDone", ENV_COUNT)

    controlled_body_tensor = make_tensor(
        runtime,
        controlled_body_buffer,
        [ENV_COUNT],
        neo.SharedBufferTensorDTypeCode.UInt,
    )
    action_tensor = make_tensor(
        runtime,
        action_buffer,
        [ENV_COUNT],
        neo.SharedBufferTensorDTypeCode.Float,
    )
    observation_tensor = make_tensor(
        runtime,
        observation_buffer,
        [ENV_COUNT],
        neo.SharedBufferTensorDTypeCode.Float,
    )
    reward_tensor = make_tensor(
        runtime,
        reward_buffer,
        [ENV_COUNT],
        neo.SharedBufferTensorDTypeCode.Float,
    )
    done_tensor = make_tensor(
        runtime,
        done_buffer,
        [ENV_COUNT],
        neo.SharedBufferTensorDTypeCode.UInt,
    )

    controlled_body_data = [mapping.rigid_body_count] * ENV_COUNT
    for body_slot, entity_id in enumerate(mapping.rigid_body_entity_ids):
        env_index = mapping.rigid_body_environment_indices[body_slot]
        if entity_id == box_entities[env_index]:
            if controlled_body_data[env_index] != mapping.rigid_body_count:
                raise RuntimeError(f"Found multiple controlled rigid bodies for env {env_index}.")
            controlled_body_data[env_index] = body_slot

    if any(body_index >= mapping.rigid_body_count for body_index in controlled_body_data):
        raise RuntimeError(
            f"Expected exactly one controlled rigid body per env, got {controlled_body_data}."
        )

    controlled_body_tensor.copy_(
        torch.tensor(
            controlled_body_data,
            device=controlled_body_tensor.device,
            dtype=controlled_body_tensor.dtype,
        )
    )
    if not runtime.sync_shared_buffer_from_cuda(controlled_body_buffer):
        raise RuntimeError("Failed to synchronize controlled-body lookup buffer from CUDA to Vulkan.")

    if not runtime.upload_world():
        raise RuntimeError("Failed to upload prepared world state.")

    pre_physics = create_pre_physics_pass(runtime, action_buffer, controlled_body_buffer)
    post_physics = create_post_physics_pass(
        runtime,
        controlled_body_buffer,
        observation_buffer,
        reward_buffer,
        done_buffer,
    )

    frame = neo.FrameContext()
    frame.delta_seconds = 1.0 / 60.0
    frame.frame_index = 0
    frame.time_seconds = 0.0

    base_actions = torch.linspace(0.1, 0.4, ENV_COUNT, device=action_tensor.device, dtype=action_tensor.dtype)

    try:
        for step_index in range(STEP_COUNT):
            action_tensor.copy_(base_actions)

            step_start = time.perf_counter()
            if not runtime.sync_shared_buffer_from_cuda(action_buffer):
                raise RuntimeError("Failed to synchronize action buffer from CUDA to Vulkan.")
            after_action_sync = time.perf_counter()
            if not runtime.execute_custom_compute_pass(pre_physics):
                raise RuntimeError("Failed to execute pre-physics pass.")
            after_pre_physics = time.perf_counter()
            if not runtime.step_physics(frame):
                raise RuntimeError("Physics step failed.")
            after_physics = time.perf_counter()
            if not runtime.execute_custom_compute_pass(post_physics):
                raise RuntimeError("Failed to execute post-physics pass.")
            after_post_physics = time.perf_counter()
            if not runtime.sync_shared_buffer_to_cuda(observation_buffer):
                raise RuntimeError("Failed to synchronize observation buffer from Vulkan to CUDA.")
            if not runtime.sync_shared_buffer_to_cuda(reward_buffer):
                raise RuntimeError("Failed to synchronize reward buffer from Vulkan to CUDA.")
            if not runtime.sync_shared_buffer_to_cuda(done_buffer):
                raise RuntimeError("Failed to synchronize done buffer from Vulkan to CUDA.")
            after_output_sync = time.perf_counter()

            torch.cuda.synchronize()
            after_cuda_sync = time.perf_counter()
            print(f"step {step_index}")
            print(
                "  timings_ms:",
                {
                    "action_sync": round((after_action_sync - step_start) * 1000.0, 3),
                    "pre_physics": round((after_pre_physics - after_action_sync) * 1000.0, 3),
                    "physics": round((after_physics - after_pre_physics) * 1000.0, 3),
                    "post_physics": round((after_post_physics - after_physics) * 1000.0, 3),
                    "output_sync": round((after_output_sync - after_post_physics) * 1000.0, 3),
                    "cuda_sync": round((after_cuda_sync - after_output_sync) * 1000.0, 3),
                    "total": round((after_cuda_sync - step_start) * 1000.0, 3),
                },
            )
            print("  action:", action_tensor.cpu())
            print("  observation:", observation_tensor.cpu())
            print("  reward:", reward_tensor.cpu())
            print("  done:", done_tensor.cpu())

            frame.frame_index += 1
            frame.time_seconds += frame.delta_seconds
    finally:
        runtime.destroy_custom_compute_pass(pre_physics)
        runtime.destroy_custom_compute_pass(post_physics)
        runtime.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
