import math
import struct

import cressim_neo as neo

from readback_viewer import ColorFramePlayer


FRAME_COUNT = 24
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

CUSTOM_SHIFT_SHADER = """
#include "structured_buffer_compat.hlsli"
#include "physics/core/physics_math.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"

cbuffer CustomRigidLateralShiftConstants
{
    float lateralShift;
    float verticalLift;
    float padding0;
    float padding1;
};

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
    uint elementStride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, elementStride);
    if (idx >= bodyCount)
    {
        return;
    }

    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, idx);
    const float4 sourceOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, idx));

    float4 targetPosition = sourcePosition;
    targetPosition.x += lateralShift;
    targetPosition.y += verticalLift;

    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetPositions, idx, targetPosition);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetOrientations, idx, sourceOrientation);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetFlags, idx, kKinematicTargetEnabled);
}
"""


def _make_material(resources, name: str, color: neo.Float3, roughness: float):
    desc = neo.MaterialResourceDesc()
    desc.debug_name = name
    desc.base_color = color
    desc.roughness = roughness
    return resources.register_material(desc)


def _author_scene(runtime: neo.Runtime, target) -> neo.GpuRenderTargetBinding:
    world = runtime.world()
    resources = runtime.resources()

    plane_mesh = resources.register_mesh(
        neo.make_plane_mesh(8.0, "Python.CustomComputeGround", 2.0)
    )
    cube_mesh = resources.register_mesh(neo.make_cube_mesh(0.35, "Python.CustomComputeCube"))

    ground_material = _make_material(
        resources, "Python.CustomComputeGroundMaterial", neo.Float3(0.76, 0.74, 0.68), 0.95
    )
    body_material = _make_material(
        resources, "Python.CustomComputeBodyMaterial", neo.Float3(0.17, 0.49, 0.93), 0.28
    )

    light_entity = world.create_entity()
    light = neo.DirectionalLightComponent()
    light.direction = neo.Float3(-0.45, -1.0, 0.25)
    light.color = neo.Float3(1.0, 1.0, 1.0)
    light.intensity = 8.0
    world.set_directional_light(light_entity, light)

    camera_entity = world.create_entity()
    camera_transform = neo.TransformComponent()
    camera_transform.world_transform.position = neo.Float3(0.0, 3.0, -10.5)
    world.set_transform(camera_entity, camera_transform)

    camera = neo.CameraComponent()
    camera.vertical_fov_degrees = 42.0
    camera.clear_color = True
    camera.clear_depth = True
    camera.output.mode = neo.RenderOutputMode.ExplicitSurface
    camera.output.binding = neo.GpuRenderTargetBinding()
    camera.output.binding.target = target
    camera.output.binding.first_layer = 0
    camera.output.binding.layer_count = 1
    camera.output_width = TARGET_WIDTH
    camera.output_height = TARGET_HEIGHT
    camera.clear_color_value = neo.Float4(0.03, 0.03, 0.04, 1.0)
    world.set_camera(camera_entity, camera)

    ground_entity = world.create_entity()
    ground_transform = neo.TransformComponent()
    ground_transform.world_transform.position = neo.Float3(0.0, -0.3, 0.0)
    world.set_transform(ground_entity, ground_transform)

    ground_renderer = neo.MeshRendererComponent()
    ground_renderer.mesh = plane_mesh
    ground_renderer.material = ground_material
    world.set_mesh_renderer(ground_entity, ground_renderer)

    ground_body = neo.RigidBodyComponent()
    ground_body.body_type = neo.RigidBodyType.Static
    world.set_rigid_body(ground_entity, ground_body)

    ground_collider = neo.ColliderComponent()
    ground_collider.shape_type = neo.ColliderShapeType.Box
    ground_collider.shape_params = neo.Float4(8.0, 0.1, 8.0, 0.0)
    world.add_collider(ground_entity, ground_collider)

    for body_index in range(6):
        entity = world.create_entity()
        transform = neo.TransformComponent()
        transform.world_transform.position = neo.Float3(
            -3.0 + 1.2 * float(body_index), 0.8 + 0.18 * float(body_index % 2), 0.0
        )
        world.set_transform(entity, transform)

        renderer = neo.MeshRendererComponent()
        renderer.mesh = cube_mesh
        renderer.material = body_material
        world.set_mesh_renderer(entity, renderer)

        body = neo.RigidBodyComponent()
        body.body_type = neo.RigidBodyType.Kinematic
        body.inverse_mass = 1.0
        world.set_rigid_body(entity, body)

        collider = neo.ColliderComponent()
        collider.shape_type = neo.ColliderShapeType.Box
        collider.shape_params = neo.Float4(0.35, 0.35, 0.35, 0.0)
        world.add_collider(entity, collider)

    return camera.output.binding


def _make_custom_pass(runtime: neo.Runtime) -> neo.CustomComputePassHandle:
    for resource in runtime.list_custom_compute_resources():
        print(
            resource.key,
            "count=",
            resource.element_count,
            "stride=",
            resource.element_stride_bytes,
            "generation=",
            resource.binding_generation,
        )

    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonRigidLateralShift"
    desc.shader_source = CUSTOM_SHIFT_SHADER
    desc.thread_group_size_x = 64
    desc.resource_bindings = [
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
        neo.CustomComputeResourceBindingDesc(),
    ]

    desc.resource_bindings[0].shader_variable_name = "g_RigidBodyPositionsInvMass"
    desc.resource_bindings[0].resource_key = "rigid.positions"
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_RigidBodyOrientations"
    desc.resource_bindings[1].resource_key = "rigid.orientations"
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[2].shader_variable_name = "g_RigidBodyKinematicTargetPositions"
    desc.resource_bindings[2].resource_key = "rigid.kinematic_target_positions"
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[3].shader_variable_name = "g_RigidBodyKinematicTargetOrientations"
    desc.resource_bindings[3].resource_key = "rigid.kinematic_target_orientations"
    desc.resource_bindings[3].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[4].shader_variable_name = "g_RigidBodyKinematicTargetFlags"
    desc.resource_bindings[4].resource_key = "rigid.kinematic_target_flags"
    desc.resource_bindings[4].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ResourceElementCount
    desc.dispatch.count_resource_key = "rigid.positions"
    desc.constant_buffer_variable_name = "CustomRigidLateralShiftConstants"
    desc.constant_buffer_size_bytes = 16
    desc.constant_data = list(struct.pack("4f", 0.0, 0.0, 0.0, 0.0))

    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create custom compute pass.")
    return handle


def main() -> int:
    config = neo.RuntimeConfig()
    config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
    config.gpu_device_desc.enable_validation = False

    runtime = neo.Runtime()
    if not runtime.initialize(config):
        raise RuntimeError("Failed to initialize runtime.")

    target_desc = neo.GpuRenderTargetDesc()
    target_desc.width = TARGET_WIDTH
    target_desc.height = TARGET_HEIGHT
    target_desc.array_size = 1
    target_desc.layered_rendering = False
    target_desc.debug_name = "PythonCustomComputeTarget"
    target = runtime.create_render_target(target_desc)

    binding = _author_scene(runtime, target)
    runtime.prepare()
    if not runtime.upload_world():
        raise RuntimeError("Failed to upload prepared world state.")
    custom_pass = _make_custom_pass(runtime)

    readbacks = []
    frame = neo.FrameContext()
    frame.delta_seconds = 1.0 / 30.0

    try:
        for frame_index in range(FRAME_COUNT):
            frame.frame_index = frame_index
            frame.time_seconds = frame_index * frame.delta_seconds

            shift = 0.22 * math.sin(frame.time_seconds * 2.8)
            constants = struct.pack("4f", shift, 0.0, 0.0, 0.0)
            if not runtime.update_custom_compute_pass_constants(custom_pass, constants):
                raise RuntimeError("Failed to update custom compute constants.")
            if not runtime.execute_custom_compute_pass(custom_pass):
                raise RuntimeError("Failed to execute custom compute pass.")
            if not runtime.step_physics(frame):
                raise RuntimeError("Physics step failed.")
            runtime.step_visual_sensors(frame)

            request = runtime.request_render_target_readback(binding)
            runtime.end_frame(frame)
            event = None
            while event is None:
                event = runtime.try_get_render_target_readback(request)
            readbacks.append(event)
    finally:
        runtime.destroy_custom_compute_pass(custom_pass)
        runtime.shutdown()

    ColorFramePlayer("Custom Compute Rigid Lateral Shift", interval=0.12).show(readbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
