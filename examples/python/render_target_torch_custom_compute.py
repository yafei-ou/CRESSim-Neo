import cressim_neo as neo
import math

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


ENV_COUNT = 4
TARGET_WIDTH = 256
TARGET_HEIGHT = 256


IMAGE_OBSERVATION_SHADER = r"""
#include "structured_buffer_compat.hlsli"

Texture2DArray<float4> g_ColorTarget;
Texture2DArray<uint> g_SegmentationTarget;
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ColorObservation);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SegmentationObservation);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    uint layers = 0u;
    g_ColorTarget.GetDimensions(width, height, layers);

    const uint x = dispatchThreadID.x;
    const uint y = dispatchThreadID.y;
    const uint envIndex = dispatchThreadID.z;
    if (x >= width || y >= height || envIndex >= layers)
    {
        return;
    }

    const uint pixelIndex = envIndex * width * height + y * width + x;

    const float4 color = saturate(g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0)));

    const uint segmentation =
        g_SegmentationTarget.Load(int4(int(x), int(y), int(envIndex), 0));

    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex, color);
    CRESSIM_SB_STORE(g_SegmentationObservation, pixelIndex, segmentation);
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
    runtime: neo.Runtime, name: str, count: int, element_stride_bytes: int
) -> neo.SharedBufferHandle:
    desc = neo.SharedBufferDesc()
    desc.debug_name = name
    desc.element_stride_bytes = element_stride_bytes
    desc.element_count = count
    desc.access = neo.SharedBufferAccess.ReadWrite
    desc.bind_flags = (
        neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
    )
    handle = runtime.create_shared_buffer(desc)
    if not handle.is_valid():
        raise RuntimeError(f"Failed to create shared buffer: {name}")
    return handle


def create_material(resources, env_index: int) -> neo.MaterialHandle:
    palette = (
        neo.Float3(0.95, 0.25, 0.20),
        neo.Float3(0.20, 0.75, 0.30),
        neo.Float3(0.20, 0.45, 0.95),
        neo.Float3(0.95, 0.75, 0.20),
    )
    material_desc = neo.MaterialResourceDesc()
    material_desc.debug_name = f"Python.RenderTargetTorch.Material.{env_index}"
    material_desc.base_color = palette[env_index % len(palette)]
    material_desc.metallic = 0.0
    material_desc.roughness = 0.45
    return resources.register_material(material_desc)


def quaternion_from_euler_degrees(x_degrees: float, y_degrees: float, z_degrees: float) -> neo.Quaternion:
    x = math.radians(x_degrees) * 0.5
    y = math.radians(y_degrees) * 0.5
    z = math.radians(z_degrees) * 0.5

    sin_x = math.sin(x)
    cos_x = math.cos(x)
    sin_y = math.sin(y)
    cos_y = math.cos(y)
    sin_z = math.sin(z)
    cos_z = math.cos(z)

    return neo.Quaternion(
        sin_z * cos_x * cos_y - cos_z * sin_x * sin_y,
        cos_z * sin_x * cos_y + sin_z * cos_x * sin_y,
        cos_z * cos_x * sin_y - sin_z * sin_x * cos_y,
        cos_z * cos_x * cos_y + sin_z * sin_x * sin_y,
    )


def author_env(
    runtime: neo.Runtime,
    env_index: int,
    color_target: neo.GpuRenderTargetHandle,
    segmentation_target: neo.GpuRenderTargetHandle,
    mesh: neo.MeshHandle,
    material: neo.MaterialHandle,
) -> None:
    world = runtime.world()

    ground_material_desc = neo.MaterialResourceDesc()
    ground_material_desc.debug_name = f"Python.RenderTargetTorch.GroundMaterial.{env_index}"
    ground_material_desc.base_color = neo.Float3(0.62, 0.62, 0.66)
    ground_material_desc.metallic = 0.0
    ground_material_desc.roughness = 0.9
    ground_material = runtime.resources().register_material(ground_material_desc)

    ground_entity = world.create_entity(env_index)
    ground_transform = neo.TransformComponent()
    ground_transform.world_transform.position = neo.Float3(0.0, -1.0, 5.4)
    ground_transform.world_transform.scale = neo.Float3(4.6, 0.18, 4.6)
    world.set_transform(ground_entity, ground_transform)

    ground_renderer = neo.MeshRendererComponent()
    ground_renderer.mesh = mesh
    ground_renderer.material = ground_material
    ground_renderer.segmentation_id = 20 + env_index
    ground_renderer.visible = True
    world.set_mesh_renderer(ground_entity, ground_renderer)

    cube_entity = world.create_entity(env_index)
    cube_transform = neo.TransformComponent()
    cube_transform.world_transform.position = neo.Float3(0.35, -0.05, 4.1)
    cube_transform.world_transform.rotation = quaternion_from_euler_degrees(22.0, -28.0, 8.0)
    world.set_transform(cube_entity, cube_transform)

    cube_renderer = neo.MeshRendererComponent()
    cube_renderer.mesh = mesh
    cube_renderer.material = material
    cube_renderer.segmentation_id = env_index + 1
    cube_renderer.visible = True
    world.set_mesh_renderer(cube_entity, cube_renderer)

    light_entity = world.create_entity(env_index)
    light = neo.DirectionalLightComponent()
    light.direction = neo.Float3(-0.55, -1.0, 0.25)
    light.color = neo.Float3(1.0, 1.0, 1.0)
    light.intensity = 7.5
    light.casts_shadows = False
    world.set_directional_light(light_entity, light)

    base_camera_transform = neo.TransformComponent()
    base_camera_transform.world_transform.position = neo.Float3(0.0, 1.9, -7.2)
    base_camera_transform.world_transform.rotation = quaternion_from_euler_degrees(0.0, 0.0, 0.0)

    color_camera_entity = world.create_entity(env_index)
    world.set_transform(color_camera_entity, base_camera_transform)
    color_camera = neo.CameraComponent()
    color_camera.product = neo.CameraProduct.ColorDepth
    color_camera.vertical_fov_degrees = 52.0
    color_camera.output.mode = neo.RenderOutputMode.ExplicitSurface
    color_camera.output.binding = neo.GpuRenderTargetBinding()
    color_camera.output.binding.target = color_target
    color_camera.output.binding.first_layer = env_index
    color_camera.output.binding.layer_count = 1
    color_camera.output_width = TARGET_WIDTH
    color_camera.output_height = TARGET_HEIGHT
    color_camera.clear_color = True
    color_camera.clear_depth = True
    color_camera.clear_color_value = neo.Float4(0.03, 0.03, 0.05, 1.0)
    world.set_camera(color_camera_entity, color_camera)

    segmentation_camera_entity = world.create_entity(env_index)
    world.set_transform(segmentation_camera_entity, base_camera_transform)
    segmentation_camera = neo.CameraComponent()
    segmentation_camera.product = neo.CameraProduct.SegmentationDepth
    segmentation_camera.vertical_fov_degrees = 52.0
    segmentation_camera.output.mode = neo.RenderOutputMode.ExplicitSurface
    segmentation_camera.output.binding = neo.GpuRenderTargetBinding()
    segmentation_camera.output.binding.target = segmentation_target
    segmentation_camera.output.binding.first_layer = env_index
    segmentation_camera.output.binding.layer_count = 1
    segmentation_camera.output_width = TARGET_WIDTH
    segmentation_camera.output_height = TARGET_HEIGHT
    segmentation_camera.clear_color = True
    segmentation_camera.clear_depth = True
    segmentation_camera.clear_color_value = neo.Float4(0.0, 0.0, 0.0, 1.0)
    world.set_camera(segmentation_camera_entity, segmentation_camera)


def create_image_observation_pass(
    runtime: neo.Runtime,
    color_target: neo.GpuRenderTargetHandle,
    segmentation_target: neo.GpuRenderTargetHandle,
    color_buffer: neo.SharedBufferHandle,
    segmentation_buffer: neo.SharedBufferHandle,
) -> neo.CustomComputePassHandle:
    desc = neo.CustomComputePassDesc()
    desc.debug_name = "PythonRenderTargetTorchImageObservation"
    desc.shader_source = IMAGE_OBSERVATION_SHADER
    desc.thread_group_size_x = 8
    desc.thread_group_size_y = 8
    desc.thread_group_size_z = 1
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(4)]

    desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
    desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
    desc.resource_bindings[0].render_target_binding.target = color_target
    desc.resource_bindings[0].render_target_binding.first_layer = 0
    desc.resource_bindings[0].render_target_binding.layer_count = ENV_COUNT
    desc.resource_bindings[0].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
    desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[1].shader_variable_name = "g_SegmentationTarget"
    desc.resource_bindings[1].render_target_binding = neo.GpuRenderTargetBinding()
    desc.resource_bindings[1].render_target_binding.target = segmentation_target
    desc.resource_bindings[1].render_target_binding.first_layer = 0
    desc.resource_bindings[1].render_target_binding.layer_count = ENV_COUNT
    desc.resource_bindings[1].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
    desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadOnly

    desc.resource_bindings[2].shader_variable_name = "g_ColorObservation"
    desc.resource_bindings[2].shared_buffer_handle = color_buffer
    desc.resource_bindings[2].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.resource_bindings[3].shader_variable_name = "g_SegmentationObservation"
    desc.resource_bindings[3].shared_buffer_handle = segmentation_buffer
    desc.resource_bindings[3].access = neo.CustomComputeResourceAccess.ReadWrite

    desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
    desc.dispatch.group_count_x = (TARGET_WIDTH + 7) // 8
    desc.dispatch.group_count_y = (TARGET_HEIGHT + 7) // 8
    desc.dispatch.group_count_z = ENV_COUNT

    handle = runtime.create_custom_compute_pass(desc)
    if not handle.is_valid():
        raise RuntimeError("Failed to create render observation custom compute pass.")
    return handle


def show_observations(rgb_tensor: "torch.Tensor", segmentation_tensor: "torch.Tensor") -> None:
    rgb_images = np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)
    segmentation_ids = segmentation_tensor.detach().cpu().numpy().astype(np.uint32)
    palette = np.zeros((64, 3), dtype=np.float32)
    palette[1] = np.array([0.90, 0.20, 0.18], dtype=np.float32)
    palette[2] = np.array([0.18, 0.72, 0.30], dtype=np.float32)
    palette[3] = np.array([0.20, 0.42, 0.92], dtype=np.float32)
    palette[4] = np.array([0.95, 0.72, 0.18], dtype=np.float32)
    palette[20] = np.array([0.45, 0.45, 0.48], dtype=np.float32)
    palette[21] = np.array([0.52, 0.52, 0.56], dtype=np.float32)
    palette[22] = np.array([0.60, 0.60, 0.64], dtype=np.float32)
    palette[23] = np.array([0.68, 0.68, 0.72], dtype=np.float32)
    segmentation_images = palette[np.clip(segmentation_ids, 0, palette.shape[0] - 1)]

    figure, axes = plt.subplots(2, ENV_COUNT, figsize=(4 * ENV_COUNT, 8), squeeze=False)
    for env_index in range(ENV_COUNT):
        axes[0, env_index].imshow(rgb_images[env_index])
        axes[0, env_index].set_title(f"Env {env_index} RGB")
        axes[0, env_index].axis("off")

        axes[1, env_index].imshow(segmentation_images[env_index])
        axes[1, env_index].set_title(f"Env {env_index} Segmentation")
        axes[1, env_index].axis("off")

    figure.tight_layout()
    plt.show()


def create_render_target(
    runtime: neo.Runtime,
    debug_name: str,
    color_format: neo.TextureFormat,
) -> neo.GpuRenderTargetHandle:
    target_desc = neo.GpuRenderTargetDesc()
    target_desc.width = TARGET_WIDTH
    target_desc.height = TARGET_HEIGHT
    target_desc.array_size = ENV_COUNT
    target_desc.color = True
    target_desc.depth = True
    target_desc.layered_rendering = True
    target_desc.shader_readable = True
    target_desc.color_format = color_format
    target_desc.debug_name = debug_name
    target = runtime.create_render_target(target_desc)
    if not runtime.is_valid_render_target(target):
        raise RuntimeError(f"Failed to create explicit render target: {debug_name}")
    return target


def main() -> int:
    config = neo.RuntimeConfig()
    config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
    config.gpu_device_desc.enable_validation = False
    config.scene_layout.env_count = ENV_COUNT

    runtime = neo.Runtime()
    if not runtime.initialize(config):
        raise RuntimeError("Failed to initialize runtime.")

    try:
        color_target = create_render_target(
            runtime,
            "Python.RenderTargetTorchCustomCompute.ColorTarget",
            neo.TextureFormat.RGBA8Unorm,
        )
        segmentation_target = create_render_target(
            runtime,
            "Python.RenderTargetTorchCustomCompute.SegmentationTarget",
            neo.TextureFormat.R32Uint,
        )

        resources = runtime.resources()
        mesh = resources.register_mesh(neo.make_cube_mesh(0.6, "Python.RenderTargetTorch.Cube"))

        for env_index in range(ENV_COUNT):
            material = create_material(resources, env_index)
            author_env(runtime, env_index, color_target, segmentation_target, mesh, material)

        runtime.prepare()

        pixel_count = ENV_COUNT * TARGET_WIDTH * TARGET_HEIGHT
        color_buffer = create_buffer(
            runtime,
            "PythonRenderTargetTorchColorObservation",
            pixel_count,
            16,
        )
        segmentation_buffer = create_buffer(
            runtime,
            "PythonRenderTargetTorchSegmentationObservation",
            pixel_count,
            4,
        )

        color_tensor = make_tensor(
            runtime,
            color_buffer,
            [ENV_COUNT, TARGET_HEIGHT, TARGET_WIDTH, 4],
            neo.SharedBufferTensorDTypeCode.Float,
        )
        segmentation_tensor = make_tensor(
            runtime,
            segmentation_buffer,
            [ENV_COUNT, TARGET_HEIGHT, TARGET_WIDTH],
            neo.SharedBufferTensorDTypeCode.UInt,
        )

        if not runtime.upload_world():
            raise RuntimeError("Failed to upload prepared world state.")

        render_pass = create_image_observation_pass(
            runtime,
            color_target,
            segmentation_target,
            color_buffer,
            segmentation_buffer,
        )

        frame = neo.FrameContext()
        frame.delta_seconds = 1.0 / 60.0
        frame.frame_index = 0
        frame.time_seconds = 0.0

        runtime.step_visual_sensors(frame)
        if not runtime.execute_custom_compute_pass(render_pass):
            raise RuntimeError("Failed to execute render observation pass.")
        if not runtime.sync_shared_buffer_to_cuda(color_buffer):
            raise RuntimeError("Failed to synchronize color buffer from Vulkan to CUDA.")
        if not runtime.sync_shared_buffer_to_cuda(segmentation_buffer):
            raise RuntimeError("Failed to synchronize segmentation buffer from Vulkan to CUDA.")
        torch.cuda.synchronize()
        runtime.end_frame(frame)

        center_ids = segmentation_tensor[:, TARGET_HEIGHT // 2, TARGET_WIDTH // 2].cpu()
        print("center segmentation ids:", center_ids)
        show_observations(color_tensor, segmentation_tensor)
        return 0
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
