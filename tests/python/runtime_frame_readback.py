import sys
import struct

import cressim_neo as neo
import matplotlib.pyplot as plt
import numpy as np


def _rgb_image_from_event(event) -> np.ndarray:
    color_bytes = bytes(event.color_bytes)
    stride = event.color_row_stride_bytes
    if stride == 0:
        raise RuntimeError("Readback event does not provide a valid color row stride.")

    if event.color_format == neo.TextureFormat.RGBA16Float:
        image = np.empty((event.color_height, event.color_width, 3), dtype=np.float32)
        for row_index in range(event.color_height):
            row_start = row_index * stride
            for pixel_index in range(event.color_width):
                src = row_start + pixel_index * 8
                r, g, b, _ = struct.unpack_from("<4e", color_bytes, src)
                image[row_index, pixel_index, 0] = max(0.0, r)
                image[row_index, pixel_index, 1] = max(0.0, g)
                image[row_index, pixel_index, 2] = max(0.0, b)

        # Readbacks default to scene-linear HDR. Apply a lightweight tonemap and gamma transform
        # so matplotlib shows a useful preview instead of an almost-black image.
        image = image / (1.0 + image)
        return np.power(np.clip(image, 0.0, 1.0), 1.0 / 2.2)

    if event.color_format in (
        neo.TextureFormat.BGRA8Unorm,
        neo.TextureFormat.BGRA8UnormSrgb,
    ):
        image = np.empty((event.color_height, event.color_width, 3), dtype=np.uint8)
        for row_index in range(event.color_height):
            row_start = row_index * stride
            row_end = row_start + (event.color_width * 4)
            bgra_row = color_bytes[row_start:row_end]
            for pixel_index in range(event.color_width):
                src = pixel_index * 4
                image[row_index, pixel_index, 0] = bgra_row[src + 2]
                image[row_index, pixel_index, 1] = bgra_row[src + 1]
                image[row_index, pixel_index, 2] = bgra_row[src + 0]
        return image

    if event.color_format not in (
        neo.TextureFormat.RGBA8Unorm,
        neo.TextureFormat.RGBA8UnormSrgb,
    ):
        raise RuntimeError(f"Unsupported readback color format for display: {event.color_format}")

    image = np.empty((event.color_height, event.color_width, 3), dtype=np.uint8)
    for row_index in range(event.color_height):
        row_start = row_index * stride
        row_end = row_start + (event.color_width * 4)
        rgba_row = color_bytes[row_start:row_end]
        for pixel_index in range(event.color_width):
            src = pixel_index * 4
            image[row_index, pixel_index, 0] = rgba_row[src + 0]
            image[row_index, pixel_index, 1] = rgba_row[src + 1]
            image[row_index, pixel_index, 2] = rgba_row[src + 2]
    return image


def _show_event_image(event) -> None:
    image = _rgb_image_from_event(event)
    center_pixel = image[event.color_height // 2, event.color_width // 2]
    print("Center pixel:", center_pixel)
    plt.figure("CRESSim-Neo Frame Readback")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def main() -> int:
    runtime = neo.Runtime()
    if not runtime.initialize():
        print("Skipping Python frame readback test because runtime initialization failed.")
        return 77

    try:
        world = runtime.world()
        resources = runtime.resources()

        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = 320
        target_desc.height = 240
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.debug_name = "Python.FrameReadback.Target"
        target = runtime.create_render_target(target_desc)
        if not runtime.is_valid_render_target(target):
            raise RuntimeError("Failed to create explicit-surface render target.")

        mesh = resources.register_mesh(neo.make_cube_mesh(0.6, "Python.FrameReadback.Cube"))

        material_desc = neo.MaterialResourceDesc()
        material_desc.debug_name = "Python.FrameReadback.Material"
        material_desc.base_color = neo.Float3(1.0, 0.15, 0.1)
        material_desc.metallic = 0.0
        material_desc.roughness = 0.45
        material = resources.register_material(material_desc)

        cube_entity = world.create_entity()
        cube_transform = neo.TransformComponent()
        cube_transform.world_transform.position = neo.Float3(0.0, 0.0, 0.5)
        world.set_transform(cube_entity, cube_transform)

        cube_renderer = neo.MeshRendererComponent()
        cube_renderer.mesh = mesh
        cube_renderer.material = material
        cube_renderer.visible = True
        world.set_mesh_renderer(cube_entity, cube_renderer)

        camera_entity = world.create_entity()
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(0.0, 0.0, -4.0)
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 52.0
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = target
        camera.output.binding.first_layer = 0
        camera.output.binding.layer_count = 1
        camera.output_width = target_desc.width
        camera.output_height = target_desc.height
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(0.02, 0.02, 0.03, 1.0)
        world.set_camera(camera_entity, camera)

        light_entity = world.create_entity()
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.35, -0.45, 1.0)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 4.0
        light.casts_shadows = False
        world.set_directional_light(light_entity, light)

        frame = neo.FrameContext()
        frame.delta_seconds = 1.0 / 60.0

        request = None
        for frame_index in range(2):
            runtime.prepare()
            request = runtime.request_render_target_readback(camera.output.binding)
            if request.id == 0:
                raise RuntimeError("Failed to queue render-target readback request.")

            frame.frame_index = frame_index
            frame.time_seconds = frame_index * frame.delta_seconds
            runtime.step_physics(frame)
            runtime.step_visual_sensors(frame)
            runtime.end_frame(frame)

        event = runtime.try_get_render_target_readback(request)
        if event is None:
            raise RuntimeError("Expected a render-target readback event.")
        if event.color_width == 0 or event.color_height == 0 or len(event.color_bytes) == 0:
            raise RuntimeError("Expected a non-empty color payload in the readback event.")
        if not any(event.color_bytes):
            raise RuntimeError("Expected rendered frame to contain non-black pixels.")

        print(
            "Python frame readback passed:",
            f"{event.color_width}x{event.color_height}",
            f"bytes={len(event.color_bytes)}",
        )
        _show_event_image(event)
        return 0
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
