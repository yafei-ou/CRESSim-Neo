import math

import matplotlib.pyplot as plt
import numpy as np

import cressim_neo as neo
from readback_viewer import color_image_from_event


ENV_COUNT = 4
FRAME_COUNT = 300
TARGET_WIDTH = 320
TARGET_HEIGHT = 240

GRID_WIDTH = 10
GRID_DEPTH = 10
LAYERS = 5
SPACING = 1.4
BASE_HEIGHT = 1.5
LAYER_HEIGHT = 2.0
ENV_WORLD_SPACING = 72.0
GROUND_HALF_EXTENT = 28.0


def _env_origin(env_index: int, env_count: int) -> neo.Float3:
    columns = max(1, int(math.ceil(math.sqrt(float(env_count)))))
    rows = max(1, (env_count + columns - 1) // columns)
    col = env_index % columns
    row = env_index // columns
    x_center = 0.5 * float(columns - 1)
    z_center = 0.5 * float(rows - 1)
    return neo.Float3(
        (float(col) - x_center) * ENV_WORLD_SPACING,
        0.0,
        (float(row) - z_center) * ENV_WORLD_SPACING,
    )


def _palette_for_env(env_index: int):
    palettes = [
        (
            neo.Float3(0.85, 0.28, 0.18),
            neo.Float3(0.18, 0.58, 0.90),
            neo.Float3(0.28, 0.82, 0.36),
            neo.Float3(0.72, 0.74, 0.77),
        ),
        (
            neo.Float3(0.95, 0.56, 0.14),
            neo.Float3(0.14, 0.76, 0.90),
            neo.Float3(0.56, 0.34, 0.88),
            neo.Float3(0.80, 0.75, 0.63),
        ),
        (
            neo.Float3(0.82, 0.18, 0.48),
            neo.Float3(0.12, 0.66, 0.44),
            neo.Float3(0.94, 0.82, 0.20),
            neo.Float3(0.64, 0.74, 0.84),
        ),
        (
            neo.Float3(0.26, 0.46, 0.92),
            neo.Float3(0.92, 0.24, 0.24),
            neo.Float3(0.20, 0.72, 0.66),
            neo.Float3(0.66, 0.68, 0.78),
        ),
        (
            neo.Float3(0.84, 0.34, 0.16),
            neo.Float3(0.38, 0.52, 0.94),
            neo.Float3(0.16, 0.78, 0.30),
            neo.Float3(0.76, 0.70, 0.60),
        ),
        (
            neo.Float3(0.72, 0.20, 0.20),
            neo.Float3(0.20, 0.76, 0.88),
            neo.Float3(0.58, 0.30, 0.82),
            neo.Float3(0.70, 0.78, 0.70),
        ),
    ]
    return palettes[env_index % len(palettes)]


def _make_material(resources, name: str, color: neo.Float3, roughness: float, metallic: float = 0.0):
    desc = neo.MaterialResourceDesc()
    desc.debug_name = name
    desc.base_color = color
    desc.roughness = roughness
    desc.metallic = metallic
    return resources.register_material(desc)


def _collider_params(shape_type):
    if shape_type == neo.ColliderShapeType.Sphere:
        return neo.Float4(0.45, 0.0, 0.0, 0.0)
    if shape_type == neo.ColliderShapeType.Capsule:
        return neo.Float4(0.28, 0.52, 0.0, 0.0)
    return neo.Float4(0.45, 0.45, 0.45, 0.0)


def _mesh_for_shape(shape_type, cube_mesh, sphere_mesh, capsule_mesh):
    if shape_type == neo.ColliderShapeType.Sphere:
        return sphere_mesh
    if shape_type == neo.ColliderShapeType.Capsule:
        return capsule_mesh
    return cube_mesh


def _material_for_shape(shape_type, materials):
    if shape_type == neo.ColliderShapeType.Sphere:
        return materials["sphere"]
    if shape_type == neo.ColliderShapeType.Capsule:
        return materials["capsule"]
    return materials["box"]


def _author_camera(world, env_index: int, env_origin: neo.Float3, target):
    entity = world.create_entity(env_index)
    transform = neo.TransformComponent()
    transform.world_transform.position = neo.Float3(env_origin.x, 5.0, env_origin.z - 26.0)
    world.set_transform(entity, transform)

    camera = neo.CameraComponent()
    camera.vertical_fov_degrees = 55.0
    camera.clear_color = True
    camera.clear_depth = True
    camera.render_order = env_index
    camera.output.mode = neo.RenderOutputMode.ExplicitSurface
    camera.output.binding = neo.GpuRenderTargetBinding()
    camera.output.binding.target = target
    camera.output.binding.first_layer = env_index
    camera.output.binding.layer_count = 1
    camera.output_width = TARGET_WIDTH
    camera.output_height = TARGET_HEIGHT
    camera.clear_color_value = neo.Float4(0.02, 0.02, 0.03, 1.0)
    world.set_camera(entity, camera)
    return camera.output.binding


def _author_ground(world, env_index: int, env_origin: neo.Float3, plane_mesh, ground_material):
    entity = world.create_entity(env_index)
    transform = neo.TransformComponent()
    transform.world_transform.position = neo.Float3(env_origin.x, -1.0, env_origin.z)
    world.set_transform(entity, transform)

    renderer = neo.MeshRendererComponent()
    renderer.mesh = plane_mesh
    renderer.material = ground_material
    world.set_mesh_renderer(entity, renderer)

    body = neo.RigidBodyComponent()
    body.body_type = neo.RigidBodyType.Static
    world.set_rigid_body(entity, body)

    collider = neo.ColliderComponent()
    collider.shape_type = neo.ColliderShapeType.Box
    collider.shape_params = neo.Float4(GROUND_HALF_EXTENT, 0.05, GROUND_HALF_EXTENT, 0.0)
    collider.friction = 0.1
    collider.static_friction = 0.2
    collider.restitution = 0.5
    world.add_collider(entity, collider)


def _author_light(world, env_index: int, env_phase: float):
    entity = world.create_entity(env_index)
    light = neo.DirectionalLightComponent()
    light.direction = neo.Float3(
        -0.45 + 0.08 * math.sin(env_phase),
        -1.0,
        0.35 + 0.08 * math.cos(env_phase),
    )
    light.color = neo.Float3(1.0, 1.0, 1.0)
    light.intensity = 8.0
    world.set_directional_light(entity, light)


def _author_dynamic_array(
    world,
    env_index: int,
    env_count: int,
    env_origin: neo.Float3,
    cube_mesh,
    sphere_mesh,
    capsule_mesh,
    materials,
):
    env_phase = float(env_index) * 0.45
    env_velocity_bias_x = math.cos(env_phase) * 0.05
    env_velocity_bias_z = math.sin(env_phase) * 0.05
    env_angular_bias = 0.20 + 0.05 * float(env_index % 5)
    env_restitution = 0.2 * float(env_index % 4)
    env_friction = 0.05 + 0.15 * float(env_index % 4)

    x_origin = -0.5 * float(GRID_WIDTH - 1) * SPACING
    z_origin = -0.5 * float(GRID_DEPTH - 1) * SPACING

    for layer in range(LAYERS):
        for z in range(GRID_DEPTH):
            for x in range(GRID_WIDTH):
                shape_index = (x + z + layer + env_index) % 3
                if shape_index == 0:
                    shape_type = neo.ColliderShapeType.Box
                elif shape_index == 1:
                    shape_type = neo.ColliderShapeType.Sphere
                else:
                    shape_type = neo.ColliderShapeType.Capsule

                entity = world.create_entity(env_index)
                transform = neo.TransformComponent()
                transform.world_transform.position = neo.Float3(
                    env_origin.x + x_origin + float(x) * SPACING,
                    BASE_HEIGHT
                    + float(layer) * LAYER_HEIGHT
                    + (0.0 if ((x + z + env_index) % 2 == 0) else 0.25)
                    + 0.05 * float(env_count % 3),
                    env_origin.z + z_origin + float(z) * SPACING,
                )
                world.set_transform(entity, transform)

                renderer = neo.MeshRendererComponent()
                renderer.mesh = _mesh_for_shape(shape_type, cube_mesh, sphere_mesh, capsule_mesh)
                renderer.material = _material_for_shape(shape_type, materials)
                world.set_mesh_renderer(entity, renderer)

                body = neo.RigidBodyComponent()
                body.inverse_mass = 1.0
                body.inverse_inertia_local = neo.Float3(1.0, 1.0, 1.0)
                body.linear_velocity = neo.Float3(
                    float((x % 3) - 1) * 0.08 + env_velocity_bias_x,
                    0.0,
                    float((z % 3) - 1) * 0.08 + env_velocity_bias_z,
                )
                body.angular_velocity = neo.Float3(0.0, env_angular_bias, 0.0)
                world.set_rigid_body(entity, body)

                collider = neo.ColliderComponent()
                collider.shape_type = shape_type
                collider.shape_params = _collider_params(shape_type)
                collider.friction = env_friction
                collider.static_friction = env_friction + 0.1
                collider.restitution = env_restitution
                world.add_collider(entity, collider)


def _stitch_images(images):
    columns = max(1, int(math.ceil(math.sqrt(len(images)))))
    rows = max(1, (len(images) + columns - 1) // columns)
    blank = np.zeros_like(images[0])
    grid_rows = []
    for row in range(rows):
        row_images = []
        for col in range(columns):
            index = row * columns + col
            row_images.append(images[index] if index < len(images) else blank)
        grid_rows.append(np.concatenate(row_images, axis=1))
    return np.concatenate(grid_rows, axis=0)


def main() -> int:
    runtime = neo.Runtime()

    config = neo.RuntimeConfig()
    config.scene_layout.env_count = ENV_COUNT
    config.scene_layout.max_renderable_objects_per_env = GRID_WIDTH * GRID_DEPTH * LAYERS + 16
    config.scene_layout.max_lights_per_env = 4
    config.scene_layout.max_cameras_per_env = 1
    config.physics_desc.rigid_rigid_contact_iterations = 20

    if not runtime.initialize(config):
        print("Skipping large-array Python scene because runtime initialization failed.")
        return 77

    try:
        world = runtime.world()
        resources = runtime.resources()

        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = TARGET_WIDTH
        target_desc.height = TARGET_HEIGHT
        target_desc.array_size = ENV_COUNT
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.debug_name = "Python.LargeArray.Target"
        target = runtime.create_render_target(target_desc)
        if not runtime.is_valid_render_target(target):
            raise RuntimeError("Failed to create large-array render target.")

        cube_mesh = resources.register_mesh(neo.make_cube_mesh(0.45, "LargeArray.CubeMesh"))
        plane_mesh = resources.register_mesh(neo.make_plane_mesh(GROUND_HALF_EXTENT, "LargeArray.PlaneMesh"))
        sphere_mesh = resources.register_mesh(neo.make_sphere_mesh(0.45, 20, 12, "LargeArray.SphereMesh"))
        capsule_mesh = resources.register_mesh(
            neo.make_capsule_mesh(0.28, 0.52, 20, 6, 2, "LargeArray.CapsuleMesh")
        )

        camera_bindings = []
        for env_index in range(ENV_COUNT):
            box_color, sphere_color, capsule_color, ground_color = _palette_for_env(env_index)
            materials = {
                "box": _make_material(resources, f"LargeArray.Box.{env_index}", box_color, 0.55),
                "sphere": _make_material(
                    resources, f"LargeArray.Sphere.{env_index}", sphere_color, 0.35, 0.08
                ),
                "capsule": _make_material(
                    resources, f"LargeArray.Capsule.{env_index}", capsule_color, 0.45, 0.02
                ),
                "ground": _make_material(resources, f"LargeArray.Ground.{env_index}", ground_color, 0.90),
            }

            env_origin = _env_origin(env_index, ENV_COUNT)
            camera_bindings.append(_author_camera(world, env_index, env_origin, target))
            _author_ground(world, env_index, env_origin, plane_mesh, materials["ground"])
            _author_light(world, env_index, float(env_index) * 0.45)
            _author_dynamic_array(
                world,
                env_index,
                ENV_COUNT,
                env_origin,
                cube_mesh,
                sphere_mesh,
                capsule_mesh,
                materials,
            )

        frame = neo.FrameContext()
        frame.delta_seconds = 1.0 / 60.0

        stitched_frames = []
        for frame_index in range(FRAME_COUNT):
            runtime.prepare()
            if not runtime.upload_world():
                raise RuntimeError("Failed to upload prepared world state.")
            requests = []
            for binding in camera_bindings:
                request = runtime.request_render_target_readback(binding)
                if request.id == 0:
                    raise RuntimeError("Failed to queue large-array render-target readback request.")
                requests.append(request)

            frame.frame_index = frame_index
            frame.time_seconds = frame_index * frame.delta_seconds
            runtime.step_physics(frame)
            runtime.step_visual_sensors(frame)
            runtime.end_frame(frame)

            images = []
            for request in requests:
                event = runtime.try_get_render_target_readback(request)
                if event is None:
                    raise RuntimeError("Expected a render-target readback event for large-array scene.")
                images.append(color_image_from_event(event))
            stitched_frames.append(_stitch_images(images))

        plt.ion()
        figure, axes = plt.subplots(num="CRESSim-Neo Large Array")
        artist = axes.imshow(stitched_frames[0])
        axes.axis("off")
        for frame_index, image in enumerate(stitched_frames):
            artist.set_data(image)
            axes.set_title(f"Large Array Frame {frame_index}")
            figure.canvas.draw_idle()
            plt.pause(0.02)
        plt.ioff()
        plt.show()

        stats = runtime.last_render_stats()
        print(
            "Large-array Python scene passed:",
            f"frames={FRAME_COUNT}",
            f"envs={ENV_COUNT}",
            f"draw_calls={stats.draw_calls}",
        )
        return 0
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
