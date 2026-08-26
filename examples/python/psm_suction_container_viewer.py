import math
from pathlib import Path
import cressim_neo as neo
from cressim_neo_envs.psm_builder import (
    PsmAuthoringConfig,
    PsmBuildResult,
    author_psm_scene,
    set_psm_joint_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


SCENE_SCALE = 2.5
PSM_SCALE = 10.0 * SCENE_SCALE
PSM_MASS_SCALE = 0.02
PSM_INERTIA_SCALE = 0.02
PSM_INSERTION_BASE_TARGET = 0.08 * PSM_SCALE
PSM_INSERTION_AMPLITUDE = 0.1 * PSM_SCALE
PSM_INSERTION_FREQUENCY_HZ = 0.20
PSM_TOOL_YAW_ZERO_POSE_Z_PER_SCALE = -0.4864
FLUID_PARTICLE_RADIUS = 0.09
FLUID_HORIZONTAL_FILL_FRACTION = 1.0
FLUID_HEIGHT_FILL_FRACTION = 1.0
CONTAINER_OPENING_CENTER_X_LOCAL = -0.105369
CONTAINER_OPENING_CENTER_Z_LOCAL = -0.174073
CONTAINER_OPENING_TOP_Y_LOCAL = -0.01413111111111111
CONTAINER_BOTTOM_Y_LOCAL = -0.569843
CONTAINER_ALIGN_X = 0.0
CONTAINER_ALIGN_Z = 0.0
CONTAINER_TOP_Y = 1.10
CONTAINER_STATIC_BAND_HEIGHT = 0.06
CONTAINER_PARTICLE_MASS = 0.12
CONTAINER_PARTICLE_RADIUS = 0.30
CONTAINER_EDGE_COMPLIANCE = 0.0
CONTAINER_VOLUME_COMPLIANCE = 8.0e-4
CONTAINER_CONTACT_FRICTION = 0.55
CONTAINER_CONTACT_STATIC_FRICTION = 0.75
CONTAINER_CONTACT_RESTITUTION = 0.05
CONTAINER_CONTACT_DAMPING = 0.8
GROUND_HALF = neo.Float3(5.6 * SCENE_SCALE, 0.08 * SCENE_SCALE, 2.4 * SCENE_SCALE)
GROUND_CLEARANCE = 0.01 * SCENE_SCALE
FLUID_DROP_GAP_Y = 0.10 * SCENE_SCALE
FLUID_DROP_EXTRA_Z = 0.0
CAMERA_POSITION = (0.0, 2.8 * SCENE_SCALE, -6.0 * SCENE_SCALE)
PSM_WORLD_OFFSET = (
    0.0,
    CONTAINER_TOP_Y * SCENE_SCALE + 0.5 * SCENE_SCALE,
    -PSM_TOOL_YAW_ZERO_POSE_Z_PER_SCALE * PSM_SCALE,
)


def _compute_regular_grid_axis(
    inner_half_extent: float,
    particle_radius: float,
    fill_fraction: float,
) -> tuple[int, float]:
    spacing = 2.0 * particle_radius
    max_count = max(
        1,
        int(math.floor((2.0 * max(0.0, inner_half_extent - particle_radius)) / spacing)) + 1,
    )
    count = max(1, min(max_count, int(math.floor(max_count * fill_fraction + 0.5))))
    return count, float(count) * spacing


def _compute_fluid_block_desc(
    block_half_extents: neo.Float3,
    particle_radius: float,
    fill_fraction_xy: float,
    fill_fraction_height: float,
) -> tuple[neo.Float3, float]:
    spacing = 2.0 * particle_radius
    _, size_x = _compute_regular_grid_axis(block_half_extents.x, particle_radius, fill_fraction_xy)
    _, size_z = _compute_regular_grid_axis(block_half_extents.z, particle_radius, fill_fraction_xy)
    _, size_y = _compute_regular_grid_axis(block_half_extents.y, particle_radius, fill_fraction_height)
    return neo.Float3(size_x, size_y, size_z), spacing


def _register_material(
    resources: neo.RenderResourceManager,
    debug_name: str,
    base_color: neo.Float3,
    roughness: float,
) -> int:
    material = neo.MaterialResourceDesc()
    material.debug_name = debug_name
    material.base_color = base_color
    material.metallic = 0.0
    material.roughness = roughness
    return resources.register_material(material)


def _container_entity_world_position(top_y: float) -> tuple[float, float, float]:
    return (
        CONTAINER_ALIGN_X - CONTAINER_OPENING_CENTER_X_LOCAL * SCENE_SCALE,
        top_y - CONTAINER_OPENING_TOP_Y_LOCAL * SCENE_SCALE,
        CONTAINER_ALIGN_Z - CONTAINER_OPENING_CENTER_Z_LOCAL * SCENE_SCALE,
    )


def _container_bottom_world_y(top_y: float) -> float:
    return top_y - CONTAINER_OPENING_TOP_Y_LOCAL * SCENE_SCALE + CONTAINER_BOTTOM_Y_LOCAL * SCENE_SCALE


def _look_rotation(position: tuple[float, float, float], target: tuple[float, float, float]) -> neo.Quaternion:
    px, py, pz = position
    tx, ty, tz = target
    forward = [tx - px, ty - py, tz - pz]
    forward_len = math.sqrt(sum(v * v for v in forward))
    if forward_len <= 1.0e-8:
        return neo.Quaternion()
    forward = [v / forward_len for v in forward]
    up = [0.0, 1.0, 0.0]
    right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    ]
    right_len = math.sqrt(sum(v * v for v in right))
    if right_len <= 1.0e-8:
        right = [1.0, 0.0, 0.0]
    else:
        right = [v / right_len for v in right]
    corrected_up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ]
    m00, m01, m02 = right[0], corrected_up[0], forward[0]
    m10, m11, m12 = right[1], corrected_up[1], forward[1]
    m20, m21, m22 = right[2], corrected_up[2], forward[2]
    trace = m00 + m11 + m22
    q = neo.Quaternion()
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q.w = 0.25 * s
        q.x = (m21 - m12) / s
        q.y = (m02 - m20) / s
        q.z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q.w = (m21 - m12) / s
        q.x = 0.25 * s
        q.y = (m01 + m10) / s
        q.z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q.w = (m02 - m20) / s
        q.x = (m01 + m10) / s
        q.y = 0.25 * s
        q.z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q.w = (m10 - m01) / s
        q.x = (m02 + m20) / s
        q.y = (m12 + m21) / s
        q.z = 0.25 * s
    return q


def _load_obj_mesh(path: Path, debug_name: str, *, reverse_winding: bool = False) -> neo.MeshResourceDesc:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    vertices: list[neo.MeshVertex] = []
    indices: list[int] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0] == "v" and len(tokens) >= 4:
                positions.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif tokens[0] == "vn" and len(tokens) >= 4:
                normals.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif tokens[0] == "vt" and len(tokens) >= 3:
                texcoords.append((float(tokens[1]), float(tokens[2])))
            elif tokens[0] == "f" and len(tokens) >= 4:
                face_vertices = tokens[1:]
                for tri_idx in range(1, len(face_vertices) - 1):
                    tri = (face_vertices[0], face_vertices[tri_idx], face_vertices[tri_idx + 1])
                    if reverse_winding:
                        tri = (tri[0], tri[2], tri[1])
                    for face_token in tri:
                        parts = face_token.split("/")
                        pos_idx = int(parts[0]) - 1
                        tex_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                        normal_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None
                        vertex = neo.MeshVertex()
                        px, py, pz = positions[pos_idx]
                        vertex.position = neo.Float3(px, py, pz)
                        if normal_idx is not None and 0 <= normal_idx < len(normals):
                            nx, ny, nz = normals[normal_idx]
                            vertex.normal = neo.Float3(nx, ny, nz)
                        else:
                            vertex.normal = neo.Float3(0.0, 1.0, 0.0)
                        if tex_idx is not None and 0 <= tex_idx < len(texcoords):
                            u, v = texcoords[tex_idx]
                            vertex.tex_coord_u = u
                            vertex.tex_coord_v = 1.0 - v
                        indices.append(len(vertices))
                        vertices.append(vertex)

    mesh = neo.MeshResourceDesc()
    mesh.debug_name = debug_name
    mesh.vertices = vertices
    mesh.indices = indices
    return mesh


def _bottom_static_particle_indices(node_path: Path, band_height: float) -> list[int]:
    with node_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    node_count = int(lines[0].split()[0])
    entries = [lines[i + 1].split() for i in range(node_count)]
    y_values = [float(entry[2]) for entry in entries]
    min_y = min(y_values)
    threshold = min_y + band_height
    return [index for index, entry in enumerate(entries) if float(entry[2]) <= threshold]


def _author_container(runtime: neo.Runtime) -> None:
    world = runtime.world()
    resources = runtime.resources()
    models_dir = REPO_ROOT / "examples" / "models"
    node_path = models_dir / "container.node"
    ele_path = models_dir / "container.ele"
    surface_path = models_dir / "container_surface.obj"

    soft_entity = world.create_entity(0)
    soft_transform = neo.TransformComponent()
    soft_transform.world_transform.position = neo.Float3(
        *_container_entity_world_position(CONTAINER_TOP_Y * SCENE_SCALE)
    )
    soft_transform.world_transform.scale = neo.Float3(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)
    world.set_transform(soft_entity, soft_transform)

    soft_body = neo.SoftBodyComponent()
    soft_body.source.kind = neo.SoftBodySourceKind.TetGenFiles
    soft_body.source.tet_gen.node_file = str(node_path)
    soft_body.source.tet_gen.ele_file = str(ele_path)
    soft_body.source.tet_gen.static_particle_indices = _bottom_static_particle_indices(
        node_path,
        CONTAINER_STATIC_BAND_HEIGHT * SCENE_SCALE,
    )
    soft_body.particle_mass = CONTAINER_PARTICLE_MASS
    soft_body.particle_radius = CONTAINER_PARTICLE_RADIUS
    soft_body.edge_compliance = CONTAINER_EDGE_COMPLIANCE
    soft_body.volume_compliance = CONTAINER_VOLUME_COMPLIANCE
    soft_body.self_collision_enabled = False
    soft_body.supports_suturing = False
    soft_body.collision_layer = 0x1
    soft_body.collision_mask = 0xFFFFFFFF
    soft_body.material.contact.friction = CONTAINER_CONTACT_FRICTION
    soft_body.material.contact.static_friction = CONTAINER_CONTACT_STATIC_FRICTION
    soft_body.material.contact.restitution = CONTAINER_CONTACT_RESTITUTION
    soft_body.material.contact.damping = CONTAINER_CONTACT_DAMPING
    if not world.set_soft_body(soft_entity, soft_body):
        raise RuntimeError("Failed to author TetGen soft-body container.")

    container_material = _register_material(
        resources,
        "FluidIsolation.ContainerMaterial",
        neo.Float3(0.76, 0.47, 0.32),
        0.88,
    )
    renderer = neo.MeshRendererComponent()
    renderer.mesh = resources.register_mesh(
        _load_obj_mesh(surface_path, "FluidIsolation.ContainerMesh", reverse_winding=True)
    )
    renderer.material = container_material
    renderer.visible = True
    world.set_mesh_renderer(soft_entity, renderer)


def _author_psm(runtime: neo.Runtime) -> PsmBuildResult:
    world = runtime.world()
    resources = runtime.resources()
    urdf_path = REPO_ROOT / "examples" / "models" / "psm" / "psm_suction_irrigator.urdf"
    build = author_psm_scene(
        world,
        resources,
        PsmAuthoringConfig(
            resolve_root=REPO_ROOT,
            urdf_path=urdf_path,
            tool_type="suction_irrigator",
            env_count=1,
            add_ground=False,
            add_default_lighting=False,
            add_default_camera=False,
            global_scale=PSM_SCALE,
        ),
    )
    set_psm_joint_targets(
        world,
        build,
        [0.0, 0.0, PSM_INSERTION_BASE_TARGET, 0.0, 0.0, 0.0, 0.0],
    )
    offset_x, offset_y, offset_z = PSM_WORLD_OFFSET
    translated_entities: set[int] = set()
    for instance in build.instances:
        for entity in instance.link_entities.values():
            if entity in translated_entities:
                continue
            translated_entities.add(entity)
            transform = world.try_get_transform(entity)
            if transform is None:
                continue
            transform.world_transform.position = neo.Float3(
                transform.world_transform.position.x + offset_x,
                transform.world_transform.position.y + offset_y,
                transform.world_transform.position.z + offset_z,
            )
            world.set_transform(entity, transform)
            rigid_body = world.try_get_rigid_body(entity)
            if rigid_body is None or rigid_body.body_type != neo.RigidBodyType.Dynamic:
                continue
            if rigid_body.inverse_mass > 0.0:
                mass = 1.0 / rigid_body.inverse_mass
                scaled_mass = max(mass * PSM_MASS_SCALE, 1.0e-6)
                rigid_body.inverse_mass = 1.0 / scaled_mass
            rigid_body.inverse_inertia_local = neo.Float3(
                rigid_body.inverse_inertia_local.x / PSM_INERTIA_SCALE
                if rigid_body.inverse_inertia_local.x > 0.0
                else 0.0,
                rigid_body.inverse_inertia_local.y / PSM_INERTIA_SCALE
                if rigid_body.inverse_inertia_local.y > 0.0
                else 0.0,
                rigid_body.inverse_inertia_local.z / PSM_INERTIA_SCALE
                if rigid_body.inverse_inertia_local.z > 0.0
                else 0.0,
            )
            world.set_rigid_body(entity, rigid_body)
    return build


def _author_scene(runtime: neo.Runtime) -> tuple[int, PsmBuildResult]:
    world = runtime.world()
    resources = runtime.resources()

    fluid_block_half = neo.Float3(0.8, 0.8, 0.8)
    fluid_size, fluid_spacing = _compute_fluid_block_desc(
        fluid_block_half,
        FLUID_PARTICLE_RADIUS,
        FLUID_HORIZONTAL_FILL_FRACTION,
        FLUID_HEIGHT_FILL_FRACTION,
    )
    container_center_x, _, container_center_z = _container_entity_world_position(CONTAINER_TOP_Y * SCENE_SCALE)

    ground_mesh = resources.register_mesh(neo.make_box_mesh(GROUND_HALF, "FluidIsolation.GroundMesh"))
    ground_material = _register_material(
        resources,
        "FluidIsolation.GroundMaterial",
        neo.Float3(0.62, 0.64, 0.68),
        0.92,
    )

    ground_entity = world.create_entity(0)
    ground_transform = neo.TransformComponent()
    ground_y = _container_bottom_world_y(CONTAINER_TOP_Y * SCENE_SCALE) - GROUND_HALF.y - GROUND_CLEARANCE
    ground_transform.world_transform.position = neo.Float3(0.0, ground_y, 0.0)
    world.set_transform(ground_entity, ground_transform)
    ground_body = neo.RigidBodyComponent()
    ground_body.body_type = neo.RigidBodyType.Static
    ground_body.inverse_mass = 0.0
    world.set_rigid_body(ground_entity, ground_body)
    ground_collider = neo.ColliderComponent()
    ground_collider.shape_type = neo.ColliderShapeType.Box
    ground_collider.shape_params = neo.Float4(GROUND_HALF.x, GROUND_HALF.y, GROUND_HALF.z, 0.0)
    world.add_collider(ground_entity, ground_collider)
    ground_renderer = neo.MeshRendererComponent()
    ground_renderer.mesh = ground_mesh
    ground_renderer.material = ground_material
    ground_renderer.visible = True
    world.set_mesh_renderer(ground_entity, ground_renderer)

    psm_build = _author_psm(runtime)
    _author_container(runtime)

    fluid_entity = world.create_entity(0)
    fluid_transform = neo.TransformComponent()
    fluid_transform.world_transform.position = neo.Float3(
        container_center_x,
        CONTAINER_TOP_Y * SCENE_SCALE + FLUID_DROP_GAP_Y + 0.5 * fluid_size.y,
        container_center_z + FLUID_DROP_EXTRA_Z,
    )
    world.set_transform(fluid_entity, fluid_transform)
    fluid = neo.FluidComponent()
    fluid.source.kind = neo.FluidSourceKind.RegularGrid
    fluid.source.regular_grid.size = fluid_size
    fluid.source.regular_grid.target_particle_spacing = fluid_spacing
    fluid.particle_radius = FLUID_PARTICLE_RADIUS
    fluid.material = neo.FluidMaterialDesc()
    fluid.material.contact = neo.ParticleContactMaterialDesc()
    fluid.material.contact.friction = 0.04
    fluid.material.contact.static_friction = 0.06
    fluid.material.contact.restitution = 0.0
    fluid.material.contact.damping = 0.2
    fluid.material.viscosity = 1.5
    fluid.material.cohesion = 0.8
    fluid.material.gravity_scale = 0.75
    fluid.material.cfl_coefficient = 1.0
    fluid.material.vorticity_confinement = 0.25
    fluid.material.surface_tension = 1.5
    particle_diameter = 2.0 * fluid.particle_radius
    fluid.particle_mass = particle_diameter * particle_diameter * particle_diameter * 10.0
    fluid.visual_color = neo.Float4(0.90, 0.16, 0.16, 0.80)
    if not world.set_fluid(fluid_entity, fluid):
        raise RuntimeError("Failed to author fluid body.")

    fluid_visuals = neo.EnvironmentFluidDesc()
    fluid_visuals.smoothness = 0.95
    fluid_visuals.specular = neo.Float3(0.38, 0.44, 0.50)
    fluid_visuals.fresnel = 0.84
    fluid_visuals.depth_edge_threshold = 0.18
    fluid_visuals.filter_radius_pixels = 4.0
    fluid_visuals.filter_world_radius = 0.18
    fluid_visuals.filter_depth_threshold = 0.11
    fluid_visuals.enable_background_refraction = True
    fluid_visuals.refraction_ior = 1.33
    fluid_visuals.refraction_view_thickness = 0.40
    world.set_environment_fluid(0, fluid_visuals)

    light_entity = world.create_entity(0)
    light = neo.DirectionalLightComponent()
    light.direction = neo.Float3(-0.45, -1.0, 0.35)
    light.color = neo.Float3(1.0, 1.0, 1.0)
    light.intensity = 7.5
    light.casts_shadows = False
    world.set_directional_light(light_entity, light)

    camera_entity = world.create_entity(0)
    camera_transform = neo.TransformComponent()
    camera_transform.world_transform.position = neo.Float3(*CAMERA_POSITION)
    camera_transform.world_transform.rotation = _look_rotation(
        CAMERA_POSITION,
        (0.0, CONTAINER_TOP_Y * SCENE_SCALE + 0.5 * SCENE_SCALE, 0.0),
    )
    world.set_transform(camera_entity, camera_transform)

    camera = neo.CameraComponent()
    camera.product = neo.CameraProduct.ColorDepth
    camera.vertical_fov_degrees = 42.0
    camera.clear_color = True
    camera.clear_depth = True
    camera.clear_color_value = neo.Float4(0.03, 0.04, 0.06, 1.0)
    world.set_camera(camera_entity, camera)
    return camera_entity, psm_build


def main() -> int:
    if not hasattr(neo, "DebugViewerApp"):
        raise RuntimeError("This build does not include the Python debug viewer bindings.")

    viewer_desc = neo.DebugViewerAppDesc()
    viewer_desc.window_title = "CRESSim-Neo Fluid RL Isolation"
    viewer_desc.width = 1600
    viewer_desc.height = 900
    viewer_desc.step_simulation = True
    viewer_desc.show_stats = True

    config = neo.RuntimeConfig()
    config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
    config.gpu_device_desc.enable_validation = False
    config.physics_desc.enable_blocking_readback = False
    config.scene_layout.env_count = 1
    config.scene_layout.max_renderable_objects_per_env = 64
    config.scene_layout.max_lights_per_env = 2
    config.scene_layout.max_cameras_per_env = 1

    viewer = neo.DebugViewerApp()
    if not viewer.initialize(viewer_desc, config):
        raise RuntimeError("Failed to initialize the debug viewer.")

    runtime = neo.Runtime()
    if not runtime.initialize(config):
        viewer.shutdown()
        raise RuntimeError("Failed to initialize the runtime.")

    try:
        camera_entity, psm_build = _author_scene(runtime)
        binding = neo.DebugViewerCameraBinding()
        binding.camera_entity = camera_entity
        callbacks = neo.DebugViewerCallbacks()

        def before_tick(frame: neo.FrameContext, current_runtime: neo.Runtime) -> None:
            insertion_target = (
                PSM_INSERTION_BASE_TARGET
                + PSM_INSERTION_AMPLITUDE
                * math.sin(2.0 * math.pi * PSM_INSERTION_FREQUENCY_HZ * float(frame.time_seconds))
            )
            set_psm_joint_targets(
                current_runtime.world(),
                psm_build,
                [0.0, 0.0, insertion_target, 0.0, 0.0, 0.0, 0.0],
            )

        callbacks.before_tick = before_tick
        viewer.run(runtime, binding, callbacks)
        return 0
    finally:
        runtime.shutdown()
        viewer.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
