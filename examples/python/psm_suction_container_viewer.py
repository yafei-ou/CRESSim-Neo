import argparse
import importlib.util
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_BIN = REPO_ROOT / "build" / "bin"
if str(BUILD_BIN) not in sys.path:
    sys.path.insert(0, str(BUILD_BIN))

import cressim_neo as neo


def _load_local_package_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_local_package_module("cressim_neo.psm_builder", REPO_ROOT / "python" / "package" / "psm_builder.py")
psm_env_module = _load_local_package_module(
    "cressim_neo.psm_env",
    REPO_ROOT / "python" / "package" / "psm_env.py",
)
PsmEnv = psm_env_module.PsmEnv


CONTAINER_TOP_Y_LOCAL = 0.025932
CONTAINER_BOTTOM_Y_LOCAL = -0.569843
CONTAINER_CENTER_X_LOCAL = -0.105369
CONTAINER_CENTER_Z_LOCAL = 1.0
CONTAINER_ALIGN_Z = -1.9456
CONTAINER_ALIGN_X = 0.0
PSM_INSERTION_STROKE_LOCAL = 0.24
DEFAULT_START_INSERTION_FRACTION = 0.10
DEFAULT_END_INSERTION_FRACTION = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a suction-irrigator PSM above a TetGen soft-body container."
    )
    parser.add_argument(
        "--robot-scale",
        type=float,
        default=6.0,
        help="Uniform world-space scale applied during PSM authoring.",
    )
    parser.add_argument(
        "--container-top-y",
        type=float,
        default=0.49,
        help="World-space height for the container rim top.",
    )
    parser.add_argument(
        "--start-insertion",
        type=float,
        default=None,
        help="Initial insertion target in authored world units. Defaults to a scale-aware stroke fraction.",
    )
    parser.add_argument(
        "--end-insertion",
        type=float,
        default=None,
        help="Final insertion target in authored world units. Defaults to a deeper scale-aware stroke fraction.",
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=3.0,
        help="Seconds spent driving the tool down into the container.",
    )
    return parser.parse_args()


def _resolve_insertion_target(explicit_value: float | None, robot_scale: float, fraction: float) -> float:
    if explicit_value is not None:
        return float(explicit_value)
    return float(PSM_INSERTION_STROKE_LOCAL * robot_scale * fraction)


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


def _author_container(env: neo.PsmEnv, top_y: float) -> None:
    runtime = env.runtime
    world = runtime.world()
    resources = runtime.resources()
    models_dir = REPO_ROOT / "examples" / "models"
    node_path = models_dir / "container.node"
    ele_path = models_dir / "container.ele"
    surface_path = models_dir / "container_surface.obj"

    soft_entity = world.create_entity(0)
    soft_transform = neo.TransformComponent()
    soft_transform.world_transform.position = neo.Float3(
        CONTAINER_ALIGN_X - CONTAINER_CENTER_X_LOCAL,
        top_y - CONTAINER_TOP_Y_LOCAL,
        CONTAINER_ALIGN_Z - CONTAINER_CENTER_Z_LOCAL,
    )
    world.set_transform(soft_entity, soft_transform)

    soft_body = neo.SoftBodyComponent()
    soft_body.source.kind = neo.SoftBodySourceKind.TetGenFiles
    soft_body.source.tet_gen.node_file = str(node_path)
    soft_body.source.tet_gen.ele_file = str(ele_path)
    soft_body.source.tet_gen.static_particle_indices = _bottom_static_particle_indices(node_path, 0.03)
    soft_body.particle_mass = 0.06
    soft_body.particle_radius = 0.025
    soft_body.edge_compliance = 8.0e-4
    soft_body.volume_compliance = 2.0e-3
    soft_body.self_collision_enabled = True
    soft_body.supports_suturing = False
    soft_body.simulated = True
    soft_body.collision_layer = 0x1
    soft_body.collision_mask = 0xFFFFFFFF
    soft_body.material.contact.friction = 0.85
    soft_body.material.contact.static_friction = 1.0
    soft_body.material.contact.damping = 0.35
    if not world.set_soft_body(soft_entity, soft_body):
        raise RuntimeError("Failed to author TetGen soft-body container.")

    material = neo.MaterialResourceDesc()
    material.debug_name = "PsmSuctionContainer.ContainerMaterial"
    material.base_color = neo.Float3(0.76, 0.47, 0.32)
    material.roughness = 0.88
    material.metallic = 0.0
    container_material = resources.register_material(material)

    renderer = neo.MeshRendererComponent()
    renderer.mesh = resources.register_mesh(
        _load_obj_mesh(
            surface_path,
            "PsmSuctionContainer.ContainerMesh",
            reverse_winding=True,
        )
    )
    renderer.material = container_material
    renderer.visible = True
    world.set_mesh_renderer(soft_entity, renderer)


def _author_ground(env: neo.PsmEnv, top_y: float) -> None:
    runtime = env.runtime
    world = runtime.world()
    resources = runtime.resources()

    ground_half_height = 0.05
    ground_y = top_y - CONTAINER_TOP_Y_LOCAL + CONTAINER_BOTTOM_Y_LOCAL - ground_half_height - 0.01

    ground_entity = world.create_entity(0)
    ground_transform = neo.TransformComponent()
    ground_transform.world_transform.position = neo.Float3(0.0, ground_y, CONTAINER_ALIGN_Z)
    world.set_transform(ground_entity, ground_transform)

    ground_material = neo.MaterialResourceDesc()
    ground_material.debug_name = "PsmSuctionContainer.GroundMaterial"
    ground_material.base_color = neo.Float3(0.58, 0.60, 0.64)
    ground_material.roughness = 0.96
    ground_material.metallic = 0.0
    ground_material_handle = resources.register_material(ground_material)

    ground_renderer = neo.MeshRendererComponent()
    ground_renderer.mesh = resources.register_mesh(
        neo.make_plane_mesh(3.5, "PsmSuctionContainer.GroundMesh", 2.0)
    )
    ground_renderer.material = ground_material_handle
    ground_renderer.visible = True
    world.set_mesh_renderer(ground_entity, ground_renderer)

    ground_body = neo.RigidBodyComponent()
    ground_body.body_type = neo.RigidBodyType.Static
    ground_body.inverse_mass = 0.0
    ground_body.inverse_inertia_local = neo.Float3(0.0, 0.0, 0.0)
    world.set_rigid_body(ground_entity, ground_body)

    ground_collider = neo.ColliderComponent()
    ground_collider.shape_type = neo.ColliderShapeType.Box
    ground_collider.shape_params = neo.Float4(3.5, ground_half_height, 3.5, 0.0)
    ground_collider.friction = 0.95
    ground_collider.static_friction = 1.1
    world.add_collider(ground_entity, ground_collider)


def _reposition_camera(env: neo.PsmEnv) -> None:
    world = env.runtime.world()
    transform = world.try_get_transform(env.camera_entity)
    if transform is None:
        return
    camera_position = (0.0, 4.2, -8.4)
    camera_target = (0.0, 0.45, CONTAINER_ALIGN_Z)
    transform.world_transform.position = neo.Float3(*camera_position)
    transform.world_transform.rotation = _look_rotation(camera_position, camera_target)
    world.set_transform(env.camera_entity, transform)


def main() -> int:
    args = parse_args()
    if not hasattr(neo, "DebugViewerApp"):
        raise RuntimeError("This build does not include the Python debug viewer bindings.")

    start_insertion = _resolve_insertion_target(
        args.start_insertion,
        args.robot_scale,
        DEFAULT_START_INSERTION_FRACTION,
    )
    end_insertion = _resolve_insertion_target(
        args.end_insertion,
        args.robot_scale,
        DEFAULT_END_INSERTION_FRACTION,
    )

    viewer_desc = neo.DebugViewerAppDesc()
    viewer_desc.window_title = "CRESSim-Neo Suction Irrigator Container Demo"
    viewer_desc.width = 1600
    viewer_desc.height = 900
    viewer_desc.step_simulation = True
    viewer_desc.show_stats = True

    env = PsmEnv(
        resolve_root=REPO_ROOT,
        tool_type="suction_irrigator",
        viewer_desc=viewer_desc,
        add_ground=False,
        global_scale=args.robot_scale,
    )

    try:
        _author_ground(env, args.container_top_y)
        _author_container(env, args.container_top_y)
        _reposition_camera(env)

        initial_targets = [0.0, 0.0, start_insertion, 0.0, 0.0, 0.0, 0.0]
        env.set_joint_targets(initial_targets)

        callbacks = neo.DebugViewerCallbacks()

        def before_tick(frame: neo.FrameContext, runtime: neo.Runtime) -> None:
            time_seconds = float(frame.time_seconds)
            alpha = min(max(time_seconds / max(args.ramp_seconds, 1.0e-4), 0.0), 1.0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            insertion = (1.0 - smooth) * start_insertion + smooth * end_insertion
            env.set_joint_targets([0.0, 0.0, insertion, 0.0, 0.0, 0.0, 0.0])

        callbacks.before_tick = before_tick
        env.run_viewer(callbacks=callbacks)
        return 0
    finally:
        env.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
