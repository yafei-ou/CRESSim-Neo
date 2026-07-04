from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from . import _cressim_neo as neo

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


_JOINT_ORDER = (
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
)

_JOINT_KIND = {
    "psm_yaw_joint": "hinge",
    "psm_pitch_end_joint": "hinge",
    "psm_main_insertion_joint": "slider",
    "psm_tool_roll_joint": "hinge",
    "psm_tool_pitch_joint": "hinge",
    "psm_tool_yaw_joint": "hinge",
}

_DRIVE_MAX_ANGULAR_VELOCITY = 4.0
_DRIVE_MAX_LINEAR_VELOCITY = 1.0
_HINGE_DRIVE_COMPLIANCE = 1.0e-6
_SLIDER_DRIVE_COMPLIANCE = 1.0e-6
_GROUND_HALF_EXTENT = 0.75
_ENV_SPACING = 2.5
_BASE_HEIGHT = 0.1524
_HIDDEN_LINKS: set[str] = set()


@dataclass
class PsmRobotInstance:
    env_index: int
    base_entity: int
    link_entities: dict[str, int]
    joint_ids: list[int]
    joint_limits: list[tuple[float, float]]


def _require_trimesh():
    if trimesh is None:
        raise RuntimeError(
            "PsmScene requires trimesh for OBJ/STL visual mesh loading. "
            "Install it with `pip install trimesh` in your Python environment."
        )
    return trimesh


def _normalize_optional_path(path: str | os.PathLike[str] | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _find_psm_urdf_path(
    resolve_root: str | os.PathLike[str] | Path | None = None,
    urdf_path: str | os.PathLike[str] | Path | None = None,
) -> Path:
    explicit_urdf_path = _normalize_optional_path(urdf_path)
    if explicit_urdf_path is not None:
        if explicit_urdf_path.exists():
            return explicit_urdf_path
        raise RuntimeError(f"Configured PSM URDF path does not exist: {explicit_urdf_path}")

    search_roots: list[Path] = []
    explicit_root = _normalize_optional_path(resolve_root)
    if explicit_root is not None:
        search_roots.append(explicit_root)

    env_root = _normalize_optional_path(os.environ.get("CRESSIM_NEO_PSM_RESOLVE_ROOT"))
    if env_root is not None:
        search_roots.append(env_root)

    current = Path(__file__).resolve()
    search_roots.extend(current.parents)

    for root in search_roots:
        candidate = root / "extern" / "SurRoL" / "surrol" / "assets" / "psm" / "psm.urdf"
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Failed to locate extern/SurRoL/surrol/assets/psm/psm.urdf. "
        "Pass `resolve_root=...`, `urdf_path=...`, or set "
        "`CRESSIM_NEO_PSM_RESOLVE_ROOT`."
    )


def _parse_floats(raw: str | None, count: int, default: Iterable[float]) -> list[float]:
    if not raw:
        return list(default)
    values = [float(token) for token in raw.split()]
    if len(values) < count:
        values.extend(list(default)[len(values):count])
    return values[:count]


def _rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _pose_matrix(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = np.asarray(list(xyz), dtype=np.float64)
    return transform


def _urdf_to_engine_root_matrix() -> np.ndarray:
    return _pose_matrix((0.0, _BASE_HEIGHT, 0.0), (-0.5 * math.pi, 0.0, 0.0))


def _matrix_to_quaternion(matrix: np.ndarray) -> neo.Quaternion:
    m = matrix[:3, :3]
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return neo.Quaternion(float(x), float(y), float(z), float(w))


def _quaternion_conjugate(quaternion: neo.Quaternion) -> neo.Quaternion:
    return neo.Quaternion(-quaternion.x, -quaternion.y, -quaternion.z, quaternion.w)


def _quaternion_rotate(quaternion: neo.Quaternion, vector: np.ndarray) -> np.ndarray:
    q = np.array([quaternion.x, quaternion.y, quaternion.z], dtype=np.float64)
    t = 2.0 * np.cross(q, vector)
    return vector + quaternion.w * t + np.cross(q, t)


def _make_joint_frame_rotation(axis_x: np.ndarray) -> neo.Quaternion:
    axis = np.asarray(axis_x, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis /= length

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, axis))) > 0.99:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    axis_y = np.cross(axis, reference)
    axis_y_norm = float(np.linalg.norm(axis_y))
    if axis_y_norm <= 1.0e-8:
        axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        axis_y /= axis_y_norm
    axis_z = np.cross(axis, axis_y)
    axis_z_norm = float(np.linalg.norm(axis_z))
    if axis_z_norm > 1.0e-8:
        axis_z /= axis_z_norm
    basis = np.eye(4, dtype=np.float64)
    basis[:3, 0] = axis
    basis[:3, 1] = axis_y
    basis[:3, 2] = axis_z
    return _matrix_to_quaternion(basis)


def _make_joint_frame_world_rotation(
    joint_world_rotation: np.ndarray,
    joint_axis_local: np.ndarray,
) -> np.ndarray:
    axis_world = joint_world_rotation @ np.asarray(joint_axis_local, dtype=np.float64)
    axis_length = float(np.linalg.norm(axis_world))
    if axis_length <= 1.0e-8:
        axis_world = joint_world_rotation[:, 0]
        axis_length = float(np.linalg.norm(axis_world))
    axis_x = axis_world / max(axis_length, 1.0e-8)

    reference_candidates = [
        joint_world_rotation[:, 0],
        joint_world_rotation[:, 1],
        joint_world_rotation[:, 2],
    ]
    axis_y = None
    for reference in reference_candidates:
        candidate = reference - axis_x * float(np.dot(reference, axis_x))
        candidate_length = float(np.linalg.norm(candidate))
        if candidate_length > 1.0e-8:
            axis_y = candidate / candidate_length
            break
    if axis_y is None:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, axis_x))) > 0.99:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidate = fallback - axis_x * float(np.dot(fallback, axis_x))
        axis_y = candidate / max(float(np.linalg.norm(candidate)), 1.0e-8)

    axis_z = np.cross(axis_x, axis_y)
    axis_z /= max(float(np.linalg.norm(axis_z)), 1.0e-8)
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= max(float(np.linalg.norm(axis_y)), 1.0e-8)

    basis = np.eye(4, dtype=np.float64)
    basis[:3, 0] = axis_x
    basis[:3, 1] = axis_y
    basis[:3, 2] = axis_z
    return basis


def _look_rotation(position: np.ndarray, target: np.ndarray) -> neo.Quaternion:
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-8:
        return neo.Quaternion()
    forward /= forward_norm
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= right_norm
    corrected_up = np.cross(forward, right)
    basis = np.eye(4, dtype=np.float64)
    basis[:3, 0] = right
    basis[:3, 1] = corrected_up
    basis[:3, 2] = forward
    return _matrix_to_quaternion(basis)


def _env_origin(env_index: int, env_count: int) -> np.ndarray:
    cols = max(1, math.ceil(math.sqrt(float(env_count))))
    row = env_index // cols
    col = env_index % cols
    x_offset = (col - 0.5 * (cols - 1)) * _ENV_SPACING
    y_offset = (row - 0.5 * (max(1, math.ceil(env_count / cols)) - 1)) * _ENV_SPACING
    return np.array([x_offset, y_offset, 0.0], dtype=np.float64)


def _combine_trimesh_objects(meshes: list) -> object:
    if not meshes:
        raise RuntimeError("Expected at least one visual mesh to combine.")
    if len(meshes) == 1:
        return meshes[0]
    return _require_trimesh().util.concatenate(meshes)


def _load_trimesh(path: Path, transform: np.ndarray, scale: list[float] | None):
    tm = _require_trimesh()
    loaded = tm.load(path, process=False)
    if isinstance(loaded, tm.Scene):
        source_meshes = [geom.copy() for geom in loaded.geometry.values()]
    else:
        source_meshes = [loaded.copy()]
    if not source_meshes:
        raise RuntimeError(f"Visual mesh contains no geometry: {path}")
    for mesh in source_meshes:
        if scale is not None:
            mesh.apply_scale(scale)
        mesh.apply_transform(transform)
    return _combine_trimesh_objects(source_meshes)


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    for i0, i1, i2 in faces:
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        face_normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(face_normal))
        if length > 1.0e-8:
            face_normal /= length
        else:
            face_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        normals[i0] += face_normal
        normals[i1] += face_normal
        normals[i2] += face_normal

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-8
    normals[valid] /= lengths[valid][:, None]
    normals[~valid] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return normals


def _mesh_to_desc(mesh, debug_name: str) -> neo.MeshResourceDesc:
    vertices_array = np.asarray(mesh.vertices, dtype=np.float64)
    faces_array = np.asarray(mesh.faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise RuntimeError(f"Mesh {debug_name} has invalid vertex layout: {vertices_array.shape}")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise RuntimeError(f"Mesh {debug_name} is not triangulated: {faces_array.shape}")
    if len(vertices_array) == 0 or len(faces_array) == 0:
        raise RuntimeError(f"Mesh {debug_name} contains no renderable geometry.")
    if faces_array.min() < 0 or faces_array.max() >= len(vertices_array):
        raise RuntimeError(f"Mesh {debug_name} references out-of-range vertex indices.")

    # Match the engine's procedural/OBJ winding convention so front faces render correctly.
    faces_array = faces_array[:, [0, 2, 1]].copy()
    normals_array = _compute_vertex_normals(vertices_array, faces_array)

    desc = neo.MeshResourceDesc()
    desc.debug_name = debug_name
    vertices = []
    for index, position in enumerate(vertices_array):
        vertex = neo.MeshVertex()
        vertex.position = neo.Float3(float(position[0]), float(position[1]), float(position[2]))
        normal = normals_array[index]
        vertex.normal = neo.Float3(float(normal[0]), float(normal[1]), float(normal[2]))
        vertices.append(vertex)
    desc.vertices = vertices
    desc.indices = faces_array.astype(np.uint32, copy=False).reshape(-1).tolist()
    return desc


def _parse_psm_urdf(urdf_path: Path) -> tuple[dict, dict, dict]:
    root = ET.parse(urdf_path).getroot()

    materials: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall("material"):
        name = material.attrib.get("name")
        color = material.find("color")
        if not name or color is None:
            continue
        rgba = _parse_floats(color.attrib.get("rgba"), 4, (1.0, 1.0, 1.0, 1.0))
        materials[name] = (rgba[0], rgba[1], rgba[2], rgba[3])

    links: dict[str, dict] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        inertial = link.find("inertial")
        inertial_mass = 1.0
        inertia_diag = (0.0, 0.0, 0.0)
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if mass is not None:
                inertial_mass = float(mass.attrib.get("value", "1.0"))
            if inertia is not None:
                inertia_diag = (
                    float(inertia.attrib.get("ixx", "0.0")),
                    float(inertia.attrib.get("iyy", "0.0")),
                    float(inertia.attrib.get("izz", "0.0")),
                )

        visuals = []
        for visual_index, visual in enumerate(link.findall("visual")):
            origin = visual.find("origin")
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = geometry.find("mesh")
            if mesh is None:
                continue
            xyz = _parse_floats(origin.attrib.get("xyz") if origin is not None else None,
                                3, (0.0, 0.0, 0.0))
            rpy = _parse_floats(origin.attrib.get("rpy") if origin is not None else None,
                                3, (0.0, 0.0, 0.0))
            scale = _parse_floats(mesh.attrib.get("scale"), 3, (1.0, 1.0, 1.0)) \
                if mesh.attrib.get("scale") else None

            material_name = "default"
            material_node = visual.find("material")
            if material_node is not None:
                material_name = material_node.attrib.get("name", material_name)
                inline_color = material_node.find("color")
                if inline_color is not None:
                    rgba = _parse_floats(inline_color.attrib.get("rgba"), 4, (1.0, 1.0, 1.0, 1.0))
                    materials[f"{link_name}.visual.{visual_index}"] = tuple(rgba)
                    material_name = f"{link_name}.visual.{visual_index}"

            visuals.append(
                {
                    "mesh_path": urdf_path.parent / mesh.attrib["filename"],
                    "origin": _pose_matrix(xyz, rpy),
                    "scale": scale,
                    "material_name": material_name,
                }
            )

        links[link_name] = {
            "mass": inertial_mass,
            "inertia_diag": inertia_diag,
            "visuals": visuals,
        }

    joints: dict[str, dict] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        xyz = _parse_floats(origin.attrib.get("xyz") if origin is not None else None,
                            3, (0.0, 0.0, 0.0))
        rpy = _parse_floats(origin.attrib.get("rpy") if origin is not None else None,
                            3, (0.0, 0.0, 0.0))
        axis_values = _parse_floats(axis.attrib.get("xyz") if axis is not None else None,
                                    3, (1.0, 0.0, 0.0))
        lower = float(limit.attrib.get("lower", "0.0")) if limit is not None else 0.0
        upper = float(limit.attrib.get("upper", "0.0")) if limit is not None else 0.0
        velocity = float(limit.attrib.get("velocity", "0.0")) if limit is not None else 0.0
        joints[name] = {
            "type": joint.attrib["type"],
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "origin": _pose_matrix(xyz, rpy),
            "axis": np.asarray(axis_values, dtype=np.float64),
            "lower": lower,
            "upper": upper,
            "velocity": velocity,
        }

    return materials, links, joints


class PsmScene:
    def __init__(
        self,
        runtime: neo.Runtime,
        runtime_config: neo.RuntimeConfig,
        env_count: int,
        resolve_root: str | os.PathLike[str] | Path | None = None,
        urdf_path: str | os.PathLike[str] | Path | None = None,
        viewer: neo.DebugViewerApp | None = None,
    ):
        self.runtime = runtime
        self.runtime_config = runtime_config
        self.viewer = viewer
        self.env_count = env_count
        self.resolve_root = _normalize_optional_path(resolve_root)
        self.urdf_path = _normalize_optional_path(urdf_path)
        self.instances: list[PsmRobotInstance] = []
        self.camera_entities: list[int] = []
        self._frame = neo.FrameContext()
        self._materials: dict[str, neo.MaterialHandle] = {}
        self._mesh_handles: dict[str, neo.MeshHandle] = {}
        self._mesh_inertia_diagonals: dict[str, tuple[float, float, float]] = {}
        self._viewer_session_started = False
        self._shutdown = False

    @classmethod
    def create(
        cls,
        env_count: int = 1,
        resolve_root: str | os.PathLike[str] | Path | None = None,
        urdf_path: str | os.PathLike[str] | Path | None = None,
        viewer_desc: neo.DebugViewerAppDesc | None = None,
    ) -> "PsmScene":
        config = neo.RuntimeConfig()
        config.scene_layout.env_count = env_count
        config.scene_layout.max_renderable_objects_per_env = 16
        config.scene_layout.max_lights_per_env = 3
        config.scene_layout.max_cameras_per_env = 1
        config.physics_desc.enable_blocking_readback = False
        config.physics_desc.substeps = 4
        config.physics_desc.default_iterations = 50

        viewer = None
        if viewer_desc is not None:
            if not hasattr(neo, "DebugViewerApp"):
                raise RuntimeError("Debug viewer bindings are unavailable in this build.")
            viewer = neo.DebugViewerApp()
            if not viewer.initialize(viewer_desc, config):
                raise RuntimeError("Failed to initialize the debug viewer.")

        runtime = neo.Runtime()
        if not runtime.initialize(config):
            if viewer is not None:
                viewer.shutdown()
            raise RuntimeError("Failed to initialize the runtime for the PSM scene.")

        scene = cls(runtime, config, env_count, resolve_root, urdf_path, viewer)
        scene.author()
        return scene

    def author(self) -> None:
        world = self.runtime.world()
        resources = self.runtime.resources()
        materials, links, joints = _parse_psm_urdf(
            _find_psm_urdf_path(resolve_root=self.resolve_root, urdf_path=self.urdf_path)
        )

        self._materials["ground"] = self._register_material(
            resources, "PsmScene.Ground", (0.35, 0.36, 0.40, 1.0), 0.95
        )

        retained_links = {"psm_base_link"}
        retained_links.update(joints[joint_name]["child"] for joint_name in _JOINT_ORDER)
        retained_joints = {joint_name: joints[joint_name] for joint_name in _JOINT_ORDER}
        material_handles = {
            name: self._register_material(
                resources,
                f"PsmScene.Material.{name}",
                materials.get(name, (0.8, 0.8, 0.8, 1.0)),
                0.55,
            )
            for name in {visual["material_name"] for link_name in retained_links
                         for visual in links[link_name]["visuals"]}
        }

        for env_index in range(self.env_count):
            env_origin = _env_origin(env_index, self.env_count)
            self._author_environment(
                world, resources, env_index, env_origin, links, retained_joints, material_handles
            )

    def _register_material(
        self,
        resources: neo.RenderResourceManager,
        debug_name: str,
        rgba: tuple[float, float, float, float],
        roughness: float,
    ) -> neo.MaterialHandle:
        if debug_name in self._materials:
            return self._materials[debug_name]
        material = neo.MaterialResourceDesc()
        material.debug_name = debug_name
        material.base_color = neo.Float3(float(rgba[0]), float(rgba[1]), float(rgba[2]))
        material.roughness = float(roughness)
        material.opacity = float(rgba[3])
        self._materials[debug_name] = resources.register_material(material)
        return self._materials[debug_name]

    def _register_link_mesh(
        self,
        resources: neo.RenderResourceManager,
        link_name: str,
        link_data: dict,
    ) -> neo.MeshHandle:
        if link_name in self._mesh_handles:
            return self._mesh_handles[link_name]

        visual_meshes = []
        for visual in link_data["visuals"]:
            visual_meshes.append(
                _load_trimesh(visual["mesh_path"], visual["origin"], visual["scale"])
            )
        combined = _combine_trimesh_objects(visual_meshes)
        bounds = np.asarray(combined.bounds, dtype=np.float64)
        extents = np.maximum(bounds[1] - bounds[0], 1.0e-4)
        self._mesh_inertia_diagonals[link_name] = (
            float((extents[1] * extents[1] + extents[2] * extents[2]) / 12.0),
            float((extents[0] * extents[0] + extents[2] * extents[2]) / 12.0),
            float((extents[0] * extents[0] + extents[1] * extents[1]) / 12.0),
        )
        mesh_desc = _mesh_to_desc(combined, f"PsmScene.Mesh.{link_name}")
        handle = resources.register_mesh(mesh_desc)
        self._mesh_handles[link_name] = handle
        return handle

    def _author_environment(
        self,
        world: neo.World,
        resources: neo.RenderResourceManager,
        env_index: int,
        env_origin: np.ndarray,
        links: dict,
        joints: dict,
        material_handles: dict[str, neo.MaterialHandle],
    ) -> None:
        ground_entity = world.create_entity(env_index)
        ground_transform = neo.TransformComponent()
        ground_transform.world_transform.position = neo.Float3(
            float(env_origin[0]), -0.02, float(env_origin[1])
        )
        ground_transform.world_transform.scale = neo.Float3(1.0, 1.0, 1.0)
        world.set_transform(ground_entity, ground_transform)
        ground_mesh = resources.register_mesh(
            neo.make_plane_mesh(_GROUND_HALF_EXTENT, f"PsmScene.Ground.{env_index}", 1.0)
        )
        ground_renderer = neo.MeshRendererComponent()
        ground_renderer.mesh = ground_mesh
        ground_renderer.material = self._materials["ground"]
        ground_renderer.visible = True
        world.set_mesh_renderer(ground_entity, ground_renderer)

        camera_entity = world.create_entity(env_index)
        camera_position = np.array(
            [env_origin[0] + 1.25, 0.95, env_origin[2] - 1.10],
            dtype=np.float64,
        )
        camera_target = np.array(
            [env_origin[0], 0.30, env_origin[2]],
            dtype=np.float64,
        )
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(
            float(camera_position[0]), float(camera_position[1]), float(camera_position[2])
        )
        camera_transform.world_transform.rotation = _look_rotation(camera_position, camera_target)
        world.set_transform(camera_entity, camera_transform)
        camera = neo.CameraComponent()
        camera.vertical_fov_degrees = 55.0
        camera.near_clip = 0.01
        camera.far_clip = 20.0
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(0.84, 0.90, 0.98, 1.0)
        world.set_camera(camera_entity, camera)
        self.camera_entities.append(camera_entity)

        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.35, -0.45, -0.82)
        light.color = neo.Float3(1.0, 0.98, 0.95)
        light.intensity = 7.5
        world.set_directional_light(light_entity, light)

        fill_light_entity = world.create_entity(env_index)
        fill_light = neo.PointLightComponent()
        fill_light.color = neo.Float3(0.76, 0.84, 1.0)
        fill_light.intensity = 28.0
        fill_light.range = 4.0
        world.set_point_light(fill_light_entity, fill_light)
        fill_transform = neo.TransformComponent()
        fill_transform.world_transform.position = neo.Float3(
            float(env_origin[0] + 0.65), 1.15, float(env_origin[1] - 0.85)
        )
        world.set_transform(fill_light_entity, fill_transform)

        rim_light_entity = world.create_entity(env_index)
        rim_light = neo.PointLightComponent()
        rim_light.color = neo.Float3(1.0, 0.92, 0.82)
        rim_light.intensity = 18.0
        rim_light.range = 3.5
        world.set_point_light(rim_light_entity, rim_light)
        rim_transform = neo.TransformComponent()
        rim_transform.world_transform.position = neo.Float3(
            float(env_origin[0] - 0.55), 0.75, float(env_origin[1] + 0.75)
        )
        world.set_transform(rim_light_entity, rim_transform)

        link_world = {"psm_base_link": _urdf_to_engine_root_matrix()}
        link_world["psm_base_link"][:3, 3] += np.array([env_origin[0], 0.0, env_origin[1]])
        for joint_name in _JOINT_ORDER:
            joint = joints[joint_name]
            link_world[joint["child"]] = link_world[joint["parent"]] @ joint["origin"]

        link_entities: dict[str, int] = {}
        base_entity = 0
        for link_name in ["psm_base_link", *(joints[name]["child"] for name in _JOINT_ORDER)]:
            link_entity = world.create_entity(env_index)
            if link_name == "psm_base_link":
                base_entity = link_entity
            link_entities[link_name] = link_entity

            transform = neo.TransformComponent()
            transform.world_transform.position = neo.Float3(
                float(link_world[link_name][0, 3]),
                float(link_world[link_name][1, 3]),
                float(link_world[link_name][2, 3]),
            )
            transform.world_transform.rotation = _matrix_to_quaternion(link_world[link_name])
            world.set_transform(link_entity, transform)

            mesh_handle = self._register_link_mesh(resources, link_name, links[link_name])
            material_name = links[link_name]["visuals"][0]["material_name"]
            renderer = neo.MeshRendererComponent()
            renderer.mesh = mesh_handle
            renderer.material = material_handles[material_name]
            renderer.visible = link_name not in _HIDDEN_LINKS
            world.set_mesh_renderer(link_entity, renderer)

            rigid_body = neo.RigidBodyComponent()
            if link_name == "psm_base_link":
                rigid_body.body_type = neo.RigidBodyType.Static
                rigid_body.inverse_mass = 0.0
                rigid_body.inverse_inertia_local = neo.Float3(0.0, 0.0, 0.0)
            else:
                mass = float(links[link_name]["mass"])
                inverse_mass = 0.0 if mass <= 0.0 else 1.0 / mass
                inertia_diag = links[link_name]["inertia_diag"]
                if inertia_diag[0] <= 0.0 or inertia_diag[1] <= 0.0 or inertia_diag[2] <= 0.0:
                    fallback = self._mesh_inertia_diagonals.get(link_name, (1.0e-4, 1.0e-4, 1.0e-4))
                    inertia_diag = (
                        max(float(mass) * fallback[0], 1.0e-6),
                        max(float(mass) * fallback[1], 1.0e-6),
                        max(float(mass) * fallback[2], 1.0e-6),
                    )
                rigid_body.body_type = neo.RigidBodyType.Dynamic
                rigid_body.inverse_mass = inverse_mass
                rigid_body.inverse_inertia_local = neo.Float3(
                    0.0 if inertia_diag[0] <= 0.0 else float(1.0 / inertia_diag[0]),
                    0.0 if inertia_diag[1] <= 0.0 else float(1.0 / inertia_diag[1]),
                    0.0 if inertia_diag[2] <= 0.0 else float(1.0 / inertia_diag[2]),
                )
            world.set_rigid_body(link_entity, rigid_body)

        joint_ids: list[int] = []
        joint_limits: list[tuple[float, float]] = []
        for joint_index, joint_name in enumerate(_JOINT_ORDER):
            joint = joints[joint_name]
            parent_entity = link_entities[joint["parent"]]
            child_entity = link_entities[joint["child"]]
            joint_world = link_world[joint["parent"]] @ joint["origin"]
            parent_world_rotation = link_world[joint["parent"]][:3, :3]
            child_world_rotation = link_world[joint["child"]][:3, :3]
            solver_joint_world = _make_joint_frame_world_rotation(
                joint_world[:3, :3],
                joint["axis"],
            )
            parent_rotation = _matrix_to_quaternion(parent_world_rotation)
            child_rotation = _matrix_to_quaternion(child_world_rotation)
            anchor_world = joint_world[:3, 3]
            parent_position = link_world[joint["parent"]][:3, 3]
            child_position = link_world[joint["child"]][:3, 3]
            local_anchor_a = _quaternion_rotate(
                _quaternion_conjugate(parent_rotation), anchor_world - parent_position
            )
            local_anchor_b = _quaternion_rotate(
                _quaternion_conjugate(child_rotation), anchor_world - child_position
            )
            local_rotation_a = _matrix_to_quaternion(parent_world_rotation.T @ solver_joint_world[:3, :3])
            local_rotation_b = _matrix_to_quaternion(child_world_rotation.T @ solver_joint_world[:3, :3])

            assigned_joint_id = env_index * 100 + joint_index + 1
            joint_ids.append(assigned_joint_id)
            joint_limits.append((joint["lower"], joint["upper"]))

            if _JOINT_KIND[joint_name] == "hinge":
                state = neo.HingeJointState()
                state.joint_id = assigned_joint_id
                state.body_a = parent_entity
                state.body_b = child_entity
                state.local_anchor_a = neo.Float3(
                    float(local_anchor_a[0]), float(local_anchor_a[1]), float(local_anchor_a[2])
                )
                state.local_anchor_b = neo.Float3(
                    float(local_anchor_b[0]), float(local_anchor_b[1]), float(local_anchor_b[2])
                )
                state.local_rotation_a = local_rotation_a
                state.local_rotation_b = local_rotation_b
                state.limit_enabled = True
                state.limit_min = float(joint["lower"])
                state.limit_max = float(joint["upper"])
                state.drive_mode = neo.RigidJointDriveMode.TargetPosition
                state.drive_target_angle = 0.0
                state.drive_compliance = _HINGE_DRIVE_COMPLIANCE
                state.drive_max_angular_velocity = max(
                    float(joint["velocity"]), _DRIVE_MAX_ANGULAR_VELOCITY
                )
                if not world.upsert_hinge_joint(state):
                    raise RuntimeError(f"Failed to author hinge joint {joint_name}.")
            else:
                state = neo.SliderJointState()
                state.joint_id = assigned_joint_id
                state.body_a = parent_entity
                state.body_b = child_entity
                state.local_anchor_a = neo.Float3(
                    float(local_anchor_a[0]), float(local_anchor_a[1]), float(local_anchor_a[2])
                )
                state.local_anchor_b = neo.Float3(
                    float(local_anchor_b[0]), float(local_anchor_b[1]), float(local_anchor_b[2])
                )
                state.local_rotation_a = local_rotation_a
                state.local_rotation_b = local_rotation_b
                state.limit_enabled = True
                state.limit_min = float(joint["lower"])
                state.limit_max = float(joint["upper"])
                state.drive_mode = neo.RigidJointDriveMode.TargetPosition
                state.drive_target_position = 0.0
                state.drive_compliance = _SLIDER_DRIVE_COMPLIANCE
                state.drive_max_velocity = max(float(joint["velocity"]), _DRIVE_MAX_LINEAR_VELOCITY)
                if not world.upsert_slider_joint(state):
                    raise RuntimeError(f"Failed to author slider joint {joint_name}.")

        self.instances.append(
            PsmRobotInstance(
                env_index=env_index,
                base_entity=base_entity,
                link_entities=link_entities,
                joint_ids=joint_ids,
                joint_limits=joint_limits,
            )
        )

    def set_joint_targets(
        self,
        targets: Iterable[float] | Iterable[Iterable[float]],
        env_index: int | None = None,
    ) -> None:
        if env_index is not None:
            self._set_env_joint_targets(env_index, list(float(value) for value in targets))
            return

        target_list = list(targets)
        if self.env_count == 1 and target_list and not isinstance(target_list[0], (list, tuple, np.ndarray)):
            self._set_env_joint_targets(0, list(float(value) for value in target_list))
            return

        if len(target_list) != self.env_count:
            raise ValueError(f"Expected {self.env_count} target vectors, got {len(target_list)}.")
        for current_env, env_targets in enumerate(target_list):
            self._set_env_joint_targets(current_env, list(float(value) for value in env_targets))

    def _set_env_joint_targets(self, env_index: int, targets: list[float]) -> None:
        if env_index < 0 or env_index >= self.env_count:
            raise ValueError(f"Environment index {env_index} is out of range.")
        if len(targets) != len(_JOINT_ORDER):
            raise ValueError(f"Expected {len(_JOINT_ORDER)} joint targets, got {len(targets)}.")

        world = self.runtime.world()
        instance = self.instances[env_index]
        for joint_name, joint_id, limits, target in zip(
            _JOINT_ORDER, instance.joint_ids, instance.joint_limits, targets
        ):
            clamped = min(max(float(target), limits[0]), limits[1])
            if _JOINT_KIND[joint_name] == "hinge":
                state = world.try_get_hinge_joint(joint_id)
                if state is None:
                    raise RuntimeError(f"Missing hinge joint state for {joint_name}.")
                state.drive_target_angle = clamped
                if not world.upsert_hinge_joint(state):
                    raise RuntimeError(f"Failed to update hinge joint {joint_name}.")
            else:
                state = world.try_get_slider_joint(joint_id)
                if state is None:
                    raise RuntimeError(f"Missing slider joint state for {joint_name}.")
                state.drive_target_position = clamped
                if not world.upsert_slider_joint(state):
                    raise RuntimeError(f"Failed to update slider joint {joint_name}.")

    def sync(self) -> None:
        self.runtime.prepare()
        if not self.runtime.upload_world():
            raise RuntimeError("Failed to upload authored PSM world state.")

    def step(self, delta_seconds: float = 1.0 / 60.0) -> None:
        self.sync()
        self._frame.frame_index += 1
        self._frame.delta_seconds = float(delta_seconds)
        self._frame.time_seconds += float(delta_seconds)
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("PSM scene physics step failed.")
        if not self.runtime.step_simulation_sensors(self._frame):
            raise RuntimeError("PSM scene simulation-sensor step failed.")
        self.runtime.step_visual_sensors(self._frame)
        self.runtime.end_frame(self._frame)

    def run_viewer(
        self,
        env_index: int = 0,
        callbacks: neo.DebugViewerCallbacks | None = None,
    ) -> bool:
        if self.viewer is None:
            raise RuntimeError("This PSM scene was not created with a debug viewer.")
        if env_index < 0 or env_index >= len(self.camera_entities):
            raise ValueError(f"Environment index {env_index} is out of range.")
        binding = neo.DebugViewerCameraBinding()
        binding.camera_entity = self.camera_entities[env_index]
        self._viewer_session_started = True
        if callbacks is None:
            return self.viewer.run(self.runtime, binding)
        return self.viewer.run(self.runtime, binding, callbacks)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        viewer = self.viewer
        self.viewer = None

        if viewer is not None and not self._viewer_session_started:
            viewer.shutdown()

        self.runtime.shutdown()


def create_psm_scene(
    env_count: int = 1,
    resolve_root: str | os.PathLike[str] | Path | None = None,
    urdf_path: str | os.PathLike[str] | Path | None = None,
    viewer_desc: neo.DebugViewerAppDesc | None = None,
) -> PsmScene:
    return PsmScene.create(
        env_count=env_count,
        resolve_root=resolve_root,
        urdf_path=urdf_path,
        viewer_desc=viewer_desc,
    )
