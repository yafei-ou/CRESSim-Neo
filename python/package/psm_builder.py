from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import struct
from typing import Iterable
import xml.etree.ElementTree as ET
import zlib

from . import _cressim_neo as neo

import numpy as np


_ARM_JOINT_ORDER = (
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
)

_GRIPPER_JOINT_ORDER = (
    "psm_tool_gripper1_joint",
    "psm_tool_gripper2_joint",
)

_PASSIVE_JOINT_ORDER = (
    "psm_pitch_back_joint",
    "psm_pitch_bottom_joint",
    "psm_pitch_top_joint",
    "psm_pitch_front_joint",
)

_PHYSICAL_JOINT_ORDER = (*_ARM_JOINT_ORDER, *_PASSIVE_JOINT_ORDER, *_GRIPPER_JOINT_ORDER)
_COMMAND_JOINT_ORDER = (*_ARM_JOINT_ORDER, "psm_jaw_joint")

_JOINT_KIND = {
    "psm_yaw_joint": "hinge",
    "psm_pitch_end_joint": "hinge",
    "psm_pitch_back_joint": "hinge",
    "psm_pitch_bottom_joint": "hinge",
    "psm_pitch_top_joint": "hinge",
    "psm_pitch_front_joint": "hinge",
    "psm_main_insertion_joint": "slider",
    "psm_tool_roll_joint": "hinge",
    "psm_tool_pitch_joint": "hinge",
    "psm_tool_yaw_joint": "hinge",
    "psm_tool_gripper1_joint": "hinge",
    "psm_tool_gripper2_joint": "hinge",
}

_DRIVE_MAX_ANGULAR_VELOCITY = 4.0
_DRIVE_MAX_LINEAR_VELOCITY = 1.0
_HINGE_DRIVE_COMPLIANCE = 1.0e-6
_SLIDER_DRIVE_COMPLIANCE = 1.0e-6
_HINGE_DRIVE_DAMPING = 0.0
_SLIDER_DRIVE_DAMPING = 0.0
_GROUND_HALF_EXTENT = 0.75
_DEFAULT_ENV_SPACING = 2.5
_BASE_HEIGHT = 0.1524
_PSM_ASSET_BASIS = np.array(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
_HIDDEN_LINKS: set[str] = set()
_DEFAULT_INERTIA_DIAG = (1.0e-4, 1.0e-4, 1.0e-4)
_FALLBACK_INERTIA_SCALE = 1.0
_PITCH_END_TOP_CLOSURE_MESH_OFFSET = np.array((0.00388, -0.036286, 0.0), dtype=np.float64)
_PITCH_BOTTOM_FRONT_CLOSURE_OFFSET = np.array((0.096164, 0.0, 0.0), dtype=np.float64)
_SUCTION_IRRIGATOR_YAW_CAPSULE_RADIUS = 0.004
_SUCTION_IRRIGATOR_YAW_CAPSULE_TOTAL_LENGTH = 0.011958
_SUCTION_IRRIGATOR_YAW_CAPSULE_HALF_HEIGHT = (
    0.5 * _SUCTION_IRRIGATOR_YAW_CAPSULE_TOTAL_LENGTH - _SUCTION_IRRIGATOR_YAW_CAPSULE_RADIUS
)
_SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION = np.array(
    (0.0, 0.5 * _SUCTION_IRRIGATOR_YAW_CAPSULE_TOTAL_LENGTH, 0.0),
    dtype=np.float64,
)

_InertiaDiag = tuple[float, float, float]
_LinkDynamicsFallback = tuple[float, _InertiaDiag]

_LINK_DYNAMICS_FALLBACKS: dict[str, _LinkDynamicsFallback] = {
    "psm_yaw_link": (1.4705, (6.0e-2, 6.0e-2, 1.2e-2)),
    "psm_pitch_end_link": (2.091, (5.0e-2, 5.0e-2, 1.6e-2)),
    "psm_main_insertion_link": (0.22491, (1.2e-2, 1.2e-2, 3.0e-3)),
    "psm_tool_roll_link": (0.02, (5.0e-4, 5.0e-4, 1.8e-4)),
    "psm_tool_pitch_link": (0.10, (4.0e-4, 4.0e-4, 1.4e-4)),
    "psm_tool_yaw_link": (0.08, (2.8e-4, 2.8e-4, 1.0e-4)),
    "psm_tool_gripper1_link": (0.03, (1.0e-4, 6.0e-5, 1.0e-4)),
    "psm_tool_gripper2_link": (0.03, (1.0e-4, 6.0e-5, 1.0e-4)),
}

_DEFAULT_PSM_TOOL_TYPE = "large_needle_driver"
_PSM_TOOL_URDF_FILENAMES = {
    "large_needle_driver": "psm.urdf",
    "suction_irrigator": "psm_suction_irrigator.urdf",
}
_PSM_TOOL_ALIASES = {
    "default": _DEFAULT_PSM_TOOL_TYPE,
    "large_needle_driver": "large_needle_driver",
    "needle_driver": "large_needle_driver",
    "lnd": "large_needle_driver",
    "suction_irrigator": "suction_irrigator",
    "suction-irrigator": "suction_irrigator",
    "suction": "suction_irrigator",
}
_MISSING_JOINT_ID = -1
_MISSING_JOINT_LIMIT = (0.0, 0.0)
_MISSING_JAW_IDS = (_MISSING_JOINT_ID, _MISSING_JOINT_ID)


def _srgb_channel_to_linear(value: float) -> float:
    """Convert an sRGB material channel to the renderer's linear working space."""
    srgb = min(max(float(value), 0.0), 1.0)
    if srgb <= 0.04045:
        return srgb / 12.92
    return ((srgb + 0.055) / 1.055) ** 2.4


def _srgb_rgba_to_linear(
    rgba: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert URDF display-color RGB while preserving its non-color alpha channel."""
    return (
        _srgb_channel_to_linear(rgba[0]),
        _srgb_channel_to_linear(rgba[1]),
        _srgb_channel_to_linear(rgba[2]),
        float(rgba[3]),
    )


@dataclass
class PsmRobotInstance:
    env_index: int
    base_entity: int
    link_entities: dict[str, int]
    arm_joint_ids: list[int]
    arm_joint_limits: list[tuple[float, float]]
    passive_joint_ids: dict[str, int]
    passive_joint_limits: dict[str, tuple[float, float]]
    jaw_joint_ids: tuple[int, int]
    jaw_limit: tuple[float, float]


@dataclass
class PsmAuthoringConfig:
    resolve_root: str | os.PathLike[str] | Path | None = None
    urdf_path: str | os.PathLike[str] | Path | None = None
    tool_type: str = _DEFAULT_PSM_TOOL_TYPE
    env_count: int = 1
    add_ground: bool = True
    add_default_lighting: bool = True
    add_default_camera: bool = True
    env_spacing: float = _DEFAULT_ENV_SPACING
    global_scale: float = 1.0


@dataclass
class PsmBuildResult:
    instances: list[PsmRobotInstance] = field(default_factory=list)
    camera_entities: list[int] = field(default_factory=list)
    ground_entities: list[int] = field(default_factory=list)
    light_entities: list[list[int]] = field(default_factory=list)
    urdf_path: Path | None = None
    env_count: int = 0


def _normalize_optional_path(path: str | os.PathLike[str] | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _find_psm_urdf_path(
    resolve_root: str | os.PathLike[str] | Path | None = None,
    urdf_path: str | os.PathLike[str] | Path | None = None,
    tool_type: str = _DEFAULT_PSM_TOOL_TYPE,
) -> Path:
    explicit_urdf_path = _normalize_optional_path(urdf_path)
    if explicit_urdf_path is not None:
        if explicit_urdf_path.exists():
            return explicit_urdf_path
        raise RuntimeError(f"Configured PSM URDF path does not exist: {explicit_urdf_path}")

    normalized_tool_type = _normalize_psm_tool_type(tool_type)
    candidate_filename = _PSM_TOOL_URDF_FILENAMES[normalized_tool_type]

    package_assets = Path(__file__).resolve().parent / "assets"
    search_roots: list[Path] = [package_assets]
    explicit_root = _normalize_optional_path(resolve_root)
    if explicit_root is not None:
        search_roots.append(explicit_root)
        search_roots.append(explicit_root / "assets")

    env_root = _normalize_optional_path(os.environ.get("CRESSIM_NEO_PSM_RESOLVE_ROOT"))
    if env_root is not None:
        search_roots.append(env_root)
        search_roots.append(env_root / "assets")

    for root in search_roots:
        local_candidate = root / "models" / "psm" / candidate_filename
        if local_candidate.exists():
            return local_candidate

        extern_candidate = root / "extern" / "SurRoL" / "surrol" / "assets" / "psm" / candidate_filename
        if extern_candidate.exists():
            return extern_candidate

        if normalized_tool_type == _DEFAULT_PSM_TOOL_TYPE:
            fallback_candidate = root / "extern" / "SurRoL" / "surrol" / "assets" / "psm" / "psm.urdf"
            if fallback_candidate.exists():
                return fallback_candidate

    raise RuntimeError(
        f"Failed to locate the PSM URDF for tool type {normalized_tool_type!r}: "
        f"assets/models/psm/{candidate_filename} or "
        f"extern/SurRoL/surrol/assets/psm/{candidate_filename}. "
        "Pass `resolve_root=...`, `urdf_path=...`, or set "
        "`CRESSIM_NEO_PSM_RESOLVE_ROOT`."
    )


def _normalize_psm_tool_type(tool_type: str | None) -> str:
    raw_value = _DEFAULT_PSM_TOOL_TYPE if tool_type is None else str(tool_type).strip().lower()
    if not raw_value:
        raw_value = _DEFAULT_PSM_TOOL_TYPE
    normalized = _PSM_TOOL_ALIASES.get(raw_value)
    if normalized is None:
        supported = ", ".join(sorted(_PSM_TOOL_URDF_FILENAMES))
        raise RuntimeError(
            f"Unsupported PSM tool type {tool_type!r}. Supported tool types: {supported}."
        )
    return normalized


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


def _scale_transform_translation(transform: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.array(transform, copy=True)
    scaled[:3, 3] *= float(scale)
    return scaled


def _urdf_to_engine_root_matrix(scale: float) -> np.ndarray:
    return _pose_matrix((0.0, _BASE_HEIGHT * scale, 0.0), (0.5 * math.pi, math.pi, 0.0))


def _convert_psm_local_frame(transform: np.ndarray) -> np.ndarray:
    return _PSM_ASSET_BASIS @ transform @ _PSM_ASSET_BASIS


def _convert_psm_local_direction(direction: Iterable[float]) -> np.ndarray:
    converted = _PSM_ASSET_BASIS[:3, :3] @ np.asarray(list(direction), dtype=np.float64)
    length = float(np.linalg.norm(converted))
    if length <= 1.0e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return converted / length


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


def _env_origin(env_index: int, env_count: int, env_spacing: float) -> np.ndarray:
    cols = max(1, math.ceil(math.sqrt(float(env_count))))
    row = env_index // cols
    col = env_index % cols
    x_offset = (col - 0.5 * (cols - 1)) * env_spacing
    y_offset = (row - 0.5 * (max(1, math.ceil(env_count / cols)) - 1)) * env_spacing
    return np.array([x_offset, y_offset, 0.0], dtype=np.float64)


def _parse_obj_mtllib_paths(obj_path: Path) -> list[Path]:
    mtllib_paths: list[Path] = []
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("mtllib "):
                    continue
                relative_path = stripped[7:].strip()
                if relative_path:
                    mtllib_paths.append((obj_path.parent / relative_path).resolve())
    except OSError:
        return []
    return mtllib_paths


def _parse_mtl_base_color_texture(mtl_path: Path) -> Path | None:
    try:
        with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("map_Kd "):
                    continue
                relative_path = stripped[7:].strip()
                if relative_path:
                    texture_path = (mtl_path.parent / relative_path).resolve()
                    if texture_path.exists():
                        return texture_path
    except OSError:
        return None
    return None


def _find_visual_base_color_texture(mesh_path: Path) -> Path | None:
    lower_suffix = mesh_path.suffix.lower()
    if lower_suffix == ".obj":
        for mtl_path in _parse_obj_mtllib_paths(mesh_path):
            texture_path = _parse_mtl_base_color_texture(mtl_path)
            if texture_path is not None and texture_path.suffix.lower() == ".png":
                return texture_path

    for suffix in (".png",):
        candidate = mesh_path.with_suffix(suffix)
        if candidate.exists():
            return candidate.resolve()
    return None


def _normalize_obj_index(raw_index: str, count: int) -> int | None:
    if not raw_index:
        return None
    index = int(raw_index)
    if index > 0:
        normalized = index - 1
    else:
        normalized = count + index
    if normalized < 0 or normalized >= count:
        return None
    return normalized


def _transform_position(
    local_position: np.ndarray,
    transform_linear: np.ndarray,
    transform_translation: np.ndarray,
    scale_vector: np.ndarray,
) -> np.ndarray:
    scaled_position = local_position * scale_vector
    return transform_linear @ (_PSM_ASSET_BASIS[:3, :3] @ scaled_position) + transform_translation


def _transform_normal(local_normal: np.ndarray, normal_transform: np.ndarray) -> np.ndarray:
    world_normal = normal_transform @ local_normal
    length = float(np.linalg.norm(world_normal))
    if length <= 1.0e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return world_normal / length


def _append_triangle_vertex(
    vertices: list,
    indices: list[int],
    position: np.ndarray,
    normal: np.ndarray,
    uv: tuple[float, float] | None = None,
) -> None:
    vertex = neo.MeshVertex()
    vertex.position = neo.Float3(float(position[0]), float(position[1]), float(position[2]))
    vertex.normal = neo.Float3(float(normal[0]), float(normal[1]), float(normal[2]))
    if uv is not None:
        vertex.tex_coord_u = float(uv[0])
        vertex.tex_coord_v = float(1.0 - uv[1])
    indices.append(len(vertices))
    vertices.append(vertex)


def _load_obj_mesh_desc(
    path: Path,
    transform: np.ndarray,
    scale: list[float] | None,
    debug_name: str,
) -> tuple[neo.MeshResourceDesc, tuple[np.ndarray, np.ndarray]]:
    positions: list[np.ndarray] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[np.ndarray] = []
    faces: list[list[tuple[int, int | None, int | None]]] = []

    scale_vector = np.asarray(scale if scale is not None else (1.0, 1.0, 1.0), dtype=np.float64)
    transform_linear = transform[:3, :3]
    transform_translation = transform[:3, 3]
    normal_transform = transform_linear @ _PSM_ASSET_BASIS[:3, :3]

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if not tokens:
                continue

            if tokens[0] == "v" and len(tokens) >= 4:
                positions.append(
                    _transform_position(
                        np.array((float(tokens[1]), float(tokens[2]), float(tokens[3])), dtype=np.float64),
                        transform_linear,
                        transform_translation,
                        scale_vector,
                    )
                )
            elif tokens[0] == "vt" and len(tokens) >= 3:
                texcoords.append((float(tokens[1]), float(tokens[2])))
            elif tokens[0] == "vn" and len(tokens) >= 4:
                normals.append(
                    _transform_normal(
                        np.array((float(tokens[1]), float(tokens[2]), float(tokens[3])), dtype=np.float64),
                        normal_transform,
                    )
                )
            elif tokens[0] == "f" and len(tokens) >= 4:
                face: list[tuple[int, int | None, int | None]] = []
                for face_token in tokens[1:]:
                    parts = face_token.split("/")
                    position_index = _normalize_obj_index(parts[0], len(positions))
                    texcoord_index = (
                        _normalize_obj_index(parts[1], len(texcoords))
                        if len(parts) > 1 and parts[1]
                        else None
                    )
                    normal_index = (
                        _normalize_obj_index(parts[2], len(normals))
                        if len(parts) > 2 and parts[2]
                        else None
                    )
                    if position_index is None:
                        raise RuntimeError(f"OBJ face in {path} references an invalid position index.")
                    face.append((position_index, texcoord_index, normal_index))
                faces.append(face)

    if not positions or not faces:
        raise RuntimeError(f"Mesh {debug_name} contains no renderable OBJ geometry.")

    desc = neo.MeshResourceDesc()
    desc.debug_name = debug_name
    vertices = []
    indices: list[int] = []
    bounds_min = np.array(positions[0], copy=True)
    bounds_max = np.array(positions[0], copy=True)
    for position in positions[1:]:
        bounds_min = np.minimum(bounds_min, position)
        bounds_max = np.maximum(bounds_max, position)

    for face in faces:
        if len(face) < 3:
            continue
        for triangle_index in range(1, len(face) - 1):
            triangle = [face[0], face[triangle_index], face[triangle_index + 1]]
            p0 = positions[triangle[0][0]]
            p1 = positions[triangle[1][0]]
            p2 = positions[triangle[2][0]]
            fallback_normal = np.cross(p1 - p0, p2 - p0)
            length = float(np.linalg.norm(fallback_normal))
            if length <= 1.0e-8:
                fallback_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                fallback_normal /= length

            for position_index, texcoord_index, normal_index in triangle:
                normal = normals[normal_index] if normal_index is not None else fallback_normal
                uv = texcoords[texcoord_index] if texcoord_index is not None else None
                _append_triangle_vertex(vertices, indices, positions[position_index], normal, uv)

    if not vertices:
        raise RuntimeError(f"Mesh {debug_name} contains no triangulated OBJ geometry.")

    desc.vertices = vertices
    desc.indices = indices
    return desc, (bounds_min, bounds_max)


def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + triangle_count * 50 == len(data)


def _load_stl_mesh_desc(
    path: Path,
    transform: np.ndarray,
    scale: list[float] | None,
    debug_name: str,
) -> tuple[neo.MeshResourceDesc, tuple[np.ndarray, np.ndarray]]:
    data = path.read_bytes()
    scale_vector = np.asarray(scale if scale is not None else (1.0, 1.0, 1.0), dtype=np.float64)
    transform_linear = transform[:3, :3]
    transform_translation = transform[:3, 3]
    normal_transform = transform_linear @ _PSM_ASSET_BASIS[:3, :3]

    desc = neo.MeshResourceDesc()
    desc.debug_name = debug_name
    vertices = []
    indices: list[int] = []
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None

    def accumulate_triangle(normal: np.ndarray, triangle_positions: list[np.ndarray]) -> None:
        nonlocal bounds_min, bounds_max
        transformed_normal = _transform_normal(normal, normal_transform)
        for position in triangle_positions:
            bounds_min = np.array(position, copy=True) if bounds_min is None else np.minimum(bounds_min, position)
            bounds_max = np.array(position, copy=True) if bounds_max is None else np.maximum(bounds_max, position)
        for corner_index in (0, 1, 2):
            _append_triangle_vertex(vertices, indices, triangle_positions[corner_index], transformed_normal)

    if _is_binary_stl(data):
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        offset = 84
        for _ in range(triangle_count):
            normal = np.array(struct.unpack_from("<3f", data, offset), dtype=np.float64)
            offset += 12
            triangle_positions = []
            for _corner in range(3):
                position = np.array(struct.unpack_from("<3f", data, offset), dtype=np.float64)
                offset += 12
                triangle_positions.append(
                    _transform_position(position, transform_linear, transform_translation, scale_vector)
                )
            offset += 2
            accumulate_triangle(normal, triangle_positions)
    else:
        current_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        current_positions: list[np.ndarray] = []
        for raw_line in data.decode("utf-8", errors="ignore").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) >= 5 and tokens[0] == "facet" and tokens[1] == "normal":
                current_normal = np.array(
                    (float(tokens[2]), float(tokens[3]), float(tokens[4])),
                    dtype=np.float64,
                )
                current_positions = []
            elif len(tokens) >= 4 and tokens[0] == "vertex":
                current_positions.append(
                    _transform_position(
                        np.array((float(tokens[1]), float(tokens[2]), float(tokens[3])), dtype=np.float64),
                        transform_linear,
                        transform_translation,
                        scale_vector,
                    )
                )
            elif tokens[0] == "endfacet" and len(current_positions) == 3:
                accumulate_triangle(current_normal, current_positions)
                current_positions = []

    if not vertices or bounds_min is None or bounds_max is None:
        raise RuntimeError(f"Mesh {debug_name} contains no STL geometry.")

    desc.vertices = vertices
    desc.indices = indices
    return desc, (bounds_min, bounds_max)


def _load_visual_mesh_desc(
    mesh_path: Path,
    transform: np.ndarray,
    scale: list[float] | None,
    debug_name: str,
) -> tuple[neo.MeshResourceDesc, tuple[np.ndarray, np.ndarray]]:
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_mesh_desc(mesh_path, transform, scale, debug_name)
    if suffix == ".stl":
        return _load_stl_mesh_desc(mesh_path, transform, scale, debug_name)
    raise RuntimeError(f"Unsupported visual mesh format for {debug_name}: {mesh_path}")


def _read_png_rgba8(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if len(data) < len(signature) or data[: len(signature)] != signature:
        raise RuntimeError(f"Unsupported texture format for {path}: expected a PNG file.")

    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace_method = 0
    idat_chunks: list[bytes] = []
    offset = len(signature)

    while offset + 8 <= len(data):
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_length
        if chunk_data_end + 4 > len(data):
            raise RuntimeError(f"Corrupted PNG texture: {path}")
        chunk_data = data[chunk_data_start:chunk_data_end]
        offset = chunk_data_end + 4

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter_method, interlace_method = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width <= 0 or height <= 0:
        raise RuntimeError(f"PNG texture is missing image dimensions: {path}")
    if bit_depth != 8:
        raise RuntimeError(f"Unsupported PNG bit depth for {path}: expected 8, got {bit_depth}.")
    if interlace_method != 0:
        raise RuntimeError(f"Unsupported interlaced PNG texture: {path}")
    if color_type not in (2, 6):
        raise RuntimeError(
            f"Unsupported PNG color type for {path}: expected RGB or RGBA, got {color_type}."
        )

    bytes_per_pixel = 4 if color_type == 6 else 3
    stride = width * bytes_per_pixel
    decoded = zlib.decompress(b"".join(idat_chunks))
    expected_size = (stride + 1) * height
    if len(decoded) != expected_size:
        raise RuntimeError(
            f"PNG texture {path} decoded to an unexpected size: "
            f"expected {expected_size}, got {len(decoded)}."
        )

    rows = bytearray(width * height * bytes_per_pixel)
    previous_row = bytearray(stride)
    source_offset = 0
    target_offset = 0
    for _ in range(height):
        filter_type = decoded[source_offset]
        source_offset += 1
        raw_row = decoded[source_offset : source_offset + stride]
        source_offset += stride
        row = bytearray(stride)

        if filter_type == 0:
            row[:] = raw_row
        elif filter_type == 1:
            for index in range(stride):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                row[index] = (raw_row[index] + left) & 0xFF
        elif filter_type == 2:
            for index in range(stride):
                row[index] = (raw_row[index] + previous_row[index]) & 0xFF
        elif filter_type == 3:
            for index in range(stride):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous_row[index]
                row[index] = (raw_row[index] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(stride):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous_row[index]
                up_left = previous_row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                row[index] = (raw_row[index] + predictor) & 0xFF
        else:
            raise RuntimeError(f"Unsupported PNG filter type {filter_type} in {path}.")

        rows[target_offset : target_offset + stride] = row
        target_offset += stride
        previous_row = row

    if color_type == 6:
        return width, height, bytes(rows)

    rgba = bytearray(width * height * 4)
    src = 0
    dst = 0
    while src < len(rows):
        rgba[dst] = rows[src]
        rgba[dst + 1] = rows[src + 1]
        rgba[dst + 2] = rows[src + 2]
        rgba[dst + 3] = 255
        src += 3
        dst += 4
    return width, height, bytes(rgba)


def _scale_inertia_diag(inertia_diag: _InertiaDiag, scale: float) -> _InertiaDiag:
    return (
        float(inertia_diag[0] * scale),
        float(inertia_diag[1] * scale),
        float(inertia_diag[2] * scale),
    )


def _resolve_link_dynamics(
    link_name: str,
    link_data: dict,
    mesh_inertia_diagonals: dict[str, _InertiaDiag],
    global_scale: float,
) -> tuple[float, _InertiaDiag]:
    mass = float(link_data["mass"])
    inertia_diag = tuple(float(value) for value in link_data["inertia_diag"])
    fallback = _LINK_DYNAMICS_FALLBACKS.get(link_name)
    has_valid_inertia = inertia_diag[0] > 0.0 and inertia_diag[1] > 0.0 and inertia_diag[2] > 0.0

    if fallback is not None and (mass <= 0.0 or not has_valid_inertia):
        fallback_mass, fallback_inertia = fallback
        scale_cubed = global_scale * global_scale * global_scale
        scale_quintic = scale_cubed * global_scale * global_scale
        return (
            fallback_mass * scale_cubed,
            _scale_inertia_diag(
                fallback_inertia,
                _FALLBACK_INERTIA_SCALE * scale_quintic,
            ),
        )

    if not has_valid_inertia:
        fallback_diag = mesh_inertia_diagonals.get(link_name, _DEFAULT_INERTIA_DIAG)
        inertia_diag = (
            max(mass * fallback_diag[0] * _FALLBACK_INERTIA_SCALE, 1.0e-6),
            max(mass * fallback_diag[1] * _FALLBACK_INERTIA_SCALE, 1.0e-6),
            max(mass * fallback_diag[2] * _FALLBACK_INERTIA_SCALE, 1.0e-6),
        )

    return mass, inertia_diag


def _resolve_jaw_limits(gripper1: dict, gripper2: dict) -> tuple[float, float]:
    gripper1_range = (-float(gripper1["upper"]), -float(gripper1["lower"]))
    gripper2_range = (float(gripper2["lower"]), float(gripper2["upper"]))
    lower = max(gripper1_range[0], gripper2_range[0])
    upper = min(gripper1_range[1], gripper2_range[1])
    if lower > upper:
        raise RuntimeError(
            "PSM gripper mimic limits are inconsistent and cannot be mapped to a single jaw command."
        )
    return lower, upper


def _scale_psm_authoring_data(links: dict, joints: dict, scale: float) -> None:
    if abs(scale - 1.0) <= 1.0e-8:
        return

    scale_squared = scale * scale
    scale_cubed = scale_squared * scale
    scale_quintic = scale_cubed * scale_squared

    for link_data in links.values():
        link_data["mass"] = float(link_data["mass"]) * scale_cubed
        inertia_diag = tuple(float(value) for value in link_data["inertia_diag"])
        link_data["inertia_diag"] = tuple(value * scale_quintic for value in inertia_diag)
        for visual in link_data["visuals"]:
            visual["origin"] = _scale_transform_translation(visual["origin"], scale)
            visual_scale = visual["scale"]
            if visual_scale is None:
                visual["scale"] = [scale, scale, scale]
            else:
                visual["scale"] = [float(value) * scale for value in visual_scale]

    for joint in joints.values():
        joint["origin"] = _scale_transform_translation(joint["origin"], scale)
        if joint["type"] == "prismatic":
            joint["lower"] = float(joint["lower"]) * scale
            joint["upper"] = float(joint["upper"]) * scale
            joint["velocity"] = float(joint["velocity"]) * scale


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
            xyz = _parse_floats(
                origin.attrib.get("xyz") if origin is not None else None,
                3,
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_floats(
                origin.attrib.get("rpy") if origin is not None else None,
                3,
                (0.0, 0.0, 0.0),
            )
            scale = (
                _parse_floats(mesh.attrib.get("scale"), 3, (1.0, 1.0, 1.0))
                if mesh.attrib.get("scale")
                else None
            )

            material_name = "default"
            material_node = visual.find("material")
            if material_node is not None:
                material_name = material_node.attrib.get("name", material_name)
                inline_color = material_node.find("color")
                if inline_color is not None:
                    rgba = _parse_floats(
                        inline_color.attrib.get("rgba"),
                        4,
                        (1.0, 1.0, 1.0, 1.0),
                    )
                    materials[f"{link_name}.visual.{visual_index}"] = tuple(rgba)
                    material_name = f"{link_name}.visual.{visual_index}"

            visuals.append(
                {
                    "mesh_path": urdf_path.parent / mesh.attrib["filename"],
                    "origin": _convert_psm_local_frame(_pose_matrix(xyz, rpy)),
                    "scale": scale,
                    "material_name": material_name,
                    "base_color_texture_path": _find_visual_base_color_texture(
                        urdf_path.parent / mesh.attrib["filename"]
                    ),
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
        xyz = _parse_floats(
            origin.attrib.get("xyz") if origin is not None else None,
            3,
            (0.0, 0.0, 0.0),
        )
        rpy = _parse_floats(
            origin.attrib.get("rpy") if origin is not None else None,
            3,
            (0.0, 0.0, 0.0),
        )
        axis_values = _parse_floats(
            axis.attrib.get("xyz") if axis is not None else None,
            3,
            (1.0, 0.0, 0.0),
        )
        lower = float(limit.attrib.get("lower", "0.0")) if limit is not None else 0.0
        upper = float(limit.attrib.get("upper", "0.0")) if limit is not None else 0.0
        velocity = float(limit.attrib.get("velocity", "0.0")) if limit is not None else 0.0
        joints[name] = {
            "type": joint.attrib["type"],
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "origin": _convert_psm_local_frame(_pose_matrix(xyz, rpy)),
            "axis": _convert_psm_local_direction(axis_values),
            "lower": lower,
            "upper": upper,
            "velocity": velocity,
        }

    return materials, links, joints


class _PsmAuthor:
    def __init__(
        self,
        world: neo.World,
        resources: neo.RenderResourceManager,
        config: PsmAuthoringConfig,
    ) -> None:
        self.world = world
        self.resources = resources
        self.config = config
        self._materials: dict[str, neo.MaterialHandle] = {}
        self._mesh_handles: dict[str, neo.MeshHandle] = {}
        self._mesh_inertia_diagonals: dict[str, tuple[float, float, float]] = {}
        self._textures: dict[str, neo.TextureHandle] = {}
        self._normalized_tool_type = _normalize_psm_tool_type(config.tool_type)

    @staticmethod
    def _authored_joint_names(joints: dict[str, dict], requested_names: Iterable[str]) -> list[str]:
        return [joint_name for joint_name in requested_names if joint_name in joints]

    def author(self) -> PsmBuildResult:
        urdf_path = _find_psm_urdf_path(
            resolve_root=self.config.resolve_root,
            urdf_path=self.config.urdf_path,
            tool_type=self.config.tool_type,
        )
        materials, links, joints = _parse_psm_urdf(urdf_path)
        _scale_psm_authoring_data(links, joints, self.config.global_scale)
        authored_physical_joint_names = self._authored_joint_names(joints, _PHYSICAL_JOINT_ORDER)

        if self.config.add_ground:
            self._materials["ground"] = self._register_material(
                "PsmBuilder.Ground",
                (0.35, 0.36, 0.40, 1.0),
                0.95,
            )

        retained_links = {"psm_base_link"}
        retained_links.update(joints[joint_name]["child"] for joint_name in authored_physical_joint_names)
        retained_joints = {joint_name: joints[joint_name] for joint_name in authored_physical_joint_names}
        material_handles = {}
        for link_name in retained_links:
            primary_visual = links[link_name]["visuals"][0]
            texture_handle = None
            texture_path = primary_visual.get("base_color_texture_path")
            if texture_path is not None:
                texture_handle = self._register_texture(
                    Path(texture_path),
                    f"PsmBuilder.Texture.{link_name}",
                )
            material_handles[link_name] = self._register_material(
                f"PsmBuilder.Material.{link_name}",
                _srgb_rgba_to_linear(
                    materials.get(primary_visual["material_name"], (0.8, 0.8, 0.8, 1.0))
                ),
                0.55,
                texture_handle,
            )

        result = PsmBuildResult(urdf_path=urdf_path, env_count=self.config.env_count)
        for env_index in range(self.config.env_count):
            env_origin = _env_origin(env_index, self.config.env_count, self.config.env_spacing)
            instance, ground_entity, camera_entity, light_entities = self._author_environment(
                env_index,
                env_origin,
                links,
                retained_joints,
                authored_physical_joint_names,
                material_handles,
            )
            result.instances.append(instance)
            if ground_entity is not None:
                result.ground_entities.append(ground_entity)
            if camera_entity is not None:
                result.camera_entities.append(camera_entity)
            result.light_entities.append(light_entities)
        return result

    def _register_material(
        self,
        debug_name: str,
        rgba: tuple[float, float, float, float],
        roughness: float,
        base_color_texture: neo.TextureHandle | None = None,
    ) -> neo.MaterialHandle:
        if debug_name in self._materials:
            return self._materials[debug_name]
        material = neo.MaterialResourceDesc()
        material.debug_name = debug_name
        if base_color_texture is not None:
            material.base_color = neo.Float3(1.0, 1.0, 1.0)
        else:
            material.base_color = neo.Float3(float(rgba[0]), float(rgba[1]), float(rgba[2]))
        material.roughness = float(roughness)
        material.opacity = float(rgba[3])
        if base_color_texture is not None:
            material.base_color_texture = base_color_texture
        self._materials[debug_name] = self.resources.register_material(material)
        return self._materials[debug_name]

    def _register_texture(self, texture_path: Path, debug_name: str) -> neo.TextureHandle:
        cache_key = str(texture_path)
        if cache_key in self._textures:
            return self._textures[cache_key]

        width, height, pixel_data = _read_png_rgba8(texture_path)
        texture = neo.TextureResourceDesc()
        texture.debug_name = debug_name
        texture.width = int(width)
        texture.height = int(height)
        texture.pixel_format = neo.TexturePixelFormat.RGBA8
        texture.color_space = neo.TextureColorSpace.Srgb
        texture_subresource = neo.TextureSubresourceDesc()
        texture_subresource.pixel_data = list(pixel_data)
        texture.subresources = [texture_subresource]
        handle = self.resources.register_texture(texture)
        self._textures[cache_key] = handle
        return handle

    def _register_link_mesh(self, link_name: str, link_data: dict) -> neo.MeshHandle:
        if link_name in self._mesh_handles:
            return self._mesh_handles[link_name]

        mesh_debug_name = f"PsmBuilder.Mesh.{link_name}"
        bounds_min: np.ndarray | None = None
        bounds_max: np.ndarray | None = None
        vertices: list[neo.MeshVertex] = []
        indices: list[int] = []

        for visual_index, visual in enumerate(link_data["visuals"]):
            visual_desc, visual_bounds = _load_visual_mesh_desc(
                Path(visual["mesh_path"]),
                visual["origin"],
                visual["scale"],
                f"{mesh_debug_name}.{visual_index}",
            )
            vertex_offset = len(vertices)
            vertices.extend(list(visual_desc.vertices))
            indices.extend(vertex_offset + int(index) for index in visual_desc.indices)

            if bounds_min is None or bounds_max is None:
                bounds_min = np.array(visual_bounds[0], copy=True)
                bounds_max = np.array(visual_bounds[1], copy=True)
            else:
                bounds_min = np.minimum(bounds_min, visual_bounds[0])
                bounds_max = np.maximum(bounds_max, visual_bounds[1])

        if bounds_min is None or bounds_max is None:
            raise RuntimeError(f"Expected at least one visual mesh for {link_name}.")

        extents = np.maximum(bounds_max - bounds_min, 1.0e-4)
        self._mesh_inertia_diagonals[link_name] = (
            float((extents[1] * extents[1] + extents[2] * extents[2]) / 12.0),
            float((extents[0] * extents[0] + extents[2] * extents[2]) / 12.0),
            float((extents[0] * extents[0] + extents[1] * extents[1]) / 12.0),
        )
        mesh_desc = neo.MeshResourceDesc()
        mesh_desc.debug_name = mesh_debug_name
        mesh_desc.vertices = vertices
        mesh_desc.indices = indices
        handle = self.resources.register_mesh(mesh_desc)
        self._mesh_handles[link_name] = handle
        return handle

    def _maybe_add_special_link_collider(self, link_name: str, link_entity: int) -> None:
        if self._normalized_tool_type != "suction_irrigator" or link_name != "psm_tool_yaw_link":
            return
        collider = neo.ColliderComponent()
        collider.shape_type = neo.ColliderShapeType.Capsule
        collider.local_position = neo.Float3(
            float(_SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[0] * self.config.global_scale),
            float(_SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[1] * self.config.global_scale),
            float(_SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[2] * self.config.global_scale),
        )
        collider.shape_params = neo.Float4(
            float(_SUCTION_IRRIGATOR_YAW_CAPSULE_RADIUS * self.config.global_scale),
            float(_SUCTION_IRRIGATOR_YAW_CAPSULE_HALF_HEIGHT * self.config.global_scale),
            0.0,
            0.0,
        )
        collider.friction = 1.0
        collider.static_friction = 1.0
        self.world.add_collider(link_entity, collider)

    def _author_environment(
        self,
        env_index: int,
        env_origin: np.ndarray,
        links: dict,
        joints: dict,
        authored_physical_joint_names: list[str],
        material_handles: dict[str, neo.MaterialHandle],
    ) -> tuple[PsmRobotInstance, int | None, int | None, list[int]]:
        ground_entity: int | None = None
        if self.config.add_ground:
            ground_entity = self.world.create_entity(env_index)
            ground_transform = neo.TransformComponent()
            ground_transform.world_transform.position = neo.Float3(
                float(env_origin[0]),
                float(-0.02 * self.config.global_scale),
                float(env_origin[1]),
            )
            ground_transform.world_transform.scale = neo.Float3(1.0, 1.0, 1.0)
            self.world.set_transform(ground_entity, ground_transform)
            ground_mesh = self.resources.register_mesh(
                neo.make_plane_mesh(
                    _GROUND_HALF_EXTENT * self.config.global_scale,
                    f"PsmBuilder.Ground.{env_index}",
                    1.0,
                )
            )
            ground_renderer = neo.MeshRendererComponent()
            ground_renderer.mesh = ground_mesh
            ground_renderer.material = self._materials["ground"]
            ground_renderer.visible = True
            self.world.set_mesh_renderer(ground_entity, ground_renderer)

        camera_entity: int | None = None
        if self.config.add_default_camera:
            camera_entity = self.world.create_entity(env_index)
            camera_offset = self.config.global_scale * np.array([1.25, 0.95, -1.10], dtype=np.float64)
            camera_target_offset = self.config.global_scale * np.array(
                [0.0, 0.30, 0.0],
                dtype=np.float64,
            )
            camera_position = np.array(
                [
                    env_origin[0] + camera_offset[0],
                    camera_offset[1],
                    env_origin[2] + camera_offset[2],
                ],
                dtype=np.float64,
            )
            camera_target = np.array(
                [
                    env_origin[0] + camera_target_offset[0],
                    camera_target_offset[1],
                    env_origin[2] + camera_target_offset[2],
                ],
                dtype=np.float64,
            )
            camera_transform = neo.TransformComponent()
            camera_transform.world_transform.position = neo.Float3(
                float(camera_position[0]),
                float(camera_position[1]),
                float(camera_position[2]),
            )
            camera_transform.world_transform.rotation = _look_rotation(camera_position, camera_target)
            self.world.set_transform(camera_entity, camera_transform)
            camera = neo.CameraComponent()
            camera.vertical_fov_degrees = 55.0
            camera.near_clip = float(0.01 * self.config.global_scale)
            camera.far_clip = float(max(20.0 * self.config.global_scale, 20.0))
            camera.clear_color = True
            camera.clear_depth = True
            camera.clear_color_value = neo.Float4(0.84, 0.90, 0.98, 1.0)
            self.world.set_camera(camera_entity, camera)

        light_entities: list[int] = []
        if self.config.add_default_lighting:
            light_entity = self.world.create_entity(env_index)
            light = neo.DirectionalLightComponent()
            light.direction = neo.Float3(-0.35, -1.0, 0.25)
            light.color = neo.Float3(1.0, 1.0, 1.0)
            light.intensity = 7.5
            self.world.set_directional_light(light_entity, light)
            light_entities.append(light_entity)

        link_world = {"psm_base_link": _urdf_to_engine_root_matrix(self.config.global_scale)}
        link_world["psm_base_link"][:3, 3] += np.array([env_origin[0], 0.0, env_origin[1]])
        for joint_name in authored_physical_joint_names:
            joint = joints[joint_name]
            link_world[joint["child"]] = link_world[joint["parent"]] @ joint["origin"]

        link_entities: dict[str, int] = {}
        base_entity = 0
        for link_name in ["psm_base_link", *(joints[name]["child"] for name in authored_physical_joint_names)]:
            link_entity = self.world.create_entity(env_index)
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
            self.world.set_transform(link_entity, transform)

            mesh_handle = self._register_link_mesh(link_name, links[link_name])
            renderer = neo.MeshRendererComponent()
            renderer.mesh = mesh_handle
            renderer.material = material_handles[link_name]
            renderer.visible = link_name not in _HIDDEN_LINKS
            self.world.set_mesh_renderer(link_entity, renderer)

            rigid_body = neo.RigidBodyComponent()
            if link_name == "psm_base_link":
                rigid_body.body_type = neo.RigidBodyType.Static
                rigid_body.inverse_mass = 0.0
                rigid_body.inverse_inertia_local = neo.Float3(0.0, 0.0, 0.0)
            else:
                mass, inertia_diag = _resolve_link_dynamics(
                    link_name,
                    links[link_name],
                    self._mesh_inertia_diagonals,
                    self.config.global_scale,
                )
                inverse_mass = 0.0 if mass <= 0.0 else 1.0 / mass
                rigid_body.body_type = neo.RigidBodyType.Dynamic
                rigid_body.inverse_mass = inverse_mass
                rigid_body.inverse_inertia_local = neo.Float3(
                    0.0 if inertia_diag[0] <= 0.0 else float(1.0 / inertia_diag[0]),
                    0.0 if inertia_diag[1] <= 0.0 else float(1.0 / inertia_diag[1]),
                    0.0 if inertia_diag[2] <= 0.0 else float(1.0 / inertia_diag[2]),
                )
            self.world.set_rigid_body(link_entity, rigid_body)
            self._maybe_add_special_link_collider(link_name, link_entity)

        physical_joint_ids: dict[str, int] = {}
        physical_joint_limits: dict[str, tuple[float, float]] = {}
        for joint_index, joint_name in enumerate(authored_physical_joint_names):
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
                _quaternion_conjugate(parent_rotation),
                anchor_world - parent_position,
            )
            local_anchor_b = _quaternion_rotate(
                _quaternion_conjugate(child_rotation),
                anchor_world - child_position,
            )
            local_rotation_a = _matrix_to_quaternion(
                parent_world_rotation.T @ solver_joint_world[:3, :3]
            )
            local_rotation_b = _matrix_to_quaternion(
                child_world_rotation.T @ solver_joint_world[:3, :3]
            )

            assigned_joint_id = env_index * 100 + joint_index + 1
            physical_joint_ids[joint_name] = assigned_joint_id
            physical_joint_limits[joint_name] = (joint["lower"], joint["upper"])

            if _JOINT_KIND[joint_name] == "hinge":
                state = neo.HingeJointState()
                state.joint_id = assigned_joint_id
                state.body_a = parent_entity
                state.body_b = child_entity
                state.local_anchor_a = neo.Float3(
                    float(local_anchor_a[0]),
                    float(local_anchor_a[1]),
                    float(local_anchor_a[2]),
                )
                state.local_anchor_b = neo.Float3(
                    float(local_anchor_b[0]),
                    float(local_anchor_b[1]),
                    float(local_anchor_b[2]),
                )
                state.local_rotation_a = local_rotation_a
                state.local_rotation_b = local_rotation_b
                state.limit_enabled = True
                state.limit_min = float(joint["lower"])
                state.limit_max = float(joint["upper"])
                if joint_name not in _PASSIVE_JOINT_ORDER:
                    state.drive_mode = neo.RigidJointDriveMode.TargetPosition
                    state.drive_target_angle = 0.0
                    state.drive_compliance = _HINGE_DRIVE_COMPLIANCE
                    state.drive_damping = _HINGE_DRIVE_DAMPING
                    state.drive_max_angular_velocity = max(
                        float(joint["velocity"]),
                        _DRIVE_MAX_ANGULAR_VELOCITY,
                    )
                if not self.world.upsert_hinge_joint(state):
                    raise RuntimeError(f"Failed to author hinge joint {joint_name}.")
            else:
                state = neo.SliderJointState()
                state.joint_id = assigned_joint_id
                state.body_a = parent_entity
                state.body_b = child_entity
                state.local_anchor_a = neo.Float3(
                    float(local_anchor_a[0]),
                    float(local_anchor_a[1]),
                    float(local_anchor_a[2]),
                )
                state.local_anchor_b = neo.Float3(
                    float(local_anchor_b[0]),
                    float(local_anchor_b[1]),
                    float(local_anchor_b[2]),
                )
                state.local_rotation_a = local_rotation_a
                state.local_rotation_b = local_rotation_b
                state.limit_enabled = True
                state.limit_min = float(joint["lower"])
                state.limit_max = float(joint["upper"])
                state.drive_mode = neo.RigidJointDriveMode.TargetPosition
                state.drive_target_position = 0.0
                state.drive_compliance = _SLIDER_DRIVE_COMPLIANCE
                state.drive_damping = _SLIDER_DRIVE_DAMPING
                state.drive_max_velocity = max(float(joint["velocity"]), _DRIVE_MAX_LINEAR_VELOCITY)
                if not self.world.upsert_slider_joint(state):
                    raise RuntimeError(f"Failed to author slider joint {joint_name}.")

        closure_joint_offset = 0
        if {"psm_pitch_bottom_link", "psm_pitch_end_link"}.issubset(link_entities):
            pitch_bottom_rotation = _matrix_to_quaternion(link_world["psm_pitch_bottom_link"])
            pitch_bottom_position = link_world["psm_pitch_bottom_link"][:3, 3]
            pitch_end_anchor_world = (
                link_world["psm_pitch_end_link"] @ links["psm_pitch_end_link"]["visuals"][0]["origin"]
            )[:3, 3]
            pitch_bottom_anchor_local = _quaternion_rotate(
                _quaternion_conjugate(pitch_bottom_rotation),
                pitch_end_anchor_world - pitch_bottom_position,
            )

            closure = neo.BallJointState()
            closure.joint_id = env_index * 100 + len(_PHYSICAL_JOINT_ORDER) + closure_joint_offset + 1
            closure.body_a = link_entities["psm_pitch_bottom_link"]
            closure.body_b = link_entities["psm_pitch_end_link"]
            closure.local_anchor_a = neo.Float3(
                float(pitch_bottom_anchor_local[0]),
                float(pitch_bottom_anchor_local[1]),
                float(pitch_bottom_anchor_local[2]),
            )
            pitch_end_anchor_local = links["psm_pitch_end_link"]["visuals"][0]["origin"][:3, 3]
            closure.local_anchor_b = neo.Float3(
                float(pitch_end_anchor_local[0]),
                float(pitch_end_anchor_local[1]),
                float(pitch_end_anchor_local[2]),
            )
            if not self.world.upsert_ball_joint(closure):
                raise RuntimeError("Failed to author pitch bottom/end closure joint.")
            closure_joint_offset += 1

        if {"psm_pitch_top_link", "psm_pitch_end_link"}.issubset(link_entities):
            pitch_top_rotation = _matrix_to_quaternion(link_world["psm_pitch_top_link"])
            pitch_top_position = link_world["psm_pitch_top_link"][:3, 3]
            pitch_end_top_anchor_world = (
                link_world["psm_pitch_end_link"]
                @ links["psm_pitch_end_link"]["visuals"][0]["origin"]
                @ _pose_matrix(
                    self.config.global_scale * _PITCH_END_TOP_CLOSURE_MESH_OFFSET,
                    (0.0, 0.0, 0.0),
                )
            )[:3, 3]
            pitch_top_anchor_local = _quaternion_rotate(
                _quaternion_conjugate(pitch_top_rotation),
                pitch_end_top_anchor_world - pitch_top_position,
            )

            closure = neo.BallJointState()
            closure.joint_id = env_index * 100 + len(_PHYSICAL_JOINT_ORDER) + closure_joint_offset + 1
            closure.body_a = link_entities["psm_pitch_top_link"]
            closure.body_b = link_entities["psm_pitch_end_link"]
            closure.local_anchor_a = neo.Float3(
                float(pitch_top_anchor_local[0]),
                float(pitch_top_anchor_local[1]),
                float(pitch_top_anchor_local[2]),
            )
            pitch_end_top_anchor_local = (
                links["psm_pitch_end_link"]["visuals"][0]["origin"]
                @ _pose_matrix(
                    self.config.global_scale * _PITCH_END_TOP_CLOSURE_MESH_OFFSET,
                    (0.0, 0.0, 0.0),
                )
            )[:3, 3]
            closure.local_anchor_b = neo.Float3(
                float(pitch_end_top_anchor_local[0]),
                float(pitch_end_top_anchor_local[1]),
                float(pitch_end_top_anchor_local[2]),
            )
            if not self.world.upsert_ball_joint(closure):
                raise RuntimeError("Failed to author pitch top/end closure joint.")
            closure_joint_offset += 1

        if {"psm_pitch_bottom_link", "psm_pitch_front_link"}.issubset(link_entities):
            pitch_bottom_rotation = _matrix_to_quaternion(link_world["psm_pitch_bottom_link"])
            pitch_bottom_position = link_world["psm_pitch_bottom_link"][:3, 3]
            pitch_front_rotation = _matrix_to_quaternion(link_world["psm_pitch_front_link"])
            pitch_front_position = link_world["psm_pitch_front_link"][:3, 3]
            pitch_bottom_front_anchor_world = (
                link_world["psm_pitch_bottom_link"]
                @ _pose_matrix(
                    self.config.global_scale * _PITCH_BOTTOM_FRONT_CLOSURE_OFFSET,
                    (0.0, 0.0, 0.0),
                )
            )[:3, 3]
            pitch_bottom_front_anchor_local = _quaternion_rotate(
                _quaternion_conjugate(pitch_bottom_rotation),
                pitch_bottom_front_anchor_world - pitch_bottom_position,
            )
            pitch_front_anchor_local = _quaternion_rotate(
                _quaternion_conjugate(pitch_front_rotation),
                pitch_bottom_front_anchor_world - pitch_front_position,
            )

            closure = neo.BallJointState()
            closure.joint_id = env_index * 100 + len(_PHYSICAL_JOINT_ORDER) + closure_joint_offset + 1
            closure.body_a = link_entities["psm_pitch_bottom_link"]
            closure.body_b = link_entities["psm_pitch_front_link"]
            closure.local_anchor_a = neo.Float3(
                float(pitch_bottom_front_anchor_local[0]),
                float(pitch_bottom_front_anchor_local[1]),
                float(pitch_bottom_front_anchor_local[2]),
            )
            closure.local_anchor_b = neo.Float3(
                float(pitch_front_anchor_local[0]),
                float(pitch_front_anchor_local[1]),
                float(pitch_front_anchor_local[2]),
            )
            if not self.world.upsert_ball_joint(closure):
                raise RuntimeError("Failed to author pitch bottom/front closure joint.")

        instance = PsmRobotInstance(
            env_index=env_index,
            base_entity=base_entity,
            link_entities=link_entities,
            arm_joint_ids=[physical_joint_ids.get(joint_name, _MISSING_JOINT_ID) for joint_name in _ARM_JOINT_ORDER],
            arm_joint_limits=[
                physical_joint_limits.get(joint_name, _MISSING_JOINT_LIMIT) for joint_name in _ARM_JOINT_ORDER
            ],
            passive_joint_ids={
                joint_name: physical_joint_ids[joint_name]
                for joint_name in _PASSIVE_JOINT_ORDER
                if joint_name in physical_joint_ids
            },
            passive_joint_limits={
                joint_name: physical_joint_limits[joint_name]
                for joint_name in _PASSIVE_JOINT_ORDER
                if joint_name in physical_joint_limits
            },
            jaw_joint_ids=(
                (
                    physical_joint_ids["psm_tool_gripper1_joint"],
                    physical_joint_ids["psm_tool_gripper2_joint"],
                )
                if "psm_tool_gripper1_joint" in physical_joint_ids
                and "psm_tool_gripper2_joint" in physical_joint_ids
                else _MISSING_JAW_IDS
            ),
            jaw_limit=(
                _resolve_jaw_limits(
                    joints["psm_tool_gripper1_joint"],
                    joints["psm_tool_gripper2_joint"],
                )
                if "psm_tool_gripper1_joint" in joints and "psm_tool_gripper2_joint" in joints
                else _MISSING_JOINT_LIMIT
            ),
        )
        return instance, ground_entity, camera_entity, light_entities


def get_psm_default_runtime_config(env_count: int = 1) -> neo.RuntimeConfig:
    if env_count < 1:
        raise ValueError(f"Expected env_count >= 1, got {env_count}.")
    config = neo.RuntimeConfig()
    config.scene_layout.env_count = env_count
    config.scene_layout.max_renderable_objects_per_env = 16
    config.scene_layout.max_lights_per_env = 3
    config.scene_layout.max_cameras_per_env = 1
    config.physics_desc.enable_blocking_readback = False
    config.physics_desc.substeps = 4
    config.physics_desc.default_iterations = 20
    return config


def author_psm_scene(
    world: neo.World,
    resources: neo.RenderResourceManager,
    config: PsmAuthoringConfig,
) -> PsmBuildResult:
    if config.env_count < 1:
        raise ValueError(f"Expected env_count >= 1, got {config.env_count}.")
    if config.global_scale <= 0.0:
        raise ValueError(f"Expected global_scale > 0, got {config.global_scale}.")
    return _PsmAuthor(world, resources, config).author()


def set_psm_joint_targets(
    world: neo.World,
    build_result: PsmBuildResult,
    targets: Iterable[float] | Iterable[Iterable[float]],
    env_index: int | None = None,
) -> None:
    if env_index is not None:
        _set_env_joint_targets(world, build_result, env_index, list(float(value) for value in targets))
        return

    target_list = list(targets)
    if build_result.env_count == 1 and target_list and not isinstance(
        target_list[0],
        (list, tuple, np.ndarray),
    ):
        _set_env_joint_targets(world, build_result, 0, list(float(value) for value in target_list))
        return

    if len(target_list) != build_result.env_count:
        raise ValueError(
            f"Expected {build_result.env_count} target vectors, got {len(target_list)}."
        )
    for current_env, env_targets in enumerate(target_list):
        _set_env_joint_targets(
            world,
            build_result,
            current_env,
            list(float(value) for value in env_targets),
        )


def _set_env_joint_targets(
    world: neo.World,
    build_result: PsmBuildResult,
    env_index: int,
    targets: list[float],
) -> None:
    if env_index < 0 or env_index >= build_result.env_count:
        raise ValueError(f"Environment index {env_index} is out of range.")
    if len(targets) not in (len(_ARM_JOINT_ORDER), len(_COMMAND_JOINT_ORDER)):
        raise ValueError(
            "Expected "
            f"{len(_ARM_JOINT_ORDER)} arm targets or {len(_COMMAND_JOINT_ORDER)} "
            f"arm-plus-jaw targets, got {len(targets)}."
        )

    instance = build_result.instances[env_index]
    arm_targets = targets[: len(_ARM_JOINT_ORDER)]
    jaw_target = 0.0 if len(targets) == len(_ARM_JOINT_ORDER) else float(targets[-1])

    for joint_name, joint_id, limits, target in zip(
        _ARM_JOINT_ORDER,
        instance.arm_joint_ids,
        instance.arm_joint_limits,
        arm_targets,
    ):
        clamped = min(max(float(target), limits[0]), limits[1])
        if int(joint_id) == _MISSING_JOINT_ID:
            continue
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

    jaw_clamped = min(max(jaw_target, instance.jaw_limit[0]), instance.jaw_limit[1])
    gripper_targets = (
        ("psm_tool_gripper1_joint", instance.jaw_joint_ids[0], jaw_clamped),
        ("psm_tool_gripper2_joint", instance.jaw_joint_ids[1], -jaw_clamped),
    )
    for joint_name, joint_id, target in gripper_targets:
        if int(joint_id) == _MISSING_JOINT_ID:
            continue
        state = world.try_get_hinge_joint(joint_id)
        if state is None:
            raise RuntimeError(f"Missing hinge joint state for {joint_name}.")
        state.drive_target_angle = float(target)
        if not world.upsert_hinge_joint(state):
            raise RuntimeError(f"Failed to update hinge joint {joint_name}.")


__all__ = [
    "PsmAuthoringConfig",
    "PsmBuildResult",
    "PsmRobotInstance",
    "author_psm_scene",
    "get_psm_default_runtime_config",
    "set_psm_joint_targets",
]
