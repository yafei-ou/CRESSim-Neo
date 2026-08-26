"""Render close-up reference screenshots of the two PSM tool types.

This deliberately still uses ``author_psm_scene``: the complete PSM is built so
the end-effector pose and tool meshes are exactly the same as in the other PSM
examples, but every non-tool mesh is hidden before rendering.

By default this writes one PNG for each tool to
``artifacts/screenshots/psm_tool_type_screenshot/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import zlib

import cressim_neo as neo
from cressim_neo_envs.psm_builder import (
    PsmAuthoringConfig,
    PsmBuildResult,
    author_psm_scene,
    get_psm_default_runtime_config,
    set_psm_joint_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts" / "screenshots" / "psm_tool_type_screenshot"
TOOL_TYPES = ("large_needle_driver", "suction_irrigator")
# Reference screenshot background: RGB (226, 226, 226).
BACKGROUND_GRAY = 226.0 / 255.0
CAMERA_ROLL_DEGREES = -42.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render isolated close-ups of PSM tool types.")
    parser.add_argument(
        "--tool-type",
        choices=(*TOOL_TYPES, "both"),
        default="both",
        help="Tool screenshot to render (default: both).",
    )
    parser.add_argument("--width", type=int, default=1400, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1000, help="Output height in pixels.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for the rendered PNG files."
    )
    return parser.parse_args()


def _look_rotation(position: tuple[float, float, float], target: tuple[float, float, float]) -> neo.Quaternion:
    """Return an engine camera rotation looking from ``position`` at ``target``."""
    forward = [target[index] - position[index] for index in range(3)]
    forward_length = math.sqrt(sum(component * component for component in forward))
    forward = [component / forward_length for component in forward]
    up = [0.0, 1.0, 0.0]
    right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    ]
    right_length = math.sqrt(sum(component * component for component in right))
    right = [component / right_length for component in right]
    corrected_up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ]
    m00, m01, m02 = right[0], corrected_up[0], forward[0]
    m10, m11, m12 = right[1], corrected_up[1], forward[1]
    m20, m21, m22 = right[2], corrected_up[2], forward[2]
    trace = m00 + m11 + m22
    rotation = neo.Quaternion()
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        rotation.w, rotation.x, rotation.y, rotation.z = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        rotation.w, rotation.x, rotation.y, rotation.z = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        rotation.w, rotation.x, rotation.y, rotation.z = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        rotation.w, rotation.x, rotation.y, rotation.z = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    return rotation


def _write_rgba_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Write the RGBA8 readback without adding Pillow as an example dependency."""
    expected_size = width * height * 4
    if len(pixels) != expected_size:
        raise RuntimeError(f"Expected {expected_size} RGBA bytes, got {len(pixels)}.")
    # Render-target readbacks are bottom-up; PNG rows are conventionally top-down.
    rows = [b"\x00" + pixels[row * width * 4 : (row + 1) * width * 4] for row in range(height - 1, -1, -1)]
    png = b"\x89PNG\r\n\x1a\n"
    for chunk_type, payload in (
        (b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
        (b"IDAT", zlib.compress(b"".join(rows), level=9)),
        (b"IEND", b""),
    ):
        png += struct.pack(">I", len(payload)) + chunk_type + payload
        png += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    path.write_bytes(png)


def _rotate_vector(rotation: neo.Quaternion, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate ``vector`` from a link-local frame into world space."""
    qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _multiply_quaternions(first: neo.Quaternion, second: neo.Quaternion) -> neo.Quaternion:
    """Compose two rotations, applying ``second`` in the first rotation's local frame."""
    result = neo.Quaternion()
    result.w = first.w * second.w - first.x * second.x - first.y * second.y - first.z * second.z
    result.x = first.w * second.x + first.x * second.w + first.y * second.z - first.z * second.y
    result.y = first.w * second.y - first.x * second.z + first.y * second.w + first.z * second.x
    result.z = first.w * second.z + first.x * second.y - first.y * second.x + first.z * second.w
    return result


def _rolled_look_rotation(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> neo.Quaternion:
    """Look at the tooltip, then roll the image to match the reference framing."""
    half_roll_radians = 0.5 * math.radians(CAMERA_ROLL_DEGREES)
    roll = neo.Quaternion()
    roll.z = math.sin(half_roll_radians)
    roll.w = math.cos(half_roll_radians)
    return _multiply_quaternions(_look_rotation(position, target), roll)


def _show_only_tool_meshes(
    world: neo.World,
    build: PsmBuildResult,
    tool_type: str,
) -> None:
    # Keep just the end-effector assembly.  The builder still authors all links
    # and joints, preserving the usual PSM kinematic chain and tool geometry.
    visible_links = {
        "psm_tool_roll_link",
        "psm_tool_pitch_link",
        "psm_tool_yaw_link",
        "psm_tool_gripper1_link",
        "psm_tool_gripper2_link",
    }
    if tool_type == "suction_irrigator":
        # This link contains the irrigator's long tube/body.  It is a part of
        # the tool assembly for this variant, not an unrelated robot section.
        visible_links.add("psm_main_insertion_link")
    for link_name, entity in build.instances[0].link_entities.items():
        renderer = world.try_get_mesh_renderer(entity)
        if renderer is None:
            continue
        renderer.visible = link_name in visible_links
        world.set_mesh_renderer(entity, renderer)


def _author_closeup_fill_lights(world: neo.World) -> None:
    """Use a soft key/fill pair that keeps the metal readable without washing it out."""
    for direction, intensity in (((0.0, -0.35, 1.0), 2.25), ((0.0, -0.35, -1.0), 0.8)):
        light_entity = world.create_entity(0)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(*direction)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = intensity
        light.casts_shadows = False
        world.set_directional_light(light_entity, light)


def _render_tool(tool_type: str, width: int, height: int, output_dir: Path) -> Path:
    runtime_config = get_psm_default_runtime_config(1)
    runtime_config.scene_layout.max_cameras_per_env = 1
    runtime = neo.Runtime()
    if not runtime.initialize(runtime_config):
        raise RuntimeError("Failed to initialize the CRESSim-Neo runtime.")

    try:
        world = runtime.world()
        build = author_psm_scene(
            world,
            runtime.resources(),
            PsmAuthoringConfig(
                resolve_root=REPO_ROOT,
                tool_type=tool_type,
                env_count=1,
                add_ground=False,
                add_default_lighting=False,
                add_default_camera=False,
            ),
        )
        # Stationary pose, with the needle-driver jaws open enough to identify it.
        set_psm_joint_targets(world, build, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55])
        _show_only_tool_meshes(world, build, tool_type)
        _author_closeup_fill_lights(world)

        tool_transform = world.try_get_transform(build.instances[0].link_entities["psm_tool_yaw_link"])
        if tool_transform is None:
            raise RuntimeError("The PSM tool-yaw link has no transform.")
        yaw_pose = tool_transform.world_transform
        # Both PSM URDFs define psm_tool_tip_joint 10.2 mm along the tool-yaw
        # link's local Z axis.  The builder converts the asset basis by flipping
        # Z, hence the negative local-Z displacement below.  Aim at this exact
        # tooltip location, rather than at the yaw-link origin/wrist.
        tip_offset = _rotate_vector(yaw_pose.rotation, (0.0, 0.0, -0.0102))
        target = (
            yaw_pose.position.x + tip_offset[0],
            yaw_pose.position.y + tip_offset[1],
            yaw_pose.position.z + tip_offset[2],
        )
        # Straight-on front view from the illuminated, opposite side of the
        # previous render.  Keeping X and Y fixed avoids side perspective and
        # vertical tilt.
        camera_position = (target[0], target[1], target[2] - 0.09)

        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width, target_desc.height = width, height
        target_desc.color, target_desc.depth = True, True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.debug_name = f"PsmToolTypeScreenshot.{tool_type}"
        render_target = runtime.create_render_target(target_desc)
        if not runtime.is_valid_render_target(render_target):
            raise RuntimeError("Failed to create the screenshot render target.")

        camera_entity = world.create_entity(0)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(*camera_position)
        camera_transform.world_transform.rotation = _rolled_look_rotation(camera_position, target)
        world.set_transform(camera_entity, camera_transform)
        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 28.0
        camera.near_clip = 0.01
        camera.far_clip = 2.0
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(BACKGROUND_GRAY, BACKGROUND_GRAY, BACKGROUND_GRAY, 1.0)
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = render_target
        camera.output.binding.first_layer = 0
        camera.output.binding.layer_count = 1
        camera.output_width, camera.output_height = width, height
        world.set_camera(camera_entity, camera)

        frame = neo.FrameContext()
        frame.delta_seconds = 1.0 / 60.0
        readback = None
        for frame_index in range(3):
            runtime.prepare()
            if not runtime.upload_world():
                raise RuntimeError("Failed to upload the PSM screenshot scene.")
            readback = runtime.request_render_target_readback(camera.output.binding)
            if readback.id == 0:
                raise RuntimeError("Failed to queue the screenshot readback.")
            frame.frame_index = frame_index
            frame.time_seconds = frame_index * frame.delta_seconds
            runtime.step_physics(frame)
            runtime.step_visual_sensors(frame)
            runtime.end_frame(frame)

        event = runtime.try_get_render_target_readback(readback)
        if event is None or not event.color_bytes:
            raise RuntimeError("The screenshot render did not return a color image.")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"psm_{tool_type}.png"
        _write_rgba_png(output_path, event.color_width, event.color_height, bytes(event.color_bytes))
        return output_path
    finally:
        runtime.shutdown()


def main() -> int:
    args = parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive.")
    tool_types = TOOL_TYPES if args.tool_type == "both" else (args.tool_type,)
    for tool_type in tool_types:
        print(f"Rendering {tool_type} close-up...")
        print(f"Saved {_render_tool(tool_type, args.width, args.height, args.output_dir.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
