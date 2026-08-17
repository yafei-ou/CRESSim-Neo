#!/usr/bin/env python3
"""Record the remaining C++ demonstration scenes shown in the paper as MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_SCENES = {
    "rigid_body_scale": ("example_physics_large_array_multi_env", ()),
    "soft_bodies": ("example_physics_soft_particles_toroid_multi_env", ()),
    "joints": ("example_physics_rigid_joints", ("--suppress-connected-collisions",)),
    "fluid_slider": ("example_physics_fluid_moving_splitter", ()),
    "suturing": ("example_physics_soft_body_arc_needle_thread_kinematic", ("--mode", "needle-thread", "--wire-ground-friction", "0.5")),
    "cdcr": ("example_physics_rigid_spherical_cdcr", ()),
    "ultrasound_demo": ("example_physics_soft_particles_ultrasound_multi_env", ("--move-probe",)),
}
SCENES = tuple(CPP_SCENES)
BATCHED_SCENES = {"rigid_body_scale", "soft_bodies", "fluid_slider", "ultrasound_demo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, action="append", help="Repeat to select scenes; defaults to all remaining C++ paper scenes.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--simulation-fps", type=int, default=60, help="Fixed physics rate; keep this independent from --fps.")
    parser.add_argument("--envs", type=int, default=1, help="Environment count for batched scenes.")
    parser.add_argument("--switch-interval", type=int, default=0, help="Switch to the next environment camera every N video frames; 0 disables switching.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "paper_videos")
    parser.add_argument("--bin-dir", type=Path, default=REPO_ROOT / "build" / "linux-release" / "bin")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record_cpp(scene: str, args: argparse.Namespace) -> None:
    executable_name, scene_args = CPP_SCENES[scene]
    executable = args.bin_dir / executable_name
    if not executable.is_file():
        raise FileNotFoundError(f"Missing {executable}; build the C++ examples first.")
    output = args.output_dir / f"{scene}_{args.width}x{args.height}_{args.fps}fps.mp4"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "--frames", str(args.frames), "--window", "off", "--window-size", f"{args.width}x{args.height}", "--capture-video", str(output), "--capture-fps", str(args.fps), "--simulation-fps", str(args.simulation_fps), "--capture-switch-interval", str(args.switch_interval)]
    if scene in BATCHED_SCENES:
        command.extend(("--envs", str(args.envs)))
    command.extend(scene_args)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    if min(args.frames, args.fps, args.simulation_fps, args.envs, args.width, args.height) <= 0:
        raise ValueError("Frames, rates, and dimensions must be positive.")
    if args.fps > args.simulation_fps:
        raise ValueError("Video fps cannot exceed simulation fps.")
    if args.switch_interval < 0:
        raise ValueError("Switch interval cannot be negative.")
    for scene in args.scene or SCENES:
        print(f"Recording {scene}...")
        record_cpp(scene, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
