#!/usr/bin/env python3
"""Record paper environments as selectable 1920x1080, 30fps MP4 videos.

The names map to the paper's CartPole, SoftBodyPush, FluidPour, TargetCenter,
TissueRetract, BloodSuction, and UltrasoundScan tasks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys

import cressim_neo as neo
from cressim_neo_envs.cartpole import CartPoleTorchVectorEnv
from cressim_neo_envs.target_center_env import TargetCenterTorchVectorEnv
from cressim_neo_envs.fluid_pour_env import FluidPourTorchVectorEnv
from cressim_neo_envs.soft_body_push_env import SoftBodyPushTorchVectorEnv
from cressim_neo_envs.ultrasound_scan_env import UltrasoundScanTorchVectorEnv
from cressim_neo_envs.blood_suction_env import BloodSuctionTorchVectorEnv
from cressim_neo_envs.tissue_retract_env import TissueRetractTorchVectorEnv
import numpy as np
import torch

from capture_utils import VideoWriter, prepare_output, rgb_tensor_to_frame


REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES = ("cartpole", "soft_body_push", "fluid_pour", "target_center", "tissue_retract", "blood_suction", "ultrasound_scan")


def simple_policy(env, observation, step, scene):
    if scene == "cartpole":
        return torch.clamp(.75 * observation[:, 0] + observation[:, 1] + 6 * observation[:, 2] + 1.25 * observation[:, 3], -1, 1)
    action = torch.zeros((1, env.ACTION_DIM), device=env.action_tensor.device, dtype=env.action_tensor.dtype)
    if scene == "soft_body_push":
        if 12 <= step < 76: action[:, 0] = 1
        elif 76 <= step < 108: action[:, 1] = .35
    elif scene == "fluid_pour":
        if 12 <= step < 62: action[:, 0] = 1
        elif 62 <= step < 212: action[:, 1] = -.5
    elif scene == "target_center":
        rgb = observation[..., :3]
        mask = (rgb[..., 0] > .35) & (rgb[..., 0] > rgb[..., 1] + .08) & (rgb[..., 0] > rgb[..., 2] + .08)
        weights, mass = mask.to(observation.dtype), mask.sum((1, 2))
        if mass[0] > 0:
            x = torch.linspace(-1, 1, env.image_width, device=observation.device).view(1, 1, -1)
            y = torch.linspace(-1, 1, env.image_height, device=observation.device).view(1, -1, 1)
            action[:, 0] = torch.clamp((weights * x).sum((1, 2)) / mass * 2.5, -1, 1)
            action[:, 1] = torch.clamp((weights * y).sum((1, 2)) / mass * 2.5, -1, 1)
        else: action[:, 0] = .35
    elif scene == "ultrasound_scan":
        action[:, 0], action[:, 1] = .75 * math.sin(.08 * step), .55 * math.cos(.064 * step)
    return action


class BloodSuctionPolicy:
    def __init__(self, env): self.env, self.step = env, 0
    def __call__(self, observation, _):
        action = torch.zeros((1, self.env.ACTION_DIM), device=self.env.action_tensor.device, dtype=self.env.action_tensor.dtype)
        inserting = observation[:, 2] < .185 * self.env.psm_scale
        action[inserting, 2] = 1
        phase = (self.step % 120) / 120
        desired = -.9 * phase if phase < .5 else -.9 * (1 - phase)
        action[:, 0] = torch.clamp((desired - observation[:, 0]) / max(self.env.rotational_action_scale, 1e-6), -1, 1)
        action[inserting, 0] = 0
        self.step += 1
        return action

    def reset(self, _env_indices):
        self.step = 0


class TissueRetractPolicy:
    """Repeatable approach, grasp, lift, and hold motion for the PSM scene."""

    def __init__(self, env, motion_scale):
        self.env = env
        self.motion_scale = motion_scale
        self.lift_step_count = math.ceil(30 / motion_scale)
        device, dtype = env.observation_tensor.device, env.observation_tensor.dtype
        self.phase = torch.zeros(env.env_count, device=device, dtype=torch.int64)
        self.last_abs_y_error = torch.full((env.env_count,), float("inf"), device=device, dtype=dtype)
        self.insertion_sign = torch.ones(env.env_count, device=device, dtype=dtype)
        self.descend_command_sign = torch.ones(env.env_count, device=device, dtype=dtype)
        self.hold_steps = torch.zeros(env.env_count, device=device, dtype=torch.int64)
        self.probed = torch.zeros(env.env_count, device=device, dtype=torch.bool)

    def __call__(self, observation, _):
        action = torch.zeros((self.env.env_count, self.env.ACTION_DIM), device=self.env.action_tensor.device, dtype=self.env.action_tensor.dtype)
        y_error = observation[:, 15] - observation[:, 22]
        abs_y_error = torch.abs(y_error)
        probe = ~self.probed
        action[probe, 2] = .6
        self.insertion_sign = torch.where(abs_y_error <= self.last_abs_y_error, self.insertion_sign, -self.insertion_sign)
        self.probed = torch.ones_like(self.probed)
        descend = self.phase == 0
        descend_action = torch.where(y_error[descend] < 0, self.insertion_sign[descend] * .6, -self.insertion_sign[descend] * .6)
        action[descend, 2] = descend_action
        self.descend_command_sign[descend] = torch.where(
            descend_action >= 0,
            torch.ones_like(descend_action),
            -torch.ones_like(descend_action),
        )
        self.phase = torch.where(descend & (abs_y_error < .035), torch.ones_like(self.phase), self.phase)
        grasp = self.phase == 1
        action[grasp, 6] = 1
        self.hold_steps = torch.where(grasp, self.hold_steps + 1, self.hold_steps)
        lift_ready = (observation[:, 28] > .5) | (self.hold_steps > 12)
        self.phase = torch.where(grasp & lift_ready, torch.full_like(self.phase, 2), self.phase)
        self.hold_steps = torch.where(grasp & lift_ready, torch.zeros_like(self.hold_steps), self.hold_steps)
        lift = self.phase == 2
        action[lift, 2] = -self.descend_command_sign[lift] * .6
        action[lift, 6] = 1
        self.hold_steps = torch.where(lift, self.hold_steps + 1, self.hold_steps)
        self.phase = torch.where(
            lift & (self.hold_steps > self.lift_step_count),
            torch.full_like(self.phase, 3),
            self.phase,
        )
        action[self.phase == 3, 6] = 1
        self.last_abs_y_error = abs_y_error
        return action

    def reset(self, env_indices):
        self.phase[env_indices] = 0
        self.last_abs_y_error[env_indices] = float("inf")
        self.insertion_sign[env_indices] = 1
        self.descend_command_sign[env_indices] = 1
        self.hold_steps[env_indices] = 0
        self.probed[env_indices] = False

def make_scene(scene, width, height, max_steps, motion_scale, ultrasound_height):
    if scene == "cartpole":
        env = CartPoleTorchVectorEnv(env_count=1, max_episode_steps=max_steps, image_width=width, image_height=height, reset_pole_angle_range_radians=.15)
        return env, lambda o, s: simple_policy(env, o, s, scene), env.render
    if scene == "soft_body_push":
        env = SoftBodyPushTorchVectorEnv(env_count=1, max_episode_steps=max_steps, action_scale=.01, enable_rgb_observation=True, image_width=width, image_height=height)
    elif scene == "fluid_pour":
        env = FluidPourTorchVectorEnv(env_count=1, max_episode_steps=max_steps, position_action_scale=.01, tilt_action_scale=.02, source_move_range_x=3.5, enable_rgb_observation=True, image_width=width, image_height=height)
    elif scene == "target_center":
        env = TargetCenterTorchVectorEnv(env_count=1, max_episode_steps=max_steps, image_width=width, image_height=height)
        return env, lambda o, s: simple_policy(env, o, s, scene), lambda: env.rgb_observation_tensor
    elif scene == "tissue_retract":
        env = TissueRetractTorchVectorEnv(env_count=1, max_episode_steps=max_steps, enable_rgb_observation=True, enable_target_marker=False, image_width=width, image_height=height, resolve_root=REPO_ROOT)
        return env, TissueRetractPolicy(env, motion_scale), env.render
    elif scene == "blood_suction":
        env = BloodSuctionTorchVectorEnv(env_count=1, max_episode_steps=max_steps, enable_rgb_observation=True, image_width=width, image_height=height, insertion_action_scale=.05, resolve_root=REPO_ROOT)
        return env, BloodSuctionPolicy(env), env.render
    else:
        env = UltrasoundScanTorchVectorEnv(env_count=1, max_episode_steps=max_steps, frame_stack=4, image_height=ultrasound_height, probe_num_scanlines=96, probe_line_length=.7, probe_scanline_spacing=.006, enable_rgb_observation=True, render_width=width, render_height=height, resolve_root=REPO_ROOT)
    return env, lambda o, s: simple_policy(env, o, s, scene), env.render


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, action="append", help="Repeat to select scenes; defaults to all.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "paper_videos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion-scale", type=float, default=1.0, help="Multiplier applied to every scripted action; values below 1.0 slow motion.")
    parser.add_argument("--ultrasound-height", type=int, default=640, help="Synthesized ultrasound image height for UltrasoundScan.")
    parser.add_argument("--stop-on-done", action="store_true", help="End a scene's video when its task terminates instead of resetting the episode.")
    parser.add_argument("--_scene-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_scene_worker(scene, args):
    """Run one scene in a fresh process so its Vulkan runtime is fully released."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scene", scene,
        "--frames", str(args.frames),
        "--fps", str(args.fps),
        "--width", str(args.width),
        "--height", str(args.height),
        "--warmup-steps", str(args.warmup_steps),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
        "--motion-scale", str(args.motion_scale),
        "--ultrasound-height", str(args.ultrasound_height),
        "--_scene-worker",
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.stop_on_done:
        command.append("--stop-on-done")
    subprocess.run(command, check=True)


def overlay_ultrasound(frame, observation):
    """Place the latest ultrasound image over the unobstructed upper-left video area."""
    ultrasound = observation[0, -1].detach().cpu().numpy()
    overlay_height = max(1, frame.shape[0] // 2)
    overlay_width = max(1, round(overlay_height * ultrasound.shape[1] / ultrasound.shape[0]))
    y_indices = np.linspace(0, ultrasound.shape[0] - 1, overlay_height).astype(np.intp)
    x_indices = np.linspace(0, ultrasound.shape[1] - 1, overlay_width).astype(np.intp)
    image = np.clip(ultrasound[np.ix_(y_indices, x_indices)], 0.0, 1.0)
    margin = max(16, frame.shape[1] // 100)
    frame[margin : margin + overlay_height, margin : margin + overlay_width] = image[..., None]
    return frame


def main():
    args = parse_args()
    if min(args.frames, args.fps, args.width, args.height, args.motion_scale, args.ultrasound_height) <= 0 or args.warmup_steps < 0: raise ValueError("Invalid capture dimensions or timing.")
    scenes = args.scene or SCENES
    if not args._scene_worker and len(scenes) > 1:
        for scene in scenes:
            run_scene_worker(scene, args)
        return 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    for scene in scenes:
        output = args.output_dir / f"{scene}_{args.width}x{args.height}_{args.fps}fps.mp4"
        prepare_output(output, overwrite=args.overwrite)
        env, policy, render = make_scene(scene, args.width, args.height, args.frames + args.warmup_steps + 1, args.motion_scale, args.ultrasound_height)
        try:
            observation = env.reset()
            for step in range(args.warmup_steps): observation, _, _, _ = env.step(policy(observation, step) * args.motion_scale)
            with VideoWriter(output, args.width, args.height, args.fps, overwrite=args.overwrite) as video:
                for step in range(args.frames):
                    frame = rgb_tensor_to_frame(render())
                    if scene == "ultrasound_scan":
                        frame = overlay_ultrasound(frame, observation)
                    video.write(frame)
                    observation, _, terminated, truncated = env.step(policy(observation, step + args.warmup_steps) * args.motion_scale)
                    done = torch.nonzero((terminated != 0) | (truncated != 0), as_tuple=False).flatten()
                    if done.numel():
                        if args.stop_on_done:
                            print(f"{scene}: task completed after {step + 1} frames; stopping capture")
                            break
                        observation = env.reset(done)
                        if hasattr(policy, "reset"):
                            policy.reset(done)
                    if (step + 1) % args.fps == 0 or step + 1 == args.frames: print(f"{scene}: {step + 1}/{args.frames} frames")
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
