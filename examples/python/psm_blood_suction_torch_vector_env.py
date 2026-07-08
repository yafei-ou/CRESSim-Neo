from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_BIN = REPO_ROOT / "build" / "bin"
if str(BUILD_BIN) not in sys.path:
    sys.path.insert(0, str(BUILD_BIN))

import cressim_neo as neo
import torch
from live_capture_utils import InteractiveImageCapture, rgb_tensor_to_numpy

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc


def create_live_figure(
    rgb_tensor: "torch.Tensor",
) -> tuple["plt.Figure", np.ndarray]:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    env_count = min(rgb_images.shape[0], 4)
    figure, axes = plt.subplots(1, env_count, figsize=(4 * env_count, 4), squeeze=False)
    image_artists: list[np.ndarray] = []
    for env_index in range(env_count):
        image_artist = axes[0, env_index].imshow(rgb_images[env_index], animated=True)
        axes[0, env_index].set_title(f"Env {env_index}")
        axes[0, env_index].axis("off")
        image_artists.append(image_artist)
    figure.tight_layout()
    plt.show(block=False)
    return figure, np.asarray(image_artists, dtype=object)


def update_live_figure(image_artists: np.ndarray, rgb_tensor: "torch.Tensor") -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])


class ScriptedPolicy:
    def __init__(self, env: "neo.PsmBloodSuctionTorchVectorEnv") -> None:
        self.env = env
        self._step = torch.zeros(env.env_count, device=env.observation_tensor.device, dtype=torch.int64)
        self._viewer_base_insertion = 0.08 * env.psm_scale
        self._viewer_max_insertion = self._viewer_base_insertion + 0.115 * env.psm_scale
        self._inserting = torch.ones(
            env.env_count, device=env.observation_tensor.device, dtype=torch.bool
        )
        self._sweeping = torch.zeros(
            env.env_count, device=env.observation_tensor.device, dtype=torch.bool
        )

    def __call__(self, observation: "torch.Tensor") -> "torch.Tensor":
        action = torch.zeros(
            (self.env.env_count, self.env.ACTION_DIM),
            device=self.env.action_tensor.device,
            dtype=self.env.action_tensor.dtype,
        )
        insertion_position = observation[:, 2]
        sweep_joint_position = observation[:, 0]
        upper_margin = 0.01 * self.env.psm_scale
        self._inserting = insertion_position < (self._viewer_max_insertion - upper_margin)
        self._sweeping = ~self._inserting
        action[self._inserting, 2] = 1.0
        sweep_phase = (self._step % 120).to(dtype=action.dtype) / 120.0
        desired_sweep_angle = torch.where(
            sweep_phase < 0.5,
            -0.45 * (sweep_phase / 0.5),
            -0.45 * (1.0 - (sweep_phase - 0.5) / 0.5),
        )
        sweep_error = desired_sweep_angle - sweep_joint_position
        action[:, 0] = torch.clamp(sweep_error / max(self.env.rotational_action_scale, 1.0e-6), -1.0, 1.0)
        action[self._inserting, 0] = 0.0

        self._step += 1
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted RGB demo for the PSM blood-suction vector env.")
    parser.add_argument("--env-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = neo.PsmBloodSuctionTorchVectorEnv(
        env_count=args.env_count,
        enable_rgb_observation=True,
        image_width=args.image_width,
        image_height=args.image_height,
        insertion_action_scale=0.05,
        resolve_root=REPO_ROOT,
    )
    try:
        observation = env.reset()
        policy = ScriptedPolicy(env)
        for _ in range(max(0, args.warmup_steps)):
            observation, _, _, _ = env.step(policy(observation))
        rgb = env.render()
        figure, image_artists = create_live_figure(rgb)
        capture = InteractiveImageCapture(figure, __file__)
        capture.update("reset", [("rgb", rgb_tensor_to_numpy(rgb), None)])
        print(f"reset observation shape: {tuple(observation.shape)}")
        print(f"rgb observation shape: {tuple(rgb.shape)}")
        for step_index in range(args.steps):
            if not plt.fignum_exists(figure.number):
                break
            action = policy(observation)
            observation, reward, terminated, truncated = env.step(action)
            rgb = env.render()
            print(f"step {step_index}")
            print(f"  action: {action[:4]}")
            print(f"  observation: {observation[:4]}")
            print(f"  reward: {reward[:4]}")
            print(f"  terminated: {terminated[:4]}")
            print(f"  truncated: {truncated[:4]}")
            if plt.fignum_exists(figure.number):
                update_live_figure(image_artists, rgb)
                capture.update(f"step_{step_index:04d}", [("rgb", rgb_tensor_to_numpy(rgb), None)])
                plt.pause(0.05)
        while plt.fignum_exists(figure.number):
            plt.pause(0.1)
    finally:
        plt.close("all")
        if "capture" in locals():
            capture.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
