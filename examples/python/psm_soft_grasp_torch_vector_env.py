from __future__ import annotations

import argparse
from pathlib import Path

import cressim_neo as neo
import torch
from live_capture_utils import (
    InteractiveImageCapture,
    create_rgb_grid_figure,
    rgb_tensor_to_numpy,
    update_rgb_grid_figure,
)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib to be installed.") from exc


class ScriptedPolicy:
    def __init__(self, env: "neo.PsmSoftGraspTorchVectorEnv") -> None:
        self.env = env
        self._insertion_action_magnitude = 0.6
        self._phase = torch.zeros(env.env_count, device=env.observation_tensor.device, dtype=torch.int64)
        self._last_abs_y_error = torch.full(
            (env.env_count,), float("inf"), device=env.observation_tensor.device
        )
        self._insertion_sign = torch.ones(
            env.env_count, device=env.observation_tensor.device, dtype=env.observation_tensor.dtype
        )
        self._descend_command_sign = torch.ones(
            env.env_count, device=env.observation_tensor.device, dtype=env.observation_tensor.dtype
        )
        self._hold_steps = torch.zeros(env.env_count, device=env.observation_tensor.device, dtype=torch.int64)
        self._probed = torch.zeros(env.env_count, device=env.observation_tensor.device, dtype=torch.bool)

    def __call__(self, observation: "torch.Tensor") -> "torch.Tensor":
        action = torch.zeros(
            (self.env.env_count, self.env.ACTION_DIM),
            device=self.env.action_tensor.device,
            dtype=self.env.action_tensor.dtype,
        )
        target_pos = observation[:, 14:17]
        tooltip_pos = observation[:, 21:24]
        attached = observation[:, 28] > 0.5
        y_error = target_pos[:, 1] - tooltip_pos[:, 1]
        abs_y_error = torch.abs(y_error)

        probe_mask = ~self._probed
        action[probe_mask, 2] = self._insertion_action_magnitude
        improved = abs_y_error <= self._last_abs_y_error
        self._insertion_sign = torch.where(improved, self._insertion_sign, -self._insertion_sign)
        self._probed = torch.ones_like(self._probed)

        descend_mask = self._phase == 0
        descend_action = torch.where(
            y_error[descend_mask] < 0.0,
            self._insertion_sign[descend_mask] * self._insertion_action_magnitude,
            -self._insertion_sign[descend_mask] * self._insertion_action_magnitude,
        )
        action[descend_mask, 2] = descend_action
        self._descend_command_sign[descend_mask] = torch.where(
            descend_action >= 0.0,
            torch.ones_like(descend_action),
            -torch.ones_like(descend_action),
        )
        close_enough = abs_y_error < 0.035
        self._phase = torch.where(descend_mask & close_enough, torch.ones_like(self._phase), self._phase)

        grasp_mask = self._phase == 1
        action[grasp_mask, 6] = 1.0
        self._hold_steps = torch.where(grasp_mask, self._hold_steps + 1, self._hold_steps)
        lift_ready = attached | (self._hold_steps > 12)
        self._phase = torch.where(grasp_mask & lift_ready, torch.full_like(self._phase, 2), self._phase)
        self._hold_steps = torch.where(grasp_mask & lift_ready, torch.zeros_like(self._hold_steps), self._hold_steps)

        lift_mask = self._phase == 2
        action[lift_mask, 2] = -self._descend_command_sign[lift_mask] * self._insertion_action_magnitude
        action[lift_mask, 6] = 1.0
        self._hold_steps = torch.where(lift_mask, self._hold_steps + 1, self._hold_steps)
        self._phase = torch.where(lift_mask & (self._hold_steps > 30), torch.full_like(self._phase, 3), self._phase)

        hold_mask = self._phase == 3
        action[hold_mask, 6] = 1.0

        self._last_abs_y_error = abs_y_error
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted RGB demo for the PSM soft-grasp vector env.")
    parser.add_argument("--env-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--image-width", type=int, default=2048)
    parser.add_argument("--image-height", type=int, default=2048)
    parser.add_argument(
        "--disable-target-marker",
        action="store_true",
        help="Hide the red target marker while keeping the task target active.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolve_root = Path(__file__).resolve().parents[2]
    env = neo.PsmSoftGraspTorchVectorEnv(
        env_count=args.env_count,
        enable_rgb_observation=True,
        enable_target_marker=not args.disable_target_marker,
        image_width=args.image_width,
        image_height=args.image_height,
        resolve_root=resolve_root,
    )
    try:
        observation = env.reset()
        policy = ScriptedPolicy(env)
        rgb = env.render()
        figure, image_artists = create_rgb_grid_figure(rgb)
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
                update_rgb_grid_figure(image_artists, rgb)
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
