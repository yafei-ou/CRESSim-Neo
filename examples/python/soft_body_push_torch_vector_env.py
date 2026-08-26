from __future__ import annotations

import cressim_neo as neo
from cressim_neo_envs.soft_body_push_env import SoftBodyPushTorchVectorEnv
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


def scripted_action(
    env: "SoftBodyPushTorchVectorEnv", step_index: int
) -> "torch.Tensor":
    action = torch.zeros(
        (env.env_count, env.ACTION_DIM),
        device=env.action_tensor.device,
        dtype=env.action_tensor.dtype,
    )
    warmup_steps = 12
    push_steps = 64
    steer_steps = 32
    if step_index < warmup_steps:
        return action
    motion_step = step_index - warmup_steps
    if motion_step < push_steps:
        action[:, 0] = 1.0
        return action
    motion_step -= push_steps
    if motion_step < steer_steps:
        action[:, 1] = 0.35
        return action
    return action


def main() -> int:
    env = SoftBodyPushTorchVectorEnv(
        env_count=1,
        max_episode_steps=180,
        action_scale=0.01,
        enable_rgb_observation=True,
        image_width=512,
        image_height=512,
    )
    try:
        observation = env.reset()
        rgb = env.render()
        figure, image_artists = create_rgb_grid_figure(rgb)
        capture = InteractiveImageCapture(figure, __file__)
        capture.update("reset", [("rgb", rgb_tensor_to_numpy(rgb), None)])
        print(f"reset observation shape: {tuple(observation.shape)}")
        print(f"rgb observation shape: {tuple(rgb.shape)}")
        for step_index in range(120):
            if not plt.fignum_exists(figure.number):
                break
            action = scripted_action(env, step_index)
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
