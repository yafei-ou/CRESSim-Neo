from __future__ import annotations

import cressim_neo as neo
from cressim_neo_envs.cartpole import CartpoleTorchVectorEnv
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
    env: "CartpoleTorchVectorEnv", observation: "torch.Tensor"
) -> "torch.Tensor":
    action = torch.zeros(
        env.env_count,
        device=observation.device,
        dtype=observation.dtype,
    )
    cart_position = observation[:, 0]
    cart_velocity = observation[:, 1]
    pole_angle = observation[:, 2]
    pole_angular_velocity = observation[:, 3]
    action.copy_(
        torch.clamp(
            0.75 * cart_position
            + 1.0 * cart_velocity
            + 6.0 * pole_angle
            + 1.25 * pole_angular_velocity,
            -1.0,
            1.0,
        )
    )
    return action


def main() -> int:
    env = CartpoleTorchVectorEnv(
        env_count=8,
        max_episode_steps=256,
        image_width=1024,
        image_height=1024,
        reset_pole_angle_range_radians=0.15,
    )
    try:
        observation = env.reset()
        rgb = env.render()
        figure, image_artists = create_rgb_grid_figure(rgb)
        capture = InteractiveImageCapture(figure, __file__)
        capture.update("reset", [("rgb", rgb_tensor_to_numpy(rgb), None)])
        print(f"reset observation shape: {tuple(observation.shape)}")
        print(f"rgb observation shape: {tuple(rgb.shape)}")

        for step_index in range(256):
            if not plt.fignum_exists(figure.number):
                break
            action = scripted_action(env, observation)
            observation, reward, terminated, truncated = env.step(action)
            rgb = env.render()
            print(f"step {step_index}")
            print(f"  action: {action[:4].cpu()}")
            print(f"  observation: {observation[:4].cpu()}")
            print(f"  reward: {reward[:4].cpu()}")
            print(f"  terminated: {terminated[:4].cpu()}")
            print(f"  truncated: {truncated[:4].cpu()}")

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                observation = env.reset(done_indices)
                rgb = env.render()

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
