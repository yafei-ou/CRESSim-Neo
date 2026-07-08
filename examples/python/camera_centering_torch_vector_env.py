from __future__ import annotations

import math

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
    env_count = rgb_images.shape[0]
    column_count = min(4, env_count)
    row_count = math.ceil(env_count / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4 * column_count, 4 * row_count),
        squeeze=False,
    )
    image_artists: list[np.ndarray] = []
    for env_index in range(env_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        image_artist = axes[row_index, column_index].imshow(
            rgb_images[env_index], animated=True
        )
        axes[row_index, column_index].set_title(f"Env {env_index}")
        axes[row_index, column_index].axis("off")
        image_artists.append(image_artist)
    for env_index in range(env_count, row_count * column_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        axes[row_index, column_index].axis("off")
    figure.tight_layout()
    plt.show(block=False)
    return figure, np.asarray(image_artists, dtype=object)


def update_live_figure(image_artists: np.ndarray, rgb_tensor: "torch.Tensor") -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])


def scripted_action(
    env: "neo.CameraCenteringTorchVectorEnv", observation: "torch.Tensor"
) -> "torch.Tensor":
    action = torch.zeros(
        (env.env_count, env.ACTION_DIM),
        device=observation.device,
        dtype=observation.dtype,
    )

    rgb = observation[..., :3]
    red_mask = (rgb[..., 0] > 0.35) & (rgb[..., 0] > rgb[..., 1] + 0.08) & (
        rgb[..., 0] > rgb[..., 2] + 0.08
    )

    x_coords = torch.linspace(
        -1.0,
        1.0,
        env.image_width,
        device=observation.device,
        dtype=observation.dtype,
    ).view(1, 1, env.image_width)
    y_coords = torch.linspace(
        -1.0,
        1.0,
        env.image_height,
        device=observation.device,
        dtype=observation.dtype,
    ).view(1, env.image_height, 1)

    weights = red_mask.to(dtype=observation.dtype)
    visible_mass = weights.sum(dim=(1, 2))
    visible_mask = visible_mass > 0.0
    if torch.any(visible_mask):
        mass = visible_mass[visible_mask]
        centroid_x = (weights[visible_mask] * x_coords).sum(dim=(1, 2)) / mass
        centroid_y = (weights[visible_mask] * y_coords).sum(dim=(1, 2)) / mass

        yaw_command = torch.clamp(centroid_x * 2.5, -1.0, 1.0)
        pitch_command = torch.clamp(centroid_y * 2.5, -1.0, 1.0)
        action[visible_mask, 0] = yaw_command
        action[visible_mask, 1] = pitch_command

    if torch.any(~visible_mask):
        action[~visible_mask, 0] = 0.35
        action[~visible_mask, 1] = 0.0

    return action


def main() -> int:
    env = neo.CameraCenteringTorchVectorEnv(
        env_count=4,
        max_episode_steps=180,
        image_width=256,
        image_height=256,
    )
    try:
        observation = env.reset()
        figure, image_artists = create_live_figure(observation)
        capture = InteractiveImageCapture(figure, __file__)
        capture.update("reset", [("rgb", rgb_tensor_to_numpy(observation), None)])
        print(f"reset observation shape: {tuple(observation.shape)}")

        for step_index in range(180):
            if not plt.fignum_exists(figure.number):
                break
            action = scripted_action(env, observation)
            observation, reward, terminated, truncated = env.step(action)
            print(f"step {step_index}")
            print(f"  action: {action[:4].cpu()}")
            print(f"  reward: {reward[:4].cpu()}")
            print(f"  terminated: {terminated[:4].cpu()}")
            print(f"  truncated: {truncated[:4].cpu()}")

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                observation = env.reset(done_indices)

            if plt.fignum_exists(figure.number):
                update_live_figure(image_artists, observation)
                capture.update(f"step_{step_index:04d}", [("rgb", rgb_tensor_to_numpy(observation), None)])
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
