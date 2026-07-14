from __future__ import annotations

import math
from pathlib import Path

import cressim_neo as neo
import torch
from live_capture_utils import InteractiveImageCapture, rgb_tensor_to_numpy

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc


def create_live_figure(
    rgb_tensor: "torch.Tensor",
    observation_tensor: "torch.Tensor",
) -> tuple["plt.Figure", np.ndarray, np.ndarray]:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    ultrasound_images = observation_tensor[:, -1].detach().cpu().numpy()
    env_count = rgb_images.shape[0]
    column_count = min(4, env_count)
    row_count = math.ceil(env_count / column_count)
    figure, axes = plt.subplots(
        row_count * 2,
        column_count,
        figsize=(4 * column_count, 6 * row_count),
        squeeze=False,
    )
    rgb_artists: list[np.ndarray] = []
    ultrasound_artists: list[np.ndarray] = []
    for env_index in range(env_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        rgb_axis = axes[row_index * 2, column_index]
        ultrasound_axis = axes[row_index * 2 + 1, column_index]
        rgb_artist = rgb_axis.imshow(
            rgb_images[env_index], animated=True
        )
        ultrasound_artist = ultrasound_axis.imshow(
            ultrasound_images[env_index], cmap="gray", vmin=0.0, vmax=1.0, animated=True
        )
        rgb_axis.set_title(f"Env {env_index} RGB")
        ultrasound_axis.set_title(f"Env {env_index} Ultrasound")
        rgb_axis.axis("off")
        ultrasound_axis.axis("off")
        rgb_artists.append(rgb_artist)
        ultrasound_artists.append(ultrasound_artist)
    for env_index in range(env_count, row_count * column_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        axes[row_index * 2, column_index].axis("off")
        axes[row_index * 2 + 1, column_index].axis("off")
    figure.tight_layout()
    plt.show(block=False)
    return (
        figure,
        np.asarray(rgb_artists, dtype=object),
        np.asarray(ultrasound_artists, dtype=object),
    )


def update_live_figure(
    rgb_artists: np.ndarray,
    ultrasound_artists: np.ndarray,
    rgb_tensor: "torch.Tensor",
    observation_tensor: "torch.Tensor",
) -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    ultrasound_images = observation_tensor[:, -1].detach().cpu().numpy()
    for env_index, rgb_artist in enumerate(rgb_artists.tolist()):
        rgb_artist.set_data(rgb_images[env_index])
        ultrasound_artists[env_index].set_data(ultrasound_images[env_index])


def scripted_action(
    env: neo.UltrasoundCenteringTorchVectorEnv,
    observation: "torch.Tensor",
    step_index: int,
) -> "torch.Tensor":
    action = torch.zeros(
        (env.env_count, env.ACTION_DIM),
        device=observation.device,
        dtype=observation.dtype,
    )
    phase = 0.08 * float(step_index)
    for env_index in range(env.env_count):
        env_phase = phase + 0.7 * float(env_index)
        action[env_index, 0] = 0.75 * math.sin(env_phase)
        action[env_index, 1] = 0.55 * math.cos(0.8 * env_phase)
    return action


def main() -> int:
    env = neo.UltrasoundCenteringTorchVectorEnv(
        env_count=4,
        max_episode_steps=160,
        frame_stack=4,
        image_height=160,
        probe_num_scanlines=96,
        probe_line_length=0.7,
        probe_scanline_spacing=0.006,
        enable_rgb_observation=True,
        render_width=1024,
        render_height=1024,
        resolve_root=REPO_ROOT,
        debug_logging=True,
    )
    try:
        observation = env.reset()
        rgb = env.render()
        figure, rgb_artists, ultrasound_artists = create_live_figure(rgb, observation)
        capture = InteractiveImageCapture(figure, __file__)
        capture.update(
            "reset",
            [
                ("rgb", rgb_tensor_to_numpy(rgb), None),
                ("ultrasound", observation[:, -1].detach().cpu().numpy(), "gray"),
            ],
        )
        print(f"reset observation shape: {tuple(observation.shape)}")
        print(f"render rgb shape: {tuple(rgb.shape)}")
        for step_index in range(160):
            if not plt.fignum_exists(figure.number):
                break
            action = scripted_action(env, observation, step_index)
            observation, reward, terminated, truncated = env.step(action)
            print(f"step {step_index}")
            print(f"  action[0]: {action[0].detach().cpu()}")
            print(f"  reward: {reward[:4].cpu()}")
            print(f"  terminated: {terminated[:4].cpu()}")
            print(f"  truncated: {truncated[:4].cpu()}")

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                observation = env.reset(done_indices)

            if plt.fignum_exists(figure.number):
                rgb = env.render()
                update_live_figure(rgb_artists, ultrasound_artists, rgb, observation)
                capture.update(
                    f"step_{step_index:04d}",
                    [
                        ("rgb", rgb_tensor_to_numpy(rgb), None),
                        ("ultrasound", observation[:, -1].detach().cpu().numpy(), "gray"),
                    ],
                )
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
