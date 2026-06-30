from __future__ import annotations

import math

import cressim_neo as neo
import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc


def create_live_figure(
    observation_tensor: "torch.Tensor",
) -> tuple["plt.Figure", np.ndarray]:
    last_frames = observation_tensor[:, -1].detach().cpu().numpy()
    env_count = last_frames.shape[0]
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
            last_frames[env_index], cmap="gray", vmin=0.0, vmax=1.0, animated=True
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


def update_live_figure(image_artists: np.ndarray, observation_tensor: "torch.Tensor") -> None:
    last_frames = observation_tensor[:, -1].detach().cpu().numpy()
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(last_frames[env_index])


def main() -> int:
    env = neo.UltrasoundCenteringTorchVectorEnv(
        env_count=4,
        max_episode_steps=160,
        frame_stack=4,
        image_height=160,
        probe_num_scanlines=96,
        probe_line_length=0.7,
        probe_scanline_spacing=0.006,
        debug_logging=True,
    )
    try:
        observation = env.reset()
        figure, image_artists = create_live_figure(observation)
        print(f"reset observation shape: {tuple(observation.shape)}")
        for step_index in range(160):
            if not plt.fignum_exists(figure.number):
                break
            action = torch.zeros(
                (env.env_count, env.ACTION_DIM),
                device=observation.device,
                dtype=observation.dtype,
            )
            observation, reward, terminated, truncated = env.step(action)
            print(f"step {step_index}")
            print(f"  reward: {reward[:4].cpu()}")
            print(f"  terminated: {terminated[:4].cpu()}")
            print(f"  truncated: {truncated[:4].cpu()}")

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                observation = env.reset(done_indices)

            if plt.fignum_exists(figure.number):
                update_live_figure(image_artists, observation)
                plt.pause(0.05)

        while plt.fignum_exists(figure.number):
            plt.pause(0.1)
    finally:
        plt.close("all")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
