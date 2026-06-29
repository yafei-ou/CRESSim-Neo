from __future__ import annotations

import cressim_neo as neo
import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc


def create_live_figure(
    rgb_tensor: "torch.Tensor",
) -> tuple["plt.Figure", np.ndarray]:
    rgb_images = np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)
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
    rgb_images = np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])


def scripted_action(
    env: "neo.CartpoleTorchVectorEnv", observation: "torch.Tensor"
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
    env = neo.CartpoleTorchVectorEnv(
        env_count=8,
        max_episode_steps=256,
        image_width=256,
        image_height=256,
        reset_pole_angle_range_radians=0.15,
    )
    try:
        observation = env.reset()
        rgb = env.render()
        figure, image_artists = create_live_figure(rgb)
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
                update_live_figure(image_artists, rgb)
                plt.pause(0.05)

        while plt.fignum_exists(figure.number):
            plt.pause(0.1)
    finally:
        plt.close("all")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
