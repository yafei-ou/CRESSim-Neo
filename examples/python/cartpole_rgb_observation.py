import cressim_neo as neo

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise RuntimeError("This example requires matplotlib and numpy to be installed.") from exc

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


def show_rgb(rgb_tensor: "torch.Tensor") -> None:
    rgb_images = np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)
    env_count = min(rgb_images.shape[0], 4)
    figure, axes = plt.subplots(1, env_count, figsize=(4 * env_count, 4), squeeze=False)
    for env_index in range(env_count):
        axes[0, env_index].imshow(rgb_images[env_index])
        axes[0, env_index].set_title(f"Env {env_index}")
        axes[0, env_index].axis("off")
    figure.tight_layout()
    plt.show()


def main() -> int:
    env = neo.CartpoleTorchVectorEnv(
        env_count=8,
        max_episode_steps=256,
        enable_rgb_observation=True,
        image_width=256,
        image_height=256,
        reset_pole_angle_range_radians=0.15,
    )
    try:
        observation = env.reset()
        rgb = env.render()
        print("reset observation shape:", tuple(observation.shape))
        print("rgb observation shape:", tuple(rgb.shape))

        for step_index in range(5):
            action = torch.empty(
                env.env_count, device=env.action_tensor.device, dtype=env.action_tensor.dtype
            ).uniform_(-1.0, 1.0)
            observation, reward, terminated, truncated = env.step(action)
            rgb = env.render()
            print(f"step {step_index}")
            print("  observation:", observation[:4].cpu())
            print("  reward:", reward[:4].cpu())
            print("  terminated:", terminated[:4].cpu())
            print("  truncated:", truncated[:4].cpu())

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                env.reset(done_indices)
                rgb = env.render()

            show_rgb(rgb)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
