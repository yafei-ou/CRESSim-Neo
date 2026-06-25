import cressim_neo as neo

try:
    import torch
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


def main() -> int:
    env = neo.CartpoleTorchVectorEnv(env_count=8, max_episode_steps=256)
    try:
        observation = env.reset()
        print("reset observation shape:", tuple(observation.shape))

        for step_index in range(8):
            action = torch.empty(
                env.env_count, device=env.action_tensor.device, dtype=env.action_tensor.dtype
            ).uniform_(-1.0, 1.0)
            observation, reward, terminated, truncated = env.step(action)
            print(f"step {step_index}")
            print("  action:", action[:4].cpu())
            print("  observation:", observation[:4].cpu())
            print("  reward:", reward[:4].cpu())
            print("  terminated:", terminated[:4].cpu())
            print("  truncated:", truncated[:4].cpu())

            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                env.reset(done_indices)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
