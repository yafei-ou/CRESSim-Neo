from __future__ import annotations

import torch

import cressim_neo as neo


def main() -> int:
    env = neo.SoftBodyTorchVectorEnv(env_count=8)
    try:
        obs = env.reset()
        print("reset observation shape:", tuple(obs.shape))
        for step_index in range(8):
            action = torch.empty(env.env_count, device=env.action_tensor.device, dtype=torch.float32)
            action.uniform_(-1.0, 1.0)
            obs, reward, terminated, truncated = env.step(action)
            print(f"step {step_index}")
            print("  action:", action[:4].detach().cpu())
            print("  observation:", obs[:4].detach().cpu())
            print("  reward:", reward[:4].detach().cpu())
            print("  terminated:", terminated[:4].detach().cpu())
            print("  truncated:", truncated[:4].detach().cpu())
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
