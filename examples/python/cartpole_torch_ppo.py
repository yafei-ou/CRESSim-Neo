import math

import cressim_neo as neo

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


class CartpoleActorCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = self.policy_mean(features).squeeze(-1)
        value = self.value_head(features).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    return -0.5 * (((action - mean).square() / variance) + 2.0 * std.log() + math.log(2.0 * math.pi))


def main() -> int:
    device = torch.device("cuda")
    env = neo.CartpoleTorchVectorEnv(
        env_count=256,
        max_episode_steps=500,
        reset_cart_position_range=0.05,
        reset_cart_velocity_range=0.05,
        reset_pole_angle_range_radians=0.08,
        reset_pole_angular_velocity_range=0.05,
    )

    rollout_steps = 128
    update_count = 500
    minibatch_size = 2048
    ppo_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    learning_rate = 3.0e-4
    max_grad_norm = 0.5

    model = CartpoleActorCritic(neo.CartpoleTorchVectorEnv.OBSERVATION_DIM).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    observation = env.reset()
    running_episode_returns = torch.zeros(env.env_count, device=device, dtype=torch.float32)
    running_episode_lengths = torch.zeros(env.env_count, device=device, dtype=torch.int32)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []

    try:
        for update_index in range(update_count):
            obs_buffer = torch.empty(
                (rollout_steps, env.env_count, neo.CartpoleTorchVectorEnv.OBSERVATION_DIM),
                device=device,
                dtype=torch.float32,
            )
            actions_buffer = torch.empty((rollout_steps, env.env_count), device=device, dtype=torch.float32)
            log_prob_buffer = torch.empty((rollout_steps, env.env_count), device=device, dtype=torch.float32)
            rewards_buffer = torch.empty((rollout_steps, env.env_count), device=device, dtype=torch.float32)
            done_buffer = torch.empty((rollout_steps, env.env_count), device=device, dtype=torch.float32)
            values_buffer = torch.empty((rollout_steps, env.env_count), device=device, dtype=torch.float32)

            for step_index in range(rollout_steps):
                obs_buffer[step_index].copy_(observation)
                with torch.no_grad():
                    mean, std, value = model(observation)
                    action = mean + std * torch.randn_like(mean)
                    log_prob = gaussian_log_prob(action, mean, std)

                next_observation, reward, terminated, truncated = env.step(action)
                done_mask = ((terminated != 0) | (truncated != 0)).to(dtype=torch.float32)

                if update_index == 0 and step_index == 0:
                    print("debug step0")
                    print("  action:", action[:8].detach().cpu())
                    print("  observation:", next_observation[:8].detach().cpu())
                    print("  reward:", reward[:8].detach().cpu())
                    print("  terminated:", terminated[:8].detach().cpu())
                    print("  truncated:", truncated[:8].detach().cpu())

                actions_buffer[step_index].copy_(action)
                log_prob_buffer[step_index].copy_(log_prob)
                rewards_buffer[step_index].copy_(reward)
                done_buffer[step_index].copy_(done_mask)
                values_buffer[step_index].copy_(value)

                running_episode_returns += reward
                running_episode_lengths += 1

                finished_mask = done_mask != 0.0
                finished_indices = torch.nonzero(finished_mask, as_tuple=False).flatten()
                if finished_indices.numel() > 0:
                    finished_returns.extend(running_episode_returns[finished_indices].detach().cpu().tolist())
                    finished_lengths.extend(running_episode_lengths[finished_indices].detach().cpu().tolist())
                    running_episode_returns[finished_indices] = 0.0
                    running_episode_lengths[finished_indices] = 0

                    reset_observation = env.reset(finished_indices)
                    next_observation = next_observation.clone()
                    next_observation[finished_indices] = reset_observation[finished_indices]

                observation = next_observation

            with torch.no_grad():
                _, _, next_value = model(observation)

            advantages = torch.empty_like(rewards_buffer)
            last_advantage = torch.zeros(env.env_count, device=device, dtype=torch.float32)
            for step_index in range(rollout_steps - 1, -1, -1):
                not_done = 1.0 - done_buffer[step_index]
                if step_index == rollout_steps - 1:
                    next_values = next_value
                else:
                    next_values = values_buffer[step_index + 1]
                delta = rewards_buffer[step_index] + gamma * next_values * not_done - values_buffer[step_index]
                last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
                advantages[step_index] = last_advantage

            returns = advantages + values_buffer

            flat_observations = obs_buffer.reshape(-1, neo.CartpoleTorchVectorEnv.OBSERVATION_DIM)
            flat_actions = actions_buffer.reshape(-1)
            flat_log_probs = log_prob_buffer.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values_buffer.reshape(-1)

            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8)

            sample_count = flat_observations.shape[0]
            for _ in range(ppo_epochs):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, minibatch_size):
                    batch_indices = permutation[start : start + minibatch_size]
                    batch_obs = flat_observations[batch_indices]
                    batch_actions = flat_actions[batch_indices]
                    batch_old_log_probs = flat_log_probs[batch_indices]
                    batch_advantages = flat_advantages[batch_indices]
                    batch_returns = flat_returns[batch_indices]
                    batch_old_values = flat_values[batch_indices]

                    mean, std, value = model(batch_obs)
                    log_prob = gaussian_log_prob(batch_actions, mean, std)
                    entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + std.log()

                    ratio = (log_prob - batch_old_log_probs).exp()
                    surrogate_1 = ratio * batch_advantages
                    surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                    policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

                    value_clipped = batch_old_values + torch.clamp(
                        value - batch_old_values, -clip_epsilon, clip_epsilon
                    )
                    value_loss_unclipped = (value - batch_returns).square()
                    value_loss_clipped = (value_clipped - batch_returns).square()
                    value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            recent_returns = finished_returns[-32:]
            recent_lengths = finished_lengths[-32:]
            mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
            mean_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
            print(
                f"update {update_index:03d}  "
                f"mean_return={mean_return:.2f}  "
                f"mean_length={mean_length:.1f}  "
                f"finished_episodes={len(finished_returns)}"
            )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
