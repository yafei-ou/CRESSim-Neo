from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


class ContinuousActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = self.policy_mean(features)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return mean, std, value


@dataclass
class PPOTrainConfig:
    name: str
    model_path: Path
    train_env_count: int
    rollout_steps: int
    update_count: int
    max_episode_steps: int
    hidden_dim: int = 128
    minibatch_size: int = 2048
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 0.5
    device: str = "cuda"


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    elementwise = -0.5 * (((action - mean).square() / variance) + 2.0 * std.log() + math.log(2.0 * math.pi))
    return elementwise.sum(dim=-1)


def save_model(
    model: ContinuousActorCritic,
    model_path: Path,
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"saved model to {model_path}")


def load_model(model_path: Path, device: torch.device) -> ContinuousActorCritic:
    checkpoint = torch.load(model_path, map_location=device)
    model = ContinuousActorCritic(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    ).to(device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def reshape_env_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action_dim == 1:
        return action.squeeze(-1)
    return action


def create_live_figure(initial_rgb: "torch.Tensor") -> tuple[object, object]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Inference mode requires matplotlib and numpy to be installed.") from exc

    rgb_images = np.clip(initial_rgb[..., :3].detach().cpu().numpy(), 0.0, 1.0)
    env_count = min(rgb_images.shape[0], 4)
    figure, axes = plt.subplots(1, env_count, figsize=(4 * env_count, 4), squeeze=False)
    image_artists: list[object] = []
    for env_index in range(env_count):
        image_artist = axes[0, env_index].imshow(rgb_images[env_index], animated=True)
        axes[0, env_index].set_title(f"Env {env_index}")
        axes[0, env_index].axis("off")
        image_artists.append(image_artist)
    figure.tight_layout()
    plt.show(block=False)
    return figure, np.asarray(image_artists, dtype=object)


def update_live_figure(image_artists: object, rgb_tensor: "torch.Tensor") -> None:
    import numpy as np

    rgb_images = np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])


def train_ppo_continuous(
    *,
    env_factory: Callable[[int, int], object],
    observation_dim: int,
    action_dim: int,
    config: PPOTrainConfig,
) -> int:
    device = torch.device(config.device)
    env = env_factory(config.train_env_count, config.max_episode_steps)
    model = ContinuousActorCritic(observation_dim, action_dim, hidden_dim=config.hidden_dim).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    observation = env.reset()
    running_episode_returns = torch.zeros(env.env_count, device=device, dtype=torch.float32)
    running_episode_lengths = torch.zeros(env.env_count, device=device, dtype=torch.int32)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []

    try:
        for update_index in range(config.update_count):
            obs_buffer = torch.empty(
                (config.rollout_steps, env.env_count, observation_dim),
                device=device,
                dtype=torch.float32,
            )
            actions_buffer = torch.empty(
                (config.rollout_steps, env.env_count, action_dim),
                device=device,
                dtype=torch.float32,
            )
            log_prob_buffer = torch.empty(
                (config.rollout_steps, env.env_count),
                device=device,
                dtype=torch.float32,
            )
            rewards_buffer = torch.empty(
                (config.rollout_steps, env.env_count),
                device=device,
                dtype=torch.float32,
            )
            done_buffer = torch.empty(
                (config.rollout_steps, env.env_count),
                device=device,
                dtype=torch.float32,
            )
            values_buffer = torch.empty(
                (config.rollout_steps, env.env_count),
                device=device,
                dtype=torch.float32,
            )

            for step_index in range(config.rollout_steps):
                obs_buffer[step_index].copy_(observation)
                with torch.no_grad():
                    mean, std, value = model(observation)
                    action = mean + std * torch.randn_like(mean)
                    log_prob = gaussian_log_prob(action, mean, std)

                next_observation, reward, terminated, truncated = env.step(
                    reshape_env_action(action, action_dim)
                )
                done_mask = ((terminated != 0) | (truncated != 0)).to(dtype=torch.float32)

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
            for step_index in range(config.rollout_steps - 1, -1, -1):
                not_done = 1.0 - done_buffer[step_index]
                next_values = next_value if step_index == config.rollout_steps - 1 else values_buffer[step_index + 1]
                delta = rewards_buffer[step_index] + config.gamma * next_values * not_done - values_buffer[step_index]
                last_advantage = delta + config.gamma * config.gae_lambda * not_done * last_advantage
                advantages[step_index] = last_advantage

            returns = advantages + values_buffer

            flat_observations = obs_buffer.reshape(-1, observation_dim)
            flat_actions = actions_buffer.reshape(-1, action_dim)
            flat_log_probs = log_prob_buffer.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values_buffer.reshape(-1)

            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8)

            sample_count = flat_observations.shape[0]
            for _ in range(config.ppo_epochs):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, config.minibatch_size):
                    batch_indices = permutation[start : start + config.minibatch_size]
                    batch_obs = flat_observations[batch_indices]
                    batch_actions = flat_actions[batch_indices]
                    batch_old_log_probs = flat_log_probs[batch_indices]
                    batch_advantages = flat_advantages[batch_indices]
                    batch_returns = flat_returns[batch_indices]
                    batch_old_values = flat_values[batch_indices]

                    mean, std, value = model(batch_obs)
                    log_prob = gaussian_log_prob(batch_actions, mean, std)
                    entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + std.log()).sum(dim=-1)

                    ratio = (log_prob - batch_old_log_probs).exp()
                    surrogate_1 = ratio * batch_advantages
                    surrogate_2 = torch.clamp(
                        ratio,
                        1.0 - config.clip_epsilon,
                        1.0 + config.clip_epsilon,
                    ) * batch_advantages
                    policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

                    value_clipped = batch_old_values + torch.clamp(
                        value - batch_old_values,
                        -config.clip_epsilon,
                        config.clip_epsilon,
                    )
                    value_loss_unclipped = (value - batch_returns).square()
                    value_loss_clipped = (value_clipped - batch_returns).square()
                    value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy.mean()
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

            recent_returns = finished_returns[-32:]
            recent_lengths = finished_lengths[-32:]
            mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
            mean_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
            print(
                f"{config.name} update {update_index:03d}  "
                f"mean_return={mean_return:.2f}  "
                f"mean_length={mean_length:.1f}  "
                f"finished_episodes={len(finished_returns)}"
            )

        save_model(
            model,
            config.model_path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
        )
    finally:
        env.close()

    return 0


def run_inference_continuous(
    *,
    env_factory: Callable[[int, int, int, int], object],
    action_dim: int,
    model_path: Path,
    infer_env_count: int,
    max_episode_steps: int,
    image_width: int,
    image_height: int,
    fps: float,
    device_name: str = "cuda",
) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Inference mode requires matplotlib to be installed.") from exc

    device = torch.device(device_name)
    model = load_model(model_path, device)
    env = env_factory(infer_env_count, max_episode_steps, image_width, image_height)
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        observation = env.reset()
        rgb = env.render()
        figure, image_artists = create_live_figure(rgb)

        while plt.fignum_exists(figure.number):
            with torch.no_grad():
                mean, _, _ = model(observation)
                action = mean.clamp_(-1.0, 1.0)

            observation, _, terminated, truncated = env.step(
                reshape_env_action(action, action_dim)
            )
            done_mask = (terminated != 0) | (truncated != 0)
            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                reset_observation = env.reset(done_indices)
                observation = observation.clone()
                observation[done_indices] = reset_observation[done_indices]

            rgb = env.render()
            update_live_figure(image_artists, rgb)
            figure.canvas.draw_idle()
            plt.pause(frame_interval)
    finally:
        plt.close("all")
        env.close()

    return 0
