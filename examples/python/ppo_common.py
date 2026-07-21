from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from process_runtime_report import print_process_runtime_report

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise RuntimeError("This example requires PyTorch to be installed.") from exc


def _normalize_observation_shape(observation_dim: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(observation_dim, int):
        return (observation_dim,)
    if len(observation_dim) == 0:
        raise ValueError("observation_dim must not be empty.")
    return tuple(int(size) for size in observation_dim)


def _normalize_observation_spec(observation_dim: Any) -> Any:
    if isinstance(observation_dim, dict):
        return {str(key): _normalize_observation_shape(value) for key, value in observation_dim.items()}
    return _normalize_observation_shape(observation_dim)


def _serialize_observation_spec(observation_spec: Any) -> Any:
    if isinstance(observation_spec, dict):
        return {key: list(value) for key, value in observation_spec.items()}
    return list(observation_spec)


def _deserialize_observation_spec(serialized: Any) -> Any:
    if isinstance(serialized, dict):
        return {str(key): tuple(int(size) for size in value) for key, value in serialized.items()}
    return tuple(int(size) for size in serialized)


def _flatten_observation_shape(observation_shape: tuple[int, ...]) -> int:
    size = 1
    for dim in observation_shape:
        size *= dim
    return size


def _is_hybrid_observation(observation: Any) -> bool:
    return isinstance(observation, dict)


def _clone_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {key: value.clone() for key, value in observation.items()}
    return observation.clone()


def _assign_observation(dest: Any, source: Any, indices: "torch.Tensor") -> None:
    if isinstance(dest, dict):
        for key in dest:
            dest[key][indices] = source[key][indices]
        return
    dest[indices] = source[indices]


def _make_observation_buffer(
    rollout_steps: int,
    env_count: int,
    observation_spec: Any,
    *,
    device: "torch.device",
    dtype: "torch.dtype" = torch.float32,
) -> Any:
    if isinstance(observation_spec, dict):
        return {
            key: torch.empty((rollout_steps, env_count, *shape), device=device, dtype=dtype)
            for key, shape in observation_spec.items()
        }
    return torch.empty((rollout_steps, env_count, *observation_spec), device=device, dtype=dtype)


def _store_observation_step(buffer: Any, step_index: int, observation: Any) -> None:
    if isinstance(buffer, dict):
        for key in buffer:
            buffer[key][step_index].copy_(observation[key])
        return
    buffer[step_index].copy_(observation)


def _flatten_observation_buffer(buffer: Any, observation_spec: Any) -> Any:
    if isinstance(buffer, dict):
        return {
            key: value.reshape(-1, *observation_spec[key])
            for key, value in buffer.items()
        }
    return buffer.reshape(-1, *observation_spec)


def _index_observation_batch(observation: Any, batch_indices: "torch.Tensor") -> Any:
    if isinstance(observation, dict):
        return {key: value[batch_indices] for key, value in observation.items()}
    return observation[batch_indices]


def _get_observation_device(observation: Any) -> "torch.device | None":
    if isinstance(observation, dict):
        for value in observation.values():
            return value.device
        return None
    return observation.device


def _synchronize_device(device: "torch.device | None") -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _reset_done_environments(env: object, next_observation: Any, terminated: "torch.Tensor", truncated: "torch.Tensor") -> Any:
    done_mask = (terminated != 0) | (truncated != 0)
    done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
    if done_indices.numel() == 0:
        return next_observation

    reset_observation = env.reset(done_indices)
    next_observation = _clone_observation(next_observation)
    _assign_observation(next_observation, reset_observation, done_indices)
    return next_observation


class ContinuousActorCritic(nn.Module):
    MODEL_KIND = "mlp"

    def __init__(
        self,
        observation_dim: int | tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_shape = _normalize_observation_shape(observation_dim)
        input_dim = _flatten_observation_shape(self.observation_shape)
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_observation = observation.reshape(observation.shape[0], -1)
        features = self.backbone(flat_observation)
        mean = self.policy_mean(features)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return mean, std, value


class ImageContinuousActorCritic(nn.Module):
    MODEL_KIND = "cnn"

    def __init__(
        self,
        observation_dim: int | tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_shape = _normalize_observation_shape(observation_dim)
        if len(self.observation_shape) != 3:
            raise ValueError("ImageContinuousActorCritic expects observation_dim=(height, width, channels).")
        height, width, channels = self.observation_shape
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            encoded_dim = int(self.encoder(dummy).shape[1])
        self.backbone = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_observation = observation.permute(0, 3, 1, 2).contiguous()
        encoded = self.encoder(image_observation)
        features = self.backbone(encoded)
        mean = self.policy_mean(features)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return mean, std, value


class ChannelsFirstImageContinuousActorCritic(nn.Module):
    MODEL_KIND = "cnn_channels_first"

    def __init__(
        self,
        observation_dim: int | tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_shape = _normalize_observation_shape(observation_dim)
        if len(self.observation_shape) != 3:
            raise ValueError(
                "ChannelsFirstImageContinuousActorCritic expects observation_dim=(channels, height, width)."
            )
        channels, height, width = self.observation_shape
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            encoded_dim = int(self.encoder(dummy).shape[1])
        self.backbone = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation.contiguous())
        features = self.backbone(encoded)
        mean = self.policy_mean(features)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return mean, std, value


class HybridImageVectorActorCritic(nn.Module):
    MODEL_KIND = "hybrid_cnn_mlp"

    def __init__(
        self,
        observation_dim: dict[str, int | tuple[int, ...]],
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_shape = _normalize_observation_spec(observation_dim)
        if not isinstance(self.observation_shape, dict):
            raise ValueError("HybridImageVectorActorCritic expects a dict observation spec.")
        if "rgb" not in self.observation_shape or "vector" not in self.observation_shape:
            raise ValueError("HybridImageVectorActorCritic expects observation keys 'rgb' and 'vector'.")
        rgb_shape = self.observation_shape["rgb"]
        vector_shape = self.observation_shape["vector"]
        if len(rgb_shape) != 3:
            raise ValueError("HybridImageVectorActorCritic expects rgb observation_dim=(height, width, channels).")
        height, width, channels = rgb_shape
        vector_dim = _flatten_observation_shape(vector_shape)
        self.action_dim = action_dim
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            rgb_encoded_dim = int(self.rgb_encoder(dummy).shape[1])
        vector_hidden = max(hidden_dim // 2, 64)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, vector_hidden),
            nn.Tanh(),
            nn.Linear(vector_hidden, vector_hidden),
            nn.Tanh(),
        )
        fusion_input_dim = rgb_encoded_dim + vector_hidden
        self.backbone = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation: dict[str, "torch.Tensor"]) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        rgb_observation = observation["rgb"].permute(0, 3, 1, 2).contiguous()
        vector_observation = observation["vector"].reshape(observation["vector"].shape[0], -1)
        rgb_features = self.rgb_encoder(rgb_observation)
        vector_features = self.vector_encoder(vector_observation)
        features = self.backbone(torch.cat((rgb_features, vector_features), dim=-1))
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
    warmup_updates: int = 0
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


@dataclass
class StepBenchmarkResult:
    env_count: int
    measured_steps: int
    warmup_steps: int
    elapsed_seconds: float
    env_steps_per_second: float


@dataclass
class PPOBenchmarkResult:
    env_count: int
    measured_updates: int
    warmup_updates: int
    measured_env_steps: int
    elapsed_seconds: float
    env_steps_per_second: float


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    elementwise = -0.5 * (((action - mean).square() / variance) + 2.0 * std.log() + math.log(2.0 * math.pi))
    return elementwise.sum(dim=-1)


def save_model(
    model: nn.Module,
    model_path: Path,
    *,
    observation_dim: Any,
    action_dim: int,
    hidden_dim: int,
    model_kind: str = ContinuousActorCritic.MODEL_KIND,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation_shape": _serialize_observation_spec(_normalize_observation_spec(observation_dim)),
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "model_kind": model_kind,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"saved model to {model_path}")


def _build_model(
    model_kind: str,
    observation_dim: Any,
    action_dim: int,
    hidden_dim: int,
) -> nn.Module:
    if model_kind == ContinuousActorCritic.MODEL_KIND:
        return ContinuousActorCritic(observation_dim, action_dim, hidden_dim=hidden_dim)
    if model_kind == ImageContinuousActorCritic.MODEL_KIND:
        return ImageContinuousActorCritic(observation_dim, action_dim, hidden_dim=hidden_dim)
    if model_kind == ChannelsFirstImageContinuousActorCritic.MODEL_KIND:
        return ChannelsFirstImageContinuousActorCritic(
            observation_dim, action_dim, hidden_dim=hidden_dim
        )
    if model_kind == HybridImageVectorActorCritic.MODEL_KIND:
        return HybridImageVectorActorCritic(observation_dim, action_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported PPO model kind: {model_kind}")


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    observation_shape = checkpoint.get("observation_shape")
    if observation_shape is None:
        observation_shape = (int(checkpoint["observation_dim"]),)
    else:
        observation_shape = _deserialize_observation_spec(observation_shape)
    model_kind = str(checkpoint.get("model_kind", ContinuousActorCritic.MODEL_KIND))
    model = _build_model(
        model_kind,
        observation_shape,
        int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    ).to(device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def reshape_env_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action_dim == 1 and action.ndim > 1 and action.shape[-1] == 1:
        return action.squeeze(-1)
    return action


def create_inference_figure(
    initial_rgb: "torch.Tensor",
    initial_observation: Any = None,
) -> tuple[object, object, object | None]:
    from live_capture_utils import create_rgb_grid_figure, create_rgb_ultrasound_grid_figure

    env_count = min(initial_rgb.shape[0], 4)
    show_ultrasound = (
        initial_observation is not None
        and not isinstance(initial_observation, dict)
        and initial_observation.ndim == 4
        and initial_observation.shape[0] >= env_count
    )
    if show_ultrasound:
        return create_rgb_ultrasound_grid_figure(
            initial_rgb[:env_count], initial_observation[:env_count]
        )
    figure, rgb_artists = create_rgb_grid_figure(initial_rgb[:env_count])
    return figure, rgb_artists, None


def update_inference_figure(
    rgb_artists: object,
    rgb_tensor: "torch.Tensor",
    observation_tensor: Any = None,
    ultrasound_artists: object | None = None,
) -> None:
    from live_capture_utils import update_rgb_grid_figure, update_rgb_ultrasound_grid_figure

    if observation_tensor is not None and not isinstance(observation_tensor, dict) and ultrasound_artists is not None:
        update_rgb_ultrasound_grid_figure(
            rgb_artists, ultrasound_artists, rgb_tensor, observation_tensor
        )
    else:
        update_rgb_grid_figure(rgb_artists, rgb_tensor)


def benchmark_env_stepping(
    *,
    env_factory: Callable[[int, int], object],
    action_dim: int,
    env_count: int,
    max_episode_steps: int,
    warmup_steps: int,
    measured_steps: int,
    device_name: str = "cuda",
    log_runtime_environment: bool = False,
) -> StepBenchmarkResult:
    if measured_steps <= 0:
        raise ValueError("measured_steps must be positive.")

    env = env_factory(env_count, max_episode_steps)
    if log_runtime_environment:
        print_process_runtime_report(f"step benchmark runtime report ({env_count} envs)")

    try:
        observation = env.reset()
        device = _get_observation_device(observation) or torch.device(device_name)
        if action_dim == 1:
            action = torch.zeros(env_count, device=device, dtype=torch.float32)
        else:
            action = torch.zeros((env_count, action_dim), device=device, dtype=torch.float32)

        for _ in range(max(0, warmup_steps)):
            next_observation, _, terminated, truncated = env.step(reshape_env_action(action, action_dim))
            observation = _reset_done_environments(env, next_observation, terminated, truncated)

        _synchronize_device(device)
        start_time = time.perf_counter()
        for _ in range(measured_steps):
            next_observation, _, terminated, truncated = env.step(reshape_env_action(action, action_dim))
            observation = _reset_done_environments(env, next_observation, terminated, truncated)
        _synchronize_device(device)
        elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-8)
    finally:
        env.close()

    return StepBenchmarkResult(
        env_count=env_count,
        measured_steps=measured_steps,
        warmup_steps=max(0, warmup_steps),
        elapsed_seconds=elapsed_seconds,
        env_steps_per_second=(measured_steps * env_count) / elapsed_seconds,
    )


def benchmark_ppo_training_throughput(
    *,
    env_factory: Callable[[int, int], object],
    observation_dim: Any,
    action_dim: int,
    config: PPOTrainConfig,
    model_kind: str = ContinuousActorCritic.MODEL_KIND,
    log_runtime_environment: bool = False,
) -> PPOBenchmarkResult:
    device = torch.device(config.device)
    observation_shape = _normalize_observation_spec(observation_dim)
    env = env_factory(config.train_env_count, config.max_episode_steps)
    if log_runtime_environment:
        print_process_runtime_report(f"{config.name} benchmark runtime report")

    model = _build_model(
        model_kind,
        observation_shape,
        action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    observation = env.reset()
    total_env_steps = 0
    measured_updates = max(0, config.update_count - max(0, config.warmup_updates))
    elapsed_seconds = 0.0

    try:
        start_time = None
        for update_index in range(config.update_count):
            if update_index == config.warmup_updates:
                _synchronize_device(device)
                start_time = time.perf_counter()
                total_env_steps = 0

            obs_buffer = _make_observation_buffer(
                config.rollout_steps,
                env.env_count,
                observation_shape,
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
                _store_observation_step(obs_buffer, step_index, observation)
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

                observation = _reset_done_environments(env, next_observation, terminated, truncated)

            if update_index >= config.warmup_updates:
                total_env_steps += config.rollout_steps * env.env_count

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
            flat_observations = _flatten_observation_buffer(obs_buffer, observation_shape)
            flat_actions = actions_buffer.reshape(-1, action_dim)
            flat_log_probs = log_prob_buffer.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values_buffer.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8)

            sample_count = (
                next(iter(flat_observations.values())).shape[0]
                if isinstance(flat_observations, dict)
                else flat_observations.shape[0]
            )
            for _ in range(config.ppo_epochs):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, config.minibatch_size):
                    batch_indices = permutation[start : start + config.minibatch_size]
                    batch_obs = _index_observation_batch(flat_observations, batch_indices)
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

        _synchronize_device(device)
        if start_time is not None:
            elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-8)
    finally:
        env.close()

    return PPOBenchmarkResult(
        env_count=config.train_env_count,
        measured_updates=measured_updates,
        warmup_updates=max(0, config.warmup_updates),
        measured_env_steps=total_env_steps,
        elapsed_seconds=elapsed_seconds,
        env_steps_per_second=(total_env_steps / elapsed_seconds) if elapsed_seconds > 0.0 else 0.0,
    )


def train_ppo_continuous(
    *,
    env_factory: Callable[[int, int], object],
    observation_dim: Any,
    action_dim: int,
    config: PPOTrainConfig,
    model_kind: str = ContinuousActorCritic.MODEL_KIND,
    log_runtime_environment: bool = False,
) -> int:
    device = torch.device(config.device)
    observation_shape = _normalize_observation_spec(observation_dim)
    env = env_factory(config.train_env_count, config.max_episode_steps)
    if log_runtime_environment:
        print_process_runtime_report(f"{config.name} runtime report")
    model = _build_model(
        model_kind,
        observation_shape,
        action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    observation = env.reset()
    running_episode_returns = torch.zeros(env.env_count, device=device, dtype=torch.float32)
    running_episode_lengths = torch.zeros(env.env_count, device=device, dtype=torch.int32)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []
    training_start_time = None
    total_env_steps = 0

    try:
        for update_index in range(config.update_count):
            if update_index == config.warmup_updates:
                _synchronize_device(device)
                training_start_time = time.perf_counter()
                total_env_steps = 0
            update_start_time = time.perf_counter()
            obs_buffer = _make_observation_buffer(
                config.rollout_steps,
                env.env_count,
                observation_shape,
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
                _store_observation_step(obs_buffer, step_index, observation)
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
                    next_observation = _clone_observation(next_observation)
                    _assign_observation(next_observation, reset_observation, finished_indices)

                observation = next_observation

            if update_index >= config.warmup_updates:
                total_env_steps += config.rollout_steps * env.env_count
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

            flat_observations = _flatten_observation_buffer(obs_buffer, observation_shape)
            flat_actions = actions_buffer.reshape(-1, action_dim)
            flat_log_probs = log_prob_buffer.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values_buffer.reshape(-1)

            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8)

            sample_count = (
                next(iter(flat_observations.values())).shape[0]
                if isinstance(flat_observations, dict)
                else flat_observations.shape[0]
            )
            for _ in range(config.ppo_epochs):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, config.minibatch_size):
                    batch_indices = permutation[start : start + config.minibatch_size]
                    batch_obs = _index_observation_batch(flat_observations, batch_indices)
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
            update_elapsed = max(time.perf_counter() - update_start_time, 1.0e-8)
            if update_index < config.warmup_updates:
                print(
                    f"{config.name} warmup {update_index:03d}  "
                    f"mean_return={mean_return:.2f}  "
                    f"mean_length={mean_length:.1f}  "
                    f"finished_episodes={len(finished_returns)}"
                )
            else:
                average_elapsed = max(time.perf_counter() - (training_start_time or time.perf_counter()), 1.0e-8)
                update_fps = (config.rollout_steps * env.env_count) / update_elapsed
                average_fps = total_env_steps / average_elapsed
                print(
                    f"{config.name} update {update_index:03d}  "
                    f"mean_return={mean_return:.2f}  "
                    f"mean_length={mean_length:.1f}  "
                    f"finished_episodes={len(finished_returns)}  "
                    f"fps={update_fps:.1f}  "
                    f"avg_fps={average_fps:.1f}"
                )

        save_model(
            model,
            config.model_path,
            observation_dim=observation_shape,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            model_kind=model_kind,
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
    log_runtime_environment: bool = False,
) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Inference mode requires matplotlib to be installed.") from exc

    device = torch.device(device_name)
    model = load_model(model_path, device)
    effective_image_width = image_width
    effective_image_height = image_height
    observation_shape = getattr(model, "observation_shape", None)
    if isinstance(observation_shape, dict):
        rgb_shape = observation_shape.get("rgb")
        if rgb_shape is not None and len(rgb_shape) == 3:
            effective_image_height = int(rgb_shape[0])
            effective_image_width = int(rgb_shape[1])
    elif isinstance(observation_shape, tuple) and len(observation_shape) == 3:
        if isinstance(model, ChannelsFirstImageContinuousActorCritic):
            effective_image_height = int(observation_shape[1])
            effective_image_width = int(observation_shape[2])
        else:
            effective_image_height = int(observation_shape[0])
            effective_image_width = int(observation_shape[1])

    env = env_factory(
        infer_env_count,
        max_episode_steps,
        effective_image_width,
        effective_image_height,
    )
    if log_runtime_environment:
        print_process_runtime_report("inference runtime report")
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        observation = env.reset()
        rgb = env.render()
        figure, rgb_artists, ultrasound_artists = create_inference_figure(rgb, observation)

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
                observation = _clone_observation(observation)
                _assign_observation(observation, reset_observation, done_indices)

            rgb = env.render()
            update_inference_figure(rgb_artists, rgb, observation, ultrasound_artists)
            plt.pause(frame_interval)
    finally:
        plt.close("all")
        env.close()

    return 0
