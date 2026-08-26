from __future__ import annotations

import argparse
from pathlib import Path

import cressim_neo as neo
from cressim_neo_envs.camera_centering_env import CameraCenteringTorchVectorEnv

from ppo_common import PPOTrainConfig, run_inference_continuous, train_ppo_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference for the CRESSim camera centering PPO example."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/camera_centering_ppo_final.pt"),
        help="Path used to save the trained model or load it for inference.",
    )
    parser.add_argument("--train-env-count", type=int, default=64)
    parser.add_argument("--infer-env-count", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-runtime-libs", action="store_true")
    return parser.parse_args()


def _make_base_env(
    env_count: int, max_episode_steps: int, image_width: int, image_height: int
) -> CameraCenteringTorchVectorEnv:
    return CameraCenteringTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        image_width=image_width,
        image_height=image_height,
        success_center_threshold=0.06,
        action_scale_yaw_degrees=0.2,
        action_scale_pitch_degrees=0.2,
    )


def make_train_env(
    env_count: int,
    max_episode_steps: int,
    image_width: int,
    image_height: int,
) -> CameraCenteringTorchVectorEnv:
    return _make_base_env(env_count, max_episode_steps, image_width, image_height)


def make_infer_env(
    env_count: int,
    max_episode_steps: int,
    image_width: int,
    image_height: int,
) -> CameraCenteringTorchVectorEnv:
    return _make_base_env(env_count, max_episode_steps, image_width, image_height)


def run_training(args: argparse.Namespace) -> int:
    observation_dim = (args.image_height, args.image_width, 4)
    return train_ppo_continuous(
        env_factory=lambda env_count, max_episode_steps: make_train_env(
            env_count,
            max_episode_steps,
            args.image_width,
            args.image_height,
        ),
        observation_dim=observation_dim,
        action_dim=CameraCenteringTorchVectorEnv.ACTION_DIM,
        model_kind="cnn",
        config=PPOTrainConfig(
            name="camera_centering",
            model_path=args.model_path,
            train_env_count=args.train_env_count,
            rollout_steps=args.rollout_steps,
            update_count=args.update_count,
            max_episode_steps=args.max_episode_steps,
            hidden_dim=args.hidden_dim,
            minibatch_size=256,
        ),
        log_runtime_environment=args.log_runtime_libs,
    )


def run_inference(args: argparse.Namespace) -> int:
    return run_inference_continuous(
        env_factory=lambda env_count, max_episode_steps, image_width, image_height: make_infer_env(
            env_count,
            max_episode_steps,
            image_width,
            image_height,
        ),
        action_dim=CameraCenteringTorchVectorEnv.ACTION_DIM,
        model_path=args.model_path,
        infer_env_count=args.infer_env_count,
        max_episode_steps=args.max_episode_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        fps=args.fps,
        log_runtime_environment=args.log_runtime_libs,
    )


def main() -> int:
    args = parse_args()
    if args.mode == "train":
        return run_training(args)
    return run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
