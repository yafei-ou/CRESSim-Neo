from __future__ import annotations

import argparse
from pathlib import Path

import cressim_neo as neo
from cressim_neo_envs.cartpole import CartpoleTorchVectorEnv

from ppo_common import PPOTrainConfig, run_inference_continuous, train_ppo_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference for the CRESSim CartPole PPO example.")
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/cartpole_ppo_final.pt"),
        help="Path used to save the trained model or load it for inference.",
    )
    parser.add_argument("--train-env-count", type=int, default=256)
    parser.add_argument("--infer-env-count", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--log-runtime-libs", action="store_true")
    return parser.parse_args()

def make_train_env(env_count: int, max_episode_steps: int) -> CartpoleTorchVectorEnv:
    return CartpoleTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        reset_cart_position_range=0.05,
        reset_cart_velocity_range=0.05,
        reset_pole_angle_range_radians=0.08,
        reset_pole_angular_velocity_range=0.05,
    )


def make_infer_env(
    env_count: int, max_episode_steps: int, image_width: int, image_height: int
) -> CartpoleTorchVectorEnv:
    return CartpoleTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        reset_cart_position_range=0.05,
        reset_cart_velocity_range=0.05,
        reset_pole_angle_range_radians=0.08,
        reset_pole_angular_velocity_range=0.05,
        image_width=image_width,
        image_height=image_height,
    )


def run_training(args: argparse.Namespace) -> int:
    return train_ppo_continuous(
        env_factory=make_train_env,
        observation_dim=CartpoleTorchVectorEnv.OBSERVATION_DIM,
        action_dim=1,
        config=PPOTrainConfig(
            name="cartpole",
            model_path=args.model_path,
            train_env_count=args.train_env_count,
            rollout_steps=args.rollout_steps,
            update_count=args.update_count,
            max_episode_steps=args.max_episode_steps,
            hidden_dim=args.hidden_dim,
        ),
        log_runtime_environment=args.log_runtime_libs,
    )


def run_inference(args: argparse.Namespace) -> int:
    return run_inference_continuous(
        env_factory=make_infer_env,
        action_dim=1,
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
