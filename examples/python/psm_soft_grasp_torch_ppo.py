from __future__ import annotations

import argparse
from pathlib import Path

import cressim_neo as neo

from ppo_common import PPOTrainConfig, run_inference_continuous, train_ppo_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference for the CRESSim PSM soft-grasp PPO example."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/psm_soft_grasp_ppo_final.pt"),
        help="Path used to save the trained model or load it for inference.",
    )
    parser.add_argument("--train-env-count", type=int, default=64)
    parser.add_argument("--infer-env-count", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--disable-target-marker",
        action="store_true",
        help="Hide the red target marker during inference rendering.",
    )
    return parser.parse_args()


def _make_base_env(
    env_count: int,
    max_episode_steps: int,
    *,
    enable_rgb_observation: bool,
    image_width: int,
    image_height: int,
    enable_target_marker: bool,
) -> neo.PsmSoftGraspTorchVectorEnv:
    resolve_root = Path(__file__).resolve().parents[2]
    return neo.PsmSoftGraspTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        enable_rgb_observation=enable_rgb_observation,
        enable_target_marker=enable_target_marker,
        image_width=image_width,
        image_height=image_height,
        psm_scale=10.0,
        rotational_action_scale=0.012,
        insertion_action_scale=0.015,
        tissue_width=0.96,
        tissue_height=0.96,
        tissue_thickness=0.08,
        tooltip_proximity_threshold=0.08,
        lift_target_distance=0.18,
        resolve_root=resolve_root,
    )


def make_train_env(env_count: int, max_episode_steps: int) -> neo.PsmSoftGraspTorchVectorEnv:
    return _make_base_env(
        env_count,
        max_episode_steps,
        enable_rgb_observation=False,
        image_width=64,
        image_height=64,
        enable_target_marker=True,
    )


def make_infer_env(
    env_count: int,
    max_episode_steps: int,
    image_width: int,
    image_height: int,
    *,
    enable_target_marker: bool,
) -> neo.PsmSoftGraspTorchVectorEnv:
    return _make_base_env(
        env_count,
        max_episode_steps,
        enable_rgb_observation=True,
        image_width=image_width,
        image_height=image_height,
        enable_target_marker=enable_target_marker,
    )


def run_training(args: argparse.Namespace) -> int:
    return train_ppo_continuous(
        env_factory=make_train_env,
        observation_dim=neo.PsmSoftGraspTorchVectorEnv.OBSERVATION_DIM,
        action_dim=neo.PsmSoftGraspTorchVectorEnv.ACTION_DIM,
        config=PPOTrainConfig(
            name="psm_soft_grasp",
            model_path=args.model_path,
            train_env_count=args.train_env_count,
            rollout_steps=args.rollout_steps,
            update_count=args.update_count,
            max_episode_steps=args.max_episode_steps,
            hidden_dim=args.hidden_dim,
            minibatch_size=1024,
        ),
    )


def run_inference(args: argparse.Namespace) -> int:
    return run_inference_continuous(
        env_factory=lambda env_count, max_episode_steps, image_width, image_height: make_infer_env(
            env_count,
            max_episode_steps,
            image_width,
            image_height,
            enable_target_marker=not args.disable_target_marker,
        ),
        action_dim=neo.PsmSoftGraspTorchVectorEnv.ACTION_DIM,
        model_path=args.model_path,
        infer_env_count=args.infer_env_count,
        max_episode_steps=args.max_episode_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        fps=args.fps,
    )


def main() -> int:
    args = parse_args()
    if args.mode == "train":
        return run_training(args)
    return run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
