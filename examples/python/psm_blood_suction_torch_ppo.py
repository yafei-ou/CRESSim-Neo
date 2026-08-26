from __future__ import annotations

import argparse
from pathlib import Path
import cressim_neo as neo
from cressim_neo_envs.psm_blood_suction_env import PsmBloodSuctionTorchVectorEnv

REPO_ROOT = Path(__file__).resolve().parents[2]

from ppo_common import PPOTrainConfig, HybridImageVectorActorCritic, run_inference_continuous, train_ppo_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference for the CRESSim PSM blood-suction PPO example."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/psm_blood_suction_ppo_final.pt"),
        help="Path used to save the trained model or load it for inference.",
    )
    parser.add_argument("--train-env-count", type=int, default=64)
    parser.add_argument("--infer-env-count", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--train-image-width", type=int, default=64)
    parser.add_argument("--train-image-height", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-runtime-libs", action="store_true")
    return parser.parse_args()


def _make_base_env(
    env_count: int,
    max_episode_steps: int,
    *,
    image_width: int,
    image_height: int,
    enable_visualization_camera: bool,
    visualization_image_width: int | None = None,
    visualization_image_height: int | None = None,
) -> PsmBloodSuctionTorchVectorEnv:
    return PsmBloodSuctionTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        enable_rgb_observation=True,
        enable_visualization_camera=enable_visualization_camera,
        return_combined_observation=True,
        image_width=image_width,
        image_height=image_height,
        visualization_image_width=visualization_image_width,
        visualization_image_height=visualization_image_height,
        insertion_action_scale=0.05,
        resolve_root=REPO_ROOT,
    )


def make_train_env(env_count: int, max_episode_steps: int) -> PsmBloodSuctionTorchVectorEnv:
    return _make_base_env(
        env_count,
        max_episode_steps,
        image_width=64,
        image_height=64,
        enable_visualization_camera=False,
    )


def make_infer_env(
    env_count: int,
    max_episode_steps: int,
    image_width: int,
    image_height: int,
    *,
    visualization_image_width: int,
    visualization_image_height: int,
) -> PsmBloodSuctionTorchVectorEnv:
    return _make_base_env(
        env_count,
        max_episode_steps,
        image_width=image_width,
        image_height=image_height,
        enable_visualization_camera=True,
        visualization_image_width=visualization_image_width,
        visualization_image_height=visualization_image_height,
    )


def run_training(args: argparse.Namespace) -> int:
    observation_dim = {
        "vector": PsmBloodSuctionTorchVectorEnv.OBSERVATION_DIM,
        "rgb": (args.train_image_height, args.train_image_width, 4),
    }

    def train_env_factory(env_count: int, max_episode_steps: int) -> PsmBloodSuctionTorchVectorEnv:
        return _make_base_env(
            env_count,
            max_episode_steps,
            image_width=args.train_image_width,
            image_height=args.train_image_height,
            enable_visualization_camera=False,
        )

    return train_ppo_continuous(
        env_factory=train_env_factory,
        observation_dim=observation_dim,
        action_dim=PsmBloodSuctionTorchVectorEnv.ACTION_DIM,
        config=PPOTrainConfig(
            name="psm_blood_suction",
            model_path=args.model_path,
            train_env_count=args.train_env_count,
            rollout_steps=args.rollout_steps,
            update_count=args.update_count,
            max_episode_steps=args.max_episode_steps,
            hidden_dim=args.hidden_dim,
            minibatch_size=1024,
        ),
        model_kind=HybridImageVectorActorCritic.MODEL_KIND,
        log_runtime_environment=args.log_runtime_libs,
    )


def run_inference(args: argparse.Namespace) -> int:
    return run_inference_continuous(
        env_factory=lambda env_count, max_episode_steps, image_width, image_height: make_infer_env(
            env_count,
            max_episode_steps,
            image_width,
            image_height,
            visualization_image_width=args.image_width,
            visualization_image_height=args.image_height,
        ),
        action_dim=PsmBloodSuctionTorchVectorEnv.ACTION_DIM,
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
