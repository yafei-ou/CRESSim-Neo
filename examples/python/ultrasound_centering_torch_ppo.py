from __future__ import annotations

import argparse
from pathlib import Path

import cressim_neo as neo
from cressim_neo_envs.ultrasound_centering_env import UltrasoundCenteringTorchVectorEnv

from ppo_common import PPOTrainConfig, run_inference_continuous, train_ppo_continuous


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference for the CRESSim ultrasound centering PPO example."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/ultrasound_centering_ppo_final.pt"),
        help="Path used to save the trained model or load it for inference.",
    )
    parser.add_argument("--train-env-count", type=int, default=8)
    parser.add_argument("--infer-env-count", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--log-runtime-libs", action="store_true")
    return parser.parse_args()


def _make_base_env(
    env_count: int,
    max_episode_steps: int,
    image_height: int,
    frame_stack: int,
    *,
    enable_rgb_observation: bool,
    render_width: int,
    render_height: int,
) -> UltrasoundCenteringTorchVectorEnv:
    return UltrasoundCenteringTorchVectorEnv(
        env_count=env_count,
        max_episode_steps=max_episode_steps,
        frame_stack=frame_stack,
        image_height=image_height,
        probe_num_scanlines=96,
        probe_line_length=0.7,
        probe_scanline_spacing=0.006,
        lateral_action_scale=0.002,
        depth_action_scale=0.002,
        reset_lateral_range=0.16,
        reset_depth_range=0.06,
        lateral_move_range=0.22,
        depth_move_range=0.22,
        success_center_threshold=0.05,
        success_slice_threshold=0.10,
        dark_sphere_radius=0.12,
        enable_rgb_observation=enable_rgb_observation,
        render_width=render_width,
        render_height=render_height,
        resolve_root=REPO_ROOT,
    )


def _resolve_observation_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    env = _make_base_env(
        1,
        args.max_episode_steps,
        args.image_height,
        args.frame_stack,
        enable_rgb_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    try:
        observation = env.reset()
        _, frame_stack, image_height, image_width = observation.shape
        return int(frame_stack), int(image_height), int(image_width)
    finally:
        env.close()


def run_training(args: argparse.Namespace) -> int:
    observation_dim = _resolve_observation_shape(args)
    return train_ppo_continuous(
        env_factory=lambda env_count, max_episode_steps: _make_base_env(
            env_count,
            max_episode_steps,
            args.image_height,
            args.frame_stack,
            enable_rgb_observation=False,
            render_width=args.render_width,
            render_height=args.render_height,
        ),
        observation_dim=observation_dim,
        action_dim=UltrasoundCenteringTorchVectorEnv.ACTION_DIM,
        model_kind="cnn_channels_first",
        config=PPOTrainConfig(
            name="ultrasound_centering",
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
        env_factory=lambda env_count, max_episode_steps, _image_width, _image_height: _make_base_env(
            env_count,
            max_episode_steps,
            args.image_height,
            args.frame_stack,
            enable_rgb_observation=True,
            render_width=args.render_width,
            render_height=args.render_height,
        ),
        action_dim=UltrasoundCenteringTorchVectorEnv.ACTION_DIM,
        model_path=args.model_path,
        infer_env_count=args.infer_env_count,
        max_episode_steps=args.max_episode_steps,
        image_width=args.render_width,
        image_height=args.render_height,
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
