from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cressim_neo as neo

from camera_centering_torch_ppo import make_train_env as make_target_center_env
from cartpole_torch_ppo import make_train_env as make_cartpole_env
from fluid_pouring_torch_ppo import make_train_env as make_fluid_pour_env
from ppo_common import (
    PPOTrainConfig,
    benchmark_env_stepping,
    benchmark_ppo_training_throughput,
)
from psm_blood_suction_torch_ppo import _make_base_env as make_blood_suction_base_env
from psm_soft_grasp_torch_ppo import _make_base_env as make_tissue_retract_base_env
from soft_body_pusher_torch_ppo import make_train_env as make_soft_body_push_env
from ultrasound_centering_torch_ppo import _make_base_env as make_ultrasound_base_env

REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class TaskBenchmarkSpec:
    display_name: str
    env_factory: Callable[[int, int], object]
    observation_dim: Any | Callable[[], Any]
    action_dim: int
    model_kind: str
    max_episode_steps: int
    rollout_steps: int
    update_count: int
    hidden_dim: int
    minibatch_size: int


@dataclass(frozen=True)
class BenchmarkRow:
    mode: str
    task: str
    env_count: int
    warmup: int
    measured: int
    elapsed_seconds: float
    env_steps_per_second: float


def _resolve_ultrasound_observation_shape() -> tuple[int, int, int]:
    env = make_ultrasound_base_env(
        1,
        160,
        160,
        4,
        enable_rgb_observation=False,
        render_width=320,
        render_height=240,
    )
    try:
        observation = env.reset()
        _, frame_stack, image_height, image_width = observation.shape
        return int(frame_stack), int(image_height), int(image_width)
    finally:
        env.close()


TASK_SPECS: dict[str, TaskBenchmarkSpec] = {
    "cartpole": TaskBenchmarkSpec(
        display_name="CartPole",
        env_factory=make_cartpole_env,
        observation_dim=neo.CartpoleTorchVectorEnv.OBSERVATION_DIM,
        action_dim=1,
        model_kind="mlp",
        max_episode_steps=500,
        rollout_steps=128,
        update_count=500,
        hidden_dim=128,
        minibatch_size=2048,
    ),
    "soft_body_push": TaskBenchmarkSpec(
        display_name="SoftBodyPush",
        env_factory=make_soft_body_push_env,
        observation_dim=neo.SoftBodyPusherTorchVectorEnv.OBSERVATION_DIM,
        action_dim=neo.SoftBodyPusherTorchVectorEnv.ACTION_DIM,
        model_kind="mlp",
        max_episode_steps=180,
        rollout_steps=128,
        update_count=400,
        hidden_dim=128,
        minibatch_size=1024,
    ),
    "fluid_pour": TaskBenchmarkSpec(
        display_name="FluidPour",
        env_factory=make_fluid_pour_env,
        observation_dim=neo.FluidPouringTorchVectorEnv.OBSERVATION_DIM,
        action_dim=neo.FluidPouringTorchVectorEnv.ACTION_DIM,
        model_kind="mlp",
        max_episode_steps=240,
        rollout_steps=128,
        update_count=400,
        hidden_dim=128,
        minibatch_size=1024,
    ),
    "target_center": TaskBenchmarkSpec(
        display_name="TargetCenter",
        env_factory=lambda env_count, max_episode_steps: make_target_center_env(
            env_count,
            max_episode_steps,
            64,
            64,
        ),
        observation_dim=(64, 64, 4),
        action_dim=neo.CameraCenteringTorchVectorEnv.ACTION_DIM,
        model_kind="cnn",
        max_episode_steps=120,
        rollout_steps=128,
        update_count=100,
        hidden_dim=256,
        minibatch_size=256,
    ),
    "tissue_retract": TaskBenchmarkSpec(
        display_name="TissueRetract",
        env_factory=lambda env_count, max_episode_steps: make_tissue_retract_base_env(
            env_count,
            max_episode_steps,
            enable_rgb_observation=False,
            image_width=64,
            image_height=64,
            enable_target_marker=True,
        ),
        observation_dim=neo.PsmSoftGraspTorchVectorEnv.OBSERVATION_DIM,
        action_dim=neo.PsmSoftGraspTorchVectorEnv.ACTION_DIM,
        model_kind="mlp",
        max_episode_steps=180,
        rollout_steps=128,
        update_count=200,
        hidden_dim=256,
        minibatch_size=1024,
    ),
    "blood_suction": TaskBenchmarkSpec(
        display_name="BloodSuction",
        env_factory=lambda env_count, max_episode_steps: make_blood_suction_base_env(
            env_count,
            max_episode_steps,
            image_width=64,
            image_height=64,
            enable_visualization_camera=False,
        ),
        observation_dim={
            "vector": neo.PsmBloodSuctionTorchVectorEnv.OBSERVATION_DIM,
            "rgb": (64, 64, 4),
        },
        action_dim=neo.PsmBloodSuctionTorchVectorEnv.ACTION_DIM,
        model_kind="hybrid_cnn_mlp",
        max_episode_steps=240,
        rollout_steps=128,
        update_count=50,
        hidden_dim=256,
        minibatch_size=1024,
    ),
    "ultrasound_scan": TaskBenchmarkSpec(
        display_name="UltrasoundScan",
        env_factory=lambda env_count, max_episode_steps: make_ultrasound_base_env(
            env_count,
            max_episode_steps,
            160,
            4,
            enable_rgb_observation=False,
            render_width=320,
            render_height=240,
        ),
        observation_dim=_resolve_ultrasound_observation_shape,
        action_dim=neo.UltrasoundCenteringTorchVectorEnv.ACTION_DIM,
        model_kind="cnn_channels_first",
        max_episode_steps=160,
        rollout_steps=128,
        update_count=50,
        hidden_dim=256,
        minibatch_size=256,
    ),
}


def parse_env_counts(value: str) -> list[int]:
    env_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not env_counts:
        raise argparse.ArgumentTypeError("At least one environment count is required.")
    return env_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CRESSim-Neo environment and PPO throughput.")
    parser.add_argument(
        "--tasks",
        default="cartpole,soft_body_push,fluid_pour,target_center,tissue_retract,blood_suction,ultrasound_scan",
        help="Comma-separated task ids or 'all'.",
    )
    parser.add_argument("--mode", choices=("step", "ppo", "both"), default="both")
    parser.add_argument("--env-counts", type=parse_env_counts, default=[1, 8, 32, 64], help="Comma-separated env counts.")
    parser.add_argument("--step-warmup-steps", type=int, default=64)
    parser.add_argument("--step-measured-steps", type=int, default=256)
    parser.add_argument("--ppo-warmup-updates", type=int, default=5)
    parser.add_argument("--ppo-update-count", type=int, default=15)
    parser.add_argument("--ppo-rollout-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=0, help="Override per-task default when > 0.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-runtime-libs", action="store_true")
    return parser.parse_args()


def resolve_tasks(task_arg: str) -> list[str]:
    if task_arg.strip().lower() == "all":
        return list(TASK_SPECS.keys())
    tasks = [item.strip() for item in task_arg.split(",") if item.strip()]
    invalid = [task for task in tasks if task not in TASK_SPECS]
    if invalid:
        raise ValueError(f"Unknown task ids: {', '.join(invalid)}")
    return tasks


def resolve_observation_dim(value: Any | Callable[[], Any]) -> Any:
    return value() if callable(value) else value


def print_summary(rows: list[BenchmarkRow]) -> None:
    if not rows:
        print("No benchmark results recorded.")
        return

    headers = (
        "mode",
        "task",
        "env_count",
        "warmup",
        "measured",
        "elapsed_s",
        "env_steps_per_s",
    )
    formatted_rows = [
        (
            row.mode,
            row.task,
            str(row.env_count),
            str(row.warmup),
            str(row.measured),
            f"{row.elapsed_seconds:.6f}",
            f"{row.env_steps_per_second:.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(header), max(len(values[index]) for values in formatted_rows))
        for index, header in enumerate(headers)
    ]

    print()
    print("Throughput benchmark summary")
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print(
        "  ".join("-" * widths[index] for index in range(len(headers)))
    )
    for values in formatted_rows:
        print("  ".join(values[index].ljust(widths[index]) for index in range(len(headers))))


def main() -> int:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    rows: list[BenchmarkRow] = []
    for task_index, task_name in enumerate(tasks):
        spec = TASK_SPECS[task_name]
        observation_dim = None
        if args.mode in ("ppo", "both"):
            observation_dim = resolve_observation_dim(spec.observation_dim)

        for env_count in args.env_counts:
            max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else spec.max_episode_steps
            should_log_runtime = args.log_runtime_libs and task_index == 0 and env_count == args.env_counts[0]

            if args.mode in ("step", "both"):
                step_result = benchmark_env_stepping(
                    env_factory=spec.env_factory,
                    action_dim=spec.action_dim,
                    env_count=env_count,
                    max_episode_steps=max_episode_steps,
                    warmup_steps=args.step_warmup_steps,
                    measured_steps=args.step_measured_steps,
                    device_name=args.device,
                    log_runtime_environment=should_log_runtime,
                )
                rows.append(
                    BenchmarkRow(
                        mode="step",
                        task=task_name,
                        env_count=step_result.env_count,
                        warmup=step_result.warmup_steps,
                        measured=step_result.measured_steps,
                        elapsed_seconds=step_result.elapsed_seconds,
                        env_steps_per_second=step_result.env_steps_per_second,
                    )
                )

            if args.mode in ("ppo", "both"):
                ppo_result = benchmark_ppo_training_throughput(
                    env_factory=spec.env_factory,
                    observation_dim=observation_dim,
                    action_dim=spec.action_dim,
                    model_kind=spec.model_kind,
                    config=PPOTrainConfig(
                        name=f"{task_name}_benchmark",
                        model_path=Path("artifacts") / f"{task_name}_benchmark_unused.pt",
                        train_env_count=env_count,
                        rollout_steps=args.ppo_rollout_steps,
                        update_count=max(args.ppo_update_count, args.ppo_warmup_updates + 1),
                        warmup_updates=args.ppo_warmup_updates,
                        max_episode_steps=max_episode_steps,
                        hidden_dim=spec.hidden_dim,
                        minibatch_size=spec.minibatch_size,
                        device=args.device,
                    ),
                    log_runtime_environment=should_log_runtime and args.mode == "ppo",
                )
                rows.append(
                    BenchmarkRow(
                        mode="ppo",
                        task=task_name,
                        env_count=ppo_result.env_count,
                        warmup=ppo_result.warmup_updates,
                        measured=ppo_result.measured_updates,
                        elapsed_seconds=ppo_result.elapsed_seconds,
                        env_steps_per_second=ppo_result.env_steps_per_second,
                    )
                )

    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
