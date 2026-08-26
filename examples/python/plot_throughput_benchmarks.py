from __future__ import annotations

import argparse
from pathlib import Path


TASK_LABELS = {
    "cartpole": "CartPole",
    "soft_body_push": "SoftBodyPush",
    "fluid_pour": "FluidPour",
    "target_center": "TargetCenter",
    "tissue_retract": "TissueRetract",
    "blood_suction": "BloodSuction",
    "ultrasound_scan": "UltrasoundScan",
}

SIMPLE_TASKS = (
    "cartpole",
    "soft_body_push",
    "fluid_pour",
    "target_center",
)

SURGICAL_TASKS = (
    "tissue_retract",
    "blood_suction",
    "ultrasound_scan",
)

TASK_COLORS = {
    "cartpole": "#1f77b4",
    "soft_body_push": "#ff7f0e",
    "fluid_pour": "#2ca02c",
    "target_center": "#d62728",
    "tissue_retract": "#9467bd",
    "blood_suction": "#8c564b",
    "ultrasound_scan": "#e377c2",
}

TASK_MARKERS = {
    "cartpole": "o",
    "soft_body_push": "s",
    "fluid_pour": "^",
    "target_center": "D",
    "tissue_retract": "o",
    "blood_suction": "s",
    "ultrasound_scan": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CRESSim-Neo throughput benchmark logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/benchmark_log.txt"),
        help="Path to the benchmark log produced by throughput_benchmark.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/throughput_plot.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--mode",
        choices=("step", "ppo"),
        default="step",
        help="Which benchmark mode to plot.",
    )
    parser.add_argument(
        "--title",
        default="CRESSim-Neo Throughput Scaling",
        help="Figure title.",
    )
    return parser.parse_args()


def parse_benchmark_log(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Benchmark log not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Throughput benchmark summary"):
                continue
            if line.startswith("mode  ") or line.startswith("----"):
                continue

            parts = line.split()
            if len(parts) != 7:
                continue
            mode, task, env_count, warmup, measured, elapsed_s, env_steps_per_s = parts
            rows.append(
                {
                    "mode": mode,
                    "task": task,
                    "env_count": env_count,
                    "warmup": warmup,
                    "measured": measured,
                    "elapsed_s": elapsed_s,
                    "env_steps_per_s": env_steps_per_s,
                }
            )
    return rows


def filter_mode(rows: list[dict[str, str]], mode: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row["mode"] != mode:
            continue
        task = row["task"]
        grouped.setdefault(task, []).append((int(row["env_count"]), float(row["env_steps_per_s"])))

    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def plot_panel(axis: object, grouped: dict[str, list[tuple[int, float]]], tasks: tuple[str, ...], title: str) -> None:
    axis.set_title(title)
    axis.set_yscale("log")
    axis.set_xlabel("# Environments")
    axis.set_ylabel("Environment steps/s")
    axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    xticks: list[int] = []
    for task in tasks:
        if task not in grouped:
            continue
        xticks.extend(item[0] for item in grouped[task])

    unique_ticks = sorted(set(xticks))
    tick_positions = {value: index for index, value in enumerate(unique_ticks)}

    for task in tasks:
        if task not in grouped:
            continue
        xs = [item[0] for item in grouped[task]]
        ys = [item[1] for item in grouped[task]]
        x_positions = [tick_positions[value] for value in xs]
        axis.plot(
            x_positions,
            ys,
            marker=TASK_MARKERS.get(task, "o"),
            color=TASK_COLORS.get(task),
            linewidth=2.0,
            markersize=6.0,
            label=TASK_LABELS.get(task, task),
        )

    if unique_ticks:
        axis.set_xticks(list(range(len(unique_ticks))))
        axis.set_xticklabels([str(value) for value in unique_ticks])
    axis.legend(frameon=False)


def main() -> int:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script requires matplotlib.") from exc

    # Embed TrueType fonts in PDF output so it does not contain Type 3 glyphs.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    rows = parse_benchmark_log(args.input)
    grouped = filter_mode(rows, args.mode)
    if not grouped:
        raise RuntimeError(f"No rows found for mode '{args.mode}' in {args.input}.")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    plot_panel(axes[0], grouped, SIMPLE_TASKS, "Simple Tasks")
    plot_panel(axes[1], grouped, SURGICAL_TASKS, "Surgical Tasks")
    figure.suptitle(args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.output.suffix.lower() != ".pdf":
        pdf_output = args.output.with_suffix(".pdf")
        figure.savefig(pdf_output, bbox_inches="tight")
    print(f"saved plot to {args.output}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
