import argparse
import math
from pathlib import Path

import cressim_neo as neo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View and scrub a scaled CRESSim-Neo PSM.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform world-space scale applied during PSM authoring.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not hasattr(neo, "DebugViewerApp"):
        raise RuntimeError(
            "This build does not include the Python debug viewer bindings."
        )

    viewer_desc = neo.DebugViewerAppDesc()
    viewer_desc.window_title = "CRESSim-Neo PSM Viewer"
    viewer_desc.width = 1600
    viewer_desc.height = 900
    viewer_desc.step_simulation = True
    viewer_desc.show_stats = True

    resolve_root = Path(__file__).resolve().parents[2]
    env = neo.PsmEnv(
        resolve_root=resolve_root,
        viewer_desc=viewer_desc,
        global_scale=args.scale,
    )

    try:
        base_targets = [
            0.05 * math.pi,
            -0.05 * math.pi,
            0.10 * args.scale, # slider joint motion should be scaled with the robot
            0.0 * math.pi,
            0.0 * math.pi,
            0.0 * math.pi,
        ]
        amplitudes = [
            0.10 * math.pi,
            0.10 * math.pi,
            0.0 * args.scale, # slider joint motion should be scaled with the robot
            0.0 * math.pi,
            0.0 * math.pi,
            0.0 * math.pi,
        ]
        frequencies = [0.55, 0.75, 0.42, 0.95, 1.15, 1.35]
        phase_offsets = [0.0, 0.7, 1.1, 1.8, 2.4, 3.1]
        jaw_base = 0.35
        jaw_amplitude = 0.0
        jaw_frequency = 0.6
        jaw_phase = 0.4

        callbacks = neo.DebugViewerCallbacks()

        def before_tick(frame: neo.FrameContext, runtime: neo.Runtime) -> None:
            time_seconds = float(frame.time_seconds)
            arm_targets = [
                base + amplitude * math.sin(2.0 * math.pi * frequency * time_seconds + phase)
                for base, amplitude, frequency, phase in zip(
                    base_targets, amplitudes, frequencies, phase_offsets
                )
            ]
            jaw_target = jaw_base + jaw_amplitude * math.sin(
                2.0 * math.pi * jaw_frequency * time_seconds + jaw_phase
            )
            env.set_joint_targets([*arm_targets, jaw_target])

        callbacks.before_tick = before_tick
        env.run_viewer(callbacks=callbacks)
        return 0
    finally:
        env.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
