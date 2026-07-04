import math
from pathlib import Path

import cressim_neo as neo


def main() -> int:
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
    scene = neo.create_psm_scene(
        env_count=1,
        resolve_root=resolve_root,
        viewer_desc=viewer_desc,
    )

    try:
        base_targets = [
            0.05 * math.pi,
            -0.05 * math.pi,
            0.10,
            0.0 * math.pi,
            0.0 * math.pi,
            0.0 * math.pi,
        ]
        amplitudes = [
            0.0 * math.pi,
            0.0 * math.pi,
            0.0,
            0.0 * math.pi,
            0.0 * math.pi,
            0.0 * math.pi,
        ]
        frequencies = [0.55, 0.75, 0.42, 0.95, 1.15, 1.35]
        phase_offsets = [0.0, 0.7, 1.1, 1.8, 2.4, 3.1]

        callbacks = neo.DebugViewerCallbacks()

        def before_tick(frame: neo.FrameContext, runtime: neo.Runtime) -> None:
            time_seconds = float(frame.time_seconds)
            targets = [
                base + amplitude * math.sin(2.0 * math.pi * frequency * time_seconds + phase)
                for base, amplitude, frequency, phase in zip(
                    base_targets, amplitudes, frequencies, phase_offsets
                )
            ]
            scene.set_joint_targets(targets, env_index=0)

        callbacks.before_tick = before_tick
        scene.run_viewer(env_index=0, callbacks=callbacks)
        return 0
    finally:
        scene.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
