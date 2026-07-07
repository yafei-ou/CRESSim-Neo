from pathlib import Path

import cressim_neo as neo


def main() -> int:
    resolve_root = Path(__file__).resolve().parents[2]

    config = neo.get_psm_default_runtime_config(env_count=2)
    runtime = neo.Runtime()
    if not runtime.initialize(config):
        raise RuntimeError("Failed to initialize runtime for PSM builder smoke.")

    try:
        build = neo.author_psm_scene(
            runtime.world(),
            runtime.resources(),
            neo.PsmAuthoringConfig(resolve_root=resolve_root, env_count=2),
        )
        assert build.env_count == 2
        assert len(build.instances) == 2
        assert len(build.camera_entities) == 2
        assert len(build.light_entities) == 2
        for env_index, instance in enumerate(build.instances):
            assert instance.env_index == env_index
            assert len(instance.arm_joint_ids) == 6
            assert len(instance.arm_joint_limits) == 6
            assert len(instance.jaw_joint_ids) == 2

        neo.set_psm_joint_targets(
            runtime.world(),
            build,
            [
                [0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.15],
                [0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.10],
            ],
        )
        runtime.prepare()
        if not runtime.upload_world():
            raise RuntimeError("Failed to upload PSM builder smoke world.")

        suction_build = neo.author_psm_scene(
            runtime.world(),
            runtime.resources(),
            neo.PsmAuthoringConfig(
                resolve_root=resolve_root,
                tool_type="suction_irrigator",
                env_count=1,
                add_ground=False,
                add_default_lighting=False,
                add_default_camera=False,
            ),
        )
        assert suction_build.env_count == 1
        assert suction_build.urdf_path is not None
        assert suction_build.urdf_path.name == "psm_suction_irrigator.urdf"
        assert len(suction_build.instances[0].arm_joint_ids) == 6
        assert int(suction_build.instances[0].arm_joint_ids[3]) == -1
        assert tuple(float(v) for v in suction_build.instances[0].arm_joint_limits[3]) == (0.0, 0.0)
        assert tuple(int(v) for v in suction_build.instances[0].jaw_joint_ids) == (-1, -1)
        assert tuple(float(v) for v in suction_build.instances[0].jaw_limit) == (0.0, 0.0)

    finally:
        runtime.shutdown()

    env = neo.PsmEnv(resolve_root=resolve_root)
    try:
        env.set_joint_targets([0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.15])
        env.step()
    finally:
        env.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
