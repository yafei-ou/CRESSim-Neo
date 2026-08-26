from __future__ import annotations

from pathlib import Path
import os

import cressim_neo as neo
from .psm_builder import (
    PsmAuthoringConfig,
    PsmBuildResult,
    PsmRobotInstance,
    author_psm_scene,
    get_psm_default_runtime_config,
    set_psm_joint_targets,
)


class PsmEnv:
    def __init__(
        self,
        resolve_root: str | os.PathLike[str] | Path | None = None,
        urdf_path: str | os.PathLike[str] | Path | None = None,
        tool_type: str = "large_needle_driver",
        viewer_desc: neo.DebugViewerAppDesc | None = None,
        runtime_config: neo.RuntimeConfig | None = None,
        add_ground: bool = True,
        add_default_lighting: bool = True,
        add_default_camera: bool = True,
        global_scale: float = 1.0,
    ) -> None:
        config = runtime_config or get_psm_default_runtime_config(1)
        config.scene_layout.env_count = 1

        self._viewer = None
        if viewer_desc is not None:
            if not hasattr(neo, "DebugViewerApp"):
                raise RuntimeError("Debug viewer bindings are unavailable in this build.")
            viewer = neo.DebugViewerApp()
            if not viewer.initialize(viewer_desc, config):
                raise RuntimeError("Failed to initialize the debug viewer.")
            self._viewer = viewer

        self._runtime = neo.Runtime()
        if not self._runtime.initialize(config):
            if self._viewer is not None:
                self._viewer.shutdown()
                self._viewer = None
            raise RuntimeError("Failed to initialize the runtime for the PSM env.")

        self._frame = neo.FrameContext()
        self._viewer_session_started = False
        self._shutdown = False
        self._build_result = author_psm_scene(
            self._runtime.world(),
            self._runtime.resources(),
            PsmAuthoringConfig(
                resolve_root=resolve_root,
                urdf_path=urdf_path,
                tool_type=tool_type,
                env_count=1,
                add_ground=add_ground,
                add_default_lighting=add_default_lighting,
                add_default_camera=add_default_camera,
                global_scale=global_scale,
            ),
        )

    @property
    def runtime(self) -> neo.Runtime:
        return self._runtime

    @property
    def build_result(self) -> PsmBuildResult:
        return self._build_result

    @property
    def instance(self) -> PsmRobotInstance:
        return self._build_result.instances[0]

    @property
    def camera_entity(self) -> int:
        if not self._build_result.camera_entities:
            raise RuntimeError("This PSM env was created without a default camera.")
        return self._build_result.camera_entities[0]

    def set_joint_targets(self, targets) -> None:
        set_psm_joint_targets(self._runtime.world(), self._build_result, targets, env_index=0)

    def sync(self) -> None:
        self._runtime.prepare()
        if not self._runtime.upload_world():
            raise RuntimeError("Failed to upload authored PSM world state.")

    def step(self, delta_seconds: float = 1.0 / 60.0) -> None:
        self.sync()
        self._frame.frame_index += 1
        self._frame.delta_seconds = float(delta_seconds)
        self._frame.time_seconds += float(delta_seconds)
        if not self._runtime.step_physics(self._frame):
            raise RuntimeError("PSM env physics step failed.")
        if not self._runtime.step_simulation_sensors(self._frame):
            raise RuntimeError("PSM env simulation-sensor step failed.")
        self._runtime.step_visual_sensors(self._frame)
        self._runtime.end_frame(self._frame)

    def run_viewer(self, callbacks: neo.DebugViewerCallbacks | None = None) -> bool:
        if self._viewer is None:
            raise RuntimeError("This PSM env was not created with a debug viewer.")
        binding = neo.DebugViewerCameraBinding()
        binding.camera_entity = self.camera_entity
        self._viewer_session_started = True
        if callbacks is None:
            return self._viewer.run(self._runtime, binding)
        return self._viewer.run(self._runtime, binding, callbacks)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        viewer = self._viewer
        self._viewer = None
        if viewer is not None and not self._viewer_session_started:
            viewer.shutdown()

        self._runtime.shutdown()


__all__ = ["PsmEnv"]
