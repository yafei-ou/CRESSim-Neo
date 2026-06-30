from __future__ import annotations

import math
import struct

from . import _cressim_neo as neo
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "cressim_neo.ultrasound_centering_env requires PyTorch to be installed."
    ) from exc


_ULTRASOUND_CENTERING_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer UltrasoundCenteringPrePhysicsConstants
{
    float lateralActionScale;
    float depthActionScale;
    float lateralMoveRange;
    float depthMoveRange;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_ProbePoseSlots);
CRESSIM_STRUCTURED_BUFFER(float4, g_ProbeBasePositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ProbeBaseOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_ProbeState);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_Actions.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const float2 action = clamp(CRESSIM_SB_LOAD(g_Actions, envIndex), float2(-1.0f, -1.0f),
                                float2(1.0f, 1.0f));
    float2 state = CRESSIM_SB_LOAD(g_ProbeState, envIndex);
    state.x = clamp(state.x + action.x * lateralActionScale, -lateralMoveRange, lateralMoveRange);
    state.y = clamp(state.y + action.y * depthActionScale, -depthMoveRange, depthMoveRange);
    CRESSIM_SB_STORE(g_ProbeState, envIndex, state);

    const float4 basePosition = CRESSIM_SB_LOAD(g_ProbeBasePositions, envIndex);
    const float4 position = float4(basePosition.x + state.x, basePosition.y,
                                   basePosition.z + state.y, 0.0f);
    const uint poseSlot = CRESSIM_SB_LOAD(g_ProbePoseSlots, envIndex);
    CRESSIM_SB_STORE(g_EntityPositions, poseSlot, position);
    CRESSIM_SB_STORE(g_EntityOrientations, poseSlot,
                     CRESSIM_SB_LOAD(g_ProbeBaseOrientations, envIndex));
}
"""


_ULTRASOUND_CENTERING_RESET_PROBE_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer UltrasoundCenteringResetProbeConstants
{
    float4 rewardParams0;
    float4 rewardParams1;
    float4 rewardParams2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_ProbePoseSlots);
CRESSIM_STRUCTURED_BUFFER(uint, g_TargetParticleIndices);
CRESSIM_STRUCTURED_BUFFER(float2, g_ResetProbeState);
CRESSIM_STRUCTURED_BUFFER(float4, g_ProbeBasePositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ProbeBaseOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_ProbeState);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_PreviousErrors);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float ComputeCombinedError(uint envIndex, float3 probePosition, float4 probeOrientation)
{
    const uint targetParticleIndex = CRESSIM_SB_LOAD(g_TargetParticleIndices, envIndex);
    const float3 targetPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, targetParticleIndex).xyz;
    const float3 relative = targetPosition - probePosition;
    const float4 inverseOrientation =
        float4(-probeOrientation.x, -probeOrientation.y, -probeOrientation.z, probeOrientation.w);
    const float3 local = QuaternionRotate(inverseOrientation, relative);
    if (local.z <= 1.0e-4f || local.z > rewardParams2.y)
    {
        return rewardParams1.y;
    }

    const float centerError = abs(local.x) / max(rewardParams1.z, 1.0e-4f);
    const float sliceError = abs(local.y) / max(rewardParams1.w, 1.0e-4f);
    return centerError + sliceError;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_ResetMask.GetDimensions(envCount, stride);
    if (envIndex >= envCount || CRESSIM_SB_LOAD(g_ResetMask, envIndex) == 0u)
    {
        return;
    }

    const float2 resetState = CRESSIM_SB_LOAD(g_ResetProbeState, envIndex);
    CRESSIM_SB_STORE(g_ProbeState, envIndex, resetState);

    const float4 basePosition = CRESSIM_SB_LOAD(g_ProbeBasePositions, envIndex);
    const float4 probeOrientation = CRESSIM_SB_LOAD(g_ProbeBaseOrientations, envIndex);
    const float4 probePosition =
        float4(basePosition.x + resetState.x, basePosition.y, basePosition.z + resetState.y, 0.0f);
    const uint poseSlot = CRESSIM_SB_LOAD(g_ProbePoseSlots, envIndex);
    CRESSIM_SB_STORE(g_EntityPositions, poseSlot, probePosition);
    CRESSIM_SB_STORE(g_EntityOrientations, poseSlot, probeOrientation);

    const float combinedError = ComputeCombinedError(envIndex, probePosition.xyz, probeOrientation);
    CRESSIM_SB_STORE(g_PreviousErrors, envIndex, combinedError);
    CRESSIM_SB_STORE(g_Rewards, envIndex, 0.0f);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);
}
"""


_ULTRASOUND_CENTERING_RESET_PARTICLES_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(float4, g_ResetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_ResetMask.GetDimensions(envCount, stride);
    if (envIndex >= envCount || CRESSIM_SB_LOAD(g_ResetMask, envIndex) == 0u)
    {
        return;
    }

    const uint particleOffset = CRESSIM_SB_LOAD(g_EnvParticleOffsets, envIndex);
    const uint particleCount = CRESSIM_SB_LOAD(g_EnvParticleCounts, envIndex);
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_ResetPositions, particleIndex);
        positionInvMass.x = resetPosition.x;
        positionInvMass.y = resetPosition.y;
        positionInvMass.z = resetPosition.z;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex,
                         float4(positionInvMass.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
}
"""


_ULTRASOUND_CENTERING_REWARD_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer UltrasoundCenteringRewardConstants
{
    float4 rewardParams0;
    float4 rewardParams1;
    float4 rewardParams2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ProbePoseSlots);
CRESSIM_STRUCTURED_BUFFER(uint, g_TargetParticleIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_PreviousErrors);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_ProbePoseSlots.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint poseSlot = CRESSIM_SB_LOAD(g_ProbePoseSlots, envIndex);
    const uint targetParticleIndex = CRESSIM_SB_LOAD(g_TargetParticleIndices, envIndex);
    const float3 probePosition = CRESSIM_SB_LOAD(g_EntityPositions, poseSlot).xyz;
    const float4 probeOrientation = CRESSIM_SB_LOAD(g_EntityOrientations, poseSlot);
    const float3 targetPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, targetParticleIndex).xyz;
    const float3 relative = targetPosition - probePosition;
    const float4 inverseOrientation =
        float4(-probeOrientation.x, -probeOrientation.y, -probeOrientation.z, probeOrientation.w);
    const float3 local = QuaternionRotate(inverseOrientation, relative);

    float combinedError = rewardParams1.y;
    float centerError = rewardParams1.y;
    float sliceError = rewardParams1.y;
    if (local.z > 1.0e-4f && local.z <= rewardParams2.y)
    {
        centerError = abs(local.x) / max(rewardParams1.z, 1.0e-4f);
        sliceError = abs(local.y) / max(rewardParams1.w, 1.0e-4f);
        combinedError = centerError + sliceError;
    }

    const float previousError = CRESSIM_SB_LOAD(g_PreviousErrors, envIndex);
    const float progressReward = (previousError - combinedError) * rewardParams1.x;
    float reward = progressReward - rewardParams0.z;
    uint terminated = 0u;
    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint truncated = nextEpisodeStep >= uint(rewardParams2.x + 0.5f) ? 1u : 0u;
    if (centerError <= rewardParams0.x && sliceError <= rewardParams0.y)
    {
        terminated = 1u;
        reward += rewardParams0.w;
    }

    CRESSIM_SB_STORE(g_PreviousErrors, envIndex, combinedError);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_ULTRASOUND_CENTERING_OBSERVATION_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

Texture2DArray<float4> g_UltrasoundImage;
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observation);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint imageWidth = 0u;
    uint imageHeight = 0u;
    uint layerCount = 0u;
    g_UltrasoundImage.GetDimensions(imageWidth, imageHeight, layerCount);

    const uint x = dispatchThreadID.x;
    const uint y = dispatchThreadID.y;
    const uint envIndex = dispatchThreadID.z;
    if (x >= imageWidth || y >= imageHeight || envIndex >= layerCount)
    {
        return;
    }

    const float4 color = saturate(g_UltrasoundImage.Load(int4(int(x), int(y), int(envIndex), 0)));
    const float grayscale = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));
    const uint pixelIndex = envIndex * imageWidth * imageHeight + y * imageWidth + x;
    CRESSIM_SB_STORE(g_Observation, pixelIndex, grayscale);
}
"""


_ULTRASOUND_CENTERING_RGB_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

Texture2DArray<float4> g_ColorTarget;
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ColorObservation);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    uint layers = 0u;
    g_ColorTarget.GetDimensions(width, height, layers);

    const uint x = dispatchThreadID.x;
    const uint y = dispatchThreadID.y;
    const uint envIndex = dispatchThreadID.z;
    if (x >= width || y >= height || envIndex >= layers)
    {
        return;
    }

    const uint pixelIndex = envIndex * width * height + y * width + x;
    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex,
                     saturate(g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0))));
}
"""


def _probe_orientation() -> neo.Quaternion:
    angle = 0.5 * math.pi
    quat = neo.Quaternion()
    quat.x = math.sin(0.5 * angle)
    quat.y = 0.0
    quat.z = 0.0
    quat.w = math.cos(0.5 * angle)
    return quat


class UltrasoundCenteringTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 2

    def __init__(
        self,
        env_count: int = 32,
        max_episode_steps: int = 160,
        frame_stack: int = 4,
        image_height: int = 128,
        image_width: int | None = None,
        probe_num_scanlines: int = 96,
        probe_line_length: float = 0.7,
        probe_scanline_spacing: float = 0.006,
        lateral_action_scale: float = 0.01,
        depth_action_scale: float = 0.01,
        reset_lateral_range: float = 0.16,
        reset_depth_range: float = 0.06,
        lateral_move_range: float = 0.22,
        depth_move_range: float = 0.22,
        success_center_threshold: float = 0.05,
        success_slice_threshold: float = 0.10,
        dark_sphere_radius: float = 0.12,
        enable_rgb_observation: bool = False,
        render_width: int = 256,
        render_height: int = 256,
        debug_logging: bool = False,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.requested_image_width = image_width
        self.image_height_request = image_height
        self.probe_num_scanlines = max(1, int(probe_num_scanlines))
        self.probe_line_length = max(0.05, float(probe_line_length))
        self.probe_scanline_spacing = max(1.0e-4, float(probe_scanline_spacing))
        self.lateral_action_scale = lateral_action_scale
        self.depth_action_scale = depth_action_scale
        self.reset_lateral_range = min(reset_lateral_range, lateral_move_range)
        self.reset_depth_range = min(reset_depth_range, depth_move_range)
        self.lateral_move_range = lateral_move_range
        self.depth_move_range = depth_move_range
        self.success_center_threshold = success_center_threshold
        self.success_slice_threshold = success_slice_threshold
        self.dark_sphere_radius = dark_sphere_radius
        self.enable_rgb_observation = enable_rgb_observation
        self.render_width = max(1, int(render_width))
        self.render_height = max(1, int(render_height))
        self.debug_logging = debug_logging

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.physics_desc.substeps = 4
        config.physics_desc.default_iterations = 16
        config.physics_desc.soft_internal_iterations = 32
        config.physics_desc.soft_contact_iterations = 16
        config.scene_layout.env_count = env_count

        self.runtime = neo.Runtime()
        runtime_info = self.runtime.get_info()
        if not runtime_info.cuda_interop_supported:
            raise RuntimeError(
                "UltrasoundCenteringTorchVectorEnv requires CUDA interop support in this build."
            )
        if not runtime_info.ultrasound_supported:
            raise RuntimeError(
                "UltrasoundCenteringTorchVectorEnv requires ultrasound support in this build."
            )
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize ultrasound-centering runtime.")

        self._probe_entities: list[int] = []
        self._soft_entities: list[int] = []
        self._probe_pose_slots: list[int] = []
        self._probe_base_positions: list[tuple[float, float, float, float]] = []
        self._probe_base_orientations: list[tuple[float, float, float, float]] = []
        self._target_particle_local_indices: list[int] = []
        self._target_particle_indices: list[int] = []
        self._target_particle_positions: list[tuple[float, float, float, float]] = []
        self._reset_probe_state_centers: list[tuple[float, float]] = []
        self._ultrasound_render_target = neo.GpuRenderTargetHandle()
        self._ultrasound_probe_layout = neo.UltrasoundProbeLayout()
        self._rgb_render_target = neo.GpuRenderTargetHandle()

        if self.enable_rgb_observation:
            self._initialize_rgb_observation_resources()

        self._author_scene(self.runtime.world())
        self.runtime.prepare()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._reset_positions = self._build_reset_positions(self.runtime.world())
        self._build_target_metadata(self.runtime.world())
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared ultrasound-centering world.")

        self._initialize_probe_results()
        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self.observation_tensor = torch.zeros(
            (self.env_count, self.frame_stack, self.image_height, self.image_width),
            device=self.current_frame_tensor.device,
            dtype=self.current_frame_tensor.dtype,
        )
        self.reset()

    def _initialize_rgb_observation_resources(self) -> None:
        resources = self.runtime.resources()
        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.render_width
        target_desc.height = self.render_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "UltrasoundCentering.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create ultrasound-centering RGB render target.")

        self._rgb_soft_mesh = resources.register_mesh(
            neo.make_box_mesh(
                neo.Float3(0.225, 0.225, 0.225),
                "UltrasoundCentering.RenderSoftBody",
            )
        )
        self._rgb_ground_mesh = resources.register_mesh(
            neo.make_box_mesh(
                neo.Float3(2.0, 0.05, 2.0),
                "UltrasoundCentering.RenderGround",
            )
        )
        lateral_span = max(
            (self.probe_num_scanlines - 1) * self.probe_scanline_spacing,
            0.04,
        )
        self._rgb_probe_mesh = resources.register_mesh(
            neo.make_box_mesh(
                neo.Float3(0.5 * (lateral_span + 0.04), 0.06, 0.04),
                "UltrasoundCentering.RenderProbe",
            )
        )

    def _make_material(
        self, debug_name: str, base_color: neo.Float3, roughness: float
    ) -> neo.MaterialHandle:
        material_desc = neo.MaterialResourceDesc()
        material_desc.debug_name = debug_name
        material_desc.base_color = base_color
        material_desc.metallic = 0.0
        material_desc.roughness = roughness
        return self.runtime.resources().register_material(material_desc)

    def _author_scene(self, world: neo.World) -> None:
        probe_orientation = _probe_orientation()
        probe_height = 0.4
        soft_size = neo.Float3(0.45, 0.45, 0.45)
        ground_center_y = -0.15
        ground_half_height = 0.05
        ground_top_y = ground_center_y + ground_half_height
        cube_half_height = 0.5 * soft_size.y
        soft_spawn_y = ground_top_y + cube_half_height + 0.02

        probe = neo.UltrasoundProbeComponent()
        probe.geometry = neo.UltrasoundProbeGeometry.Linear
        probe.num_scanlines = self.probe_num_scanlines
        probe.line_length = self.probe_line_length
        probe.scanline_spacing = self.probe_scanline_spacing

        renderer_template = neo.UltrasoundRendererComponent()
        renderer_template.enabled = True
        renderer_template.output_width = 0 if self.requested_image_width is None else self.requested_image_width
        renderer_template.output_height = self.image_height_request

        probe_layout = self.runtime.compute_ultrasound_probe_layout(probe, renderer_template)
        if probe_layout.image_width <= 0 or probe_layout.image_height <= 0:
            raise RuntimeError("Failed to compute ultrasound probe layout.")
        self._ultrasound_probe_layout = probe_layout

        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = probe_layout.image_width
        target_desc.height = probe_layout.image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = False
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.unordered_access = True
        target_desc.color_format = probe_layout.color_format
        self._ultrasound_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._ultrasound_render_target):
            raise RuntimeError("Failed to create layered ultrasound render target.")

        ground_material = None
        soft_material = None
        probe_material = None
        if self.enable_rgb_observation:
            ground_material = self._make_material(
                "UltrasoundCentering.GroundMaterial",
                neo.Float3(0.70, 0.73, 0.78),
                0.92,
            )
            soft_material = self._make_material(
                "UltrasoundCentering.SoftMaterial",
                neo.Float3(0.82, 0.54, 0.47),
                0.68,
            )
            probe_material = self._make_material(
                "UltrasoundCentering.ProbeMaterial",
                neo.Float3(0.22, 0.27, 0.33),
                0.34,
            )

        for env_index in range(self.env_count):
            if self.enable_rgb_observation:
                light_entity = world.create_entity(env_index)
                light = neo.DirectionalLightComponent()
                light.direction = neo.Float3(-0.30, -1.0, 0.18)
                light.color = neo.Float3(1.0, 1.0, 1.0)
                light.intensity = 7.5
                light.casts_shadows = True
                world.set_directional_light(light_entity, light)

            ground_entity = world.create_entity(env_index)
            ground_transform = neo.TransformComponent()
            ground_transform.world_transform.position = neo.Float3(0.0, ground_center_y, 0.0)
            world.set_transform(ground_entity, ground_transform)
            ground_body = neo.RigidBodyComponent()
            ground_body.body_type = neo.RigidBodyType.Static
            ground_body.inverse_mass = 0.0
            ground_body.simulated = True
            world.set_rigid_body(ground_entity, ground_body)
            ground_collider = neo.ColliderComponent()
            ground_collider.shape_type = neo.ColliderShapeType.Box
            ground_collider.shape_params = neo.Float4(2.0, 0.05, 2.0, 0.0)
            ground_collider.friction = 2.5
            world.add_collider(ground_entity, ground_collider)
            if self.enable_rgb_observation:
                ground_renderer = neo.MeshRendererComponent()
                ground_renderer.mesh = self._rgb_ground_mesh
                ground_renderer.material = ground_material
                ground_renderer.segmentation_id = 100 + env_index
                ground_renderer.visible = True
                world.set_mesh_renderer(ground_entity, ground_renderer)

            soft_entity = world.create_entity(env_index)
            soft_transform = neo.TransformComponent()
            soft_transform.world_transform.position = neo.Float3(0.0, soft_spawn_y, 0.0)
            world.set_transform(soft_entity, soft_transform)
            if self.enable_rgb_observation:
                soft_renderer = neo.MeshRendererComponent()
                soft_renderer.mesh = self._rgb_soft_mesh
                soft_renderer.material = soft_material
                soft_renderer.segmentation_id = 200 + env_index
                soft_renderer.visible = True
                world.set_mesh_renderer(soft_entity, soft_renderer)
            soft = neo.SoftBodyComponent()
            soft.source.kind = neo.SoftBodySourceKind.RegularGrid
            soft.source.regular_grid.size = soft_size
            soft.source.regular_grid.target_particle_spacing = 0.04
            soft.particle_mass = 0.01
            soft.particle_radius = 0.02
            soft.edge_compliance = 0.0
            soft.volume_compliance = 8.0e-4
            soft.simulated = True
            soft.self_collision_enabled = False
            soft.collision_layer = 0x1
            soft.collision_mask = 0xFFFFFFFF
            if not world.set_soft_body(soft_entity, soft):
                raise RuntimeError(f"Failed to author ultrasound soft body for env {env_index}.")
            scatterer = neo.UltrasoundScattererSourceComponent()
            scatterer.density = 1_000_000.0
            scatterer.point_distance_override = 0.0
            world.set_ultrasound_scatterer_source(soft_entity, scatterer)
            self._soft_entities.append(soft_entity)

            authoring_particles = world.try_get_soft_body_authoring_particles(soft_entity)
            if authoring_particles is None or authoring_particles.particle_count == 0:
                raise RuntimeError(
                    f"Authoring particles were unavailable for ultrasound soft body {soft_entity}."
                )
            rest_positions = authoring_particles.rest_positions
            min_x = max_x = rest_positions[0].x
            min_y = max_y = rest_positions[0].y
            min_z = max_z = rest_positions[0].z
            for position in rest_positions[1:]:
                min_x = min(min_x, position.x)
                max_x = max(max_x, position.x)
                min_y = min(min_y, position.y)
                max_y = max(max_y, position.y)
                min_z = min(min_z, position.z)
                max_z = max(max_z, position.z)
            center_x = 0.5 * (min_x + max_x)
            center_y = 0.5 * (min_y + max_y)
            center_z = 0.5 * (min_z + max_z)

            best_index = 0
            best_distance_sq = float("inf")
            for index, position in enumerate(rest_positions):
                dx = position.x - center_x
                dy = position.y - center_y
                dz = position.z - center_z
                distance_sq = dx * dx + dy * dy + dz * dz
                if distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_index = index

            target_position = rest_positions[best_index]
            if self.debug_logging:
                print(
                    "[UltrasoundCentering] "
                    f"env={env_index} soft_entity={soft_entity} "
                    f"center_particle_index={best_index} "
                    f"particle_pos=({target_position.x:.4f}, {target_position.y:.4f}, {target_position.z:.4f}) "
                    f"bbox_center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f}) "
                    f"dist2={best_distance_sq:.6f}"
                )
            self._target_particle_local_indices.append(best_index)
            self._target_particle_positions.append(
                (
                    target_position.x,
                    target_position.y,
                    target_position.z,
                    0.0,
                )
            )
            self._reset_probe_state_centers.append((target_position.x, target_position.z))

            amplitude_ranges: list[neo.UltrasoundAmplitudeRange] = []
            radius_sq = self.dark_sphere_radius * self.dark_sphere_radius
            for position in rest_positions:
                dx = position.x - target_position.x
                dy = position.y - target_position.y
                dz = position.z - target_position.z
                inside = dx * dx + dy * dy + dz * dz <= radius_sq
                if inside:
                    amplitude_ranges.append(neo.UltrasoundAmplitudeRange(0.0, 0.0))
                else:
                    amplitude_ranges.append(neo.UltrasoundAmplitudeRange(0.0, 1.0))
            world.set_ultrasound_scatterer_amplitude_ranges(soft_entity, amplitude_ranges)

            probe_entity = world.create_entity(env_index)
            probe_transform = neo.TransformComponent()
            probe_transform.world_transform.position = neo.Float3(0.0, probe_height, 0.0)
            probe_transform.world_transform.rotation = probe_orientation
            world.set_transform(probe_entity, probe_transform)
            world.set_ultrasound_probe(probe_entity, probe)
            renderer = neo.UltrasoundRendererComponent()
            renderer.enabled = True
            renderer.output_width = renderer_template.output_width
            renderer.output_height = renderer_template.output_height
            renderer.output.mode = neo.RenderOutputMode.ExplicitSurface
            renderer.output.binding = neo.GpuRenderTargetBinding()
            renderer.output.binding.target = self._ultrasound_render_target
            renderer.output.binding.first_layer = env_index
            renderer.output.binding.layer_count = 1
            world.set_ultrasound_renderer(probe_entity, renderer)
            if self.enable_rgb_observation:
                probe_renderer = neo.MeshRendererComponent()
                probe_renderer.mesh = self._rgb_probe_mesh
                probe_renderer.material = probe_material
                probe_renderer.segmentation_id = 300 + env_index
                probe_renderer.visible = True
                world.set_mesh_renderer(probe_entity, probe_renderer)
            self._probe_entities.append(probe_entity)
            self._probe_pose_slots.append(world.entity_pose_slot(probe_entity))
            self._probe_base_positions.append((0.0, probe_height, 0.0, 0.0))
            self._probe_base_orientations.append(
                (probe_orientation.x, probe_orientation.y, probe_orientation.z, probe_orientation.w)
            )

            if self.enable_rgb_observation:
                self._author_rgb_camera(world, env_index)

    def _author_rgb_camera(self, world: neo.World, env_index: int) -> None:
        camera_entity = world.create_entity(env_index)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(0.0, 1.6, -2.8)
        tilt = neo.Quaternion()
        tilt_angle = math.radians(12.0)
        tilt.x = math.sin(tilt_angle * 0.5)
        tilt.y = 0.0
        tilt.z = 0.0
        tilt.w = math.cos(tilt_angle * 0.5)
        camera_transform.world_transform.rotation = tilt
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 48.0
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = self._rgb_render_target
        camera.output.binding.first_layer = env_index
        camera.output.binding.layer_count = 1
        camera.output_width = self.render_width
        camera.output_height = self.render_height
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(0.03, 0.04, 0.06, 1.0)
        world.set_camera(camera_entity, camera)

    def _build_reset_positions(self, world: neo.World) -> list[tuple[float, float, float, float]]:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        reset_positions = [
            (0.0, 0.0, 0.0, 0.0) for _ in range(self._particle_mapping.particle_count)
        ]
        for entity in self._soft_entities:
            authoring_particles = world.try_get_soft_body_authoring_particles(entity)
            if authoring_particles is None:
                raise RuntimeError(
                    f"Authoring particles were unavailable for soft body {entity}."
                )
            slot = slot_by_entity[entity]
            particle_offset = self._particle_mapping.soft_body_particle_offsets[slot]
            if self._particle_mapping.soft_body_particle_counts[slot] != authoring_particles.particle_count:
                raise RuntimeError("Prepared soft-body particle count did not match authoring data.")
            for local_index, position in enumerate(authoring_particles.rest_positions):
                reset_positions[particle_offset + local_index] = (
                    position.x,
                    position.y,
                    position.z,
                    0.0,
                )
        return reset_positions

    def _build_target_metadata(self, world: neo.World) -> None:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        for env_index, soft_entity in enumerate(self._soft_entities):
            slot = slot_by_entity[soft_entity]
            particle_offset = self._particle_mapping.soft_body_particle_offsets[slot]
            self._target_particle_indices.append(
                particle_offset + self._target_particle_local_indices[env_index]
            )

    def _initialize_probe_results(self) -> None:
        results: list[neo.UltrasoundProbeResult] = []
        for env_index, probe_entity in enumerate(self._probe_entities):
            result = self.runtime.world().try_get_ultrasound_probe_result(probe_entity)
            if result is None or not result.prepared or not result.image_binding.is_valid():
                raise RuntimeError("Prepared ultrasound probe outputs are unavailable.")
            if result.image_binding.target.id != self._ultrasound_render_target.id:
                raise RuntimeError("Prepared ultrasound probe output bound the wrong render target.")
            if result.image_binding.first_layer != env_index or result.image_binding.layer_count != 1:
                raise RuntimeError("Prepared ultrasound probe output bound the wrong render target layer.")
            results.append(result)

        self.image_width = results[0].image_width
        self.image_height = results[0].image_height
        if self.image_width != self._ultrasound_probe_layout.image_width:
            raise RuntimeError("Prepared ultrasound image width did not match the queried layout.")
        if self.image_height != self._ultrasound_probe_layout.image_height:
            raise RuntimeError("Prepared ultrasound image height did not match the queried layout.")
        for result in results[1:]:
            if result.image_width != self.image_width or result.image_height != self.image_height:
                raise RuntimeError("Ultrasound image sizes differed across environments.")
        self._probe_results = results

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.Actions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.current_frame_buffer, self.current_frame_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.CurrentFrame",
            self.env_count * self.image_width * self.image_height,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.image_height, self.image_width],
        )
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.Rewards",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.Terminated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.Truncated",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.EpisodeSteps",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.previous_errors_buffer, self.previous_errors_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.PreviousErrors",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ResetMask",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.probe_pose_slots_buffer, self.probe_pose_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ProbePoseSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.probe_base_positions_buffer, self.probe_base_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ProbeBasePositions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.probe_base_orientations_buffer, self.probe_base_orientations_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ProbeBaseOrientations",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.probe_state_buffer, self.probe_state_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ProbeState",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.reset_probe_state_buffer, self.reset_probe_state_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ResetProbeState",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.target_particle_indices_buffer, self.target_particle_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.TargetParticleIndices",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.env_particle_offsets_buffer, self.env_particle_offsets_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.EnvParticleOffsets",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.env_particle_counts_buffer, self.env_particle_counts_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.EnvParticleCounts",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count],
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "UltrasoundCentering.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
                self.runtime,
                "UltrasoundCentering.RgbObservation",
                self.env_count * self.render_width * self.render_height,
                neo.SharedBufferTensorDTypeCode.Float,
                element_stride_bytes=16,
                shape=[self.env_count, self.render_height, self.render_width, 4],
            )

    def _populate_lookup_buffers(self) -> None:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        device = self.action_tensor.device
        self.action_tensor.zero_()
        self.current_frame_tensor.zero_()
        self.reward_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.previous_errors_tensor.zero_()
        self.reset_mask_tensor.zero_()
        self.probe_pose_slots_tensor.copy_(
            torch.tensor(
                self._probe_pose_slots,
                device=device,
                dtype=self.probe_pose_slots_tensor.dtype,
            )
        )
        self.probe_base_positions_tensor.copy_(
            torch.tensor(
                self._probe_base_positions,
                device=device,
                dtype=self.probe_base_positions_tensor.dtype,
            )
        )
        self.probe_base_orientations_tensor.copy_(
            torch.tensor(
                self._probe_base_orientations,
                device=device,
                dtype=self.probe_base_orientations_tensor.dtype,
            )
        )
        self.probe_state_tensor.zero_()
        self.reset_probe_state_tensor.zero_()
        self.target_particle_indices_tensor.copy_(
            torch.tensor(
                self._target_particle_indices,
                device=device,
                dtype=self.target_particle_indices_tensor.dtype,
            )
        )
        self.env_particle_offsets_tensor.copy_(
            torch.tensor(
                [
                    self._particle_mapping.soft_body_particle_offsets[slot_by_entity[entity]]
                    for entity in self._soft_entities
                ],
                device=device,
                dtype=self.env_particle_offsets_tensor.dtype,
            )
        )
        self.env_particle_counts_tensor.copy_(
            torch.tensor(
                [
                    self._particle_mapping.soft_body_particle_counts[slot_by_entity[entity]]
                    for entity in self._soft_entities
                ],
                device=device,
                dtype=self.env_particle_counts_tensor.dtype,
            )
        )
        self.reset_positions_tensor.copy_(
            torch.tensor(
                self._reset_positions,
                device=device,
                dtype=self.reset_positions_tensor.dtype,
            )
        )
        self._sync_from_cuda(
            self.runtime,
            [
                self.action_buffer,
                self.current_frame_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
                self.previous_errors_buffer,
                self.reset_mask_buffer,
                self.probe_pose_slots_buffer,
                self.probe_base_positions_buffer,
                self.probe_base_orientations_buffer,
                self.probe_state_buffer,
                self.reset_probe_state_buffer,
                self.target_particle_indices_buffer,
                self.env_particle_offsets_buffer,
                self.env_particle_counts_buffer,
                self.reset_positions_buffer,
            ],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_tensor.zero_()
            self._sync_from_cuda(self.runtime, [self.rgb_observation_buffer])

    def _create_custom_passes(self) -> None:
        def bind(
            desc: neo.CustomComputePassDesc,
            specs: list[tuple[str, object, str, object]],
        ) -> None:
            desc.resource_bindings = [
                neo.CustomComputeResourceBindingDesc() for _ in range(len(specs))
            ]
            for binding, (name, handle, key, access) in zip(desc.resource_bindings, specs):
                binding.shader_variable_name = name
                binding.access = access
                if handle is not None:
                    binding.shared_buffer_handle = handle
                else:
                    binding.resource_key = key

        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "UltrasoundCentering.PrePhysicsControl"
        pre_desc.shader_source = _ULTRASOUND_CENTERING_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        bind(
            pre_desc,
            [
                ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbePoseSlots", self.probe_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeBasePositions", self.probe_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeBaseOrientations", self.probe_base_orientations_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeState", self.probe_state_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        pre_desc.constant_buffer_variable_name = "UltrasoundCenteringPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.lateral_action_scale,
                self.depth_action_scale,
                self.lateral_move_range,
                self.depth_move_range,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        reward_constants = struct.pack(
            "<12f",
            self.success_center_threshold,
            self.success_slice_threshold,
            0.01,
            1.0,
            1.0,
            4.0,
            0.5
            * max(
                (self.probe_num_scanlines - 1) * self.probe_scanline_spacing,
                1.0e-4,
            ),
            self.dark_sphere_radius,
            float(self.max_episode_steps),
            self.probe_line_length,
            0.0,
            0.0,
        )

        reset_probe_desc = neo.CustomComputePassDesc()
        reset_probe_desc.debug_name = "UltrasoundCentering.ResetProbe"
        reset_probe_desc.shader_source = _ULTRASOUND_CENTERING_RESET_PROBE_SHADER
        reset_probe_desc.thread_group_size_x = 64
        bind(
            reset_probe_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbePoseSlots", self.probe_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetParticleIndices", self.target_particle_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ResetProbeState", self.reset_probe_state_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeBasePositions", self.probe_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeBaseOrientations", self.probe_base_orientations_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ProbeState", self.probe_state_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_PreviousErrors", self.previous_errors_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_probe_desc.constant_buffer_variable_name = "UltrasoundCenteringResetProbeConstants"
        reset_probe_desc.constant_buffer_size_bytes = 48
        reset_probe_desc.constant_data = list(reward_constants)
        reset_probe_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_probe_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_probe_pass = self._register_custom_pass(self.runtime, reset_probe_desc)

        reset_particles_desc = neo.CustomComputePassDesc()
        reset_particles_desc.debug_name = "UltrasoundCentering.ResetParticles"
        reset_particles_desc.shader_source = _ULTRASOUND_CENTERING_RESET_PARTICLES_SHADER
        reset_particles_desc.thread_group_size_x = 64
        bind(
            reset_particles_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ResetPositions", self.reset_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticlePreviousPositions", None, "particle.previous_positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_particles_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_particles_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_particles_pass = self._register_custom_pass(self.runtime, reset_particles_desc)

        reward_desc = neo.CustomComputePassDesc()
        reward_desc.debug_name = "UltrasoundCentering.Reward"
        reward_desc.shader_source = _ULTRASOUND_CENTERING_REWARD_SHADER
        reward_desc.thread_group_size_x = 64
        bind(
            reward_desc,
            [
                ("g_ProbePoseSlots", self.probe_pose_slots_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetParticleIndices", self.target_particle_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EntityOrientations", None, "entity.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_PreviousErrors", self.previous_errors_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reward_desc.constant_buffer_variable_name = "UltrasoundCenteringRewardConstants"
        reward_desc.constant_buffer_size_bytes = 48
        reward_desc.constant_data = list(reward_constants)
        reward_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reward_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reward_pass = self._register_custom_pass(self.runtime, reward_desc)

        obs_desc = neo.CustomComputePassDesc()
        obs_desc.debug_name = "UltrasoundCentering.Observation"
        obs_desc.shader_source = _ULTRASOUND_CENTERING_OBSERVATION_SHADER
        obs_desc.thread_group_size_x = 8
        obs_desc.thread_group_size_y = 8
        obs_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
        obs_desc.resource_bindings[0].shader_variable_name = "g_UltrasoundImage"
        obs_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
        obs_desc.resource_bindings[0].render_target_binding.target = self._ultrasound_render_target
        obs_desc.resource_bindings[0].render_target_binding.first_layer = 0
        obs_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
        obs_desc.resource_bindings[0].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
        obs_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
        obs_desc.resource_bindings[1].shader_variable_name = "g_Observation"
        obs_desc.resource_bindings[1].shared_buffer_handle = self.current_frame_buffer
        obs_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
        obs_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        obs_desc.dispatch.group_count_x = (self.image_width + 7) // 8
        obs_desc.dispatch.group_count_y = (self.image_height + 7) // 8
        obs_desc.dispatch.group_count_z = self.env_count
        self._observation_pass = self._register_custom_pass(self.runtime, obs_desc)

        if self.enable_rgb_observation:
            rgb_desc = neo.CustomComputePassDesc()
            rgb_desc.debug_name = "UltrasoundCentering.RgbObservation"
            rgb_desc.shader_source = _ULTRASOUND_CENTERING_RGB_SHADER
            rgb_desc.thread_group_size_x = 8
            rgb_desc.thread_group_size_y = 8
            rgb_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
            rgb_desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
            rgb_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
            rgb_desc.resource_bindings[0].render_target_binding.target = self._rgb_render_target
            rgb_desc.resource_bindings[0].render_target_binding.first_layer = 0
            rgb_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
            rgb_desc.resource_bindings[0].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
            rgb_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
            rgb_desc.resource_bindings[1].shader_variable_name = "g_ColorObservation"
            rgb_desc.resource_bindings[1].shared_buffer_handle = self.rgb_observation_buffer
            rgb_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
            rgb_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
            rgb_desc.dispatch.group_count_x = (self.render_width + 7) // 8
            rgb_desc.dispatch.group_count_y = (self.render_height + 7) // 8
            rgb_desc.dispatch.group_count_z = self.env_count
            self._rgb_render_pass = self._register_custom_pass(self.runtime, rgb_desc)

    def _sample_reset_probe_state(self, env_indices: torch.Tensor) -> None:
        sampled = torch.empty(
            (env_indices.numel(), 2),
            device=self.reset_probe_state_tensor.device,
            dtype=self.reset_probe_state_tensor.dtype,
        )
        center_states = torch.tensor(
            [self._reset_probe_state_centers[int(env_index)] for env_index in env_indices.tolist()],
            device=self.reset_probe_state_tensor.device,
            dtype=self.reset_probe_state_tensor.dtype,
        )
        sampled[:, 0].uniform_(-self.reset_lateral_range, self.reset_lateral_range)
        sampled[:, 1].uniform_(-self.reset_depth_range, self.reset_depth_range)
        sampled += center_states
        sampled[:, 0].clamp_(-self.lateral_move_range, self.lateral_move_range)
        sampled[:, 1].clamp_(-self.depth_move_range, self.depth_move_range)
        self.reset_probe_state_tensor[env_indices] = sampled

    def _capture_current_frame(self) -> None:
        if not self.runtime.execute_custom_compute_pass(self._observation_pass):
            raise RuntimeError("Failed to execute ultrasound observation pass.")

    def _update_observation_stack(self, env_indices: torch.Tensor | None = None, *, fill: bool) -> None:
        if env_indices is None:
            current = self.current_frame_tensor
            if fill:
                self.observation_tensor.copy_(current.unsqueeze(1).expand_as(self.observation_tensor))
            else:
                self.observation_tensor[:, :-1].copy_(self.observation_tensor[:, 1:].clone())
                self.observation_tensor[:, -1].copy_(current)
            return

        if env_indices.numel() == 0:
            return
        current = self.current_frame_tensor[env_indices]
        if fill:
            self.observation_tensor[env_indices] = current.unsqueeze(1).expand(
                env_indices.numel(), self.frame_stack, self.image_height, self.image_width
            )
        else:
            for local_index, env_index in enumerate(env_indices.tolist()):
                self.observation_tensor[env_index, :-1].copy_(
                    self.observation_tensor[env_index, 1:].clone()
                )
                self.observation_tensor[env_index, -1].copy_(current[local_index])

    def _sync_step_outputs_to_cuda(self) -> None:
        self._sync_to_cuda(
            self.runtime,
            [
                self.current_frame_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
            ],
            device=self.current_frame_tensor.device,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_indices = torch.arange(
                self.env_count, device=self.action_tensor.device, dtype=torch.int64
            )
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(
                list(env_ids), device=self.action_tensor.device, dtype=torch.int64
            )
        if env_indices.numel() == 0:
            return self.observation_tensor

        self._sample_reset_probe_state(env_indices)
        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        self.action_tensor.zero_()
        self._sync_from_cuda(
            self.runtime,
            [
                self.reset_mask_buffer,
                self.action_buffer,
                self.reset_probe_state_buffer,
            ],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_particles_pass):
            raise RuntimeError("Failed to execute ultrasound particle reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_probe_pass):
            raise RuntimeError("Failed to execute ultrasound probe reset pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Failed to step physics during ultrasound reset.")
        if not self.runtime.step_simulation_sensors(self._frame):
            raise RuntimeError("Failed to step ultrasound sensors during reset.")
        self._capture_current_frame()
        self._sync_step_outputs_to_cuda()
        self._update_observation_stack(env_indices, fill=True)
        self._end_frame(self.runtime, advance=False)
        self.reset_mask_tensor.zero_()
        self._sync_from_cuda(self.runtime, [self.reset_mask_buffer])
        return self.observation_tensor

    def step(
        self, action_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if list(action_tensor.shape) != [self.env_count, self.ACTION_DIM]:
            raise ValueError(
                f"Expected action tensor shape [{self.env_count}, {self.ACTION_DIM}], "
                f"got {list(action_tensor.shape)}."
            )
        self.action_tensor.copy_(
            action_tensor.to(device=self.action_tensor.device, dtype=self.action_tensor.dtype)
        )
        self._sync_from_cuda(self.runtime, [self.action_buffer])
        if not self.runtime.execute_custom_compute_pass(self._pre_pass):
            raise RuntimeError("Failed to execute ultrasound pre-physics control pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Failed to step ultrasound physics.")
        if not self.runtime.step_simulation_sensors(self._frame):
            raise RuntimeError("Failed to step ultrasound simulation sensors.")
        if not self.runtime.execute_custom_compute_pass(self._reward_pass):
            raise RuntimeError("Failed to execute ultrasound reward pass.")
        self._capture_current_frame()
        self._sync_step_outputs_to_cuda()
        self._update_observation_stack(fill=False)
        self._end_frame(self.runtime, advance=True)
        return (
            self.observation_tensor,
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def render(self) -> torch.Tensor:
        if not self.enable_rgb_observation:
            raise RuntimeError("RGB observations were not enabled for this ultrasound env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute ultrasound RGB observation pass.")
        self._sync_to_cuda(
            self.runtime,
            [self.rgb_observation_buffer],
            device=self.rgb_observation_tensor.device,
        )
        self._end_frame(self.runtime, advance=False)
        return self.rgb_observation_tensor

    def close(self) -> None:
        self.close_runtime(getattr(self, "runtime", None))
        self.runtime = None
