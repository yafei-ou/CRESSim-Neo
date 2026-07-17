from __future__ import annotations

import math
import struct

from . import _cressim_neo as neo
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "cressim_neo.soft_body_pusher_env requires PyTorch to be installed."
    ) from exc


_SOFT_BODY_PUSHER_PRE_PHYSICS_SHADER = r"""
#include "structured_buffer_compat.hlsli"

cbuffer SoftBodyPusherPrePhysicsConstants
{
    float actionScale;
    float moveRangeX;
    float moveRangeZ;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_PusherBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_PusherBasePositions);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_PusherTargetState);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_KinematicTargetFlags);

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
    const uint bodyIndex = CRESSIM_SB_LOAD(g_PusherBodyIndices, envIndex);
    const float4 basePosition = CRESSIM_SB_LOAD(g_PusherBasePositions, envIndex);
    float2 targetState = CRESSIM_SB_LOAD(g_PusherTargetState, envIndex);
    targetState.x = clamp(targetState.x + action.x * actionScale, -1.0f, 1.0f);
    targetState.y = clamp(targetState.y + action.y * actionScale, -1.0f, 1.0f);
    CRESSIM_SB_STORE(g_PusherTargetState, envIndex, targetState);
    const float4 targetPosition =
        float4(basePosition.x + targetState.x * moveRangeX, basePosition.y,
               basePosition.z + targetState.y * moveRangeZ, 0.0f);
    CRESSIM_SB_STORE(g_KinematicTargetPositions, bodyIndex, targetPosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, bodyIndex, float4(0.0f, 0.0f, 0.0f, 1.0f));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, bodyIndex, 1u);
}
"""


_SOFT_BODY_PUSHER_POST_PHYSICS_SHADER = r"""
#include "structured_buffer_compat.hlsli"

cbuffer SoftBodyPusherPostPhysicsConstants
{
    float successFraction;
    uint maxEpisodeSteps;
    float centroidRewardScale;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_PusherBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMin);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMax);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_EnvParticleOffsets.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint particleOffset = CRESSIM_SB_LOAD(g_EnvParticleOffsets, envIndex);
    const uint particleCount = CRESSIM_SB_LOAD(g_EnvParticleCounts, envIndex);
    if (particleCount == 0u)
    {
        return;
    }

    const float4 targetMin = CRESSIM_SB_LOAD(g_TargetBoundsMin, envIndex);
    const float4 targetMax = CRESSIM_SB_LOAD(g_TargetBoundsMax, envIndex);
    const float targetCenterX = 0.5f * (targetMin.x + targetMax.x);
    const float targetCenterZ = 0.5f * (targetMin.z + targetMax.z);
    float centroidX = 0.0f;
    float centroidZ = 0.0f;
    uint insideCount = 0u;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float4 position = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        centroidX += position.x;
        centroidZ += position.z;
        if (position.x >= targetMin.x && position.x <= targetMax.x &&
            position.y >= targetMin.y && position.y <= targetMax.y &&
            position.z >= targetMin.z && position.z <= targetMax.z)
        {
            insideCount += 1u;
        }
    }

    const float invCount = 1.0f / float(particleCount);
    centroidX *= invCount;
    centroidZ *= invCount;
    const float occupancyFraction = float(insideCount) * invCount;
    const float centroidDistance =
        length(float2(centroidX - targetCenterX, centroidZ - targetCenterZ));
    const float shapingReward = max(0.0f, 1.0f - centroidDistance * centroidRewardScale);
    const float reward = occupancyFraction + 0.25f * shapingReward;
    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint terminated = occupancyFraction >= successFraction ? 1u : 0u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const uint pusherBodyIndex = CRESSIM_SB_LOAD(g_PusherBodyIndices, envIndex);
    const float4 pusherPosition = CRESSIM_SB_LOAD(g_RigidPositions, pusherBodyIndex);
    const uint obsBase = envIndex * 5u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, pusherPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, pusherPosition.z - targetCenterZ);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, centroidZ - targetCenterZ);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, occupancyFraction);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_SOFT_BODY_PUSHER_RGB_SHADER = r"""
#include "structured_buffer_compat.hlsli"

Texture2DArray<float4> g_ColorTarget;
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ColorObservation);

float toneMapReinhard(float value)
{
    return value / (1.0 + value);
}

float linearToSrgb(float value)
{
    if (value <= 0.0031308)
    {
        return value * 12.92;
    }
    return 1.055 * pow(abs(value), 1.0 / 2.4) - 0.055;
}

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
    float4 color = g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0));
    color.rgb = max(color.rgb, 0.0);
    color.r = linearToSrgb(toneMapReinhard(color.r));
    color.g = linearToSrgb(toneMapReinhard(color.g));
    color.b = linearToSrgb(toneMapReinhard(color.b));
    color = saturate(color);
    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex, color);
}
"""


_SOFT_BODY_PUSHER_RESET_PARTICLES_SHADER = r"""
#include "structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(float2, g_ResetOffsets);
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
    const float2 resetOffset = CRESSIM_SB_LOAD(g_ResetOffsets, envIndex);
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_ResetPositions, particleIndex);
        positionInvMass.x = resetPosition.x + resetOffset.x;
        positionInvMass.y = resetPosition.y;
        positionInvMass.z = resetPosition.z + resetOffset.y;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex,
                         float4(positionInvMass.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
}
"""


_SOFT_BODY_PUSHER_RESET_RIGID_SHADER = r"""
#include "structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_PusherBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_PusherBasePositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidAngularVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_KinematicTargetFlags);

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

    const uint pusherBodyIndex = CRESSIM_SB_LOAD(g_PusherBodyIndices, envIndex);
    const float4 pusherBasePosition = CRESSIM_SB_LOAD(g_PusherBasePositions, envIndex);
    CRESSIM_SB_STORE(g_RigidPositions, pusherBodyIndex, pusherBasePosition);
    CRESSIM_SB_STORE(g_RigidOrientations, pusherBodyIndex, float4(0.0f, 0.0f, 0.0f, 1.0f));
    CRESSIM_SB_STORE(g_RigidLinearVelocities, pusherBodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    CRESSIM_SB_STORE(g_RigidAngularVelocities, pusherBodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    CRESSIM_SB_STORE(g_KinematicTargetPositions, pusherBodyIndex, pusherBasePosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, pusherBodyIndex,
                     float4(0.0f, 0.0f, 0.0f, 1.0f));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, pusherBodyIndex, 1u);
}
"""


_SOFT_BODY_PUSHER_RESET_OUTPUTS_SHADER = r"""
#include "structured_buffer_compat.hlsli"

cbuffer SoftBodyPusherResetOutputsConstants
{
    float centroidRewardScale;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_PusherBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMin);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMax);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

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
    const float4 targetMin = CRESSIM_SB_LOAD(g_TargetBoundsMin, envIndex);
    const float4 targetMax = CRESSIM_SB_LOAD(g_TargetBoundsMax, envIndex);
    const float targetCenterX = 0.5f * (targetMin.x + targetMax.x);
    const float targetCenterZ = 0.5f * (targetMin.z + targetMax.z);
    float centroidX = 0.0f;
    float centroidZ = 0.0f;
    uint insideCount = 0u;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        centroidX += positionInvMass.x;
        centroidZ += positionInvMass.z;
        if (positionInvMass.x >= targetMin.x && positionInvMass.x <= targetMax.x &&
            positionInvMass.y >= targetMin.y && positionInvMass.y <= targetMax.y &&
            positionInvMass.z >= targetMin.z && positionInvMass.z <= targetMax.z)
        {
            insideCount += 1u;
        }
    }

    centroidX /= float(particleCount);
    centroidZ /= float(particleCount);
    const float occupancyFraction = float(insideCount) / float(particleCount);
    const float centroidDistance =
        length(float2(centroidX - targetCenterX, centroidZ - targetCenterZ));
    const float reward = occupancyFraction +
                         0.25f * max(0.0f, 1.0f - centroidDistance * centroidRewardScale);
    const uint pusherBodyIndex = CRESSIM_SB_LOAD(g_PusherBodyIndices, envIndex);
    const float4 pusherPosition = CRESSIM_SB_LOAD(g_RigidPositions, pusherBodyIndex);
    const uint obsBase = envIndex * 5u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, pusherPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, pusherPosition.z - targetCenterZ);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, centroidZ - targetCenterZ);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, occupancyFraction);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);
}
"""


def _make_box_collider(
    half_extents: neo.Float3,
    *,
    local_position: neo.Float3 | None = None,
    collision_layer: int = 0xFFFFFFFF,
    collision_mask: int = 0xFFFFFFFF,
) -> neo.ColliderComponent:
    collider = neo.ColliderComponent()
    collider.shape_type = neo.ColliderShapeType.Box
    collider.shape_params = neo.Float4(half_extents.x, half_extents.y, half_extents.z, 0.0)
    collider.local_position = local_position or neo.Float3(0.0, 0.0, 0.0)
    collider.collision_layer = collision_layer
    collider.collision_mask = collision_mask
    return collider


class SoftBodyPusherTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 2
    OBSERVATION_DIM = 5
    SOFT_SIZE = neo.Float3(0.45, 0.45, 0.45)
    SOFT_PARTICLE_SPACING = 0.08
    SOFT_PARTICLE_RADIUS = 0.04

    def __init__(
        self,
        env_count: int = 32,
        max_episode_steps: int = 240,
        success_fraction: float = 0.7,
        reset_position_range: float = 0.1,
        action_scale: float = 1.0,
        pusher_move_range_x: float = 1.0,
        pusher_move_range_z: float = 0.35,
        enable_rgb_observation: bool = False,
        image_width: int = 160,
        image_height: int = 160,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.success_fraction = success_fraction
        self.reset_position_range = reset_position_range
        self.action_scale = action_scale
        self.pusher_move_range_x = pusher_move_range_x
        self.pusher_move_range_z = pusher_move_range_z
        self.enable_rgb_observation = enable_rgb_observation
        self.image_width = image_width
        self.image_height = image_height

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.physics_desc.substeps = 4
        config.physics_desc.default_iterations = 16
        config.physics_desc.soft_internal_iterations = 32
        config.physics_desc.soft_contact_iterations = 16
        config.physics_desc.rigid_rigid_contact_iterations = 8
        config.scene_layout.env_count = env_count
        if enable_rgb_observation:
            config.scene_layout.max_renderable_objects_per_env = 8
            config.scene_layout.max_lights_per_env = 2
            config.scene_layout.max_cameras_per_env = 1

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize soft-body pusher runtime.")

        if self.enable_rgb_observation:
            self._initialize_rgb_observation_resources()

        self._soft_entities: list[int] = []
        self._pusher_entities: list[int] = []
        self._pusher_base_positions: list[tuple[float, float, float, float]] = []
        self._target_bounds_min: list[tuple[float, float, float, float]] = []
        self._target_bounds_max: list[tuple[float, float, float, float]] = []
        self._author_scene(self.runtime.world())
        self.runtime.prepare()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._rigid_mapping = self.runtime.get_prepared_rigid_layout_mapping()
        self._reset_positions = self._build_reset_positions(self.runtime.world())
        self._pusher_body_indices = self._build_pusher_body_indices()
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared soft-body pusher world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self._end_frame(self.runtime, advance=False)

    def _initialize_rgb_observation_resources(self) -> None:
        resources = self.runtime.resources()
        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.image_width
        target_desc.height = self.image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA16Float
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "SoftBodyPusher.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create soft-body pusher RGB render target.")

        self._rgb_soft_mesh = resources.register_mesh(
            neo.make_box_mesh(
                neo.Float3(
                    0.5 * self.SOFT_SIZE.x,
                    0.5 * self.SOFT_SIZE.y,
                    0.5 * self.SOFT_SIZE.z,
                ),
                "SoftBodyPusher.RenderSoftBody",
            )
        )
        self._rgb_capsule_mesh = resources.register_mesh(
            neo.make_capsule_mesh(0.10, 0.18, 24, 8, 4, "SoftBodyPusher.RenderCapsule")
        )
        ground_half = neo.Float3(1.25, 0.05, 0.85)
        self._rgb_plane_mesh = resources.register_mesh(
            neo.make_box_mesh(ground_half, "SoftBodyPusher.RenderGround")
        )

    def _author_scene(self, world: neo.World) -> None:
        ground_half = neo.Float3(1.25, 0.05, 0.85)
        target_half = neo.Float3(0.18, 0.3, 0.18)
        soft_position_x = -0.25
        pusher_position_x = -0.90
        ground_top = 0.0
        soft_half_height = 0.5 * self.SOFT_SIZE.y
        soft_spawn_height = ground_top + soft_half_height + 0.08
        pusher_radius = 0.10
        pusher_half_height = 0.18
        pusher_height = ground_top + pusher_radius + pusher_half_height + 0.02

        for env_index in range(self.env_count):
            z_offset = float(env_index) * 2.5

            ground_entity = world.create_entity(env_index)
            ground_transform = neo.TransformComponent()
            ground_transform.world_transform.position = neo.Float3(0.0, -0.05, z_offset)
            world.set_transform(ground_entity, ground_transform)
            ground_body = neo.RigidBodyComponent()
            ground_body.body_type = neo.RigidBodyType.Static
            ground_body.inverse_mass = 0.0
            ground_body.simulated = True
            world.set_rigid_body(ground_entity, ground_body)
            ground_collider = _make_box_collider(ground_half)
            ground_collider.friction = 0.90
            ground_collider.static_friction = 1.10
            world.add_collider(ground_entity, ground_collider)
            if self.enable_rgb_observation:
                ground_renderer = neo.MeshRendererComponent()
                ground_renderer.mesh = self._rgb_plane_mesh
                ground_renderer.material = self._make_material(
                    f"SoftBodyPusher.GroundMaterial.{env_index}",
                    neo.Float3(0.62, 0.64, 0.68),
                    0.92,
                )
                ground_renderer.segmentation_id = 100 + env_index
                ground_renderer.visible = True
                world.set_mesh_renderer(ground_entity, ground_renderer)

            soft_entity = world.create_entity(env_index)
            soft_transform = neo.TransformComponent()
            soft_transform.world_transform.position = neo.Float3(
                soft_position_x, soft_spawn_height, z_offset
            )
            world.set_transform(soft_entity, soft_transform)
            if self.enable_rgb_observation:
                soft_renderer = neo.MeshRendererComponent()
                soft_renderer.mesh = self._rgb_soft_mesh
                soft_renderer.material = self._make_material(
                    f"SoftBodyPusher.SoftMaterial.{env_index}",
                    neo.Float3(0.26, 0.58, 0.92),
                    0.45,
                )
                soft_renderer.segmentation_id = 200 + env_index
                soft_renderer.visible = True
                world.set_mesh_renderer(soft_entity, soft_renderer)
            soft = neo.SoftBodyComponent()
            soft.source.kind = neo.SoftBodySourceKind.RegularGrid
            soft.source.regular_grid.size = self.SOFT_SIZE
            soft.source.regular_grid.target_particle_spacing = self.SOFT_PARTICLE_SPACING
            soft.particle_mass = 0.01
            soft.particle_radius = self.SOFT_PARTICLE_RADIUS
            soft.edge_compliance = 0.0
            soft.volume_compliance = 8.0e-4
            soft.material.contact.friction = 0.82
            soft.material.contact.static_friction = 1.05
            soft.material.contact.damping = 0.35
            soft.simulated = True
            soft.self_collision_enabled = True
            soft.collision_layer = 0x1
            soft.collision_mask = 0xFFFFFFFF
            if not world.set_soft_body(soft_entity, soft):
                raise RuntimeError(f"Failed to author soft body for env {env_index}.")
            self._soft_entities.append(soft_entity)

            pusher_entity = world.create_entity(env_index)
            pusher_transform = neo.TransformComponent()
            pusher_transform.world_transform.position = neo.Float3(
                pusher_position_x, pusher_height, z_offset
            )
            world.set_transform(pusher_entity, pusher_transform)
            pusher_body = neo.RigidBodyComponent()
            pusher_body.body_type = neo.RigidBodyType.Kinematic
            pusher_body.inverse_mass = 1.0
            pusher_body.simulated = True
            pusher_body.kinematic_target_enabled = True
            pusher_body.kinematic_target_position = pusher_transform.world_transform.position
            world.set_rigid_body(pusher_entity, pusher_body)
            pusher_collider = neo.ColliderComponent()
            pusher_collider.shape_type = neo.ColliderShapeType.Capsule
            pusher_collider.shape_params = neo.Float4(
                pusher_radius, pusher_half_height, 0.0, 0.0
            )
            pusher_collider.friction = 0.90
            pusher_collider.static_friction = 1.10
            world.add_collider(pusher_entity, pusher_collider)
            if self.enable_rgb_observation:
                pusher_renderer = neo.MeshRendererComponent()
                pusher_renderer.mesh = self._rgb_capsule_mesh
                pusher_renderer.material = self._make_material(
                    f"SoftBodyPusher.PusherMaterial.{env_index}",
                    neo.Float3(0.86, 0.34, 0.24),
                    0.40,
                )
                pusher_renderer.segmentation_id = 300 + env_index
                pusher_renderer.visible = True
                world.set_mesh_renderer(pusher_entity, pusher_renderer)
            self._pusher_entities.append(pusher_entity)
            self._pusher_base_positions.append(
                (pusher_position_x, pusher_height, z_offset, 0.0)
            )

            target_center = neo.Float3(0.45, 0.3, z_offset)
            self._target_bounds_min.append(
                (
                    target_center.x - target_half.x,
                    0.0,
                    target_center.z - target_half.z,
                    0.0,
                )
            )
            self._target_bounds_max.append(
                (
                    target_center.x + target_half.x,
                    target_center.y + target_half.y,
                    target_center.z + target_half.z,
                    0.0,
                )
            )
            if self.enable_rgb_observation:
                self._author_rgb_camera(world, env_index, z_offset)

    def _make_material(
        self,
        debug_name: str,
        base_color: neo.Float3,
        roughness: float,
    ) -> neo.MaterialHandle:
        material_desc = neo.MaterialResourceDesc()
        material_desc.debug_name = debug_name
        material_desc.base_color = base_color
        material_desc.metallic = 0.0
        material_desc.roughness = roughness
        return self.runtime.resources().register_material(material_desc)

    def _author_rgb_camera(self, world: neo.World, env_index: int, z_offset: float) -> None:
        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.35, -1.0, 0.25)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 7.5
        light.casts_shadows = True
        world.set_directional_light(light_entity, light)

        camera_entity = world.create_entity(env_index)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(0.0, 1.65, z_offset - 3.35)
        tilt = neo.Quaternion()
        tilt_angle = math.radians(24.0)
        tilt.x = math.sin(tilt_angle * 0.5)
        tilt.y = 0.0
        tilt.z = 0.0
        tilt.w = math.cos(tilt_angle * 0.5)
        camera_transform.world_transform.rotation = tilt
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 40.0
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = self._rgb_render_target
        camera.output.binding.first_layer = env_index
        camera.output.binding.layer_count = 1
        camera.output_width = self.image_width
        camera.output_height = self.image_height
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
            particle_count = self._particle_mapping.soft_body_particle_counts[slot]
            if particle_count != authoring_particles.particle_count:
                raise RuntimeError(
                    "Prepared soft-body particle count did not match authoring data."
                )
            for local_index, position in enumerate(authoring_particles.rest_positions):
                reset_positions[particle_offset + local_index] = (
                    position.x,
                    position.y,
                    position.z,
                    0.0,
                )
        return reset_positions

    def _build_pusher_body_indices(self) -> list[int]:
        slot_by_entity = {
            entity_id: slot for slot, entity_id in enumerate(self._rigid_mapping.rigid_body_entity_ids)
        }
        return [slot_by_entity[entity] for entity in self._pusher_entities]

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.Actions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.observation_buffer, observation_flat = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.Terminated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.Truncated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.EpisodeSteps", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.ResetMask", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_offsets_buffer, self.reset_offsets_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.ResetOffsets",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.env_particle_offsets_buffer, self.env_particle_offsets_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.ParticleOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.env_particle_counts_buffer, self.env_particle_counts_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.ParticleCounts", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.pusher_body_indices_buffer, self.pusher_body_indices_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyPusher.BodyIndices", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.pusher_base_positions_buffer, self.pusher_base_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.BasePositions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.pusher_target_state_buffer, self.pusher_target_state_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.TargetState",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.target_bounds_min_buffer, self.target_bounds_min_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.TargetBoundsMin",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.target_bounds_max_buffer, self.target_bounds_max_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.TargetBoundsMax",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyPusher.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
                self.runtime,
                "SoftBodyPusher.RgbObservation",
                self.env_count * self.image_width * self.image_height,
                neo.SharedBufferTensorDTypeCode.Float,
                element_stride_bytes=16,
                shape=[self.env_count, self.image_height, self.image_width, 4],
            )

    def _populate_lookup_buffers(self) -> None:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        device = self.action_tensor.device
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
        self.pusher_body_indices_tensor.copy_(
            torch.tensor(
                self._pusher_body_indices,
                device=device,
                dtype=self.pusher_body_indices_tensor.dtype,
            )
        )
        self.pusher_base_positions_tensor.copy_(
            torch.tensor(
                self._pusher_base_positions,
                device=device,
                dtype=self.pusher_base_positions_tensor.dtype,
            )
        )
        self.pusher_target_state_tensor.zero_()
        self.target_bounds_min_tensor.copy_(
            torch.tensor(
                self._target_bounds_min,
                device=device,
                dtype=self.target_bounds_min_tensor.dtype,
            )
        )
        self.target_bounds_max_tensor.copy_(
            torch.tensor(
                self._target_bounds_max,
                device=device,
                dtype=self.target_bounds_max_tensor.dtype,
            )
        )
        self.action_tensor.zero_()
        self.observation_tensor.zero_()
        self.reward_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.reset_mask_tensor.zero_()
        self.reset_offsets_tensor.zero_()
        self.reset_positions_tensor.copy_(
            torch.tensor(self._reset_positions, device=device, dtype=self.reset_positions_tensor.dtype)
        )
        self._sync_from_cuda(
            self.runtime,
            [
                self.env_particle_offsets_buffer,
                self.env_particle_counts_buffer,
                self.pusher_body_indices_buffer,
                self.pusher_base_positions_buffer,
                self.pusher_target_state_buffer,
                self.target_bounds_min_buffer,
                self.target_bounds_max_buffer,
                self.action_buffer,
                self.observation_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
                self.reset_mask_buffer,
                self.reset_offsets_buffer,
                self.reset_positions_buffer,
            ],
        )

    def _create_custom_passes(self) -> None:
        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "SoftBodyPusher.PrePhysicsControl"
        pre_desc.shader_source = _SOFT_BODY_PUSHER_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        pre_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(7)]
        specs = [
            ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PusherBodyIndices", self.pusher_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PusherBasePositions", self.pusher_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PusherTargetState", self.pusher_target_state_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_KinematicTargetPositions", None, "rigid.kinematic_target_positions", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_KinematicTargetOrientations", None, "rigid.kinematic_target_orientations", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_KinematicTargetFlags", None, "rigid.kinematic_target_flags", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(pre_desc.resource_bindings, specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        pre_desc.constant_buffer_variable_name = "SoftBodyPusherPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.action_scale,
                self.pusher_move_range_x,
                self.pusher_move_range_z,
                0.0,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "SoftBodyPusher.PostPhysicsObservations"
        post_desc.shader_source = _SOFT_BODY_PUSHER_POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        post_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(12)]
        specs = [
            ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_PusherBodyIndices", self.pusher_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_TargetBoundsMin", self.target_bounds_min_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_TargetBoundsMax", self.target_bounds_max_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidPositions", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(post_desc.resource_bindings, specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        post_desc.constant_buffer_variable_name = "SoftBodyPusherPostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 16
        post_desc.constant_data = list(
            struct.pack("<fIff", self.success_fraction, self.max_episode_steps, 2.0, 0.0)
        )
        post_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._post_pass = self._register_custom_pass(self.runtime, post_desc)

        def bind(desc: neo.CustomComputePassDesc, specs: list[tuple[str, object, str, object]]) -> None:
            desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(len(specs))]
            for binding, (name, handle, key, access) in zip(desc.resource_bindings, specs):
                binding.shader_variable_name = name
                binding.access = access
                if handle is not None:
                    binding.shared_buffer_handle = handle
                else:
                    binding.resource_key = key

        reset_particles_desc = neo.CustomComputePassDesc()
        reset_particles_desc.debug_name = "SoftBodyPusher.ResetParticles"
        reset_particles_desc.shader_source = _SOFT_BODY_PUSHER_RESET_PARTICLES_SHADER
        reset_particles_desc.thread_group_size_x = 64
        bind(
            reset_particles_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ResetOffsets", self.reset_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
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

        reset_rigid_desc = neo.CustomComputePassDesc()
        reset_rigid_desc.debug_name = "SoftBodyPusher.ResetRigid"
        reset_rigid_desc.shader_source = _SOFT_BODY_PUSHER_RESET_RIGID_SHADER
        reset_rigid_desc.thread_group_size_x = 64
        bind(
            reset_rigid_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_PusherBodyIndices", self.pusher_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_PusherBasePositions", self.pusher_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositions", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidLinearVelocities", None, "rigid.linear_velocities", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidAngularVelocities", None, "rigid.angular_velocities", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_KinematicTargetPositions", None, "rigid.kinematic_target_positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_KinematicTargetOrientations", None, "rigid.kinematic_target_orientations", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_KinematicTargetFlags", None, "rigid.kinematic_target_flags", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_rigid_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_rigid_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_rigid_pass = self._register_custom_pass(self.runtime, reset_rigid_desc)

        reset_outputs_desc = neo.CustomComputePassDesc()
        reset_outputs_desc.debug_name = "SoftBodyPusher.ResetOutputs"
        reset_outputs_desc.shader_source = _SOFT_BODY_PUSHER_RESET_OUTPUTS_SHADER
        reset_outputs_desc.thread_group_size_x = 64
        bind(
            reset_outputs_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_PusherBodyIndices", self.pusher_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetBoundsMin", self.target_bounds_min_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetBoundsMax", self.target_bounds_max_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositions", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_outputs_desc.constant_buffer_variable_name = "SoftBodyPusherResetOutputsConstants"
        reset_outputs_desc.constant_buffer_size_bytes = 16
        reset_outputs_desc.constant_data = list(struct.pack("<4f", 2.0, 0.0, 0.0, 0.0))
        reset_outputs_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_outputs_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_outputs_pass = self._register_custom_pass(self.runtime, reset_outputs_desc)

        if self.enable_rgb_observation:
            render_desc = neo.CustomComputePassDesc()
            render_desc.debug_name = "SoftBodyPusher.RgbObservation"
            render_desc.shader_source = _SOFT_BODY_PUSHER_RGB_SHADER
            render_desc.thread_group_size_x = 8
            render_desc.thread_group_size_y = 8
            render_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
            render_desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
            render_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
            render_desc.resource_bindings[0].render_target_binding.target = self._rgb_render_target
            render_desc.resource_bindings[0].render_target_binding.first_layer = 0
            render_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
            render_desc.resource_bindings[0].render_target_texture_plane = neo.GpuRenderTargetTexturePlane.Color
            render_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
            render_desc.resource_bindings[1].shader_variable_name = "g_ColorObservation"
            render_desc.resource_bindings[1].shared_buffer_handle = self.rgb_observation_buffer
            render_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
            render_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
            render_desc.dispatch.group_count_x = (self.image_width + 7) // 8
            render_desc.dispatch.group_count_y = (self.image_height + 7) // 8
            render_desc.dispatch.group_count_z = self.env_count
            self._rgb_render_pass = self._register_custom_pass(self.runtime, render_desc)

    def _sync_outputs_to_cuda(self) -> None:
        self._sync_to_cuda(
            self.runtime,
            [
                self.observation_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
            ],
            device=self.observation_tensor.device,
        )

    def reset(self, env_ids: "torch.Tensor | list[int] | None" = None) -> "torch.Tensor":
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

        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        self.reset_offsets_tensor.zero_()
        self.action_tensor.zero_()
        self.pusher_target_state_tensor.index_fill_(0, env_indices, 0.0)
        sampled_offsets = torch.empty(
            (env_indices.numel(), 2),
            device=self.reset_offsets_tensor.device,
            dtype=self.reset_offsets_tensor.dtype,
        )
        sampled_offsets.uniform_(-self.reset_position_range, self.reset_position_range)
        self.reset_offsets_tensor.index_copy_(0, env_indices, sampled_offsets)
        self._sync_from_cuda(
            self.runtime,
            [
                self.reset_mask_buffer,
                self.reset_offsets_buffer,
                self.pusher_target_state_buffer,
                self.action_buffer,
            ],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_particles_pass):
            raise RuntimeError("Failed to execute soft-body pusher particle reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_rigid_pass):
            raise RuntimeError("Failed to execute soft-body pusher rigid reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_outputs_pass):
            raise RuntimeError("Failed to execute soft-body pusher output reset pass.")
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=False)
        return self.observation_tensor

    def step(
        self, action_tensor: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
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
            raise RuntimeError("Failed to execute soft-body pusher pre-physics pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Soft-body pusher physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute soft-body pusher post-physics pass.")
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=True)
        return (
            self.observation_tensor,
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def close(self) -> None:
        self.close_runtime(getattr(self, "runtime", None))
        self.runtime = None

    def render(self) -> "torch.Tensor":
        if not self.enable_rgb_observation:
            raise RuntimeError("RGB observations were not enabled for this soft-body env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute soft-body pusher RGB observation pass.")
        if not self.runtime.sync_shared_buffer_to_cuda(self.rgb_observation_buffer):
            raise RuntimeError("Failed to synchronize soft-body pusher RGB observation buffer to CUDA.")
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor
