from __future__ import annotations

import math
import struct

from . import _cressim_neo as neo
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "cressim_neo.fluid_pouring_env requires PyTorch to be installed."
    ) from exc


_FLUID_POURING_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer FluidPouringPrePhysicsConstants
{
    float positionActionScale;
    float tiltActionScale;
    float moveRangeX;
    float maxTiltRadians;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_SourceBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_SourceBasePositions);
CRESSIM_RW_STRUCTURED_BUFFER(float2, g_SourceTargetState);
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
    const uint bodyIndex = CRESSIM_SB_LOAD(g_SourceBodyIndices, envIndex);
    const float4 basePosition = CRESSIM_SB_LOAD(g_SourceBasePositions, envIndex);
    float2 targetState = CRESSIM_SB_LOAD(g_SourceTargetState, envIndex);
    targetState.x = clamp(targetState.x + action.x * positionActionScale, -1.0f, 1.0f);
    targetState.y = clamp(targetState.y + action.y * tiltActionScale, -1.0f, 1.0f);
    CRESSIM_SB_STORE(g_SourceTargetState, envIndex, targetState);
    const float tiltAngle = targetState.y * maxTiltRadians;
    float sinHalf = 0.0f;
    float cosHalf = 1.0f;
    sincos(tiltAngle * 0.5f, sinHalf, cosHalf);
    const float4 targetPosition =
        float4(basePosition.x + targetState.x * moveRangeX, basePosition.y,
               basePosition.z, 0.0f);
    CRESSIM_SB_STORE(g_KinematicTargetPositions, bodyIndex, targetPosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, bodyIndex,
                     float4(0.0f, 0.0f, sinHalf, cosHalf));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, bodyIndex, 1u);
}
"""


_FLUID_POURING_POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer FluidPouringPostPhysicsConstants
{
    float successFraction;
    uint maxEpisodeSteps;
    float spillPenalty;
    float spillPlaneY;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_SourceBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMin);
CRESSIM_STRUCTURED_BUFFER(float4, g_TargetBoundsMax);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
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
    float centroidX = 0.0f;
    float centroidY = 0.0f;
    uint targetCount = 0u;
    uint spillCount = 0u;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float4 position = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        centroidX += position.x;
        centroidY += position.y;
        if (position.x >= targetMin.x && position.x <= targetMax.x &&
            position.y >= targetMin.y && position.y <= targetMax.y &&
            position.z >= targetMin.z && position.z <= targetMax.z)
        {
            targetCount += 1u;
        }
        else if (position.y < spillPlaneY)
        {
            spillCount += 1u;
        }
    }

    const float invCount = 1.0f / float(particleCount);
    centroidX *= invCount;
    centroidY *= invCount;
    const float targetFraction = float(targetCount) * invCount;
    const float spillFraction = float(spillCount) * invCount;
    const float reward = max(0.0f, targetFraction - spillPenalty * spillFraction);
    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint terminated = targetFraction >= successFraction ? 1u : 0u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const uint sourceBodyIndex = CRESSIM_SB_LOAD(g_SourceBodyIndices, envIndex);
    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidPositions, sourceBodyIndex);
    const float4 sourceOrientation = CRESSIM_SB_LOAD(g_RigidOrientations, sourceBodyIndex);
    const float tiltAngle = 2.0f * atan2(sourceOrientation.z, sourceOrientation.w);
    const uint obsBase = envIndex * 6u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, sourcePosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, tiltAngle);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, centroidY);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, targetFraction);
    CRESSIM_SB_STORE(g_Observations, obsBase + 5u, spillFraction);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_FLUID_POURING_RGB_SHADER = r"""
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
    const float4 color = saturate(g_ColorTarget.Load(int4(int(x), int(y), int(envIndex), 0)));
    CRESSIM_SB_STORE(g_ColorObservation, pixelIndex, color);
}
"""


_FLUID_POURING_RESET_PARTICLES_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(float, g_ResetOffsets);
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
    const float resetOffset = CRESSIM_SB_LOAD(g_ResetOffsets, envIndex);
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_ResetPositions, particleIndex);
        positionInvMass.x = resetPosition.x + resetOffset;
        positionInvMass.y = resetPosition.y;
        positionInvMass.z = resetPosition.z;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex,
                         float4(positionInvMass.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
}
"""


_FLUID_POURING_RESET_RIGID_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_SourceBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_SourceBasePositions);
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

    const uint sourceBodyIndex = CRESSIM_SB_LOAD(g_SourceBodyIndices, envIndex);
    const float4 sourceBasePosition = CRESSIM_SB_LOAD(g_SourceBasePositions, envIndex);
    CRESSIM_SB_STORE(g_RigidPositions, sourceBodyIndex, sourceBasePosition);
    CRESSIM_SB_STORE(g_RigidOrientations, sourceBodyIndex, float4(0.0f, 0.0f, 0.0f, 1.0f));
    CRESSIM_SB_STORE(g_RigidLinearVelocities, sourceBodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    CRESSIM_SB_STORE(g_RigidAngularVelocities, sourceBodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    CRESSIM_SB_STORE(g_KinematicTargetPositions, sourceBodyIndex, sourceBasePosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, sourceBodyIndex,
                     float4(0.0f, 0.0f, 0.0f, 1.0f));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, sourceBodyIndex, 1u);
}
"""


_FLUID_POURING_RESET_OUTPUTS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer FluidPouringResetOutputsConstants
{
    float spillPenalty;
    float spillPlaneY;
    float padding0;
    float padding1;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_SourceBodyIndices);
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
    float centroidX = 0.0f;
    float centroidY = 0.0f;
    uint targetCount = 0u;
    uint spillCount = 0u;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        centroidX += positionInvMass.x;
        centroidY += positionInvMass.y;
        if (positionInvMass.x >= targetMin.x && positionInvMass.x <= targetMax.x &&
            positionInvMass.y >= targetMin.y && positionInvMass.y <= targetMax.y &&
            positionInvMass.z >= targetMin.z && positionInvMass.z <= targetMax.z)
        {
            targetCount += 1u;
        }
        else if (positionInvMass.y < spillPlaneY)
        {
            spillCount += 1u;
        }
    }

    centroidX /= float(particleCount);
    centroidY /= float(particleCount);
    const float targetFraction = float(targetCount) / float(particleCount);
    const float spillFraction = float(spillCount) / float(particleCount);
    const float reward = max(0.0f, targetFraction - spillPenalty * spillFraction);
    const uint sourceBodyIndex = CRESSIM_SB_LOAD(g_SourceBodyIndices, envIndex);
    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidPositions, sourceBodyIndex);
    const uint obsBase = envIndex * 6u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, sourcePosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, centroidY);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, targetFraction);
    CRESSIM_SB_STORE(g_Observations, obsBase + 5u, spillFraction);
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


def _make_container_colliders(
    inner_half: neo.Float3, wall_thickness: float
) -> list[neo.ColliderComponent]:
    return [
        _make_box_collider(half_extents, local_position=local_position)
        for local_position, half_extents in _container_wall_specs(inner_half, wall_thickness)
    ]


def _container_wall_specs(
    inner_half: neo.Float3, wall_thickness: float
) -> list[tuple[neo.Float3, neo.Float3]]:
    outer_x = inner_half.x + 2.0 * wall_thickness
    outer_z = inner_half.z + 2.0 * wall_thickness
    wall_half_y = inner_half.y
    return [
        (
            neo.Float3(0.0, -inner_half.y - wall_thickness, 0.0),
            neo.Float3(outer_x, wall_thickness, outer_z),
        ),
        (
            neo.Float3(-inner_half.x - wall_thickness, 0.0, 0.0),
            neo.Float3(wall_thickness, wall_half_y, inner_half.z),
        ),
        (
            neo.Float3(inner_half.x + wall_thickness, 0.0, 0.0),
            neo.Float3(wall_thickness, wall_half_y, inner_half.z),
        ),
        (
            neo.Float3(0.0, 0.0, -inner_half.z - wall_thickness),
            neo.Float3(outer_x, wall_half_y, wall_thickness),
        ),
        (
            neo.Float3(0.0, 0.0, inner_half.z + wall_thickness),
            neo.Float3(outer_x, wall_half_y, wall_thickness),
        ),
    ]


def _container_render_box_specs(
    inner_half: neo.Float3, wall_thickness: float
) -> list[tuple[neo.Float3, neo.Float3]]:
    return _container_wall_specs(inner_half, wall_thickness)


def _make_container_mesh(
    inner_half: neo.Float3, wall_thickness: float, debug_name: str
) -> neo.MeshResourceDesc:
    mesh = neo.MeshResourceDesc()
    mesh.debug_name = debug_name
    vertices: list[neo.MeshVertex] = []
    indices: list[int] = []
    base_mesh = neo.make_cube_mesh(1.0, "FluidPouring.ContainerWall")
    for local_position, half_extents in _container_render_box_specs(
        inner_half, wall_thickness
    ):
        base_vertex = len(vertices)
        for vertex in base_mesh.vertices:
            combined = neo.MeshVertex()
            combined.position = neo.Float3(
                local_position.x + vertex.position.x * half_extents.x,
                local_position.y + vertex.position.y * half_extents.y,
                local_position.z + vertex.position.z * half_extents.z,
            )
            combined.normal = vertex.normal
            combined.tex_coord_u = vertex.tex_coord_u
            combined.tex_coord_v = vertex.tex_coord_v
            combined.tangent = vertex.tangent
            vertices.append(combined)
        for index in base_mesh.indices:
            indices.append(base_vertex + index)
    mesh.vertices = vertices
    mesh.indices = indices
    return mesh


def _compute_regular_grid_axis(
    inner_half_extent: float, particle_radius: float, fill_fraction: float
) -> tuple[int, float]:
    spacing = 2.0 * particle_radius
    max_count = max(
        1,
        int(math.floor((2.0 * max(0.0, inner_half_extent - particle_radius)) / spacing)) + 1,
    )
    count = max(1, min(max_count, int(math.floor(max_count * fill_fraction + 0.5))))
    return count, float(count) * spacing


def _compute_fluid_block_desc(
    cup_inner_half: neo.Float3,
    particle_radius: float,
    fill_fraction_xy: float,
    fill_fraction_height: float,
) -> tuple[neo.Float3, float]:
    spacing = 2.0 * particle_radius
    _, size_x = _compute_regular_grid_axis(cup_inner_half.x, particle_radius, fill_fraction_xy)
    _, size_z = _compute_regular_grid_axis(cup_inner_half.z, particle_radius, fill_fraction_xy)
    _, size_y = _compute_regular_grid_axis(cup_inner_half.y, particle_radius, fill_fraction_height)
    return neo.Float3(size_x, size_y, size_z), spacing


class FluidPouringTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 2
    OBSERVATION_DIM = 6
    FLUID_PARTICLE_RADIUS = 0.09
    FLUID_HORIZONTAL_FILL_FRACTION = 1.0
    FLUID_HEIGHT_FILL_FRACTION = 1.0

    def __init__(
        self,
        env_count: int = 32,
        max_episode_steps: int = 240,
        success_fraction: float = 0.40,
        reset_position_range: float = 0.0,
        position_action_scale: float = 1.0,
        tilt_action_scale: float = 1.0,
        source_move_range_x: float = 1.1,
        max_tilt_radians: float = 0.5 * math.pi,
        enable_rgb_observation: bool = False,
        image_width: int = 160,
        image_height: int = 160,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.success_fraction = success_fraction
        self.reset_position_range = reset_position_range
        self.position_action_scale = position_action_scale
        self.tilt_action_scale = tilt_action_scale
        self.source_move_range_x = source_move_range_x
        self.max_tilt_radians = max_tilt_radians
        self.enable_rgb_observation = enable_rgb_observation
        self.image_width = image_width
        self.image_height = image_height

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.scene_layout.env_count = env_count
        if enable_rgb_observation:
            config.scene_layout.max_renderable_objects_per_env = 16
            config.scene_layout.max_lights_per_env = 2
            config.scene_layout.max_cameras_per_env = 1

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize fluid pouring runtime.")

        if self.enable_rgb_observation:
            self._initialize_rgb_observation_resources()

        self._fluid_specs: list[tuple[int, neo.Float3, neo.Float3, float]] = []
        self._source_entities: list[int] = []
        self._source_base_positions: list[tuple[float, float, float, float]] = []
        self._target_bounds_min: list[tuple[float, float, float, float]] = []
        self._target_bounds_max: list[tuple[float, float, float, float]] = []
        self._author_scene(self.runtime.world())
        self.runtime.prepare()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._rigid_mapping = self.runtime.get_prepared_rigid_layout_mapping()
        self._source_body_indices = self._build_source_body_indices()
        self._reset_positions = self._build_reset_positions()
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared fluid pouring world.")

        self._create_shared_buffers()
        self._create_custom_passes()
        self._populate_lookup_buffers()
        self.reset()

    def _initialize_rgb_observation_resources(self) -> None:
        resources = self.runtime.resources()
        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.image_width
        target_desc.height = self.image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA8UnormSrgb
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "FluidPouring.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create fluid pouring RGB render target.")

        ground_half = neo.Float3(5.6, 0.08, 2.4)
        self._rgb_plane_mesh = resources.register_mesh(
            neo.make_box_mesh(ground_half, "FluidPouring.RenderGround")
        )
        cup_inner_half = neo.Float3(1.15, 0.95, 0.80)
        wall_thickness = 0.24
        self._rgb_cup_mesh = resources.register_mesh(
            _make_container_mesh(cup_inner_half, wall_thickness, "FluidPouring.RenderCup")
        )

    def _author_scene(self, world: neo.World) -> None:
        ground_half = neo.Float3(5.6, 0.08, 2.4)
        cup_inner_half = neo.Float3(1.15, 0.95, 0.80)
        wall_thickness = 0.24
        fluid_size, fluid_spacing = _compute_fluid_block_desc(
            cup_inner_half,
            self.FLUID_PARTICLE_RADIUS,
            self.FLUID_HORIZONTAL_FILL_FRACTION,
            self.FLUID_HEIGHT_FILL_FRACTION,
        )
        source_position_x = -3.90
        target_position_x = 0.0
        source_cup_height = 3.50
        target_cup_height = 1.10

        for env_index in range(self.env_count):
            z_offset = float(env_index) * 6.4

            ground_entity = world.create_entity(env_index)
            ground_transform = neo.TransformComponent()
            ground_transform.world_transform.position = neo.Float3(0.0, -0.05, z_offset)
            world.set_transform(ground_entity, ground_transform)
            ground_body = neo.RigidBodyComponent()
            ground_body.body_type = neo.RigidBodyType.Static
            ground_body.inverse_mass = 0.0
            ground_body.simulated = True
            world.set_rigid_body(ground_entity, ground_body)
            world.add_collider(ground_entity, _make_box_collider(ground_half))
            if self.enable_rgb_observation:
                ground_renderer = neo.MeshRendererComponent()
                ground_renderer.mesh = self._rgb_plane_mesh
                ground_renderer.material = self._make_material(
                    f"FluidPouring.GroundMaterial.{env_index}",
                    neo.Float3(0.62, 0.64, 0.68),
                    0.92,
                )
                ground_renderer.segmentation_id = 100 + env_index
                ground_renderer.visible = True
                world.set_mesh_renderer(ground_entity, ground_renderer)

            source_entity = world.create_entity(env_index)
            source_transform = neo.TransformComponent()
            source_transform.world_transform.position = neo.Float3(
                source_position_x, source_cup_height, z_offset
            )
            world.set_transform(source_entity, source_transform)
            source_body = neo.RigidBodyComponent()
            source_body.body_type = neo.RigidBodyType.Kinematic
            source_body.inverse_mass = 0.0
            source_body.inverse_inertia_local = neo.Float3(0.0, 0.0, 0.0)
            source_body.simulated = True
            source_body.kinematic_target_enabled = True
            source_body.kinematic_target_position = source_transform.world_transform.position
            world.set_rigid_body(source_entity, source_body)
            if not world.replace_colliders(
                source_entity, _make_container_colliders(cup_inner_half, wall_thickness)
            ):
                raise RuntimeError(f"Failed to author source container for env {env_index}.")
            if self.enable_rgb_observation:
                self._set_container_renderer(
                    world, source_entity, env_index, "Source", neo.Float3(0.82, 0.42, 0.22), 200 + env_index
                )
            self._source_entities.append(source_entity)
            self._source_base_positions.append(
                (source_position_x, source_cup_height, z_offset, 0.0)
            )

            target_entity = world.create_entity(env_index)
            target_transform = neo.TransformComponent()
            target_transform.world_transform.position = neo.Float3(
                target_position_x, target_cup_height, z_offset
            )
            world.set_transform(target_entity, target_transform)
            target_body = neo.RigidBodyComponent()
            target_body.body_type = neo.RigidBodyType.Static
            target_body.inverse_mass = 0.0
            target_body.simulated = True
            world.set_rigid_body(target_entity, target_body)
            if not world.replace_colliders(
                target_entity, _make_container_colliders(cup_inner_half, wall_thickness)
            ):
                raise RuntimeError(f"Failed to author target container for env {env_index}.")
            if self.enable_rgb_observation:
                self._set_container_renderer(
                    world, target_entity, env_index, "Target", neo.Float3(0.26, 0.68, 0.40), 300 + env_index
                )

            fluid_entity = world.create_entity(env_index)
            fluid_transform = neo.TransformComponent()
            source_floor_y = source_cup_height - cup_inner_half.y
            fluid_position = neo.Float3(
                source_position_x,
                source_floor_y + fluid_size.y * 0.5,
                z_offset,
            )
            fluid_transform.world_transform.position = fluid_position
            world.set_transform(fluid_entity, fluid_transform)
            fluid = neo.FluidComponent()
            fluid.source.kind = neo.FluidSourceKind.RegularGrid
            fluid.source.regular_grid.size = fluid_size
            fluid.source.regular_grid.target_particle_spacing = fluid_spacing
            fluid.particle_radius = self.FLUID_PARTICLE_RADIUS
            fluid.material = neo.FluidMaterialDesc()
            fluid.material.contact = neo.ParticleContactMaterialDesc()
            fluid.material.contact.friction = 0.04
            fluid.material.contact.static_friction = 0.06
            fluid.material.contact.restitution = 0.0
            fluid.material.contact.damping = 0.2
            fluid.material.viscosity = 1.5
            fluid.material.cohesion = 0.8
            fluid.material.gravity_scale = 0.75
            fluid.material.cfl_coefficient = 1.0
            fluid.material.vorticity_confinement = 0.25
            fluid.material.surface_tension = 1.5
            particle_diameter = 2.0 * fluid.particle_radius
            fluid.particle_mass = particle_diameter * particle_diameter * particle_diameter * 1000.0
            fluid.simulated = True
            fluid.visual_color = neo.Float4(0.16, 0.56, 0.96, 0.4)
            if not world.set_fluid(fluid_entity, fluid):
                raise RuntimeError(f"Failed to author fluid body for env {env_index}.")
            self._fluid_specs.append(
                (
                    fluid_entity,
                    fluid_position,
                    fluid_size,
                    fluid_spacing,
                )
            )
            self._target_bounds_min.append(
                (
                    target_position_x - cup_inner_half.x,
                    target_cup_height - cup_inner_half.y + wall_thickness * 0.5,
                    z_offset - cup_inner_half.z,
                    0.0,
                )
            )
            self._target_bounds_max.append(
                (
                    target_position_x + cup_inner_half.x,
                    target_cup_height + cup_inner_half.y,
                    z_offset + cup_inner_half.z,
                    0.0,
                )
            )
            if self.enable_rgb_observation:
                fluid_visuals = neo.EnvironmentFluidDesc()
                fluid_visuals.smoothness = 0.95
                fluid_visuals.specular = neo.Float3(0.38, 0.44, 0.50)
                fluid_visuals.fresnel = 0.84
                fluid_visuals.depth_edge_threshold = 0.18
                fluid_visuals.filter_radius_pixels = 4.0
                fluid_visuals.filter_world_radius = 0.18
                fluid_visuals.filter_depth_threshold = 0.11
                fluid_visuals.enable_background_refraction = True
                fluid_visuals.refraction_ior = 1.33
                fluid_visuals.refraction_view_thickness = 0.40
                world.set_environment_fluid(env_index, fluid_visuals)
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

    def _set_container_renderer(
        self,
        world: neo.World,
        entity: int,
        env_index: int,
        name_prefix: str,
        base_color: neo.Float3,
        segmentation_id: int,
    ) -> None:
        material = self._make_material(
            f"FluidPouring.{name_prefix}Material.{env_index}",
            base_color,
            0.58,
        )
        renderer = neo.MeshRendererComponent()
        renderer.mesh = self._rgb_cup_mesh
        renderer.material = material
        renderer.segmentation_id = segmentation_id
        renderer.visible = True
        world.set_mesh_renderer(entity, renderer)

    def _author_rgb_camera(self, world: neo.World, env_index: int, z_offset: float) -> None:
        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.35, -1.0, 0.25)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 7.5
        light.casts_shadows = False
        world.set_directional_light(light_entity, light)

        camera_entity = world.create_entity(env_index)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(0.0, 16.0, z_offset - 14.0)
        tilt = neo.Quaternion()
        tilt_angle = math.radians(45.0)
        tilt.x = math.sin(tilt_angle * 0.5)
        tilt.y = 0.0
        tilt.z = 0.0
        tilt.w = math.cos(tilt_angle * 0.5)
        camera_transform.world_transform.rotation = tilt
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 34.0
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

    def _build_source_body_indices(self) -> list[int]:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._rigid_mapping.rigid_body_entity_ids)
        }
        return [slot_by_entity[entity] for entity in self._source_entities]

    def _regular_grid_positions(
        self, center: neo.Float3, size: neo.Float3, spacing: float
    ) -> list[tuple[float, float, float, float]]:
        nx = max(1, int(math.floor(size.x / spacing + 0.5)))
        ny = max(1, int(math.floor(size.y / spacing + 0.5)))
        nz = max(1, int(math.floor(size.z / spacing + 0.5)))
        positions: list[tuple[float, float, float, float]] = []
        for z_index in range(nz):
            for y_index in range(ny):
                for x_index in range(nx):
                    x = center.x - size.x * 0.5 + (x_index + 0.5) * spacing
                    y = center.y - size.y * 0.5 + (y_index + 0.5) * spacing
                    z = center.z - size.z * 0.5 + (z_index + 0.5) * spacing
                    positions.append((x, y, z, 0.0))
        return positions

    def _build_reset_positions(self) -> list[tuple[float, float, float, float]]:
        slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.fluid_entity_ids)
        }
        reset_positions = [
            (0.0, 0.0, 0.0, 0.0) for _ in range(self._particle_mapping.particle_count)
        ]
        for entity, position, size, spacing in self._fluid_specs:
            slot = slot_by_entity[entity]
            particle_offset = self._particle_mapping.fluid_particle_offsets[slot]
            particle_count = self._particle_mapping.fluid_particle_counts[slot]
            positions = self._regular_grid_positions(position, size, spacing)
            if particle_count != len(positions):
                raise RuntimeError("Prepared fluid particle count did not match authored grid.")
            for local_index, reset_position in enumerate(positions):
                reset_positions[particle_offset + local_index] = reset_position
        return reset_positions

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.Actions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.observation_buffer, observation_flat = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.Terminated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.Truncated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.EpisodeSteps", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.ResetMask", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_offsets_buffer, self.reset_offsets_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.ResetOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.env_particle_offsets_buffer, self.env_particle_offsets_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.ParticleOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.env_particle_counts_buffer, self.env_particle_counts_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.ParticleCounts", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.source_body_indices_buffer, self.source_body_indices_tensor = self._register_shared_buffer(
            self.runtime, "FluidPouring.BodyIndices", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.source_base_positions_buffer, self.source_base_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.BasePositions",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.source_target_state_buffer, self.source_target_state_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.TargetState",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, 2],
        )
        self.target_bounds_min_buffer, self.target_bounds_min_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.TargetBoundsMin",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.target_bounds_max_buffer, self.target_bounds_max_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.TargetBoundsMax",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "FluidPouring.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
                self.runtime,
                "FluidPouring.RgbObservation",
                self.env_count * self.image_width * self.image_height,
                neo.SharedBufferTensorDTypeCode.Float,
                element_stride_bytes=16,
                shape=[self.env_count, self.image_height, self.image_width, 4],
            )

    def _populate_lookup_buffers(self) -> None:
        slot_by_entity = {
            entity_id: slot for slot, entity_id in enumerate(self._particle_mapping.fluid_entity_ids)
        }
        device = self.action_tensor.device
        self.env_particle_offsets_tensor.copy_(
            torch.tensor(
                [
                    self._particle_mapping.fluid_particle_offsets[slot_by_entity[entity]]
                    for entity, _, _, _ in self._fluid_specs
                ],
                device=device,
                dtype=self.env_particle_offsets_tensor.dtype,
            )
        )
        self.env_particle_counts_tensor.copy_(
            torch.tensor(
                [
                    self._particle_mapping.fluid_particle_counts[slot_by_entity[entity]]
                    for entity, _, _, _ in self._fluid_specs
                ],
                device=device,
                dtype=self.env_particle_counts_tensor.dtype,
            )
        )
        self.source_body_indices_tensor.copy_(
            torch.tensor(
                self._source_body_indices,
                device=device,
                dtype=self.source_body_indices_tensor.dtype,
            )
        )
        self.source_base_positions_tensor.copy_(
            torch.tensor(
                self._source_base_positions,
                device=device,
                dtype=self.source_base_positions_tensor.dtype,
            )
        )
        self.source_target_state_tensor.zero_()
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
        self.reset_positions_tensor.copy_(
            torch.tensor(
                self._reset_positions,
                device=device,
                dtype=self.reset_positions_tensor.dtype,
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
        self._sync_from_cuda(
            self.runtime,
            [
                self.env_particle_offsets_buffer,
                self.env_particle_counts_buffer,
                self.source_body_indices_buffer,
                self.source_base_positions_buffer,
                self.source_target_state_buffer,
                self.target_bounds_min_buffer,
                self.target_bounds_max_buffer,
                self.reset_positions_buffer,
                self.action_buffer,
                self.observation_buffer,
                self.reward_buffer,
                self.terminated_buffer,
                self.truncated_buffer,
                self.episode_steps_buffer,
                self.reset_mask_buffer,
                self.reset_offsets_buffer,
            ],
        )

    def _create_custom_passes(self) -> None:
        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "FluidPouring.PrePhysicsControl"
        pre_desc.shader_source = _FLUID_POURING_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        pre_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(7)]
        specs = [
            ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_SourceBodyIndices", self.source_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_SourceBasePositions", self.source_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_SourceTargetState", self.source_target_state_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
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
        pre_desc.constant_buffer_variable_name = "FluidPouringPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.position_action_scale,
                self.tilt_action_scale,
                self.source_move_range_x,
                self.max_tilt_radians,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "FluidPouring.PostPhysicsObservations"
        post_desc.shader_source = _FLUID_POURING_POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        post_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(13)]
        specs = [
            ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_SourceBodyIndices", self.source_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_TargetBoundsMin", self.target_bounds_min_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_TargetBoundsMax", self.target_bounds_max_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidPositions", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
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
        post_desc.constant_buffer_variable_name = "FluidPouringPostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 16
        post_desc.constant_data = list(
            struct.pack("<fIff", self.success_fraction, self.max_episode_steps, 0.5, 0.05)
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
        reset_particles_desc.debug_name = "FluidPouring.ResetParticles"
        reset_particles_desc.shader_source = _FLUID_POURING_RESET_PARTICLES_SHADER
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
        reset_rigid_desc.debug_name = "FluidPouring.ResetRigid"
        reset_rigid_desc.shader_source = _FLUID_POURING_RESET_RIGID_SHADER
        reset_rigid_desc.thread_group_size_x = 64
        bind(
            reset_rigid_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_SourceBodyIndices", self.source_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_SourceBasePositions", self.source_base_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
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
        reset_outputs_desc.debug_name = "FluidPouring.ResetOutputs"
        reset_outputs_desc.shader_source = _FLUID_POURING_RESET_OUTPUTS_SHADER
        reset_outputs_desc.thread_group_size_x = 64
        bind(
            reset_outputs_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_SourceBodyIndices", self.source_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
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
        reset_outputs_desc.constant_buffer_variable_name = "FluidPouringResetOutputsConstants"
        reset_outputs_desc.constant_buffer_size_bytes = 16
        reset_outputs_desc.constant_data = list(struct.pack("<4f", 0.5, 0.05, 0.0, 0.0))
        reset_outputs_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_outputs_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_outputs_pass = self._register_custom_pass(self.runtime, reset_outputs_desc)

        if self.enable_rgb_observation:
            render_desc = neo.CustomComputePassDesc()
            render_desc.debug_name = "FluidPouring.RgbObservation"
            render_desc.shader_source = _FLUID_POURING_RGB_SHADER
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
        self.source_target_state_tensor.index_fill_(0, env_indices, 0.0)
        sampled_offsets = torch.empty(
            env_indices.numel(),
            device=self.reset_offsets_tensor.device,
            dtype=self.reset_offsets_tensor.dtype,
        )
        sampled_offsets.uniform_(-self.reset_position_range, self.reset_position_range)
        self.reset_offsets_tensor.index_copy_(0, env_indices, sampled_offsets)
        self._sync_from_cuda(
            self.runtime,
            [self.reset_mask_buffer, self.reset_offsets_buffer, self.source_target_state_buffer],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_particles_pass):
            raise RuntimeError("Failed to execute fluid pouring particle reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_rigid_pass):
            raise RuntimeError("Failed to execute fluid pouring rigid reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_outputs_pass):
            raise RuntimeError("Failed to execute fluid pouring output reset pass.")
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=False)
        self.reset_mask_tensor.zero_()
        self._sync_from_cuda(self.runtime, [self.reset_mask_buffer])
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
            raise RuntimeError("Failed to execute fluid pouring pre-physics pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Fluid pouring physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute fluid pouring post-physics pass.")
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
            raise RuntimeError("RGB observations were not enabled for this fluid env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute fluid pouring RGB observation pass.")
        if not self.runtime.sync_shared_buffer_to_cuda(self.rgb_observation_buffer):
            raise RuntimeError("Failed to synchronize fluid pouring RGB observation buffer to CUDA.")
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor
