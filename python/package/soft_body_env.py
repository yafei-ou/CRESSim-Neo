from __future__ import annotations

import struct

from . import _cressim_neo as neo
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError("cressim_neo.soft_body_env requires PyTorch to be installed.") from exc


_SOFT_BODY_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer SoftBodyPrePhysicsConstants
{
    float actionScale;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(float, g_Actions);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);

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

    const uint particleOffset = CRESSIM_SB_LOAD(g_EnvParticleOffsets, envIndex);
    const uint particleCount = CRESSIM_SB_LOAD(g_EnvParticleCounts, envIndex);
    const float targetVelocityX = CRESSIM_SB_LOAD(g_Actions, envIndex) * actionScale;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        float4 velocity = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex);
        velocity.x = targetVelocityX;
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, velocity);
    }
}
"""


_SOFT_BODY_POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer SoftBodyPostPhysicsConstants
{
    float terminatePosition;
    uint maxEpisodeSteps;
    float rewardScale;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
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

    float centroidX = 0.0f;
    float averageVelocityX = 0.0f;
    float minX = 3.402823466e+38f;
    float maxX = -3.402823466e+38f;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float x = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).x;
        const float vx = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex).x;
        centroidX += x;
        averageVelocityX += vx;
        minX = min(minX, x);
        maxX = max(maxX, x);
    }

    const float invCount = 1.0f / float(particleCount);
    centroidX *= invCount;
    averageVelocityX *= invCount;
    const float reward = max(0.0f, 1.0f - abs(centroidX) * rewardScale);
    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint terminated = abs(centroidX) > terminatePosition ? 1u : 0u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const uint obsBase = envIndex * 4u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, averageVelocityX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, minX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, maxX);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_SOFT_BODY_RESET_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

cbuffer SoftBodyResetConstants
{
    float rewardScale;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(float, g_ResetOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvParticleCounts);
CRESSIM_STRUCTURED_BUFFER(float4, g_ResetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
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
    const float resetOffset = CRESSIM_SB_LOAD(g_ResetOffsets, envIndex);
    if (particleCount == 0u)
    {
        return;
    }

    float centroidX = 0.0f;
    float minX = 3.402823466e+38f;
    float maxX = -3.402823466e+38f;
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
        centroidX += positionInvMass.x;
        minX = min(minX, positionInvMass.x);
        maxX = max(maxX, positionInvMass.x);
    }

    centroidX /= float(particleCount);
    const uint obsBase = envIndex * 4u;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, centroidX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, minX);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, maxX);
    CRESSIM_SB_STORE(g_Rewards, envIndex, max(0.0f, 1.0f - abs(centroidX) * rewardScale));
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);
}
"""


class SoftBodyTorchVectorEnv(TorchStagedVectorEnvBase):
    OBSERVATION_DIM = 4

    def __init__(
        self,
        env_count: int = 32,
        max_episode_steps: int = 240,
        action_scale: float = 1.5,
        terminate_position: float = 1.25,
        reset_position_range: float = 0.25,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.terminate_position = terminate_position
        self.reset_position_range = reset_position_range

        config = neo.RuntimeConfig()
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.scene_layout.env_count = env_count

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize soft-body runtime.")

        self._soft_entities = self._author_scene(self.runtime.world())
        self.runtime.prepare()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._reset_positions = self._build_reset_positions(self.runtime.world())
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared soft-body world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self.reset()

    def _author_scene(self, world: neo.World) -> list[int]:
        entities: list[int] = []
        for env_index in range(self.env_count):
            entity = world.create_entity(env_index)
            transform = neo.TransformComponent()
            transform.world_transform.position = neo.Float3(0.0, 0.0, float(env_index) * 2.5)
            world.set_transform(entity, transform)

            soft = neo.SoftBodyComponent()
            soft.source.kind = neo.SoftBodySourceKind.RegularGrid
            soft.source.regular_grid.size = neo.Float3(0.6, 0.6, 0.6)
            soft.source.regular_grid.target_particle_spacing = 0.2
            soft.particle_mass = 1.0
            soft.particle_radius = 0.09
            soft.edge_compliance = 5.0e-5
            soft.volume_compliance = 2.0e-4
            soft.simulated = True
            soft.self_collision_enabled = False
            if not world.set_soft_body(entity, soft):
                raise RuntimeError(f"Failed to author soft body for env {env_index}.")
            entities.append(entity)
        return entities

    def _build_reset_positions(self, world: neo.World) -> list[tuple[float, float, float, float]]:
        slot_by_entity = {
            entity_id: slot for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        reset_positions = [(0.0, 0.0, 0.0, 0.0) for _ in range(self._particle_mapping.particle_count)]
        for entity in self._soft_entities:
            authoring_particles = world.try_get_soft_body_authoring_particles(entity)
            if authoring_particles is None:
                raise RuntimeError(f"Authoring particles were unavailable for soft body {entity}.")
            slot = slot_by_entity[entity]
            particle_offset = self._particle_mapping.soft_body_particle_offsets[slot]
            particle_count = self._particle_mapping.soft_body_particle_counts[slot]
            if particle_count != authoring_particles.particle_count:
                raise RuntimeError("Prepared soft-body particle count did not match authoring data.")
            for local_index, position in enumerate(authoring_particles.rest_positions):
                reset_positions[particle_offset + local_index] = (position.x, position.y, position.z, 0.0)
        return reset_positions

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.Actions", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.observation_buffer, observation_flat = self._register_shared_buffer(
            self.runtime,
            "SoftBodyEnv.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.Terminated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.Truncated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.EpisodeSteps", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.ResetMask", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_offsets_buffer, self.reset_offsets_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.ResetOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.env_particle_offsets_buffer, self.env_particle_offsets_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.ParticleOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.env_particle_counts_buffer, self.env_particle_counts_tensor = self._register_shared_buffer(
            self.runtime, "SoftBodyEnv.ParticleCounts", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "SoftBodyEnv.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )

    def _populate_lookup_buffers(self) -> None:
        slot_by_entity = {
            entity_id: slot for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
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
        pre_desc.debug_name = "SoftBodyEnv.PrePhysicsControl"
        pre_desc.shader_source = _SOFT_BODY_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        pre_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(4)]
        specs = [
            ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(pre_desc.resource_bindings, specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        pre_desc.constant_buffer_variable_name = "SoftBodyPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(struct.pack("<4f", self.action_scale, 0.0, 0.0, 0.0))
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)
        self.runtime.update_custom_compute_pass_constants(
            self._pre_pass, struct.pack("<4f", self.action_scale, 0.0, 0.0, 0.0)
        )

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "SoftBodyEnv.PostPhysicsObservations"
        post_desc.shader_source = _SOFT_BODY_POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        post_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(9)]
        specs = [
            ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadOnly),
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
        post_desc.constant_buffer_variable_name = "SoftBodyPostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 16
        post_desc.constant_data = list(
            struct.pack("<fIff", self.terminate_position, self.max_episode_steps, 1.0, 0.0)
        )
        post_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._post_pass = self._register_custom_pass(self.runtime, post_desc)
        self.runtime.update_custom_compute_pass_constants(
            self._post_pass,
            struct.pack("<fIff", self.terminate_position, self.max_episode_steps, 1.0, 0.0),
        )

        reset_desc = neo.CustomComputePassDesc()
        reset_desc.debug_name = "SoftBodyEnv.Reset"
        reset_desc.shader_source = _SOFT_BODY_RESET_SHADER
        reset_desc.thread_group_size_x = 64
        reset_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(13)]
        specs = [
            ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ResetOffsets", self.reset_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleOffsets", self.env_particle_offsets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_EnvParticleCounts", self.env_particle_counts_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ResetPositions", self.reset_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
            ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_ParticlePreviousPositions", None, "particle.previous_positions", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
        ]
        for binding, (name, handle, key, access) in zip(reset_desc.resource_bindings, specs):
            binding.shader_variable_name = name
            binding.access = access
            if handle is not None:
                binding.shared_buffer_handle = handle
            else:
                binding.resource_key = key
        reset_desc.constant_buffer_variable_name = "SoftBodyResetConstants"
        reset_desc.constant_buffer_size_bytes = 16
        reset_desc.constant_data = list(struct.pack("<4f", 1.0, 0.0, 0.0, 0.0))
        reset_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_pass = self._register_custom_pass(self.runtime, reset_desc)
        self.runtime.update_custom_compute_pass_constants(
            self._reset_pass, struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        )

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
            env_indices = torch.arange(self.env_count, device=self.action_tensor.device, dtype=torch.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(list(env_ids), device=self.action_tensor.device, dtype=torch.int64)
        if env_indices.numel() == 0:
            return self.observation_tensor

        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        self.reset_offsets_tensor.zero_()
        sampled_offsets = torch.empty(
            env_indices.numel(),
            device=self.reset_offsets_tensor.device,
            dtype=self.reset_offsets_tensor.dtype,
        )
        sampled_offsets.uniform_(-self.reset_position_range, self.reset_position_range)
        self.reset_offsets_tensor.index_copy_(0, env_indices, sampled_offsets)
        self._sync_from_cuda(self.runtime, [self.reset_mask_buffer, self.reset_offsets_buffer])
        if not self.runtime.execute_custom_compute_pass(self._reset_pass):
            raise RuntimeError("Failed to execute soft-body reset pass.")
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=False)
        self.reset_mask_tensor.zero_()
        self._sync_from_cuda(self.runtime, [self.reset_mask_buffer])
        return self.observation_tensor

    def step(
        self, action_tensor: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if list(action_tensor.shape) != [self.env_count]:
            raise ValueError(f"Expected action tensor shape [{self.env_count}], got {list(action_tensor.shape)}.")
        self.action_tensor.copy_(action_tensor.to(device=self.action_tensor.device, dtype=self.action_tensor.dtype))
        self._sync_from_cuda(self.runtime, [self.action_buffer])
        if not self.runtime.execute_custom_compute_pass(self._pre_pass):
            raise RuntimeError("Failed to execute soft-body pre-physics pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("Soft-body physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute soft-body post-physics pass.")
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
