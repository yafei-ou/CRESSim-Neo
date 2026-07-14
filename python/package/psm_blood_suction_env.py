from __future__ import annotations

import math
import struct
from pathlib import Path

from . import _cressim_neo as neo
from .psm_builder import (
    PsmAuthoringConfig,
    author_psm_scene,
    get_psm_default_runtime_config,
    set_psm_joint_targets,
)
from .torch_env import TorchStagedVectorEnvBase

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "cressim_neo.psm_blood_suction_env requires PyTorch to be installed."
    ) from exc


_PSM_BLOOD_SUCTION_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmBloodSuctionPrePhysicsConstants
{
    float rotationalActionScale;
    float insertionActionScale;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(float, g_Actions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_CurrentJointTargets);
CRESSIM_STRUCTURED_BUFFER(float2, g_JointLimits);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_RW_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EnvStepU32);

static const uint kCommandJointCount = 5u;
static const uint kEnvMetadataU32Stride = 12u;
static const uint kEnvStepStride = 4u;
static const uint kNearestDistanceSqInitBits = 0x7F7FFFFFu;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataU32Stride + offset);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_EnvMetadataU32.GetDimensions(envCount, stride);
    envCount /= kEnvMetadataU32Stride;
    if (envIndex >= envCount)
    {
        return;
    }

    const uint actionBase = envIndex * kCommandJointCount;
    float commandTargets[kCommandJointCount];
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        commandTargets[i] = CRESSIM_SB_LOAD(g_CurrentJointTargets, actionBase + i);
    }

    commandTargets[0u] += clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 0u), -1.0f, 1.0f) * rotationalActionScale;
    commandTargets[1u] += clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 1u), -1.0f, 1.0f) * rotationalActionScale;
    commandTargets[2u] += clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 2u), -1.0f, 1.0f) * insertionActionScale;
    commandTargets[3u] += clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 3u), -1.0f, 1.0f) * rotationalActionScale;
    commandTargets[4u] += clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 4u), -1.0f, 1.0f) * rotationalActionScale;
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        const float2 limits = CRESSIM_SB_LOAD(g_JointLimits, actionBase + i);
        commandTargets[i] = clamp(commandTargets[i], limits.x, limits.y);
        CRESSIM_SB_STORE(g_CurrentJointTargets, actionBase + i, commandTargets[i]);
    }

    const uint hingeSlot0 = envMetaU32(envIndex, 0u);
    const uint hingeSlot1 = envMetaU32(envIndex, 1u);
    const uint sliderSlot = envMetaU32(envIndex, 2u);
    const uint hingeSlot2 = envMetaU32(envIndex, 3u);
    const uint hingeSlot3 = envMetaU32(envIndex, 4u);

    if (hingeSlot0 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot0);
        joint.driveTargetParams.x = commandTargets[0u];
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot0, joint);
    }
    if (hingeSlot1 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot1);
        joint.driveTargetParams.x = commandTargets[1u];
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot1, joint);
    }
    if (sliderSlot != kInvalidIndex)
    {
        GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, sliderSlot);
        joint.driveTargetParams.x = commandTargets[2u];
        CRESSIM_SB_STORE(g_SliderJoints, sliderSlot, joint);
    }
    if (hingeSlot2 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot2);
        joint.driveTargetParams.x = commandTargets[3u];
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot2, joint);
    }
    if (hingeSlot3 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot3);
        joint.driveTargetParams.x = commandTargets[4u];
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot3, joint);
    }
    const uint envStepBase = envIndex * kEnvStepStride;
    CRESSIM_SB_STORE(g_EnvStepU32, envStepBase + 0u, 0u);
    CRESSIM_SB_STORE(g_EnvStepU32, envStepBase + 1u, 0u);
    CRESSIM_SB_STORE(g_EnvStepU32, envStepBase + 2u, kNearestDistanceSqInitBits);
    CRESSIM_SB_STORE(g_EnvStepU32, envStepBase + 3u, kInvalidIndex);
}
"""


_PSM_BLOOD_SUCTION_PRE_PARTICLE_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"

cbuffer PsmBloodSuctionPreParticleConstants
{
    float suctionRadius;
    float removalRadius;
    float suctionVelocityScale;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_FluidParticleIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidParticleEnvIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_FluidActiveMask);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EnvStepU32);

static const uint kEnvMetadataU32Stride = 12u;
static const uint kEnvStepStride = 4u;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataU32Stride + offset);
}

float4 envMetaF32(uint envIndex, uint slot)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex * 2u + slot);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint fluidListIndex = dispatchThreadID.x;
    uint fluidParticleCount = 0u;
    uint stride = 0u;
    g_FluidParticleIndices.GetDimensions(fluidParticleCount, stride);
    if (fluidListIndex >= fluidParticleCount)
    {
        return;
    }

    const uint particleIndex = CRESSIM_SB_LOAD(g_FluidParticleIndices, fluidListIndex);
    if (CRESSIM_SB_LOAD(g_FluidActiveMask, particleIndex) == 0u)
    {
        return;
    }

    const uint envIndex = CRESSIM_SB_LOAD(g_FluidParticleEnvIndices, fluidListIndex);
    const uint tooltipBodyIndex = envMetaU32(envIndex, 5u);
    const float4 rigidPosition = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = envMetaF32(envIndex, 0u).xyz;
    const float3 tooltipPosition =
        rigidPosition.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);

    float4 particlePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float3 delta = particlePosition.xyz - tooltipPosition;
    const float distanceSq = dot(delta, delta);
    const float distance = sqrt(max(distanceSq, 1.0e-12f));
    const uint envStepBase = envIndex * kEnvStepStride;

    if (distance <= removalRadius)
    {
        const float3 removalTeleport = envMetaF32(envIndex, 1u).xyz;
        particlePosition.xyz = removalTeleport;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, particlePosition);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex, float4(removalTeleport, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
        CRESSIM_SB_STORE(g_FluidActiveMask, particleIndex, 0u);
        InterlockedAdd(CRESSIM_SB_REF(g_EnvStepU32, envStepBase + 0u), 1u);
        return;
    }

    InterlockedAdd(CRESSIM_SB_REF(g_EnvStepU32, envStepBase + 1u), 1u);
    const float safeRadius = max(suctionRadius, 1.0e-4f);
    if (distance > safeRadius)
    {
        return;
    }

    const float falloff = max(0.0f, 1.0f - distance / safeRadius);
    const float3 direction = distance > 1.0e-5f ? (-delta / distance) : float3(0.0f, -1.0f, 0.0f);
    float4 velocity = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex);
    velocity.xyz += direction * (suctionVelocityScale * falloff);
    CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, velocity);
}
"""


_PSM_BLOOD_SUCTION_POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmBloodSuctionPostPhysicsConstants
{
    float removedRewardScale;
    float progressRewardScale;
    float stepPenalty;
    float completionBonus;
    float completionRemainingThreshold;
    uint maxEpisodeSteps;
    float observationDistanceLimit;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidActiveMask);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_STRUCTURED_BUFFER(GpuHingeJointRuntimeState, g_HingeJointRuntimeStates);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJointRuntimeState, g_SliderJointRuntimeStates);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EnvStats);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvStepU32);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

static const uint kCommandJointCount = 5u;
static const uint kObservationDim = 13u;
static const uint kEnvMetadataU32Stride = 12u;
static const uint kEnvStepStride = 4u;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataU32Stride + offset);
}

float4 envMetaF32(uint envIndex, uint slot)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex * 2u + slot);
}

float2 ReadHingeState(uint slot)
{
    if (slot == kInvalidIndex)
    {
        return float2(0.0f, 0.0f);
    }
    const float4 angleState = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, slot).angleState;
    return float2(angleState.y, angleState.z);
}

float2 ReadSliderState(uint slot)
{
    if (slot == kInvalidIndex)
    {
        return float2(0.0f, 0.0f);
    }
    const float4 state = CRESSIM_SB_LOAD(g_SliderJointRuntimeStates, slot).state;
    return float2(state.x, state.y);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_EnvStats.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint hingeSlot0 = envMetaU32(envIndex, 0u);
    const uint hingeSlot1 = envMetaU32(envIndex, 1u);
    const uint sliderSlot = envMetaU32(envIndex, 2u);
    const uint hingeSlot2 = envMetaU32(envIndex, 3u);
    const uint hingeSlot3 = envMetaU32(envIndex, 4u);

    const float2 hingeState0 = ReadHingeState(hingeSlot0);
    const float2 hingeState1 = ReadHingeState(hingeSlot1);
    const float2 sliderState = ReadSliderState(sliderSlot);
    const float2 hingeState2 = ReadHingeState(hingeSlot2);
    const float2 hingeState3 = ReadHingeState(hingeSlot3);

    const uint tooltipBodyIndex = envMetaU32(envIndex, 5u);
    const float4 rigidPosition = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = envMetaF32(envIndex, 0u).xyz;
    const float3 tooltipPosition =
        rigidPosition.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);

    const uint envStepBase = envIndex * kEnvStepStride;
    const uint nearestDistanceSqBits = CRESSIM_SB_LOAD(g_EnvStepU32, envStepBase + 2u);
    float currentNearest = observationDistanceLimit;
    if (CRESSIM_SB_LOAD(g_EnvStepU32, envStepBase + 3u) != kInvalidIndex)
    {
        currentNearest = sqrt(asfloat(nearestDistanceSqBits));
    }

    float4 stats = CRESSIM_SB_LOAD(g_EnvStats, envIndex);
    const float previousNearest = stats.x;
    const uint removedCount = CRESSIM_SB_LOAD(g_EnvStepU32, envStepBase + 0u);
    const uint remainingCount = CRESSIM_SB_LOAD(g_EnvStepU32, envStepBase + 1u);
    float progressReward = 0.0f;
    if (removedCount == 0u &&
        previousNearest < observationDistanceLimit &&
        currentNearest < observationDistanceLimit)
    {
        progressReward = progressRewardScale * (previousNearest - currentNearest);
    }

    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const uint terminated = 0u;
    float reward = float(removedCount) * removedRewardScale + progressReward;

    stats.x = currentNearest;
    stats.y = currentNearest;
    stats.z = float(removedCount);
    stats.w = float(remainingCount);
    CRESSIM_SB_STORE(g_EnvStats, envIndex, stats);

    const uint obsBase = envIndex * kObservationDim;
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, hingeState0.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, hingeState1.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, sliderState.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, hingeState2.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, hingeState3.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 5u, hingeState0.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 6u, hingeState1.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 7u, sliderState.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 8u, hingeState2.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 9u, hingeState3.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 10u, tooltipPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 11u, tooltipPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 12u, tooltipPosition.z);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_PSM_BLOOD_SUCTION_POST_PARTICLE_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_FluidParticleIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidParticleEnvIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidActiveMask);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EnvStepU32);

static const uint kEnvMetadataU32Stride = 12u;
static const uint kEnvStepStride = 4u;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataU32Stride + offset);
}

float4 envMetaF32(uint envIndex, uint slot)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex * 2u + slot);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint fluidListIndex = dispatchThreadID.x;
    uint fluidParticleCount = 0u;
    uint stride = 0u;
    g_FluidParticleIndices.GetDimensions(fluidParticleCount, stride);
    if (fluidListIndex >= fluidParticleCount)
    {
        return;
    }

    const uint particleIndex = CRESSIM_SB_LOAD(g_FluidParticleIndices, fluidListIndex);
    if (CRESSIM_SB_LOAD(g_FluidActiveMask, particleIndex) == 0u)
    {
        return;
    }

    const uint envIndex = CRESSIM_SB_LOAD(g_FluidParticleEnvIndices, fluidListIndex);
    const uint tooltipBodyIndex = envMetaU32(envIndex, 5u);
    const float4 rigidPosition = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = envMetaF32(envIndex, 0u).xyz;
    const float3 tooltipPosition =
        rigidPosition.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);
    const float3 particlePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const uint distanceSqBits = asuint(dot(particlePosition - tooltipPosition, particlePosition - tooltipPosition));
    const uint envStepBase = envIndex * kEnvStepStride;
    const uint nearestDistanceSlot = envStepBase + 2u;
    const uint nearestIndexSlot = envStepBase + 3u;
    uint compareBits = CRESSIM_SB_LOAD(g_EnvStepU32, nearestDistanceSlot);
    [allow_uav_condition]
    while (distanceSqBits < compareBits)
    {
        uint originalBits = 0u;
        InterlockedCompareExchange(
            CRESSIM_SB_REF(g_EnvStepU32, nearestDistanceSlot),
            compareBits,
            distanceSqBits,
            originalBits
        );
        if (originalBits == compareBits)
        {
            CRESSIM_SB_STORE(g_EnvStepU32, nearestIndexSlot, particleIndex);
            break;
        }
        compareBits = originalBits;
    }
}
"""


_PSM_BLOOD_SUCTION_RESET_PARTICLES_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_ResetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_FluidActiveMask);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * 12u + offset);
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

    const uint containerOffset = envMetaU32(envIndex, 6u);
    const uint containerCount = envMetaU32(envIndex, 7u);
    for (uint i = 0u; i < containerCount; ++i)
    {
        const uint particleIndex = containerOffset + i;
        float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_ResetPositions, particleIndex);
        positionInvMass.xyz = resetPosition.xyz;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex, float4(resetPosition.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
        CRESSIM_SB_STORE(g_FluidActiveMask, particleIndex, 0u);
    }

    const uint fluidOffset = envMetaU32(envIndex, 8u);
    const uint fluidCount = envMetaU32(envIndex, 9u);
    for (uint i = 0u; i < fluidCount; ++i)
    {
        const uint particleIndex = fluidOffset + i;
        float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_ResetPositions, particleIndex);
        positionInvMass.xyz = resetPosition.xyz;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex, float4(resetPosition.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
        CRESSIM_SB_STORE(g_FluidActiveMask, particleIndex, 1u);
    }
}
"""


_PSM_BLOOD_SUCTION_RESET_RIGID_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmBloodSuctionResetRigidConstants
{
    uint rigidBodyCountPerEnv;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidResetBodyIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidResetPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidResetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidAngularVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_CurrentJointTargets);
CRESSIM_STRUCTURED_BUFFER(float, g_ResetJointTargets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_RW_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);

static const uint kCommandJointCount = 5u;
static const uint kEnvMetadataU32Stride = 12u;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataU32Stride + offset);
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

    const uint rigidBase = envIndex * rigidBodyCountPerEnv;
    for (uint i = 0u; i < rigidBodyCountPerEnv; ++i)
    {
        const uint bodyIndex = CRESSIM_SB_LOAD(g_RigidResetBodyIndices, rigidBase + i);
        const float4 resetPosition = CRESSIM_SB_LOAD(g_RigidResetPositions, rigidBase + i);
        const float4 resetOrientation = CRESSIM_SB_LOAD(g_RigidResetOrientations, rigidBase + i);
        float4 rigidPosition = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, bodyIndex);
        rigidPosition.xyz = resetPosition.xyz;
        CRESSIM_SB_STORE(g_RigidPositionsInvMass, bodyIndex, rigidPosition);
        CRESSIM_SB_STORE(g_RigidOrientations, bodyIndex, resetOrientation);
        CRESSIM_SB_STORE(g_RigidLinearVelocities, bodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
        CRESSIM_SB_STORE(g_RigidAngularVelocities, bodyIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    const uint targetBase = envIndex * kCommandJointCount;
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        CRESSIM_SB_STORE(
            g_CurrentJointTargets,
            targetBase + i,
            CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + i)
        );
    }

    const uint hingeSlot0 = envMetaU32(envIndex, 0u);
    const uint hingeSlot1 = envMetaU32(envIndex, 1u);
    const uint sliderSlot = envMetaU32(envIndex, 2u);
    const uint hingeSlot2 = envMetaU32(envIndex, 3u);
    const uint hingeSlot3 = envMetaU32(envIndex, 4u);

    if (hingeSlot0 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot0);
        joint.driveTargetParams.x = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 0u);
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot0, joint);
    }
    if (hingeSlot1 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot1);
        joint.driveTargetParams.x = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 1u);
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot1, joint);
    }
    if (sliderSlot != kInvalidIndex)
    {
        GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, sliderSlot);
        joint.driveTargetParams.x = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 2u);
        CRESSIM_SB_STORE(g_SliderJoints, sliderSlot, joint);
    }
    if (hingeSlot2 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot2);
        joint.driveTargetParams.x = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 3u);
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot2, joint);
    }
    if (hingeSlot3 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot3);
        joint.driveTargetParams.x = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 4u);
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot3, joint);
    }
}
"""


_PSM_BLOOD_SUCTION_RESET_OUTPUTS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"

cbuffer PsmBloodSuctionResetOutputsConstants
{
    float observationDistanceLimit;
    float padding0;
    float padding1;
    float padding2;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(float, g_CurrentJointTargets);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidActiveMask);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EnvStats);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);

static const uint kCommandJointCount = 5u;
static const uint kObservationDim = 13u;

uint envMetaU32(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * 12u + offset);
}

float4 envMetaF32(uint envIndex, uint slot)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex * 2u + slot);
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

    const uint tooltipBodyIndex = envMetaU32(envIndex, 5u);
    const float4 rigidPosition = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = envMetaF32(envIndex, 0u).xyz;
    const float3 tooltipPosition =
        rigidPosition.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);

    const uint particleOffset = envMetaU32(envIndex, 8u);
    const uint particleCount = envMetaU32(envIndex, 9u);
    float bestDistanceSq = 1.0e30f;
    float3 bestDelta = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        if (CRESSIM_SB_LOAD(g_FluidActiveMask, particleIndex) == 0u)
        {
            continue;
        }
        const float3 particlePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
        const float3 delta = particlePosition - tooltipPosition;
        const float distanceSq = dot(delta, delta);
        if (distanceSq < bestDistanceSq)
        {
            bestDistanceSq = distanceSq;
            bestDelta = delta;
        }
    }

    const float nearestDistance =
        bestDistanceSq < 1.0e29f ? sqrt(bestDistanceSq) : observationDistanceLimit;
    const uint remainingCount = envMetaU32(envIndex, 9u);
    CRESSIM_SB_STORE(
        g_EnvStats,
        envIndex,
        float4(nearestDistance, nearestDistance, 0.0f, float(remainingCount))
    );
    CRESSIM_SB_STORE(g_Rewards, envIndex, 0.0f);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);

    const uint obsBase = envIndex * kObservationDim;
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        CRESSIM_SB_STORE(g_Observations, obsBase + i, CRESSIM_SB_LOAD(g_CurrentJointTargets, envIndex * kCommandJointCount + i));
    }
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        CRESSIM_SB_STORE(g_Observations, obsBase + 5u + i, 0.0f);
    }
    CRESSIM_SB_STORE(g_Observations, obsBase + 10u, tooltipPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 11u, tooltipPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 12u, tooltipPosition.z);
}
"""


_PSM_BLOOD_SUCTION_RGB_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"

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


_CONTAINER_OPENING_CENTER_X_LOCAL = -0.105369
_CONTAINER_OPENING_CENTER_Z_LOCAL = -0.174073
_CONTAINER_OPENING_TOP_Y_LOCAL = -0.01413111111111111
_CONTAINER_BOTTOM_Y_LOCAL = -0.569843

_SUCTION_IRRIGATOR_YAW_CAPSULE_TOTAL_LENGTH = 0.011958
_SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION = (
    0.0,
    0.5 * _SUCTION_IRRIGATOR_YAW_CAPSULE_TOTAL_LENGTH,
    0.0,
)


def _compute_regular_grid_axis(
    inner_half_extent: float,
    particle_radius: float,
    fill_fraction: float,
) -> tuple[int, float]:
    spacing = 2.0 * particle_radius
    max_count = max(
        1,
        int(math.floor((2.0 * max(0.0, inner_half_extent - particle_radius)) / spacing)) + 1,
    )
    count = max(1, min(max_count, int(math.floor(max_count * fill_fraction + 0.5))))
    return count, float(count) * spacing


def _compute_fluid_block_desc(
    block_half_extents: neo.Float3,
    particle_radius: float,
    fill_fraction_xy: float,
    fill_fraction_height: float,
) -> tuple[neo.Float3, float]:
    spacing = 2.0 * particle_radius
    _, size_x = _compute_regular_grid_axis(block_half_extents.x, particle_radius, fill_fraction_xy)
    _, size_z = _compute_regular_grid_axis(block_half_extents.z, particle_radius, fill_fraction_xy)
    _, size_y = _compute_regular_grid_axis(block_half_extents.y, particle_radius, fill_fraction_height)
    return neo.Float3(size_x, size_y, size_z), spacing


def _bind_resources(
    desc: neo.CustomComputePassDesc,
    specs: list[tuple[str, object | None, str, object]],
) -> None:
    desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(len(specs))]
    for binding, (name, handle, key, access) in zip(desc.resource_bindings, specs):
        binding.shader_variable_name = name
        binding.access = access
        if handle is not None:
            binding.shared_buffer_handle = handle
        else:
            binding.resource_key = key


def _find_repo_root(
    resolve_root: str | Path | None,
    required_relative_path: str,
) -> Path:
    search_roots: list[Path] = []
    if resolve_root is not None:
        search_roots.append(Path(resolve_root).expanduser().resolve())
    search_roots.extend(Path(__file__).resolve().parents)
    for root in search_roots:
        if (root / required_relative_path).exists():
            return root
    raise RuntimeError(
        f"Failed to locate {required_relative_path}. Pass resolve_root=... to the env constructor."
    )


def _container_entity_world_position(
    env_center_x: float,
    env_center_z: float,
    top_y: float,
    scene_scale: float,
) -> tuple[float, float, float]:
    return (
        env_center_x - _CONTAINER_OPENING_CENTER_X_LOCAL * scene_scale,
        top_y - _CONTAINER_OPENING_TOP_Y_LOCAL * scene_scale,
        env_center_z - _CONTAINER_OPENING_CENTER_Z_LOCAL * scene_scale,
    )


def _container_bottom_world_y(top_y: float, scene_scale: float) -> float:
    return top_y - _CONTAINER_OPENING_TOP_Y_LOCAL * scene_scale + _CONTAINER_BOTTOM_Y_LOCAL * scene_scale


def _look_rotation(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> neo.Quaternion:
    px, py, pz = position
    tx, ty, tz = target
    forward = [tx - px, ty - py, tz - pz]
    forward_len = math.sqrt(sum(v * v for v in forward))
    if forward_len <= 1.0e-8:
        return neo.Quaternion()
    forward = [v / forward_len for v in forward]
    up = [0.0, 1.0, 0.0]
    right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    ]
    right_len = math.sqrt(sum(v * v for v in right))
    if right_len <= 1.0e-8:
        right = [1.0, 0.0, 0.0]
    else:
        right = [v / right_len for v in right]
    corrected_up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ]
    m00, m01, m02 = right[0], corrected_up[0], forward[0]
    m10, m11, m12 = right[1], corrected_up[1], forward[1]
    m20, m21, m22 = right[2], corrected_up[2], forward[2]
    trace = m00 + m11 + m22
    q = neo.Quaternion()
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q.w = 0.25 * s
        q.x = (m21 - m12) / s
        q.y = (m02 - m20) / s
        q.z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q.w = (m21 - m12) / s
        q.x = 0.25 * s
        q.y = (m01 + m10) / s
        q.z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q.w = (m02 - m20) / s
        q.x = (m01 + m10) / s
        q.y = 0.25 * s
        q.z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q.w = (m10 - m01) / s
        q.x = (m02 + m20) / s
        q.y = (m12 + m21) / s
        q.z = 0.25 * s
    return q


def _load_obj_mesh(path: Path, debug_name: str, *, reverse_winding: bool = False) -> neo.MeshResourceDesc:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    vertices: list[neo.MeshVertex] = []
    indices: list[int] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0] == "v" and len(tokens) >= 4:
                positions.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif tokens[0] == "vn" and len(tokens) >= 4:
                normals.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif tokens[0] == "vt" and len(tokens) >= 3:
                texcoords.append((float(tokens[1]), float(tokens[2])))
            elif tokens[0] == "f" and len(tokens) >= 4:
                face_vertices = tokens[1:]
                for tri_idx in range(1, len(face_vertices) - 1):
                    tri = (face_vertices[0], face_vertices[tri_idx], face_vertices[tri_idx + 1])
                    if reverse_winding:
                        tri = (tri[0], tri[2], tri[1])
                    for face_token in tri:
                        parts = face_token.split("/")
                        pos_idx = int(parts[0]) - 1
                        tex_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                        normal_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None
                        vertex = neo.MeshVertex()
                        px, py, pz = positions[pos_idx]
                        vertex.position = neo.Float3(px, py, pz)
                        if normal_idx is not None and 0 <= normal_idx < len(normals):
                            nx, ny, nz = normals[normal_idx]
                            vertex.normal = neo.Float3(nx, ny, nz)
                        else:
                            vertex.normal = neo.Float3(0.0, 1.0, 0.0)
                        if tex_idx is not None and 0 <= tex_idx < len(texcoords):
                            u, v = texcoords[tex_idx]
                            vertex.tex_coord_u = u
                            vertex.tex_coord_v = 1.0 - v
                        indices.append(len(vertices))
                        vertices.append(vertex)

    mesh = neo.MeshResourceDesc()
    mesh.debug_name = debug_name
    mesh.vertices = vertices
    mesh.indices = indices
    return mesh


def _bottom_static_particle_indices(node_path: Path, band_height: float) -> list[int]:
    with node_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    node_count = int(lines[0].split()[0])
    entries = [lines[i + 1].split() for i in range(node_count)]
    y_values = [float(entry[2]) for entry in entries]
    min_y = min(y_values)
    threshold = min_y + band_height
    return [index for index, entry in enumerate(entries) if float(entry[2]) <= threshold]


class PsmBloodSuctionTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 5
    OBSERVATION_DIM = 13
    OBS_DIM = OBSERVATION_DIM
    DEFAULT_SCENE_SCALE = 2.5
    DEFAULT_PSM_SCALE = 10.0 * DEFAULT_SCENE_SCALE
    DEFAULT_PSM_MASS_SCALE = 0.005
    DEFAULT_PSM_INERTIA_SCALE = 0.005
    DEFAULT_PSM_INSERTION_RATIO = 0.08
    DEFAULT_PSM_TOOL_YAW_ZERO_POSE_Z_PER_SCALE = -0.4864
    DEFAULT_FLUID_PARTICLE_RADIUS = 0.09
    DEFAULT_FLUID_HORIZONTAL_FILL_FRACTION = 1.0
    DEFAULT_FLUID_HEIGHT_FILL_FRACTION = 1.0
    DEFAULT_CONTAINER_TOP_Y = 1.10
    DEFAULT_CONTAINER_STATIC_BAND_HEIGHT = 0.06
    DEFAULT_CONTAINER_PARTICLE_MASS = 0.12
    DEFAULT_CONTAINER_PARTICLE_RADIUS = 0.30
    DEFAULT_CONTAINER_EDGE_COMPLIANCE = 0.0
    DEFAULT_CONTAINER_VOLUME_COMPLIANCE = 8.0e-4
    DEFAULT_CONTAINER_CONTACT_FRICTION = 0.55
    DEFAULT_CONTAINER_CONTACT_STATIC_FRICTION = 0.75
    DEFAULT_CONTAINER_CONTACT_RESTITUTION = 0.05
    DEFAULT_CONTAINER_CONTACT_DAMPING = 0.8
    DEFAULT_GROUND_CLEARANCE = 0.01
    DEFAULT_FLUID_DROP_GAP_Y = 0.10
    DEFAULT_FLUID_DROP_EXTRA_Z = 0.0
    DEFAULT_PSM_RESET_HEIGHT_ABOVE_CONTAINER = 0.65
    ACTIVE_ARM_JOINT_INDICES = (0, 1, 2, 4, 5)

    def __init__(
        self,
        env_count: int = 16,
        max_episode_steps: int = 240,
        enable_rgb_observation: bool = False,
        enable_visualization_camera: bool = False,
        return_combined_observation: bool = False,
        image_width: int = 160,
        image_height: int = 160,
        visualization_image_width: int | None = None,
        visualization_image_height: int | None = None,
        scene_scale: float = DEFAULT_SCENE_SCALE,
        psm_scale: float = DEFAULT_PSM_SCALE,
        psm_mass_scale: float = DEFAULT_PSM_MASS_SCALE,
        psm_inertia_scale: float = DEFAULT_PSM_INERTIA_SCALE,
        rotational_action_scale: float = 0.01,
        insertion_action_scale: float = 0.04,
        suction_radius: float = 1.00,
        removal_radius: float = 0.25,
        suction_velocity_scale: float = 0.40,
        removed_reward_scale: float = 0.03,
        progress_reward_scale: float = 1.5,
        step_penalty: float = 0.0,
        completion_bonus: float = 0.0,
        completion_remaining_threshold: int = 0,
        env_spacing: float | None = None,
        resolve_root=None,
        urdf_path=None,
        tool_type: str = "suction_irrigator",
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = int(max_episode_steps)
        self.enable_rgb_observation = enable_rgb_observation
        self.enable_visualization_camera = bool(enable_visualization_camera)
        self.return_combined_observation = bool(return_combined_observation)
        if self.return_combined_observation and not self.enable_rgb_observation:
            raise ValueError("return_combined_observation=True requires enable_rgb_observation=True.")
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.visualization_image_width = (
            int(visualization_image_width) if visualization_image_width is not None else self.image_width
        )
        self.visualization_image_height = (
            int(visualization_image_height) if visualization_image_height is not None else self.image_height
        )
        self.scene_scale = float(scene_scale)
        self.psm_scale = float(psm_scale)
        self.psm_mass_scale = float(psm_mass_scale)
        self.psm_inertia_scale = float(psm_inertia_scale)
        self.rotational_action_scale = float(rotational_action_scale)
        # Slider targets are authored in scaled world units already.
        self.insertion_action_scale = float(insertion_action_scale)
        self.suction_radius = float(suction_radius)
        self.removal_radius = float(removal_radius)
        self.suction_velocity_scale = float(suction_velocity_scale)
        self.removed_reward_scale = float(removed_reward_scale)
        self.progress_reward_scale = float(progress_reward_scale)
        self.step_penalty = float(step_penalty)
        self.completion_bonus = float(completion_bonus)
        self.completion_remaining_threshold = int(completion_remaining_threshold)
        self.env_spacing = (
            float(env_spacing)
            if env_spacing is not None
            else max(18.0 * self.scene_scale, 1.6 * self.psm_scale)
        )
        self.resolve_root = resolve_root
        self.urdf_path = urdf_path
        self.tool_type = str(tool_type)
        self.repo_root = _find_repo_root(resolve_root, "examples/models/container.node")

        self.container_top_y = self.DEFAULT_CONTAINER_TOP_Y
        self.fluid_particle_radius = self.DEFAULT_FLUID_PARTICLE_RADIUS
        self.fluid_horizontal_fill_fraction = self.DEFAULT_FLUID_HORIZONTAL_FILL_FRACTION
        self.fluid_height_fill_fraction = self.DEFAULT_FLUID_HEIGHT_FILL_FRACTION
        self.ground_half = neo.Float3(
            5.6 * self.scene_scale,
            0.08 * self.scene_scale,
            2.4 * self.scene_scale,
        )
        self.camera_position_local = (
            0.0,
            2.4 * self.scene_scale,
            -3.4 * self.scene_scale,
        )
        self.initial_insertion = self.DEFAULT_PSM_INSERTION_RATIO * self.psm_scale
        self.psm_world_offset = (
            0.0,
            self.container_top_y * self.scene_scale
            + self.DEFAULT_PSM_RESET_HEIGHT_ABOVE_CONTAINER * self.scene_scale,
            -self.DEFAULT_PSM_TOOL_YAW_ZERO_POSE_Z_PER_SCALE * self.psm_scale,
        )
        self.tooltip_local_anchor = (
            _SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[0] * self.psm_scale,
            _SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[1] * self.psm_scale,
            _SUCTION_IRRIGATOR_YAW_CAPSULE_LOCAL_POSITION[2] * self.psm_scale,
            0.0,
        )
        self.observation_distance_limit = max(4.0 * self.suction_radius, 1.0)

        config = get_psm_default_runtime_config(env_count)
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.physics_desc.substeps = 4
        config.physics_desc.default_iterations = 10
        config.physics_desc.rigid_joint_iterations = 50
        camera_count = (1 if enable_rgb_observation else 0) + (1 if self.enable_visualization_camera else 0)
        any_camera_enabled = camera_count > 0
        config.scene_layout.max_renderable_objects_per_env = 32 if any_camera_enabled else 24
        config.scene_layout.max_lights_per_env = 1 if any_camera_enabled else 0
        config.scene_layout.max_cameras_per_env = camera_count

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize PSM blood-suction runtime.")

        self._psm_build = None
        self._container_entities: list[int] = []
        self._fluid_specs: list[tuple[int, neo.Float3, neo.Float3, float]] = []
        self._tooltip_body_entities: list[int] = []
        self._env_scene_centers: list[tuple[float, float]] = []
        self._reset_joint_targets: list[list[float]] = []
        self._rigid_reset_body_indices: list[int] = []
        self._rigid_reset_positions: list[tuple[float, float, float, float]] = []
        self._rigid_reset_orientations: list[tuple[float, float, float, float]] = []

        if self.enable_rgb_observation:
            self._initialize_rgb_observation_resources()
        if self.enable_visualization_camera:
            self._initialize_visualization_render_resources()

        self._author_scene()
        self.runtime.prepare()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._rigid_mapping = self.runtime.get_prepared_rigid_layout_mapping()
        self._joint_mapping = self.runtime.get_prepared_joint_layout_mapping()
        self._resolve_runtime_slots()
        self._reset_positions = self._build_reset_positions(self.runtime.world())
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared PSM blood-suction world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self._end_frame(self.runtime, advance=False)

    def _initialize_rgb_observation_resources(self) -> None:
        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.image_width
        target_desc.height = self.image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA16Float
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "PsmBloodSuction.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create PSM blood-suction RGB render target.")

    def _initialize_visualization_render_resources(self) -> None:
        target_desc = neo.GpuRenderTargetDesc()
        target_desc.width = self.visualization_image_width
        target_desc.height = self.visualization_image_height
        target_desc.array_size = self.env_count
        target_desc.color = True
        target_desc.depth = True
        target_desc.color_format = neo.TextureFormat.RGBA16Float
        target_desc.layered_rendering = True
        target_desc.shader_readable = True
        target_desc.debug_name = "PsmBloodSuction.VisualizationRenderTarget"
        self._visualization_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._visualization_render_target):
            raise RuntimeError("Failed to create PSM blood-suction visualization render target.")

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

    def _author_scene(self) -> None:
        world = self.runtime.world()
        resources = self.runtime.resources()
        models_dir = self.repo_root / "examples" / "models"
        node_path = models_dir / "container.node"
        ele_path = models_dir / "container.ele"
        surface_path = models_dir / "container_surface.obj"

        self._ground_mesh = resources.register_mesh(
            neo.make_box_mesh(self.ground_half, "PsmBloodSuction.GroundMesh")
        )
        self._ground_material = self._make_material(
            "PsmBloodSuction.GroundMaterial",
            neo.Float3(0.62, 0.64, 0.68),
            0.92,
        )
        self._container_mesh = resources.register_mesh(
            _load_obj_mesh(surface_path, "PsmBloodSuction.ContainerMesh", reverse_winding=True)
        )
        self._container_material = self._make_material(
            "PsmBloodSuction.ContainerMaterial",
            neo.Float3(0.76, 0.47, 0.32),
            0.88,
        )
        self._container_static_particle_indices = _bottom_static_particle_indices(
            node_path,
            self.DEFAULT_CONTAINER_STATIC_BAND_HEIGHT * self.scene_scale,
        )
        self._container_node_path = node_path
        self._container_ele_path = ele_path

        self._psm_build = author_psm_scene(
            world,
            resources,
            PsmAuthoringConfig(
                resolve_root=self.resolve_root,
                urdf_path=self.urdf_path,
                tool_type=self.tool_type,
                env_count=self.env_count,
                add_ground=False,
                add_default_lighting=False,
                add_default_camera=False,
                env_spacing=self.env_spacing,
                global_scale=self.psm_scale,
            ),
        )

        fluid_block_half = neo.Float3(1.0, 0.45, 1.0)
        fluid_size, fluid_spacing = _compute_fluid_block_desc(
            fluid_block_half,
            self.fluid_particle_radius,
            self.fluid_horizontal_fill_fraction,
            self.fluid_height_fill_fraction,
        )

        translated_entities: set[int] = set()
        full_default_targets = [0.0, 0.0, self.initial_insertion, 0.0, 0.0, 0.0]
        active_default_targets = [full_default_targets[index] for index in self.ACTIVE_ARM_JOINT_INDICES]
        for env_index, instance in enumerate(self._psm_build.instances):
            self._reset_joint_targets.append(active_default_targets[:])
            set_psm_joint_targets(
                world,
                self._psm_build,
                full_default_targets,
                env_index=env_index,
            )
            base_transform = world.try_get_transform(instance.base_entity)
            if base_transform is None:
                raise RuntimeError("Failed to resolve authored PSM base transform.")
            env_center_x = float(base_transform.world_transform.position.x)
            env_center_z = float(base_transform.world_transform.position.z)
            self._env_scene_centers.append((env_center_x, env_center_z))

            for entity in instance.link_entities.values():
                if entity in translated_entities:
                    continue
                translated_entities.add(entity)
                transform = world.try_get_transform(entity)
                if transform is None:
                    continue
                transform.world_transform.position = neo.Float3(
                    transform.world_transform.position.x + self.psm_world_offset[0],
                    transform.world_transform.position.y + self.psm_world_offset[1],
                    transform.world_transform.position.z + self.psm_world_offset[2],
                )
                world.set_transform(entity, transform)
                rigid_body = world.try_get_rigid_body(entity)
                if rigid_body is None or rigid_body.body_type != neo.RigidBodyType.Dynamic:
                    continue
                if rigid_body.inverse_mass > 0.0:
                    mass = 1.0 / rigid_body.inverse_mass
                    scaled_mass = max(mass * self.psm_mass_scale, 1.0e-6)
                    rigid_body.inverse_mass = 1.0 / scaled_mass
                rigid_body.inverse_inertia_local = neo.Float3(
                    rigid_body.inverse_inertia_local.x / self.psm_inertia_scale
                    if rigid_body.inverse_inertia_local.x > 0.0
                    else 0.0,
                    rigid_body.inverse_inertia_local.y / self.psm_inertia_scale
                    if rigid_body.inverse_inertia_local.y > 0.0
                    else 0.0,
                    rigid_body.inverse_inertia_local.z / self.psm_inertia_scale
                    if rigid_body.inverse_inertia_local.z > 0.0
                    else 0.0,
                )
                world.set_rigid_body(entity, rigid_body)

            self._tooltip_body_entities.append(instance.link_entities["psm_tool_yaw_link"])
            self._author_env_support(world, env_index, env_center_x, env_center_z)
            self._author_container(world, env_index, env_center_x, env_center_z)
            self._author_fluid(
                world,
                env_index,
                env_center_x,
                env_center_z,
                fluid_size,
                fluid_spacing,
            )
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
            if self.enable_rgb_observation:
                self._author_rgb_camera(
                    world,
                    env_index,
                    env_center_x,
                    env_center_z,
                    render_target=self._rgb_render_target,
                    image_width=self.image_width,
                    image_height=self.image_height,
                )
            if self.enable_visualization_camera:
                self._author_rgb_camera(
                    world,
                    env_index,
                    env_center_x,
                    env_center_z,
                    render_target=self._visualization_render_target,
                    image_width=self.visualization_image_width,
                    image_height=self.visualization_image_height,
                )

    def _author_env_support(
        self,
        world: neo.World,
        env_index: int,
        env_center_x: float,
        env_center_z: float,
    ) -> None:
        ground_entity = world.create_entity(env_index)
        ground_transform = neo.TransformComponent()
        ground_y = (
            _container_bottom_world_y(self.container_top_y * self.scene_scale, self.scene_scale)
            - self.ground_half.y
            - self.DEFAULT_GROUND_CLEARANCE * self.scene_scale
        )
        ground_transform.world_transform.position = neo.Float3(env_center_x, ground_y, env_center_z)
        world.set_transform(ground_entity, ground_transform)
        ground_body = neo.RigidBodyComponent()
        ground_body.body_type = neo.RigidBodyType.Static
        ground_body.inverse_mass = 0.0
        ground_body.simulated = True
        world.set_rigid_body(ground_entity, ground_body)
        ground_collider = neo.ColliderComponent()
        ground_collider.shape_type = neo.ColliderShapeType.Box
        ground_collider.shape_params = neo.Float4(
            self.ground_half.x,
            self.ground_half.y,
            self.ground_half.z,
            0.0,
        )
        world.add_collider(ground_entity, ground_collider)
        renderer = neo.MeshRendererComponent()
        renderer.mesh = self._ground_mesh
        renderer.material = self._ground_material
        renderer.visible = True
        world.set_mesh_renderer(ground_entity, renderer)

    def _author_container(
        self,
        world: neo.World,
        env_index: int,
        env_center_x: float,
        env_center_z: float,
    ) -> None:
        soft_entity = world.create_entity(env_index)
        soft_transform = neo.TransformComponent()
        soft_transform.world_transform.position = neo.Float3(
            *_container_entity_world_position(
                env_center_x,
                env_center_z,
                self.container_top_y * self.scene_scale,
                self.scene_scale,
            )
        )
        soft_transform.world_transform.scale = neo.Float3(
            self.scene_scale,
            self.scene_scale,
            self.scene_scale,
        )
        world.set_transform(soft_entity, soft_transform)

        soft_body = neo.SoftBodyComponent()
        soft_body.source.kind = neo.SoftBodySourceKind.TetGenFiles
        soft_body.source.tet_gen.node_file = str(self._container_node_path)
        soft_body.source.tet_gen.ele_file = str(self._container_ele_path)
        soft_body.source.tet_gen.static_particle_indices = self._container_static_particle_indices
        soft_body.particle_mass = self.DEFAULT_CONTAINER_PARTICLE_MASS
        soft_body.particle_radius = self.DEFAULT_CONTAINER_PARTICLE_RADIUS
        soft_body.edge_compliance = self.DEFAULT_CONTAINER_EDGE_COMPLIANCE
        soft_body.volume_compliance = self.DEFAULT_CONTAINER_VOLUME_COMPLIANCE
        soft_body.self_collision_enabled = False
        soft_body.supports_suturing = False
        soft_body.simulated = True
        soft_body.collision_layer = 0x1
        soft_body.collision_mask = 0xFFFFFFFF
        soft_body.material.contact.friction = self.DEFAULT_CONTAINER_CONTACT_FRICTION
        soft_body.material.contact.static_friction = self.DEFAULT_CONTAINER_CONTACT_STATIC_FRICTION
        soft_body.material.contact.restitution = self.DEFAULT_CONTAINER_CONTACT_RESTITUTION
        soft_body.material.contact.damping = self.DEFAULT_CONTAINER_CONTACT_DAMPING
        if not world.set_soft_body(soft_entity, soft_body):
            raise RuntimeError(f"Failed to author TetGen soft body for env {env_index}.")

        renderer = neo.MeshRendererComponent()
        renderer.mesh = self._container_mesh
        renderer.material = self._container_material
        renderer.visible = True
        world.set_mesh_renderer(soft_entity, renderer)
        self._container_entities.append(soft_entity)

    def _author_fluid(
        self,
        world: neo.World,
        env_index: int,
        env_center_x: float,
        env_center_z: float,
        fluid_size: neo.Float3,
        fluid_spacing: float,
    ) -> None:
        fluid_entity = world.create_entity(env_index)
        fluid_transform = neo.TransformComponent()
        fluid_transform.world_transform.position = neo.Float3(
            env_center_x,
            self.container_top_y * self.scene_scale
            + self.DEFAULT_FLUID_DROP_GAP_Y * self.scene_scale
            + 0.5 * fluid_size.y,
            env_center_z + self.DEFAULT_FLUID_DROP_EXTRA_Z,
        )
        world.set_transform(fluid_entity, fluid_transform)
        fluid = neo.FluidComponent()
        fluid.source.kind = neo.FluidSourceKind.RegularGrid
        fluid.source.regular_grid.size = fluid_size
        fluid.source.regular_grid.target_particle_spacing = fluid_spacing
        fluid.particle_radius = self.fluid_particle_radius
        fluid.material = neo.FluidMaterialDesc()
        fluid.material.contact = neo.ParticleContactMaterialDesc()
        fluid.material.contact.friction = 0.04
        fluid.material.contact.static_friction = 0.06
        fluid.material.contact.restitution = 0.0
        fluid.material.contact.damping = 0.2
        fluid.material.viscosity = 1.5
        fluid.material.cohesion = 0.2
        fluid.material.gravity_scale = 0.75
        fluid.material.cfl_coefficient = 1.0
        fluid.material.vorticity_confinement = 0.25
        fluid.material.surface_tension = 1.5
        particle_diameter = 2.0 * fluid.particle_radius
        fluid.particle_mass = particle_diameter * particle_diameter * particle_diameter * 10.0
        fluid.simulated = True
        fluid.visual_color = neo.Float4(0.90, 0.16, 0.16, 0.80)
        if not world.set_fluid(fluid_entity, fluid):
            raise RuntimeError(f"Failed to author fluid body for env {env_index}.")
        self._fluid_specs.append(
            (
                fluid_entity,
                fluid_transform.world_transform.position,
                fluid_size,
                fluid_spacing,
            )
        )

    def _author_rgb_camera(
        self,
        world: neo.World,
        env_index: int,
        env_center_x: float,
        env_center_z: float,
        *,
        render_target: neo.GpuRenderTargetHandle,
        image_width: int,
        image_height: int,
    ) -> None:
        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.45, -1.0, 0.35)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 7.5
        light.casts_shadows = False
        world.set_directional_light(light_entity, light)

        camera_entity = world.create_entity(env_index)
        camera_position = (
            env_center_x + self.camera_position_local[0],
            self.camera_position_local[1],
            env_center_z + self.camera_position_local[2],
        )
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(*camera_position)
        camera_transform.world_transform.rotation = _look_rotation(
            camera_position,
            (
                env_center_x,
                self.container_top_y * self.scene_scale + 0.5 * self.scene_scale,
                env_center_z,
            ),
        )
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 46.0
        camera.output.mode = neo.RenderOutputMode.ExplicitSurface
        camera.output.binding = neo.GpuRenderTargetBinding()
        camera.output.binding.target = render_target
        camera.output.binding.first_layer = env_index
        camera.output.binding.layer_count = 1
        camera.output_width = image_width
        camera.output_height = image_height
        camera.clear_color = True
        camera.clear_depth = True
        camera.clear_color_value = neo.Float4(0.03, 0.04, 0.06, 1.0)
        world.set_camera(camera_entity, camera)

    def _regular_grid_positions(
        self,
        center: neo.Float3,
        size: neo.Float3,
        spacing: float,
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

    def _resolve_runtime_slots(self) -> None:
        rigid_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._rigid_mapping.rigid_body_entity_ids)
        }
        hinge_slot_by_id = {
            int(joint_id): slot
            for slot, joint_id in enumerate(self._joint_mapping.hinge_joint_ids)
        }
        slider_slot_by_id = {
            int(joint_id): slot
            for slot, joint_id in enumerate(self._joint_mapping.slider_joint_ids)
        }
        soft_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        fluid_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.fluid_entity_ids)
        }

        self._container_particle_offsets: list[int] = []
        self._container_particle_counts: list[int] = []
        self._fluid_particle_offsets: list[int] = []
        self._fluid_particle_counts: list[int] = []
        self._fluid_particle_indices_flat: list[int] = []
        self._fluid_particle_env_indices_flat: list[int] = []
        self._arm_hinge_slots: list[list[int]] = []
        self._arm_slider_slots: list[int] = []
        self._tooltip_body_indices: list[int] = []
        self._removal_teleport_positions: list[tuple[float, float, float, float]] = []

        world = self.runtime.world()
        body_entity_order = [
            "psm_base_link",
            "psm_yaw_link",
            "psm_pitch_back_link",
            "psm_pitch_bottom_link",
            "psm_pitch_end_link",
            "psm_pitch_top_link",
            "psm_pitch_front_link",
            "psm_main_insertion_link",
            "psm_tool_roll_link",
            "psm_tool_pitch_link",
            "psm_tool_yaw_link",
            "psm_tool_gripper1_link",
            "psm_tool_gripper2_link",
        ]

        for env_index, instance in enumerate(self._psm_build.instances):
            container_slot = soft_slot_by_entity[self._container_entities[env_index]]
            self._container_particle_offsets.append(
                int(self._particle_mapping.soft_body_particle_offsets[container_slot])
            )
            self._container_particle_counts.append(
                int(self._particle_mapping.soft_body_particle_counts[container_slot])
            )

            fluid_entity = self._fluid_specs[env_index][0]
            fluid_slot = fluid_slot_by_entity[fluid_entity]
            self._fluid_particle_offsets.append(
                int(self._particle_mapping.fluid_particle_offsets[fluid_slot])
            )
            self._fluid_particle_counts.append(
                int(self._particle_mapping.fluid_particle_counts[fluid_slot])
            )
            fluid_particle_offset = self._fluid_particle_offsets[-1]
            fluid_particle_count = self._fluid_particle_counts[-1]
            self._fluid_particle_indices_flat.extend(
                fluid_particle_offset + local_index for local_index in range(fluid_particle_count)
            )
            self._fluid_particle_env_indices_flat.extend(env_index for _ in range(fluid_particle_count))

            self._arm_hinge_slots.append(
                [
                    hinge_slot_by_id.get(int(instance.arm_joint_ids[0]), 0xFFFFFFFF),
                    hinge_slot_by_id.get(int(instance.arm_joint_ids[1]), 0xFFFFFFFF),
                    hinge_slot_by_id.get(int(instance.arm_joint_ids[4]), 0xFFFFFFFF),
                    hinge_slot_by_id.get(int(instance.arm_joint_ids[5]), 0xFFFFFFFF),
                ]
            )
            self._arm_slider_slots.append(
                slider_slot_by_id.get(int(instance.arm_joint_ids[2]), 0xFFFFFFFF)
            )

            tooltip_body_index = rigid_slot_by_entity[self._tooltip_body_entities[env_index]]
            self._tooltip_body_indices.append(tooltip_body_index)
            env_center_x, env_center_z = self._env_scene_centers[env_index]
            self._removal_teleport_positions.append(
                (
                    env_center_x + 5.0 * self.scene_scale,
                    -5.0 * self.scene_scale,
                    env_center_z - 5.0 * self.scene_scale,
                    0.0,
                )
            )

            rigid_reset_count_for_env = 0
            for link_name in body_entity_order:
                if link_name != "psm_base_link" and link_name not in instance.link_entities:
                    continue
                entity = (
                    instance.base_entity
                    if link_name == "psm_base_link"
                    else instance.link_entities[link_name]
                )
                slot = rigid_slot_by_entity[entity]
                transform = world.try_get_transform(entity)
                if transform is None:
                    raise RuntimeError(f"Missing transform for rigid reset entity {entity}.")
                self._rigid_reset_body_indices.append(slot)
                self._rigid_reset_positions.append(
                    (
                        float(transform.world_transform.position.x),
                        float(transform.world_transform.position.y),
                        float(transform.world_transform.position.z),
                        0.0,
                    )
                )
                rotation = transform.world_transform.rotation
                self._rigid_reset_orientations.append(
                    (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
                )
                rigid_reset_count_for_env += 1
            if env_index == 0:
                self._rigid_body_reset_count = rigid_reset_count_for_env

    def _build_reset_positions(self, world: neo.World) -> list[tuple[float, float, float, float]]:
        soft_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }
        fluid_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.fluid_entity_ids)
        }
        reset_positions = [
            (0.0, 0.0, 0.0, 0.0) for _ in range(self._particle_mapping.particle_count)
        ]

        for entity in self._container_entities:
            authoring_particles = world.try_get_soft_body_authoring_particles(entity)
            if authoring_particles is None:
                raise RuntimeError(f"Authoring particles were unavailable for soft body {entity}.")
            slot = soft_slot_by_entity[entity]
            particle_offset = self._particle_mapping.soft_body_particle_offsets[slot]
            particle_count = self._particle_mapping.soft_body_particle_counts[slot]
            if particle_count != authoring_particles.particle_count:
                raise RuntimeError("Prepared soft-body particle count did not match authoring data.")
            for local_index, position in enumerate(authoring_particles.rest_positions):
                reset_positions[particle_offset + local_index] = (
                    float(position.x),
                    float(position.y),
                    float(position.z),
                    0.0,
                )

        for entity, position, size, spacing in self._fluid_specs:
            slot = fluid_slot_by_entity[entity]
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
            "PsmBloodSuction.Actions",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.observation_buffer, observation_flat = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime, "PsmBloodSuction.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime, "PsmBloodSuction.Terminated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime, "PsmBloodSuction.Truncated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime, "PsmBloodSuction.EpisodeSteps", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime, "PsmBloodSuction.ResetMask", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.current_joint_targets_buffer, self.current_joint_targets_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.CurrentJointTargets",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.reset_joint_targets_buffer, self.reset_joint_targets_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.ResetJointTargets",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.joint_limits_buffer, self.joint_limits_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.JointLimits",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM, 2],
        )
        self.env_metadata_u32_buffer, self.env_metadata_u32_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.EnvMetadataU32",
            self.env_count * 12,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, 12],
        )
        self.env_metadata_f32_buffer, self.env_metadata_f32_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.EnvMetadataF32",
            self.env_count * 2,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 2, 4],
        )
        self.env_stats_buffer, self.env_stats_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.EnvStats",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.env_step_u32_buffer, self.env_step_u32_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.EnvStepU32",
            self.env_count * 4,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, 4],
        )
        self.fluid_active_mask_buffer, self.fluid_active_mask_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.FluidActiveMask",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.fluid_particle_indices_buffer, self.fluid_particle_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.FluidParticleIndices",
            len(self._fluid_particle_indices_flat),
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.fluid_particle_env_indices_buffer, self.fluid_particle_env_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.FluidParticleEnvIndices",
            len(self._fluid_particle_env_indices_flat),
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )
        self.rigid_reset_body_indices_buffer, self.rigid_reset_body_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.RigidResetBodyIndices",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, self._rigid_body_reset_count],
        )
        self.rigid_reset_positions_buffer, self.rigid_reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.RigidResetPositions",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self._rigid_body_reset_count, 4],
        )
        self.rigid_reset_orientations_buffer, self.rigid_reset_orientations_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmBloodSuction.RigidResetOrientations",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self._rigid_body_reset_count, 4],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
                self.runtime,
                "PsmBloodSuction.RgbObservation",
                self.env_count * self.image_width * self.image_height,
                neo.SharedBufferTensorDTypeCode.Float,
                element_stride_bytes=16,
                shape=[self.env_count, self.image_height, self.image_width, 4],
            )
        if self.enable_visualization_camera:
            self.visualization_rgb_observation_buffer, self.visualization_rgb_observation_tensor = (
                self._register_shared_buffer(
                    self.runtime,
                    "PsmBloodSuction.VisualizationRgbObservation",
                    self.env_count * self.visualization_image_width * self.visualization_image_height,
                    neo.SharedBufferTensorDTypeCode.Float,
                    element_stride_bytes=16,
                    shape=[
                        self.env_count,
                        self.visualization_image_height,
                        self.visualization_image_width,
                        4,
                    ],
                )
            )

    def _populate_lookup_buffers(self) -> None:
        device = self.action_tensor.device
        self.action_tensor.zero_()
        self.reset_mask_tensor.zero_()
        self.reward_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.env_stats_tensor.zero_()
        self.env_step_u32_tensor.zero_()
        self.fluid_active_mask_tensor.zero_()

        self.current_joint_targets_tensor.copy_(
            torch.tensor(self._reset_joint_targets, device=device, dtype=self.current_joint_targets_tensor.dtype)
        )
        self.reset_joint_targets_tensor.copy_(self.current_joint_targets_tensor)
        self.joint_limits_tensor.copy_(
            torch.tensor(
                [
                    [instance.arm_joint_limits[index] for index in self.ACTIVE_ARM_JOINT_INDICES]
                    for instance in self._psm_build.instances
                ],
                device=device,
                dtype=self.joint_limits_tensor.dtype,
            )
        )
        env_metadata_u32 = []
        env_metadata_f32 = []
        for env_index in range(self.env_count):
            env_metadata_u32.append(
                [
                    int(self._arm_hinge_slots[env_index][0]),
                    int(self._arm_hinge_slots[env_index][1]),
                    int(self._arm_slider_slots[env_index]),
                    int(self._arm_hinge_slots[env_index][2]),
                    int(self._arm_hinge_slots[env_index][3]),
                    int(self._tooltip_body_indices[env_index]),
                    int(self._container_particle_offsets[env_index]),
                    int(self._container_particle_counts[env_index]),
                    int(self._fluid_particle_offsets[env_index]),
                    int(self._fluid_particle_counts[env_index]),
                    int(self._fluid_particle_counts[env_index]),
                    0,
                ]
            )
            env_metadata_f32.append(
                [
                    list(self.tooltip_local_anchor),
                    list(self._removal_teleport_positions[env_index]),
                ]
            )
        self.env_metadata_u32_tensor.copy_(
            torch.tensor(env_metadata_u32, device=device, dtype=self.env_metadata_u32_tensor.dtype)
        )
        self.env_metadata_f32_tensor.copy_(
            torch.tensor(env_metadata_f32, device=device, dtype=self.env_metadata_f32_tensor.dtype)
        )
        self.reset_positions_tensor.copy_(
            torch.tensor(
                self._reset_positions,
                device=device,
                dtype=self.reset_positions_tensor.dtype,
            )
        )
        self.fluid_particle_indices_tensor.copy_(
            torch.tensor(
                self._fluid_particle_indices_flat,
                device=device,
                dtype=self.fluid_particle_indices_tensor.dtype,
            )
        )
        self.fluid_particle_env_indices_tensor.copy_(
            torch.tensor(
                self._fluid_particle_env_indices_flat,
                device=device,
                dtype=self.fluid_particle_env_indices_tensor.dtype,
            )
        )
        self.rigid_reset_body_indices_tensor.copy_(
            torch.tensor(
                self._rigid_reset_body_indices,
                device=device,
                dtype=self.rigid_reset_body_indices_tensor.dtype,
            ).view(self.env_count, self._rigid_body_reset_count)
        )
        self.rigid_reset_positions_tensor.copy_(
            torch.tensor(
                self._rigid_reset_positions,
                device=device,
                dtype=self.rigid_reset_positions_tensor.dtype,
            ).view(self.env_count, self._rigid_body_reset_count, 4)
        )
        self.rigid_reset_orientations_tensor.copy_(
            torch.tensor(
                self._rigid_reset_orientations,
                device=device,
                dtype=self.rigid_reset_orientations_tensor.dtype,
            ).view(self.env_count, self._rigid_body_reset_count, 4)
        )

        handles = [
            self.action_buffer,
            self.reset_mask_buffer,
            self.observation_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.episode_steps_buffer,
            self.current_joint_targets_buffer,
            self.reset_joint_targets_buffer,
            self.joint_limits_buffer,
            self.env_metadata_u32_buffer,
            self.env_metadata_f32_buffer,
            self.env_stats_buffer,
            self.env_step_u32_buffer,
            self.fluid_active_mask_buffer,
            self.fluid_particle_indices_buffer,
            self.fluid_particle_env_indices_buffer,
            self.reset_positions_buffer,
            self.rigid_reset_body_indices_buffer,
            self.rigid_reset_positions_buffer,
            self.rigid_reset_orientations_buffer,
        ]
        if self.enable_rgb_observation:
            self.rgb_observation_tensor.zero_()
            handles.append(self.rgb_observation_buffer)
        if self.enable_visualization_camera:
            self.visualization_rgb_observation_tensor.zero_()
            handles.append(self.visualization_rgb_observation_buffer)
        self._sync_from_cuda(self.runtime, handles)

    def _create_custom_passes(self) -> None:
        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "PsmBloodSuction.PrePhysics"
        pre_desc.shader_source = _PSM_BLOOD_SUCTION_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        _bind_resources(
            pre_desc,
            [
                ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CurrentJointTargets", self.current_joint_targets_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_JointLimits", self.joint_limits_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_HingeJoints", None, "joint.hinge", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_SliderJoints", None, "joint.slider", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EnvStepU32", self.env_step_u32_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        pre_desc.constant_buffer_variable_name = "PsmBloodSuctionPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 32
        pre_desc.constant_data = list(
            struct.pack(
                "<8f",
                self.rotational_action_scale,
                self.insertion_action_scale,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        pre_particle_desc = neo.CustomComputePassDesc()
        pre_particle_desc.debug_name = "PsmBloodSuction.PreParticle"
        pre_particle_desc.shader_source = _PSM_BLOOD_SUCTION_PRE_PARTICLE_SHADER
        pre_particle_desc.thread_group_size_x = 64
        _bind_resources(
            pre_particle_desc,
            [
                ("g_FluidParticleIndices", self.fluid_particle_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidParticleEnvIndices", self.fluid_particle_env_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticlePreviousPositions", None, "particle.previous_positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidActiveMask", self.fluid_active_mask_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EnvStepU32", self.env_step_u32_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        pre_particle_desc.constant_buffer_variable_name = "PsmBloodSuctionPreParticleConstants"
        pre_particle_desc.constant_buffer_size_bytes = 16
        pre_particle_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.suction_radius,
                self.removal_radius,
                self.suction_velocity_scale,
                0.0,
            )
        )
        pre_particle_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_particle_desc.dispatch.group_count_x = (len(self._fluid_particle_indices_flat) + 63) // 64
        self._pre_particle_pass = self._register_custom_pass(self.runtime, pre_particle_desc)

        post_particle_desc = neo.CustomComputePassDesc()
        post_particle_desc.debug_name = "PsmBloodSuction.PostParticle"
        post_particle_desc.shader_source = _PSM_BLOOD_SUCTION_POST_PARTICLE_SHADER
        post_particle_desc.thread_group_size_x = 64
        _bind_resources(
            post_particle_desc,
            [
                ("g_FluidParticleIndices", self.fluid_particle_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidParticleEnvIndices", self.fluid_particle_env_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidActiveMask", self.fluid_active_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvStepU32", self.env_step_u32_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        post_particle_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_particle_desc.dispatch.group_count_x = (len(self._fluid_particle_indices_flat) + 63) // 64
        self._post_particle_pass = self._register_custom_pass(self.runtime, post_particle_desc)

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "PsmBloodSuction.PostPhysics"
        post_desc.shader_source = _PSM_BLOOD_SUCTION_POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        _bind_resources(
            post_desc,
            [
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_HingeJointRuntimeStates", None, "joint.hinge_runtime", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_SliderJointRuntimeStates", None, "joint.slider_runtime", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvStats", self.env_stats_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EnvStepU32", self.env_step_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        post_desc.constant_buffer_variable_name = "PsmBloodSuctionPostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 32
        post_desc.constant_data = list(
            struct.pack(
                "<5fI2f",
                self.removed_reward_scale,
                self.progress_reward_scale,
                self.step_penalty,
                self.completion_bonus,
                float(self.completion_remaining_threshold),
                self.max_episode_steps,
                self.observation_distance_limit,
                0.0,
            )
        )
        post_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._post_pass = self._register_custom_pass(self.runtime, post_desc)

        reset_particles_desc = neo.CustomComputePassDesc()
        reset_particles_desc.debug_name = "PsmBloodSuction.ResetParticles"
        reset_particles_desc.shader_source = _PSM_BLOOD_SUCTION_RESET_PARTICLES_SHADER
        reset_particles_desc.thread_group_size_x = 64
        _bind_resources(
            reset_particles_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ResetPositions", self.reset_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidActiveMask", self.fluid_active_mask_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticlePreviousPositions", None, "particle.previous_positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticleVelocities", None, "particle.velocities", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_particles_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_particles_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_particles_pass = self._register_custom_pass(self.runtime, reset_particles_desc)

        reset_rigid_desc = neo.CustomComputePassDesc()
        reset_rigid_desc.debug_name = "PsmBloodSuction.ResetRigid"
        reset_rigid_desc.shader_source = _PSM_BLOOD_SUCTION_RESET_RIGID_SHADER
        reset_rigid_desc.thread_group_size_x = 64
        _bind_resources(
            reset_rigid_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidResetBodyIndices", self.rigid_reset_body_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidResetPositions", self.rigid_reset_positions_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidResetOrientations", self.rigid_reset_orientations_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidLinearVelocities", None, "rigid.linear_velocities", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_RigidAngularVelocities", None, "rigid.angular_velocities", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_CurrentJointTargets", self.current_joint_targets_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ResetJointTargets", self.reset_joint_targets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_HingeJoints", None, "joint.hinge", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_SliderJoints", None, "joint.slider", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_rigid_desc.constant_buffer_variable_name = "PsmBloodSuctionResetRigidConstants"
        reset_rigid_desc.constant_buffer_size_bytes = 16
        reset_rigid_desc.constant_data = list(
            struct.pack("<I3f", self._rigid_body_reset_count, 0.0, 0.0, 0.0)
        )
        reset_rigid_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_rigid_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_rigid_pass = self._register_custom_pass(self.runtime, reset_rigid_desc)

        reset_outputs_desc = neo.CustomComputePassDesc()
        reset_outputs_desc.debug_name = "PsmBloodSuction.ResetOutputs"
        reset_outputs_desc.shader_source = _PSM_BLOOD_SUCTION_RESET_OUTPUTS_SHADER
        reset_outputs_desc.thread_group_size_x = 64
        _bind_resources(
            reset_outputs_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CurrentJointTargets", self.current_joint_targets_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_FluidActiveMask", self.fluid_active_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvStats", self.env_stats_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_outputs_desc.constant_buffer_variable_name = "PsmBloodSuctionResetOutputsConstants"
        reset_outputs_desc.constant_buffer_size_bytes = 16
        reset_outputs_desc.constant_data = list(
            struct.pack("<4f", self.observation_distance_limit, 0.0, 0.0, 0.0)
        )
        reset_outputs_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_outputs_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_outputs_pass = self._register_custom_pass(self.runtime, reset_outputs_desc)

        if self.enable_rgb_observation:
            render_desc = neo.CustomComputePassDesc()
            render_desc.debug_name = "PsmBloodSuction.RgbObservation"
            render_desc.shader_source = _PSM_BLOOD_SUCTION_RGB_SHADER
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

        if self.enable_visualization_camera:
            visualization_render_desc = neo.CustomComputePassDesc()
            visualization_render_desc.debug_name = "PsmBloodSuction.VisualizationRgbObservation"
            visualization_render_desc.shader_source = _PSM_BLOOD_SUCTION_RGB_SHADER
            visualization_render_desc.thread_group_size_x = 8
            visualization_render_desc.thread_group_size_y = 8
            visualization_render_desc.resource_bindings = [neo.CustomComputeResourceBindingDesc() for _ in range(2)]
            visualization_render_desc.resource_bindings[0].shader_variable_name = "g_ColorTarget"
            visualization_render_desc.resource_bindings[0].render_target_binding = neo.GpuRenderTargetBinding()
            visualization_render_desc.resource_bindings[0].render_target_binding.target = self._visualization_render_target
            visualization_render_desc.resource_bindings[0].render_target_binding.first_layer = 0
            visualization_render_desc.resource_bindings[0].render_target_binding.layer_count = self.env_count
            visualization_render_desc.resource_bindings[0].render_target_texture_plane = (
                neo.GpuRenderTargetTexturePlane.Color
            )
            visualization_render_desc.resource_bindings[0].access = neo.CustomComputeResourceAccess.ReadOnly
            visualization_render_desc.resource_bindings[1].shader_variable_name = "g_ColorObservation"
            visualization_render_desc.resource_bindings[1].shared_buffer_handle = (
                self.visualization_rgb_observation_buffer
            )
            visualization_render_desc.resource_bindings[1].access = neo.CustomComputeResourceAccess.ReadWrite
            visualization_render_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
            visualization_render_desc.dispatch.group_count_x = (self.visualization_image_width + 7) // 8
            visualization_render_desc.dispatch.group_count_y = (self.visualization_image_height + 7) // 8
            visualization_render_desc.dispatch.group_count_z = self.env_count
            self._visualization_rgb_render_pass = self._register_custom_pass(
                self.runtime, visualization_render_desc
            )

    def _sync_outputs_to_cuda(self) -> None:
        handles = [
            self.observation_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.episode_steps_buffer,
        ]
        if self.return_combined_observation and self.enable_rgb_observation:
            handles.append(self.rgb_observation_buffer)
        self._sync_to_cuda(
            self.runtime,
            handles,
            device=self.observation_tensor.device,
        )

    def _render_rgb_observation(self) -> None:
        if not self.enable_rgb_observation:
            raise RuntimeError("RGB observations were not enabled for this PSM blood-suction env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute PSM blood-suction RGB observation pass.")

    def _render_visualization_rgb_observation(self) -> None:
        if not self.enable_visualization_camera:
            raise RuntimeError("Visualization camera was not enabled for this PSM blood-suction env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._visualization_rgb_render_pass):
            raise RuntimeError("Failed to execute PSM blood-suction visualization RGB observation pass.")

    def _make_observation_output(self) -> "torch.Tensor | dict[str, torch.Tensor]":
        if self.return_combined_observation:
            return {
                "vector": self.observation_tensor,
                "rgb": self.rgb_observation_tensor,
            }
        return self.observation_tensor

    def reset(self, env_ids: "torch.Tensor | list[int] | None" = None) -> "torch.Tensor | dict[str, torch.Tensor]":
        if env_ids is None:
            env_indices = torch.arange(self.env_count, device=self.action_tensor.device, dtype=torch.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(list(env_ids), device=self.action_tensor.device, dtype=torch.int64)
        if env_indices.numel() == 0:
            return self._make_observation_output()

        self.reset_mask_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        self.action_tensor.zero_()
        self.current_joint_targets_tensor.copy_(self.reset_joint_targets_tensor)
        self._sync_from_cuda(
            self.runtime,
            [
                self.reset_mask_buffer,
                self.action_buffer,
                self.current_joint_targets_buffer,
            ],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_particles_pass):
            raise RuntimeError("Failed to execute PSM blood-suction particle reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_rigid_pass):
            raise RuntimeError("Failed to execute PSM blood-suction rigid reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_outputs_pass):
            raise RuntimeError("Failed to execute PSM blood-suction output reset pass.")
        if self.return_combined_observation:
            self._render_rgb_observation()
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=False)
        return self._make_observation_output()

    def step(
        self, action_tensor: "torch.Tensor"
    ) -> tuple["torch.Tensor | dict[str, torch.Tensor]", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
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
            raise RuntimeError("Failed to execute PSM blood-suction pre-physics pass.")
        if not self.runtime.execute_custom_compute_pass(self._pre_particle_pass):
            raise RuntimeError("Failed to execute PSM blood-suction pre-particle pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("PSM blood-suction physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_particle_pass):
            raise RuntimeError("Failed to execute PSM blood-suction post-particle pass.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute PSM blood-suction post-physics pass.")
        if self.return_combined_observation:
            self._render_rgb_observation()
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=True)
        return (
            self._make_observation_output(),
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def render(self) -> "torch.Tensor":
        if self.enable_visualization_camera:
            self._render_visualization_rgb_observation()
            if not self.runtime.sync_shared_buffer_to_cuda(self.visualization_rgb_observation_buffer):
                raise RuntimeError(
                    "Failed to synchronize PSM blood-suction visualization RGB observation buffer to CUDA."
                )
            self.runtime.end_frame(self._frame)
            torch.cuda.synchronize(device=self.visualization_rgb_observation_tensor.device)
            return self.visualization_rgb_observation_tensor
        if not self.enable_rgb_observation:
            raise RuntimeError("RGB observations were not enabled for this PSM blood-suction env.")
        self._render_rgb_observation()
        if not self.runtime.sync_shared_buffer_to_cuda(self.rgb_observation_buffer):
            raise RuntimeError("Failed to synchronize PSM blood-suction RGB observation buffer to CUDA.")
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor

    def close(self) -> None:
        self.close_runtime(getattr(self, "runtime", None))
        self.runtime = None


__all__ = ["PsmBloodSuctionTorchVectorEnv"]
