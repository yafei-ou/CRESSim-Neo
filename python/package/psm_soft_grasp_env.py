from __future__ import annotations

import math
import struct

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
        "cressim_neo.psm_soft_grasp_env requires PyTorch to be installed."
    ) from exc


_PSM_SOFT_GRASP_PRE_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmSoftGraspPrePhysicsConstants
{
    float rotationalActionScale;
    float insertionActionScale;
    float tooltipProximityThreshold;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(float, g_Actions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_CurrentJointTargets);
CRESSIM_STRUCTURED_BUFFER(float2, g_JointLimits);
CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PreviousGraspClosed);
CRESSIM_RW_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidParticleAttachmentConstraint, g_Attachments);

static const uint kCommandJointCount = 7u;
static const uint kArmHingeCount = 5u;
static const uint kEnvMetadataStride = 14u;

uint jointBaseIndex(uint envIndex)
{
    return envIndex * kCommandJointCount;
}

uint armHingeBaseIndex(uint envIndex)
{
    return envIndex * kArmHingeCount;
}

uint envMeta(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataStride + offset);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_EnvMetadataF32.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint actionBase = envIndex * kCommandJointCount;
    const uint targetBase = jointBaseIndex(envIndex);

    float commandTargets[kCommandJointCount];
    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        commandTargets[i] = CRESSIM_SB_LOAD(g_CurrentJointTargets, targetBase + i);
    }

    const float yawAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 0u), -1.0f, 1.0f);
    const float pitchAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 1u), -1.0f, 1.0f);
    const float insertionAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 2u), -1.0f, 1.0f);
    const float rollAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 3u), -1.0f, 1.0f);
    const float wristPitchAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 4u), -1.0f, 1.0f);
    const float wristYawAction = clamp(CRESSIM_SB_LOAD(g_Actions, actionBase + 5u), -1.0f, 1.0f);
    const float graspAction = CRESSIM_SB_LOAD(g_Actions, actionBase + 6u);

    commandTargets[0u] += yawAction * rotationalActionScale;
    commandTargets[1u] += pitchAction * rotationalActionScale;
    commandTargets[2u] += insertionAction * insertionActionScale;
    commandTargets[3u] += rollAction * rotationalActionScale;
    commandTargets[4u] += wristPitchAction * rotationalActionScale;
    commandTargets[5u] += wristYawAction * rotationalActionScale;

    [unroll] for (uint i = 0u; i < 6u; ++i)
    {
        const float2 limits = CRESSIM_SB_LOAD(g_JointLimits, targetBase + i);
        commandTargets[i] = clamp(commandTargets[i], limits.x, limits.y);
    }

    const float2 jawLimits = CRESSIM_SB_LOAD(g_JointLimits, targetBase + 6u);
    commandTargets[6u] = graspAction > 0.0f ? jawLimits.x : jawLimits.y;

    [unroll] for (uint i = 0u; i < kCommandJointCount; ++i)
    {
        CRESSIM_SB_STORE(g_CurrentJointTargets, targetBase + i, commandTargets[i]);
    }

    const uint armHingeBase = armHingeBaseIndex(envIndex);
    const uint hingeSlot0 = envMeta(envIndex, 0u);
    const uint hingeSlot1 = envMeta(envIndex, 1u);
    const uint hingeSlot2 = envMeta(envIndex, 2u);
    const uint hingeSlot3 = envMeta(envIndex, 3u);
    const uint hingeSlot4 = envMeta(envIndex, 4u);
    const uint sliderSlot = envMeta(envIndex, 5u);
    const uint jawSlotA = envMeta(envIndex, 6u);
    const uint jawSlotB = envMeta(envIndex, 7u);

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
    if (hingeSlot4 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot4);
        joint.driveTargetParams.x = commandTargets[5u];
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot4, joint);
    }
    if (jawSlotA != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jawSlotA);
        joint.driveTargetParams.x = commandTargets[6u];
        CRESSIM_SB_STORE(g_HingeJoints, jawSlotA, joint);
    }
    if (jawSlotB != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jawSlotB);
        joint.driveTargetParams.x = -commandTargets[6u];
        CRESSIM_SB_STORE(g_HingeJoints, jawSlotB, joint);
    }

    const uint attachmentSlot = envMeta(envIndex, 10u);
    GpuRigidParticleAttachmentConstraint attachment =
        CRESSIM_SB_LOAD(g_Attachments, attachmentSlot);
    const uint wasClosed = CRESSIM_SB_LOAD(g_PreviousGraspClosed, envIndex);
    if (graspAction <= 0.0f)
    {
        CRESSIM_SB_STORE(g_PreviousGraspClosed, envIndex, 0u);
        attachment.enabled = 0u;
        CRESSIM_SB_STORE(g_Attachments, attachmentSlot, attachment);
        return;
    }

    CRESSIM_SB_STORE(g_PreviousGraspClosed, envIndex, 1u);
    if (wasClosed != 0u || attachment.enabled != 0u)
    {
        return;
    }

    const uint tooltipBodyIndex = envMeta(envIndex, 11u);
    const float4 rigidPosInvMass = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex).xyz;
    const float3 tooltipPosition =
        rigidPosInvMass.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);

    const uint particleOffset = envMeta(envIndex, 8u);
    const uint particleCount = envMeta(envIndex, 9u);
    if (particleCount == 0u)
    {
        return;
    }

    float bestDistanceSq = 1.0e30f;
    uint bestParticleIndex = kInvalidIndex;
    for (uint i = 0u; i < particleCount; ++i)
    {
        const uint particleIndex = particleOffset + i;
        const float3 particlePosition =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
        const float distanceSq = dot(particlePosition - tooltipPosition, particlePosition - tooltipPosition);
        if (distanceSq < bestDistanceSq)
        {
            bestDistanceSq = distanceSq;
            bestParticleIndex = particleIndex;
        }
    }

    if (bestParticleIndex == kInvalidIndex ||
        bestDistanceSq > tooltipProximityThreshold * tooltipProximityThreshold)
    {
        return;
    }

    attachment.particleIndex = bestParticleIndex;
    attachment.enabled = 1u;
    CRESSIM_SB_STORE(g_Attachments, attachmentSlot, attachment);
}
"""


_PSM_SOFT_GRASP_POST_PHYSICS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmSoftGraspPostPhysicsConstants
{
    float liftTargetDistance;
    uint maxEpisodeSteps;
    float tooltipRewardScale;
    float padding0;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_STRUCTURED_BUFFER(GpuHingeJointRuntimeState, g_HingeJointRuntimeStates);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJointRuntimeState, g_SliderJointRuntimeStates);
CRESSIM_STRUCTURED_BUFFER(GpuRigidParticleAttachmentConstraint, g_Attachments);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PreviousGraspClosed);

static const uint kArmHingeCount = 5u;
static const uint kEnvMetadataStride = 14u;

uint obsBaseIndex(uint envIndex)
{
    return envIndex * 29u;
}

uint envMeta(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataStride + offset);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_EnvMetadataF32.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    const uint armHingeBase = envIndex * kArmHingeCount;
    const uint targetParticleIndex = envMeta(envIndex, 13u);
    const float3 targetPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, targetParticleIndex).xyz;
    const float targetRestHeight = CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex).w;
    const uint tooltipBodyIndex = envMeta(envIndex, 11u);
    const float4 rigidPosInvMass = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex).xyz;
    const float3 tooltipPosition =
        rigidPosInvMass.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);
    const uint attachmentSlot = envMeta(envIndex, 10u);
    const GpuRigidParticleAttachmentConstraint attachment =
        CRESSIM_SB_LOAD(g_Attachments, attachmentSlot);

    const uint hingeSlot0 = envMeta(envIndex, 0u);
    const uint hingeSlot1 = envMeta(envIndex, 1u);
    const uint hingeSlot2 = envMeta(envIndex, 2u);
    const uint hingeSlot3 = envMeta(envIndex, 3u);
    const uint hingeSlot4 = envMeta(envIndex, 4u);
    const uint sliderSlot = envMeta(envIndex, 5u);
    const uint jawSlot = envMeta(envIndex, 6u);

    const float jointPos0 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot0).angleState.y;
    const float jointVel0 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot0).angleState.z;
    const float jointPos1 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot1).angleState.y;
    const float jointVel1 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot1).angleState.z;
    const float jointPos2 = CRESSIM_SB_LOAD(g_SliderJointRuntimeStates, sliderSlot).state.x;
    const float jointVel2 = CRESSIM_SB_LOAD(g_SliderJointRuntimeStates, sliderSlot).state.y;
    const float jointPos3 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot2).angleState.y;
    const float jointVel3 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot2).angleState.z;
    const float jointPos4 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot3).angleState.y;
    const float jointVel4 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot3).angleState.z;
    const float jointPos5 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot4).angleState.y;
    const float jointVel5 = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, hingeSlot4).angleState.z;
    const float jawPos = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, jawSlot).angleState.y;
    const float jawVel = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, jawSlot).angleState.z;

    const float tooltipDistance = length(targetPosition - tooltipPosition);
    const float liftAmount = max(0.0f, targetPosition.y - targetRestHeight);
    const float reward =
        -tooltipDistance * tooltipRewardScale +
        (attachment.enabled != 0u ? 0.5f : 0.0f) +
        liftAmount * 4.0f;

    const uint nextEpisodeStep = CRESSIM_SB_LOAD(g_EpisodeSteps, envIndex) + 1u;
    const uint terminated =
        (attachment.enabled != 0u && liftAmount >= liftTargetDistance) ? 1u : 0u;
    const uint truncated = nextEpisodeStep >= maxEpisodeSteps ? 1u : 0u;
    const uint markerPoseSlot = envMeta(envIndex, 12u);
    if (markerPoseSlot != kInvalidIndex)
    {
        CRESSIM_SB_STORE(g_EntityPositions, markerPoseSlot, float4(targetPosition, 0.0f));
    }

    const uint obsBase = obsBaseIndex(envIndex);
    CRESSIM_SB_STORE(g_Observations, obsBase + 0u, jointPos0);
    CRESSIM_SB_STORE(g_Observations, obsBase + 1u, jointPos1);
    CRESSIM_SB_STORE(g_Observations, obsBase + 2u, jointPos2);
    CRESSIM_SB_STORE(g_Observations, obsBase + 3u, jointPos3);
    CRESSIM_SB_STORE(g_Observations, obsBase + 4u, jointPos4);
    CRESSIM_SB_STORE(g_Observations, obsBase + 5u, jointPos5);
    CRESSIM_SB_STORE(g_Observations, obsBase + 6u, jawPos);
    CRESSIM_SB_STORE(g_Observations, obsBase + 7u, jointVel0);
    CRESSIM_SB_STORE(g_Observations, obsBase + 8u, jointVel1);
    CRESSIM_SB_STORE(g_Observations, obsBase + 9u, jointVel2);
    CRESSIM_SB_STORE(g_Observations, obsBase + 10u, jointVel3);
    CRESSIM_SB_STORE(g_Observations, obsBase + 11u, jointVel4);
    CRESSIM_SB_STORE(g_Observations, obsBase + 12u, jointVel5);
    CRESSIM_SB_STORE(g_Observations, obsBase + 13u, jawVel);
    CRESSIM_SB_STORE(g_Observations, obsBase + 14u, targetPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 15u, targetPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 16u, targetPosition.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 17u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 18u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 19u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 20u, 1.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 21u, tooltipPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 22u, tooltipPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 23u, tooltipPosition.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 24u, rigidOrientation.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 25u, rigidOrientation.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 26u, rigidOrientation.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 27u, rigidOrientation.w);
    CRESSIM_SB_STORE(g_Observations, obsBase + 28u, attachment.enabled != 0u ? 1.0f : 0.0f);
    CRESSIM_SB_STORE(g_Rewards, envIndex, reward);
    CRESSIM_SB_STORE(g_Terminated, envIndex, terminated);
    CRESSIM_SB_STORE(g_Truncated, envIndex, truncated);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, nextEpisodeStep);
}
"""


_PSM_SOFT_GRASP_RESET_PARTICLES_SHADER = r"""
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
        positionInvMass.xyz = resetPosition.xyz;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex, positionInvMass);
        CRESSIM_SB_STORE(g_ParticlePreviousPositions, particleIndex, float4(resetPosition.xyz, 0.0f));
        CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
}
"""


_PSM_SOFT_GRASP_RESET_RIGID_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer PsmSoftGraspResetRigidConstants
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

static const uint kCommandJointCount = 7u;
static const uint kArmHingeCount = 5u;
static const uint kEnvMetadataStride = 14u;

uint envMeta(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataStride + offset);
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
        CRESSIM_SB_STORE(g_CurrentJointTargets, targetBase + i,
                         CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + i));
    }

    const uint armHingeBase = envIndex * kArmHingeCount;
    const float target0 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 0u);
    const float target1 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 1u);
    const float target2 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 2u);
    const float target3 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 3u);
    const float target4 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 4u);
    const float target5 = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 5u);
    const float jawTarget = CRESSIM_SB_LOAD(g_ResetJointTargets, targetBase + 6u);

    const uint hingeSlot0 = envMeta(envIndex, 0u);
    const uint hingeSlot1 = envMeta(envIndex, 1u);
    const uint hingeSlot2 = envMeta(envIndex, 2u);
    const uint hingeSlot3 = envMeta(envIndex, 3u);
    const uint hingeSlot4 = envMeta(envIndex, 4u);
    const uint sliderSlot = envMeta(envIndex, 5u);
    const uint jawSlotA = envMeta(envIndex, 6u);
    const uint jawSlotB = envMeta(envIndex, 7u);

    if (hingeSlot0 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot0);
        joint.driveTargetParams.x = target0;
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot0, joint);
    }
    if (hingeSlot1 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot1);
        joint.driveTargetParams.x = target1;
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot1, joint);
    }
    if (sliderSlot != kInvalidIndex)
    {
        GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, sliderSlot);
        joint.driveTargetParams.x = target2;
        CRESSIM_SB_STORE(g_SliderJoints, sliderSlot, joint);
    }
    if (hingeSlot2 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot2);
        joint.driveTargetParams.x = target3;
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot2, joint);
    }
    if (hingeSlot3 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot3);
        joint.driveTargetParams.x = target4;
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot3, joint);
    }
    if (hingeSlot4 != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, hingeSlot4);
        joint.driveTargetParams.x = target5;
        CRESSIM_SB_STORE(g_HingeJoints, hingeSlot4, joint);
    }
    if (jawSlotA != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jawSlotA);
        joint.driveTargetParams.x = jawTarget;
        CRESSIM_SB_STORE(g_HingeJoints, jawSlotA, joint);
    }
    if (jawSlotB != kInvalidIndex)
    {
        GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jawSlotB);
        joint.driveTargetParams.x = -jawTarget;
        CRESSIM_SB_STORE(g_HingeJoints, jawSlotB, joint);
    }
}
"""


_PSM_SOFT_GRASP_RESET_OUTPUTS_SHADER = r"""
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ResetMask);
CRESSIM_STRUCTURED_BUFFER(uint, g_TargetChoice);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateLocalIndices);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EnvMetadataU32);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EnvMetadataF32);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidParticleAttachmentConstraint, g_Attachments);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Observations);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_Rewards);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Terminated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Truncated);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_EpisodeSteps);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PreviousGraspClosed);

static const uint kCommandJointCount = 7u;
static const uint kEnvMetadataStride = 14u;

uint envMeta(uint envIndex, uint offset)
{
    return CRESSIM_SB_LOAD(g_EnvMetadataU32, envIndex * kEnvMetadataStride + offset);
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

    uint candidateCount = 0u;
    g_CandidateLocalIndices.GetDimensions(candidateCount, stride);
    const uint particleOffset = envMeta(envIndex, 8u);
    const uint candidateIndex =
        candidateCount > 0u ? CRESSIM_SB_LOAD(g_TargetChoice, envIndex) % candidateCount : 0u;
    const uint localParticleIndex =
        candidateCount > 0u ? CRESSIM_SB_LOAD(g_CandidateLocalIndices, candidateIndex) : 0u;
    const uint targetParticleIndex = particleOffset + localParticleIndex;
    const float3 targetPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, targetParticleIndex).xyz;
    CRESSIM_SB_STORE(g_EnvMetadataU32, envIndex * kEnvMetadataStride + 13u, targetParticleIndex);
    float4 envFloat = CRESSIM_SB_LOAD(g_EnvMetadataF32, envIndex);
    envFloat.w = targetPosition.y;
    CRESSIM_SB_STORE(g_EnvMetadataF32, envIndex, envFloat);

    const uint attachmentSlot = envMeta(envIndex, 10u);
    GpuRigidParticleAttachmentConstraint attachment =
        CRESSIM_SB_LOAD(g_Attachments, attachmentSlot);
    attachment.particleIndex = targetParticleIndex;
    attachment.enabled = 0u;
    CRESSIM_SB_STORE(g_Attachments, attachmentSlot, attachment);
    CRESSIM_SB_STORE(g_PreviousGraspClosed, envIndex, 0u);

    const uint tooltipBodyIndex = envMeta(envIndex, 11u);
    const float4 rigidPosInvMass = CRESSIM_SB_LOAD(g_RigidPositionsInvMass, tooltipBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidOrientations, tooltipBodyIndex));
    const float3 tooltipLocalAnchor = envFloat.xyz;
    const float3 tooltipPosition =
        rigidPosInvMass.xyz + QuaternionRotate(rigidOrientation, tooltipLocalAnchor);
    const uint markerPoseSlot = envMeta(envIndex, 12u);
    if (markerPoseSlot != kInvalidIndex)
    {
        CRESSIM_SB_STORE(g_EntityPositions, markerPoseSlot, float4(targetPosition, 0.0f));
    }

    const uint obsBase = envIndex * 29u;
    [unroll] for (uint i = 0u; i < 7u; ++i)
    {
        CRESSIM_SB_STORE(g_Observations, obsBase + i, 0.0f);
    }
    [unroll] for (uint i = 7u; i < 14u; ++i)
    {
        CRESSIM_SB_STORE(g_Observations, obsBase + i, 0.0f);
    }
    CRESSIM_SB_STORE(g_Observations, obsBase + 14u, targetPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 15u, targetPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 16u, targetPosition.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 17u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 18u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 19u, 0.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 20u, 1.0f);
    CRESSIM_SB_STORE(g_Observations, obsBase + 21u, tooltipPosition.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 22u, tooltipPosition.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 23u, tooltipPosition.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 24u, rigidOrientation.x);
    CRESSIM_SB_STORE(g_Observations, obsBase + 25u, rigidOrientation.y);
    CRESSIM_SB_STORE(g_Observations, obsBase + 26u, rigidOrientation.z);
    CRESSIM_SB_STORE(g_Observations, obsBase + 27u, rigidOrientation.w);
    CRESSIM_SB_STORE(g_Observations, obsBase + 28u, 0.0f);
    CRESSIM_SB_STORE(g_Rewards, envIndex, 0.0f);
    CRESSIM_SB_STORE(g_Terminated, envIndex, 0u);
    CRESSIM_SB_STORE(g_Truncated, envIndex, 0u);
    CRESSIM_SB_STORE(g_EpisodeSteps, envIndex, 0u);
}
"""


_PSM_SOFT_GRASP_RGB_SHADER = r"""
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


def _quat_tuple(quaternion: neo.Quaternion) -> tuple[float, float, float, float]:
    return (float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))


def _quat_conjugate(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def _quat_rotate(
    quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


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


class PsmSoftGraspTorchVectorEnv(TorchStagedVectorEnvBase):
    ACTION_DIM = 7
    OBSERVATION_DIM = 29
    OBS_DIM = OBSERVATION_DIM
    JOINT_POSITION_SLICE = slice(0, 7)
    JOINT_VELOCITY_SLICE = slice(7, 14)
    TARGET_POSE_SLICE = slice(14, 21)
    TOOLTIP_POSE_SLICE = slice(21, 28)
    ATTACHMENT_FLAG_INDEX = 28
    DEFAULT_TISSUE_PARTICLE_SPACING = 0.08
    DEFAULT_TISSUE_PARTICLE_RADIUS = 0.04
    DEFAULT_TABLE_SURFACE_Y = 0.0
    DEFAULT_INITIAL_TISSUE_CLEARANCE = 0.04
    DEFAULT_INITIAL_INSERTION = 0.10
    DEFAULT_TISSUE_OFFSET_X = 0.40
    DEFAULT_TISSUE_OFFSET_Z = 0.40
    DEFAULT_JAW_CLOSE_TARGET = 0.05
    DEFAULT_JAW_OPEN_TARGET = 0.65
    INVALID_INDEX = 0xFFFFFFFF

    def __init__(
        self,
        env_count: int = 16,
        max_episode_steps: int = 180,
        enable_rgb_observation: bool = False,
        enable_target_marker: bool = True,
        image_width: int = 160,
        image_height: int = 160,
        psm_scale: float = 10.0,
        rotational_action_scale: float = 0.02,
        insertion_action_scale: float = 0.02,
        tissue_width: float = 0.96,
        tissue_height: float = 0.96,
        tissue_thickness: float = 0.08,
        tissue_particle_spacing: float = DEFAULT_TISSUE_PARTICLE_SPACING,
        tissue_particle_radius: float = DEFAULT_TISSUE_PARTICLE_RADIUS,
        tooltip_proximity_threshold: float = 0.08,
        lift_target_distance: float = 0.25,
        target_sampling_fraction_x: float = 0.0,
        target_sampling_fraction_z: float = 0.0,
        resolve_root=None,
        urdf_path=None,
    ) -> None:
        super().__init__(env_count)
        self.max_episode_steps = max_episode_steps
        self.enable_rgb_observation = enable_rgb_observation
        self.enable_target_marker = enable_target_marker
        self.image_width = image_width
        self.image_height = image_height
        self.psm_scale = psm_scale
        self.rotational_action_scale = rotational_action_scale
        # Slider joint limits/targets are already authored in scaled world units,
        # so applying psm_scale again here overdrives insertion motion.
        self.insertion_action_scale = insertion_action_scale
        self.tissue_width = tissue_width
        self.tissue_height = tissue_height
        self.tissue_thickness = tissue_thickness
        self.tissue_particle_spacing = tissue_particle_spacing
        self.tissue_particle_radius = tissue_particle_radius
        self.tooltip_proximity_threshold = tooltip_proximity_threshold
        self.lift_target_distance = lift_target_distance
        self.target_sampling_fraction_x = max(0.0, min(1.0, float(target_sampling_fraction_x)))
        self.target_sampling_fraction_z = max(0.0, min(1.0, float(target_sampling_fraction_z)))

        config = get_psm_default_runtime_config(env_count)
        config.gpu_device_desc.preferred_backend = neo.GpuBackend.Vulkan
        config.gpu_device_desc.enable_validation = False
        config.physics_desc.enable_blocking_readback = False
        config.physics_desc.substeps = 4
        config.physics_desc.default_iterations = 32
        config.physics_desc.soft_internal_iterations = 32
        config.physics_desc.soft_contact_iterations = 16
        config.physics_desc.rigid_rigid_contact_iterations = 8
        if enable_rgb_observation:
            config.scene_layout.max_renderable_objects_per_env = 24
            config.scene_layout.max_lights_per_env = 1
            config.scene_layout.max_cameras_per_env = 1
        else:
            config.scene_layout.max_renderable_objects_per_env = 16
            config.scene_layout.max_lights_per_env = 0
            config.scene_layout.max_cameras_per_env = 0

        self.runtime = neo.Runtime()
        if not self.runtime.initialize(config):
            raise RuntimeError("Failed to initialize PSM soft-grasp runtime.")

        if self.enable_rgb_observation:
            self._initialize_rgb_observation_resources()

        self._psm_build = None
        self._tissue_entities: list[int] = []
        self._target_marker_entities: list[int] = []
        self._attachment_constraint_ids: list[int] = []
        self._target_marker_pose_slots: list[int] = []
        self._tooltip_body_entities: list[int] = []
        self._tooltip_local_anchors: list[tuple[float, float, float, float]] = []
        self._jaw_command_ranges: list[tuple[float, float]] = []
        self._rigid_reset_body_indices: list[int] = []
        self._rigid_reset_positions: list[tuple[float, float, float, float]] = []
        self._rigid_reset_orientations: list[tuple[float, float, float, float]] = []
        self._reset_joint_targets: list[list[float]] = []
        self._attachment_slot_indices: list[int] = []
        self._target_candidate_local_indices: list[int] = []

        self._author_scene(resolve_root=resolve_root, urdf_path=urdf_path)
        self.runtime.prepare()
        self._constraint_mapping = self.runtime.get_prepared_constraint_layout_mapping()
        self._particle_mapping = self.runtime.get_prepared_particle_layout_mapping()
        self._rigid_mapping = self.runtime.get_prepared_rigid_layout_mapping()
        self._joint_mapping = self.runtime.get_prepared_joint_layout_mapping()
        self._reset_positions = self._build_reset_positions(self.runtime.world())
        self._resolve_runtime_slots()
        if not self.runtime.upload_world():
            self.close()
            raise RuntimeError("Failed to upload prepared PSM soft-grasp world.")

        self._create_shared_buffers()
        self._populate_lookup_buffers()
        self._create_custom_passes()
        self.reset()

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
        target_desc.debug_name = "PsmSoftGrasp.RgbObservationTarget"
        self._rgb_render_target = self.runtime.create_render_target(target_desc)
        if not self.runtime.is_valid_render_target(self._rgb_render_target):
            raise RuntimeError("Failed to create PSM soft-grasp RGB render target.")

        resources = self.runtime.resources()
        self._ground_mesh = resources.register_mesh(
            neo.make_box_mesh(neo.Float3(1.25, 0.05, 1.25), "PsmSoftGrasp.GroundMesh")
        )
        self._target_mesh = resources.register_mesh(
            neo.make_sphere_mesh(0.05, 16, 12, "PsmSoftGrasp.TargetMesh")
        )
        self._tissue_mesh = resources.register_mesh(
            neo.make_box_mesh(
                neo.Float3(0.5 * self.tissue_width, 0.5 * self.tissue_thickness, 0.5 * self.tissue_height),
                "PsmSoftGrasp.TissueMesh",
            )
        )

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

    def _author_scene(self, *, resolve_root=None, urdf_path=None) -> None:
        world = self.runtime.world()
        self._psm_build = author_psm_scene(
            world,
            self.runtime.resources(),
            PsmAuthoringConfig(
                resolve_root=resolve_root,
                urdf_path=urdf_path,
                env_count=self.env_count,
                add_ground=False,
                add_default_lighting=False,
                add_default_camera=False,
                env_spacing=max(4.0, 0.75 * self.psm_scale),
                global_scale=self.psm_scale,
            ),
        )

        default_joint_targets = [0.0] * 6
        default_joint_targets[2] = self.DEFAULT_INITIAL_INSERTION * self.psm_scale
        for env_index, instance in enumerate(self._psm_build.instances):
            jaw_close_target = max(float(instance.jaw_limit[0]), self.DEFAULT_JAW_CLOSE_TARGET)
            jaw_open_target = min(float(instance.jaw_limit[1]), self.DEFAULT_JAW_OPEN_TARGET)
            if jaw_open_target < jaw_close_target:
                jaw_close_target = float(instance.jaw_limit[0])
                jaw_open_target = float(instance.jaw_limit[1])
            self._jaw_command_ranges.append((jaw_close_target, jaw_open_target))
            self._reset_joint_targets.append([*default_joint_targets, jaw_open_target])
            set_psm_joint_targets(
                world,
                self._psm_build,
                [*default_joint_targets, jaw_open_target],
                env_index=env_index,
            )

            gripper_a = world.try_get_transform(instance.link_entities["psm_tool_gripper1_link"])
            gripper_b = world.try_get_transform(instance.link_entities["psm_tool_gripper2_link"])
            tooltip_body_entity = instance.link_entities["psm_tool_yaw_link"]
            tooltip_body_transform = world.try_get_transform(tooltip_body_entity)
            if gripper_a is None or gripper_b is None or tooltip_body_transform is None:
                raise RuntimeError("Failed to resolve authored PSM tooltip transforms.")

            tooltip_midpoint = (
                0.5 * (float(gripper_a.world_transform.position.x) + float(gripper_b.world_transform.position.x)),
                0.5 * (float(gripper_a.world_transform.position.y) + float(gripper_b.world_transform.position.y)),
                0.5 * (float(gripper_a.world_transform.position.z) + float(gripper_b.world_transform.position.z)),
            )
            body_position = (
                float(tooltip_body_transform.world_transform.position.x),
                float(tooltip_body_transform.world_transform.position.y),
                float(tooltip_body_transform.world_transform.position.z),
            )
            body_rotation = _quat_tuple(tooltip_body_transform.world_transform.rotation)
            local_anchor = _quat_rotate(
                _quat_conjugate(body_rotation),
                (
                    tooltip_midpoint[0] - body_position[0],
                    tooltip_midpoint[1] - body_position[1],
                    tooltip_midpoint[2] - body_position[2],
                ),
            )
            self._tooltip_body_entities.append(tooltip_body_entity)
            self._tooltip_local_anchors.append(
                (local_anchor[0], local_anchor[1], local_anchor[2], 0.0)
            )

            tissue_center = neo.Float3(
                tooltip_midpoint[0] + self.DEFAULT_TISSUE_OFFSET_X,
                self.DEFAULT_TABLE_SURFACE_Y
                + 0.5 * self.tissue_thickness
                + self.DEFAULT_INITIAL_TISSUE_CLEARANCE,
                tooltip_midpoint[2] + self.DEFAULT_TISSUE_OFFSET_Z,
            )
            self._author_env_support(world, env_index, tissue_center)
            tissue_entity = self._author_tissue(world, env_index, tissue_center)
            self._tissue_entities.append(tissue_entity)
            if self.enable_target_marker:
                marker_entity = self._author_target_marker(world, env_index, tissue_center)
                self._target_marker_entities.append(marker_entity)
                self._target_marker_pose_slots.append(world.entity_pose_slot(marker_entity))
            else:
                self._target_marker_pose_slots.append(self.INVALID_INDEX)
            if self.enable_rgb_observation:
                self._author_rgb_camera(world, env_index, tissue_center)

            attachment = neo.AuthoredRigidParticleAttachmentConstraintState()
            attachment.constraint_id = 10000 + env_index
            attachment.rigid_body_entity_id = tooltip_body_entity
            attachment.particle = neo.AuthoredParticleReference()
            attachment.particle.entity_id = tissue_entity
            attachment.particle.type = neo.AuthoredParticleReferenceType.SoftBodyParticle
            attachment.particle.local_particle_index = 0
            attachment.local_anchor = neo.Float3(local_anchor[0], local_anchor[1], local_anchor[2])
            attachment.compliance = 0.0
            attachment.enabled = False
            if not world.upsert_rigid_particle_attachment_constraint(attachment):
                raise RuntimeError(f"Failed to author rigid-particle attachment for env {env_index}.")
            self._attachment_constraint_ids.append(attachment.constraint_id)

    def _author_env_support(self, world: neo.World, env_index: int, tissue_center: neo.Float3) -> None:
        ground_entity = world.create_entity(env_index)
        ground_transform = neo.TransformComponent()
        ground_transform.world_transform.position = neo.Float3(
            tissue_center.x,
            self.DEFAULT_TABLE_SURFACE_Y - 0.05,
            tissue_center.z,
        )
        world.set_transform(ground_entity, ground_transform)
        ground_body = neo.RigidBodyComponent()
        ground_body.body_type = neo.RigidBodyType.Static
        ground_body.inverse_mass = 0.0
        ground_body.simulated = True
        world.set_rigid_body(ground_entity, ground_body)
        ground_collider = neo.ColliderComponent()
        ground_collider.shape_type = neo.ColliderShapeType.Box
        ground_collider.shape_params = neo.Float4(1.25, 0.05, 1.25, 0.0)
        ground_collider.friction = 0.9
        ground_collider.static_friction = 1.1
        world.add_collider(ground_entity, ground_collider)
        if self.enable_rgb_observation:
            renderer = neo.MeshRendererComponent()
            renderer.mesh = self._ground_mesh
            renderer.material = self._make_material(
                f"PsmSoftGrasp.GroundMaterial.{env_index}",
                neo.Float3(0.60, 0.63, 0.68),
                0.9,
            )
            renderer.visible = True
            renderer.segmentation_id = 100 + env_index
            world.set_mesh_renderer(ground_entity, renderer)

    def _author_tissue(self, world: neo.World, env_index: int, tissue_center: neo.Float3) -> int:
        tissue_entity = world.create_entity(env_index)
        transform = neo.TransformComponent()
        transform.world_transform.position = tissue_center
        world.set_transform(tissue_entity, transform)
        if self.enable_rgb_observation:
            renderer = neo.MeshRendererComponent()
            renderer.mesh = self._tissue_mesh
            renderer.material = self._make_material(
                f"PsmSoftGrasp.TissueMaterial.{env_index}",
                neo.Float3(0.30, 0.68, 0.44),
                0.55,
            )
            renderer.visible = True
            renderer.segmentation_id = 200 + env_index
            world.set_mesh_renderer(tissue_entity, renderer)
        soft = neo.SoftBodyComponent()
        soft.source.kind = neo.SoftBodySourceKind.RegularGrid
        soft.source.regular_grid.size = neo.Float3(
            self.tissue_width, self.tissue_thickness, self.tissue_height
        )
        soft.source.regular_grid.target_particle_spacing = self.tissue_particle_spacing
        soft.particle_mass = 0.01
        soft.particle_radius = self.tissue_particle_radius
        # Keep the tissue visibly soft, but cohesive enough that a lifted particle
        # drags the surrounding patch instead of peeling away on its own.
        soft.edge_compliance = 4.5e-3
        soft.volume_compliance = 2.6e-3
        soft.material.contact.friction = 0.82
        soft.material.contact.static_friction = 1.05
        soft.material.contact.damping = 0.40
        soft.simulated = True
        soft.self_collision_enabled = True
        soft.collision_layer = 0x1
        soft.collision_mask = 0xFFFFFFFF
        if not world.set_soft_body(tissue_entity, soft):
            raise RuntimeError(f"Failed to author soft tissue for env {env_index}.")
        return tissue_entity

    def _author_target_marker(self, world: neo.World, env_index: int, tissue_center: neo.Float3) -> int:
        marker_entity = world.create_entity(env_index)
        transform = neo.TransformComponent()
        transform.world_transform.position = neo.Float3(
            tissue_center.x, tissue_center.y + 0.5 * self.tissue_thickness, tissue_center.z
        )
        world.set_transform(marker_entity, transform)
        if self.enable_rgb_observation:
            renderer = neo.MeshRendererComponent()
            renderer.mesh = self._target_mesh
            renderer.material = self._make_material(
                f"PsmSoftGrasp.TargetMaterial.{env_index}",
                neo.Float3(0.92, 0.22, 0.18),
                0.25,
            )
            renderer.visible = True
            renderer.segmentation_id = 300 + env_index
            world.set_mesh_renderer(marker_entity, renderer)
        return marker_entity

    def _author_rgb_camera(self, world: neo.World, env_index: int, tissue_center: neo.Float3) -> None:
        light_entity = world.create_entity(env_index)
        light = neo.DirectionalLightComponent()
        light.direction = neo.Float3(-0.35, -1.0, 0.25)
        light.color = neo.Float3(1.0, 1.0, 1.0)
        light.intensity = 7.5
        light.casts_shadows = True
        world.set_directional_light(light_entity, light)

        camera_entity = world.create_entity(env_index)
        camera_transform = neo.TransformComponent()
        camera_transform.world_transform.position = neo.Float3(
            tissue_center.x,
            tissue_center.y + 1.55,
            tissue_center.z - max(3.2, 0.32 * self.psm_scale),
        )
        tilt = neo.Quaternion()
        tilt_angle = math.radians(22.0)
        tilt.x = math.sin(tilt_angle * 0.5)
        tilt.y = 0.0
        tilt.z = 0.0
        tilt.w = math.cos(tilt_angle * 0.5)
        camera_transform.world_transform.rotation = tilt
        world.set_transform(camera_entity, camera_transform)

        camera = neo.CameraComponent()
        camera.product = neo.CameraProduct.ColorDepth
        camera.vertical_fov_degrees = 38.0
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
        for entity in self._tissue_entities:
            authoring_particles = world.try_get_soft_body_authoring_particles(entity)
            if authoring_particles is None:
                raise RuntimeError(f"Authoring particles were unavailable for soft body {entity}.")
            slot = slot_by_entity[entity]
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
        return reset_positions

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
        self._env_particle_offsets: list[int] = []
        self._env_particle_counts: list[int] = []
        self._arm_hinge_slots: list[list[int]] = []
        self._arm_slider_slots: list[int] = []
        self._jaw_hinge_slots: list[list[int]] = []
        self._tooltip_body_indices: list[int] = []

        particle_slot_by_entity = {
            entity_id: slot
            for slot, entity_id in enumerate(self._particle_mapping.soft_body_entity_ids)
        }

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
            tissue_slot = particle_slot_by_entity[self._tissue_entities[env_index]]
            self._env_particle_offsets.append(
                int(self._particle_mapping.soft_body_particle_offsets[tissue_slot])
            )
            self._env_particle_counts.append(
                int(self._particle_mapping.soft_body_particle_counts[tissue_slot])
            )

            arm_hinge_slots = [
                hinge_slot_by_id[int(instance.arm_joint_ids[0])],
                hinge_slot_by_id[int(instance.arm_joint_ids[1])],
                hinge_slot_by_id[int(instance.arm_joint_ids[3])],
                hinge_slot_by_id[int(instance.arm_joint_ids[4])],
                hinge_slot_by_id[int(instance.arm_joint_ids[5])],
            ]
            self._arm_hinge_slots.append(arm_hinge_slots)
            self._arm_slider_slots.append(slider_slot_by_id[int(instance.arm_joint_ids[2])])
            self._jaw_hinge_slots.append(
                [
                    hinge_slot_by_id[int(instance.jaw_joint_ids[0])],
                    hinge_slot_by_id[int(instance.jaw_joint_ids[1])],
                ]
            )
            tooltip_body_index = rigid_slot_by_entity[self._tooltip_body_entities[env_index]]
            self._tooltip_body_indices.append(tooltip_body_index)

            attachment_slot = None
            for slot in range(self._constraint_mapping.rigid_particle_attachments.count):
                if (
                    int(self._constraint_mapping.rigid_particle_attachments.environment_indices[slot]) == env_index
                    and int(self._constraint_mapping.rigid_particle_attachments.particle_entity_ids[slot])
                    == self._tissue_entities[env_index]
                    and int(self._constraint_mapping.rigid_particle_attachments.rigid_body_indices[slot])
                    == tooltip_body_index
                ):
                    attachment_slot = slot
                    break
            if attachment_slot is None:
                authored_id = self._attachment_constraint_ids[env_index]
                mapped_ids = list(self._constraint_mapping.rigid_particle_attachments.constraint_ids)
                raise RuntimeError(
                    "Failed to resolve rigid-particle attachment slot for "
                    f"env {env_index}. Authored id={authored_id}, prepared ids={mapped_ids}."
                )
            self._attachment_slot_indices.append(attachment_slot)

            for link_name in body_entity_order:
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

        self._rigid_body_reset_count = len(body_entity_order)
        self._target_candidate_local_indices = self._build_target_candidate_local_indices(world)

    def _build_target_candidate_local_indices(self, world: neo.World) -> list[int]:
        if not self._tissue_entities:
            return [0]
        authoring_particles = world.try_get_soft_body_authoring_particles(self._tissue_entities[0])
        center_transform = world.try_get_transform(self._tissue_entities[0])
        if authoring_particles is None or center_transform is None:
            raise RuntimeError("Failed to resolve tissue authoring particles for target selection.")
        center = center_transform.world_transform.position
        local_positions = []
        max_y = -1.0e30
        for local_index, position in enumerate(authoring_particles.rest_positions):
            local_x = float(position.x) - float(center.x)
            local_y = float(position.y) - float(center.y)
            local_z = float(position.z) - float(center.z)
            max_y = max(max_y, local_y)
            local_positions.append((local_index, local_x, local_y, local_z))
        top_positions = [entry for entry in local_positions if abs(entry[2] - max_y) <= 0.5 * self.tissue_particle_spacing + 1.0e-5]
        if not top_positions:
            return [0]

        if self.target_sampling_fraction_x <= 1.0e-6 and self.target_sampling_fraction_z <= 1.0e-6:
            target_corner_x = -0.5 * self.tissue_width + self.tissue_particle_spacing
            target_corner_z = -0.5 * self.tissue_height + self.tissue_particle_spacing
            best = min(
                top_positions,
                key=lambda entry: (entry[1] - target_corner_x) * (entry[1] - target_corner_x)
                + (entry[3] - target_corner_z) * (entry[3] - target_corner_z),
            )
            return [int(best[0])]

        x_limit = 0.5 * self.tissue_width * self.target_sampling_fraction_x
        z_limit = 0.5 * self.tissue_height * self.target_sampling_fraction_z
        candidates = [
            int(local_index)
            for local_index, local_x, _local_y, local_z in top_positions
            if abs(local_x) <= x_limit and abs(local_z) <= z_limit
        ]
        if candidates:
            return candidates
        best = min(top_positions, key=lambda entry: entry[1] * entry[1] + entry[3] * entry[3])
        return [int(best[0])]

    def _create_shared_buffers(self) -> None:
        self.action_buffer, self.action_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.Actions",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.observation_buffer, observation_flat = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.Observations",
            self.env_count * self.OBSERVATION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.observation_tensor = observation_flat.view(self.env_count, self.OBSERVATION_DIM)
        self.reward_buffer, self.reward_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.Rewards", self.env_count, neo.SharedBufferTensorDTypeCode.Float
        )
        self.terminated_buffer, self.terminated_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.Terminated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.truncated_buffer, self.truncated_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.Truncated", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.episode_steps_buffer, self.episode_steps_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.EpisodeSteps", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_mask_buffer, self.reset_mask_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.ResetMask", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.target_choice_buffer, self.target_choice_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.TargetChoice", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.previous_grasp_closed_buffer, self.previous_grasp_closed_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.PreviousGraspClosed", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.env_metadata_u32_buffer, self.env_metadata_u32_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.EnvMetadataU32",
            self.env_count * 14,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, 14],
        )
        self.env_metadata_f32_buffer, self.env_metadata_f32_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.EnvMetadataF32",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.env_particle_offsets_buffer, self.env_particle_offsets_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.ParticleOffsets", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.env_particle_counts_buffer, self.env_particle_counts_tensor = self._register_shared_buffer(
            self.runtime, "PsmSoftGrasp.ParticleCounts", self.env_count, neo.SharedBufferTensorDTypeCode.UInt
        )
        self.reset_positions_buffer, self.reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.ResetPositions",
            self._particle_mapping.particle_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self._particle_mapping.particle_count, 4],
        )
        self.arm_hinge_slots_buffer, self.arm_hinge_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.ArmHingeSlots",
            self.env_count * 5,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, 5],
        )
        self.arm_slider_slots_buffer, self.arm_slider_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.ArmSliderSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.jaw_hinge_slots_buffer, self.jaw_hinge_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.JawHingeSlots",
            self.env_count * 2,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, 2],
        )
        self.attachment_slots_buffer, self.attachment_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.AttachmentSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.target_pose_slots_buffer, self.target_pose_slots_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TargetPoseSlots",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.tooltip_body_indices_buffer, self.tooltip_body_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TooltipBodyIndices",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.tooltip_local_anchors_buffer, self.tooltip_local_anchors_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TooltipLocalAnchors",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, 4],
        )
        self.target_particle_indices_buffer, self.target_particle_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TargetParticleIndices",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.target_rest_heights_buffer, self.target_rest_heights_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TargetRestHeights",
            self.env_count,
            neo.SharedBufferTensorDTypeCode.Float,
        )
        self.candidate_local_indices_buffer, self.candidate_local_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.TargetCandidateLocalIndices",
            len(self._target_candidate_local_indices),
            neo.SharedBufferTensorDTypeCode.UInt,
        )
        self.current_joint_targets_buffer, self.current_joint_targets_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.CurrentJointTargets",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.reset_joint_targets_buffer, self.reset_joint_targets_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.ResetJointTargets",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            shape=[self.env_count, self.ACTION_DIM],
        )
        self.joint_limits_buffer, self.joint_limits_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.JointLimits",
            self.env_count * self.ACTION_DIM,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=8,
            shape=[self.env_count, self.ACTION_DIM, 2],
        )
        self.rigid_reset_body_indices_buffer, self.rigid_reset_body_indices_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.RigidResetBodyIndices",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.UInt,
            shape=[self.env_count, self._rigid_body_reset_count],
        )
        self.rigid_reset_positions_buffer, self.rigid_reset_positions_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.RigidResetPositions",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self._rigid_body_reset_count, 4],
        )
        self.rigid_reset_orientations_buffer, self.rigid_reset_orientations_tensor = self._register_shared_buffer(
            self.runtime,
            "PsmSoftGrasp.RigidResetOrientations",
            self.env_count * self._rigid_body_reset_count,
            neo.SharedBufferTensorDTypeCode.Float,
            element_stride_bytes=16,
            shape=[self.env_count, self._rigid_body_reset_count, 4],
        )
        if self.enable_rgb_observation:
            self.rgb_observation_buffer, self.rgb_observation_tensor = self._register_shared_buffer(
                self.runtime,
                "PsmSoftGrasp.RgbObservation",
                self.env_count * self.image_width * self.image_height,
                neo.SharedBufferTensorDTypeCode.Float,
                element_stride_bytes=16,
                shape=[self.env_count, self.image_height, self.image_width, 4],
            )

    def _populate_lookup_buffers(self) -> None:
        device = self.action_tensor.device
        self.action_tensor.zero_()
        self.observation_tensor.zero_()
        self.reward_tensor.zero_()
        self.terminated_tensor.zero_()
        self.truncated_tensor.zero_()
        self.episode_steps_tensor.zero_()
        self.reset_mask_tensor.zero_()
        self.target_choice_tensor.zero_()
        self.previous_grasp_closed_tensor.zero_()
        self.env_metadata_u32_tensor.zero_()
        self.env_metadata_f32_tensor.zero_()

        self.env_particle_offsets_tensor.copy_(
            torch.tensor(self._env_particle_offsets, device=device, dtype=self.env_particle_offsets_tensor.dtype)
        )
        self.env_particle_counts_tensor.copy_(
            torch.tensor(self._env_particle_counts, device=device, dtype=self.env_particle_counts_tensor.dtype)
        )
        self.reset_positions_tensor.copy_(
            torch.tensor(self._reset_positions, device=device, dtype=self.reset_positions_tensor.dtype)
        )
        self.arm_hinge_slots_tensor.copy_(
            torch.tensor(self._arm_hinge_slots, device=device, dtype=self.arm_hinge_slots_tensor.dtype)
        )
        self.arm_slider_slots_tensor.copy_(
            torch.tensor(self._arm_slider_slots, device=device, dtype=self.arm_slider_slots_tensor.dtype)
        )
        self.jaw_hinge_slots_tensor.copy_(
            torch.tensor(self._jaw_hinge_slots, device=device, dtype=self.jaw_hinge_slots_tensor.dtype)
        )
        self.attachment_slots_tensor.copy_(
            torch.tensor(self._attachment_slot_indices, device=device, dtype=self.attachment_slots_tensor.dtype)
        )
        self.target_pose_slots_tensor.copy_(
            torch.tensor(self._target_marker_pose_slots, device=device, dtype=self.target_pose_slots_tensor.dtype)
        )
        self.tooltip_body_indices_tensor.copy_(
            torch.tensor(self._tooltip_body_indices, device=device, dtype=self.tooltip_body_indices_tensor.dtype)
        )
        self.tooltip_local_anchors_tensor.copy_(
            torch.tensor(self._tooltip_local_anchors, device=device, dtype=self.tooltip_local_anchors_tensor.dtype)
        )
        self.env_metadata_u32_tensor[:, 0:5].copy_(self.arm_hinge_slots_tensor)
        self.env_metadata_u32_tensor[:, 5].copy_(self.arm_slider_slots_tensor)
        self.env_metadata_u32_tensor[:, 6:8].copy_(self.jaw_hinge_slots_tensor)
        self.env_metadata_u32_tensor[:, 8].copy_(self.env_particle_offsets_tensor)
        self.env_metadata_u32_tensor[:, 9].copy_(self.env_particle_counts_tensor)
        self.env_metadata_u32_tensor[:, 10].copy_(self.attachment_slots_tensor)
        self.env_metadata_u32_tensor[:, 11].copy_(self.tooltip_body_indices_tensor)
        self.env_metadata_u32_tensor[:, 12].copy_(self.target_pose_slots_tensor)
        self.env_metadata_f32_tensor[:, 0:3].copy_(self.tooltip_local_anchors_tensor[:, 0:3])
        self.candidate_local_indices_tensor.copy_(
            torch.tensor(
                self._target_candidate_local_indices,
                device=device,
                dtype=self.candidate_local_indices_tensor.dtype,
            )
        )
        self.current_joint_targets_tensor.copy_(
            torch.tensor(self._reset_joint_targets, device=device, dtype=self.current_joint_targets_tensor.dtype)
        )
        self.reset_joint_targets_tensor.copy_(
            torch.tensor(self._reset_joint_targets, device=device, dtype=self.reset_joint_targets_tensor.dtype)
        )

        joint_limits = []
        for instance, jaw_limits in zip(self._psm_build.instances, self._jaw_command_ranges):
            joint_limits.append(
                [
                    (float(instance.arm_joint_limits[0][0]), float(instance.arm_joint_limits[0][1])),
                    (float(instance.arm_joint_limits[1][0]), float(instance.arm_joint_limits[1][1])),
                    (float(instance.arm_joint_limits[2][0]), float(instance.arm_joint_limits[2][1])),
                    (float(instance.arm_joint_limits[3][0]), float(instance.arm_joint_limits[3][1])),
                    (float(instance.arm_joint_limits[4][0]), float(instance.arm_joint_limits[4][1])),
                    (float(instance.arm_joint_limits[5][0]), float(instance.arm_joint_limits[5][1])),
                    jaw_limits,
                ]
            )
        self.joint_limits_tensor.copy_(
            torch.tensor(joint_limits, device=device, dtype=self.joint_limits_tensor.dtype)
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
        self.target_particle_indices_tensor.zero_()
        self.target_rest_heights_tensor.zero_()

        handles = [
            self.action_buffer,
            self.observation_buffer,
            self.reward_buffer,
            self.terminated_buffer,
            self.truncated_buffer,
            self.episode_steps_buffer,
            self.reset_mask_buffer,
            self.target_choice_buffer,
            self.previous_grasp_closed_buffer,
            self.env_metadata_u32_buffer,
            self.env_metadata_f32_buffer,
            self.env_particle_offsets_buffer,
            self.env_particle_counts_buffer,
            self.reset_positions_buffer,
            self.arm_hinge_slots_buffer,
            self.arm_slider_slots_buffer,
            self.jaw_hinge_slots_buffer,
            self.attachment_slots_buffer,
            self.target_pose_slots_buffer,
            self.tooltip_body_indices_buffer,
            self.tooltip_local_anchors_buffer,
            self.target_particle_indices_buffer,
            self.target_rest_heights_buffer,
            self.candidate_local_indices_buffer,
            self.current_joint_targets_buffer,
            self.reset_joint_targets_buffer,
            self.joint_limits_buffer,
            self.rigid_reset_body_indices_buffer,
            self.rigid_reset_positions_buffer,
            self.rigid_reset_orientations_buffer,
        ]
        if self.enable_rgb_observation:
            self.rgb_observation_tensor.zero_()
            handles.append(self.rgb_observation_buffer)
        self._sync_from_cuda(self.runtime, handles)

    def _create_custom_passes(self) -> None:
        pre_desc = neo.CustomComputePassDesc()
        pre_desc.debug_name = "PsmSoftGrasp.PrePhysicsControl"
        pre_desc.shader_source = _PSM_SOFT_GRASP_PRE_PHYSICS_SHADER
        pre_desc.thread_group_size_x = 64
        _bind_resources(
            pre_desc,
            [
                ("g_Actions", self.action_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CurrentJointTargets", self.current_joint_targets_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_JointLimits", self.joint_limits_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_PreviousGraspClosed", self.previous_grasp_closed_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_HingeJoints", None, "joint.hinge", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_SliderJoints", None, "joint.slider", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Attachments", None, "constraint.rigid_particle_attachments", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        pre_desc.constant_buffer_variable_name = "PsmSoftGraspPrePhysicsConstants"
        pre_desc.constant_buffer_size_bytes = 16
        pre_desc.constant_data = list(
            struct.pack(
                "<4f",
                self.rotational_action_scale,
                self.insertion_action_scale,
                self.tooltip_proximity_threshold,
                0.0,
            )
        )
        pre_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        pre_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._pre_pass = self._register_custom_pass(self.runtime, pre_desc)

        post_desc = neo.CustomComputePassDesc()
        post_desc.debug_name = "PsmSoftGrasp.PostPhysicsObservations"
        post_desc.shader_source = _PSM_SOFT_GRASP_POST_PHYSICS_SHADER
        post_desc.thread_group_size_x = 64
        _bind_resources(
            post_desc,
            [
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_HingeJointRuntimeStates", None, "joint.hinge_runtime", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_SliderJointRuntimeStates", None, "joint.slider_runtime", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_Attachments", None, "constraint.rigid_particle_attachments", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        post_desc.constant_buffer_variable_name = "PsmSoftGraspPostPhysicsConstants"
        post_desc.constant_buffer_size_bytes = 16
        post_desc.constant_data = list(
            struct.pack(
                "<fIff",
                self.lift_target_distance,
                self.max_episode_steps,
                1.5,
                0.0,
            )
        )
        post_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        post_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._post_pass = self._register_custom_pass(self.runtime, post_desc)

        reset_particles_desc = neo.CustomComputePassDesc()
        reset_particles_desc.debug_name = "PsmSoftGrasp.ResetParticles"
        reset_particles_desc.shader_source = _PSM_SOFT_GRASP_RESET_PARTICLES_SHADER
        reset_particles_desc.thread_group_size_x = 64
        _bind_resources(
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

        reset_rigid_desc = neo.CustomComputePassDesc()
        reset_rigid_desc.debug_name = "PsmSoftGrasp.ResetRigid"
        reset_rigid_desc.shader_source = _PSM_SOFT_GRASP_RESET_RIGID_SHADER
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
        reset_rigid_desc.constant_buffer_variable_name = "PsmSoftGraspResetRigidConstants"
        reset_rigid_desc.constant_buffer_size_bytes = 16
        reset_rigid_desc.constant_data = list(
            struct.pack("<I3f", self._rigid_body_reset_count, 0.0, 0.0, 0.0)
        )
        reset_rigid_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_rigid_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_rigid_pass = self._register_custom_pass(self.runtime, reset_rigid_desc)

        reset_outputs_desc = neo.CustomComputePassDesc()
        reset_outputs_desc.debug_name = "PsmSoftGrasp.ResetOutputs"
        reset_outputs_desc.shader_source = _PSM_SOFT_GRASP_RESET_OUTPUTS_SHADER
        reset_outputs_desc.thread_group_size_x = 64
        _bind_resources(
            reset_outputs_desc,
            [
                ("g_ResetMask", self.reset_mask_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_TargetChoice", self.target_choice_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_CandidateLocalIndices", self.candidate_local_indices_buffer, "", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_EnvMetadataU32", self.env_metadata_u32_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EnvMetadataF32", self.env_metadata_f32_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_ParticlePositionsInvMass", None, "particle.positions_inv_mass", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidPositionsInvMass", None, "rigid.positions", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_RigidOrientations", None, "rigid.orientations", neo.CustomComputeResourceAccess.ReadOnly),
                ("g_Attachments", None, "constraint.rigid_particle_attachments", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EntityPositions", None, "entity.positions", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Observations", self.observation_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Rewards", self.reward_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Terminated", self.terminated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_Truncated", self.truncated_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_EpisodeSteps", self.episode_steps_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
                ("g_PreviousGraspClosed", self.previous_grasp_closed_buffer, "", neo.CustomComputeResourceAccess.ReadWrite),
            ],
        )
        reset_outputs_desc.dispatch.mode = neo.CustomComputeDispatchMode.ExplicitGroupCount
        reset_outputs_desc.dispatch.group_count_x = (self.env_count + 63) // 64
        self._reset_outputs_pass = self._register_custom_pass(self.runtime, reset_outputs_desc)

        if self.enable_rgb_observation:
            render_desc = neo.CustomComputePassDesc()
            render_desc.debug_name = "PsmSoftGrasp.RgbObservation"
            render_desc.shader_source = _PSM_SOFT_GRASP_RGB_SHADER
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
            env_indices = torch.arange(self.env_count, device=self.action_tensor.device, dtype=torch.int64)
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.to(device=self.action_tensor.device, dtype=torch.int64)
        else:
            env_indices = torch.tensor(list(env_ids), device=self.action_tensor.device, dtype=torch.int64)
        if env_indices.numel() == 0:
            return self.observation_tensor

        self.reset_mask_tensor.zero_()
        self.target_choice_tensor.zero_()
        for env_index in env_indices.tolist():
            self.reset_mask_tensor[int(env_index)] = 1
        if len(self._target_candidate_local_indices) > 1:
            sampled = torch.randint(
                0,
                len(self._target_candidate_local_indices),
                (env_indices.numel(),),
                device=self.target_choice_tensor.device,
                dtype=self.target_choice_tensor.dtype,
            )
            self.target_choice_tensor.index_copy_(0, env_indices, sampled)
        self.action_tensor.zero_()
        self.current_joint_targets_tensor.copy_(self.reset_joint_targets_tensor)
        self._sync_from_cuda(
            self.runtime,
            [
                self.reset_mask_buffer,
                self.target_choice_buffer,
                self.action_buffer,
                self.current_joint_targets_buffer,
            ],
        )
        if not self.runtime.execute_custom_compute_pass(self._reset_particles_pass):
            raise RuntimeError("Failed to execute PSM soft-grasp particle reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_rigid_pass):
            raise RuntimeError("Failed to execute PSM soft-grasp rigid reset pass.")
        if not self.runtime.execute_custom_compute_pass(self._reset_outputs_pass):
            raise RuntimeError("Failed to execute PSM soft-grasp output reset pass.")
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
            raise RuntimeError("Failed to execute PSM soft-grasp pre-physics pass.")
        if not self.runtime.step_physics(self._frame):
            raise RuntimeError("PSM soft-grasp physics step failed.")
        if not self.runtime.execute_custom_compute_pass(self._post_pass):
            raise RuntimeError("Failed to execute PSM soft-grasp post-physics pass.")
        self._sync_outputs_to_cuda()
        self._end_frame(self.runtime, advance=True)
        return (
            self.observation_tensor,
            self.reward_tensor,
            self.terminated_tensor,
            self.truncated_tensor,
        )

    def render(self) -> "torch.Tensor":
        if not self.enable_rgb_observation:
            raise RuntimeError("RGB observations were not enabled for this PSM soft-grasp env.")
        self.runtime.step_visual_sensors(self._frame)
        if not self.runtime.execute_custom_compute_pass(self._rgb_render_pass):
            raise RuntimeError("Failed to execute PSM soft-grasp RGB observation pass.")
        if not self.runtime.sync_shared_buffer_to_cuda(self.rgb_observation_buffer):
            raise RuntimeError("Failed to synchronize PSM soft-grasp RGB observation buffer to CUDA.")
        self.runtime.end_frame(self._frame)
        torch.cuda.synchronize(device=self.rgb_observation_tensor.device)
        return self.rgb_observation_tensor

    def close(self) -> None:
        self.close_runtime(getattr(self, "runtime", None))
        self.runtime = None


__all__ = ["PsmSoftGraspTorchVectorEnv"]
