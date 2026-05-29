#include "physics/physics_world.h"

#include "common/logger.h"
#include "common/math_utils_runtime.h"
#include "physics/particle_phase.h"
#include "physics/soft_body_authoring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kBroadPhaseContributionNone   = 0u;
constexpr std::uint32_t kBroadPhaseContributionMoving = 1u;
constexpr std::uint32_t kBroadPhaseContributionStatic = 2u;

Diligent::float4 toPositionInvMass(const RigidBodyState &state)
{
    return Diligent::float4{state.position.x, state.position.y, state.position.z,
                            state.inverseMass};
}

Diligent::float4 toOrientation(const RigidBodyState &state)
{
    return Diligent::float4{state.rotation.q.x, state.rotation.q.y, state.rotation.q.z,
                            state.rotation.q.w};
}

Diligent::float4 toLinearVelocity(const RigidBodyState &state)
{
    return Diligent::float4{state.linearVelocity.x, state.linearVelocity.y, state.linearVelocity.z,
                            0.0f};
}

Diligent::float4 toScale(const RigidBodyState &state)
{
    return Diligent::float4{state.scale.x, state.scale.y, state.scale.z, 0.0f};
}

Diligent::float4 toAngularVelocity(const RigidBodyState &state)
{
    return Diligent::float4{state.angularVelocity.x, state.angularVelocity.y,
                            state.angularVelocity.z, 0.0f};
}

Diligent::float4 toInverseInertiaLocal(const RigidBodyState &state)
{
    return Diligent::float4{state.inverseInertiaLocal.x, state.inverseInertiaLocal.y,
                            state.inverseInertiaLocal.z, 0.0f};
}

Diligent::float4 toKinematicTargetPosition(const RigidBodyState &state)
{
    return Diligent::float4{state.kinematicTargetPosition.x, state.kinematicTargetPosition.y,
                            state.kinematicTargetPosition.z, 0.0f};
}

Diligent::float4 toKinematicTargetOrientation(const RigidBodyState &state)
{
    return Diligent::float4{state.kinematicTargetRotation.q.x, state.kinematicTargetRotation.q.y,
                            state.kinematicTargetRotation.q.z, state.kinematicTargetRotation.q.w};
}

Diligent::float4 toColliderLocalPosition(const ColliderState &state)
{
    return Diligent::float4{state.localPosition.x, state.localPosition.y, state.localPosition.z,
                            0.0f};
}

Diligent::float4 toColliderLocalOrientation(const ColliderState &state)
{
    return Diligent::float4{state.localRotation.q.x, state.localRotation.q.y,
                            state.localRotation.q.z, state.localRotation.q.w};
}

Diligent::float4 toColliderMaterial(const ColliderState &state)
{
    return Diligent::float4{state.friction, state.restitution, 0.0f, state.staticFriction};
}

Diligent::float4 toParticleContactMaterial(const ParticleContactMaterialDesc &material)
{
    return Diligent::float4{material.friction, material.restitution, material.damping,
                            material.staticFriction};
}

constexpr float kFluidKernelPi                  = 3.14159265358979323846f;
constexpr float kFluidContactDistanceMultiplier = 2.0f / 0.6f;

float fluidSpikyWeight(float distance, float supportRadius) noexcept
{
    const float oneMinusQ = 1.0f - distance / supportRadius;
    const float k = 15.0f / (kFluidKernelPi * supportRadius * supportRadius * supportRadius);
    return k * oneMinusQ * oneMinusQ;
}

float fluidSpikyWeightDerivative(float distance, float supportRadius) noexcept
{
    const float oneMinusQ = 1.0f - distance / supportRadius;
    const float k = 15.0f / (kFluidKernelPi * supportRadius * supportRadius * supportRadius);
    return -k * 2.0f * oneMinusQ / supportRadius;
}

std::uint32_t tightPack3D(float radius, float separation,
                          std::array<Diligent::float3, 2048> &points) noexcept
{
    const int dim        = static_cast<int>(std::ceil(radius / separation));
    std::uint32_t count  = 0u;
    const float rowScale = std::sqrt(0.75f) * separation;

    for (int z = -dim; z <= dim; ++z)
    {
        for (int y = -dim; y <= dim; ++y)
        {
            for (int x = -dim; x <= dim; ++x)
            {
                const float xpos = x * separation + (((y + z) & 1) != 0 ? separation * 0.5f : 0.0f);
                const Diligent::float3 sample{xpos, y * rowScale, z * rowScale};
                const float magnitudeSq =
                    sample.x * sample.x + sample.y * sample.y + sample.z * sample.z;
                if (magnitudeSq == 0.0f)
                {
                    continue;
                }

                if (std::sqrt(magnitudeSq) <= radius)
                {
                    if (count == points.size())
                    {
                        return count;
                    }
                    points[count++] = sample;
                }
            }
        }
    }

    return count;
}

void calculateFluidSolveScales(float restDistance, float supportRadius, float &restDensity,
                               float &densityConstraintScale,
                               float &surfaceConstraintScale) noexcept
{
    std::array<Diligent::float3, 2048> samples{};
    const std::uint32_t count = tightPack3D(supportRadius, restDistance, samples);

    restDensity            = 0.0f;
    densityConstraintScale = 0.0f;
    float a                = 0.0f;
    float b                = 0.0f;

    for (std::uint32_t i = 0u; i < count; ++i)
    {
        const Diligent::float3 sample = samples[i];
        const float distance =
            std::sqrt(sample.x * sample.x + sample.y * sample.y + sample.z * sample.z);

        restDensity += fluidSpikyWeight(distance, supportRadius);
        const float derivative = fluidSpikyWeightDerivative(distance, supportRadius);
        densityConstraintScale += derivative * derivative;

        if (sample.y <= 0.0f && distance > 0.0f)
        {
            const float cosTheta = sample.y / distance;
            a += derivative * cosTheta;
            b -= distance * cosTheta;
        }
    }

    if (restDensity <= 1.0e-6f)
    {
        restDensity = 1.0f;
    }
    if (densityConstraintScale <= 1.0e-6f)
    {
        densityConstraintScale = 1.0f;
    }
    surfaceConstraintScale =
        (std::abs(b) > 1.0e-6f && std::abs(a) > 1.0e-6f) ? std::abs(a / b) : 1.0f;
}

FluidMaterialGpu toFluidSolverMaterial(const FluidMaterialDesc &material,
                                       float particleRadius) noexcept
{
    const float fluidRestDistance = std::max(2.0f * particleRadius, 1.0e-4f);
    // PhysX-like shared density model: support/contact distance is derived from spacing
    // rather than taken directly from the authored material.
    const float smoothingRadius =
        std::max(kFluidContactDistanceMultiplier * particleRadius, fluidRestDistance);
    float restDensity            = 0.0f;
    float densityConstraintScale = 0.0f;
    float surfaceConstraintScale = 0.0f;
    calculateFluidSolveScales(fluidRestDistance, smoothingRadius, restDensity,
                              densityConstraintScale, surfaceConstraintScale);

    const float invRestDensity = 1.0f / restDensity;
    const float restRatio      = std::clamp(fluidRestDistance / smoothingRadius, 1.0e-3f, 0.999f);
    const float restRatioSq    = restRatio * restRatio;
    const float cohesion1      = -(1.0f + restRatio) / restRatioSq;
    const float cohesion2      = (restRatioSq + restRatio + 1.0f) / restRatioSq;

    return FluidMaterialGpu{restDensity,
                            invRestDensity,
                            smoothingRadius,
                            1.0f / std::max(densityConstraintScale, 1.0e-6f),
                            material.viscosity * invRestDensity,
                            material.cohesion * smoothingRadius,
                            cohesion1,
                            cohesion2,
                            (material.surfaceTension * invRestDensity) / surfaceConstraintScale,
                            material.vorticityConfinement * invRestDensity,
                            material.gravityScale,
                            material.cflCoefficient * smoothingRadius};
}

bool nearlyEqual(float a, float b, float epsilon = 1.0e-4f) noexcept
{
    const float scale = std::max({1.0f, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= epsilon * scale;
}

void normalizeParticleContactMaterial(ParticleContactMaterialDesc &material) noexcept
{
    material.friction       = std::max(material.friction, 0.0f);
    material.staticFriction = material.staticFriction < 0.0f
                                  ? material.friction
                                  : std::max(material.staticFriction, 0.0f);
    material.restitution    = std::clamp(material.restitution, 0.0f, 1.0f);
    material.damping        = std::max(material.damping, 0.0f);
}

void normalizeFluidMaterial(FluidMaterialDesc &material) noexcept
{
    normalizeParticleContactMaterial(material.contact);
    material.viscosity            = std::max(material.viscosity, 0.0f);
    material.cohesion             = std::max(material.cohesion, 0.0f);
    material.surfaceTension       = std::max(material.surfaceTension, 0.0f);
    material.vorticityConfinement = std::max(material.vorticityConfinement, 0.0f);
    material.gravityScale         = std::max(material.gravityScale, 0.0f);
    material.cflCoefficient       = std::max(material.cflCoefficient, 0.0f);
}

std::uint32_t findOrAppendParticleContactMaterial(std::vector<Diligent::float4> &materials,
                                                  const Diligent::float4 &material) noexcept
{
    for (std::uint32_t materialIndex = 0u;
         materialIndex < static_cast<std::uint32_t>(materials.size()); ++materialIndex)
    {
        if (materials[materialIndex] == material)
        {
            return materialIndex;
        }
    }

    materials.push_back(material);
    return static_cast<std::uint32_t>(materials.size() - 1u);
}

std::uint32_t findOrAppendFluidMaterial(std::vector<FluidMaterialGpu> &materials,
                                        const FluidMaterialGpu &material) noexcept
{
    for (std::uint32_t materialIndex = 0u;
         materialIndex < static_cast<std::uint32_t>(materials.size()); ++materialIndex)
    {
        if (materials[materialIndex] == material)
        {
            return materialIndex;
        }
    }

    materials.push_back(material);
    return static_cast<std::uint32_t>(materials.size() - 1u);
}

Diligent::float4 toFloat4(const Diligent::float3 &value, float w = 0.0f) noexcept
{
    return Diligent::float4{value.x, value.y, value.z, w};
}

Diligent::float3 safeNormalize(const Diligent::float3 &value,
                               const Diligent::float3 &fallback) noexcept
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= 1.0e-8f)
    {
        return fallback;
    }
    return value / std::sqrt(lengthSq);
}

Diligent::float3 quaternionRotate(const Diligent::QuaternionF &q,
                                  const Diligent::float3 &v) noexcept
{
    const Diligent::float3 qv{q.q.x, q.q.y, q.q.z};
    const Diligent::float3 t = 2.0f * Diligent::cross(qv, v);
    return v + q.q.w * t + Diligent::cross(qv, t);
}

Diligent::float3 quaternionInverseRotate(const Diligent::QuaternionF &q,
                                         const Diligent::float3 &v) noexcept
{
    const Diligent::QuaternionF conjugate{-q.q.x, -q.q.y, -q.q.z, q.q.w};
    return quaternionRotate(conjugate, v);
}

Diligent::QuaternionF quaternionConjugate(const Diligent::QuaternionF &q) noexcept
{
    return Diligent::QuaternionF{-q.q.x, -q.q.y, -q.q.z, q.q.w};
}

Diligent::QuaternionF quaternionMultiply(const Diligent::QuaternionF &a,
                                         const Diligent::QuaternionF &b) noexcept
{
    return Diligent::QuaternionF{a.q.w * b.q.x + a.q.x * b.q.w + a.q.y * b.q.z - a.q.z * b.q.y,
                                 a.q.w * b.q.y - a.q.x * b.q.z + a.q.y * b.q.w + a.q.z * b.q.x,
                                 a.q.w * b.q.z + a.q.x * b.q.y - a.q.y * b.q.x + a.q.z * b.q.w,
                                 a.q.w * b.q.w - a.q.x * b.q.x - a.q.y * b.q.y - a.q.z * b.q.z};
}

void computeMatrixQ(const Diligent::QuaternionF &q, float outQ[4][4]) noexcept
{
    outQ[0][0] = q.q.w;
    outQ[0][1] = -q.q.x;
    outQ[0][2] = -q.q.y;
    outQ[0][3] = -q.q.z;
    outQ[1][0] = q.q.x;
    outQ[1][1] = q.q.w;
    outQ[1][2] = -q.q.z;
    outQ[1][3] = q.q.y;
    outQ[2][0] = q.q.y;
    outQ[2][1] = q.q.z;
    outQ[2][2] = q.q.w;
    outQ[2][3] = -q.q.x;
    outQ[3][0] = q.q.z;
    outQ[3][1] = -q.q.y;
    outQ[3][2] = q.q.x;
    outQ[3][3] = q.q.w;
}

void computeMatrixQHat(const Diligent::QuaternionF &q, float outQ[4][4]) noexcept
{
    outQ[0][0] = q.q.w;
    outQ[0][1] = -q.q.x;
    outQ[0][2] = -q.q.y;
    outQ[0][3] = -q.q.z;
    outQ[1][0] = q.q.x;
    outQ[1][1] = q.q.w;
    outQ[1][2] = q.q.z;
    outQ[1][3] = -q.q.y;
    outQ[2][0] = q.q.y;
    outQ[2][1] = -q.q.z;
    outQ[2][2] = q.q.w;
    outQ[2][3] = q.q.x;
    outQ[3][0] = q.q.z;
    outQ[3][1] = q.q.y;
    outQ[3][2] = -q.q.x;
    outQ[3][3] = q.q.w;
}

void computeProjectionRowsFromLocalFrames(const Diligent::QuaternionF &localRotationA,
                                          const Diligent::QuaternionF &localRotationB,
                                          std::uint32_t rowStart, std::uint32_t rowCount,
                                          Diligent::float4 *rowsOut) noexcept
{
    const Diligent::QuaternionF q00 = quaternionConjugate(localRotationA);
    const Diligent::QuaternionF q10 = quaternionConjugate(localRotationB);

    float Q[4][4];
    float QHat[4][4];
    computeMatrixQ(q00, Q);
    computeMatrixQHat(q10, QHat);

    for (std::uint32_t row = 0u; row < rowCount; ++row)
    {
        const std::uint32_t srcRow = rowStart + row;
        Diligent::float4 outRow{};
        for (std::uint32_t col = 0u; col < 4u; ++col)
        {
            outRow[col] = QHat[0][srcRow] * Q[0][col] + QHat[1][srcRow] * Q[1][col] +
                          QHat[2][srcRow] * Q[2][col] + QHat[3][srcRow] * Q[3][col];
        }
        rowsOut[row] = outRow;
    }
}

} // namespace

namespace
{

void enqueueDirtyIndex(std::uint32_t index, std::vector<std::uint32_t> &dirtyIndices,
                       std::vector<std::uint8_t> &dirtyBits) noexcept
{
    if (index >= dirtyBits.size())
    {
        dirtyBits.resize(index + 1u, 0u);
    }
    if (dirtyBits[index] != 0u)
    {
        return;
    }

    dirtyBits[index] = 1u;
    dirtyIndices.push_back(index);
}

void clearDirtyIndices(std::vector<std::uint32_t> &dirtyIndices,
                       std::vector<std::uint8_t> &dirtyBits) noexcept
{
    for (const std::uint32_t index : dirtyIndices)
    {
        if (index < dirtyBits.size())
        {
            dirtyBits[index] = 0u;
        }
    }
    dirtyIndices.clear();
}

} // namespace

namespace
{

bool equalSoftBodyRegularGridSource(const SoftBodyRegularGridSource &lhs,
                                    const SoftBodyRegularGridSource &rhs) noexcept
{
    return lhs.size == rhs.size && lhs.targetParticleSpacing == rhs.targetParticleSpacing &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodyTetMeshSource(const SoftBodyTetMeshSource &lhs,
                                const SoftBodyTetMeshSource &rhs) noexcept
{
    return lhs.objectSpaceRestPositions == rhs.objectSpaceRestPositions &&
           lhs.tetVertexIndices == rhs.tetVertexIndices &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodyTetGenSource(const SoftBodyTetGenSource &lhs,
                               const SoftBodyTetGenSource &rhs) noexcept
{
    return lhs.nodeFile == rhs.nodeFile && lhs.eleFile == rhs.eleFile &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodySourceDesc(const SoftBodySourceDesc &lhs, const SoftBodySourceDesc &rhs) noexcept
{
    if (lhs.kind != rhs.kind)
    {
        return false;
    }

    switch (lhs.kind)
    {
    case SoftBodySourceKind::RegularGrid:
        return equalSoftBodyRegularGridSource(lhs.regularGrid, rhs.regularGrid);
    case SoftBodySourceKind::TetMesh:
        return equalSoftBodyTetMeshSource(lhs.tetMesh, rhs.tetMesh);
    case SoftBodySourceKind::TetGenFiles:
        return equalSoftBodyTetGenSource(lhs.tetGen, rhs.tetGen);
    }

    return false;
}

bool equalFluidSourceDesc(const FluidSourceDesc &lhs, const FluidSourceDesc &rhs) noexcept
{
    if (lhs.kind != rhs.kind)
    {
        return false;
    }

    switch (lhs.kind)
    {
    case FluidSourceKind::RegularGrid:
        return lhs.regularGrid.size == rhs.regularGrid.size &&
               lhs.regularGrid.targetParticleSpacing == rhs.regularGrid.targetParticleSpacing;
    }

    return false;
}

} // namespace

void PhysicsWorld::clear()
{
    mRigidBodies.clear();
    mColliders.clear();
    mBodyColliderMapping.clear();
    mRigidJointScene.clear();
    mJointCollisionSuppression.clear();
    mEntityToRigidBodyIndex.clear();
    mRigidBodyIdToIndex.clear();
    mColliderIdToIndex.clear();
    mEntityToColliderIds.clear();
    mEntityToSoftBodyIndex.clear();
    mEntityToFluidIndex.clear();
    mTetGenMeshCache.clear();
    mRigidBodySnapshot.clear();
    mColliderSnapshot.clear();
    mSoftBodySnapshot.clear();
    mFluidSnapshot.clear();
    mBallJointSnapshot.clear();
    mHingeJointSnapshot.clear();
    mSliderJointSnapshot.clear();
    mSoftBodyDerivedCaches.clear();
    mFluidDerivedCaches.clear();
    mParticles.clear();
    mParticleContactMaterials.clear();
    mFluidMaterials.clear();
    mSoftEdges.clear();
    mSoftTets.clear();
    mSoftRenderData.clear();
    mRigidBodyDirtyIndices.clear();
    mColliderDirtyIndices.clear();
    mRigidBodyDirtyBits.clear();
    mColliderDirtyBits.clear();
    mRigidBodyCountDirty            = true;
    mColliderCountDirty             = true;
    mFullRigidBodyUploadRequired    = true;
    mFullColliderUploadRequired     = true;
    mBodyColliderMappingDirty       = true;
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    mSoftBodyDerivedStateDirty      = true;
    mStaticBroadPhaseDirty          = true;
    mActiveMovingColliderCount      = 0u;
    mStaticColliderCount            = 0u;
    mParticleGridCellSize           = 0.1f;
    mSoftBodyBoundsChunkCount       = 0u;
    mNextRigidBodyId                = 1u;
    mNextColliderId                 = 1u;
    mNextBallJointId                = 1u;
    mNextHingeJointId               = 1u;
    mNextSliderJointId              = 1u;
    ++mRigidBodyTopologyRevision;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    ++mSimulationRevision;
}

RigidBodyState &PhysicsWorld::upsertRigidBody(const RigidBodyState &state)
{
    RigidBodyState normalizedState = state;
    normalizeRigidBodyState(normalizedState);

    auto it = mEntityToRigidBodyIndex.find(normalizedState.entityId);
    if (it == mEntityToRigidBodyIndex.end())
    {
        normalizedState.rigidBodyId = mNextRigidBodyId++;

        const std::uint32_t index = static_cast<std::uint32_t>(mRigidBodies.size());
        mEntityToRigidBodyIndex.emplace(normalizedState.entityId, index);
        mRigidBodyIdToIndex.emplace(normalizedState.rigidBodyId, index);
        mRigidBodySnapshot.push_back(normalizedState);
        mRigidBodies.rigidBodyIds.push_back(normalizedState.rigidBodyId);
        mRigidBodies.entityIds.push_back(normalizedState.entityId);
        mRigidBodies.positionsInvMass.push_back(toPositionInvMass(normalizedState));
        mRigidBodies.orientations.push_back(toOrientation(normalizedState));
        mRigidBodies.scales.push_back(toScale(normalizedState));
        mRigidBodies.linearVelocities.push_back(toLinearVelocity(normalizedState));
        mRigidBodies.angularVelocities.push_back(toAngularVelocity(normalizedState));
        mRigidBodies.inverseInertiaLocal.push_back(toInverseInertiaLocal(normalizedState));
        mRigidBodies.bodyTypes.push_back(static_cast<std::uint32_t>(normalizedState.bodyType));
        mRigidBodies.kinematicTargetPositions.push_back(toKinematicTargetPosition(normalizedState));
        mRigidBodies.kinematicTargetOrientations.push_back(
            toKinematicTargetOrientation(normalizedState));
        mRigidBodies.kinematicTargetFlags.push_back(normalizedState.kinematicTargetEnabled ? 1u
                                                                                           : 0u);
        markRigidBodyDirty(index);
        markRigidBodyCountDirty();
        mBodyColliderMappingDirty       = true;
        mJointCollisionSuppressionDirty = true;
        mStaticBroadPhaseDirty          = mStaticBroadPhaseDirty || isStaticBody(normalizedState);
        ++mRigidBodyTopologyRevision;
        ++mAuthoredRevision;
        return mRigidBodySnapshot.back();
    }

    const std::uint32_t index          = it->second;
    normalizedState.rigidBodyId        = mRigidBodySnapshot[index].rigidBodyId;
    const RigidBodyState previousState = mRigidBodySnapshot[index];
    if (previousState.bodyType != normalizedState.bodyType)
    {
        const std::uint32_t enabledColliderCount =
            enabledColliderCountForEntity(previousState.entityId);
        if (isStaticBody(previousState))
        {
            mStaticColliderCount -= enabledColliderCount;
        }
        else if (isMovingBody(previousState))
        {
            mActiveMovingColliderCount -= enabledColliderCount;
        }

        if (isStaticBody(normalizedState))
        {
            mStaticColliderCount += enabledColliderCount;
        }
        else if (isMovingBody(normalizedState))
        {
            mActiveMovingColliderCount += enabledColliderCount;
        }
    }
    writeRigidBodySoAAt(mRigidBodies, index, normalizedState);
    mRigidBodySnapshot[index] = normalizedState;
    markRigidBodyDirty(index);
    if (previousState.environmentIndex != normalizedState.environmentIndex)
    {
        auto colliderHandlesIt = mEntityToColliderIds.find(previousState.entityId);
        if (colliderHandlesIt != mEntityToColliderIds.end())
        {
            for (const ColliderId colliderId : colliderHandlesIt->second)
            {
                const auto colliderIt = mColliderIdToIndex.find(colliderId);
                if (colliderIt != mColliderIdToIndex.end())
                {
                    const std::uint32_t colliderIndex = colliderIt->second;
                    if (colliderIndex < mColliders.environmentIndices.size())
                    {
                        mColliders.environmentIndices[colliderIndex] =
                            normalizedState.environmentIndex;
                        markColliderDirty(colliderIndex);
                    }
                }
            }
        }
        mRigidJointSceneDirty           = true;
        mJointCollisionSuppressionDirty = true;
        ++mRigidJointSceneRevision;
        ++mRigidJointModeRevision;
        ++mRigidJointTopologyRevision;
    }
    mStaticBroadPhaseDirty =
        mStaticBroadPhaseDirty || staticBodyPoseChanged(previousState, normalizedState);
    ++mAuthoredRevision;
    return mRigidBodySnapshot[index];
}

bool PhysicsWorld::removeRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    if (it == mEntityToRigidBodyIndex.end())
    {
        return false;
    }

    removeCollidersForEntity(entityId);

    const std::uint32_t index       = it->second;
    const std::uint32_t last        = static_cast<std::uint32_t>(mRigidBodies.size() - 1u);
    const RigidBodyId removedBodyId = mRigidBodySnapshot[index].rigidBodyId;
    const bool removedStatic        = isStaticBody(mRigidBodySnapshot[index]);

    if (index != last)
    {
        mRigidBodySnapshot[index]                    = mRigidBodySnapshot[last];
        mRigidBodies.rigidBodyIds[index]             = mRigidBodies.rigidBodyIds[last];
        mRigidBodies.entityIds[index]                = mRigidBodies.entityIds[last];
        mRigidBodies.positionsInvMass[index]         = mRigidBodies.positionsInvMass[last];
        mRigidBodies.orientations[index]             = mRigidBodies.orientations[last];
        mRigidBodies.scales[index]                   = mRigidBodies.scales[last];
        mRigidBodies.linearVelocities[index]         = mRigidBodies.linearVelocities[last];
        mRigidBodies.angularVelocities[index]        = mRigidBodies.angularVelocities[last];
        mRigidBodies.inverseInertiaLocal[index]      = mRigidBodies.inverseInertiaLocal[last];
        mRigidBodies.bodyTypes[index]                = mRigidBodies.bodyTypes[last];
        mRigidBodies.kinematicTargetPositions[index] = mRigidBodies.kinematicTargetPositions[last];
        mRigidBodies.kinematicTargetOrientations[index] =
            mRigidBodies.kinematicTargetOrientations[last];
        mRigidBodies.kinematicTargetFlags[index] = mRigidBodies.kinematicTargetFlags[last];
        mEntityToRigidBodyIndex[mRigidBodies.entityIds[index]] = index;
        mRigidBodyIdToIndex[mRigidBodies.rigidBodyIds[index]]  = index;
        markRigidBodyDirty(index);
    }

    mEntityToRigidBodyIndex.erase(it);
    mRigidBodyIdToIndex.erase(removedBodyId);
    mRigidBodySnapshot.pop_back();
    mRigidBodies.rigidBodyIds.pop_back();
    mRigidBodies.entityIds.pop_back();
    mRigidBodies.positionsInvMass.pop_back();
    mRigidBodies.orientations.pop_back();
    mRigidBodies.scales.pop_back();
    mRigidBodies.linearVelocities.pop_back();
    mRigidBodies.angularVelocities.pop_back();
    mRigidBodies.inverseInertiaLocal.pop_back();
    mRigidBodies.bodyTypes.pop_back();
    mRigidBodies.kinematicTargetPositions.pop_back();
    mRigidBodies.kinematicTargetOrientations.pop_back();
    mRigidBodies.kinematicTargetFlags.pop_back();

    markRigidBodyCountDirty();
    markColliderCountDirty(true);
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || removedStatic;
    pruneRigidJointsForBody(removedBodyId);
    ++mRigidBodyTopologyRevision;
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

void PhysicsWorld::upsertCollider(const ColliderState &state)
{
    RigidBodyId ownerRigidBodyId = state.ownerRigidBodyId;
    std::uint32_t ownerBodyIndex = 0xffffffffu;

    if (ownerRigidBodyId != kInvalidRigidBodyId)
    {
        const auto bodyIt = mRigidBodyIdToIndex.find(ownerRigidBodyId);
        if (bodyIt != mRigidBodyIdToIndex.end())
        {
            ownerBodyIndex = bodyIt->second;
        }
    }

    if (ownerBodyIndex == 0xffffffffu)
    {
        const auto bodyIt = mEntityToRigidBodyIndex.find(state.entityId);
        if (bodyIt == mEntityToRigidBodyIndex.end())
        {
            removeCollider(state.colliderId);
            return;
        }
        ownerBodyIndex   = bodyIt->second;
        ownerRigidBodyId = mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;
    }

    ColliderState normalizedState = state;
    normalizeColliderState(normalizedState);
    normalizedState.entityId         = mRigidBodySnapshot[ownerBodyIndex].entityId;
    normalizedState.ownerRigidBodyId = ownerRigidBodyId;
    if (normalizedState.colliderId == kInvalidColliderId)
    {
        normalizedState.colliderId = mNextColliderId++;
    }

    const bool ownerIsStatic = isStaticBody(mRigidBodySnapshot[ownerBodyIndex]);
    const auto colliderIt    = mColliderIdToIndex.find(normalizedState.colliderId);
    if (colliderIt == mColliderIdToIndex.end())
    {
        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mColliders.size());
        mColliderSnapshot.push_back(normalizedState);
        writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                           mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mColliderIdToIndex.emplace(normalizedState.colliderId, colliderIndex);
        auto &entityColliderIds = mEntityToColliderIds[normalizedState.entityId];
        entityColliderIds.push_back(normalizedState.colliderId);
        const std::uint32_t contribution =
            colliderBroadPhaseContribution(normalizedState, &mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
        markColliderDirty(colliderIndex);
        markColliderCountDirty();
        mBodyColliderMappingDirty = true;
        mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || ownerIsStatic;
        ++mAuthoredRevision;
        return;
    }

    const std::uint32_t colliderIndex        = colliderIt->second;
    const ColliderState previousState        = mColliderSnapshot[colliderIndex];
    const std::uint32_t previousContribution = broadPhaseContributionForCollider(previousState);
    const std::uint32_t newContribution =
        colliderBroadPhaseContribution(normalizedState, &mRigidBodySnapshot[ownerBodyIndex]);
    const bool staticContributionChanged = previousContribution == kBroadPhaseContributionStatic ||
                                           newContribution == kBroadPhaseContributionStatic;
    if (previousContribution != newContribution)
    {
        if (previousContribution == kBroadPhaseContributionStatic)
        {
            --mStaticColliderCount;
        }
        else if (previousContribution == kBroadPhaseContributionMoving)
        {
            --mActiveMovingColliderCount;
        }

        if (newContribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (newContribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
    }
    writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                       mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
    mColliderSnapshot[colliderIndex] = normalizedState;
    markColliderDirty(colliderIndex);
    if (previousState.ownerRigidBodyId != normalizedState.ownerRigidBodyId)
    {
        mBodyColliderMappingDirty = true;
    }
    const bool staticColliderShapeChanged =
        previousState.shapeType != normalizedState.shapeType ||
        previousState.shapeParams.x != normalizedState.shapeParams.x ||
        previousState.shapeParams.y != normalizedState.shapeParams.y ||
        previousState.shapeParams.z != normalizedState.shapeParams.z ||
        previousState.shapeParams.w != normalizedState.shapeParams.w ||
        previousState.localPosition.x != normalizedState.localPosition.x ||
        previousState.localPosition.y != normalizedState.localPosition.y ||
        previousState.localPosition.z != normalizedState.localPosition.z ||
        previousState.localRotation.q.x != normalizedState.localRotation.q.x ||
        previousState.localRotation.q.y != normalizedState.localRotation.q.y ||
        previousState.localRotation.q.z != normalizedState.localRotation.q.z ||
        previousState.localRotation.q.w != normalizedState.localRotation.q.w ||
        previousState.enabled != normalizedState.enabled;
    if (staticContributionChanged || ((previousContribution == kBroadPhaseContributionStatic ||
                                       newContribution == kBroadPhaseContributionStatic) &&
                                      staticColliderShapeChanged))
    {
        mStaticBroadPhaseDirty = true;
    }
    ++mAuthoredRevision;
}

bool PhysicsWorld::removeCollider(ColliderId colliderId)
{
    const auto it = mColliderIdToIndex.find(colliderId);
    if (it == mColliderIdToIndex.end())
    {
        return false;
    }

    bool removedStaticOwner            = false;
    const std::uint32_t ownerBodyIndex = mColliders.ownerRigidBodyIndices[it->second];
    if (ownerBodyIndex != 0xffffffffu && ownerBodyIndex < rigidBodyCount())
    {
        removedStaticOwner = isStaticBody(mRigidBodySnapshot[ownerBodyIndex]);
    }
    removeColliderAtIndex(it->second);
    markColliderCountDirty();
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || removedStaticOwner;
    ++mAuthoredRevision;
    return true;
}

void PhysicsWorld::replaceColliders(common::EntityId entityId,
                                    const std::vector<ColliderState> &colliders)
{
    removeCollidersForEntity(entityId);

    const auto bodyIt = mEntityToRigidBodyIndex.find(entityId);
    if (bodyIt == mEntityToRigidBodyIndex.end() || colliders.empty())
    {
        markColliderCountDirty(true);
        mBodyColliderMappingDirty = true;
        ++mAuthoredRevision;
        return;
    }

    const std::uint32_t ownerBodyIndex = bodyIt->second;
    const RigidBodyId ownerRigidBodyId = mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;

    auto &entityColliderIds = mEntityToColliderIds[entityId];
    entityColliderIds.reserve(colliders.size());

    for (ColliderState collider : colliders)
    {
        normalizeColliderState(collider);
        collider.colliderId       = mNextColliderId++;
        collider.entityId         = entityId;
        collider.ownerRigidBodyId = ownerRigidBodyId;

        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mColliders.size());
        mColliderSnapshot.push_back(collider);
        writeColliderSoAAt(mColliders, colliderIndex, collider, ownerBodyIndex,
                           mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mColliderIdToIndex.emplace(collider.colliderId, colliderIndex);
        entityColliderIds.push_back(collider.colliderId);
        const std::uint32_t contribution =
            colliderBroadPhaseContribution(collider, &mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
    }

    markColliderCountDirty(true);
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = true;
    ++mAuthoredRevision;
}

bool PhysicsWorld::upsertSoftBody(const SoftBodyState &state)
{
    SoftBodyState normalizedState = state;
    normalizeSoftBodyState(normalizedState);

    const auto it                      = mEntityToSoftBodyIndex.find(normalizedState.entityId);
    const SoftBodyState *previousState = nullptr;
    if (it != mEntityToSoftBodyIndex.end())
    {
        previousState = &mSoftBodySnapshot[it->second];
    }

    if (previousState != nullptr && classifySoftBodyChange(*previousState, normalizedState) ==
                                        SoftBodyChangeKind::RuntimePropertiesOnly)
    {
        applySoftBodyRuntimeProperties(it->second, normalizedState);
        ++mSoftParticleRevision;
        ++mAuthoredRevision;
        return true;
    }

    SoftBodyDerivedCache derivedCache;
    if (!prepareSoftBodyStateForInsert(normalizedState, previousState, derivedCache))
    {
        return false;
    }

    if (it == mEntityToSoftBodyIndex.end())
    {
        mEntityToSoftBodyIndex.emplace(normalizedState.entityId,
                                       static_cast<std::uint32_t>(mSoftBodySnapshot.size()));
        mSoftBodySnapshot.push_back(normalizedState);
        mSoftBodyDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mSoftBodySnapshot[it->second]      = normalizedState;
        mSoftBodyDerivedCaches[it->second] = std::move(derivedCache);
    }

    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertFluid(const FluidState &state)
{
    FluidState normalizedState = state;
    normalizeFluidState(normalizedState);

    const auto it = mEntityToFluidIndex.find(normalizedState.entityId);
    const FluidState *previousState =
        it != mEntityToFluidIndex.end() ? &mFluidSnapshot[it->second] : nullptr;
    if (!validateFluidMaterialCompatibility(normalizedState, previousState))
    {
        return false;
    }

    FluidDerivedCache derivedCache;
    if (!prepareFluidStateForInsert(normalizedState, derivedCache))
    {
        return false;
    }

    if (it == mEntityToFluidIndex.end())
    {
        mEntityToFluidIndex.emplace(normalizedState.entityId,
                                    static_cast<std::uint32_t>(mFluidSnapshot.size()));
        mFluidSnapshot.push_back(normalizedState);
        mFluidDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mFluidSnapshot[it->second]      = normalizedState;
        mFluidDerivedCaches[it->second] = std::move(derivedCache);
    }
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeSoftBody(common::EntityId entityId)
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    if (it == mEntityToSoftBodyIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mSoftBodySnapshot.size() - 1u);
    if (index != last)
    {
        mSoftBodySnapshot[index]      = mSoftBodySnapshot[last];
        mSoftBodyDerivedCaches[index] = std::move(mSoftBodyDerivedCaches[last]);
        mEntityToSoftBodyIndex[mSoftBodySnapshot[index].entityId] = index;
    }
    mSoftBodySnapshot.pop_back();
    mSoftBodyDerivedCaches.pop_back();
    mEntityToSoftBodyIndex.erase(it);
    mTetGenMeshCache.erase(entityId);
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeFluid(common::EntityId entityId)
{
    const auto it = mEntityToFluidIndex.find(entityId);
    if (it == mEntityToFluidIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mFluidSnapshot.size() - 1u);
    if (index != last)
    {
        mFluidSnapshot[index]                               = mFluidSnapshot[last];
        mFluidDerivedCaches[index]                          = std::move(mFluidDerivedCaches[last]);
        mEntityToFluidIndex[mFluidSnapshot[index].entityId] = index;
    }
    mFluidSnapshot.pop_back();
    mFluidDerivedCaches.pop_back();
    mEntityToFluidIndex.erase(it);
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertBallJoint(const BallJointState &state)
{
    BallJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    if (normalized.jointId == kInvalidBallJointId)
    {
        normalized.jointId = mNextBallJointId++;
    }
    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it             = std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                                       [&](const BallJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mBallJointSnapshot.end();
    if (it == mBallJointSnapshot.end())
    {
        mBallJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifyBallJointChange(inserted));
    return true;
}

bool PhysicsWorld::upsertHingeJoint(const HingeJointState &state)
{
    HingeJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    normalized.localRotationA =
        common::runtime_math::normalizeQuaternion(normalized.localRotationA);
    normalized.localRotationB =
        common::runtime_math::normalizeQuaternion(normalized.localRotationB);
    normalized.driveTargetAngle =
        std::remainder(normalized.driveTargetAngle, 2.0f * Diligent::PI_F);
    if (!normalized.limitEnabled)
    {
        normalized.limitMin = 0.0f;
        normalized.limitMax = 0.0f;
    }
    normalized.constraintCompliance = std::max(normalized.constraintCompliance, 0.0f);
    normalized.driveCompliance      = std::max(normalized.driveCompliance, 0.0f);
    if (normalized.jointId == kInvalidHingeJointId)
    {
        normalized.jointId = mNextHingeJointId++;
    }
    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it             = std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                                       [&](const HingeJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mHingeJointSnapshot.end();
    const HingeJointState previousState = inserted ? HingeJointState{} : *it;
    if (it == mHingeJointSnapshot.end())
    {
        mHingeJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifyHingeJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::upsertSliderJoint(const SliderJointState &state)
{
    SliderJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    normalized.localRotationA =
        common::runtime_math::normalizeQuaternion(normalized.localRotationA);
    normalized.localRotationB =
        common::runtime_math::normalizeQuaternion(normalized.localRotationB);
    if (normalized.limitEnabled && normalized.limitMin > normalized.limitMax)
    {
        std::swap(normalized.limitMin, normalized.limitMax);
    }
    if (!normalized.limitEnabled)
    {
        normalized.limitMin = 0.0f;
        normalized.limitMax = 0.0f;
    }
    normalized.constraintCompliance = std::max(normalized.constraintCompliance, 0.0f);
    normalized.driveCompliance      = std::max(normalized.driveCompliance, 0.0f);

    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    const RigidBodyState &bodyA = mRigidBodySnapshot[bodyAIt->second];
    const RigidBodyState &bodyB = mRigidBodySnapshot[bodyBIt->second];
    if (bodyA.environmentIndex != bodyB.environmentIndex)
    {
        return false;
    }
    const float anchorAMagSq = Diligent::dot(normalized.localAnchorA, normalized.localAnchorA);
    const float anchorBMagSq = Diligent::dot(normalized.localAnchorB, normalized.localAnchorB);
    if (anchorAMagSq <= 1.0e-8f && anchorBMagSq <= 1.0e-8f)
    {
        const Diligent::float3 worldAnchor = bodyB.position;
        normalized.localAnchorA =
            quaternionInverseRotate(bodyA.rotation, worldAnchor - bodyA.position);
        normalized.localAnchorB =
            quaternionInverseRotate(bodyB.rotation, worldAnchor - bodyB.position);
    }
    if (normalized.jointId == kInvalidSliderJointId)
    {
        normalized.jointId = mNextSliderJointId++;
    }

    auto it             = std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                                       [&](const SliderJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mSliderJointSnapshot.end();
    const SliderJointState previousState = inserted ? SliderJointState{} : *it;
    if (it == mSliderJointSnapshot.end())
    {
        mSliderJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifySliderJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::removeBallJoint(BallJointId jointId)
{
    auto it = std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                           [&](const BallJointState &state) { return state.jointId == jointId; });
    if (it != mBallJointSnapshot.end())
    {
        mBallJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeHingeJoint(HingeJointId jointId)
{
    auto it = std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                           [&](const HingeJointState &state) { return state.jointId == jointId; });
    if (it != mHingeJointSnapshot.end())
    {
        mHingeJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeSliderJoint(SliderJointId jointId)
{
    auto it = std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                           [&](const SliderJointState &state) { return state.jointId == jointId; });
    if (it != mSliderJointSnapshot.end())
    {
        mSliderJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const ColliderState *PhysicsWorld::tryGetCollider(ColliderId colliderId) const
{
    const auto it = mColliderIdToIndex.find(colliderId);
    return it == mColliderIdToIndex.end() ? nullptr : &mColliderSnapshot[it->second];
}

const BallJointState *PhysicsWorld::tryGetBallJoint(BallJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                     [&](const BallJointState &state) { return state.jointId == jointId; });
    return it == mBallJointSnapshot.end() ? nullptr : &(*it);
}

const HingeJointState *PhysicsWorld::tryGetHingeJoint(HingeJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                     [&](const HingeJointState &state) { return state.jointId == jointId; });
    return it == mHingeJointSnapshot.end() ? nullptr : &(*it);
}

const SliderJointState *PhysicsWorld::tryGetSliderJoint(SliderJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                     [&](const SliderJointState &state) { return state.jointId == jointId; });
    return it == mSliderJointSnapshot.end() ? nullptr : &(*it);
}

SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId)
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    return it == mEntityToSoftBodyIndex.end() ? nullptr : &mSoftBodySnapshot[it->second];
}

const SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId) const
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    return it == mEntityToSoftBodyIndex.end() ? nullptr : &mSoftBodySnapshot[it->second];
}

bool PhysicsWorld::tryGetSoftBodyAuthoringRestPositions(
    common::EntityId entityId, std::vector<Diligent::float3> &outRestPositions) const
{
    outRestPositions.clear();

    const SoftBodyState *softBody = tryGetSoftBody(entityId);
    if (softBody == nullptr)
    {
        return false;
    }

    if (!softBody->restPositions.empty())
    {
        outRestPositions = softBody->restPositions;
        return true;
    }

    ResolvedSoftBodyTopology topology{};
    std::string errorMessage;
    TetMeshData tetGenMeshCache{};
    const TetGenMeshCache *tetGenCache = tryGetTetGenMeshCache(entityId);
    const TetMeshData *cacheView       = nullptr;
    if (tetGenCache != nullptr)
    {
        tetGenMeshCache.objectSpaceRestPositions = tetGenCache->objectSpaceRestPositions;
        tetGenMeshCache.tetVertexIndices         = tetGenCache->tetVertexIndices;
        cacheView                                = &tetGenMeshCache;
    }

    if (!resolveSoftBodyTopology(*softBody, cacheView, topology, errorMessage))
    {
        CRESSIM_LOG_ERROR("Failed to resolve authored soft-body particles for entity ", entityId,
                          ": ", errorMessage);
        return false;
    }

    outRestPositions = std::move(topology.restPositions);
    return true;
}

FluidState *PhysicsWorld::tryGetFluid(common::EntityId entityId)
{
    const auto it = mEntityToFluidIndex.find(entityId);
    return it == mEntityToFluidIndex.end() ? nullptr : &mFluidSnapshot[it->second];
}

const FluidState *PhysicsWorld::tryGetFluid(common::EntityId entityId) const
{
    const auto it = mEntityToFluidIndex.find(entityId);
    return it == mEntityToFluidIndex.end() ? nullptr : &mFluidSnapshot[it->second];
}

const std::vector<RigidBodyState> &PhysicsWorld::rigidBodySnapshot() const noexcept
{
    return mRigidBodySnapshot;
}

const std::vector<ColliderState> &PhysicsWorld::colliderSnapshot() const noexcept
{
    return mColliderSnapshot;
}

const std::vector<SoftBodyState> &PhysicsWorld::softBodySnapshot() const noexcept
{
    return mSoftBodySnapshot;
}

const std::vector<FluidState> &PhysicsWorld::fluidSnapshot() const noexcept
{
    return mFluidSnapshot;
}

const std::vector<BallJointState> &PhysicsWorld::ballJointSnapshot() const noexcept
{
    return mBallJointSnapshot;
}

const std::vector<HingeJointState> &PhysicsWorld::hingeJointSnapshot() const noexcept
{
    return mHingeJointSnapshot;
}

const std::vector<SliderJointState> &PhysicsWorld::sliderJointSnapshot() const noexcept
{
    return mSliderJointSnapshot;
}

const RigidBodySoAHost &PhysicsWorld::rigidBodySoA() const noexcept
{
    return mRigidBodies;
}

const ColliderSoAHost &PhysicsWorld::colliderSoA() const noexcept
{
    ensureDerivedStateUpToDate();
    return mColliders;
}

const BodyColliderMappingHost &PhysicsWorld::bodyColliderMapping() const noexcept
{
    ensureDerivedStateUpToDate();
    return mBodyColliderMapping;
}

const RigidJointSceneHost &PhysicsWorld::rigidJointScene() const noexcept
{
    ensureDerivedStateUpToDate();
    return mRigidJointScene;
}

const JointCollisionSuppressionHost &PhysicsWorld::jointCollisionSuppression() const noexcept
{
    ensureDerivedStateUpToDate();
    return mJointCollisionSuppression;
}

const ParticleSoAHost &PhysicsWorld::particles() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticles;
}

const std::vector<Diligent::float4> &PhysicsWorld::particleContactMaterials() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticleContactMaterials;
}

const std::vector<FluidMaterialGpu> &PhysicsWorld::fluidMaterials() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mFluidMaterials;
}

const std::vector<SoftEdge> &PhysicsWorld::softEdges() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftEdges;
}

const std::vector<SoftTet> &PhysicsWorld::softTets() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftTets;
}

const SoftRenderDataHost &PhysicsWorld::softRenderData() const noexcept
{
    return mSoftRenderData;
}

void PhysicsWorld::setSoftRenderData(const SoftRenderDataHost &data)
{
    mSoftRenderData = data;
    recomputeSoftBodyBoundsChunkCount();
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::ensureDerivedStateUpToDate() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    if (mBodyColliderMappingDirty)
    {
        rebuildBodyColliderMapping();
        mBodyColliderMappingDirty = false;
    }
    if (mRigidJointSceneDirty)
    {
        rebuildRigidJointScene();
        mRigidJointSceneDirty = false;
    }
    if (mJointCollisionSuppressionDirty)
    {
        rebuildJointCollisionSuppression();
        mJointCollisionSuppressionDirty = false;
    }
}

void PhysicsWorld::ensureSoftBodyDerivedStateUpToDate() noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        rebuildSoftBodyDerivedState();
    }
}

const std::vector<std::uint32_t> &PhysicsWorld::rigidBodyDirtyIndices() const noexcept
{
    return mRigidBodyDirtyIndices;
}

const std::vector<std::uint32_t> &PhysicsWorld::colliderDirtyIndices() const noexcept
{
    return mColliderDirtyIndices;
}

std::uint32_t PhysicsWorld::rigidBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mRigidBodies.size());
}

std::uint32_t PhysicsWorld::colliderCount() const noexcept
{
    return static_cast<std::uint32_t>(mColliders.size());
}

std::uint32_t PhysicsWorld::softBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mSoftBodySnapshot.size());
}

std::uint32_t PhysicsWorld::fluidCount() const noexcept
{
    return static_cast<std::uint32_t>(mFluidSnapshot.size());
}

bool PhysicsWorld::rigidBodyCountDirty() const noexcept
{
    return mRigidBodyCountDirty;
}

bool PhysicsWorld::colliderCountDirty() const noexcept
{
    return mColliderCountDirty;
}

bool PhysicsWorld::fullRigidBodyUploadRequired() const noexcept
{
    return mFullRigidBodyUploadRequired;
}

bool PhysicsWorld::fullColliderUploadRequired() const noexcept
{
    return mFullColliderUploadRequired;
}

void PhysicsWorld::clearRigidBodyUploadState() noexcept
{
    clearDirtyIndices(mRigidBodyDirtyIndices, mRigidBodyDirtyBits);
    mRigidBodyCountDirty         = false;
    mFullRigidBodyUploadRequired = false;
}

void PhysicsWorld::clearColliderUploadState() noexcept
{
    clearDirtyIndices(mColliderDirtyIndices, mColliderDirtyBits);
    mColliderCountDirty         = false;
    mFullColliderUploadRequired = false;
}

bool PhysicsWorld::staticBroadPhaseDirty() const noexcept
{
    return mStaticBroadPhaseDirty;
}

void PhysicsWorld::clearStaticBroadPhaseDirty() noexcept
{
    mStaticBroadPhaseDirty = false;
}

std::uint32_t PhysicsWorld::activeMovingColliderCount() const noexcept
{
    return mActiveMovingColliderCount;
}

std::uint32_t PhysicsWorld::staticColliderCount() const noexcept
{
    return mStaticColliderCount;
}

float PhysicsWorld::particleGridCellSize() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticleGridCellSize;
}

std::uint32_t PhysicsWorld::softBodyBoundsChunkCount() const noexcept
{
    return mSoftBodyBoundsChunkCount;
}

void PhysicsWorld::integrateRigidBodiesCpu(float dt) noexcept
{
    if (mRigidBodies.empty())
    {
        return;
    }

    for (std::uint32_t i = 0; i < rigidBodyCount(); ++i)
    {
        Diligent::float4 &positionInvMass     = mRigidBodies.positionsInvMass[i];
        const Diligent::float4 linearVelocity = mRigidBodies.linearVelocities[i];
        positionInvMass.x += linearVelocity.x * dt;
        positionInvMass.y += linearVelocity.y * dt;
        positionInvMass.z += linearVelocity.z * dt;

        RigidBodyState &state = mRigidBodySnapshot[i];
        state.position.x      = positionInvMass.x;
        state.position.y      = positionInvMass.y;
        state.position.z      = positionInvMass.z;
    }

    markAllRigidBodiesDirty();
    ++mAuthoredRevision;
}

bool PhysicsWorld::syncRigidBodyStateFromSimulation(
    std::uint32_t index, const Diligent::float4 &positionInvMass,
    const Diligent::float4 &orientation, const Diligent::float4 &linearVelocity,
    const Diligent::float4 &angularVelocity) noexcept
{
    if (index >= rigidBodyCount())
    {
        return false;
    }

    mRigidBodies.positionsInvMass[index]  = positionInvMass;
    mRigidBodies.orientations[index]      = orientation;
    mRigidBodies.linearVelocities[index]  = linearVelocity;
    mRigidBodies.angularVelocities[index] = angularVelocity;
    RigidBodyState &state                 = mRigidBodySnapshot[index];
    state.position    = Diligent::float3{positionInvMass.x, positionInvMass.y, positionInvMass.z};
    state.inverseMass = positionInvMass.w;
    state.rotation =
        Diligent::QuaternionF{orientation.x, orientation.y, orientation.z, orientation.w};
    state.linearVelocity = Diligent::float3{linearVelocity.x, linearVelocity.y, linearVelocity.z};
    state.angularVelocity =
        Diligent::float3{angularVelocity.x, angularVelocity.y, angularVelocity.z};
    return true;
}

void PhysicsWorld::finalizeRigidBodyWriteback() noexcept
{
    if (mRigidBodies.empty())
    {
        return;
    }
    ++mSimulationRevision;
}

bool PhysicsWorld::syncParticleStateFromSimulation(std::uint32_t index,
                                                   const Diligent::float4 &positionInvMass,
                                                   const Diligent::float4 &previousPosition,
                                                   const Diligent::float4 &velocity) noexcept
{
    if (index >= mParticles.size())
    {
        return false;
    }

    mParticles.positionsInvMass[index]  = positionInvMass;
    mParticles.previousPositions[index] = previousPosition;
    mParticles.velocities[index]        = velocity;
    return true;
}

void PhysicsWorld::finalizeParticleWriteback() noexcept
{
    if (mParticles.empty())
    {
        return;
    }
    ++mSimulationRevision;
}

std::uint64_t PhysicsWorld::authoredRevision() const noexcept
{
    return mAuthoredRevision;
}

std::uint64_t PhysicsWorld::simulationRevision() const noexcept
{
    return mSimulationRevision;
}

std::uint64_t PhysicsWorld::rigidBodyTopologyRevision() const noexcept
{
    return mRigidBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointTopologyRevision() const noexcept
{
    return mRigidJointTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointSceneRevision() const noexcept
{
    return mRigidJointSceneRevision;
}

std::uint64_t PhysicsWorld::rigidJointModeRevision() const noexcept
{
    return mRigidJointModeRevision;
}

std::uint64_t PhysicsWorld::softBodyTopologyRevision() const noexcept
{
    return mSoftBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::softParticleRevision() const noexcept
{
    return mSoftParticleRevision;
}

std::uint64_t PhysicsWorld::softGpuTopologyRevision() const noexcept
{
    return mSoftGpuTopologyRevision;
}

void PhysicsWorld::writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                       const RigidBodyState &state)
{
    soa.rigidBodyIds[index]                = state.rigidBodyId;
    soa.entityIds[index]                   = state.entityId;
    soa.positionsInvMass[index]            = toPositionInvMass(state);
    soa.orientations[index]                = toOrientation(state);
    soa.scales[index]                      = toScale(state);
    soa.linearVelocities[index]            = toLinearVelocity(state);
    soa.angularVelocities[index]           = toAngularVelocity(state);
    soa.inverseInertiaLocal[index]         = toInverseInertiaLocal(state);
    soa.bodyTypes[index]                   = static_cast<std::uint32_t>(state.bodyType);
    soa.kinematicTargetPositions[index]    = toKinematicTargetPosition(state);
    soa.kinematicTargetOrientations[index] = toKinematicTargetOrientation(state);
    soa.kinematicTargetFlags[index]        = state.kinematicTargetEnabled ? 1u : 0u;
}

void PhysicsWorld::writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                      const ColliderState &state, std::uint32_t ownerBodyIndex,
                                      std::uint32_t ownerEnvironmentIndex)
{
    if (index == soa.size())
    {
        soa.colliderIds.push_back(state.colliderId);
        soa.entityIds.push_back(state.entityId);
        soa.ownerRigidBodyIds.push_back(state.ownerRigidBodyId);
        soa.ownerRigidBodyIndices.push_back(ownerBodyIndex);
        soa.environmentIndices.push_back(ownerEnvironmentIndex);
        soa.shapeTypes.push_back(static_cast<std::uint32_t>(state.shapeType));
        soa.shapeParams.push_back(state.shapeParams);
        soa.localPositions.push_back(toColliderLocalPosition(state));
        soa.localOrientations.push_back(toColliderLocalOrientation(state));
        soa.enabledFlags.push_back(state.enabled ? 1u : 0u);
        soa.frictionRestitution.push_back(toColliderMaterial(state));
        soa.collisionLayers.push_back(state.collisionLayer);
        soa.collisionMasks.push_back(state.collisionMask);
        return;
    }

    soa.colliderIds[index]           = state.colliderId;
    soa.entityIds[index]             = state.entityId;
    soa.ownerRigidBodyIds[index]     = state.ownerRigidBodyId;
    soa.ownerRigidBodyIndices[index] = ownerBodyIndex;
    soa.environmentIndices[index]    = ownerEnvironmentIndex;
    soa.shapeTypes[index]            = static_cast<std::uint32_t>(state.shapeType);
    soa.shapeParams[index]           = state.shapeParams;
    soa.localPositions[index]        = toColliderLocalPosition(state);
    soa.localOrientations[index]     = toColliderLocalOrientation(state);
    soa.enabledFlags[index]          = state.enabled ? 1u : 0u;
    soa.frictionRestitution[index]   = toColliderMaterial(state);
    soa.collisionLayers[index]       = state.collisionLayer;
    soa.collisionMasks[index]        = state.collisionMask;
}

bool PhysicsWorld::isStaticBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Static;
}

bool PhysicsWorld::isMovingBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Kinematic || state.bodyType == RigidBodyType::Dynamic;
}

std::uint32_t PhysicsWorld::colliderBroadPhaseContribution(const ColliderState &collider,
                                                           const RigidBodyState *owner) noexcept
{
    if (!collider.enabled || owner == nullptr)
    {
        return kBroadPhaseContributionNone;
    }
    if (isStaticBody(*owner))
    {
        return kBroadPhaseContributionStatic;
    }
    if (isMovingBody(*owner))
    {
        return kBroadPhaseContributionMoving;
    }
    return kBroadPhaseContributionNone;
}

bool PhysicsWorld::staticBodyPoseChanged(const RigidBodyState &before,
                                         const RigidBodyState &after) noexcept
{
    if (isStaticBody(before) != isStaticBody(after))
    {
        return true;
    }
    if (!isStaticBody(before))
    {
        return false;
    }

    return before.position.x != after.position.x || before.position.y != after.position.y ||
           before.position.z != after.position.z || before.rotation.q.x != after.rotation.q.x ||
           before.rotation.q.y != after.rotation.q.y || before.rotation.q.z != after.rotation.q.z ||
           before.rotation.q.w != after.rotation.q.w || before.scale.x != after.scale.x ||
           before.scale.y != after.scale.y || before.scale.z != after.scale.z;
}

void PhysicsWorld::normalizeRigidBodyState(RigidBodyState &state) noexcept
{
    if (state.bodyType == RigidBodyType::Dynamic && state.inverseMass <= 0.0f)
    {
        state.inverseMass = 1.0f;
    }

    if (state.bodyType != RigidBodyType::Kinematic)
    {
        state.kinematicTargetPosition = state.position;
        state.kinematicTargetRotation = state.rotation;
        state.kinematicTargetEnabled  = false;
    }
    else if (!state.kinematicTargetEnabled)
    {
        state.kinematicTargetPosition = state.position;
        state.kinematicTargetRotation = state.rotation;
    }
}

void PhysicsWorld::normalizeColliderState(ColliderState &state) noexcept
{
    state.friction = std::max(state.friction, 0.0f);
    state.staticFriction =
        state.staticFriction < 0.0f ? state.friction : std::max(state.staticFriction, 0.0f);
    state.restitution = std::clamp(state.restitution, 0.0f, 1.0f);
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeSoftBodyState(SoftBodyState &state) noexcept
{
    normalizeParticleContactMaterial(state.material.contact);
    state.particleMass     = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius   = std::max(state.particleRadius, 1.0e-4f);
    state.edgeCompliance   = std::max(state.edgeCompliance, 0.0f);
    state.volumeCompliance = std::max(state.volumeCompliance, 0.0f);

    const auto clampScale = [](float value) -> float
    {
        const float sign = value < 0.0f ? -1.0f : 1.0f;
        return sign * std::max(std::abs(value), 1.0e-4f);
    };
    state.restTransform.scale.x = clampScale(state.restTransform.scale.x);
    state.restTransform.scale.y = clampScale(state.restTransform.scale.y);
    state.restTransform.scale.z = clampScale(state.restTransform.scale.z);

    if (state.source.kind == SoftBodySourceKind::RegularGrid)
    {
        state.source.regularGrid.size.x = std::max(state.source.regularGrid.size.x, 1.0e-4f);
        state.source.regularGrid.size.y = std::max(state.source.regularGrid.size.y, 1.0e-4f);
        state.source.regularGrid.size.z = std::max(state.source.regularGrid.size.z, 1.0e-4f);
        state.source.regularGrid.targetParticleSpacing =
            std::max(state.source.regularGrid.targetParticleSpacing, 1.0e-4f);
    }
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeFluidState(FluidState &state) noexcept
{
    normalizeFluidMaterial(state.material);
    state.particleMass   = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius = std::max(state.particleRadius, 1.0e-4f);
    state.visualColor.x  = std::clamp(state.visualColor.x, 0.0f, 1.0f);
    state.visualColor.y  = std::clamp(state.visualColor.y, 0.0f, 1.0f);
    state.visualColor.z  = std::clamp(state.visualColor.z, 0.0f, 1.0f);
    state.visualColor.w  = std::clamp(state.visualColor.w, 0.0f, 1.0f);

    const auto clampScale = [](float value) -> float
    {
        const float sign = value < 0.0f ? -1.0f : 1.0f;
        return sign * std::max(std::abs(value), 1.0e-4f);
    };
    state.restTransform.scale.x = clampScale(state.restTransform.scale.x);
    state.restTransform.scale.y = clampScale(state.restTransform.scale.y);
    state.restTransform.scale.z = clampScale(state.restTransform.scale.z);

    if (state.source.kind == FluidSourceKind::RegularGrid)
    {
        state.source.regularGrid.size.x = std::max(state.source.regularGrid.size.x, 1.0e-4f);
        state.source.regularGrid.size.y = std::max(state.source.regularGrid.size.y, 1.0e-4f);
        state.source.regularGrid.size.z = std::max(state.source.regularGrid.size.z, 1.0e-4f);
        state.source.regularGrid.targetParticleSpacing =
            std::max(state.source.regularGrid.targetParticleSpacing, 1.0e-4f);
    }
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

bool PhysicsWorld::validateFluidMaterialCompatibility(
    const FluidState &candidate, const FluidState *previousState) const noexcept
{
    for (const FluidState &existing : mFluidSnapshot)
    {
        if (previousState != nullptr && existing.entityId == previousState->entityId)
        {
            continue;
        }

        if (!nearlyEqual(existing.particleRadius, candidate.particleRadius))
        {
            CRESSIM_LOG_ERROR(
                "Fluid bodies must currently share particleRadius because the fluid solve "
                "model is spacing-derived. Entity ",
                candidate.entityId, " is incompatible with existing fluid entity ",
                existing.entityId, ".");
            return false;
        }
    }

    return true;
}

PhysicsWorld::SoftBodyChangeKind PhysicsWorld::classifySoftBodyChange(
    const SoftBodyState &previousState, const SoftBodyState &candidate) noexcept
{
    if (!equalSoftBodySourceDesc(previousState.source, candidate.source) ||
        previousState.restTransform != candidate.restTransform)
    {
        return SoftBodyChangeKind::TopologyRebuild;
    }

    return SoftBodyChangeKind::RuntimePropertiesOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifyBallJointChange(bool inserted) noexcept
{
    return inserted ? RigidJointChangeKind::TopologyRebuild : RigidJointChangeKind::PayloadOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifyHingeJointChange(
    const HingeJointState &previousState, const HingeJointState &candidate, bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.enabled != candidate.enabled ||
        previousState.driveMode != candidate.driveMode)
    {
        return RigidJointChangeKind::ModeRebuild;
    }

    return RigidJointChangeKind::PayloadOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifySliderJointChange(
    const SliderJointState &previousState, const SliderJointState &candidate,
    bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.enabled != candidate.enabled ||
        previousState.driveMode != candidate.driveMode)
    {
        return RigidJointChangeKind::ModeRebuild;
    }

    return RigidJointChangeKind::PayloadOnly;
}

void PhysicsWorld::applySoftBodyRuntimeProperties(std::uint32_t index,
                                                  const SoftBodyState &normalizedState) noexcept
{
    if (index >= mSoftBodySnapshot.size())
    {
        return;
    }

    if (mSoftBodyDerivedStateDirty)
    {
        rebuildSoftBodyDerivedState();
    }

    if (index >= mSoftBodySnapshot.size() || index >= mSoftBodyDerivedCaches.size())
    {
        return;
    }

    SoftBodyState &softBody       = mSoftBodySnapshot[index];
    softBody.environmentIndex     = normalizedState.environmentIndex;
    softBody.collisionLayer       = normalizedState.collisionLayer;
    softBody.collisionMask        = normalizedState.collisionMask;
    softBody.material             = normalizedState.material;
    softBody.particleMass         = normalizedState.particleMass;
    softBody.particleRadius       = normalizedState.particleRadius;
    softBody.edgeCompliance       = normalizedState.edgeCompliance;
    softBody.volumeCompliance     = normalizedState.volumeCompliance;
    softBody.simulated            = normalizedState.simulated;
    softBody.selfCollisionEnabled = normalizedState.selfCollisionEnabled;

    softBody.contactMaterialIndex = findOrAppendParticleContactMaterial(
        mParticleContactMaterials, toParticleContactMaterial(softBody.material.contact));

    const std::uint32_t phase         = packParticlePhase(index, softBody.selfCollisionEnabled);
    const std::uint32_t particleBegin = softBody.particleOffset;
    const std::uint32_t particleEnd   = std::min<std::uint32_t>(
        particleBegin + softBody.particleCount, static_cast<std::uint32_t>(mParticles.size()));
    for (std::uint32_t particleIndex = particleBegin; particleIndex < particleEnd; ++particleIndex)
    {
        mParticles.positionsInvMass[particleIndex].w =
            softBody.particleMass > 0.0f ? 1.0f / softBody.particleMass : 0.0f;
        mParticles.radii[particleIndex]                   = softBody.particleRadius;
        mParticles.environmentIndices[particleIndex]      = softBody.environmentIndex;
        mParticles.collisionLayers[particleIndex]         = softBody.collisionLayer;
        mParticles.collisionMasks[particleIndex]          = softBody.collisionMask;
        mParticles.phases[particleIndex]                  = phase;
        mParticles.particleMaterialIndices[particleIndex] = softBody.contactMaterialIndex;
    }

    const std::uint32_t edgeBegin = softBody.edgeOffset;
    const std::uint32_t edgeEnd   = std::min<std::uint32_t>(
        edgeBegin + softBody.edgeCount, static_cast<std::uint32_t>(mSoftEdges.size()));
    for (std::uint32_t edgeIndex = edgeBegin; edgeIndex < edgeEnd; ++edgeIndex)
    {
        mSoftEdges[edgeIndex].compliance = softBody.edgeCompliance;
    }

    const std::uint32_t tetBegin = softBody.tetOffset;
    const std::uint32_t tetEnd   = std::min<std::uint32_t>(
        tetBegin + softBody.tetCount, static_cast<std::uint32_t>(mSoftTets.size()));
    for (std::uint32_t tetIndex = tetBegin; tetIndex < tetEnd; ++tetIndex)
    {
        mSoftTets[tetIndex].compliance = softBody.volumeCompliance;
    }

    recomputeParticleGridCellSize();
}

void PhysicsWorld::removeCollidersForEntity(common::EntityId entityId) noexcept
{
    const auto idsIt = mEntityToColliderIds.find(entityId);
    if (idsIt == mEntityToColliderIds.end())
    {
        return;
    }

    while (!idsIt->second.empty())
    {
        const ColliderId colliderId = idsIt->second.back();
        const auto colliderIt       = mColliderIdToIndex.find(colliderId);
        if (colliderIt == mColliderIdToIndex.end())
        {
            idsIt->second.pop_back();
            continue;
        }
        removeColliderAtIndex(colliderIt->second);
    }

    mEntityToColliderIds.erase(entityId);
}

void PhysicsWorld::removeColliderAtIndex(std::uint32_t index) noexcept
{
    if (index >= colliderCount())
    {
        return;
    }

    const std::uint32_t last                = colliderCount() - 1u;
    const ColliderState removed             = mColliderSnapshot[index];
    const std::uint32_t removedContribution = broadPhaseContributionForCollider(removed);
    if (removedContribution == kBroadPhaseContributionStatic)
    {
        --mStaticColliderCount;
    }
    else if (removedContribution == kBroadPhaseContributionMoving)
    {
        --mActiveMovingColliderCount;
    }

    auto removeHandleFromEntity = [&](common::EntityId entityId, ColliderId colliderId)
    {
        const auto handlesIt = mEntityToColliderIds.find(entityId);
        if (handlesIt == mEntityToColliderIds.end())
        {
            return;
        }
        auto &handles = handlesIt->second;
        handles.erase(std::remove(handles.begin(), handles.end(), colliderId), handles.end());
    };

    removeHandleFromEntity(removed.entityId, removed.colliderId);
    mColliderIdToIndex.erase(removed.colliderId);

    if (index != last)
    {
        const ColliderState moved               = mColliderSnapshot[last];
        mColliderSnapshot[index]                = moved;
        mColliders.colliderIds[index]           = mColliders.colliderIds[last];
        mColliders.entityIds[index]             = mColliders.entityIds[last];
        mColliders.ownerRigidBodyIds[index]     = mColliders.ownerRigidBodyIds[last];
        mColliders.ownerRigidBodyIndices[index] = mColliders.ownerRigidBodyIndices[last];
        mColliders.environmentIndices[index]    = mColliders.environmentIndices[last];
        mColliders.shapeTypes[index]            = mColliders.shapeTypes[last];
        mColliders.shapeParams[index]           = mColliders.shapeParams[last];
        mColliders.localPositions[index]        = mColliders.localPositions[last];
        mColliders.localOrientations[index]     = mColliders.localOrientations[last];
        mColliders.enabledFlags[index]          = mColliders.enabledFlags[last];
        mColliders.frictionRestitution[index]   = mColliders.frictionRestitution[last];
        mColliders.collisionLayers[index]       = mColliders.collisionLayers[last];
        mColliders.collisionMasks[index]        = mColliders.collisionMasks[last];
        mColliderIdToIndex[moved.colliderId]    = index;
        markColliderDirty(index);
    }

    mColliderSnapshot.pop_back();
    mColliders.colliderIds.pop_back();
    mColliders.entityIds.pop_back();
    mColliders.ownerRigidBodyIds.pop_back();
    mColliders.ownerRigidBodyIndices.pop_back();
    mColliders.environmentIndices.pop_back();
    mColliders.shapeTypes.pop_back();
    mColliders.shapeParams.pop_back();
    mColliders.localPositions.pop_back();
    mColliders.localOrientations.pop_back();
    mColliders.enabledFlags.pop_back();
    mColliders.frictionRestitution.pop_back();
    mColliders.collisionLayers.pop_back();
    mColliders.collisionMasks.pop_back();
}

std::uint32_t PhysicsWorld::broadPhaseContributionForCollider(
    const ColliderState &collider) const noexcept
{
    const auto bodyIt = mRigidBodyIdToIndex.find(collider.ownerRigidBodyId);
    const RigidBodyState *owner =
        bodyIt != mRigidBodyIdToIndex.end() ? &mRigidBodySnapshot[bodyIt->second] : nullptr;
    return colliderBroadPhaseContribution(collider, owner);
}

std::uint32_t PhysicsWorld::enabledColliderCountForEntity(common::EntityId entityId) const noexcept
{
    const auto handlesIt = mEntityToColliderIds.find(entityId);
    if (handlesIt == mEntityToColliderIds.end())
    {
        return 0u;
    }

    std::uint32_t count = 0u;
    for (const ColliderId colliderId : handlesIt->second)
    {
        const auto colliderIt = mColliderIdToIndex.find(colliderId);
        if (colliderIt == mColliderIdToIndex.end())
        {
            continue;
        }
        if (mColliderSnapshot[colliderIt->second].enabled)
        {
            ++count;
        }
    }
    return count;
}

bool PhysicsWorld::pruneRigidJointsForBody(RigidBodyId rigidBodyId) noexcept
{
    const auto removeByBody = [rigidBodyId](auto &container) noexcept
    {
        const auto originalSize = container.size();
        container.erase(
            std::remove_if(container.begin(), container.end(), [rigidBodyId](const auto &joint)
                           { return joint.bodyA == rigidBodyId || joint.bodyB == rigidBodyId; }),
            container.end());
        return container.size() != originalSize;
    };

    const bool removedBall   = removeByBody(mBallJointSnapshot);
    const bool removedHinge  = removeByBody(mHingeJointSnapshot);
    const bool removedSlider = removeByBody(mSliderJointSnapshot);
    return removedBall || removedHinge || removedSlider;
}

void PhysicsWorld::rebuildBodyColliderMapping() const noexcept
{
    mBodyColliderMapping.colliderOffsets.assign(rigidBodyCount(), 0u);
    mBodyColliderMapping.colliderCounts.assign(rigidBodyCount(), 0u);
    mBodyColliderMapping.colliderIndices.assign(colliderCount(), 0u);

    for (std::uint32_t colliderIndex = 0; colliderIndex < colliderCount(); ++colliderIndex)
    {
        const RigidBodyId ownerRigidBodyId = mColliderSnapshot[colliderIndex].ownerRigidBodyId;
        const auto bodyIt                  = mRigidBodyIdToIndex.find(ownerRigidBodyId);
        const std::uint32_t bodyIndex =
            bodyIt != mRigidBodyIdToIndex.end() ? bodyIt->second : 0xffffffffu;
        mColliders.ownerRigidBodyIndices[colliderIndex] = bodyIndex;
        mColliders.environmentIndices[colliderIndex] =
            bodyIndex != 0xffffffffu ? mRigidBodySnapshot[bodyIndex].environmentIndex : 0u;
        if (bodyIndex != 0xffffffffu)
        {
            ++mBodyColliderMapping.colliderCounts[bodyIndex];
        }
    }

    std::uint32_t runningOffset = 0u;
    for (std::uint32_t bodyIndex = 0; bodyIndex < rigidBodyCount(); ++bodyIndex)
    {
        mBodyColliderMapping.colliderOffsets[bodyIndex] = runningOffset;
        runningOffset += mBodyColliderMapping.colliderCounts[bodyIndex];
    }

    std::vector<std::uint32_t> cursor = mBodyColliderMapping.colliderOffsets;
    for (std::uint32_t colliderIndex = 0; colliderIndex < colliderCount(); ++colliderIndex)
    {
        const std::uint32_t bodyIndex = mColliders.ownerRigidBodyIndices[colliderIndex];
        if (bodyIndex == 0xffffffffu || bodyIndex >= rigidBodyCount())
        {
            continue;
        }
        mBodyColliderMapping.colliderIndices[cursor[bodyIndex]++] = colliderIndex;
    }
}

void PhysicsWorld::rebuildRigidJointScene() const noexcept
{
    auto *self = const_cast<PhysicsWorld *>(this);
    self->mRigidJointScene.clear();

    auto appendBall = [&](const BallJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        self->mRigidJointScene.ball.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.ball.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.ball.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.ball.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.ball.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
    };

    auto appendHinge = [&](const HingeJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        Diligent::float4 projectionRows[3];
        computeProjectionRowsFromLocalFrames(joint.localRotationA, joint.localRotationB, 1u, 3u,
                                             projectionRows);
        const Diligent::float3 axisA0 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{1.0f, 0.0f, 0.0f}),
            Diligent::float3{1.0f, 0.0f, 0.0f});

        self->mRigidJointScene.hinge.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.hinge.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.hinge.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.hinge.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.hinge.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.hinge.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.hinge.localAxesA0.push_back(toFloat4(axisA0, 0.0f));
        self->mRigidJointScene.hinge.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.hinge.limitMins.push_back(joint.limitMin);
        self->mRigidJointScene.hinge.limitMaxs.push_back(joint.limitMax);
        self->mRigidJointScene.hinge.constraintCompliances.push_back(joint.constraintCompliance);
        self->mRigidJointScene.hinge.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.hinge.driveTargetAngles.push_back(joint.driveTargetAngle);
        self->mRigidJointScene.hinge.driveTargetAngularVelocities.push_back(
            joint.driveTargetAngularVelocity);
        self->mRigidJointScene.hinge.projectionRow0.push_back(projectionRows[0]);
        self->mRigidJointScene.hinge.projectionRow1.push_back(projectionRows[1]);
        self->mRigidJointScene.hinge.projectionRow2.push_back(projectionRows[2]);
    };

    auto appendSlider = [&](const SliderJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        const Diligent::float3 axisA0 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{1.0f, 0.0f, 0.0f}),
            Diligent::float3{1.0f, 0.0f, 0.0f});
        const Diligent::float3 axisA1 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{0.0f, 1.0f, 0.0f}),
            Diligent::float3{0.0f, 1.0f, 0.0f});
        const Diligent::float3 axisA2 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{0.0f, 0.0f, 1.0f}),
            Diligent::float3{0.0f, 0.0f, 1.0f});
        const Diligent::float3 worldAnchorA =
            self->mRigidBodySnapshot[bodyIndexA].position +
            quaternionRotate(self->mRigidBodySnapshot[bodyIndexA].rotation, joint.localAnchorA);
        const Diligent::float3 worldAnchorB =
            self->mRigidBodySnapshot[bodyIndexB].position +
            quaternionRotate(self->mRigidBodySnapshot[bodyIndexB].rotation, joint.localAnchorB);
        const float driveRestOffset = Diligent::dot(worldAnchorA - worldAnchorB, axisA0);
        Diligent::float4 projectionRows[3];
        computeProjectionRowsFromLocalFrames(joint.localRotationA, joint.localRotationB, 1u, 3u,
                                             projectionRows);
        self->mRigidJointScene.slider.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.slider.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.slider.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.slider.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.slider.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.slider.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.slider.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.slider.limitMins.push_back(joint.limitMin);
        self->mRigidJointScene.slider.limitMaxs.push_back(joint.limitMax);
        self->mRigidJointScene.slider.constraintCompliances.push_back(joint.constraintCompliance);
        self->mRigidJointScene.slider.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.slider.driveTargetPositions.push_back(joint.driveTargetPosition);
        self->mRigidJointScene.slider.driveTargetVelocities.push_back(joint.driveTargetVelocity);
        self->mRigidJointScene.slider.driveRestOffsets.push_back(driveRestOffset);
        self->mRigidJointScene.slider.localAxesA0.push_back(toFloat4(axisA0, 0.0f));
        self->mRigidJointScene.slider.localAxesA1.push_back(toFloat4(axisA1, 0.0f));
        self->mRigidJointScene.slider.localAxesA2.push_back(toFloat4(axisA2, 0.0f));
        self->mRigidJointScene.slider.projectionRow0.push_back(projectionRows[0]);
        self->mRigidJointScene.slider.projectionRow1.push_back(projectionRows[1]);
        self->mRigidJointScene.slider.projectionRow2.push_back(projectionRows[2]);
    };

    for (const BallJointState &joint : self->mBallJointSnapshot)
    {
        appendBall(joint);
    }
    for (const HingeJointState &joint : self->mHingeJointSnapshot)
    {
        appendHinge(joint);
    }
    for (const SliderJointState &joint : self->mSliderJointSnapshot)
    {
        appendSlider(joint);
    }
}

void PhysicsWorld::rebuildJointCollisionSuppression() const noexcept
{
    auto *self = const_cast<PhysicsWorld *>(this);
    self->mJointCollisionSuppression.clear();
    self->mJointCollisionSuppression.neighborOffsets.assign(rigidBodyCount() + 1u, 0u);
    if (rigidBodyCount() == 0u)
    {
        return;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> directedPairs;
    directedPairs.reserve(
        (mBallJointSnapshot.size() + mHingeJointSnapshot.size() + mSliderJointSnapshot.size()) *
        2u);

    const auto appendPair = [&](RigidBodyId bodyAId, RigidBodyId bodyBId)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(bodyAId);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(bodyBId);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }

        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (bodyIndexA == bodyIndexB || self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
                                            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        directedPairs.emplace_back(bodyIndexA, bodyIndexB);
        directedPairs.emplace_back(bodyIndexB, bodyIndexA);
    };

    for (const BallJointState &joint : mBallJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }
    for (const HingeJointState &joint : mHingeJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }
    for (const SliderJointState &joint : mSliderJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }

    std::sort(directedPairs.begin(), directedPairs.end());
    directedPairs.erase(std::unique(directedPairs.begin(), directedPairs.end()),
                        directedPairs.end());

    for (const auto &[bodyIndexA, bodyIndexB] : directedPairs)
    {
        ++self->mJointCollisionSuppression.neighborOffsets[bodyIndexA + 1u];
        self->mJointCollisionSuppression.neighbors.push_back(bodyIndexB);
    }

    for (std::uint32_t bodyIndex = 0u; bodyIndex < rigidBodyCount(); ++bodyIndex)
    {
        self->mJointCollisionSuppression.neighborOffsets[bodyIndex + 1u] +=
            self->mJointCollisionSuppression.neighborOffsets[bodyIndex];
    }
}

void PhysicsWorld::rebuildSoftBodyDerivedState() noexcept
{
    mParticles.clear();
    mParticleContactMaterials.clear();
    mFluidMaterials.clear();
    mSoftEdges.clear();
    mSoftTets.clear();
    std::vector<std::vector<std::uint32_t>> adjacencyLists;

    for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
         ++softBodyIndex)
    {
        SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
        normalizeSoftBodyState(softBody);
        softBody.contactMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, toParticleContactMaterial(softBody.material.contact));

        softBody.particleOffset = static_cast<std::uint32_t>(mParticles.size());
        softBody.edgeOffset     = static_cast<std::uint32_t>(mSoftEdges.size());
        softBody.tetOffset      = static_cast<std::uint32_t>(mSoftTets.size());

        if (softBodyIndex >= mSoftBodyDerivedCaches.size())
        {
            CRESSIM_LOG_ERROR("Failed to rebuild derived soft-body state for entity ",
                              softBody.entityId, ": missing cached resolved topology.");
            softBody.particleCount = 0u;
            softBody.edgeCount     = 0u;
            softBody.tetCount      = 0u;
            continue;
        }
        const SoftBodyDerivedCache &topology = mSoftBodyDerivedCaches[softBodyIndex];

        const float inverseMass =
            softBody.particleMass > 0.0f ? 1.0f / softBody.particleMass : 0.0f;
        std::unordered_set<std::uint32_t> staticParticles(topology.staticParticleIndices.begin(),
                                                          topology.staticParticleIndices.end());
        softBody.restPositions = topology.restPositions;
        softBody.boundaryFaces = topology.boundaryFaces;

        softBody.particleCount = static_cast<std::uint32_t>(topology.restPositions.size());
        for (std::uint32_t localParticleIndex = 0u; localParticleIndex < softBody.particleCount;
             ++localParticleIndex)
        {
            const Diligent::float3 &position = topology.restPositions[localParticleIndex];
            const float particleInvMass =
                staticParticles.find(localParticleIndex) != staticParticles.end() ? 0.0f
                                                                                  : inverseMass;

            mParticles.positionsInvMass.push_back(
                Diligent::float4{position.x, position.y, position.z, particleInvMass});
            mParticles.previousPositions.push_back(
                Diligent::float4{position.x, position.y, position.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(softBody.particleRadius);
            mParticles.environmentIndices.push_back(softBody.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::SoftSolid));
            mParticles.ownerTypes.push_back(
                static_cast<std::uint32_t>(ParticleOwnerType::SoftBody));
            mParticles.ownerIndices.push_back(softBodyIndex);
            mParticles.owningSoftBodyIndices.push_back(softBodyIndex);
            mParticles.particleMaterialIndices.push_back(softBody.contactMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(0xffffffffu);
            mParticles.phases.push_back(
                packParticlePhase(softBodyIndex, softBody.selfCollisionEnabled));
            mParticles.collisionLayers.push_back(softBody.collisionLayer);
            mParticles.collisionMasks.push_back(softBody.collisionMask);
            adjacencyLists.emplace_back();
        }

        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(topology.adjacencyLists.size());
             ++localParticleIndex)
        {
            auto &globalAdjacency = adjacencyLists[softBody.particleOffset + localParticleIndex];
            globalAdjacency.reserve(topology.adjacencyLists[localParticleIndex].size());
            for (const std::uint32_t localNeighbor : topology.adjacencyLists[localParticleIndex])
            {
                globalAdjacency.push_back(softBody.particleOffset + localNeighbor);
            }
        }

        for (const auto &edgeDesc : topology.edges)
        {
            const std::uint32_t globalA = softBody.particleOffset + edgeDesc[0];
            const std::uint32_t globalB = softBody.particleOffset + edgeDesc[1];
            const Diligent::float3 delta{
                topology.restPositions[edgeDesc[1]].x - topology.restPositions[edgeDesc[0]].x,
                topology.restPositions[edgeDesc[1]].y - topology.restPositions[edgeDesc[0]].y,
                topology.restPositions[edgeDesc[1]].z - topology.restPositions[edgeDesc[0]].z,
            };

            SoftEdge edge{};
            edge.particleA  = globalA;
            edge.particleB  = globalB;
            edge.restLength = std::sqrt(Diligent::dot(delta, delta));
            edge.compliance = softBody.edgeCompliance;
            mSoftEdges.push_back(edge);
        }

        for (const auto &tetDesc : topology.tets)
        {
            const Diligent::float3 &p0 = topology.restPositions[tetDesc[0]];
            const Diligent::float3 &p1 = topology.restPositions[tetDesc[1]];
            const Diligent::float3 &p2 = topology.restPositions[tetDesc[2]];
            const Diligent::float3 &p3 = topology.restPositions[tetDesc[3]];

            SoftTet tet{};
            tet.particleIndices = {
                softBody.particleOffset + tetDesc[0], softBody.particleOffset + tetDesc[1],
                softBody.particleOffset + tetDesc[2], softBody.particleOffset + tetDesc[3]};
            tet.restVolume =
                std::abs(Diligent::dot(Diligent::cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0f;
            tet.compliance = softBody.volumeCompliance;
            mSoftTets.push_back(tet);
        }

        softBody.edgeCount = static_cast<std::uint32_t>(mSoftEdges.size()) - softBody.edgeOffset;
        softBody.tetCount  = static_cast<std::uint32_t>(mSoftTets.size()) - softBody.tetOffset;
    }

    for (std::uint32_t fluidIndex = 0u; fluidIndex < mFluidSnapshot.size(); ++fluidIndex)
    {
        FluidState &fluid = mFluidSnapshot[fluidIndex];
        normalizeFluidState(fluid);
        fluid.contactMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, toParticleContactMaterial(fluid.material.contact));
        fluid.fluidMaterialIndex = findOrAppendFluidMaterial(
            mFluidMaterials, toFluidSolverMaterial(fluid.material, fluid.particleRadius));
        fluid.particleOffset = static_cast<std::uint32_t>(mParticles.size());

        if (fluidIndex >= mFluidDerivedCaches.size())
        {
            fluid.particleCount = 0u;
            continue;
        }

        const FluidDerivedCache &cache = mFluidDerivedCaches[fluidIndex];
        fluid.restPositions            = cache.restPositions;
        fluid.particleCount            = static_cast<std::uint32_t>(cache.restPositions.size());
        const float inverseMass = fluid.particleMass > 0.0f ? 1.0f / fluid.particleMass : 0.0f;
        for (const Diligent::float3 &position : cache.restPositions)
        {
            mParticles.positionsInvMass.push_back(
                Diligent::float4{position.x, position.y, position.z, inverseMass});
            mParticles.previousPositions.push_back(
                Diligent::float4{position.x, position.y, position.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(fluid.particleRadius);
            mParticles.environmentIndices.push_back(fluid.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::Fluid));
            mParticles.ownerTypes.push_back(
                static_cast<std::uint32_t>(ParticleOwnerType::FluidBody));
            mParticles.ownerIndices.push_back(fluidIndex);
            mParticles.owningSoftBodyIndices.push_back(0xffffffffu);
            mParticles.particleMaterialIndices.push_back(fluid.contactMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(fluid.fluidMaterialIndex);
            mParticles.phases.push_back(
                packParticlePhase(mSoftBodySnapshot.size() + fluidIndex, true));
            mParticles.collisionLayers.push_back(fluid.collisionLayer);
            mParticles.collisionMasks.push_back(fluid.collisionMask);
            adjacencyLists.emplace_back();
        }
    }

    mParticles.adjacencyOffsets.resize(adjacencyLists.size());
    mParticles.adjacencyCounts.resize(adjacencyLists.size());
    mParticles.adjacencyIndices.clear();
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(adjacencyLists.size()); ++particleIndex)
    {
        auto &neighbors = adjacencyLists[particleIndex];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        mParticles.adjacencyOffsets[particleIndex] =
            static_cast<std::uint32_t>(mParticles.adjacencyIndices.size());
        mParticles.adjacencyCounts[particleIndex] = static_cast<std::uint32_t>(neighbors.size());
        mParticles.adjacencyIndices.insert(mParticles.adjacencyIndices.end(), neighbors.begin(),
                                           neighbors.end());
    }

    recomputeParticleGridCellSize();
    mSoftBodyDerivedStateDirty = false;
}

void PhysicsWorld::recomputeParticleGridCellSize() noexcept
{
    float gridCellSize = 0.0f;
    for (std::size_t particleIndex = 0; particleIndex < mParticles.radii.size(); ++particleIndex)
    {
        gridCellSize = std::max(gridCellSize, mParticles.radii[particleIndex] * 2.0f);
        if (particleIndex < mParticles.fluidMaterialIndices.size() &&
            mParticles.fluidMaterialIndices[particleIndex] != 0xffffffffu &&
            mParticles.fluidMaterialIndices[particleIndex] < mFluidMaterials.size())
        {
            gridCellSize = std::max(
                gridCellSize,
                mFluidMaterials[mParticles.fluidMaterialIndices[particleIndex]].smoothingRadius);
        }
    }

    mParticleGridCellSize = std::max(gridCellSize, 0.1f);
}

void PhysicsWorld::recomputeSoftBodyBoundsChunkCount() noexcept
{
    mSoftBodyBoundsChunkCount = 0u;
    for (const Diligent::uint2 &range : mSoftRenderData.softBodyParticleRanges)
    {
        mSoftBodyBoundsChunkCount += (range.y + 64u - 1u) / 64u;
    }
}

bool PhysicsWorld::prepareFluidStateForInsert(const FluidState &candidate,
                                              FluidDerivedCache &derivedCache) noexcept
{
    if (candidate.source.kind != FluidSourceKind::RegularGrid)
    {
        return false;
    }

    const Diligent::float3 size = candidate.source.regularGrid.size;
    const float spacing         = candidate.source.regularGrid.targetParticleSpacing;
    const std::uint32_t nx      = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.x / spacing + 0.5f)));
    const std::uint32_t ny = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.y / spacing + 0.5f)));
    const std::uint32_t nz = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.z / spacing + 0.5f)));

    derivedCache.restPositions.clear();
    derivedCache.restPositions.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (std::uint32_t z = 0u; z < nz; ++z)
    {
        for (std::uint32_t y = 0u; y < ny; ++y)
        {
            for (std::uint32_t x = 0u; x < nx; ++x)
            {
                const Diligent::float3 objectSpacePosition{
                    -size.x * 0.5f + (static_cast<float>(x) + 0.5f) * spacing,
                    -size.y * 0.5f + (static_cast<float>(y) + 0.5f) * spacing,
                    -size.z * 0.5f + (static_cast<float>(z) + 0.5f) * spacing};
                derivedCache.restPositions.push_back(common::runtime_math::applyTransform(
                    candidate.restTransform, objectSpacePosition));
            }
        }
    }
    return true;
}

bool PhysicsWorld::prepareSoftBodyStateForInsert(const SoftBodyState &candidate,
                                                 const SoftBodyState *previousState,
                                                 SoftBodyDerivedCache &derivedCache) noexcept
{
    TetMeshData tetGenCache;
    TetGenMeshCache cacheEntry;
    const TetMeshData *tetGenCachePtr = nullptr;
    if (candidate.source.kind == SoftBodySourceKind::TetGenFiles)
    {
        bool needReload = true;
        if (previousState != nullptr &&
            previousState->source.kind == SoftBodySourceKind::TetGenFiles)
        {
            const auto &before = previousState->source.tetGen;
            const auto &after  = candidate.source.tetGen;
            if (before.nodeFile == after.nodeFile && before.eleFile == after.eleFile)
            {
                if (const TetGenMeshCache *existing = tryGetTetGenMeshCache(candidate.entityId))
                {
                    cacheEntry = *existing;
                    needReload = false;
                }
            }
        }

        std::string errorMessage;
        if (needReload)
        {
            if (!loadTetGenFiles(candidate.source.tetGen.nodeFile, candidate.source.tetGen.eleFile,
                                 tetGenCache, errorMessage))
            {
                CRESSIM_LOG_ERROR("Failed to author soft body for entity ", candidate.entityId,
                                  ": ", errorMessage);
                return false;
            }
            cacheEntry.nodeFile                 = candidate.source.tetGen.nodeFile;
            cacheEntry.eleFile                  = candidate.source.tetGen.eleFile;
            cacheEntry.objectSpaceRestPositions = tetGenCache.objectSpaceRestPositions;
            cacheEntry.tetVertexIndices         = tetGenCache.tetVertexIndices;
        }
        else
        {
            tetGenCache.objectSpaceRestPositions = cacheEntry.objectSpaceRestPositions;
            tetGenCache.tetVertexIndices         = cacheEntry.tetVertexIndices;
        }
        tetGenCachePtr = &tetGenCache;
    }

    std::string errorMessage;
    ResolvedSoftBodyTopology resolvedTopology;
    if (!resolveSoftBodyTopology(candidate, tetGenCachePtr, resolvedTopology, errorMessage))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for entity ", candidate.entityId, ": ",
                          errorMessage);
        return false;
    }

    derivedCache.restPositions         = std::move(resolvedTopology.restPositions);
    derivedCache.edges                 = std::move(resolvedTopology.edges);
    derivedCache.tets                  = std::move(resolvedTopology.tets);
    derivedCache.boundaryFaces         = std::move(resolvedTopology.boundaryFaces);
    derivedCache.adjacencyLists        = std::move(resolvedTopology.adjacencyLists);
    derivedCache.staticParticleIndices = std::move(resolvedTopology.staticParticleIndices);

    if (candidate.source.kind == SoftBodySourceKind::TetGenFiles)
    {
        mTetGenMeshCache[candidate.entityId] = std::move(cacheEntry);
    }
    else
    {
        mTetGenMeshCache.erase(candidate.entityId);
    }

    return true;
}

const PhysicsWorld::TetGenMeshCache *PhysicsWorld::tryGetTetGenMeshCache(
    common::EntityId entityId) const noexcept
{
    const auto it = mTetGenMeshCache.find(entityId);
    return it == mTetGenMeshCache.end() ? nullptr : &it->second;
}

void PhysicsWorld::markAllRigidBodiesDirty() noexcept
{
    mFullRigidBodyUploadRequired = true;
    mRigidBodyCountDirty         = true;
    mRigidBodyDirtyBits.assign(rigidBodyCount(), 1u);
    mRigidBodyDirtyIndices.resize(rigidBodyCount());
    for (std::uint32_t i = 0; i < rigidBodyCount(); ++i)
    {
        mRigidBodyDirtyIndices[i] = i;
    }
}

void PhysicsWorld::markAllCollidersDirty() noexcept
{
    mFullColliderUploadRequired = true;
    mColliderCountDirty         = true;
    mColliderDirtyBits.assign(colliderCount(), 1u);
    mColliderDirtyIndices.resize(colliderCount());
    for (std::uint32_t i = 0; i < colliderCount(); ++i)
    {
        mColliderDirtyIndices[i] = i;
    }
}

void PhysicsWorld::markRigidBodyDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mRigidBodyDirtyIndices, mRigidBodyDirtyBits);
}

void PhysicsWorld::markColliderDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mColliderDirtyIndices, mColliderDirtyBits);
}

void PhysicsWorld::markRigidBodyCountDirty(bool fullUploadRequired) noexcept
{
    mRigidBodyCountDirty         = true;
    mFullRigidBodyUploadRequired = mFullRigidBodyUploadRequired || fullUploadRequired;
}

void PhysicsWorld::markColliderCountDirty(bool fullUploadRequired) noexcept
{
    mColliderCountDirty         = true;
    mFullColliderUploadRequired = mFullColliderUploadRequired || fullUploadRequired;
}

void PhysicsWorld::applyRigidJointChange(RigidJointChangeKind changeKind) noexcept
{
    switch (changeKind)
    {
    case RigidJointChangeKind::PayloadOnly:
        markJointSceneDirty();
        return;
    case RigidJointChangeKind::ModeRebuild:
        markJointModeDirty();
        return;
    case RigidJointChangeKind::TopologyRebuild:
        markJointTopologyDirty();
        return;
    }
}

void PhysicsWorld::markJointSceneDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::markJointModeDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::markJointTopologyDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mAuthoredRevision;
}

} // namespace cressim::neo::physics
