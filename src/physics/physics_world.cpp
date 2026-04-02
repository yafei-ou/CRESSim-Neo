#include "physics/physics_world.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace cressim::neo::physics
{

namespace
{

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
    return Diligent::float4{state.friction, state.restitution, 0.0f, 0.0f};
}

std::uint32_t flattenGridIndex(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                               const UInt3 &resolution)
{
    return x * resolution.y * resolution.z + y * resolution.z + z;
}

Diligent::float3 rotatePoint(const Diligent::QuaternionF &rotation, const Diligent::float3 &point)
{
    return rotation.RotateVector(point);
}

Diligent::float3 transformPoint(const Diligent::float3 &position,
                                const Diligent::QuaternionF &rotation,
                                const Diligent::float3 &point)
{
    return position + rotatePoint(rotation, point);
}

float tetSignedVolume(const Diligent::float3 &a, const Diligent::float3 &b,
                      const Diligent::float3 &c, const Diligent::float3 &d)
{
    return Diligent::dot(Diligent::cross(b - a, c - a), d - a) / 6.0f;
}

std::uint64_t makeSortedEdgeKey(std::uint32_t a, std::uint32_t b)
{
    const std::uint32_t lo = std::min(a, b);
    const std::uint32_t hi = std::max(a, b);
    return (static_cast<std::uint64_t>(lo) << 32u) | hi;
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

void PhysicsWorld::clear()
{
    mRigidBodies.clear();
    mColliders.clear();
    mBodyColliderMapping.clear();
    mEntityToRigidBodyIndex.clear();
    mRigidBodyIdToIndex.clear();
    mColliderIdToIndex.clear();
    mEntityToColliderIds.clear();
    mEntityToSoftBodyIndex.clear();
    mRigidBodySnapshot.clear();
    mColliderSnapshot.clear();
    mSoftBodySnapshot.clear();
    mSoftParticles.clear();
    mSoftEdges.clear();
    mSoftTets.clear();
    mRigidSurfaceParticles.clear();
    mRigidBodyDirtyIndices.clear();
    mColliderDirtyIndices.clear();
    mRigidBodyDirtyBits.clear();
    mColliderDirtyBits.clear();
    mRigidBodyCountDirty         = true;
    mColliderCountDirty          = true;
    mFullRigidBodyUploadRequired = true;
    mFullColliderUploadRequired  = true;
    mBodyColliderMappingDirty    = true;
    mSoftBodyDerivedStateDirty   = true;
    mRigidSurfaceParticlesDirty  = true;
    mStaticBroadPhaseDirty       = true;
    ++mRigidBodyTopologyRevision;
    ++mSoftBodyTopologyRevision;
    ++mRevision;
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
        mBodyColliderMappingDirty   = true;
        mRigidSurfaceParticlesDirty = true;
        mStaticBroadPhaseDirty      = mStaticBroadPhaseDirty || isStaticBody(normalizedState);
        ++mRigidBodyTopologyRevision;
        ++mRevision;
        return mRigidBodySnapshot.back();
    }

    const std::uint32_t index          = it->second;
    normalizedState.rigidBodyId        = mRigidBodySnapshot[index].rigidBodyId;
    const RigidBodyState previousState = mRigidBodySnapshot[index];
    writeRigidBodySoAAt(mRigidBodies, index, normalizedState);
    mRigidBodySnapshot[index] = normalizedState;
    markRigidBodyDirty(index);
    mStaticBroadPhaseDirty =
        mStaticBroadPhaseDirty || staticBodyPoseChanged(previousState, normalizedState);
    mRigidSurfaceParticlesDirty = true;
    ++mRevision;
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

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mRigidBodies.size() - 1u);
    const bool removedStatic  = isStaticBody(mRigidBodySnapshot[index]);

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
    mRigidBodyIdToIndex.erase(mRigidBodies.rigidBodyIds[last]);
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
    mBodyColliderMappingDirty   = true;
    mRigidSurfaceParticlesDirty = true;
    mStaticBroadPhaseDirty      = mStaticBroadPhaseDirty || removedStatic;
    ++mRigidBodyTopologyRevision;
    ++mRevision;
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
        writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex);
        mColliderIdToIndex.emplace(normalizedState.colliderId, colliderIndex);
        auto &entityColliderIds = mEntityToColliderIds[normalizedState.entityId];
        entityColliderIds.push_back(normalizedState.colliderId);
        markColliderDirty(colliderIndex);
        markColliderCountDirty();
        mBodyColliderMappingDirty   = true;
        mRigidSurfaceParticlesDirty = true;
        mStaticBroadPhaseDirty      = mStaticBroadPhaseDirty || ownerIsStatic;
        ++mRevision;
        return;
    }

    const std::uint32_t colliderIndex = colliderIt->second;
    const ColliderState previousState = mColliderSnapshot[colliderIndex];
    writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex);
    mColliderSnapshot[colliderIndex] = normalizedState;
    markColliderDirty(colliderIndex);
    if (previousState.shapeType != normalizedState.shapeType ||
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
        previousState.enabled != normalizedState.enabled)
    {
        mRigidSurfaceParticlesDirty = true;
    }
    if (previousState.ownerRigidBodyId != normalizedState.ownerRigidBodyId)
    {
        mBodyColliderMappingDirty   = true;
        mRigidSurfaceParticlesDirty = true;
    }
    if (ownerIsStatic && (previousState.shapeType != normalizedState.shapeType ||
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
                          previousState.enabled != normalizedState.enabled))
    {
        mStaticBroadPhaseDirty = true;
    }
    ++mRevision;
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
    mBodyColliderMappingDirty   = true;
    mRigidSurfaceParticlesDirty = true;
    mStaticBroadPhaseDirty      = mStaticBroadPhaseDirty || removedStaticOwner;
    ++mRevision;
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
        ++mRevision;
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
        writeColliderSoAAt(mColliders, colliderIndex, collider, ownerBodyIndex);
        mColliderIdToIndex.emplace(collider.colliderId, colliderIndex);
        entityColliderIds.push_back(collider.colliderId);
    }

    markColliderCountDirty(true);
    mBodyColliderMappingDirty   = true;
    mRigidSurfaceParticlesDirty = true;
    mStaticBroadPhaseDirty      = true;
    ++mRevision;
}

SoftBodyState &PhysicsWorld::upsertSoftBody(const SoftBodyState &state)
{
    SoftBodyState normalizedState = state;
    normalizeSoftBodyState(normalizedState);

    const auto it = mEntityToSoftBodyIndex.find(normalizedState.entityId);
    if (it == mEntityToSoftBodyIndex.end())
    {
        mEntityToSoftBodyIndex.emplace(normalizedState.entityId,
                                       static_cast<std::uint32_t>(mSoftBodySnapshot.size()));
        mSoftBodySnapshot.push_back(normalizedState);
    }
    else
    {
        mSoftBodySnapshot[it->second] = normalizedState;
    }

    rebuildSoftBodyDerivedState();
    mRigidSurfaceParticlesDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mRevision;
    return mSoftBodySnapshot[mEntityToSoftBodyIndex[normalizedState.entityId]];
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
        mSoftBodySnapshot[index]                                  = mSoftBodySnapshot[last];
        mEntityToSoftBodyIndex[mSoftBodySnapshot[index].entityId] = index;
    }
    mSoftBodySnapshot.pop_back();
    mEntityToSoftBodyIndex.erase(it);
    rebuildSoftBodyDerivedState();
    mRigidSurfaceParticlesDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mRevision;
    return true;
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

const SoftParticleSoAHost &PhysicsWorld::softParticles() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftParticles;
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

const RigidSurfaceParticleSoAHost &PhysicsWorld::rigidSurfaceParticles() const noexcept
{
    if (mRigidSurfaceParticlesDirty)
    {
        rebuildRigidSurfaceParticles();
        mRigidSurfaceParticlesDirty = false;
    }
    return mRigidSurfaceParticles;
}

void PhysicsWorld::ensureDerivedStateUpToDate() const noexcept
{
    if (mBodyColliderMappingDirty)
    {
        rebuildBodyColliderMapping();
        mBodyColliderMappingDirty = false;
    }
    if (mRigidSurfaceParticlesDirty)
    {
        rebuildRigidSurfaceParticles();
        mRigidSurfaceParticlesDirty = false;
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
    ++mRevision;
}

bool PhysicsWorld::writeBackRigidBodyState(std::uint32_t index,
                                           const Diligent::float4 &positionInvMass,
                                           const Diligent::float4 &orientation,
                                           const Diligent::float4 &linearVelocity,
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
    markRigidBodyDirty(index);

    RigidBodyState &state = mRigidBodySnapshot[index];
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
    ++mRevision;
}

bool PhysicsWorld::writeBackSoftParticleState(std::uint32_t index,
                                              const Diligent::float4 &positionInvMass,
                                              const Diligent::float4 &previousPosition,
                                              const Diligent::float4 &velocity) noexcept
{
    if (index >= mSoftParticles.size())
    {
        return false;
    }

    mSoftParticles.positionsInvMass[index]  = positionInvMass;
    mSoftParticles.previousPositions[index] = previousPosition;
    mSoftParticles.velocities[index]        = velocity;
    return true;
}

void PhysicsWorld::finalizeSoftParticleWriteback() noexcept
{
    if (mSoftParticles.empty())
    {
        return;
    }
    ++mRevision;
}

std::uint64_t PhysicsWorld::revision() const noexcept
{
    return mRevision;
}

std::uint64_t PhysicsWorld::rigidBodyTopologyRevision() const noexcept
{
    return mRigidBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::softBodyTopologyRevision() const noexcept
{
    return mSoftBodyTopologyRevision;
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
                                      const ColliderState &state, std::uint32_t ownerBodyIndex)
{
    if (index == soa.size())
    {
        soa.colliderIds.push_back(state.colliderId);
        soa.entityIds.push_back(state.entityId);
        soa.ownerRigidBodyIds.push_back(state.ownerRigidBodyId);
        soa.ownerRigidBodyIndices.push_back(ownerBodyIndex);
        soa.environmentIndices.push_back(state.environmentIndex);
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
    soa.environmentIndices[index]    = state.environmentIndex;
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
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeSoftBodyState(SoftBodyState &state) noexcept
{
    state.size.x          = std::max(state.size.x, 1.0e-3f);
    state.size.y          = std::max(state.size.y, 1.0e-3f);
    state.size.z          = std::max(state.size.z, 1.0e-3f);
    state.particleSpacing = std::max(state.particleSpacing, 1.0e-4f);
    state.particleMass    = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius  = std::max(state.particleRadius, 1.0e-4f);

    const auto deriveResolution = [&](float extent) -> std::uint32_t
    {
        return std::max<std::uint32_t>(
            2u, static_cast<std::uint32_t>(std::floor(extent / state.particleSpacing)) + 1u);
    };
    state.gridResolution.x = deriveResolution(state.size.x);
    state.gridResolution.y = deriveResolution(state.size.y);
    state.gridResolution.z = deriveResolution(state.size.z);
}

float PhysicsWorld::referenceParticleSpacing() const noexcept
{
    float spacing = std::numeric_limits<float>::max();
    for (const SoftBodyState &softBody : mSoftBodySnapshot)
    {
        spacing = std::min(spacing, std::max(softBody.particleSpacing, 1.0e-4f));
    }

    if (!std::isfinite(spacing) || spacing <= 0.0f)
    {
        return 0.25f;
    }
    return spacing;
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

    const std::uint32_t last    = colliderCount() - 1u;
    const ColliderState removed = mColliderSnapshot[index];

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

void PhysicsWorld::rebuildSoftBodyDerivedState() noexcept
{
    mSoftParticles.clear();
    mSoftEdges.clear();
    mSoftTets.clear();

    for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
         ++softBodyIndex)
    {
        SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
        normalizeSoftBodyState(softBody);

        softBody.particleOffset = static_cast<std::uint32_t>(mSoftParticles.size());
        softBody.edgeOffset     = static_cast<std::uint32_t>(mSoftEdges.size());
        softBody.tetOffset      = static_cast<std::uint32_t>(mSoftTets.size());

        const UInt3 resolution            = softBody.gridResolution;
        const std::uint32_t particleCount = resolution.x * resolution.y * resolution.z;
        softBody.particleCount            = particleCount;

        const Diligent::float3 spacing{
            resolution.x > 1u ? softBody.size.x / static_cast<float>(resolution.x - 1u) : 0.0f,
            resolution.y > 1u ? softBody.size.y / static_cast<float>(resolution.y - 1u) : 0.0f,
            resolution.z > 1u ? softBody.size.z / static_cast<float>(resolution.z - 1u) : 0.0f};
        const float inverseMass =
            softBody.particleMass > 0.0f ? 1.0f / softBody.particleMass : 0.0f;

        for (std::uint32_t x = 0u; x < resolution.x; ++x)
        {
            for (std::uint32_t y = 0u; y < resolution.y; ++y)
            {
                for (std::uint32_t z = 0u; z < resolution.z; ++z)
                {
                    const Diligent::float3 position =
                        Diligent::float3{softBody.origin.x + spacing.x * static_cast<float>(x),
                                         softBody.origin.y + spacing.y * static_cast<float>(y),
                                         softBody.origin.z + spacing.z * static_cast<float>(z)};
                    mSoftParticles.positionsInvMass.push_back(
                        Diligent::float4{position.x, position.y, position.z, inverseMass});
                    mSoftParticles.previousPositions.push_back(
                        Diligent::float4{position.x, position.y, position.z, 0.0f});
                    mSoftParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
                    mSoftParticles.radii.push_back(softBody.particleRadius);
                    mSoftParticles.environmentIndices.push_back(softBody.environmentIndex);
                    mSoftParticles.owningSoftBodyIndices.push_back(softBodyIndex);
                    mSoftParticles.collisionLayers.push_back(1u);
                    mSoftParticles.collisionMasks.push_back(0xffffffffu);
                }
            }
        }

        std::set<std::uint64_t> uniqueEdges;
        auto addTet = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2, std::uint32_t i3)
        {
            SoftTet tet{};
            tet.particleIndices = {softBody.particleOffset + i0, softBody.particleOffset + i1,
                                   softBody.particleOffset + i2, softBody.particleOffset + i3};
            tet.compliance      = softBody.volumeCompliance;
            const Diligent::float3 p0{mSoftParticles.positionsInvMass[tet.particleIndices[0]].x,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[0]].y,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[0]].z};
            const Diligent::float3 p1{mSoftParticles.positionsInvMass[tet.particleIndices[1]].x,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[1]].y,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[1]].z};
            const Diligent::float3 p2{mSoftParticles.positionsInvMass[tet.particleIndices[2]].x,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[2]].y,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[2]].z};
            const Diligent::float3 p3{mSoftParticles.positionsInvMass[tet.particleIndices[3]].x,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[3]].y,
                                      mSoftParticles.positionsInvMass[tet.particleIndices[3]].z};
            tet.restVolume = std::abs(tetSignedVolume(p0, p1, p2, p3));
            mSoftTets.push_back(tet);

            const std::array<std::pair<std::uint32_t, std::uint32_t>, 6> edges = {{
                {tet.particleIndices[0], tet.particleIndices[1]},
                {tet.particleIndices[0], tet.particleIndices[2]},
                {tet.particleIndices[0], tet.particleIndices[3]},
                {tet.particleIndices[1], tet.particleIndices[2]},
                {tet.particleIndices[1], tet.particleIndices[3]},
                {tet.particleIndices[2], tet.particleIndices[3]},
            }};
            for (const auto &edgeIndices : edges)
            {
                if (uniqueEdges.insert(makeSortedEdgeKey(edgeIndices.first, edgeIndices.second))
                        .second)
                {
                    SoftEdge edge{};
                    edge.particleA = edgeIndices.first;
                    edge.particleB = edgeIndices.second;
                    const Diligent::float3 delta{
                        mSoftParticles.positionsInvMass[edge.particleB].x -
                            mSoftParticles.positionsInvMass[edge.particleA].x,
                        mSoftParticles.positionsInvMass[edge.particleB].y -
                            mSoftParticles.positionsInvMass[edge.particleA].y,
                        mSoftParticles.positionsInvMass[edge.particleB].z -
                            mSoftParticles.positionsInvMass[edge.particleA].z};
                    edge.restLength = std::sqrt(Diligent::dot(delta, delta));
                    edge.compliance = softBody.edgeCompliance;
                    mSoftEdges.push_back(edge);
                }
            }
        };

        for (std::uint32_t x = 0u; x + 1u < resolution.x; ++x)
        {
            for (std::uint32_t y = 0u; y + 1u < resolution.y; ++y)
            {
                for (std::uint32_t z = 0u; z + 1u < resolution.z; ++z)
                {
                    const std::uint32_t p0 = flattenGridIndex(x, y, z, resolution);
                    const std::uint32_t p1 = flattenGridIndex(x, y, z + 1u, resolution);
                    const std::uint32_t p3 = flattenGridIndex(x + 1u, y, z, resolution);
                    const std::uint32_t p2 = flattenGridIndex(x + 1u, y, z + 1u, resolution);
                    const std::uint32_t p7 = flattenGridIndex(x + 1u, y + 1u, z, resolution);
                    const std::uint32_t p6 = flattenGridIndex(x + 1u, y + 1u, z + 1u, resolution);
                    const std::uint32_t p4 = flattenGridIndex(x, y + 1u, z, resolution);
                    const std::uint32_t p5 = flattenGridIndex(x, y + 1u, z + 1u, resolution);

                    if (((x + y + z) & 1u) != 0u)
                    {
                        addTet(p2, p1, p6, p3);
                        addTet(p6, p3, p4, p7);
                        addTet(p4, p1, p6, p5);
                        addTet(p3, p1, p4, p0);
                        addTet(p6, p1, p4, p3);
                    }
                    else
                    {
                        addTet(p0, p2, p5, p1);
                        addTet(p7, p2, p0, p3);
                        addTet(p5, p2, p7, p6);
                        addTet(p7, p0, p5, p4);
                        addTet(p0, p2, p7, p5);
                    }
                }
            }
        }

        softBody.edgeCount = static_cast<std::uint32_t>(mSoftEdges.size()) - softBody.edgeOffset;
        softBody.tetCount  = static_cast<std::uint32_t>(mSoftTets.size()) - softBody.tetOffset;
    }

    mSoftBodyDerivedStateDirty = false;
}

void PhysicsWorld::rebuildRigidSurfaceParticles() const noexcept
{
    mRigidSurfaceParticles.clear();
    const float referenceSpacing = referenceParticleSpacing();

    for (std::uint32_t colliderIndex = 0u; colliderIndex < colliderCount(); ++colliderIndex)
    {
        const std::uint32_t bodyIndex = mColliders.ownerRigidBodyIndices[colliderIndex];
        if (bodyIndex == 0xffffffffu || bodyIndex >= rigidBodyCount())
        {
            continue;
        }

        const ColliderState &collider = mColliderSnapshot[colliderIndex];
        if (!collider.enabled)
        {
            continue;
        }

        const RigidBodyState &body = mRigidBodySnapshot[bodyIndex];
        const Diligent::float3 colliderPosition =
            body.position + rotatePoint(body.rotation, collider.localPosition);
        const Diligent::QuaternionF colliderRotation = body.rotation * collider.localRotation;

        auto emitSample = [&](const Diligent::float3 &localPoint, float sampleRadius)
        {
            const Diligent::float3 worldPoint =
                transformPoint(colliderPosition, colliderRotation, localPoint);
            mRigidSurfaceParticles.localPositions.push_back(
                Diligent::float4{localPoint.x, localPoint.y, localPoint.z, 0.0f});
            mRigidSurfaceParticles.worldPositions.push_back(
                Diligent::float4{worldPoint.x, worldPoint.y, worldPoint.z, 0.0f});
            mRigidSurfaceParticles.owningRigidBodyIndices.push_back(bodyIndex);
            mRigidSurfaceParticles.owningRigidBodyIds.push_back(body.rigidBodyId);
            mRigidSurfaceParticles.owningColliderIndices.push_back(colliderIndex);
            mRigidSurfaceParticles.owningColliderIds.push_back(collider.colliderId);
            mRigidSurfaceParticles.sampleRadii.push_back(sampleRadius);
            mRigidSurfaceParticles.environmentIndices.push_back(collider.environmentIndex);
            mRigidSurfaceParticles.collisionLayers.push_back(collider.collisionLayer);
            mRigidSurfaceParticles.collisionMasks.push_back(collider.collisionMask);
        };

        switch (collider.shapeType)
        {
        case ColliderShapeType::Sphere:
        {
            const float radius        = std::max(collider.shapeParams.x * body.scale.x, 1.0e-4f);
            const float sampleSpacing = std::max(referenceSpacing, 1.0e-3f);
            const float sampleRadius  = std::max(sampleSpacing * 0.5f, 0.05f);
            const std::uint32_t latitudeCount = std::max<std::uint32_t>(
                3u, static_cast<std::uint32_t>(std::ceil(Diligent::PI_F * radius / sampleSpacing)));
            const std::uint32_t longitudeCount =
                std::max<std::uint32_t>(6u, static_cast<std::uint32_t>(std::ceil(
                                                2.0f * Diligent::PI_F * radius / sampleSpacing)));

            emitSample({0.0f, radius, 0.0f}, sampleRadius);
            emitSample({0.0f, -radius, 0.0f}, sampleRadius);

            for (std::uint32_t latIndex = 1u; latIndex < latitudeCount; ++latIndex)
            {
                const float v   = static_cast<float>(latIndex) / static_cast<float>(latitudeCount);
                const float phi = Diligent::PI_F * v;
                const float y   = std::cos(phi) * radius;
                const float ringRadius = std::sin(phi) * radius;

                for (std::uint32_t lonIndex = 0u; lonIndex < longitudeCount; ++lonIndex)
                {
                    const float u =
                        static_cast<float>(lonIndex) / static_cast<float>(longitudeCount);
                    const float theta = 2.0f * Diligent::PI_F * u;
                    emitSample({std::cos(theta) * ringRadius, y, std::sin(theta) * ringRadius},
                               sampleRadius);
                }
            }
            break;
        }
        case ColliderShapeType::Box:
        {
            const Diligent::float3 halfExtents{collider.shapeParams.x * body.scale.x,
                                               collider.shapeParams.y * body.scale.y,
                                               collider.shapeParams.z * body.scale.z};
            const float sampleSpacing = std::max(referenceSpacing, 1.0e-3f);
            const float sampleRadius  = std::max(sampleSpacing * 0.5f, 0.05f);

            auto emitFaceGrid = [&](float fixedCoord, std::uint32_t axis, float uMin, float uMax,
                                    float vMin, float vMax)
            {
                const float uExtent        = uMax - uMin;
                const float vExtent        = vMax - vMin;
                const std::uint32_t uCount = std::max<std::uint32_t>(
                    2u, static_cast<std::uint32_t>(std::ceil(uExtent / sampleSpacing)) + 1u);
                const std::uint32_t vCount = std::max<std::uint32_t>(
                    2u, static_cast<std::uint32_t>(std::ceil(vExtent / sampleSpacing)) + 1u);

                for (std::uint32_t uIndex = 0u; uIndex < uCount; ++uIndex)
                {
                    const float uT =
                        (uCount > 1u) ? static_cast<float>(uIndex) / static_cast<float>(uCount - 1u)
                                      : 0.0f;
                    const float u = uMin + (uMax - uMin) * uT;

                    for (std::uint32_t vIndex = 0u; vIndex < vCount; ++vIndex)
                    {
                        const float vT = (vCount > 1u) ? static_cast<float>(vIndex) /
                                                             static_cast<float>(vCount - 1u)
                                                       : 0.0f;
                        const float v  = vMin + (vMax - vMin) * vT;

                        Diligent::float3 localPoint{};
                        if (axis == 0u)
                        {
                            localPoint = {fixedCoord, u, v};
                        }
                        else if (axis == 1u)
                        {
                            localPoint = {u, fixedCoord, v};
                        }
                        else
                        {
                            localPoint = {u, v, fixedCoord};
                        }
                        emitSample(localPoint, sampleRadius);
                    }
                }
            };

            emitFaceGrid(-halfExtents.x, 0u, -halfExtents.y, halfExtents.y, -halfExtents.z,
                         halfExtents.z);
            emitFaceGrid(halfExtents.x, 0u, -halfExtents.y, halfExtents.y, -halfExtents.z,
                         halfExtents.z);
            emitFaceGrid(-halfExtents.y, 1u, -halfExtents.x, halfExtents.x, -halfExtents.z,
                         halfExtents.z);
            emitFaceGrid(halfExtents.y, 1u, -halfExtents.x, halfExtents.x, -halfExtents.z,
                         halfExtents.z);
            emitFaceGrid(-halfExtents.z, 2u, -halfExtents.x, halfExtents.x, -halfExtents.y,
                         halfExtents.y);
            emitFaceGrid(halfExtents.z, 2u, -halfExtents.x, halfExtents.x, -halfExtents.y,
                         halfExtents.y);
            break;
        }
        case ColliderShapeType::Capsule:
        {
            const float radius =
                std::max(collider.shapeParams.x * std::max(body.scale.x, body.scale.z), 1.0e-4f);
            const float halfHeight        = std::max(collider.shapeParams.y * body.scale.y, 0.0f);
            const float sampleSpacing     = std::max(referenceSpacing, 1.0e-3f);
            const float sampleRadius      = std::max(sampleSpacing * 0.5f, 0.05f);
            const std::uint32_t ringCount = std::max<std::uint32_t>(
                2u,
                static_cast<std::uint32_t>(std::ceil((halfHeight * 2.0f) / sampleSpacing)) + 1u);
            const std::uint32_t angularCount =
                std::max<std::uint32_t>(6u, static_cast<std::uint32_t>(std::ceil(
                                                2.0f * Diligent::PI_F * radius / sampleSpacing)));

            for (std::uint32_t ringIndex = 0u; ringIndex < ringCount; ++ringIndex)
            {
                const float t = (ringCount > 1u) ? static_cast<float>(ringIndex) /
                                                       static_cast<float>(ringCount - 1u)
                                                 : 0.5f;
                const float y = -halfHeight + (halfHeight * 2.0f) * t;
                for (std::uint32_t angleIndex = 0u; angleIndex < angularCount; ++angleIndex)
                {
                    const float u =
                        static_cast<float>(angleIndex) / static_cast<float>(angularCount);
                    const float theta = 2.0f * Diligent::PI_F * u;
                    emitSample({std::cos(theta) * radius, y, std::sin(theta) * radius},
                               sampleRadius);
                }
            }

            emitSample({0.0f, halfHeight + radius, 0.0f}, sampleRadius);
            emitSample({0.0f, -halfHeight - radius, 0.0f}, sampleRadius);
            break;
        }
        }
    }
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

} // namespace cressim::neo::physics
