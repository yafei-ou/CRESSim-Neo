#include "physics/physics_world.h"

#include <algorithm>

namespace cressim::neo::physics
{

namespace
{

Diligent::float4 toPositionInvMass(const RigidBodyState& state)
{
    return Diligent::float4{state.position.x, state.position.y, state.position.z,
                            state.inverseMass};
}

Diligent::float4 toOrientation(const RigidBodyState& state)
{
    return Diligent::float4{state.rotation.q.x, state.rotation.q.y, state.rotation.q.z,
                            state.rotation.q.w};
}

Diligent::float4 toLinearVelocity(const RigidBodyState& state)
{
    return Diligent::float4{state.linearVelocity.x, state.linearVelocity.y, state.linearVelocity.z,
                            0.0f};
}

Diligent::float4 toScale(const RigidBodyState& state)
{
    return Diligent::float4{state.scale.x, state.scale.y, state.scale.z, 0.0f};
}

Diligent::float4 toAngularVelocity(const RigidBodyState& state)
{
    return Diligent::float4{state.angularVelocity.x, state.angularVelocity.y,
                            state.angularVelocity.z, 0.0f};
}

Diligent::float4 toInverseInertiaLocal(const RigidBodyState& state)
{
    return Diligent::float4{state.inverseInertiaLocal.x, state.inverseInertiaLocal.y,
                            state.inverseInertiaLocal.z, 0.0f};
}

Diligent::float4 toKinematicTargetPosition(const RigidBodyState& state)
{
    return Diligent::float4{state.kinematicTargetPosition.x, state.kinematicTargetPosition.y,
                            state.kinematicTargetPosition.z, 0.0f};
}

Diligent::float4 toKinematicTargetOrientation(const RigidBodyState& state)
{
    return Diligent::float4{state.kinematicTargetRotation.q.x, state.kinematicTargetRotation.q.y,
                            state.kinematicTargetRotation.q.z, state.kinematicTargetRotation.q.w};
}

Diligent::float4 toColliderLocalPosition(const ColliderState& state)
{
    return Diligent::float4{state.localPosition.x, state.localPosition.y, state.localPosition.z,
                            0.0f};
}

Diligent::float4 toColliderLocalOrientation(const ColliderState& state)
{
    return Diligent::float4{state.localRotation.q.x, state.localRotation.q.y,
                            state.localRotation.q.z, state.localRotation.q.w};
}

Diligent::float4 toColliderMaterial(const ColliderState& state)
{
    return Diligent::float4{state.friction, state.restitution, 0.0f, 0.0f};
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
    mRigidBodySnapshot.clear();
    mColliderSnapshot.clear();
    mRigidBodyDirtyRange.clear();
    mColliderDirtyRange.clear();
    mStaticBroadPhaseDirty = true;
    ++mRevision;
}

RigidBodyState& PhysicsWorld::upsertRigidBody(const RigidBodyState& state)
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
        mBodyColliderMapping.colliderOffsets.push_back(
            static_cast<std::uint32_t>(mBodyColliderMapping.colliderIndices.size()));
        mBodyColliderMapping.colliderCounts.push_back(0u);
        mRigidBodyDirtyRange.include(index);
        mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || isStaticBody(normalizedState);
        ++mRevision;
        return mRigidBodySnapshot.back();
    }

    const std::uint32_t index          = it->second;
    normalizedState.rigidBodyId        = mRigidBodySnapshot[index].rigidBodyId;
    const RigidBodyState previousState = mRigidBodySnapshot[index];
    writeRigidBodySoAAt(mRigidBodies, index, normalizedState);
    mRigidBodySnapshot[index] = normalizedState;
    mRigidBodyDirtyRange.include(index);
    mStaticBroadPhaseDirty =
        mStaticBroadPhaseDirty || staticBodyPoseChanged(previousState, normalizedState);
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

    rebuildBodyColliderMapping();
    markAllRigidBodiesDirty();
    mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || removedStatic;
    ++mRevision;
    return true;
}

void PhysicsWorld::upsertCollider(const ColliderState& state)
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
        auto& entityColliderIds = mEntityToColliderIds[normalizedState.entityId];
        entityColliderIds.push_back(normalizedState.colliderId);
        rebuildBodyColliderMapping();
        markAllCollidersDirty();
        mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || ownerIsStatic;
        ++mRevision;
        return;
    }

    const std::uint32_t colliderIndex = colliderIt->second;
    const ColliderState previousState = mColliderSnapshot[colliderIndex];
    writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex);
    mColliderSnapshot[colliderIndex] = normalizedState;
    mColliderDirtyRange.include(colliderIndex);
    rebuildBodyColliderMapping();
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
    rebuildBodyColliderMapping();
    markAllCollidersDirty();
    mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || removedStaticOwner;
    ++mRevision;
    return true;
}

void PhysicsWorld::replaceColliders(common::EntityId entityId,
                                    const std::vector<ColliderState>& colliders)
{
    removeCollidersForEntity(entityId);

    const auto bodyIt = mEntityToRigidBodyIndex.find(entityId);
    if (bodyIt == mEntityToRigidBodyIndex.end() || colliders.empty())
    {
        rebuildBodyColliderMapping();
        ++mRevision;
        return;
    }

    const std::uint32_t ownerBodyIndex = bodyIt->second;
    const RigidBodyId ownerRigidBodyId = mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;

    auto& entityColliderIds = mEntityToColliderIds[entityId];
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

    rebuildBodyColliderMapping();
    markAllCollidersDirty();
    mStaticBroadPhaseDirty = true;
    ++mRevision;
}

RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const ColliderState* PhysicsWorld::tryGetCollider(ColliderId colliderId) const
{
    const auto it = mColliderIdToIndex.find(colliderId);
    return it == mColliderIdToIndex.end() ? nullptr : &mColliderSnapshot[it->second];
}

const std::vector<RigidBodyState>& PhysicsWorld::rigidBodySnapshot() const noexcept
{
    return mRigidBodySnapshot;
}

const std::vector<ColliderState>& PhysicsWorld::colliderSnapshot() const noexcept
{
    return mColliderSnapshot;
}

const RigidBodySoAHost& PhysicsWorld::rigidBodySoA() const noexcept
{
    return mRigidBodies;
}

const ColliderSoAHost& PhysicsWorld::colliderSoA() const noexcept
{
    return mColliders;
}

const BodyColliderMappingHost& PhysicsWorld::bodyColliderMapping() const noexcept
{
    return mBodyColliderMapping;
}

const PhysicsSoADirtyRange& PhysicsWorld::rigidBodyDirtyRange() const noexcept
{
    return mRigidBodyDirtyRange;
}

const PhysicsSoADirtyRange& PhysicsWorld::colliderDirtyRange() const noexcept
{
    return mColliderDirtyRange;
}

std::uint32_t PhysicsWorld::rigidBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mRigidBodies.size());
}

std::uint32_t PhysicsWorld::colliderCount() const noexcept
{
    return static_cast<std::uint32_t>(mColliders.size());
}

void PhysicsWorld::clearRigidBodyDirtyRange() noexcept
{
    mRigidBodyDirtyRange.clear();
}

void PhysicsWorld::clearColliderDirtyRange() noexcept
{
    mColliderDirtyRange.clear();
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
        Diligent::float4& positionInvMass     = mRigidBodies.positionsInvMass[i];
        const Diligent::float4 linearVelocity = mRigidBodies.linearVelocities[i];
        positionInvMass.x += linearVelocity.x * dt;
        positionInvMass.y += linearVelocity.y * dt;
        positionInvMass.z += linearVelocity.z * dt;

        RigidBodyState& state = mRigidBodySnapshot[i];
        state.position.x      = positionInvMass.x;
        state.position.y      = positionInvMass.y;
        state.position.z      = positionInvMass.z;
    }

    markAllRigidBodiesDirty();
    ++mRevision;
}

bool PhysicsWorld::writeBackRigidBodyState(std::uint32_t index,
                                           const Diligent::float4& positionInvMass,
                                           const Diligent::float4& orientation,
                                           const Diligent::float4& linearVelocity,
                                           const Diligent::float4& angularVelocity) noexcept
{
    if (index >= rigidBodyCount())
    {
        return false;
    }

    mRigidBodies.positionsInvMass[index]  = positionInvMass;
    mRigidBodies.orientations[index]      = orientation;
    mRigidBodies.linearVelocities[index]  = linearVelocity;
    mRigidBodies.angularVelocities[index] = angularVelocity;
    mRigidBodyDirtyRange.include(index);

    RigidBodyState& state = mRigidBodySnapshot[index];
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

std::uint64_t PhysicsWorld::revision() const noexcept
{
    return mRevision;
}

void PhysicsWorld::writeRigidBodySoAAt(RigidBodySoAHost& soa, std::uint32_t index,
                                       const RigidBodyState& state)
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

void PhysicsWorld::writeColliderSoAAt(ColliderSoAHost& soa, std::uint32_t index,
                                      const ColliderState& state, std::uint32_t ownerBodyIndex)
{
    if (index == soa.size())
    {
        soa.colliderIds.push_back(state.colliderId);
        soa.entityIds.push_back(state.entityId);
        soa.ownerRigidBodyIds.push_back(state.ownerRigidBodyId);
        soa.ownerRigidBodyIndices.push_back(ownerBodyIndex);
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
    soa.shapeTypes[index]            = static_cast<std::uint32_t>(state.shapeType);
    soa.shapeParams[index]           = state.shapeParams;
    soa.localPositions[index]        = toColliderLocalPosition(state);
    soa.localOrientations[index]     = toColliderLocalOrientation(state);
    soa.enabledFlags[index]          = state.enabled ? 1u : 0u;
    soa.frictionRestitution[index]   = toColliderMaterial(state);
    soa.collisionLayers[index]       = state.collisionLayer;
    soa.collisionMasks[index]        = state.collisionMask;
}

bool PhysicsWorld::isStaticBody(const RigidBodyState& state) noexcept
{
    return state.bodyType == RigidBodyType::Static;
}

bool PhysicsWorld::staticBodyPoseChanged(const RigidBodyState& before,
                                         const RigidBodyState& after) noexcept
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

void PhysicsWorld::normalizeRigidBodyState(RigidBodyState& state) noexcept
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

void PhysicsWorld::normalizeColliderState(ColliderState& state) noexcept
{
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
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
        auto& handles = handlesIt->second;
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
        mColliders.shapeTypes[index]            = mColliders.shapeTypes[last];
        mColliders.shapeParams[index]           = mColliders.shapeParams[last];
        mColliders.localPositions[index]        = mColliders.localPositions[last];
        mColliders.localOrientations[index]     = mColliders.localOrientations[last];
        mColliders.enabledFlags[index]          = mColliders.enabledFlags[last];
        mColliders.frictionRestitution[index]   = mColliders.frictionRestitution[last];
        mColliders.collisionLayers[index]       = mColliders.collisionLayers[last];
        mColliders.collisionMasks[index]        = mColliders.collisionMasks[last];
        mColliderIdToIndex[moved.colliderId]    = index;
    }

    mColliderSnapshot.pop_back();
    mColliders.colliderIds.pop_back();
    mColliders.entityIds.pop_back();
    mColliders.ownerRigidBodyIds.pop_back();
    mColliders.ownerRigidBodyIndices.pop_back();
    mColliders.shapeTypes.pop_back();
    mColliders.shapeParams.pop_back();
    mColliders.localPositions.pop_back();
    mColliders.localOrientations.pop_back();
    mColliders.enabledFlags.pop_back();
    mColliders.frictionRestitution.pop_back();
    mColliders.collisionLayers.pop_back();
    mColliders.collisionMasks.pop_back();
}

void PhysicsWorld::rebuildBodyColliderMapping() noexcept
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

void PhysicsWorld::markAllRigidBodiesDirty() noexcept
{
    mRigidBodyDirtyRange.valid = true;
    mRigidBodyDirtyRange.begin = 0;
    mRigidBodyDirtyRange.end   = rigidBodyCount();
}

void PhysicsWorld::markAllCollidersDirty() noexcept
{
    mColliderDirtyRange.valid = true;
    mColliderDirtyRange.begin = 0;
    mColliderDirtyRange.end   = colliderCount();
}

} // namespace cressim::neo::physics
