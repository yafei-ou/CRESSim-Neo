#include "physics/physics_world.h"

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

} // namespace

void PhysicsWorld::clear()
{
    mRigidBodies.clear();
    mEntityToIndex.clear();
    mRigidBodySnapshot.clear();
    markAllRigidBodiesDirty();
    ++mRevision;
}

RigidBodyState& PhysicsWorld::upsertRigidBody(const RigidBodyState& state)
{
    auto it = mEntityToIndex.find(state.entityId);
    if (it == mEntityToIndex.end())
    {
        const std::uint32_t index = static_cast<std::uint32_t>(mRigidBodies.size());
        mEntityToIndex.emplace(state.entityId, index);
        mRigidBodies.entityIds.push_back(state.entityId);
        mRigidBodies.positionsInvMass.push_back(toPositionInvMass(state));
        mRigidBodies.orientations.push_back(toOrientation(state));
        mRigidBodies.scales.push_back(toScale(state));
        mRigidBodies.linearVelocities.push_back(toLinearVelocity(state));
        mRigidBodies.angularVelocities.push_back(toAngularVelocity(state));
        mRigidBodies.inverseInertiaLocal.push_back(toInverseInertiaLocal(state));
        mRigidBodies.colliderShapeTypes.push_back(static_cast<std::uint32_t>(state.colliderShape));
        mRigidBodies.colliderParams.push_back(state.colliderParams);
        mRigidBodySnapshot.push_back(state);
        mRigidBodyDirtyRange.include(index);
        ++mRevision;
        return mRigidBodySnapshot.back();
    }

    const std::uint32_t index = it->second;
    writeRigidBodySoAAt(mRigidBodies, index, state);
    mRigidBodySnapshot[index] = state;
    mRigidBodyDirtyRange.include(index);
    ++mRevision;
    return mRigidBodySnapshot[index];
}

bool PhysicsWorld::removeRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToIndex.find(entityId);
    if (it == mEntityToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mRigidBodies.size() - 1u);

    if (index != last)
    {
        mRigidBodies.entityIds[index]                 = mRigidBodies.entityIds[last];
        mRigidBodies.positionsInvMass[index]          = mRigidBodies.positionsInvMass[last];
        mRigidBodies.orientations[index]              = mRigidBodies.orientations[last];
        mRigidBodies.scales[index]                    = mRigidBodies.scales[last];
        mRigidBodies.linearVelocities[index]          = mRigidBodies.linearVelocities[last];
        mRigidBodies.angularVelocities[index]         = mRigidBodies.angularVelocities[last];
        mRigidBodies.inverseInertiaLocal[index]       = mRigidBodies.inverseInertiaLocal[last];
        mRigidBodies.colliderShapeTypes[index]        = mRigidBodies.colliderShapeTypes[last];
        mRigidBodies.colliderParams[index]            = mRigidBodies.colliderParams[last];
        mRigidBodySnapshot[index]                     = mRigidBodySnapshot[last];
        mEntityToIndex[mRigidBodies.entityIds[index]] = index;
    }

    mEntityToIndex.erase(it);
    mRigidBodies.entityIds.pop_back();
    mRigidBodies.positionsInvMass.pop_back();
    mRigidBodies.orientations.pop_back();
    mRigidBodies.scales.pop_back();
    mRigidBodies.linearVelocities.pop_back();
    mRigidBodies.angularVelocities.pop_back();
    mRigidBodies.inverseInertiaLocal.pop_back();
    mRigidBodies.colliderShapeTypes.pop_back();
    mRigidBodies.colliderParams.pop_back();
    mRigidBodySnapshot.pop_back();

    markAllRigidBodiesDirty();
    ++mRevision;
    return true;
}

RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToIndex.find(entityId);
    return it == mEntityToIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mEntityToIndex.find(entityId);
    return it == mEntityToIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const std::vector<RigidBodyState>& PhysicsWorld::rigidBodySnapshot() const noexcept
{
    return mRigidBodySnapshot;
}

const RigidBodySoAHost& PhysicsWorld::rigidBodySoA() const noexcept
{
    return mRigidBodies;
}

const PhysicsSoADirtyRange& PhysicsWorld::rigidBodyDirtyRange() const noexcept
{
    return mRigidBodyDirtyRange;
}

std::uint32_t PhysicsWorld::rigidBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mRigidBodies.size());
}

void PhysicsWorld::clearRigidBodyDirtyRange() noexcept
{
    mRigidBodyDirtyRange.clear();
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
    soa.entityIds[index]          = state.entityId;
    soa.positionsInvMass[index]   = toPositionInvMass(state);
    soa.orientations[index]       = toOrientation(state);
    soa.scales[index]             = toScale(state);
    soa.linearVelocities[index]   = toLinearVelocity(state);
    soa.angularVelocities[index]  = toAngularVelocity(state);
    soa.inverseInertiaLocal[index] = toInverseInertiaLocal(state);
    soa.colliderShapeTypes[index] = static_cast<std::uint32_t>(state.colliderShape);
    soa.colliderParams[index]     = state.colliderParams;
}

RigidBodyState PhysicsWorld::readRigidBodySoAAt(const RigidBodySoAHost& soa, std::uint32_t index)
{
    RigidBodyState state{};
    state.entityId = soa.entityIds[index];

    const Diligent::float4 positionInvMass = soa.positionsInvMass[index];
    state.position    = Diligent::float3{positionInvMass.x, positionInvMass.y, positionInvMass.z};
    state.inverseMass = positionInvMass.w;

    const Diligent::float4 orientation = soa.orientations[index];
    state.rotation =
        Diligent::QuaternionF{orientation.x, orientation.y, orientation.z, orientation.w};

    const Diligent::float4 scale = soa.scales[index];
    state.scale = Diligent::float3{scale.x, scale.y, scale.z};

    const Diligent::float4 linearVelocity = soa.linearVelocities[index];
    state.linearVelocity = Diligent::float3{linearVelocity.x, linearVelocity.y, linearVelocity.z};

    const Diligent::float4 angularVelocity = soa.angularVelocities[index];
    state.angularVelocity =
        Diligent::float3{angularVelocity.x, angularVelocity.y, angularVelocity.z};

    const Diligent::float4 inverseInertiaLocal = soa.inverseInertiaLocal[index];
    state.inverseInertiaLocal =
        Diligent::float3{inverseInertiaLocal.x, inverseInertiaLocal.y, inverseInertiaLocal.z};

    state.colliderShape  = static_cast<ColliderShapeType>(soa.colliderShapeTypes[index]);
    state.colliderParams = soa.colliderParams[index];
    return state;
}

void PhysicsWorld::markAllRigidBodiesDirty() noexcept
{
    mRigidBodyDirtyRange.valid = true;
    mRigidBodyDirtyRange.begin = 0;
    mRigidBodyDirtyRange.end   = rigidBodyCount();
}

} // namespace cressim::neo::physics
