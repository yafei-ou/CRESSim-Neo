#include "physics/physics_world.h"

#include <algorithm>

namespace cressim::neo::physics
{

void PhysicsWorld::clear()
{
    mRigidBodies.clear();
    ++mRevision;
}

RigidBodyState& PhysicsWorld::upsertRigidBody(const RigidBodyState& state)
{
    auto it = std::find_if(mRigidBodies.begin(), mRigidBodies.end(),
                           [&](const RigidBodyState& rb) { return rb.entityId == state.entityId; });
    if (it == mRigidBodies.end())
    {
        mRigidBodies.push_back(state);
        ++mRevision;
        return mRigidBodies.back();
    }

    *it = state;
    ++mRevision;
    return *it;
}

bool PhysicsWorld::removeRigidBody(common::EntityId entityId)
{
    const auto before = mRigidBodies.size();
    mRigidBodies.erase(std::remove_if(mRigidBodies.begin(), mRigidBodies.end(),
                                      [&](const RigidBodyState& rb)
                                      { return rb.entityId == entityId; }),
                       mRigidBodies.end());

    if (mRigidBodies.size() == before)
    {
        return false;
    }

    ++mRevision;
    return true;
}

RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    auto it = std::find_if(mRigidBodies.begin(), mRigidBodies.end(),
                           [&](const RigidBodyState& rb) { return rb.entityId == entityId; });
    return it == mRigidBodies.end() ? nullptr : &(*it);
}

const RigidBodyState* PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    auto it = std::find_if(mRigidBodies.begin(), mRigidBodies.end(),
                           [&](const RigidBodyState& rb) { return rb.entityId == entityId; });
    return it == mRigidBodies.end() ? nullptr : &(*it);
}

const std::vector<RigidBodyState>& PhysicsWorld::rigidBodies() const noexcept
{
    return mRigidBodies;
}

std::vector<RigidBodyState>& PhysicsWorld::rigidBodies() noexcept
{
    return mRigidBodies;
}

std::uint64_t PhysicsWorld::revision() const noexcept
{
    return mRevision;
}

} // namespace cressim::neo::physics
