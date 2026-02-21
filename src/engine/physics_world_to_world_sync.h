#ifndef CRESSIM_NEO_ENGINE_PHYSICS_WORLD_TO_WORLD_SYNC_H
#define CRESSIM_NEO_ENGINE_PHYSICS_WORLD_TO_WORLD_SYNC_H

#include "engine/world.h"
#include "physics/physics_world.h"

namespace cressim::neo::engine::detail
{

void syncPhysicsWorldToWorld(const physics::PhysicsWorld& physicsWorld, World& world);

} // namespace cressim::neo::engine::detail

#endif // CRESSIM_NEO_ENGINE_PHYSICS_WORLD_TO_WORLD_SYNC_H
