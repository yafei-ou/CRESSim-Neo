#ifndef CRESSIM_NEO_ENGINE_WORLD_TO_PHYSICS_WORLD_SYNC_H
#define CRESSIM_NEO_ENGINE_WORLD_TO_PHYSICS_WORLD_SYNC_H

#include "engine/world.h"
#include "physics/physics_world.h"

namespace cressim::neo::engine::detail
{

void syncWorldToPhysicsWorld(const World& world, physics::PhysicsWorld& physicsWorld);

} // namespace cressim::neo::engine::detail

#endif // CRESSIM_NEO_ENGINE_WORLD_TO_PHYSICS_WORLD_SYNC_H
