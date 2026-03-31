#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H

#include "physics/export.h"

#include "common/scene_primitives.h"

#include <cstdint>

namespace cressim::neo::physics
{

struct PhysicsGpuRigidSceneView
{
    common::PoseBufferView poses{};
    std::uint32_t colliderCount = 0;
};

struct CRESSIM_NEO_PHYSICS_API PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
