#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H

#include "physics/export.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <cstdint>

namespace cressim::neo::physics
{

struct PhysicsGpuRigidSceneView
{
    Diligent::IBuffer* positionsBuffer    = nullptr;
    Diligent::IBuffer* orientationsBuffer = nullptr;
    Diligent::IBuffer* scalesBuffer       = nullptr;
    std::uint32_t rigidBodyCount          = 0;
    std::uint32_t colliderCount           = 0;
};

struct CRESSIM_NEO_PHYSICS_API PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
