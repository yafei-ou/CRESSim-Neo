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
    std::uint32_t colliderCount     = 0;
    std::uint64_t bindingGeneration = 0;
};

struct PhysicsGpuParticleBufferView
{
    Diligent::IBuffer *positionsInvMassBuffer      = nullptr;
    Diligent::IBuffer *previousPositionsBuffer     = nullptr;
    Diligent::IBuffer *velocitiesBuffer            = nullptr;
    Diligent::IBuffer *materialsBuffer             = nullptr;
    Diligent::IBuffer *radiiBuffer                 = nullptr;
    Diligent::IBuffer *environmentIndicesBuffer    = nullptr;
    Diligent::IBuffer *particleKindsBuffer         = nullptr;
    Diligent::IBuffer *ownerTypesBuffer            = nullptr;
    Diligent::IBuffer *ownerIndicesBuffer          = nullptr;
    Diligent::IBuffer *owningSoftBodyIndicesBuffer = nullptr;
    Diligent::IBuffer *fluidRestDensitiesBuffer    = nullptr;
    Diligent::IBuffer *fluidViscositiesBuffer      = nullptr;
    Diligent::IBuffer *fluidSmoothingRadiiBuffer   = nullptr;
    Diligent::IBuffer *phasesBuffer                = nullptr;
    Diligent::IBuffer *collisionLayersBuffer       = nullptr;
    Diligent::IBuffer *collisionMasksBuffer        = nullptr;
    Diligent::IBuffer *adjacencyOffsetsBuffer      = nullptr;
    Diligent::IBuffer *adjacencyCountsBuffer       = nullptr;
    Diligent::IBuffer *adjacencyIndicesBuffer      = nullptr;
    std::uint32_t count                            = 0;
};

struct PhysicsGpuSoftSceneView
{
    PhysicsGpuParticleBufferView particles{};
    Diligent::IBuffer *edgesBuffer         = nullptr;
    Diligent::IBuffer *tetsBuffer          = nullptr;
    Diligent::IBuffer *renderNormalsBuffer = nullptr;
    Diligent::IBuffer *worldAabbsBuffer    = nullptr;
    std::uint32_t softBodyCount            = 0;
    std::uint32_t edgeCount                = 0;
    std::uint32_t tetCount                 = 0;
    std::uint64_t bindingGeneration        = 0;
};

struct CRESSIM_NEO_PHYSICS_API PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};
    PhysicsGpuSoftSceneView soft{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
