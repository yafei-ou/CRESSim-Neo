#ifndef CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H
#define CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <cstdint>

/// @file scene_primitives.h
/// @brief Multi-environment scene layout configuration and GPU pose buffer binding views.

namespace cressim::neo::common
{

/// @brief Capacity descriptor specifying limits and environment counts for multi-environment
/// scenes.
struct SceneLayoutDesc
{
    std::uint32_t envCount = 1u; ///< Number of parallel simulation and rendering environments.
    std::uint32_t maxRenderableObjectsPerEnv =
        4096u; ///< Maximum number of renderable mesh objects in each environment.
    std::uint32_t maxLightsPerEnv  = 4u; ///< Maximum number of light sources per environment.
    std::uint32_t maxCamerasPerEnv = 4u; ///< Maximum number of cameras per environment.

    /// @brief Computes the total renderable object capacity across all environments.
    /// @return Total renderable object slots (`envCount * maxRenderableObjectsPerEnv`).
    std::uint32_t totalRenderableObjectCapacity() const noexcept
    {
        return envCount * maxRenderableObjectsPerEnv;
    }

    /// @brief Computes the total light capacity across all environments.
    /// @return Total light slots (`envCount * maxLightsPerEnv`).
    std::uint32_t totalLightCapacity() const noexcept
    {
        return envCount * maxLightsPerEnv;
    }

    /// @brief Computes the total camera capacity across all environments.
    /// @return Total camera slots (`envCount * maxCamerasPerEnv`).
    std::uint32_t totalCameraCapacity() const noexcept
    {
        return envCount * maxCamerasPerEnv;
    }
};

/// @brief Non-owning view of GPU buffers holding structured instance transformation data.
struct PoseBufferView
{
    Diligent::IBuffer *positionsBuffer =
        nullptr; ///< GPU buffer containing instance 3D position vectors.
    Diligent::IBuffer *orientationsBuffer =
        nullptr; ///< GPU buffer containing instance orientation quaternions.
    Diligent::IBuffer *scalesBuffer = nullptr; ///< GPU buffer containing instance 3D scale vectors.
    std::uint32_t count             = 0; ///< Number of valid transform instances in the buffers.
    std::uint64_t bindingGeneration =
        0; ///< Revision counter tracking buffer reallocation/rebind events.
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H
