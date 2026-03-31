#ifndef CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H
#define CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <cstdint>

namespace cressim::neo::common
{

struct SceneLayoutDesc
{
    std::uint32_t envCount                   = 1u;
    std::uint32_t maxRenderableObjectsPerEnv = 4096u;
    std::uint32_t maxLightsPerEnv            = 4u;
    std::uint32_t maxCamerasPerEnv           = 4u;

    std::uint32_t totalRenderableObjectCapacity() const noexcept
    {
        return envCount * maxRenderableObjectsPerEnv;
    }
    std::uint32_t totalLightCapacity() const noexcept
    {
        return envCount * maxLightsPerEnv;
    }
    std::uint32_t totalCameraCapacity() const noexcept
    {
        return envCount * maxCamerasPerEnv;
    }
};

struct PoseBufferView
{
    Diligent::IBuffer *positionsBuffer    = nullptr;
    Diligent::IBuffer *orientationsBuffer = nullptr;
    Diligent::IBuffer *scalesBuffer       = nullptr;
    std::uint32_t count                   = 0;
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_SCENE_PRIMITIVES_H
