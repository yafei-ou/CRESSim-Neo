#ifndef CRESSIM_NEO_GPU_GPU_BUFFER_UTILS_H
#define CRESSIM_NEO_GPU_GPU_BUFFER_UTILS_H

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <algorithm>
#include <cstdint>

namespace cressim::neo::gpu::detail
{

inline bool ensureStructuredBufferCapacity(
    Diligent::IRenderDevice* renderDevice, const char* name, std::uint32_t elementStride,
    std::uint32_t requiredElementCount, std::uint32_t minimumCapacity,
    Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage, Diligent::CPU_ACCESS_FLAGS cpuAccess,
    Diligent::Uint64 immediateContextMask, Diligent::RefCntAutoPtr<Diligent::IBuffer>& inOutBuffer,
    std::uint32_t& inOutCapacity)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    std::uint32_t currentCapacity = inOutCapacity;
    if (inOutBuffer != nullptr && elementStride > 0u)
    {
        const Diligent::BufferDesc& desc = inOutBuffer->GetDesc();
        currentCapacity                  = static_cast<std::uint32_t>(desc.Size / elementStride);
        inOutCapacity                    = currentCapacity;
    }

    const std::uint32_t requiredCapacity =
        std::max(requiredElementCount, std::max(minimumCapacity, 1u));
    if (inOutBuffer != nullptr && currentCapacity >= requiredCapacity)
    {
        return true;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = name;
    desc.Size                 = static_cast<Diligent::Uint64>(requiredCapacity) * elementStride;
    desc.BindFlags            = bindFlags;
    desc.Usage                = usage;
    desc.CPUAccessFlags       = cpuAccess;
    desc.ImmediateContextMask = immediateContextMask;
    if (usage != Diligent::USAGE_STAGING)
    {
        desc.Mode              = Diligent::BUFFER_MODE_STRUCTURED;
        desc.ElementByteStride = elementStride;
    }

    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    renderDevice->CreateBuffer(desc, nullptr, &buffer);
    if (buffer == nullptr)
    {
        return false;
    }

    inOutBuffer   = std::move(buffer);
    inOutCapacity = requiredCapacity;
    return true;
}

} // namespace cressim::neo::gpu::detail

#endif // CRESSIM_NEO_GPU_GPU_BUFFER_UTILS_H
