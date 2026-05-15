#ifndef CRESSIM_NEO_GRAPHICS_RENDER_TARGET_CACHE_KEY_H
#define CRESSIM_NEO_GRAPHICS_RENDER_TARGET_CACHE_KEY_H

#include "gpu/gpu_types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace cressim::neo::graphics::detail
{

template <typename T>
inline void hashCombine(std::size_t &seed, const T &value) noexcept
{
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

struct RenderTargetFamilyKey
{
    std::uint32_t width                  = 0u;
    std::uint32_t height                 = 0u;
    bool color                           = true;
    bool depth                           = true;
    bool shaderReadable                  = true;
    bool unorderedAccess                 = false;
    bool layeredRendering                = true;
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
    std::string debugName{};

    bool operator==(const RenderTargetFamilyKey &rhs) const noexcept
    {
        return width == rhs.width && height == rhs.height && color == rhs.color &&
               depth == rhs.depth && shaderReadable == rhs.shaderReadable &&
               unorderedAccess == rhs.unorderedAccess &&
               layeredRendering == rhs.layeredRendering && colorFormat == rhs.colorFormat &&
               depthFormat == rhs.depthFormat && debugName == rhs.debugName;
    }
};

struct RenderTargetFamilyKeyHasher
{
    std::size_t operator()(const RenderTargetFamilyKey &key) const noexcept
    {
        std::size_t seed = 0u;
        hashCombine(seed, key.width);
        hashCombine(seed, key.height);
        hashCombine(seed, key.color);
        hashCombine(seed, key.depth);
        hashCombine(seed, key.shaderReadable);
        hashCombine(seed, key.unorderedAccess);
        hashCombine(seed, key.layeredRendering);
        hashCombine(seed, static_cast<std::uint32_t>(key.colorFormat));
        hashCombine(seed, static_cast<std::uint32_t>(key.depthFormat));
        hashCombine(seed, key.debugName);
        return seed;
    }
};

struct RenderTargetCacheKey
{
    RenderTargetFamilyKey family{};
    std::uint32_t arraySize = 1u;

    bool operator==(const RenderTargetCacheKey &rhs) const noexcept
    {
        return family == rhs.family && arraySize == rhs.arraySize;
    }
};

struct RenderTargetCacheKeyHasher
{
    std::size_t operator()(const RenderTargetCacheKey &key) const noexcept
    {
        std::size_t seed = RenderTargetFamilyKeyHasher{}(key.family);
        hashCombine(seed, key.arraySize);
        return seed;
    }
};

inline RenderTargetFamilyKey makeRenderTargetFamilyKey(const gpu::GpuRenderTargetDesc &desc)
{
    return RenderTargetFamilyKey{desc.width,          desc.height,         desc.color,
                                 desc.depth,          desc.shaderReadable, desc.unorderedAccess,
                                 desc.layeredRendering, desc.colorFormat, desc.depthFormat,
                                 desc.debugName};
}

inline RenderTargetCacheKey makeRenderTargetCacheKey(const gpu::GpuRenderTargetDesc &desc)
{
    return RenderTargetCacheKey{makeRenderTargetFamilyKey(desc), desc.arraySize};
}

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDER_TARGET_CACHE_KEY_H
