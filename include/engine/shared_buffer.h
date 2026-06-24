#ifndef CRESSIM_NEO_ENGINE_SHARED_BUFFER_H
#define CRESSIM_NEO_ENGINE_SHARED_BUFFER_H

#include "engine/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cressim::neo::engine
{

enum class SharedBufferAccess
{
    ReadOnly,
    WriteOnly,
    ReadWrite,
};

enum class SharedBufferBindFlags : std::uint32_t
{
    None            = 0u,
    ShaderResource  = 1u << 0u,
    UnorderedAccess = 1u << 1u,
};

inline constexpr SharedBufferBindFlags operator|(const SharedBufferBindFlags lhs,
                                                 const SharedBufferBindFlags rhs) noexcept
{
    return static_cast<SharedBufferBindFlags>(static_cast<std::uint32_t>(lhs) |
                                              static_cast<std::uint32_t>(rhs));
}

inline constexpr SharedBufferBindFlags operator&(const SharedBufferBindFlags lhs,
                                                 const SharedBufferBindFlags rhs) noexcept
{
    return static_cast<SharedBufferBindFlags>(static_cast<std::uint32_t>(lhs) &
                                              static_cast<std::uint32_t>(rhs));
}

struct SharedBufferHandle
{
    std::uint64_t id = 0u;

    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

struct SharedBufferDesc
{
    std::string debugName;
    std::uint32_t elementStrideBytes = 0u;
    std::uint32_t elementCount       = 0u;
    std::uint32_t minimumCapacity    = 0u;
    SharedBufferAccess access        = SharedBufferAccess::ReadWrite;
    SharedBufferBindFlags bindFlags =
        SharedBufferBindFlags::ShaderResource | SharedBufferBindFlags::UnorderedAccess;
};

struct SharedBufferInfo
{
    SharedBufferHandle handle{};
    std::string debugName;
    std::uint32_t elementStrideBytes = 0u;
    std::uint32_t elementCount       = 0u;
    std::uint32_t capacity           = 0u;
    std::uint64_t sizeBytes          = 0u;
    SharedBufferAccess access        = SharedBufferAccess::ReadWrite;
    SharedBufferBindFlags bindFlags  = SharedBufferBindFlags::None;
    bool exportable                  = false;
    bool importedIntoCuda            = false;
};

struct SharedBufferCudaView
{
    void *devicePointer        = nullptr;
    std::uint64_t sizeBytes    = 0u;
    std::int32_t deviceOrdinal = -1;

    bool isValid() const noexcept
    {
        return devicePointer != nullptr && sizeBytes > 0u && deviceOrdinal >= 0;
    }
};

enum class SharedBufferTensorDTypeCode
{
    Int,
    UInt,
    Float,
    Bool,
};

struct SharedBufferTensorDesc
{
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    SharedBufferTensorDTypeCode dtypeCode = SharedBufferTensorDTypeCode::Float;
    std::uint8_t dtypeBits                = 32u;
    std::uint16_t dtypeLanes              = 1u;
    std::uint64_t byteOffset              = 0u;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_SHARED_BUFFER_H
