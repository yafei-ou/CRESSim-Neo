#ifndef CRESSIM_NEO_ENGINE_SHARED_BUFFER_H
#define CRESSIM_NEO_ENGINE_SHARED_BUFFER_H

#include "engine/export.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Declared custom-compute access policy for shared engine GPU buffers.
enum class SharedBufferAccess
{
    ReadOnly,  ///< Allows read-only custom-compute bindings.
    WriteOnly, ///< Allows write-only custom-compute bindings.
    ReadWrite, ///< Allows read/write custom-compute bindings.
};

/// @brief Binding usage flags for shared GPU buffers.
enum class SharedBufferBindFlags : std::uint32_t
{
    None            = 0u,       ///< No binding flags specified.
    ShaderResource  = 1u << 0u, ///< Bindable as a Shader Resource View (SRV).
    UnorderedAccess = 1u << 1u, ///< Bindable as an Unordered Access View (UAV).
};

/// @brief Bitwise OR operator for SharedBufferBindFlags.
/// @return The union of @p lhs and @p rhs.
inline constexpr SharedBufferBindFlags operator|(const SharedBufferBindFlags lhs,
                                                 const SharedBufferBindFlags rhs) noexcept
{
    return static_cast<SharedBufferBindFlags>(static_cast<std::uint32_t>(lhs) |
                                              static_cast<std::uint32_t>(rhs));
}

/// @brief Bitwise AND operator for SharedBufferBindFlags.
/// @return The intersection of @p lhs and @p rhs.
inline constexpr SharedBufferBindFlags operator&(const SharedBufferBindFlags lhs,
                                                 const SharedBufferBindFlags rhs) noexcept
{
    return static_cast<SharedBufferBindFlags>(static_cast<std::uint32_t>(lhs) &
                                              static_cast<std::uint32_t>(rhs));
}

/// @brief Handle wrapper identifying an engine-owned shared GPU buffer.
struct SharedBufferHandle
{
    std::uint64_t id = 0u; ///< Unique handle identifier.

    /// @brief Checks if the shared buffer handle is valid.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

/// @brief Descriptor for creating an engine-owned shared GPU buffer.
struct SharedBufferDesc
{
    std::string debugName;                 ///< Debug label for GPU diagnostic tools.
    std::uint32_t elementStrideBytes = 0u; ///< Non-zero stride per element in bytes.
    std::uint32_t elementCount       = 0u; ///< Non-zero initial element count.
    std::uint32_t minimumCapacity    = 0u; ///< Optional lower bound on allocated element capacity.
    SharedBufferAccess access = SharedBufferAccess::ReadWrite; ///< Custom-compute access policy;
                                                               ///< bindFlags control GPU views.
    SharedBufferBindFlags bindFlags = SharedBufferBindFlags::ShaderResource |
                                      SharedBufferBindFlags::UnorderedAccess; ///< Must not be None.
};

/// @brief Detailed information and runtime state for a shared GPU buffer.
struct SharedBufferInfo
{
    SharedBufferHandle handle{};           ///< Buffer handle.
    std::string debugName;                 ///< Debug label.
    std::uint32_t elementStrideBytes = 0u; ///< Element stride in bytes.
    std::uint32_t elementCount       = 0u; ///< Current element count.
    std::uint32_t capacity           = 0u; ///< Total element capacity.
    std::uint64_t sizeBytes          = 0u; ///< Total allocated size in bytes.
    SharedBufferAccess access =
        SharedBufferAccess::ReadWrite; ///< Declared custom-compute access policy.
    SharedBufferBindFlags bindFlags = SharedBufferBindFlags::None; ///< Binding flags.
    bool exportable       = false; ///< True if buffer is exportable for CUDA/DLPack interop.
    bool importedIntoCuda = false; ///< True if buffer is currently imported into CUDA.
};

/// @brief View descriptor exposing CUDA device memory pointer and size for interop.
struct SharedBufferCudaView
{
    void *devicePointer        = nullptr; ///< Raw CUDA device pointer.
    std::uint64_t sizeBytes    = 0u;      ///< Size of accessible device memory in bytes.
    std::int32_t deviceOrdinal = -1;      ///< CUDA device ordinal index.

    /// @brief Checks if the CUDA device view is valid.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return devicePointer != nullptr && sizeBytes > 0u && deviceOrdinal >= 0;
    }
};

/// @brief RAII lease keeping shared GPU buffer memory alive across external DLPack/Torch consumers.
class CRESSIM_NEO_ENGINE_API SharedBufferLease
{
public:
    SharedBufferLease() = default;

    /// @brief Checks if the buffer lease is active and valid.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return static_cast<bool>(mKeepAlive);
    }

private:
    friend class Runtime;

    explicit SharedBufferLease(std::shared_ptr<void> keepAlive) : mKeepAlive(std::move(keepAlive))
    {
    }

    std::shared_ptr<void> mKeepAlive;
};

/// @brief DLPack tensor data type codes for shared buffer tensor view export.
enum class SharedBufferTensorDTypeCode
{
    Int,   ///< Signed integer.
    UInt,  ///< Unsigned integer.
    Float, ///< Floating point.
    Bool,  ///< Boolean.
};

/// @brief Tensor metadata descriptor for exporting a shared buffer to DLPack / PyTorch.
///
/// The Python export API requires a non-empty shape and byte-aligned, non-zero dtype metadata.
struct SharedBufferTensorDesc
{
    std::vector<std::int64_t> shape;   ///< Tensor dimensions shape array.
    std::vector<std::int64_t> strides; ///< Tensor strides array.
    SharedBufferTensorDTypeCode dtypeCode =
        SharedBufferTensorDTypeCode::Float; ///< Element data type code.
    std::uint8_t dtypeBits   = 32u;         ///< Element bit width (e.g. 32 for float32).
    std::uint16_t dtypeLanes = 1u;          ///< Vector lanes count per element.
    std::uint64_t byteOffset = 0u;          ///< Byte offset from buffer start.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_SHARED_BUFFER_H
