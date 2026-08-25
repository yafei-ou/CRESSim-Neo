#ifndef CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H
#define CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H

#include "engine/export.h"
#include "engine/shared_buffer.h"
#include "gpu/gpu_types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Resource type for custom compute shader bindings.
enum class CustomComputeResourceKind
{
    Buffer,  ///< Structured or raw GPU buffer resource.
    Texture, ///< Texture image resource.
};

/// @brief Access permission mode for custom compute shader resources.
enum class CustomComputeResourceAccess
{
    ReadOnly,  ///< Read-only shader resource.
    WriteOnly, ///< Write-only unordered access resource.
    ReadWrite, ///< Read-write unordered access resource.
};

/// @brief Handle wrapper identifying a registered custom compute shader pass.
struct CustomComputePassHandle
{
    std::uint64_t id = 0u; ///< Unique handle identifier.

    /// @brief Checks if the custom compute pass handle is valid.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

/// @brief Metadata descriptor for a custom compute GPU resource.
struct CustomComputeResourceDesc
{
    std::string key; ///< Unique resource string identifier key.
    CustomComputeResourceKind kind =
        CustomComputeResourceKind::Buffer; ///< Resource type kind (Buffer, Texture).
    CustomComputeResourceAccess access = CustomComputeResourceAccess::ReadOnly; ///< Access mode.
    std::uint32_t elementCount         = 0u; ///< Number of elements in resource.
    std::uint32_t elementStrideBytes   = 0u; ///< Stride per element in bytes.
    std::uint64_t bindingGeneration    = 0u; ///< Resource binding invalidation generation key.
};

/// @brief Binding descriptor mapping a GPU resource to a compute shader variable name.
struct CustomComputeResourceBindingDesc
{
    std::string shaderVariableName;          ///< HLSL shader variable name.
    std::string resourceKey;                 ///< Registered engine resource key.
    SharedBufferHandle sharedBufferHandle{}; ///< Optional shared engine GPU buffer handle.
    gpu::GpuRenderTargetBinding renderTargetBinding{}; ///< Optional render target binding.
    gpu::GpuRenderTargetTexturePlane renderTargetTexturePlane =
        gpu::GpuRenderTargetTexturePlane::Color; ///< Texture plane selection (Color, Depth).
    CustomComputeResourceAccess access =
        CustomComputeResourceAccess::ReadOnly; ///< Resource access mode.
};

/// @brief Compute dispatch mode for executing custom compute passes.
enum class CustomComputeDispatchMode
{
    ExplicitGroupCount,   ///< Dispatch using explicit 3D thread group counts.
    ResourceElementCount, ///< Compute thread group count dynamically from resource element count.
};

/// @brief Dispatch execution parameters for custom compute passes.
struct CustomComputeDispatchDesc
{
    CustomComputeDispatchMode mode =
        CustomComputeDispatchMode::ExplicitGroupCount; ///< Dispatch mode.
    std::uint32_t groupCountX = 1u; ///< Dispatch thread group count along X dimension.
    std::uint32_t groupCountY = 1u; ///< Dispatch thread group count along Y dimension.
    std::uint32_t groupCountZ = 1u; ///< Dispatch thread group count along Z dimension.
    std::string countResourceKey;   ///< Resource key used for element count dispatch mode.
};

/// @brief Complete descriptor for compiling and instantiating a custom compute shader pass.
struct CustomComputePassDesc
{
    std::string debugName;                 ///< Debug label for diagnostics.
    std::filesystem::path shaderDirectory; ///< Root directory for shader source files.
    std::string shaderPath;                ///< Shader source path relative to shaderDirectory.
    std::string shaderSource;              ///< Optional raw HLSL shader source code string.
    std::vector<std::filesystem::path>
        includeDirectories;                  ///< Custom shader search paths for HLSL includes.
    std::string entryPoint         = "main"; ///< Shader entry point function name.
    std::uint32_t threadGroupSizeX = 1u;     ///< Workgroup size X dimension.
    std::uint32_t threadGroupSizeY = 1u;     ///< Workgroup size Y dimension.
    std::uint32_t threadGroupSizeZ = 1u;     ///< Workgroup size Z dimension.
    std::vector<CustomComputeResourceBindingDesc> resourceBindings; ///< Resource bindings array.
    std::string constantBufferVariableName;     ///< Constant buffer HLSL variable name.
    std::uint32_t constantBufferSizeBytes = 0u; ///< Constant buffer size in bytes.
    std::vector<std::uint8_t> constantData;     ///< Initial constant buffer binary data payload.
    CustomComputeDispatchDesc dispatch{};       ///< Dispatch parameters.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H
