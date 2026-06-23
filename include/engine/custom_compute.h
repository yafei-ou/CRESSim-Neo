#ifndef CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H
#define CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H

#include "engine/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cressim::neo::engine
{

enum class CustomComputeResourceKind
{
    Buffer,
};

enum class CustomComputeResourceAccess
{
    ReadOnly,
    WriteOnly,
    ReadWrite,
};

struct CustomComputePassHandle
{
    std::uint64_t id = 0u;

    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

struct CustomComputeResourceDesc
{
    std::string key;
    CustomComputeResourceKind kind     = CustomComputeResourceKind::Buffer;
    CustomComputeResourceAccess access = CustomComputeResourceAccess::ReadOnly;
    std::uint32_t elementCount         = 0u;
    std::uint32_t elementStrideBytes   = 0u;
    std::uint64_t bindingGeneration    = 0u;
};

struct CustomComputeResourceBindingDesc
{
    std::string shaderVariableName;
    std::string resourceKey;
    CustomComputeResourceAccess access = CustomComputeResourceAccess::ReadOnly;
};

enum class CustomComputeDispatchMode
{
    ExplicitGroupCount,
    ResourceElementCount,
};

struct CustomComputeDispatchDesc
{
    CustomComputeDispatchMode mode = CustomComputeDispatchMode::ExplicitGroupCount;
    std::uint32_t groupCountX      = 1u;
    std::uint32_t groupCountY      = 1u;
    std::uint32_t groupCountZ      = 1u;
    std::string countResourceKey;
};

struct CustomComputePassDesc
{
    std::string debugName;
    std::string shaderPath;
    std::string shaderSource;
    std::string entryPoint = "main";
    std::uint32_t threadGroupSizeX = 1u;
    std::uint32_t threadGroupSizeY = 1u;
    std::uint32_t threadGroupSizeZ = 1u;
    std::vector<CustomComputeResourceBindingDesc> resourceBindings;
    std::string constantBufferVariableName;
    std::uint32_t constantBufferSizeBytes = 0u;
    std::vector<std::uint8_t> constantData;
    CustomComputeDispatchDesc dispatch{};
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_CUSTOM_COMPUTE_H
