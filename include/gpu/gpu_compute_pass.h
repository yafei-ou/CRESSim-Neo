#ifndef CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H
#define CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H

#include "gpu/export.h"
#include "gpu/gpu_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// @file gpu_compute_pass.h
/// @brief Compute pipeline state object (PSO) wrapper, shader resource bindings (SRB), and
/// threadgroup dispatch helpers.

namespace cressim::neo::gpu
{

/// @brief Binding descriptor mapping a GPU buffer to a named compute shader resource variable.
struct GpuBufferBinding
{
    const char *variableName =
        nullptr; ///< Shader HLSL resource variable name (e.g. `g_Positions`).
    Diligent::IBuffer *buffer = nullptr; ///< Pointer to target GPU buffer.
    Diligent::BUFFER_VIEW_TYPE viewType =
        Diligent::BUFFER_VIEW_UNDEFINED; ///< Buffer view type (`BUFFER_VIEW_SHADER_RESOURCE` or
                                         ///< `BUFFER_VIEW_UNORDERED_ACCESS`).
};

/// @brief Binding descriptor mapping a GPU texture view to a named compute shader variable.
struct GpuTextureBinding
{
    const char *variableName     = nullptr; ///< Shader HLSL variable name.
    Diligent::ITextureView *view = nullptr; ///< Pointer to texture view (SRV or UAV).
};

/// @brief Blueprint descriptor for compiling and creating a compute pipeline state object.
struct GpuComputePassDefinition
{
    const char *shaderPath = nullptr; ///< Filesystem path to HLSL compute shader file.
    const char *shaderName = nullptr; ///< Debug identifier for the compiled compute shader.
    const char *psoName    = nullptr; ///< Debug identifier for the compute pipeline state.
    const Diligent::ShaderResourceVariableDesc *variables =
        nullptr; ///< Array of dynamic/mutable shader variable descriptors.
    std::size_t variableCount           = 0u;      ///< Number of variable descriptors.
    const Diligent::ShaderMacro *macros = nullptr; ///< Preprocessor macro definitions array.
    std::size_t macroCount              = 0u;      ///< Number of macro definitions.
    const char *entryPoint   = nullptr; ///< Entry point function name (e.g. "main" or "CSMain").
    const char *shaderSource = nullptr; ///< Optional in-memory raw HLSL source string.
};

/// @brief Wrapper managing a compute pipeline state object (PSO) and reusable shader resource
/// binding (SRB) variants.
class CRESSIM_NEO_GPU_API GpuComputePass
{
public:
    /// @brief Compiles the shader and initializes the compute pipeline state object.
    /// @param device GPU device reference.
    /// @param streamFactory Shader file stream provider factory.
    /// @param immediateContextMask Bitmask of device contexts that can execute this pass.
    /// @param definition Pass definition containing shader source paths, macros, and variable
    /// layouts.
    /// @return True on success.
    bool initialize(GpuDevice &device, Diligent::IShaderSourceInputStreamFactory *streamFactory,
                    Diligent::Uint64 immediateContextMask,
                    const GpuComputePassDefinition &definition);

    /// @brief Creates an additional shader resource binding (SRB) variant for concurrent/distinct
    /// parameter sets.
    /// @return True on success.
    bool createVariant();
    /// @brief Preallocates a specified total number of SRB variants.
    /// @param totalVariantCount Desired number of variants.
    /// @return True on success.
    bool createVariants(std::size_t totalVariantCount);
    /// @brief Forces recreation of all allocated SRB variant instances from the parent PSO.
    /// @return True on success.
    bool forceRecreateAllVariants();

    /// @brief Retrieves the raw Diligent pipeline state object.
    /// @return Pointer to Diligent::IPipelineState.
    Diligent::IPipelineState *pipelineState() const
    {
        return mPso.RawPtr();
    }
    /// @brief Retrieves the default (index 0) shader resource binding.
    /// @return Pointer to Diligent::IShaderResourceBinding or nullptr.
    Diligent::IShaderResourceBinding *defaultSrb() const
    {
        return mVariants.empty() ? nullptr : mVariants[0].srb.RawPtr();
    }
    /// @brief Retrieves the shader resource binding for a specific variant index.
    /// @param index Zero-based variant index.
    /// @return Pointer to Diligent::IShaderResourceBinding or nullptr.
    Diligent::IShaderResourceBinding *variantSrb(std::size_t index) const;

    /// @brief Binds an array of buffer variables to an SRB variant without dispatching.
    /// @tparam N Number of buffer bindings.
    /// @param variantIndex Target SRB variant index.
    /// @param bindings Array of buffer bindings.
    /// @return True on success.
    template <std::size_t N>
    bool bindVariant(std::size_t variantIndex, const std::array<GpuBufferBinding, N> &bindings);

    /// @brief Binds buffer resources and dispatches compute threadgroups.
    /// @tparam N Number of buffer bindings.
    /// @param computeContext Device context to execute dispatch on.
    /// @param variantIndex SRB variant index.
    /// @param bindings Array of buffer bindings.
    /// @param groupCountX Number of threadgroups in X dimension.
    /// @param groupCountY Number of threadgroups in Y dimension.
    /// @param groupCountZ Number of threadgroups in Z dimension.
    /// @return True on success.
    template <std::size_t N>
    bool dispatch(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                  const std::array<GpuBufferBinding, N> &bindings, std::uint32_t groupCountX,
                  std::uint32_t groupCountY = 1u, std::uint32_t groupCountZ = 1u);

    /// @brief Binds buffer resources and dispatches indirect compute threadgroups from GPU argument
    /// buffer.
    /// @tparam N Number of buffer bindings.
    /// @param computeContext Device context executing dispatch.
    /// @param variantIndex SRB variant index.
    /// @param bindings Array of buffer bindings.
    /// @param indirectArgsBuffer GPU buffer containing dispatch argument counts.
    /// @param indirectArgsOffset Byte offset into indirect argument buffer.
    /// @return True on success.
    template <std::size_t N>
    bool dispatchIndirect(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                          const std::array<GpuBufferBinding, N> &bindings,
                          Diligent::IBuffer *indirectArgsBuffer,
                          Diligent::Uint64 indirectArgsOffset = 0u);

    /// @brief Binds both buffer and texture resources, committing and dispatching compute
    /// threadgroups.
    /// @tparam NB Number of buffer bindings.
    /// @tparam NT Number of texture bindings.
    /// @param computeContext Device context executing dispatch.
    /// @param variantIndex SRB variant index.
    /// @param bufferBindings Array of buffer bindings.
    /// @param textureBindings Array of texture bindings.
    /// @param groupCountX Number of threadgroups in X dimension.
    /// @param groupCountY Number of threadgroups in Y dimension.
    /// @param groupCountZ Number of threadgroups in Z dimension.
    /// @return True on success.
    template <std::size_t NB, std::size_t NT>
    bool dispatchResources(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                           const std::array<GpuBufferBinding, NB> &bufferBindings,
                           const std::array<GpuTextureBinding, NT> &textureBindings,
                           std::uint32_t groupCountX, std::uint32_t groupCountY = 1u,
                           std::uint32_t groupCountZ = 1u);

private:
    struct VariantState
    {
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    };

    bool recreateVariant(std::size_t variantIndex);
    bool recreateAllVariants();
    static bool bindBufferVariable(Diligent::IShaderResourceBinding *srb, const char *variableName,
                                   Diligent::IBuffer *buffer, Diligent::BUFFER_VIEW_TYPE viewType);
    static bool bindTextureVariable(Diligent::IShaderResourceBinding *srb, const char *variableName,
                                    Diligent::ITextureView *view);

    template <std::size_t N>
    bool bindBufferVariables(std::size_t variantIndex,
                             const std::array<GpuBufferBinding, N> &bindings);
    template <std::size_t N>
    bool bindTextureVariables(std::size_t variantIndex,
                              const std::array<GpuTextureBinding, N> &bindings);

private:
    std::string mShaderPath;
    std::string mShaderName;
    std::string mPsoName;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPso;
    std::vector<VariantState> mVariants;
};

template <std::size_t N>
bool GpuComputePass::bindBufferVariables(std::size_t variantIndex,
                                         const std::array<GpuBufferBinding, N> &bindings)
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (srb == nullptr)
    {
        return false;
    }

    for (const GpuBufferBinding &binding : bindings)
    {
        if (!bindBufferVariable(srb, binding.variableName, binding.buffer, binding.viewType))
        {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool GpuComputePass::bindVariant(std::size_t variantIndex,
                                 const std::array<GpuBufferBinding, N> &bindings)
{
    return bindBufferVariables(variantIndex, bindings);
}

template <std::size_t N>
bool GpuComputePass::bindTextureVariables(std::size_t variantIndex,
                                          const std::array<GpuTextureBinding, N> &bindings)
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (srb == nullptr)
    {
        return false;
    }

    for (const GpuTextureBinding &binding : bindings)
    {
        if (!bindTextureVariable(srb, binding.variableName, binding.view))
        {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool GpuComputePass::dispatch(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                              const std::array<GpuBufferBinding, N> &bindings,
                              std::uint32_t groupCountX, std::uint32_t groupCountY,
                              std::uint32_t groupCountZ)
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (computeContext == nullptr || mPso == nullptr || srb == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(variantIndex, bindings))
    {
        return false;
    }

    srb = variantSrb(variantIndex);
    if (srb == nullptr)
    {
        return false;
    }

    computeContext->SetPipelineState(mPso);
    computeContext->CommitShaderResources(srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{groupCountX, groupCountY, groupCountZ});
    return true;
}

template <std::size_t N>
bool GpuComputePass::dispatchIndirect(Diligent::IDeviceContext *computeContext,
                                      std::size_t variantIndex,
                                      const std::array<GpuBufferBinding, N> &bindings,
                                      Diligent::IBuffer *indirectArgsBuffer,
                                      Diligent::Uint64 indirectArgsOffset)
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (computeContext == nullptr || mPso == nullptr || srb == nullptr ||
        indirectArgsBuffer == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(variantIndex, bindings))
    {
        return false;
    }

    srb = variantSrb(variantIndex);
    if (srb == nullptr)
    {
        return false;
    }

    computeContext->SetPipelineState(mPso);
    computeContext->CommitShaderResources(srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchComputeIndirect(Diligent::DispatchComputeIndirectAttribs{
        indirectArgsBuffer, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        indirectArgsOffset});
    return true;
}

template <std::size_t NB, std::size_t NT>
bool GpuComputePass::dispatchResources(Diligent::IDeviceContext *computeContext,
                                       std::size_t variantIndex,
                                       const std::array<GpuBufferBinding, NB> &bufferBindings,
                                       const std::array<GpuTextureBinding, NT> &textureBindings,
                                       std::uint32_t groupCountX, std::uint32_t groupCountY,
                                       std::uint32_t groupCountZ)
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (computeContext == nullptr || mPso == nullptr || srb == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(variantIndex, bufferBindings) ||
        !bindTextureVariables(variantIndex, textureBindings))
    {
        return false;
    }

    srb = variantSrb(variantIndex);
    if (srb == nullptr)
    {
        return false;
    }

    computeContext->SetPipelineState(mPso);
    computeContext->CommitShaderResources(srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{groupCountX, groupCountY, groupCountZ});
    return true;
}

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H
