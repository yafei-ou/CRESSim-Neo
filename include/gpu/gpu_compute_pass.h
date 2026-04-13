#ifndef CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H
#define CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H

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

namespace cressim::neo::gpu
{

struct GpuBufferBinding
{
    const char *variableName            = nullptr;
    Diligent::IBuffer *buffer           = nullptr;
    Diligent::BUFFER_VIEW_TYPE viewType = Diligent::BUFFER_VIEW_UNDEFINED;
};

struct GpuComputePassDefinition
{
    const char *shaderPath                                = nullptr;
    const char *shaderName                                = nullptr;
    const char *psoName                                   = nullptr;
    const Diligent::ShaderResourceVariableDesc *variables = nullptr;
    std::size_t variableCount                             = 0u;
};

class GpuComputePass
{
public:
    bool initialize(GpuDevice &device, Diligent::IShaderSourceInputStreamFactory *streamFactory,
                    Diligent::Uint64 immediateContextMask,
                    const GpuComputePassDefinition &definition);

    bool createVariant();
    bool createVariants(std::size_t totalVariantCount);

    Diligent::IPipelineState *pipelineState() const
    {
        return mPso.RawPtr();
    }
    Diligent::IShaderResourceBinding *defaultSrb() const
    {
        return mSrbs.empty() ? nullptr : mSrbs[0].RawPtr();
    }
    Diligent::IShaderResourceBinding *variantSrb(std::size_t index) const;

    template <std::size_t N>
    bool bindVariant(std::size_t variantIndex,
                     const std::array<GpuBufferBinding, N> &bindings) const;

    template <std::size_t N>
    bool dispatch(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                  const std::array<GpuBufferBinding, N> &bindings, std::uint32_t groupCountX,
                  std::uint32_t groupCountY = 1u, std::uint32_t groupCountZ = 1u) const;

    template <std::size_t N>
    bool dispatchIndirect(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                          const std::array<GpuBufferBinding, N> &bindings,
                          Diligent::IBuffer *indirectArgsBuffer,
                          Diligent::Uint64 indirectArgsOffset = 0u) const;

private:
    static bool bindBufferVariable(Diligent::IShaderResourceBinding *srb, const char *variableName,
                                   Diligent::IBuffer *buffer, Diligent::BUFFER_VIEW_TYPE viewType);

    template <std::size_t N>
    static bool bindBufferVariables(Diligent::IShaderResourceBinding *srb,
                                    const std::array<GpuBufferBinding, N> &bindings);

private:
    std::string mShaderPath;
    std::string mShaderName;
    std::string mPsoName;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPso;
    std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>> mSrbs;
};

template <std::size_t N>
bool GpuComputePass::bindBufferVariables(Diligent::IShaderResourceBinding *srb,
                                         const std::array<GpuBufferBinding, N> &bindings)
{
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
                                 const std::array<GpuBufferBinding, N> &bindings) const
{
    return bindBufferVariables(variantSrb(variantIndex), bindings);
}

template <std::size_t N>
bool GpuComputePass::dispatch(Diligent::IDeviceContext *computeContext, std::size_t variantIndex,
                              const std::array<GpuBufferBinding, N> &bindings,
                              std::uint32_t groupCountX, std::uint32_t groupCountY,
                              std::uint32_t groupCountZ) const
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (computeContext == nullptr || mPso == nullptr || srb == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(srb, bindings))
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
                                      Diligent::Uint64 indirectArgsOffset) const
{
    Diligent::IShaderResourceBinding *srb = variantSrb(variantIndex);
    if (computeContext == nullptr || mPso == nullptr || srb == nullptr ||
        indirectArgsBuffer == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(srb, bindings))
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

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_COMPUTE_PASS_H
