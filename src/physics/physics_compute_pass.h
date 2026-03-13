#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_COMPUTE_PASS_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_COMPUTE_PASS_H

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cressim::neo::physics
{

struct ComputeBufferBinding
{
    const char* variableName;
    Diligent::IBuffer* buffer;
    Diligent::BUFFER_VIEW_TYPE viewType;
};

struct ComputePassDefinition
{
    const char* shaderPath;
    const char* shaderName;
    const char* psoName;
    const Diligent::ShaderResourceVariableDesc* variables;
    std::size_t variableCount;
};

class ComputePass
{
public:
    bool initialize(Diligent::IRenderDevice* renderDevice,
                    Diligent::IShaderSourceInputStreamFactory* streamFactory,
                    Diligent::Uint64 immediateContextMask, const ComputePassDefinition& definition);

    bool createVariant();
    bool createVariants(std::size_t totalVariantCount);

    Diligent::IPipelineState* pipelineState() const
    {
        return mPso;
    }
    Diligent::IShaderResourceBinding* defaultSrb() const
    {
        return mSrbs.empty() ? nullptr : mSrbs[0];
    }
    Diligent::IShaderResourceBinding* variantSrb(std::size_t index) const;

    template <std::size_t N>
    bool bindVariant(std::size_t variantIndex,
                     const std::array<ComputeBufferBinding, N>& bindings) const;

    template <std::size_t N>
    bool dispatch(Diligent::IDeviceContext* computeContext, std::size_t variantIndex,
                  const std::array<ComputeBufferBinding, N>& bindings, std::uint32_t groupCountX,
                  std::uint32_t groupCountY = 1u, std::uint32_t groupCountZ = 1u) const;

private:
    static bool bindBufferVariable(Diligent::IShaderResourceBinding* srb, const char* variableName,
                                   Diligent::IBuffer* buffer, Diligent::BUFFER_VIEW_TYPE viewType);

    template <std::size_t N>
    static bool bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                                    const std::array<ComputeBufferBinding, N>& bindings);

private:
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPso;
    std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>> mSrbs;
};

template <std::size_t N>
bool ComputePass::bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                                      const std::array<ComputeBufferBinding, N>& bindings)
{
    for (const ComputeBufferBinding& binding : bindings)
    {
        if (!bindBufferVariable(srb, binding.variableName, binding.buffer, binding.viewType))
        {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool ComputePass::bindVariant(std::size_t variantIndex,
                              const std::array<ComputeBufferBinding, N>& bindings) const
{
    return bindBufferVariables(variantSrb(variantIndex), bindings);
}

template <std::size_t N>
bool ComputePass::dispatch(Diligent::IDeviceContext* computeContext, std::size_t variantIndex,
                           const std::array<ComputeBufferBinding, N>& bindings,
                           std::uint32_t groupCountX, std::uint32_t groupCountY,
                           std::uint32_t groupCountZ) const
{
    Diligent::IShaderResourceBinding* srb = variantSrb(variantIndex);
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

} // namespace cressim::neo::physics

#endif // !CRESSIM_NEO_PHYSICS_PHYSICS_COMPUTE_PASS_H