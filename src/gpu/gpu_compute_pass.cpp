#include "gpu/gpu_compute_pass.h"

#include "common/logger.h"

namespace cressim::neo::gpu
{

bool GpuComputePass::initialize(GpuDevice &device,
                                Diligent::IShaderSourceInputStreamFactory *streamFactory,
                                Diligent::Uint64 immediateContextMask,
                                const GpuComputePassDefinition &definition)
{
    mShaderPath = definition.shaderPath != nullptr ? definition.shaderPath : "";
    mShaderName = definition.shaderName != nullptr ? definition.shaderName : "";
    mPsoName    = definition.psoName != nullptr ? definition.psoName : "";
    mPso        = nullptr;
    mVariants.clear();

    if (streamFactory == nullptr)
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = mShaderName.c_str();
    shaderCreateInfo.FilePath                        = mShaderPath.c_str();
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> computeShader;
    if (!device.createShader(shaderCreateInfo, &computeShader))
    {
        computeShader = nullptr;
    }
    if (computeShader == nullptr)
    {
        return false;
    }

    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                 = mPsoName.c_str();
    psoCreateInfo.PSODesc.PipelineType         = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ImmediateContextMask = immediateContextMask;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    psoCreateInfo.PSODesc.ResourceLayout.Variables = definition.variables;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(definition.variableCount);
    psoCreateInfo.pCS = computeShader;

    if (!device.createComputePipelineState(psoCreateInfo, &mPso))
    {
        mPso = nullptr;
    }
    if (mPso == nullptr)
    {
        return false;
    }

    return createVariant();
}

bool GpuComputePass::createVariant()
{
    if (mPso == nullptr)
    {
        return false;
    }

    VariantState variantState;
    mPso->CreateShaderResourceBinding(&variantState.srb, true);

    if (variantState.srb == nullptr)
    {
        return false;
    }

    mVariants.push_back(std::move(variantState));
    return true;
}

bool GpuComputePass::createVariants(std::size_t totalVariantCount)
{
    if (totalVariantCount == 0u)
    {
        return false;
    }

    while (mVariants.size() < totalVariantCount)
    {
        if (!createVariant())
        {
            return false;
        }
    }
    return true;
}

bool GpuComputePass::forceRecreateAllVariants()
{
    return recreateAllVariants();
}

Diligent::IShaderResourceBinding *GpuComputePass::variantSrb(std::size_t index) const
{
    if (index >= mVariants.size())
    {
        return nullptr;
    }
    return mVariants[index].srb.RawPtr();
}

bool GpuComputePass::recreateVariant(std::size_t variantIndex)
{
    VariantState newVariantState;
    if (variantIndex >= mVariants.size() || mPso == nullptr)
    {
        return false;
    }

    mPso->CreateShaderResourceBinding(&newVariantState.srb, true);
    if (newVariantState.srb == nullptr)
    {
        return false;
    }

    mVariants[variantIndex] = std::move(newVariantState);
    return true;
}

bool GpuComputePass::recreateAllVariants()
{
    for (std::size_t i = 0; i < mVariants.size(); ++i)
    {
        if (!recreateVariant(i))
        {
            return false;
        }
    }
    return true;
}

bool GpuComputePass::bindBufferVariable(Diligent::IShaderResourceBinding *srb,
                                        const char *variableName, Diligent::IBuffer *buffer,
                                        Diligent::BUFFER_VIEW_TYPE viewType)
{
    if (srb == nullptr || buffer == nullptr)
    {
        CRESSIM_LOG_ERROR("GpuComputePass: invalid buffer binding for '", variableName, "'.");
        return false;
    }

    Diligent::IShaderResourceVariable *variable =
        srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, variableName);
    if (variable == nullptr)
    {
        CRESSIM_LOG_ERROR("GpuComputePass: shader variable not found: '", variableName, "'.");
        return false;
    }

    Diligent::IBufferView *view = buffer->GetDefaultView(viewType);
    if (view != nullptr)
    {
        variable->Set(view);
    }
    else
    {
        variable->Set(buffer);
    }
    return true;
}

} // namespace cressim::neo::gpu
