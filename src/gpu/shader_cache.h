#ifndef CRESSIM_NEO_GPU_SHADER_CACHE_H
#define CRESSIM_NEO_GPU_SHADER_CACHE_H

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsTools/interface/RenderStateCache.h"

#include <filesystem>

namespace cressim::neo::gpu
{

class ShaderCache
{
public:
    bool initialize(Diligent::IRenderDevice *renderDevice);
    void shutdown();

    bool createShader(const Diligent::ShaderCreateInfo &createInfo, Diligent::IShader **shader);
    bool createGraphicsPipelineState(const Diligent::GraphicsPipelineStateCreateInfo &createInfo,
                                     Diligent::IPipelineState **pipelineState);
    bool createComputePipelineState(const Diligent::ComputePipelineStateCreateInfo &createInfo,
                                    Diligent::IPipelineState **pipelineState);

private:
    void loadCache();
    void saveCache();
    std::filesystem::path makeCacheFilePath(Diligent::IRenderDevice *renderDevice) const;

private:
    Diligent::RefCntAutoPtr<Diligent::IRenderStateCache> mStateCache;
    std::filesystem::path mCacheFilePath;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHADER_CACHE_H
