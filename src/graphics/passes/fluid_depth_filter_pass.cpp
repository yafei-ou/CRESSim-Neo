#include "graphics/passes/fluid_depth_filter_pass.h"

#include "gpu/shader_library.h"

#include <cstring>

namespace cressim::neo::graphics::detail
{

namespace
{

constexpr Diligent::ShaderResourceVariableDesc kFilterVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsFluidDepthFilter",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SourceDepth",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FilteredDepthRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kFilterPassDefinition = {
    "graphics/fluid_depth_filter.cs.hlsl",
    "CRESSimNeo.Graphics.FluidDepthFilter.CS",
    "CRESSimNeo.Graphics.FluidDepthFilter.PSO",
    kFilterVars,
    std::size(kFilterVars),
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + 7u) / 8u;
}

} // namespace

FluidDepthFilterPass::FluidDepthFilterPass(gpu::GpuDevice &device) : mDevice(device) {}

bool FluidDepthFilterPass::initialize()
{
    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }
    const Diligent::Uint64 graphicsContextMask = gpu::contextMaskForId(backendContext.contextId);

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr ||
        !mFilterPass.initialize(mDevice, streamFactory, graphicsContextMask, kFilterPassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.FluidDepthFilterPass.Constants";
    constantsDesc.Size                 = sizeof(FilterConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = graphicsContextMask;
    backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &mConstantsBuffer);
    mInitialized = mConstantsBuffer != nullptr;
    return mInitialized;
}

bool FluidDepthFilterPass::filter(Diligent::ITextureView *sourceSrv, Diligent::ITextureView *destUav,
                                  const ResolvedCameraView &camera, std::uint32_t sourceLayer,
                                  const RenderFrameOptions::FluidRenderingOptions &options)
{
    if (!mInitialized || sourceSrv == nullptr || destUav == nullptr)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.graphicsContext == nullptr)
    {
        return false;
    }

    FilterConstants constants{};
    constants.layer = sourceLayer;
    constants.outputWidth = camera.outputTargetDesc.width;
    constants.outputHeight = camera.outputTargetDesc.height;
    constants.filterRadius = static_cast<std::uint32_t>(options.filterRadiusPixels < 1.0f
                                                            ? 1.0f
                                                            : options.filterRadiusPixels);
    constants.depthEdgeThreshold = options.depthEdgeThreshold;

    void *mapped = nullptr;
    backendContext.graphicsContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    const std::array bufferBindings{
        gpu::GpuBufferBinding{"GraphicsFluidDepthFilter", mConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };
    const std::array textureBindings{
        gpu::GpuTextureBinding{"g_SourceDepth", sourceSrv},
        gpu::GpuTextureBinding{"g_FilteredDepthRW", destUav},
    };

    return mFilterPass.dispatchResources(backendContext.graphicsContext, 0u, bufferBindings,
                                         textureBindings, dispatchGroupCount(constants.outputWidth),
                                         dispatchGroupCount(constants.outputHeight), 1u);
}

} // namespace cressim::neo::graphics::detail
