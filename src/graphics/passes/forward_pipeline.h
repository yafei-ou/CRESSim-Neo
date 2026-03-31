#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H

#include "gpu/gpu_device.h"
#include "graphics/host_scene.h"
#include "graphics/passes/render_pass_types.h"
#include "graphics/render_target_cache_key.h"

#include <memory>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class ForwardOpaquePass;
class ShadowPass;

class ForwardPipeline
{
public:
    ForwardPipeline(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                    IblQualityTier iblQualityTier);
    ~ForwardPipeline();

    bool initialize();
    bool executeBatch(const common::FrameContext &frameContext, const CameraBatchView &batchView,
                      const HostSceneView &sceneView,
                      const std::vector<EnvMainLightState> &envMainLights,
                      ForwardPassExecutionStats &outStats);

private:
    struct GpuIndirectState;
    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    IblQualityTier mIblQualityTier = IblQualityTier::Off;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::unique_ptr<GpuIndirectState> mGpuIndirectState;
    std::unordered_map<RenderTargetCacheKey, gpu::GpuRenderTargetHandle, RenderTargetCacheKeyHasher>
        mLayeredTargetCache;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H
