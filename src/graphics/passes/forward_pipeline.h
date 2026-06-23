#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H

#include "gpu/gpu_device.h"
#include "graphics/host_scene.h"
#include "graphics/passes/render_pass_types.h"
#include "graphics/render_target_cache_key.h"

#include <memory>
#include <unordered_map>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics::detail
{

class ForwardOpaquePass;
class CameraDepthPass;
class CameraSegmentationPass;
class ShadowPass;
class DebugParticlePass;
class DebugRoutedCablePass;
class DebugStrandFramePass;
class SkyboxPass;
class FluidDepthPass;
class FluidColorPass;
class FluidDepthFilterPass;
class FluidCompositePass;

class ForwardPipeline
{
public:
    ForwardPipeline(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                    IblQualityTier iblQualityTier);
    ~ForwardPipeline();

    bool initialize();
    bool executeBatch(const common::FrameContext &frameContext, const CameraBatchView &batchView,
                      const HostSceneView &sceneView,
                      const cressim::neo::physics::PhysicsGpuSceneView *physicsScene,
                      const RenderFrameOptions &options,
                      const std::vector<EnvMainLightState> &envMainLights,
                      ForwardPassExecutionStats &outStats);

private:
    struct GpuIndirectState;
    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    IblQualityTier mIblQualityTier = IblQualityTier::Off;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<CameraDepthPass> mCameraDepthPass;
    std::unique_ptr<CameraSegmentationPass> mCameraSegmentationPass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::unique_ptr<DebugParticlePass> mDebugParticlePass;
    std::unique_ptr<DebugRoutedCablePass> mDebugRoutedCablePass;
    std::unique_ptr<DebugStrandFramePass> mDebugStrandFramePass;
    std::unique_ptr<SkyboxPass> mSkyboxPass;
    std::unique_ptr<FluidDepthPass> mFluidDepthPass;
    std::unique_ptr<FluidColorPass> mFluidColorPass;
    std::unique_ptr<FluidDepthFilterPass> mFluidDepthFilterPass;
    std::unique_ptr<FluidCompositePass> mFluidCompositePass;
    std::unique_ptr<GpuIndirectState> mGpuIndirectState;
    std::unordered_map<RenderTargetCacheKey, gpu::GpuRenderTargetHandle, RenderTargetCacheKeyHasher>
        mLayeredTargetCache;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_PIPELINE_H
