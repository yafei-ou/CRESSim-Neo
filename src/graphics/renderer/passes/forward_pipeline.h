#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "graphics/host_scene.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardOpaquePass;
class ShadowPass;
class CameraBatchPresentPass;

class ForwardPipeline
{
public:
    ForwardPipeline(gpu::GpuDevice& device, RenderResourceManager& resourceManager);
    ~ForwardPipeline();

    bool initialize();
    bool executeBatch(const common::FrameContext& frameContext, const CameraBatchView& batchView,
                      const HostSceneView& sceneView, ForwardPassExecutionStats& outStats);

private:
    struct GpuIndirectState;
    struct LayeredTargetKey
    {
        std::uint32_t width = 0u;
        std::uint32_t height = 0u;
        std::uint32_t arraySize = 1u;
        bool color = true;
        bool depth = true;
        bool shaderReadable = true;
        bool layeredRendering = true;
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
        std::string debugName{};

        bool operator==(const LayeredTargetKey& rhs) const noexcept
        {
            return width == rhs.width && height == rhs.height && arraySize == rhs.arraySize &&
                   color == rhs.color && depth == rhs.depth &&
                   shaderReadable == rhs.shaderReadable &&
                   layeredRendering == rhs.layeredRendering &&
                   colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat &&
                   debugName == rhs.debugName;
        }
    };
    struct LayeredTargetKeyHasher
    {
        std::size_t operator()(const LayeredTargetKey& key) const noexcept;
    };

    gpu::GpuDevice& mDevice;
    RenderResourceManager& mResourceManager;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::unique_ptr<CameraBatchPresentPass> mCameraBatchPresentPass;
    std::unique_ptr<GpuIndirectState> mGpuIndirectState;
    std::unordered_map<LayeredTargetKey, gpu::GpuRenderTargetHandle, LayeredTargetKeyHasher>
        mLayeredTargetCache;
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
