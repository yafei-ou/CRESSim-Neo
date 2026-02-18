#ifndef CRESSIM_NEO_GRAPHICS_DEVICE_DILIGENT_GRAPHICS_DEVICE_H
#define CRESSIM_NEO_GRAPHICS_DEVICE_DILIGENT_GRAPHICS_DEVICE_H

#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cressim::neo::graphics
{

class DiligentGraphicsDevice final : public IGraphicsDevice
{
public:
    bool initialize(const GraphicsDeviceDesc& desc) override;
    void shutdown() override;

    void resizeDefaultRenderTarget(std::uint32_t width, std::uint32_t height) override;
    RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) override;
    bool resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) override;
    void destroyRenderTarget(RenderTargetHandle target) override;
    bool isValidRenderTarget(RenderTargetHandle target) const override;
    RenderTargetHandle defaultRenderTarget() const override;

    void beginFrame(const common::FrameContext& frameContext) override;
    void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) override;
    void beginRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override;
    bool drawPbr(RenderTargetHandle target, const PbrDrawCommand& drawCommand) override;
    void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override;
    void requestReadback(RenderTargetHandle target) override;
    bool tryPopReadbackEvent(RenderTargetReadbackEvent& outEvent) override;
    void endFrame(const common::FrameContext& frameContext) override;

    GraphicsBackend backend() const override;

private:
    struct RenderTargetResources
    {
        RenderTargetDesc desc{};
        RenderViewport viewport{};
        Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
        Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
    };

    struct PendingReadbackCopy
    {
        RenderTargetHandle target{};
        std::uint64_t frameIndex = 0;
        std::uint64_t fenceValue = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    };

    struct CachedMeshGpuData
    {
        std::uint64_t version = 0;
        std::uint32_t indexCount = 0;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> vertexBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> indexBuffer;
    };

    struct PbrPipelineResources
    {
        Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipelineState;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> shaderResourceBinding;
    };

    struct PbrDrawConstants
    {
        float modelMatrix[16] = {};
        float viewProjectionMatrix[16] = {};
        float cameraPositionMetallic[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float lightDirectionIntensity[4] = {0.0f, -1.0f, 0.0f, 1.0f};
        float lightColorRoughness[4] = {1.0f, 1.0f, 1.0f, 0.5f};
        float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    };

    bool initializeVulkan();
    bool createDebugViewerSwapChain();
    bool createDefaultRenderTarget();
    RenderTargetDesc normalizeTargetDesc(const RenderTargetDesc& desc) const;
    bool createRenderTargetTextures(const RenderTargetDesc& desc, RenderTargetResources& resources);

    CachedMeshGpuData* getOrCreateMeshBuffers(const PbrDrawCommand& drawCommand);

    bool createPbrPipeline(bool hasDepthTarget, Diligent::TEXTURE_FORMAT colorFormat, PbrPipelineResources& outResources);
    PbrPipelineResources* getOrCreatePbrPipeline(bool hasDepthTarget, Diligent::TEXTURE_FORMAT colorFormat);

    bool queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex);

    bool resolveShaderDirectory();
    bool loadShaderSource(const char* relativePath, const char* fallbackSource, std::string& outSource);

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;
    bool mHasActiveRenderTarget = false;
    bool mActiveRenderTargetHasDepth = false;
    Diligent::TEXTURE_FORMAT mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    bool mShaderDirectoryResolved = false;
    common::ResourceId mNextRenderTargetId = 1;
    RenderTargetHandle mDefaultRenderTarget{};
    RenderTargetHandle mActiveRenderTarget{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    std::unordered_map<common::ResourceId, CachedMeshGpuData> mCachedMeshes;
    // Targets that requested readback and are waiting for endRenderTarget().
    std::unordered_set<common::ResourceId> mPendingReadbacks;
    // GPU->CPU copy jobs collected during render target completion and consumed in endFrame().
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    // FIFO completion metadata consumed through tryPopReadbackEvent().
    std::deque<RenderTargetReadbackEvent> mCompletedReadbacks;

    std::unordered_map<std::uint64_t, PbrPipelineResources> mPbrPipelineCache;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPbrConstantBuffer;
    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    std::uint64_t mNextReadbackFenceValue = 1;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mSwapChain;

    std::string mResolvedShaderDirectory;
    std::unordered_map<std::string, std::string> mShaderSourceCache;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_DEVICE_DILIGENT_GRAPHICS_DEVICE_H
