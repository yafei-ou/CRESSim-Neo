#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cressim::neo::graphics
{

namespace
{

std::uint32_t clampExtent(std::uint32_t value)
{
    return std::max<std::uint32_t>(value, 1u);
}

float clampNormalized(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    RenderViewport normalized{};
    normalized.x = clampNormalized(viewport.x);
    normalized.y = clampNormalized(viewport.y);
    normalized.width = clampNormalized(viewport.width);
    normalized.height = clampNormalized(viewport.height);

    const float maxWidth = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width = std::min(normalized.width, maxWidth);
    normalized.height = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y = 0.0f;
    }

    return normalized;
}

struct RenderTargetResources
{
    RenderTargetDesc desc{};
    RenderViewport viewport{};
    Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
    Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
};

class DiligentGraphicsDevice final : public IGraphicsDevice
{
public:
    bool initialize(const GraphicsDeviceDesc& desc) override
    {
        shutdown();

        mDesc = desc;
        mDesc.initialWidth = clampExtent(mDesc.initialWidth);
        mDesc.initialHeight = clampExtent(mDesc.initialHeight);

        if (mDesc.preferredBackend == GraphicsBackend::Null)
        {
            mBackend = GraphicsBackend::Null;
            mInitialized = true;
            if (!createDefaultRenderTarget())
            {
                shutdown();
                return false;
            }
            return mInitialized;
        }

        // Vulkan-only backend path.
        if (mDesc.preferredBackend != GraphicsBackend::Vulkan)
        {
            return false;
        }

        if (!initializeVulkan())
        {
            shutdown();
            return false;
        }

        mInitialized = true;
        if (!createDefaultRenderTarget())
        {
            shutdown();
            return false;
        }

        return mInitialized;
    }

    void shutdown() override
    {
        mImmediateContext = nullptr;
        mRenderDevice = nullptr;

        mRenderTargets.clear();
        mPendingReadbacks.clear();
        mCompletedReadbacks.clear();
        mDefaultRenderTarget = {};
        mNextRenderTargetId = 1;

        mBackend = GraphicsBackend::Null;
        mInitialized = false;
    }

    void resizeDefaultRenderTarget(std::uint32_t width, std::uint32_t height) override
    {
        mDesc.initialWidth = clampExtent(width);
        mDesc.initialHeight = clampExtent(height);

        if (!mInitialized || !isValidRenderTarget(mDefaultRenderTarget))
        {
            return;
        }

        (void)resizeRenderTarget(mDefaultRenderTarget, mDesc.initialWidth, mDesc.initialHeight);
    }

    RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) override
    {
        if (!mInitialized)
        {
            return {};
        }

        if (mBackend == GraphicsBackend::Vulkan && !mRenderDevice)
        {
            return {};
        }

        RenderTargetResources resources{};
        resources.desc = normalizeTargetDesc(desc);
        resources.viewport = normalizeViewport(RenderViewport{});

        if (mBackend == GraphicsBackend::Vulkan)
        {
            if (!createRenderTargetTextures(resources.desc, resources))
            {
                return {};
            }
        }

        const common::ResourceId id = mNextRenderTargetId++;
        mRenderTargets.emplace(id, std::move(resources));
        return RenderTargetHandle{id};
    }

    bool resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) override
    {
        if (!mInitialized)
        {
            return false;
        }

        const auto it = mRenderTargets.find(target.id);
        if (it == mRenderTargets.end())
        {
            return false;
        }

        RenderTargetDesc resizedDesc = it->second.desc;
        resizedDesc.width = clampExtent(width == 0 ? resizedDesc.width : width);
        resizedDesc.height = clampExtent(height == 0 ? resizedDesc.height : height);

        if (mBackend == GraphicsBackend::Vulkan)
        {
            RenderTargetResources resizedResources{};
            resizedResources.desc = resizedDesc;
            if (!createRenderTargetTextures(resizedDesc, resizedResources))
            {
                return false;
            }
            it->second.colorTexture = std::move(resizedResources.colorTexture);
            it->second.depthTexture = std::move(resizedResources.depthTexture);
        }

        it->second.desc = resizedDesc;

        if (target.id == mDefaultRenderTarget.id)
        {
            mDesc.initialWidth = resizedDesc.width;
            mDesc.initialHeight = resizedDesc.height;
        }

        return true;
    }

    void destroyRenderTarget(RenderTargetHandle target) override
    {
        if (target.id == common::kInvalidResourceId || target.id == mDefaultRenderTarget.id)
        {
            return;
        }

        mPendingReadbacks.erase(target.id);
        mCompletedReadbacks.erase(
            std::remove_if(
                mCompletedReadbacks.begin(),
                mCompletedReadbacks.end(),
                [&](const RenderTargetReadbackEvent& event) { return event.target.id == target.id; }),
            mCompletedReadbacks.end());
        mRenderTargets.erase(target.id);
    }

    bool isValidRenderTarget(RenderTargetHandle target) const override
    {
        if (target.id == common::kInvalidResourceId)
        {
            return false;
        }
        return mRenderTargets.find(target.id) != mRenderTargets.end();
    }

    RenderTargetHandle defaultRenderTarget() const override
    {
        return mDefaultRenderTarget;
    }

    void beginFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;
    }

    void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) override
    {
        const auto it = mRenderTargets.find(target.id);
        if (it == mRenderTargets.end())
        {
            return;
        }

        it->second.viewport = normalizeViewport(viewport);
    }

    void beginRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override
    {
        (void)frameContext;

        if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
        {
            return;
        }

        const auto it = mRenderTargets.find(target.id);
        if (it == mRenderTargets.end())
        {
            return;
        }

        Diligent::ITextureView* colorRtv = nullptr;
        Diligent::ITextureView* depthDsv = nullptr;
        if (it->second.colorTexture != nullptr)
        {
            colorRtv = it->second.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
        }
        if (it->second.depthTexture != nullptr)
        {
            depthDsv = it->second.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
        }

        if (colorRtv == nullptr && depthDsv == nullptr)
        {
            return;
        }

        if (colorRtv != nullptr)
        {
            mImmediateContext->SetRenderTargets(1, &colorRtv, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            constexpr float clearColor[4] = {0.02f, 0.02f, 0.03f, 1.0f};
            mImmediateContext->ClearRenderTarget(colorRtv, clearColor, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
        else
        {
            mImmediateContext->SetRenderTargets(0, nullptr, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        if (depthDsv != nullptr)
        {
            mImmediateContext->ClearDepthStencil(depthDsv, Diligent::CLEAR_DEPTH_FLAG, 1.0f, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        const float targetWidth = static_cast<float>(it->second.desc.width);
        const float targetHeight = static_cast<float>(it->second.desc.height);
        const RenderViewport viewport = it->second.viewport;

        Diligent::Viewport diligentViewport{};
        diligentViewport.TopLeftX = viewport.x * targetWidth;
        diligentViewport.TopLeftY = viewport.y * targetHeight;
        diligentViewport.Width = viewport.width * targetWidth;
        diligentViewport.Height = viewport.height * targetHeight;
        diligentViewport.MinDepth = 0.0f;
        diligentViewport.MaxDepth = 1.0f;
        mImmediateContext->SetViewports(1, &diligentViewport, it->second.desc.width, it->second.desc.height);
    }

    void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override
    {
        // Transition readback request state for this target: pending -> completed.
        // NOTE: completion is currently logical (render pass ended), not GPU-fence backed.
        // NOTE: duplicate events for the same target/frame are possible if multiple passes
        // write to that target; this is acceptable for scaffolding and will be deduplicated
        // when readback moves to real copy/fence completion.
        const auto pendingReadbackIt = mPendingReadbacks.find(target.id);
        if (pendingReadbackIt == mPendingReadbacks.end())
        {
            return;
        }

        RenderTargetReadbackEvent event{};
        event.target = target;
        event.frameIndex = frameContext.frameIndex;
        mCompletedReadbacks.push_back(event);
        mPendingReadbacks.erase(pendingReadbackIt);
    }

    void requestReadback(RenderTargetHandle target) override
    {
        const auto it = mRenderTargets.find(target.id);
        if (it == mRenderTargets.end())
        {
            return;
        }
        if (!it->second.desc.cpuReadback)
        {
            return;
        }

        // Deduplicates repeated requests in the same frame.
        mPendingReadbacks.insert(target.id);
    }

    bool tryPopReadbackEvent(RenderTargetReadbackEvent& outEvent) override
    {
        if (mCompletedReadbacks.empty())
        {
            return false;
        }

        outEvent = mCompletedReadbacks.front();
        mCompletedReadbacks.pop_front();
        return true;
    }

    void endFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;

        if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
        {
            // Requests that did not get rendered this frame are discarded.
            // TODO: keep/age these requests if later behavior needs cross-frame persistence.
            mPendingReadbacks.clear();
            return;
        }

        // Placeholder hook for async staging-buffer copies when CPU readback is requested.
        // TODO: execute copy-to-staging + fence signaling and only then push completion events.
        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();

        mPendingReadbacks.clear();
    }

    GraphicsBackend backend() const override
    {
        return mBackend;
    }

private:
    bool initializeVulkan()
    {
        Diligent::IEngineFactoryVk* factoryVk = Diligent::LoadAndGetEngineFactoryVk();
        if (factoryVk == nullptr)
        {
            return false;
        }

        Diligent::EngineVkCreateInfo engineCreateInfo{};
        engineCreateInfo.EnableValidation = static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);
        factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, &mImmediateContext);
        if (!mRenderDevice || !mImmediateContext)
        {
            return false;
        }

        mBackend = GraphicsBackend::Vulkan;
        return true;
    }

    bool createDefaultRenderTarget()
    {
        RenderTargetDesc defaultDesc{};
        defaultDesc.width = mDesc.initialWidth;
        defaultDesc.height = mDesc.initialHeight;
        defaultDesc.color = true;
        defaultDesc.depth = true;
        defaultDesc.shaderReadable = true;
        defaultDesc.debugName = "CRESSimNeo.Headless.Default";

        mDefaultRenderTarget = createRenderTarget(defaultDesc);
        return isValidRenderTarget(mDefaultRenderTarget);
    }

    RenderTargetDesc normalizeTargetDesc(const RenderTargetDesc& desc) const
    {
        RenderTargetDesc normalized = desc;
        normalized.width = clampExtent(normalized.width == 0 ? mDesc.initialWidth : normalized.width);
        normalized.height = clampExtent(normalized.height == 0 ? mDesc.initialHeight : normalized.height);
        if (!normalized.color && !normalized.depth)
        {
            normalized.color = true;
        }
        if (normalized.debugName.empty())
        {
            normalized.debugName = "CRESSimNeo.RenderTarget";
        }
        return normalized;
    }

    bool createRenderTargetTextures(const RenderTargetDesc& desc, RenderTargetResources& resources)
    {
        if (!mRenderDevice || (!desc.color && !desc.depth))
        {
            return false;
        }

        resources.colorTexture = nullptr;
        resources.depthTexture = nullptr;

        if (desc.color)
        {
            Diligent::TextureDesc colorDesc{};
            const std::string colorName = desc.debugName + ".Color";
            colorDesc.Name = colorName.c_str();
            colorDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
            colorDesc.Width = desc.width;
            colorDesc.Height = desc.height;
            colorDesc.MipLevels = 1;
            colorDesc.ArraySize = 1;
            colorDesc.Format = Diligent::TEX_FORMAT_RGBA8_UNORM;
            colorDesc.BindFlags = Diligent::BIND_RENDER_TARGET;
            if (desc.shaderReadable)
            {
                colorDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
            }
            colorDesc.Usage = Diligent::USAGE_DEFAULT;

            mRenderDevice->CreateTexture(colorDesc, nullptr, &resources.colorTexture);
            if (!resources.colorTexture || resources.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET) == nullptr)
            {
                return false;
            }
        }

        if (desc.depth)
        {
            Diligent::TextureDesc depthDesc{};
            const std::string depthName = desc.debugName + ".Depth";
            depthDesc.Name = depthName.c_str();
            depthDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
            depthDesc.Width = desc.width;
            depthDesc.Height = desc.height;
            depthDesc.MipLevels = 1;
            depthDesc.ArraySize = 1;
            depthDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
            depthDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL;
            if (desc.shaderReadable)
            {
                depthDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
            }
            depthDesc.Usage = Diligent::USAGE_DEFAULT;

            mRenderDevice->CreateTexture(depthDesc, nullptr, &resources.depthTexture);
            if (!resources.depthTexture || resources.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL) == nullptr)
            {
                return false;
            }
        }

        return true;
    }

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;
    common::ResourceId mNextRenderTargetId = 1;
    RenderTargetHandle mDefaultRenderTarget{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    // Targets that requested readback and are waiting for endRenderTarget().
    std::unordered_set<common::ResourceId> mPendingReadbacks;
    // FIFO completion metadata consumed through tryPopReadbackEvent().
    std::deque<RenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
};
} // namespace

std::unique_ptr<IGraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<DiligentGraphicsDevice>();
}

} // namespace cressim::neo::graphics
