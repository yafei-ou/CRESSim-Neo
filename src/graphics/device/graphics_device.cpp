#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

struct PendingReadbackCopy
{
    RenderTargetHandle target{};
    std::uint64_t frameIndex = 0;
    std::uint64_t fenceValue = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
};

constexpr char kDebugTriangleVsSource[] = R"(
struct VSOutput
{
    float4 Position : SV_Position;
    float3 Color : COLOR0;
};

void main(in uint VertexId : SV_VertexID, out VSOutput Out)
{
    float2 positions[3] = {
        float2(0.0, 0.6),
        float2(0.6, -0.6),
        float2(-0.6, -0.6)
    };
    float3 colors[3] = {
        float3(1.0, 0.2, 0.2),
        float3(0.2, 1.0, 0.2),
        float3(0.2, 0.3, 1.0)
    };

    Out.Position = float4(positions[VertexId], 0.0, 1.0);
    Out.Color = colors[VertexId];
}
)";

constexpr char kDebugTrianglePsSource[] = R"(
struct VSOutput
{
    float4 Position : SV_Position;
    float3 Color : COLOR0;
};

float4 main(in VSOutput In) : SV_Target
{
    return float4(In.Color, 1.0);
}
)";

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
        mDebugTrianglePsoWithDepth = nullptr;
        mDebugTrianglePsoNoDepth = nullptr;
        mReadbackFence = nullptr;
        mNextReadbackFenceValue = 1;
        mImmediateContext = nullptr;
        mRenderDevice = nullptr;

        mRenderTargets.clear();
        mPendingReadbacks.clear();
        mPendingReadbackCopies.clear();
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
        mPendingReadbackCopies.erase(
            std::remove_if(
                mPendingReadbackCopies.begin(),
                mPendingReadbackCopies.end(),
                [&](const PendingReadbackCopy& copy) { return copy.target.id == target.id; }),
            mPendingReadbackCopies.end());
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

    bool drawDebugTriangle(RenderTargetHandle target) override
    {
        if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext || !mRenderDevice)
        {
            return false;
        }

        const auto it = mRenderTargets.find(target.id);
        if (it == mRenderTargets.end())
        {
            return false;
        }
        if (it->second.colorTexture == nullptr)
        {
            return false;
        }

        Diligent::IPipelineState* trianglePso = getDebugTrianglePso(it->second.depthTexture != nullptr);
        if (trianglePso == nullptr)
        {
            return false;
        }

        mImmediateContext->SetPipelineState(trianglePso);

        Diligent::DrawAttribs drawAttribs{};
        drawAttribs.NumVertices = 3;
        drawAttribs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
        mImmediateContext->Draw(drawAttribs);
        return true;
    }

    void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override
    {
        const auto pendingReadbackIt = mPendingReadbacks.find(target.id);
        if (pendingReadbackIt == mPendingReadbacks.end())
        {
            return;
        }

        mPendingReadbacks.erase(pendingReadbackIt);

        if (mBackend == GraphicsBackend::Vulkan && mImmediateContext != nullptr)
        {
            mImmediateContext->SetRenderTargets(0, nullptr, nullptr, Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        }

        if (queueReadbackCopy(target, frameContext.frameIndex))
        {
            return;
        }

        // Fallback path for targets/backends without pixel payload support.
        RenderTargetReadbackEvent event{};
        event.target = target;
        event.frameIndex = frameContext.frameIndex;
        mCompletedReadbacks.push_back(std::move(event));
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
            mPendingReadbackCopies.clear();
            return;
        }

        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();

        for (const PendingReadbackCopy& copy : mPendingReadbackCopies)
        {
            RenderTargetReadbackEvent event{};
            event.target = copy.target;
            event.frameIndex = copy.frameIndex;

            if (copy.stagingTexture != nullptr && copy.width > 0 && copy.height > 0)
            {
                if (mReadbackFence != nullptr && copy.fenceValue > 0)
                {
                    mReadbackFence->Wait(copy.fenceValue);
                }

                Diligent::MappedTextureSubresource mappedData{};
                mImmediateContext->MapTextureSubresource(
                    copy.stagingTexture,
                    0,
                    0,
                    Diligent::MAP_READ,
                    Diligent::MAP_FLAG_DO_NOT_WAIT,
                    nullptr,
                    mappedData);

                if (mappedData.pData != nullptr)
                {
                    event.width = copy.width;
                    event.height = copy.height;
                    event.rowStrideBytes = copy.width * 4u;
                    event.colorRgba8.resize(static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height));

                    const auto* srcRows = static_cast<const std::uint8_t*>(mappedData.pData);
                    auto* dstRows = event.colorRgba8.data();
                    for (std::uint32_t y = 0; y < event.height; ++y)
                    {
                        std::memcpy(
                            dstRows + static_cast<std::size_t>(y) * event.rowStrideBytes,
                            srcRows + static_cast<std::size_t>(y) * static_cast<std::size_t>(mappedData.Stride),
                            event.rowStrideBytes);
                    }

                    mImmediateContext->UnmapTextureSubresource(copy.stagingTexture, 0, 0);
                }
            }

            mCompletedReadbacks.push_back(std::move(event));
        }

        mPendingReadbacks.clear();
        mPendingReadbackCopies.clear();
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

        Diligent::FenceDesc readbackFenceDesc{};
        readbackFenceDesc.Name = "CRESSimNeo.ReadbackFence";
        readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
        mRenderDevice->CreateFence(readbackFenceDesc, &mReadbackFence);
        if (mReadbackFence == nullptr)
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

    Diligent::IPipelineState* getDebugTrianglePso(bool hasDepthTarget)
    {
        auto& pso = hasDepthTarget ? mDebugTrianglePsoWithDepth : mDebugTrianglePsoNoDepth;
        if (pso != nullptr)
        {
            return pso;
        }

        if (!createDebugTrianglePso(hasDepthTarget, pso))
        {
            return nullptr;
        }

        return pso;
    }

    bool createDebugTrianglePso(bool hasDepthTarget, Diligent::RefCntAutoPtr<Diligent::IPipelineState>& outPso)
    {
        if (!mRenderDevice)
        {
            return false;
        }

        Diligent::ShaderCreateInfo shaderCreateInfo{};
        shaderCreateInfo.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
        shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
        shaderCreateInfo.EntryPoint = "main";

        Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
        shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
        shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.DebugTriangle.VS.Depth" : "CRESSimNeo.DebugTriangle.VS.NoDepth";
        shaderCreateInfo.Source = kDebugTriangleVsSource;
        mRenderDevice->CreateShader(shaderCreateInfo, &vertexShader);
        if (vertexShader == nullptr)
        {
            return false;
        }

        Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
        shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
        shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.DebugTriangle.PS.Depth" : "CRESSimNeo.DebugTriangle.PS.NoDepth";
        shaderCreateInfo.Source = kDebugTrianglePsSource;
        mRenderDevice->CreateShader(shaderCreateInfo, &pixelShader);
        if (pixelShader == nullptr)
        {
            return false;
        }

        Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
        psoCreateInfo.PSODesc.Name = hasDepthTarget ? "CRESSimNeo.DebugTrianglePSO.Depth" : "CRESSimNeo.DebugTrianglePSO.NoDepth";
        psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
        psoCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
        psoCreateInfo.GraphicsPipeline.RTVFormats[0] = Diligent::TEX_FORMAT_RGBA8_UNORM;
        psoCreateInfo.GraphicsPipeline.DSVFormat = hasDepthTarget ? Diligent::TEX_FORMAT_D32_FLOAT : Diligent::TEX_FORMAT_UNKNOWN;
        psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
        psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = hasDepthTarget ? Diligent::True : Diligent::False;
        psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = hasDepthTarget ? Diligent::True : Diligent::False;
        psoCreateInfo.pVS = vertexShader;
        psoCreateInfo.pPS = pixelShader;

        mRenderDevice->CreateGraphicsPipelineState(psoCreateInfo, &outPso);
        return outPso != nullptr;
    }

    bool queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex)
    {
        if (!mRenderDevice || !mImmediateContext || !mReadbackFence || mBackend != GraphicsBackend::Vulkan)
        {
            return false;
        }

        const auto targetIt = mRenderTargets.find(target.id);
        if (targetIt == mRenderTargets.end())
        {
            return false;
        }

        const RenderTargetResources& resources = targetIt->second;
        if (!resources.desc.cpuReadback || resources.colorTexture == nullptr)
        {
            return false;
        }

        Diligent::TextureDesc stagingDesc = resources.colorTexture->GetDesc();
        const std::string stagingName = resources.desc.debugName + ".Readback";
        stagingDesc.Name = stagingName.c_str();
        stagingDesc.BindFlags = Diligent::BIND_NONE;
        stagingDesc.Usage = Diligent::USAGE_STAGING;
        stagingDesc.CPUAccessFlags = Diligent::CPU_ACCESS_READ;
        stagingDesc.MiscFlags = Diligent::MISC_TEXTURE_FLAG_NONE;

        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
        mRenderDevice->CreateTexture(stagingDesc, nullptr, &stagingTexture);
        if (stagingTexture == nullptr)
        {
            return false;
        }

        Diligent::CopyTextureAttribs copyAttribs{
            resources.colorTexture,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            stagingTexture,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
        mImmediateContext->CopyTexture(copyAttribs);

        const std::uint64_t fenceValue = mNextReadbackFenceValue++;
        mImmediateContext->EnqueueSignal(mReadbackFence, fenceValue);

        PendingReadbackCopy readbackCopy{};
        readbackCopy.target = target;
        readbackCopy.frameIndex = frameIndex;
        readbackCopy.fenceValue = fenceValue;
        readbackCopy.width = resources.desc.width;
        readbackCopy.height = resources.desc.height;
        readbackCopy.stagingTexture = std::move(stagingTexture);
        mPendingReadbackCopies.push_back(std::move(readbackCopy));
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
    // GPU->CPU copy jobs collected during render target completion and consumed in endFrame().
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    // FIFO completion metadata consumed through tryPopReadbackEvent().
    std::deque<RenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mDebugTrianglePsoWithDepth;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mDebugTrianglePsoNoDepth;
    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    std::uint64_t mNextReadbackFenceValue = 1;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
};
} // namespace

std::unique_ptr<IGraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<DiligentGraphicsDevice>();
}

} // namespace cressim::neo::graphics
