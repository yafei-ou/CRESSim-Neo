#include "graphics/renderer.h"
#include "graphics/device/graphics_device_impl.h"
#include "graphics/math/diligent_math_utils.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/services/debug_view_presenter.h"

#include <algorithm>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{

float clamp01(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

float clampPositive(float value, float fallback)
{
    return value > 0.0f ? value : fallback;
}

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    RenderViewport normalized{};
    normalized.x = clamp01(viewport.x);
    normalized.y = clamp01(viewport.y);
    normalized.width = clamp01(viewport.width);
    normalized.height = clamp01(viewport.height);

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

struct ValidRenderable
{
    const RenderableInstance* instance = nullptr;
    const MeshResourceDesc* mesh = nullptr;
    const MaterialResourceDesc* material = nullptr;
};

std::vector<ValidRenderable> gatherValidRenderables(const std::vector<RenderableInstance>& renderables, const RenderResourceManager& resources)
{
    std::vector<ValidRenderable> valid;
    valid.reserve(renderables.size());

    for (const RenderableInstance& renderable : renderables)
    {
        const MeshResourceDesc* mesh = resources.tryGetMesh(renderable.mesh);
        const MaterialResourceDesc* material = resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr)
        {
            continue;
        }
        if (mesh->vertices.empty() || mesh->indices.size() < 3)
        {
            continue;
        }

        ValidRenderable validRenderable{};
        validRenderable.instance = &renderable;
        validRenderable.mesh = mesh;
        validRenderable.material = material;
        valid.push_back(validRenderable);
    }

    return valid;
}

ForwardDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights)
{
    ForwardDirectionalLightData out{};
    if (lights.empty())
    {
        return out;
    }

    const DirectionalLightData& light = lights.front();
    out.direction[0] = light.direction.x;
    out.direction[1] = light.direction.y;
    out.direction[2] = light.direction.z;
    out.color[0] = light.color.x;
    out.color[1] = light.color.y;
    out.color[2] = light.color.z;
    out.intensity = light.intensity;
    return out;
}

bool buildDrawCommand(
    const ValidRenderable& renderable,
    const Diligent::float4x4& viewProjectionMatrix,
    const Diligent::float3& cameraPosition,
    const ForwardDirectionalLightData& light,
    const RenderResourceManager& resources,
    ForwardDrawCommand& outCommand)
{
    if (renderable.instance == nullptr || renderable.mesh == nullptr || renderable.material == nullptr)
    {
        return false;
    }

    const auto& mesh = *renderable.mesh;
    const auto& material = *renderable.material;
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        return false;
    }

    const Diligent::float4x4 modelMatrix = math::transformMatrix(renderable.instance->worldTransform);

    outCommand = {};
    outCommand.shadingModel = ForwardShadingModel::Pbr;
    outCommand.meshId = renderable.instance->mesh.id;
    outCommand.meshVersion = resources.meshVersion(renderable.instance->mesh);
    outCommand.vertexData = mesh.vertices.data();
    outCommand.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData = mesh.indices.data();
    outCommand.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
    math::copyMatrixRowMajor(outCommand.modelMatrix, modelMatrix);
    math::copyMatrixRowMajor(outCommand.viewProjectionMatrix, viewProjectionMatrix);
    outCommand.cameraPosition[0] = cameraPosition.x;
    outCommand.cameraPosition[1] = cameraPosition.y;
    outCommand.cameraPosition[2] = cameraPosition.z;
    outCommand.material.baseColor[0] = material.baseColor.x;
    outCommand.material.baseColor[1] = material.baseColor.y;
    outCommand.material.baseColor[2] = material.baseColor.z;
    outCommand.material.metallic = material.metallic;
    outCommand.material.roughness = material.roughness;
    outCommand.light = light;
    return true;
}

Diligent::float4x4 buildViewProjection(const CameraData& camera)
{
    const float outputWidth = camera.outputWidth > 0 ? static_cast<float>(camera.outputWidth) : 1280.0f;
    const float outputHeight = camera.outputHeight > 0 ? static_cast<float>(camera.outputHeight) : 720.0f;
    const float aspect = outputWidth / clampPositive(outputHeight, 1.0f);

    const Diligent::float4x4 view = math::viewMatrixFromTransform(camera.worldTransform);
    const Diligent::float4x4 projection = math::perspectiveMatrix(camera.verticalFovDegrees, aspect, camera.nearClip, camera.farClip);
    return view * projection;
}

Diligent::float3 cameraPosition(const CameraData& camera)
{
    return math::toFloat3(camera.worldTransform.position);
}

CameraData defaultCamera()
{
    CameraData camera{};
    camera.verticalFovDegrees = 60.0f;
    camera.nearClip = 0.01f;
    camera.farClip = 1000.0f;
    camera.viewport = {};
    return camera;
}

std::vector<CameraData> sortedCameras(const RenderWorld& world)
{
    std::vector<CameraData> cameras = world.cameras();
    std::sort(cameras.begin(), cameras.end(), [](const CameraData& lhs, const CameraData& rhs) {
        if (lhs.renderOrder != rhs.renderOrder)
        {
            return lhs.renderOrder < rhs.renderOrder;
        }
        return lhs.entityId < rhs.entityId;
    });
    return cameras;
}

} // namespace

Renderer::Renderer(GraphicsDevice& device, RenderResourceManager& resourceManager, const RendererDesc& desc) :
    mDevice(device),
    mResourceManager(resourceManager),
    mDesc(desc)
{
}

Renderer::~Renderer() = default;

bool Renderer::initialize()
{
    GraphicsDeviceImpl& deviceImpl = static_cast<GraphicsDeviceImpl&>(mDevice);
    mForwardPipeline = std::make_unique<detail::ForwardPipeline>(deviceImpl);
    if (!mForwardPipeline->initialize())
    {
        mForwardPipeline.reset();
        return false;
    }

    if (mDesc.debugViewer.enabled)
    {
        mDebugViewPresenter = std::make_unique<detail::DebugViewPresenter>(deviceImpl, mDesc.debugViewer);
        if (!mDebugViewPresenter->initialize())
        {
            mDebugViewPresenter.reset();
            mForwardPipeline.reset();
            return false;
        }
    }

    mInitialized = true;
    return mInitialized;
}

RenderStats Renderer::render(const common::FrameContext& frameContext, const RenderWorld& world)
{
    RenderStats stats{};

    if (!mInitialized)
    {
        return stats;
    }

    mDevice.beginFrame(frameContext);

    const auto& renderables = world.renderables();
    const auto validRenderables = gatherValidRenderables(renderables, mResourceManager);
    const ForwardDirectionalLightData lightData = buildMainLight(world.directionalLights());

    stats.renderableCount = static_cast<std::uint32_t>(renderables.size());
    stats.validRenderableCount = static_cast<std::uint32_t>(validRenderables.size());
    stats.lightCount = static_cast<std::uint32_t>(world.directionalLights().size());

    std::vector<CameraData> cameras = sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(defaultCamera());
    }

    const auto renderCamera = [&](const CameraData& camera) {
        RenderTargetHandle target = camera.outputTarget;
        if (!mDevice.isValidRenderTarget(target))
        {
            target = mDevice.defaultRenderTarget();
        }
        if (!mDevice.isValidRenderTarget(target))
        {
            return;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            (void)mDevice.resizeRenderTarget(target, camera.outputWidth, camera.outputHeight);
        }

        mDevice.setRenderTargetViewport(target, normalizeViewport(camera.viewport));

        if (camera.requestReadback)
        {
            mDevice.requestReadback(target);
            ++stats.readbackRequests;
        }

        const Diligent::float4x4 viewProjectionMatrix = buildViewProjection(camera);
        const Diligent::float3 cameraWorldPosition = cameraPosition(camera);

        mDevice.beginRenderTarget(target, frameContext);
        for (const ValidRenderable& renderable : validRenderables)
        {
            ForwardDrawCommand drawCommand{};
            if (!buildDrawCommand(renderable, viewProjectionMatrix, cameraWorldPosition, lightData, mResourceManager, drawCommand))
            {
                continue;
            }

            if (mForwardPipeline != nullptr && mForwardPipeline->draw(target, drawCommand))
            {
                ++stats.drawCalls;
            }
        }
        mDevice.endRenderTarget(target, frameContext);

        ++stats.cameraCount;
    };

    for (const CameraData& camera : cameras)
    {
        renderCamera(camera);
    }

    mDevice.endFrame(frameContext);
    if (mDebugViewPresenter != nullptr)
    {
        (void)mDebugViewPresenter->present(mDevice.defaultRenderTarget());
    }

    return stats;
}

} // namespace cressim::neo::graphics
