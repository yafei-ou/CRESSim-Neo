#include "graphics/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{

struct Matrix4f
{
    float m[16] = {};
};

float clamp01(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

float clampPositive(float value, float fallback)
{
    return value > 0.0f ? value : fallback;
}

common::Vec3f subtract(const common::Vec3f& lhs, const common::Vec3f& rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

float dot(const common::Vec3f& lhs, const common::Vec3f& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

common::Vec3f cross(const common::Vec3f& lhs, const common::Vec3f& rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x};
}

common::Vec3f normalize(const common::Vec3f& value)
{
    const float lengthSq = dot(value, value);
    if (lengthSq <= 1.0e-12f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float invLength = 1.0f / std::sqrt(lengthSq);
    return {value.x * invLength, value.y * invLength, value.z * invLength};
}

common::Quatf normalizeQuaternion(const common::Quatf& value)
{
    const float lengthSq = value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
    if (lengthSq <= 1.0e-12f)
    {
        return {};
    }

    const float invLength = 1.0f / std::sqrt(lengthSq);
    return {value.x * invLength, value.y * invLength, value.z * invLength, value.w * invLength};
}

common::Vec3f rotateVector(const common::Quatf& rotation, const common::Vec3f& value)
{
    const common::Quatf q = normalizeQuaternion(rotation);
    const common::Vec3f u{q.x, q.y, q.z};
    const float s = q.w;

    const common::Vec3f uxv = cross(u, value);
    const common::Vec3f uxu = cross(u, uxv);

    return {
        value.x + 2.0f * (s * uxv.x + uxu.x),
        value.y + 2.0f * (s * uxv.y + uxu.y),
        value.z + 2.0f * (s * uxv.z + uxu.z)};
}

Matrix4f identityMatrix()
{
    Matrix4f out{};
    out.m[0] = 1.0f;
    out.m[5] = 1.0f;
    out.m[10] = 1.0f;
    out.m[15] = 1.0f;
    return out;
}

Matrix4f multiply(const Matrix4f& lhs, const Matrix4f& rhs)
{
    Matrix4f out{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            out.m[row * 4 + col] =
                lhs.m[row * 4 + 0] * rhs.m[0 * 4 + col] +
                lhs.m[row * 4 + 1] * rhs.m[1 * 4 + col] +
                lhs.m[row * 4 + 2] * rhs.m[2 * 4 + col] +
                lhs.m[row * 4 + 3] * rhs.m[3 * 4 + col];
        }
    }
    return out;
}

Matrix4f transformMatrix(const common::Transform& transform)
{
    const common::Vec3f right = rotateVector(transform.rotation, common::Vec3f{1.0f, 0.0f, 0.0f});
    const common::Vec3f up = rotateVector(transform.rotation, common::Vec3f{0.0f, 1.0f, 0.0f});
    const common::Vec3f forward = rotateVector(transform.rotation, common::Vec3f{0.0f, 0.0f, 1.0f});

    Matrix4f out = identityMatrix();
    out.m[0] = right.x * transform.scale.x;
    out.m[1] = up.x * transform.scale.y;
    out.m[2] = forward.x * transform.scale.z;
    out.m[4] = right.y * transform.scale.x;
    out.m[5] = up.y * transform.scale.y;
    out.m[6] = forward.y * transform.scale.z;
    out.m[8] = right.z * transform.scale.x;
    out.m[9] = up.z * transform.scale.y;
    out.m[10] = forward.z * transform.scale.z;
    out.m[12] = transform.position.x;
    out.m[13] = transform.position.y;
    out.m[14] = transform.position.z;
    return out;
}

Matrix4f viewMatrixFromTransform(const common::Transform& cameraTransform)
{
    const common::Vec3f eye = cameraTransform.position;
    const common::Vec3f forward = normalize(rotateVector(cameraTransform.rotation, common::Vec3f{0.0f, 0.0f, -1.0f}));
    const common::Vec3f up = normalize(rotateVector(cameraTransform.rotation, common::Vec3f{0.0f, 1.0f, 0.0f}));

    const common::Vec3f at{
        eye.x + forward.x,
        eye.y + forward.y,
        eye.z + forward.z};

    const common::Vec3f zAxis = normalize(subtract(eye, at));
    const common::Vec3f xAxis = normalize(cross(up, zAxis));
    const common::Vec3f yAxis = cross(zAxis, xAxis);

    Matrix4f out = identityMatrix();
    out.m[0] = xAxis.x;
    out.m[1] = yAxis.x;
    out.m[2] = zAxis.x;
    out.m[4] = xAxis.y;
    out.m[5] = yAxis.y;
    out.m[6] = zAxis.y;
    out.m[8] = xAxis.z;
    out.m[9] = yAxis.z;
    out.m[10] = zAxis.z;
    out.m[12] = -dot(xAxis, eye);
    out.m[13] = -dot(yAxis, eye);
    out.m[14] = -dot(zAxis, eye);
    return out;
}

Matrix4f perspectiveMatrix(float verticalFovDegrees, float aspectRatio, float nearClip, float farClip)
{
    const float fovRadians = verticalFovDegrees * 0.017453292519943295769f;
    const float tanHalfFov = std::tan(0.5f * fovRadians);
    const float yScale = 1.0f / clampPositive(tanHalfFov, 1.0f);
    const float xScale = yScale / clampPositive(aspectRatio, 1.0f);
    const float nearPlane = std::max(nearClip, 0.001f);
    const float farPlane = std::max(farClip, nearPlane + 0.001f);
    const float clipRange = nearPlane - farPlane;

    Matrix4f out{};
    out.m[0] = xScale;
    out.m[5] = yScale;
    out.m[10] = farPlane / clipRange;
    out.m[11] = -1.0f;
    out.m[14] = (nearPlane * farPlane) / clipRange;
    return out;
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

PbrDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights)
{
    PbrDirectionalLightData out{};
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

void copyMatrix(float* dst, const Matrix4f& matrix)
{
    std::memcpy(dst, matrix.m, sizeof(matrix.m));
}

bool buildDrawCommand(
    const ValidRenderable& renderable,
    const Matrix4f& viewProjectionMatrix,
    const common::Vec3f& cameraPosition,
    const PbrDirectionalLightData& light,
    const RenderResourceManager& resources,
    PbrDrawCommand& outCommand)
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

    const Matrix4f modelMatrix = transformMatrix(renderable.instance->worldTransform);

    outCommand = {};
    outCommand.meshId = renderable.instance->mesh.id;
    outCommand.meshVersion = resources.meshVersion(renderable.instance->mesh);
    outCommand.vertexData = mesh.vertices.data();
    outCommand.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData = mesh.indices.data();
    outCommand.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
    copyMatrix(outCommand.modelMatrix, modelMatrix);
    copyMatrix(outCommand.viewProjectionMatrix, viewProjectionMatrix);
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

Matrix4f buildViewProjection(const CameraData& camera)
{
    const float outputWidth = camera.outputWidth > 0 ? static_cast<float>(camera.outputWidth) : 1280.0f;
    const float outputHeight = camera.outputHeight > 0 ? static_cast<float>(camera.outputHeight) : 720.0f;
    const float aspect = outputWidth / clampPositive(outputHeight, 1.0f);

    const Matrix4f view = viewMatrixFromTransform(camera.worldTransform);
    const Matrix4f projection = perspectiveMatrix(camera.verticalFovDegrees, aspect, camera.nearClip, camera.farClip);
    return multiply(view, projection);
}

common::Vec3f cameraPosition(const CameraData& camera)
{
    return camera.worldTransform.position;
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

Renderer::Renderer(IGraphicsDevice& device, RenderResourceManager& resourceManager) :
    mDevice(device),
    mResourceManager(resourceManager)
{
}

bool Renderer::initialize()
{
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
    const PbrDirectionalLightData lightData = buildMainLight(world.directionalLights());

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

        const Matrix4f viewProjectionMatrix = buildViewProjection(camera);
        const common::Vec3f cameraWorldPosition = cameraPosition(camera);

        mDevice.beginRenderTarget(target, frameContext);
        for (const ValidRenderable& renderable : validRenderables)
        {
            PbrDrawCommand drawCommand{};
            if (!buildDrawCommand(renderable, viewProjectionMatrix, cameraWorldPosition, lightData, mResourceManager, drawCommand))
            {
                continue;
            }

            if (mDevice.drawPbr(target, drawCommand))
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

    return stats;
}

} // namespace cressim::neo::graphics
