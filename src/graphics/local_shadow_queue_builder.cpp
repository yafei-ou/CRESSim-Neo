#include "graphics/local_shadow_queue_builder.h"

#include "graphics/render_resource_manager.h"
#include "graphics/renderer_internal.h"

#include <cmath>
#include <map>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

namespace
{

constexpr std::uint32_t kLocalShadowMapResolution = 1024u;
constexpr std::uint32_t kPointShadowMapResolution = 512u;

struct DrawBucketKey
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit;
    std::uint32_t materialFeatureFlags  = 0u;
    common::ResourceId materialId       = common::kInvalidResourceId;
    common::ResourceId meshId           = common::kInvalidResourceId;

    [[nodiscard]] bool operator<(const DrawBucketKey &rhs) const noexcept
    {
        if (programFamily != rhs.programFamily)
        {
            return static_cast<std::uint32_t>(programFamily) <
                   static_cast<std::uint32_t>(rhs.programFamily);
        }
        if (materialFeatureFlags != rhs.materialFeatureFlags)
        {
            return materialFeatureFlags < rhs.materialFeatureFlags;
        }
        if (materialId != rhs.materialId)
        {
            return materialId < rhs.materialId;
        }
        return meshId < rhs.meshId;
    }
};

struct EnvShadowBounds
{
    Diligent::float3 center{0.0f, 0.0f, 0.0f};
    float radius = 10.0f;
    bool valid   = false;
};

float dot3(const Diligent::float3 &lhs, const Diligent::float3 &rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

std::vector<EnvShadowBounds> buildEnvShadowBounds(const HostSceneView &sceneView)
{
    std::uint32_t envCount = 1u;
    if (sceneView.gpuEntityScene != nullptr)
    {
        envCount = sceneView.gpuEntityScene->layout.envCount;
    }
    std::vector<EnvShadowBounds> bounds(envCount);
    if (sceneView.renderables == nullptr)
    {
        return bounds;
    }

    std::vector<Diligent::float3> minPoints(envCount, Diligent::float3{1e9f, 1e9f, 1e9f});
    std::vector<Diligent::float3> maxPoints(envCount, Diligent::float3{-1e9f, -1e9f, -1e9f});
    for (const RenderableInstance &renderable : *sceneView.renderables)
    {
        if (renderable.entityId == common::kInvalidEntityId || !renderable.visible ||
            renderable.envIndex >= envCount)
        {
            continue;
        }
        const Diligent::float3 p          = renderable.worldTransform.position;
        minPoints[renderable.envIndex].x  = std::min(minPoints[renderable.envIndex].x, p.x);
        minPoints[renderable.envIndex].y  = std::min(minPoints[renderable.envIndex].y, p.y);
        minPoints[renderable.envIndex].z  = std::min(minPoints[renderable.envIndex].z, p.z);
        maxPoints[renderable.envIndex].x  = std::max(maxPoints[renderable.envIndex].x, p.x);
        maxPoints[renderable.envIndex].y  = std::max(maxPoints[renderable.envIndex].y, p.y);
        maxPoints[renderable.envIndex].z  = std::max(maxPoints[renderable.envIndex].z, p.z);
        bounds[renderable.envIndex].valid = true;
    }

    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        if (!bounds[envIndex].valid)
        {
            continue;
        }
        bounds[envIndex].center        = (minPoints[envIndex] + maxPoints[envIndex]) * 0.5f;
        const Diligent::float3 extents = maxPoints[envIndex] - minPoints[envIndex];
        bounds[envIndex].radius = std::max(5.0f, std::sqrt(dot3(extents, extents)) * 0.5f + 2.0f);
    }
    return bounds;
}

bool lightAffectsObject(const LightData &light, const RenderableInstance &renderable)
{
    if (light.type == gpu::GpuLightType::Directional)
    {
        return true;
    }

    const Diligent::float3 toObject = renderable.worldTransform.position - light.position;
    const float distanceSq          = dot3(toObject, toObject);
    const float range               = std::max(light.range, 1.0f);
    if (distanceSq > range * range)
    {
        return false;
    }

    if (light.type == gpu::GpuLightType::Spot)
    {
        const Diligent::float3 dir =
            normalizeOrFallback(light.direction, Diligent::float3{0.0f, -1.0f, 0.0f});
        const Diligent::float3 toObjDir =
            normalizeOrFallback(toObject, Diligent::float3{0.0f, 0.0f, 1.0f});
        const float outerCos = std::cos(light.outerConeAngle * Diligent::PI_F / 180.0f);
        return dot3(dir, toObjDir) >= outerCos;
    }

    return true;
}

} // namespace

LocalShadowBuildResult buildLocalShadowData(const HostSceneView &sceneView,
                                            RenderResourceManager &resourceManager)
{
    LocalShadowBuildResult result{};
    const std::size_t lightCount = sceneView.lights != nullptr ? sceneView.lights->size() : 0u;
    result.lightAssignments.assign(lightCount, gpu::GpuLightShadowAssignment{});
    if (sceneView.lights == nullptr || sceneView.renderables == nullptr)
    {
        return result;
    }

    const std::vector<EnvShadowBounds> envBounds = buildEnvShadowBounds(sceneView);
    std::unordered_map<std::uint32_t, std::uint32_t> shadowedLocalCounts;
    std::unordered_map<std::uint32_t, std::uint32_t> shadowedPointCounts;
    std::unordered_map<std::uint32_t, std::map<DrawBucketKey, std::vector<std::uint32_t>>>
        envBuckets;

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(sceneView.renderables->size()); ++objectIndex)
    {
        const RenderableInstance &renderable = (*sceneView.renderables)[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId || !renderable.visible)
        {
            continue;
        }

        const MeshResourceDesc *mesh         = resourceManager.tryGetMesh(renderable.mesh);
        const MaterialResourceDesc *material = resourceManager.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr || !material->castsShadows ||
            material->blendMode == BlendMode::Transparent || mesh->indices.size() < 3)
        {
            continue;
        }

        const DrawBucketKey key{material->pipeline.programFamily,
                                static_cast<std::uint32_t>(material->pipeline.featureFlags),
                                renderable.material.id, renderable.mesh.id};
        envBuckets[renderable.envIndex][key].push_back(objectIndex);
    }

    for (std::uint32_t lightIndex = 0u;
         lightIndex < static_cast<std::uint32_t>(sceneView.lights->size()); ++lightIndex)
    {
        const LightData &light = (*sceneView.lights)[lightIndex];
        if (light.entityId == common::kInvalidEntityId || light.lightSlot == 0xffffffffu ||
            light.lightSlot == gpu::kMainDirectionalLightSlot || !light.castsShadows ||
            light.intensity <= 0.0f)
        {
            continue;
        }

        if (light.type == gpu::GpuLightType::Point)
        {
            if (shadowedPointCounts[light.envIndex] >= gpu::kShadowedPointLightCap)
            {
                continue;
            }
        }
        else if (shadowedLocalCounts[light.envIndex] >= gpu::kShadowedLocalLightCap)
        {
            continue;
        }

        gpu::GpuLocalShadowView shadowView{};
        shadowView.lightIndex = lightIndex;
        shadowView.envIndex   = light.envIndex;
        shadowView.active     = 1u;
        shadowView.lightType  = static_cast<std::uint32_t>(light.type);
        shadowView.lightPositionRange =
            Diligent::float4{light.position.x, light.position.y, light.position.z, light.range};
        const Diligent::float3 lightDir =
            normalizeOrFallback(light.direction, Diligent::float3{0.0f, -1.0f, 0.0f});
        shadowView.lightDirection = Diligent::float4{lightDir.x, lightDir.y, lightDir.z, 0.0f};

        const EnvShadowBounds envBound =
            light.envIndex < envBounds.size() ? envBounds[light.envIndex] : EnvShadowBounds{};
        if (light.type == gpu::GpuLightType::Point)
        {
            shadowView.firstLayer = result.pointLayerCount;
            shadowView.layerCount = 6u;
            shadowView.shadowParams =
                Diligent::float4{1.0f / kPointShadowMapResolution, 1.0f / kPointShadowMapResolution,
                                 0.05f, std::max(light.range, 1.0f)};
            const Diligent::float3 pos        = light.position;
            const Diligent::float3 targets[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                                 {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
            const Diligent::float3 ups[6]     = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1},
                                                 {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
            for (std::uint32_t faceIndex = 0u; faceIndex < 6u; ++faceIndex)
            {
                const Diligent::float4x4 view =
                    buildLookAtMatrix(pos, pos + targets[faceIndex], ups[faceIndex]);
                const Diligent::float4x4 proj = Diligent::float4x4::Projection(
                    Diligent::PI_F * 0.5f, 1.0f, 0.05f, std::max(light.range, 1.0f), false);
                shadowView.lightViewProjectionMatrices[faceIndex] = view * proj;
            }
            result.pointLayerCount += 6u;
            ++shadowedPointCounts[light.envIndex];
            result.lightAssignments[lightIndex].shadowMode =
                static_cast<std::uint32_t>(gpu::GpuLightShadowMode::Point);
        }
        else if (light.type == gpu::GpuLightType::Spot)
        {
            shadowView.firstLayer = result.local2DLayerCount;
            shadowView.layerCount = 1u;
            shadowView.shadowParams =
                Diligent::float4{1.0f / kLocalShadowMapResolution, 1.0f / kLocalShadowMapResolution,
                                 0.05f, std::max(light.range, 1.0f)};
            const Diligent::float4x4 view = buildLookAtMatrix(
                light.position, light.position + lightDir, Diligent::float3{0, 1, 0});
            const float fovRadians = std::max(light.outerConeAngle * 2.0f * Diligent::PI_F / 180.0f,
                                              Diligent::PI_F / 180.0f);
            const Diligent::float4x4 proj = Diligent::float4x4::Projection(
                fovRadians, 1.0f, 0.05f, std::max(light.range, 1.0f), false);
            shadowView.lightViewProjectionMatrices[0] = view * proj;
            ++result.local2DLayerCount;
            ++shadowedLocalCounts[light.envIndex];
            result.lightAssignments[lightIndex].shadowMode =
                static_cast<std::uint32_t>(gpu::GpuLightShadowMode::Local2D);
        }
        else
        {
            const float radius = envBound.valid ? envBound.radius : std::max(light.range, 10.0f);
            const Diligent::float3 center =
                envBound.valid ? envBound.center : Diligent::float3{0, 0, 0};
            const Diligent::float3 eye = center - lightDir * (radius * 2.0f + 4.0f);
            shadowView.firstLayer      = result.local2DLayerCount;
            shadowView.layerCount      = 1u;
            shadowView.shadowParams =
                Diligent::float4{1.0f / kLocalShadowMapResolution, 1.0f / kLocalShadowMapResolution,
                                 0.1f, radius * 4.0f + 8.0f};
            const Diligent::float4x4 view =
                buildLookAtMatrix(eye, center,
                                  std::abs(lightDir.y) > 0.98f ? Diligent::float3{0, 0, 1}
                                                               : Diligent::float3{0, 1, 0});
            const Diligent::float4x4 proj = Diligent::float4x4::OrthoOffCenter(
                -radius, radius, -radius, radius, 0.1f, radius * 4.0f + 8.0f, false);
            shadowView.lightViewProjectionMatrices[0] = view * proj;
            ++result.local2DLayerCount;
            ++shadowedLocalCounts[light.envIndex];
            result.lightAssignments[lightIndex].shadowMode =
                static_cast<std::uint32_t>(gpu::GpuLightShadowMode::Local2D);
        }

        result.lightAssignments[lightIndex].shadowViewIndex =
            static_cast<std::uint32_t>(result.shadowViews.size());
        const std::uint32_t shadowViewIndex = static_cast<std::uint32_t>(result.shadowViews.size());
        result.shadowViews.push_back(shadowView);
        result.shadowViewCommandOffsets.push_back(
            static_cast<std::uint32_t>(result.commands.size()));
        result.shadowViewCommandCounts.push_back(0u);

        const auto envBucketsIt = envBuckets.find(light.envIndex);
        if (envBucketsIt == envBuckets.end())
        {
            continue;
        }

        for (const auto &[key, objectIndices] : envBucketsIt->second)
        {
            const MeshResourceDesc *mesh = resourceManager.tryGetMesh(MeshHandle{key.meshId});
            if (mesh == nullptr)
            {
                continue;
            }

            const std::uint32_t drawListOffset =
                static_cast<std::uint32_t>(result.visiblePairs.size());
            std::uint32_t instanceCount = 0u;
            for (const std::uint32_t objectIndex : objectIndices)
            {
                const RenderableInstance &renderable = (*sceneView.renderables)[objectIndex];
                if (!lightAffectsObject(light, renderable))
                {
                    continue;
                }
                gpu::GpuVisiblePairInstance pair{};
                pair.objectIndex      = objectIndex;
                pair.batchCameraIndex = shadowViewIndex;
                pair.bucketIndex      = static_cast<std::uint32_t>(result.commands.size());
                result.visiblePairs.push_back(pair);
                ++instanceCount;
            }

            if (instanceCount == 0u)
            {
                continue;
            }

            LocalShadowCommand command{};
            command.shadowViewIndex               = shadowViewIndex;
            command.drawListOffset                = drawListOffset;
            command.instanceCount                 = instanceCount;
            command.drawCommand.useDrawListBuffer = 1u;
            command.drawCommand.programFamily     = key.programFamily;
            command.drawCommand.materialFeatureFlags =
                static_cast<MaterialFeatureFlags>(key.materialFeatureFlags);
            command.drawCommand.meshId      = key.meshId;
            command.drawCommand.materialId  = key.materialId;
            command.drawCommand.meshVersion = resourceManager.meshVersion(MeshHandle{key.meshId});
            command.drawCommand.indexCount  = static_cast<std::uint32_t>(mesh->indices.size());
            result.commands.push_back(command);
            ++result.shadowViewCommandCounts[shadowViewIndex];
        }
    }

    return result;
}

} // namespace cressim::neo::graphics::detail
