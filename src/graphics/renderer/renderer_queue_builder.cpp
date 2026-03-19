#include "graphics/renderer/renderer_internal.h"

#include "common/math_utils_runtime.h"

#include <map>

namespace cressim::neo::graphics::detail
{

namespace
{

bool buildDrawCommand(std::uint32_t objectIndex, const RenderableInstance& renderable,
                      const MeshResourceDesc& mesh, const MaterialResourceDesc& material,
                      const RenderResourceManager& resources, ForwardDrawCommand& outCommand)
{
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        return false;
    }

    outCommand                      = {};
    outCommand.instanceIndex        = objectIndex;
    outCommand.programFamily        = material.pipeline.programFamily;
    outCommand.materialFeatureFlags = material.pipeline.featureFlags;
    outCommand.meshId               = renderable.mesh.id;
    outCommand.materialId           = renderable.material.id;
    outCommand.meshVersion          = resources.meshVersion(renderable.mesh);
    outCommand.vertexData           = mesh.vertices.data();
    outCommand.vertexCount          = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes    = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData            = mesh.indices.data();
    outCommand.indexCount           = static_cast<std::uint32_t>(mesh.indices.size());
    outCommand.material.baseColor   = material.baseColor;
    outCommand.material.metallic    = material.metallic;
    outCommand.material.roughness   = material.roughness;
    outCommand.material.opacity     = common::runtime_math::clamp01(material.opacity);
    outCommand.material.alphaCutoff = common::runtime_math::clamp01(material.pipeline.alphaCutoff);
    outCommand.material.receivesShadows = material.receivesShadows ? 1.0f : 0.0f;
    return true;
}

struct DrawBucketKey
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit;
    std::uint32_t materialFeatureFlags  = 0u;
    common::ResourceId materialId       = common::kInvalidResourceId;
    common::ResourceId meshId           = common::kInvalidResourceId;

    [[nodiscard]] bool operator<(const DrawBucketKey& rhs) const noexcept
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

void buildGpuBuckets(const std::vector<QueuedDraw>& sortedDraws,
                     std::vector<GpuIndirectBucket>& outBuckets,
                     std::vector<GpuIndirectCandidate>& outCandidates, bool shadowMode)
{
    outBuckets.clear();
    outCandidates.clear();
    outBuckets.reserve(sortedDraws.size());
    outCandidates.reserve(sortedDraws.size() * (shadowMode ? kShadowCascadeCount : 1u));

    std::uint32_t nextDrawListOffset = 0u;
    DrawBucketKey currentKey{};
    bool hasCurrentKey = false;
    for (const QueuedDraw& draw : sortedDraws)
    {
        if (draw.objectIndex == 0xffffffffu || draw.drawCommand.instanceIndex == 0xffffffffu)
        {
            continue;
        }

        const DrawBucketKey drawKey{
            draw.drawCommand.programFamily,
            static_cast<std::uint32_t>(draw.drawCommand.materialFeatureFlags), draw.materialId,
            draw.meshId};
        if (outBuckets.empty() || !hasCurrentKey || currentKey < drawKey || drawKey < currentKey)
        {
            GpuIndirectBucket bucket{};
            bucket.drawCommand                   = draw.drawCommand;
            bucket.drawCommand.useDrawListBuffer = 1u;
            bucket.candidateOffset = static_cast<std::uint32_t>(sortedDraws.size()); // placeholder
            bucket.candidateCount  = 0u;
            bucket.drawListOffset  = nextDrawListOffset;
            bucket.commandIndex    = static_cast<std::uint32_t>(outBuckets.size());
            outBuckets.push_back(bucket);
            currentKey    = drawKey;
            hasCurrentKey = true;
        }

        GpuIndirectBucket& bucket = outBuckets.back();
        if (bucket.candidateCount == 0u)
        {
            bucket.candidateOffset = static_cast<std::uint32_t>(outCandidates.size());
        }

        if (!shadowMode)
        {
            outCandidates.push_back(
                GpuIndirectCandidate{draw.objectIndex, bucket.commandIndex, 0u, 0u});
            ++bucket.candidateCount;
            ++nextDrawListOffset;
            continue;
        }

        for (std::uint32_t cascadeIndex = 0u; cascadeIndex < kShadowCascadeCount; ++cascadeIndex)
        {
            outCandidates.push_back(GpuIndirectCandidate{
                draw.objectIndex, bucket.commandIndex * kShadowCascadeCount + cascadeIndex,
                1u << cascadeIndex, 0u});
            ++nextDrawListOffset;
        }
        ++bucket.candidateCount;
    }

    if (shadowMode && !outBuckets.empty())
    {
        std::vector<GpuIndirectBucket> expandedBuckets;
        expandedBuckets.reserve(outBuckets.size() * kShadowCascadeCount);
        std::uint32_t nextShadowOffset = 0u;
        for (const GpuIndirectBucket& sourceBucket : outBuckets)
        {
            for (std::uint32_t cascadeIndex = 0u; cascadeIndex < kShadowCascadeCount;
                 ++cascadeIndex)
            {
                GpuIndirectBucket bucket = sourceBucket;
                bucket.commandIndex      = static_cast<std::uint32_t>(expandedBuckets.size());
                bucket.drawListOffset    = nextShadowOffset;
                nextShadowOffset += sourceBucket.candidateCount;
                expandedBuckets.push_back(bucket);
            }
        }
        outBuckets = std::move(expandedBuckets);
    }
}

} // namespace

CameraRenderQueues buildCameraRenderQueues(const std::vector<RenderableInstance>& renderables,
                                           const RenderResourceManager& resources,
                                           RenderStats& stats)
{
    CameraRenderQueues queues{};
    std::map<DrawBucketKey, std::vector<QueuedDraw>> opaqueBucketsByKey;
    std::map<DrawBucketKey, std::vector<QueuedDraw>> shadowBucketsByKey;

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(renderables.size()); ++objectIndex)
    {
        const RenderableInstance& renderable = renderables[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId ||
            renderable.objectSlot == 0xffffffffu || !renderable.visible)
        {
            continue;
        }

        const MeshResourceDesc* mesh         = resources.tryGetMesh(renderable.mesh);
        const MaterialResourceDesc* material = resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr)
        {
            continue;
        }
        if (material->blendMode == BlendMode::Transparent)
        {
            continue;
        }

        ForwardDrawCommand drawCommand{};
        if (!buildDrawCommand(objectIndex, renderable, *mesh, *material, resources, drawCommand))
        {
            continue;
        }
        ++stats.validRenderableCount;

        const QueuedDraw queuedDraw{objectIndex, renderable.mesh.id, renderable.material.id,
                                    material->castsShadows, drawCommand};
        const DrawBucketKey bucketKey{drawCommand.programFamily,
                                      static_cast<std::uint32_t>(drawCommand.materialFeatureFlags),
                                      queuedDraw.materialId, queuedDraw.meshId};
        opaqueBucketsByKey[bucketKey].push_back(queuedDraw);
        if (material->castsShadows)
        {
            shadowBucketsByKey[bucketKey].push_back(queuedDraw);
        }
    }

    std::vector<QueuedDraw> gpuOpaqueDraws;
    std::vector<QueuedDraw> gpuShadowDraws;
    for (auto& [key, draws] : opaqueBucketsByKey)
    {
        (void)key;
        gpuOpaqueDraws.insert(gpuOpaqueDraws.end(), draws.begin(), draws.end());
    }
    for (auto& [key, draws] : shadowBucketsByKey)
    {
        (void)key;
        gpuShadowDraws.insert(gpuShadowDraws.end(), draws.begin(), draws.end());
    }

    buildGpuBuckets(gpuOpaqueDraws, queues.gpuOpaqueBuckets, queues.gpuOpaqueCandidates, false);
    buildGpuBuckets(gpuShadowDraws, queues.gpuShadowBuckets, queues.gpuShadowCandidates, true);

    stats.opaqueQueueCount += static_cast<std::uint32_t>(gpuOpaqueDraws.size());
    stats.shadowCasterQueueCount += static_cast<std::uint32_t>(gpuShadowDraws.size());
    return queues;
}

} // namespace cressim::neo::graphics::detail
