#include "graphics/renderer/renderer_internal.h"

#include "common/math_utils_runtime.h"

#include <algorithm>

namespace cressim::neo::graphics::detail
{

namespace
{

float squaredDistanceToCamera(const common::Transform& transform,
                              const Diligent::float3& cameraWorldPosition)
{
    const float dx = transform.position.x - cameraWorldPosition.x;
    const float dy = transform.position.y - cameraWorldPosition.y;
    const float dz = transform.position.z - cameraWorldPosition.z;
    return dx * dx + dy * dy + dz * dz;
}

bool buildDrawCommand(const PreparedRenderable& renderable, const RenderResourceManager& resources,
                      ForwardDrawCommand& outCommand)
{
    if (renderable.instance == nullptr || renderable.mesh == nullptr ||
        renderable.material == nullptr)
    {
        return false;
    }

    const auto& mesh     = *renderable.mesh;
    const auto& material = *renderable.material;
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        return false;
    }

    outCommand                      = {};
    outCommand.programFamily        = material.pipeline.programFamily;
    outCommand.materialFeatureFlags = material.pipeline.featureFlags;
    outCommand.meshId               = renderable.instance->mesh.id;
    outCommand.materialId           = renderable.instance->material.id;
    outCommand.meshVersion          = resources.meshVersion(renderable.instance->mesh);
    outCommand.vertexData           = mesh.vertices.data();
    outCommand.vertexCount          = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes    = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData            = mesh.indices.data();
    outCommand.indexCount           = static_cast<std::uint32_t>(mesh.indices.size());
    outCommand.modelMatrix          = renderable.modelMatrix;
    outCommand.normalMatrix         = renderable.normalMatrix;
    outCommand.material.baseColor   = material.baseColor;
    outCommand.material.metallic    = material.metallic;
    outCommand.material.roughness   = material.roughness;
    outCommand.material.opacity     = common::runtime_math::clamp01(material.opacity);
    outCommand.material.alphaCutoff = common::runtime_math::clamp01(material.pipeline.alphaCutoff);
    outCommand.material.receivesShadows = material.receivesShadows ? 1.0f : 0.0f;
    return true;
}

} // namespace

std::vector<PreparedRenderable> buildPreparedRenderables(
    const std::vector<RenderableInstance>& renderables, const RenderResourceManager& resources)
{
    std::vector<PreparedRenderable> prepared;
    prepared.reserve(renderables.size());

    for (const RenderableInstance& renderable : renderables)
    {
        const MeshResourceDesc* mesh         = resources.tryGetMesh(renderable.mesh);
        const MaterialResourceDesc* material = resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr)
        {
            continue;
        }
        if (mesh->vertices.empty() || mesh->indices.size() < 3)
        {
            continue;
        }

        PreparedRenderable entry{};
        entry.instance     = &renderable;
        entry.mesh         = mesh;
        entry.material     = material;
        entry.modelMatrix  = worldMatrixFromTransform(renderable.worldTransform);
        entry.normalMatrix = normalMatrixFromModelMatrix(entry.modelMatrix);

        Diligent::float3 localBoundsMin{};
        Diligent::float3 localBoundsMax{};
        if (resources.tryGetMeshLocalBounds(renderable.mesh, localBoundsMin, localBoundsMax))
        {
            const Diligent::BoundBox localBounds{localBoundsMin, localBoundsMax};
            entry.worldBounds    = localBounds.Transform(entry.modelMatrix);
            entry.hasWorldBounds = true;
        }

        prepared.push_back(entry);
    }

    return prepared;
}

CameraRenderQueues buildCameraRenderQueues(
    const std::vector<PreparedRenderable>& preparedRenderables, const FrameViewData& frameView,
    const RenderResourceManager& resources, RenderStats& stats)
{
    CameraRenderQueues queues{};
    queues.opaque.reserve(preparedRenderables.size());
    queues.transparent.reserve(preparedRenderables.size());
    queues.shadowCasters.reserve(preparedRenderables.size());

    for (const PreparedRenderable& renderable : preparedRenderables)
    {
        if (renderable.instance == nullptr || renderable.material == nullptr)
        {
            continue;
        }

        const bool cameraVisible  = isVisibleByFrustum(renderable, frameView.viewFrustum);
        const bool transparent    = (renderable.material->blendMode == BlendMode::Transparent);
        const bool canCastShadows = renderable.material->castsShadows && !transparent;
        std::uint32_t shadowCascadeMask = 0;
        if (frameView.hasDirectionalLight)
        {
            for (std::uint32_t cascadeIdx = 0; cascadeIdx < frameView.shadowCascadeCount;
                 ++cascadeIdx)
            {
                if (isVisibleByFrustum(renderable, frameView.lightFrustums[cascadeIdx]))
                {
                    shadowCascadeMask |= (1u << cascadeIdx);
                }
            }
        }
        const bool lightVisible = shadowCascadeMask != 0;

        if (!cameraVisible)
        {
            ++stats.culledRenderableCount;
        }

        const bool needsMainPass   = cameraVisible;
        const bool needsShadowPass = canCastShadows && lightVisible;
        if (!needsMainPass && !needsShadowPass)
        {
            continue;
        }

        ForwardDrawCommand drawCommand{};
        if (!buildDrawCommand(renderable, resources, drawCommand))
        {
            continue;
        }

        QueuedDraw queuedDraw{};
        queuedDraw.entityId        = renderable.instance->entityId;
        queuedDraw.meshId          = renderable.instance->mesh.id;
        queuedDraw.materialId      = renderable.instance->material.id;
        queuedDraw.depth           = squaredDistanceToCamera(renderable.instance->worldTransform,
                                                             frameView.cameraWorldPosition);
        queuedDraw.castsShadows    = renderable.material->castsShadows;
        queuedDraw.receivesShadows = renderable.material->receivesShadows;
        queuedDraw.transparent     = transparent;
        queuedDraw.mainPassClass =
            transparent ? MainPassClass::ForwardTransparent : MainPassClass::ForwardOpaque;
        queuedDraw.shadowCascadeMask = shadowCascadeMask;
        queuedDraw.drawCommand       = drawCommand;

        if (needsMainPass && queuedDraw.transparent)
        {
            queues.transparent.push_back(queuedDraw);
        }
        else if (needsMainPass)
        {
            queues.opaque.push_back(queuedDraw);
        }

        if (needsShadowPass)
        {
            queues.shadowCasters.push_back(queuedDraw);
        }
    }

    std::sort(queues.opaque.begin(), queues.opaque.end(),
              [](const QueuedDraw& lhs, const QueuedDraw& rhs)
              {
                  if (lhs.drawCommand.programFamily != rhs.drawCommand.programFamily)
                  {
                      return static_cast<std::uint32_t>(lhs.drawCommand.programFamily) <
                             static_cast<std::uint32_t>(rhs.drawCommand.programFamily);
                  }
                  if (lhs.drawCommand.materialFeatureFlags != rhs.drawCommand.materialFeatureFlags)
                  {
                      return lhs.drawCommand.materialFeatureFlags <
                             rhs.drawCommand.materialFeatureFlags;
                  }
                  if (lhs.materialId != rhs.materialId)
                  {
                      return lhs.materialId < rhs.materialId;
                  }
                  if (lhs.meshId != rhs.meshId)
                  {
                      return lhs.meshId < rhs.meshId;
                  }
                  if (lhs.depth != rhs.depth)
                  {
                      return lhs.depth < rhs.depth;
                  }
                  return lhs.entityId < rhs.entityId;
              });

    std::sort(queues.transparent.begin(), queues.transparent.end(),
              [](const QueuedDraw& lhs, const QueuedDraw& rhs)
              {
                  if (lhs.depth != rhs.depth)
                  {
                      return lhs.depth > rhs.depth;
                  }
                  if (lhs.materialId != rhs.materialId)
                  {
                      return lhs.materialId < rhs.materialId;
                  }
                  if (lhs.meshId != rhs.meshId)
                  {
                      return lhs.meshId < rhs.meshId;
                  }
                  return lhs.entityId < rhs.entityId;
              });

    std::sort(queues.shadowCasters.begin(), queues.shadowCasters.end(),
              [](const QueuedDraw& lhs, const QueuedDraw& rhs)
              {
                  if (lhs.materialId != rhs.materialId)
                  {
                      return lhs.materialId < rhs.materialId;
                  }
                  if (lhs.meshId != rhs.meshId)
                  {
                      return lhs.meshId < rhs.meshId;
                  }
                  return lhs.entityId < rhs.entityId;
              });

    stats.opaqueQueueCount += static_cast<std::uint32_t>(queues.opaque.size());
    stats.transparentQueueCount += static_cast<std::uint32_t>(queues.transparent.size());
    stats.shadowCasterQueueCount += static_cast<std::uint32_t>(queues.shadowCasters.size());
    return queues;
}

} // namespace cressim::neo::graphics::detail
