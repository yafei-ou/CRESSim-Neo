#include "common/logger.h"
#include "graphics/flags.h"
#include "graphics/render_resource_manager.h"

#include <cmath>

namespace
{

using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RenderResourceManager;
using cressim::neo::graphics::TextureHandle;

float lengthSq(const Diligent::float3 &value)
{
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

bool almostEqual(float lhs, float rhs, float epsilon = 1.0e-4f)
{
    return std::fabs(lhs - rhs) <= epsilon;
}

} // namespace

int main()
{
    RenderResourceManager resources;

    MeshResourceDesc generatedDesc{};
    generatedDesc.debugName = "MeshTangents.Generated";
    generatedDesc.vertices = {
        {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 1.0f},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 1.0f},
    };
    generatedDesc.indices = {0u, 1u, 2u, 0u, 2u, 3u};

    const auto generatedMesh = resources.registerMesh(generatedDesc);
    const MeshResourceDesc *generatedMeshDesc = resources.tryGetMesh(generatedMesh);
    if (generatedMeshDesc == nullptr || generatedMeshDesc->vertices.size() != 4u)
    {
        CRESSIM_LOG_ERROR("Generated tangent mesh was not stored.\n");
        return 1;
    }

    for (const MeshResourceDesc::Vertex &vertex : generatedMeshDesc->vertices)
    {
        const Diligent::float3 tangent{vertex.tangent.x, vertex.tangent.y, vertex.tangent.z};
        if (!almostEqual(lengthSq(tangent), 1.0f) || !almostEqual(vertex.tangent.x, 1.0f) ||
            !almostEqual(vertex.tangent.y, 0.0f) || !almostEqual(vertex.tangent.z, 0.0f) ||
            !almostEqual(vertex.tangent.w, 1.0f))
        {
            CRESSIM_LOG_ERROR("Generated tangents were not normalized or not aligned with UVs.\n");
            return 1;
        }
    }

    if (resources.meshVersion(generatedMesh) != 1u || resources.meshVersion(generatedMesh) != 1u)
    {
        CRESSIM_LOG_ERROR("Mesh version changed unexpectedly after tangent generation.\n");
        return 1;
    }

    MeshResourceDesc explicitDesc{};
    explicitDesc.debugName = "MeshTangents.Explicit";
    explicitDesc.vertices = {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f, -1.0f}},
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f, {0.0f, 0.0f, 1.0f, -1.0f}},
        {{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, {0.0f, 0.0f, 1.0f, -1.0f}},
    };
    explicitDesc.indices = {0u, 1u, 2u};

    const auto explicitMesh = resources.registerMesh(explicitDesc);
    const MeshResourceDesc *explicitMeshDesc = resources.tryGetMesh(explicitMesh);
    if (explicitMeshDesc == nullptr || explicitMeshDesc->vertices.size() != explicitDesc.vertices.size())
    {
        CRESSIM_LOG_ERROR("Explicit tangent mesh was not stored.\n");
        return 1;
    }

    for (std::size_t i = 0; i < explicitDesc.vertices.size(); ++i)
    {
        const Diligent::float4 &expected = explicitDesc.vertices[i].tangent;
        const Diligent::float4 &actual = explicitMeshDesc->vertices[i].tangent;
        if (!almostEqual(actual.x, expected.x) || !almostEqual(actual.y, expected.y) ||
            !almostEqual(actual.z, expected.z) || !almostEqual(actual.w, expected.w))
        {
            CRESSIM_LOG_ERROR("Explicit tangents were unexpectedly regenerated.\n");
            return 1;
        }
    }

    MaterialResourceDesc materialDesc{};
    materialDesc.normalTexture = TextureHandle{7u};
    const auto material = resources.registerMaterial(materialDesc);
    const MaterialResourceDesc *storedMaterial = resources.tryGetMaterial(material);
    if (storedMaterial == nullptr ||
        !cressim::neo::graphics::hasFlag(storedMaterial->pipeline.featureFlags,
                                         MaterialFeatureFlags::NormalMap))
    {
        CRESSIM_LOG_ERROR("Materials with normal textures did not gain the NormalMap feature flag.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Mesh tangent generation checks passed.\n");
    return 0;
}
