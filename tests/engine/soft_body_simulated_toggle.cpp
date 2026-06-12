#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "graphics/render_resource_manager.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    world.setSceneLayout(common::SceneLayoutDesc{});

    graphics::RenderResourceManager resources;

    graphics::MeshResourceDesc meshDesc{};
    meshDesc.debugName = "SoftToggle.Mesh";
    meshDesc.vertices = {
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, 0.0f, 1.0f},
        {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, 1.0f, 1.0f},
        {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, 0.0f, 1.0f},
        {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, 1.0f, 1.0f},
    };
    meshDesc.indices = {
        0u, 1u, 3u, 0u, 3u, 2u, 4u, 6u, 7u, 4u, 7u, 5u, 0u, 4u, 5u, 0u, 5u, 1u,
        2u, 3u, 7u, 2u, 7u, 6u, 0u, 2u, 6u, 0u, 6u, 4u, 1u, 5u, 7u, 1u, 7u, 3u,
    };
    const graphics::MeshHandle mesh = resources.registerMesh(meshDesc);

    graphics::MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "SoftToggle.Material";
    const graphics::MaterialHandle material = resources.registerMaterial(materialDesc);

    const common::EntityId entity = world.createEntity();
    world.setMeshRenderer(entity, engine::MeshRendererComponent{mesh, material, true});

    engine::SoftBodyComponent softBody{};
    softBody.source.kind                          = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size              = {1.0f, 1.0f, 1.0f};
    softBody.source.regularGrid.targetParticleSpacing = 1.0f;
    if (!world.setSoftBody(entity, softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for simulated-toggle test.");
        return 1;
    }

    world.ensureRenderStateUpToDate(resources);

    const auto &beforeMetadata = world.renderableMetadata();
    if (beforeMetadata.empty() ||
        beforeMetadata[0].deformableType !=
            static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::SoftBody) ||
        beforeMetadata[0].deformableIndex == 0xffffffffu ||
        beforeMetadata[0].deformVertexBase == 0xffffffffu ||
        beforeMetadata[0].deformVertexCount == 0u)
    {
        CRESSIM_LOG_ERROR("Soft body render metadata was not populated before toggle.");
        return 1;
    }

    softBody.simulated = false;
    if (!world.setSoftBody(entity, softBody))
    {
        CRESSIM_LOG_ERROR("Disabling soft body simulation should succeed.");
        return 1;
    }

    world.ensureRenderStateUpToDate(resources);

    const auto &afterMetadata = world.renderableMetadata();
    if (afterMetadata.empty())
    {
        CRESSIM_LOG_ERROR("Renderable metadata unexpectedly disappeared after toggle.");
        return 1;
    }

    if (world.physicsWorld().tryGetSoftBody(entity) != nullptr)
    {
        CRESSIM_LOG_ERROR("Soft body still exists in physics world after simulated=false.");
        return 1;
    }

    if (afterMetadata[0].deformableType !=
            static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::None) ||
        afterMetadata[0].deformableIndex != 0xffffffffu ||
        afterMetadata[0].deformVertexBase != 0xffffffffu ||
        afterMetadata[0].deformNormalBase != 0xffffffffu ||
        afterMetadata[0].deformVertexCount != 0u)
    {
        CRESSIM_LOG_ERROR("Soft body render metadata was not cleared after simulated=false.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft body simulated toggle cleanup checks passed.");
    return 0;
}
