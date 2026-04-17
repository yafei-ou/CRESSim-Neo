#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "physics/physics_world.h"

int main()
{
    using namespace cressim::neo;

    {
        physics::PhysicsWorld world;
        physics::SoftBodyState valid{};
        valid.entityId                             = 5001u;
        valid.source.kind                          = physics::SoftBodySourceKind::RegularGrid;
        valid.source.regularGrid.size              = {1.0f, 1.0f, 1.0f};
        valid.source.regularGrid.targetParticleSpacing = 1.0f;
        if (!world.upsertSoftBody(valid))
        {
            CRESSIM_LOG_ERROR("Failed to author initial valid soft body for failure test.");
            return 1;
        }

        const physics::SoftBodyState before = *world.tryGetSoftBody(5001u);

        physics::SoftBodyState invalid = before;
        invalid.source.kind            = physics::SoftBodySourceKind::TetMesh;
        invalid.source.tetMesh.objectSpaceRestPositions = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
        };
        invalid.source.tetMesh.tetVertexIndices = {0u, 1u, 2u};

        if (world.upsertSoftBody(invalid))
        {
            CRESSIM_LOG_ERROR("Malformed tet index buffer should have failed.");
            return 1;
        }

        invalid.source.tetMesh.tetVertexIndices = {0u, 1u, 2u, 3u};
        invalid.source.tetMesh.objectSpaceRestPositions[3] = {0.5f, 0.5f, 0.0f};
        if (world.upsertSoftBody(invalid))
        {
            CRESSIM_LOG_ERROR("Degenerate tetrahedron should have failed.");
            return 1;
        }

        const physics::SoftBodyState *after = world.tryGetSoftBody(5001u);
        if (after == nullptr || after->source.kind != physics::SoftBodySourceKind::RegularGrid ||
            after->particleCount != before.particleCount || after->tetCount != before.tetCount)
        {
            CRESSIM_LOG_ERROR("PhysicsWorld did not preserve the previous soft body on failure.");
            return 1;
        }
    }

    {
        engine::World world;
        common::SceneLayoutDesc layout{};
        layout.envCount = 1u;
        world.setSceneLayout(layout);

        const common::EntityId entity = world.createEntity();

        engine::SoftBodyComponent valid{};
        valid.source.kind                          = physics::SoftBodySourceKind::RegularGrid;
        valid.source.regularGrid.size              = {1.0f, 1.0f, 1.0f};
        valid.source.regularGrid.targetParticleSpacing = 1.0f;
        if (!world.setSoftBody(entity, valid))
        {
            CRESSIM_LOG_ERROR("World failed to author initial valid soft body for failure test.");
            return 1;
        }

        engine::SoftBodyComponent invalid{};
        invalid.source.kind            = physics::SoftBodySourceKind::TetGenFiles;
        invalid.source.tetGen.nodeFile = "/definitely/missing_soft_body.node";
        invalid.source.tetGen.eleFile  = "/definitely/missing_soft_body.ele";
        if (world.setSoftBody(entity, invalid))
        {
            CRESSIM_LOG_ERROR("World should have rejected missing TetGen files.");
            return 1;
        }

        const std::optional<engine::SoftBodyComponent> after = world.tryGetSoftBody(entity);
        const physics::SoftBodyState *stored                 = world.physicsWorld().tryGetSoftBody(entity);
        if (!after.has_value() || stored == nullptr ||
            after->source.kind != physics::SoftBodySourceKind::RegularGrid ||
            stored->source.kind != physics::SoftBodySourceKind::RegularGrid)
        {
            CRESSIM_LOG_ERROR("World did not preserve the previous soft body on setSoftBody failure.");
            return 1;
        }
    }

    CRESSIM_LOG_INFO("Soft-body failure handling checks passed.");
    return 0;
}
