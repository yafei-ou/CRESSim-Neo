#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>

int main()
{
    using namespace cressim::neo::physics;

    PhysicsWorld world;

    SoftBodyState softBody{};
    softBody.entityId             = 2001u;
    softBody.environmentIndex     = 1u;
    softBody.collisionLayer       = 0x6u;
    softBody.collisionMask        = 0x19u;
    softBody.source.kind          = SoftBodySourceKind::TetMesh;
    softBody.source.tetMesh.objectSpaceRestPositions = {
        {-0.5f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.0f, 0.75f, 0.0f},
        {0.0f, 0.0f, 0.75f},
        {0.0f, 0.0f, -0.75f},
    };
    softBody.source.tetMesh.tetVertexIndices = {0u, 1u, 2u, 3u, 0u, 1u, 4u, 2u};
    softBody.source.tetMesh.staticParticleIndices = {0u, 4u};
    softBody.particleMass       = 3.0f;
    softBody.particleRadius     = 0.2f;
    softBody.edgeCompliance     = 0.01f;
    softBody.volumeCompliance   = 0.02f;
    softBody.selfCollisionEnabled = true;

    if (!world.upsertSoftBody(softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author in-memory tet soft body.");
        return 1;
    }

    const SoftBodyState *stored = world.tryGetSoftBody(2001u);
    if (stored == nullptr || stored->source.kind != SoftBodySourceKind::TetMesh)
    {
        CRESSIM_LOG_ERROR("Stored tet-mesh soft body metadata was not preserved.");
        return 1;
    }

    const auto &particles = world.softParticles();
    const auto &edges     = world.softEdges();
    const auto &tets      = world.softTets();
    stored                = world.tryGetSoftBody(2001u);
    if (stored == nullptr || stored->particleCount != 5u || stored->tetCount != 2u ||
        stored->edgeCount != 9u)
    {
        CRESSIM_LOG_ERROR("Derived tet-mesh soft body counts were not populated.");
        return 1;
    }

    if (particles.size() != 5u || edges.size() != 9u || tets.size() != 2u)
    {
        CRESSIM_LOG_ERROR("Unexpected tet-mesh topology counts.");
        return 1;
    }

    if (particles.positionsInvMass[0].w != 0.0f || particles.positionsInvMass[4].w != 0.0f ||
        std::abs(particles.positionsInvMass[1].w - (1.0f / 3.0f)) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Tet-mesh static particle pinning or inverse mass is incorrect.");
        return 1;
    }

    const std::uint32_t expectedAdjacencyCounts[5] = {4u, 4u, 4u, 3u, 3u};
    for (std::uint32_t i = 0u; i < 5u; ++i)
    {
        if (particles.adjacencyCounts[i] != expectedAdjacencyCounts[i] ||
            particles.environmentIndices[i] != 1u || particles.collisionLayers[i] != 0x6u ||
            particles.collisionMasks[i] != 0x19u)
        {
            CRESSIM_LOG_ERROR("Tet-mesh particle adjacency or collision metadata mismatch.");
            return 1;
        }
    }

    if (tets[0].restVolume <= 0.0f || tets[1].restVolume <= 0.0f)
    {
        CRESSIM_LOG_ERROR("Tet-mesh rest volumes were not computed.");
        return 1;
    }

    CRESSIM_LOG_INFO("Tet-mesh soft-body authoring checks passed.");
    return 0;
}
