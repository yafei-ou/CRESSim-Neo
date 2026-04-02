#include "physics/physics_world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo::physics;

    PhysicsWorld world;

    SoftBodyState softBody{};
    softBody.entityId = 1001u;
    softBody.environmentIndex = 2u;
    softBody.origin = {1.0f, 2.0f, 3.0f};
    softBody.size = {2.0f, 2.0f, 2.0f};
    softBody.particleSpacing = 1.0f;
    softBody.particleMass = 2.0f;
    softBody.particleRadius = 0.15f;
    softBody.edgeCompliance = 0.01f;
    softBody.volumeCompliance = 0.02f;

    world.upsertSoftBody(softBody);

    const auto &softBodies = world.softBodySnapshot();
    const auto &particles = world.softParticles();
    const auto &edges = world.softEdges();
    const auto &tets = world.softTets();
    if (softBodies.size() != 1u || particles.size() != 27u || tets.size() != 40u)
    {
        CRESSIM_LOG_ERROR("Unexpected soft-body counts: bodies=", softBodies.size(),
                          " particles=", particles.size(), " tets=", tets.size(), ".");
        return 1;
    }
    if (edges.empty())
    {
        CRESSIM_LOG_ERROR("Expected non-empty edge list.");
        return 1;
    }

    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        if (particles.environmentIndices[i] != 2u)
        {
            CRESSIM_LOG_ERROR("Soft particle environment metadata mismatch.");
            return 1;
        }
    }

    const SoftBodyState *stored = world.tryGetSoftBody(1001u);
    if (stored == nullptr || stored->particleCount != 27u || stored->tetCount != 40u)
    {
        CRESSIM_LOG_ERROR("Stored soft-body offsets/counts were not populated.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft-body regular tet grid generation checks passed.");
    return 0;
}
