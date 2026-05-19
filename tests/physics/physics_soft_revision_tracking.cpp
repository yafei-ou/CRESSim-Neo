#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::physics::SoftBodyState;
using cressim::neo::physics::SoftRenderDataHost;

bool nearlyEqual(float lhs, float rhs)
{
    return std::abs(lhs - rhs) <= 1.0e-6f;
}

SoftBodyState makeSoftBody(EntityId entityId)
{
    SoftBodyState state{};
    state.entityId                             = entityId;
    state.source.kind                          = SoftBodySourceKind::RegularGrid;
    state.source.regularGrid.size              = {1.0f, 1.0f, 1.0f};
    state.source.regularGrid.targetParticleSpacing = 1.0f;
    state.particleMass                         = 1.0f;
    state.particleRadius                       = 0.1f;
    state.edgeCompliance                       = 0.02f;
    state.volumeCompliance                     = 0.03f;
    return state;
}

} // namespace

int main()
{
    PhysicsWorld world;

    const EntityId entity = 2001u;
    if (!world.upsertSoftBody(makeSoftBody(entity)))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for revision tracking test.");
        return 1;
    }

    if (!nearlyEqual(world.particleGridCellSize(), 0.2f))
    {
        CRESSIM_LOG_ERROR("Unexpected initial cached particle grid cell size.");
        return 1;
    }

    const std::uint64_t particleRevisionBeforeRender = world.softParticleRevision();
    const std::uint64_t topologyRevisionBeforeRender = world.softGpuTopologyRevision();

    SoftRenderDataHost renderData{};
    renderData.fallbackNormals.emplace_back(0.0f, 1.0f, 0.0f, 0.0f);
    renderData.softBodyParticleRanges.emplace_back(0u, 65u);
    world.setSoftRenderData(renderData);

    if (world.softBodyBoundsChunkCount() != 2u)
    {
        CRESSIM_LOG_ERROR("Soft render data update did not refresh cached bounds chunk count.");
        return 1;
    }

    if (world.softParticleRevision() != particleRevisionBeforeRender)
    {
        CRESSIM_LOG_ERROR("Soft render data update should not bump soft particle revision.");
        return 1;
    }
    if (world.softGpuTopologyRevision() != topologyRevisionBeforeRender + 1u)
    {
        CRESSIM_LOG_ERROR("Soft render data update should bump soft GPU topology revision.");
        return 1;
    }

    SoftBodyState updated = *world.tryGetSoftBody(entity);
    updated.particleRadius = 0.2f;
    if (!world.upsertSoftBody(updated))
    {
        CRESSIM_LOG_ERROR("Failed to apply runtime soft body update.");
        return 1;
    }

    if (world.softParticleRevision() != particleRevisionBeforeRender + 1u)
    {
        CRESSIM_LOG_ERROR("Runtime soft body update should bump soft particle revision.");
        return 1;
    }
    if (!nearlyEqual(world.particleGridCellSize(), 0.4f))
    {
        CRESSIM_LOG_ERROR("Runtime soft body update did not refresh cached particle grid cell size.");
        return 1;
    }
    if (world.softGpuTopologyRevision() != topologyRevisionBeforeRender + 1u)
    {
        CRESSIM_LOG_ERROR("Runtime soft body update should not bump soft GPU topology revision.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft revision tracking checks passed.");
    return 0;
}
