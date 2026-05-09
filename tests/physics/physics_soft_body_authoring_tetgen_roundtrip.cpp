#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>
#include <filesystem>

namespace
{

cressim::neo::physics::SoftBodyState makeEquivalentTetMeshState()
{
    using namespace cressim::neo::physics;

    SoftBodyState softBody{};
    softBody.entityId         = 3001u;
    softBody.source.kind      = SoftBodySourceKind::TetMesh;
    softBody.source.tetMesh.objectSpaceRestPositions = {
        {-0.5f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.0f, 0.75f, 0.0f},
        {0.0f, 0.0f, 0.75f},
        {0.0f, 0.0f, -0.75f},
    };
    softBody.source.tetMesh.tetVertexIndices = {0u, 1u, 2u, 3u, 0u, 1u, 4u, 2u};
    softBody.source.tetMesh.staticParticleIndices = {1u};
    softBody.particleMass     = 2.0f;
    softBody.particleRadius   = 0.1f;
    softBody.edgeCompliance   = 0.01f;
    softBody.volumeCompliance = 0.02f;
    return softBody;
}

bool compareWorlds(const cressim::neo::physics::PhysicsWorld &lhs,
                   const cressim::neo::physics::PhysicsWorld &rhs)
{
    using namespace cressim::neo::physics;

    const auto &lhsParticles = lhs.particles();
    const auto &rhsParticles = rhs.particles();
    const auto &lhsEdges     = lhs.softEdges();
    const auto &rhsEdges     = rhs.softEdges();
    const auto &lhsTets      = lhs.softTets();
    const auto &rhsTets      = rhs.softTets();

    if (lhsParticles.size() != rhsParticles.size() || lhsEdges.size() != rhsEdges.size() ||
        lhsTets.size() != rhsTets.size())
    {
        return false;
    }

    for (std::size_t i = 0u; i < lhsParticles.size(); ++i)
    {
        const auto &lp = lhsParticles.positionsInvMass[i];
        const auto &rp = rhsParticles.positionsInvMass[i];
        if (std::abs(lp.x - rp.x) > 1.0e-6f || std::abs(lp.y - rp.y) > 1.0e-6f ||
            std::abs(lp.z - rp.z) > 1.0e-6f || std::abs(lp.w - rp.w) > 1.0e-6f)
        {
            return false;
        }
    }

    for (std::size_t i = 0u; i < lhsEdges.size(); ++i)
    {
        if (lhsEdges[i].particleA != rhsEdges[i].particleA ||
            lhsEdges[i].particleB != rhsEdges[i].particleB ||
            std::abs(lhsEdges[i].restLength - rhsEdges[i].restLength) > 1.0e-6f)
        {
            return false;
        }
    }

    for (std::size_t i = 0u; i < lhsTets.size(); ++i)
    {
        if (lhsTets[i].particleIndices.x != rhsTets[i].particleIndices.x ||
            lhsTets[i].particleIndices.y != rhsTets[i].particleIndices.y ||
            lhsTets[i].particleIndices.z != rhsTets[i].particleIndices.z ||
            lhsTets[i].particleIndices.w != rhsTets[i].particleIndices.w ||
            std::abs(lhsTets[i].restVolume - rhsTets[i].restVolume) > 1.0e-6f)
        {
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    using namespace cressim::neo::physics;

    const std::filesystem::path fixtureDir =
        std::filesystem::path(__FILE__).parent_path() / "fixtures";
    const std::filesystem::path nodeFile = fixtureDir / "soft_body_simple.node";
    const std::filesystem::path eleFileIds = fixtureDir / "soft_body_simple_ids.ele";
    const std::filesystem::path eleFileDense = fixtureDir / "soft_body_simple_dense1.ele";

    PhysicsWorld referenceWorld;
    SoftBodyState tetMeshState = makeEquivalentTetMeshState();
    if (!referenceWorld.upsertSoftBody(tetMeshState))
    {
        CRESSIM_LOG_ERROR("Failed to author reference TetMesh soft body.");
        return 1;
    }

    PhysicsWorld idMappedWorld;
    SoftBodyState tetGenState = tetMeshState;
    tetGenState.entityId      = 3002u;
    tetGenState.source.kind   = SoftBodySourceKind::TetGenFiles;
    tetGenState.source.tetGen.nodeFile = nodeFile.string();
    tetGenState.source.tetGen.eleFile  = eleFileIds.string();
    tetGenState.source.tetGen.staticParticleIndices = {1u};
    if (!idMappedWorld.upsertSoftBody(tetGenState))
    {
        CRESSIM_LOG_ERROR("Failed to author TetGen soft body from arbitrary-id fixture.");
        return 1;
    }

    if (!compareWorlds(referenceWorld, idMappedWorld))
    {
        CRESSIM_LOG_ERROR("TetGen arbitrary-id fixture did not match equivalent TetMesh topology.");
        return 1;
    }

    PhysicsWorld denseOneWorld;
    tetGenState.entityId               = 3003u;
    tetGenState.source.tetGen.eleFile  = eleFileDense.string();
    if (denseOneWorld.upsertSoftBody(tetGenState))
    {
        CRESSIM_LOG_ERROR("Dense reindexed TetGen fixture should have been rejected.");
        return 1;
    }

    CRESSIM_LOG_INFO("TetGen soft-body round-trip and format checks passed.");
    return 0;
}
