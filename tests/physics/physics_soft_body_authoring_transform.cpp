#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>
#include <filesystem>

namespace
{

Diligent::float3 applyTransform(const cressim::neo::common::Transform &transform,
                                const Diligent::float3 &point)
{
    const Diligent::float3 scaled{point.x * transform.scale.x, point.y * transform.scale.y,
                                  point.z * transform.scale.z};
    return transform.rotation.RotateVector(scaled) + transform.position;
}

bool nearlyEqual(const Diligent::float4 &lhs, const Diligent::float3 &rhs)
{
    return std::abs(lhs.x - rhs.x) <= 1.0e-5f && std::abs(lhs.y - rhs.y) <= 1.0e-5f &&
           std::abs(lhs.z - rhs.z) <= 1.0e-5f;
}

} // namespace

int main()
{
    using namespace cressim::neo::physics;

    const cressim::neo::common::Transform transform{
        {3.0f, -2.0f, 5.0f},
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, 0.5f),
        {1.5f, 2.0f, 0.75f},
    };

    {
        PhysicsWorld world;
        SoftBodyState regularGrid{};
        regularGrid.entityId                             = 4001u;
        regularGrid.source.kind                          = SoftBodySourceKind::RegularGrid;
        regularGrid.source.regularGrid.size              = {2.0f, 2.0f, 2.0f};
        regularGrid.source.regularGrid.targetParticleSpacing = 2.0f;
        regularGrid.restTransform                        = transform;
        if (!world.upsertSoftBody(regularGrid))
        {
            CRESSIM_LOG_ERROR("Failed to author transformed RegularGrid soft body.");
            return 1;
        }

        const auto &particles = world.particles();
        if (particles.size() != 8u ||
            !nearlyEqual(particles.positionsInvMass[0], applyTransform(transform, {-1.0f, -1.0f, -1.0f})) ||
            !nearlyEqual(particles.positionsInvMass[7], applyTransform(transform, {1.0f, 1.0f, 1.0f})))
        {
            CRESSIM_LOG_ERROR("RegularGrid source did not apply the full world transform.");
            return 1;
        }
    }

    const std::vector<Diligent::float3> localTetPositions = {
        {-0.5f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.0f, 0.75f, 0.0f},
        {0.0f, 0.0f, 0.75f},
        {0.0f, 0.0f, -0.75f},
    };
    const std::vector<std::uint32_t> tetIndices = {0u, 1u, 2u, 3u, 0u, 1u, 4u, 2u};

    {
        PhysicsWorld world;
        SoftBodyState tetMesh{};
        tetMesh.entityId                             = 4002u;
        tetMesh.source.kind                          = SoftBodySourceKind::TetMesh;
        tetMesh.source.tetMesh.objectSpaceRestPositions = localTetPositions;
        tetMesh.source.tetMesh.tetVertexIndices         = tetIndices;
        tetMesh.restTransform                        = transform;
        if (!world.upsertSoftBody(tetMesh))
        {
            CRESSIM_LOG_ERROR("Failed to author transformed TetMesh soft body.");
            return 1;
        }

        const auto &particles = world.particles();
        for (std::size_t i = 0u; i < localTetPositions.size(); ++i)
        {
            if (!nearlyEqual(particles.positionsInvMass[i], applyTransform(transform, localTetPositions[i])))
            {
                CRESSIM_LOG_ERROR("TetMesh source did not apply the full world transform.");
                return 1;
            }
        }
    }

    {
        const std::filesystem::path fixtureDir =
            std::filesystem::path(__FILE__).parent_path() / "fixtures";
        PhysicsWorld world;
        SoftBodyState tetGen{};
        tetGen.entityId                    = 4003u;
        tetGen.source.kind                 = SoftBodySourceKind::TetGenFiles;
        tetGen.source.tetGen.nodeFile      = (fixtureDir / "soft_body_simple.node").string();
        tetGen.source.tetGen.eleFile       = (fixtureDir / "soft_body_simple_ids.ele").string();
        tetGen.restTransform               = transform;
        if (!world.upsertSoftBody(tetGen))
        {
            CRESSIM_LOG_ERROR("Failed to author transformed TetGen soft body.");
            return 1;
        }

        const auto &particles = world.particles();
        for (std::size_t i = 0u; i < localTetPositions.size(); ++i)
        {
            if (!nearlyEqual(particles.positionsInvMass[i], applyTransform(transform, localTetPositions[i])))
            {
                CRESSIM_LOG_ERROR("TetGen source did not apply the full world transform.");
                return 1;
            }
        }
    }

    CRESSIM_LOG_INFO("Soft-body transform propagation checks passed.");
    return 0;
}
