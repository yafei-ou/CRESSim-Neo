#include "physics/physics_world.h"

#include <cmath>

int main()
{
    using namespace cressim::neo::physics;
    PhysicsWorld world;
    ClothState cloth{};
    cloth.entityId                        = 42u;
    cloth.source.objectSpaceRestPositions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    cloth.source.triangleVertexIndices    = {0, 1, 2, 2, 1, 3};
    cloth.source.staticParticleIndices    = {1, 0, 1};
    cloth.particleMass                    = 2.0f;
    cloth.selfCollisionEnabled            = true;
    if (!world.upsertCloth(cloth)) return 1;
    const auto &particles    = world.particles();
    const auto &edges        = world.distanceConstraints();
    const auto &hinges       = world.clothDihedralConstraints();
    const ClothState *stored = world.tryGetCloth(42u);
    if (stored == nullptr || stored->particleCount != 4u ||
        stored->structuralConstraintCount != 5u || stored->dihedralConstraintCount != 1u)
        return 2;
    if (particles.size() != 4u || edges.size() != 5u || hinges.size() != 1u ||
        !world.bendConstraints().empty())
        return 3;
    if (particles.positionsInvMass[0].w != 0.0f || particles.positionsInvMass[1].w != 0.0f ||
        std::abs(particles.positionsInvMass[2].w - 0.5f) > 1.0e-6f)
        return 4;
    if (std::abs(hinges[0].restAngle) > 1.0e-6f) return 5;
    if (particles.ownerTypes[0] != static_cast<std::uint32_t>(ParticleOwnerType::Cloth)) return 6;

    const std::uint32_t dynamicParticle = stored->particleOffset + 2u;
    const Diligent::float4 displaced{2.0f, 3.0f, 4.0f, 0.5f};
    const Diligent::float4 previous{1.5f, 2.5f, 3.5f, 0.0f};
    const Diligent::float4 velocity{0.5f, -1.0f, 2.0f, 0.0f};
    if (!world.syncParticleStateFromSimulation(dynamicParticle, displaced, previous, velocity))
        return 7;
    world.finalizeParticleWriteback();

    const std::uint64_t renderRevision   = world.clothTopologyRevision();
    ClothState runtimeUpdate             = *world.tryGetCloth(42u);
    runtimeUpdate.particleMass           = 4.0f;
    runtimeUpdate.particleRadius         = 0.2f;
    runtimeUpdate.structuralCompliance   = 0.1f;
    runtimeUpdate.bendCompliance         = 0.2f;
    runtimeUpdate.renderVertexToParticle = {0u, 1u, 2u, 3u};
    if (!world.upsertCloth(runtimeUpdate)) return 8;
    const auto &updatedParticles = world.particles();
    stored                       = world.tryGetCloth(42u);
    if (stored == nullptr || updatedParticles.positionsInvMass[dynamicParticle].x != displaced.x ||
        updatedParticles.positionsInvMass[dynamicParticle].y != displaced.y ||
        updatedParticles.positionsInvMass[dynamicParticle].z != displaced.z ||
        std::abs(updatedParticles.positionsInvMass[dynamicParticle].w - 0.25f) > 1.0e-6f ||
        updatedParticles.previousPositions[dynamicParticle] != previous ||
        updatedParticles.velocities[dynamicParticle] != velocity ||
        updatedParticles.positionsInvMass[stored->particleOffset].w != 0.0f ||
        std::abs(world.distanceConstraints()[stored->structuralConstraintOffset].compliance -
                 0.1f) > 1.0e-6f ||
        std::abs(world.clothDihedralConstraints()[stored->dihedralConstraintOffset].compliance -
                 0.2f) > 1.0e-6f ||
        std::abs(world.particleGridCellSize() - 0.4f) > 1.0e-6f ||
        world.clothTopologyRevision() != renderRevision + 1u)
        return 9;

    ClothState invalid                   = *stored;
    invalid.source.triangleVertexIndices = {0, 0, 2};
    if (world.upsertCloth(invalid) || world.tryGetCloth(42u)->particleCount != 4u) return 10;

    ClothState seam{};
    seam.entityId                        = 43u;
    seam.source.objectSpaceRestPositions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                                            {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    seam.source.triangleVertexIndices    = {0, 1, 2, 4, 3, 5};
    if (!world.upsertCloth(seam)) return 11;
    world.ensureDerivedStateUpToDate();
    const ClothState *seamStored = world.tryGetCloth(43u);
    if (seamStored == nullptr || seamStored->dihedralConstraintCount != 0u ||
        seamStored->structuralConstraintCount != 6u)
        return 12;
    return 0;
}
