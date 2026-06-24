#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

namespace
{

std::vector<Diligent::float3> makeParticleBlock(std::uint32_t sideCount, float spacing)
{
    std::vector<Diligent::float3> particles;
    particles.reserve(sideCount * sideCount * sideCount);

    const float centerOffset = 0.5f * spacing * static_cast<float>(sideCount - 1u);
    for (std::uint32_t z = 0u; z < sideCount; ++z)
    {
        for (std::uint32_t y = 0u; y < sideCount; ++y)
        {
            for (std::uint32_t x = 0u; x < sideCount; ++x)
            {
                particles.push_back({static_cast<float>(x) * spacing - centerOffset,
                                     static_cast<float>(y) * spacing - centerOffset,
                                     static_cast<float>(z) * spacing - centerOffset});
            }
        }
    }

    return particles;
}

bool finiteFloat4(const Diligent::float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           std::isfinite(value.w);
}

float distance3(const Diligent::float4 &lhs, const Diligent::float4 &rhs)
{
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    const float dz = lhs.z - rhs.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.physicsDesc.enableBlockingReadback      = true;
    config.physicsDesc.substeps                    = 4u;
    config.physicsDesc.defaultIterations           = 16u;
    config.physicsDesc.softInternalIterations      = 32u;
    config.physicsDesc.softContactIterations       = 16u;
    config.physicsDesc.rigidRigidContactIterations = 0u;

    engine::Runtime runtime;
    try
    {
        if (!runtime.initialize(config))
        {
            CRESSIM_LOG_WARNING(
                "Skipping meshfree XPBD distance smoke because runtime initialization failed.\n");
            return 0;
        }
    }
    catch (const std::exception &)
    {
        CRESSIM_LOG_WARNING(
            "Skipping meshfree XPBD distance smoke because runtime initialization failed.\n");
        return 0;
    }

    engine::World &world = runtime.getWorld();

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.8f, -4.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, engine::CameraComponent{});

    constexpr float kGroundY     = -0.72f;
    constexpr float kGroundHalfY = 0.05f;
    const common::EntityId groundEntity = world.createEntity();
    engine::TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, kGroundY, 0.0f};
    world.setTransform(groundEntity, groundTransform);

    engine::RigidBodyComponent groundBody{};
    groundBody.bodyType            = physics::RigidBodyType::Static;
    groundBody.inverseMass         = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);

    engine::ColliderComponent groundCollider{};
    groundCollider.shapeType      = physics::ColliderShapeType::Box;
    groundCollider.shapeParams    = {4.0f, kGroundHalfY, 4.0f, 0.0f};
    groundCollider.friction       = 0.55f;
    groundCollider.staticFriction = 0.75f;
    groundCollider.collisionLayer = 0x1u;
    groundCollider.collisionMask  = 0x2u;
    world.addCollider(groundEntity, groundCollider);

    const common::EntityId softEntity = world.createEntity();
    engine::TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, 1.05f, 0.0f};
    world.setTransform(softEntity, softTransform);

    engine::MeshfreeSoftBodyComponent soft{};
    soft.particles                       = makeParticleBlock(3u, 0.34f);
    soft.neighbourCount                  = 12u;
    soft.particleRadius                  = 0.06f;
    soft.particleMass                    = 0.04f;
    soft.compliance                      = 2.0e-5f;
    soft.material.contact.friction       = 0.45f;
    soft.material.contact.staticFriction = 0.60f;
    soft.material.contact.damping        = 0.60f;
    soft.selfCollisionEnabled            = false;
    soft.collisionLayer                  = 0x2u;
    soft.collisionMask                   = 0x1u;
    if (!world.setMeshfreeSoftBody(softEntity, soft))
    {
        CRESSIM_LOG_ERROR("Failed to author meshfree XPBD distance body.\n");
        runtime.shutdown();
        return 1;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const physics::SoftBodyState *initialSoft = world.physicsWorld().tryGetSoftBody(softEntity);
    if (initialSoft == nullptr || initialSoft->particleCount != 27u ||
        initialSoft->edgeCount == 0u || initialSoft->tetCount != 0u)
    {
        CRESSIM_LOG_ERROR("Meshfree XPBD smoke did not author distance-only topology.\n");
        runtime.shutdown();
        return 1;
    }

    const auto &initialParticles = world.physicsWorld().particles();
    float initialAverageY        = 0.0f;
    for (std::uint32_t i = 0u; i < initialSoft->particleCount; ++i)
    {
        initialAverageY += initialParticles.positionsInvMass[initialSoft->particleOffset + i].y;
    }
    initialAverageY /= static_cast<float>(initialSoft->particleCount);

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    for (std::uint32_t i = 0u; i < 90u; ++i)
    {
        frame.frameIndex  = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.prepare();
        const bool physicsStepSucceeded = runtime.uploadWorld() && runtime.stepPhysics(frame);
        if (physicsStepSucceeded)
        {
            (void)runtime.stepSimulationSensors(frame);
        }
        runtime.stepVisualSensors(frame);
        runtime.endFrame(frame);
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const physics::SoftBodyState *finalSoft = world.physicsWorld().tryGetSoftBody(softEntity);
    if (finalSoft == nullptr || finalSoft->particleCount != 27u || finalSoft->tetCount != 0u)
    {
        CRESSIM_LOG_ERROR("Meshfree XPBD soft body disappeared or gained tetrahedra.\n");
        runtime.shutdown();
        return 1;
    }

    const auto &finalParticles = world.physicsWorld().particles();
    const auto &edges          = world.physicsWorld().softEdges();
    float finalAverageY        = 0.0f;
    float minY                 = std::numeric_limits<float>::max();
    for (std::uint32_t i = 0u; i < finalSoft->particleCount; ++i)
    {
        const Diligent::float4 &particle =
            finalParticles.positionsInvMass[finalSoft->particleOffset + i];
        if (!finiteFloat4(particle))
        {
            CRESSIM_LOG_ERROR("Meshfree XPBD particle state became non-finite.\n");
            runtime.shutdown();
            return 1;
        }
        finalAverageY += particle.y;
        minY = std::min(minY, particle.y);
    }
    finalAverageY /= static_cast<float>(finalSoft->particleCount);

    if (!(finalAverageY < initialAverageY - 0.2f))
    {
        CRESSIM_LOG_ERROR("Meshfree XPBD body did not fall under gravity.\n");
        runtime.shutdown();
        return 1;
    }

    const float groundTop = kGroundY + kGroundHalfY;
    if (minY < groundTop + soft.particleRadius - 0.15f)
    {
        CRESSIM_LOG_ERROR("Meshfree XPBD body tunneled too far through the ground.\n");
        runtime.shutdown();
        return 1;
    }

    float maxStretchRatio = 0.0f;
    for (std::uint32_t edgeIndex = finalSoft->edgeOffset;
         edgeIndex < finalSoft->edgeOffset + finalSoft->edgeCount; ++edgeIndex)
    {
        const physics::SoftEdge &edge = edges[edgeIndex];
        if (edge.restLength <= 1.0e-5f)
        {
            continue;
        }
        const float currentLength = distance3(finalParticles.positionsInvMass[edge.particleA],
                                              finalParticles.positionsInvMass[edge.particleB]);
        maxStretchRatio = std::max(maxStretchRatio, currentLength / edge.restLength);
    }
    if (!(maxStretchRatio < 2.5f))
    {
        CRESSIM_LOG_ERROR("Meshfree XPBD distance constraints diverged; max stretch ratio was ",
                          maxStretchRatio, ".\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Meshfree XPBD distance smoke checks passed.\n");
    return 0;
}
