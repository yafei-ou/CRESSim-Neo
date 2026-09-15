#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <optional>

namespace
{

float closestSegmentDistance(const Diligent::float3 &a0, const Diligent::float3 &a1,
                             const Diligent::float3 &b0, const Diligent::float3 &b1)
{
    const Diligent::float3 d1 = a1 - a0;
    const Diligent::float3 d2 = b1 - b0;
    const Diligent::float3 r  = a0 - b0;
    const float a             = Diligent::dot(d1, d1);
    const float e             = Diligent::dot(d2, d2);
    const float f             = Diligent::dot(d2, r);
    constexpr float kEpsilon  = 1.0e-8f;

    float s = 0.0f;
    float t = 0.0f;
    if (a <= kEpsilon && e <= kEpsilon)
    {
        return Diligent::length(a0 - b0);
    }
    if (a <= kEpsilon)
    {
        t = std::clamp(f / e, 0.0f, 1.0f);
    }
    else
    {
        const float c = Diligent::dot(d1, r);
        if (e <= kEpsilon)
        {
            s = std::clamp(-c / a, 0.0f, 1.0f);
        }
        else
        {
            const float b           = Diligent::dot(d1, d2);
            const float denominator = a * e - b * b;
            if (std::abs(denominator) > kEpsilon)
            {
                s = std::clamp((b * f - c * e) / denominator, 0.0f, 1.0f);
            }
            const float tNumerator = b * s + f;
            if (tNumerator < 0.0f)
            {
                t = 0.0f;
                s = std::clamp(-c / a, 0.0f, 1.0f);
            }
            else if (tNumerator > e)
            {
                t = 1.0f;
                s = std::clamp((b - c) / a, 0.0f, 1.0f);
            }
            else
            {
                t = tNumerator / e;
            }
        }
    }

    const Diligent::float3 pointA = a0 + d1 * s;
    const Diligent::float3 pointB = b0 + d2 * t;
    return Diligent::length(pointB - pointA);
}

std::optional<float> runCrossingScenario(bool enableCapsuleSelfCollision)
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.physicsDesc.gravity = {0.0f, 0.0f, 0.0f};
    config.physicsDesc.substeps               = 1u;
    config.physicsDesc.defaultIterations      = 1u;
    config.physicsDesc.softInternalIterations = 1u;
    config.physicsDesc.softContactIterations  = 8u;
    config.physicsDesc.contact.slop            = 0.0f;
    config.physicsDesc.enableExperimentalStrandCapsuleSelfCollision =
        enableCapsuleSelfCollision;
    config.physicsDesc.experimentalStrandCapsuleSelfCollisionPasses = 8u;

    engine::Runtime runtime;
    try
    {
        if (!runtime.initialize(config))
        {
            return std::nullopt;
        }
    }
    catch (const std::exception &)
    {
        return std::nullopt;
    }

    engine::World &world          = runtime.getWorld();
    const common::EntityId entity = world.createEntity();
    engine::StrandComponent strand{};
    // Segment 0 runs along X at z=0. Segment 3 crosses it along Y at z=0.02. Their
    // particles are far apart, so node-sphere contacts cannot resolve this intersection.
    strand.restPositions = {{-0.5f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f},
                            {0.5f, 0.5f, 0.02f}, {0.0f, 0.5f, 0.02f},
                            {0.0f, -0.5f, 0.02f}, {-0.5f, -0.5f, 0.02f}};
    strand.particleMass           = 1.0f;
    strand.particleRadius         = 0.05f;
    strand.stretchShearCompliance = 1.0f;
    strand.bendCompliance         = 1.0f;
    strand.twistCompliance        = 1.0f;
    strand.distanceCompliance     = 1.0f;
    strand.selfCollisionEnabled   = false;
    if (!world.setStrand(entity, strand))
    {
        runtime.shutdown();
        return std::nullopt;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    runtime.prepare();
    const bool stepSucceeded = runtime.uploadWorld() && runtime.stepPhysics(frame);
    runtime.endFrame(frame);
    if (!stepSucceeded)
    {
        runtime.shutdown();
        return std::nullopt;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const physics::StrandState *state = world.physicsWorld().tryGetStrand(entity);
    if (state == nullptr || state->particleCount != strand.restPositions.size())
    {
        runtime.shutdown();
        return std::nullopt;
    }
    const auto &positions = world.physicsWorld().particles().positionsInvMass;
    const std::uint32_t offset = state->particleOffset;
    const Diligent::float3 a0{positions[offset].x, positions[offset].y, positions[offset].z};
    const Diligent::float3 a1{positions[offset + 1u].x, positions[offset + 1u].y,
                              positions[offset + 1u].z};
    const Diligent::float3 b0{positions[offset + 3u].x, positions[offset + 3u].y,
                              positions[offset + 3u].z};
    const Diligent::float3 b1{positions[offset + 4u].x, positions[offset + 4u].y,
                              positions[offset + 4u].z};
    const float distance = closestSegmentDistance(a0, a1, b0, b1);
    runtime.shutdown();
    return distance;
}

} // namespace

int main()
{
    const std::optional<float> baselineDistance = runCrossingScenario(false);
    const std::optional<float> capsuleDistance  = runCrossingScenario(true);
    if (!baselineDistance.has_value() || !capsuleDistance.has_value())
    {
        CRESSIM_LOG_WARNING(
            "Skipping experimental strand capsule test because runtime execution failed.\n");
        return 0;
    }

    if (*baselineDistance > 0.03f)
    {
        CRESSIM_LOG_ERROR("Baseline crossing unexpectedly separated to ", *baselineDistance,
                          ".\n");
        return 1;
    }
    if (*capsuleDistance < 0.075f || *capsuleDistance <= *baselineDistance + 0.04f)
    {
        CRESSIM_LOG_ERROR("Experimental capsule self-collision did not separate crossing links: "
                          "baseline=",
                          *baselineDistance, ", capsule=", *capsuleDistance, ".\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Experimental strand capsule self-collision passed: baseline=",
                     *baselineDistance, ", capsule=", *capsuleDistance, ".\n");
    return 0;
}
