#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::AuthoredParticleDistanceConstraintState;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredSuturingSequenceState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ParticleContactMaterialDesc;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;

std::uint32_t flattenGridIndex(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                               const Diligent::uint3 &resolution)
{
    return x * resolution.y * resolution.z + y * resolution.z + z;
}

std::vector<std::uint32_t> makeRightSideStaticIndices(const Diligent::float3 &size, float spacing)
{
    const float clampedSpacing = std::max(spacing, 1.0e-4f);
    const auto deriveResolution = [clampedSpacing](const float extent) -> std::uint32_t
    {
        return std::max<std::uint32_t>(2u, static_cast<std::uint32_t>(std::ceil(extent / clampedSpacing)) +
                                               1u);
    };

    const Diligent::uint3 resolution{
        deriveResolution(std::max(size.x, 1.0e-4f)),
        deriveResolution(std::max(size.y, 1.0e-4f)),
        deriveResolution(std::max(size.z, 1.0e-4f)),
    };

    std::vector<std::uint32_t> result;
    result.reserve(static_cast<std::size_t>(resolution.y) * resolution.z);
    const std::uint32_t rightX = resolution.x - 1u;
    for (std::uint32_t y = 0u; y < resolution.y; ++y)
    {
        for (std::uint32_t z = 0u; z < resolution.z; ++z)
        {
            result.push_back(flattenGridIndex(rightX, y, z, resolution));
        }
    }
    return result;
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName, "  Debug particle rendering is enabled by default.\n", false);
}

std::vector<Diligent::float3> makeArcProxyParticles(float arcRadius, float startAngleRadians,
                                                    float endAngleRadians,
                                                    std::uint32_t sampleCount)
{
    const std::uint32_t count = std::max(sampleCount, 1u);
    std::vector<Diligent::float3> points;
    points.reserve(count);

    for (std::uint32_t i = 0u; i < count; ++i)
    {
        const float t = count > 1u ? static_cast<float>(i) / static_cast<float>(count - 1u) : 0.0f;
        const float angle = startAngleRadians + (endAngleRadians - startAngleRadians) * t;
        points.push_back({std::cos(angle) * arcRadius, std::sin(angle) * arcRadius, 0.0f});
    }

    return points;
}

struct ProxyMassProperties
{
    Diligent::float3 centerOfMass{0.0f, 0.0f, 0.0f};
    Diligent::float3 inverseInertiaLocal{0.0f, 0.0f, 0.0f};
    std::vector<Diligent::float3> centeredPoints{};
};

ProxyMassProperties computeProxyMassProperties(const std::vector<Diligent::float3> &points,
                                               float particleRadius, float inverseMass)
{
    ProxyMassProperties properties{};
    if (points.empty() || inverseMass <= 0.0f)
    {
        return properties;
    }

    for (const Diligent::float3 &point : points)
    {
        properties.centerOfMass += point;
    }
    const float pointCountInv = 1.0f / static_cast<float>(points.size());
    properties.centerOfMass *= pointCountInv;

    properties.centeredPoints.reserve(points.size());
    const float totalMass = 1.0f / inverseMass;
    const float particleMass = totalMass * pointCountInv;
    const float particleSelfInertia = 0.4f * particleMass * particleRadius * particleRadius;

    Diligent::float3 inertia{0.0f, 0.0f, 0.0f};
    for (const Diligent::float3 &point : points)
    {
        const Diligent::float3 centered = point - properties.centerOfMass;
        properties.centeredPoints.push_back(centered);
        inertia.x += particleMass * (centered.y * centered.y + centered.z * centered.z) +
                     particleSelfInertia;
        inertia.y += particleMass * (centered.x * centered.x + centered.z * centered.z) +
                     particleSelfInertia;
        inertia.z += particleMass * (centered.x * centered.x + centered.y * centered.y) +
                     particleSelfInertia;
    }

    properties.inverseInertiaLocal = {
        inertia.x > 0.0f ? 1.0f / inertia.x : 0.0f, inertia.y > 0.0f ? 1.0f / inertia.y : 0.0f,
        inertia.z > 0.0f ? 1.0f / inertia.z : 0.0f};
    return properties;
}

float signedAngleXY(const Diligent::float3 &from, const Diligent::float3 &to)
{
    const Diligent::float2 a = Diligent::normalize(Diligent::float2{from.x, from.y});
    const Diligent::float2 b = Diligent::normalize(Diligent::float2{to.x, to.y});
    const float dot = std::clamp(a.x * b.x + a.y * b.y, -1.0f, 1.0f);
    const float det = a.x * b.y - a.y * b.x;
    return std::atan2(det, dot);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
            {
                continue;
            }

            printUsage(argv[0]);
            return 2;
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.physicsDesc.substeps = 6u;
    config.physicsDesc.defaultIterations = 14u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Kinematic Arc Needle With Strand";
    viewerDefaults.showStats = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.enableDebugParticles = true;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    auto &world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.9f, -8.5f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.22f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 38.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(Diligent::float3{-0.4f, -1.0f, 0.3f});
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 6.5f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(12.0f, "KinematicArcNeedleThread.GroundMesh"));

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "KinematicArcNeedleThread.GroundMaterial";
    groundMaterialDesc.baseColor = {0.76f, 0.79f, 0.82f};
    groundMaterialDesc.roughness = 0.94f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.1f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{groundMesh, groundMaterial, true});

    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);

    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {12.0f, 0.08f, 12.0f, 0.0f};
    groundCollider.friction = 0.55f;
    groundCollider.staticFriction = 0.7f;
    world.addCollider(groundEntity, groundCollider);

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, -0.44f, 0.0f};
    world.setTransform(softEntity, softTransform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {2.0f, 1.0f, 1.2f};
    softBody.source.regularGrid.targetParticleSpacing = 0.24f;
    softBody.source.regularGrid.staticParticleIndices = makeRightSideStaticIndices(
        softBody.source.regularGrid.size, softBody.source.regularGrid.targetParticleSpacing);
    softBody.particleMass = 0.08f;
    softBody.particleRadius = 0.08f;
    softBody.edgeCompliance = 0.0005f;
    softBody.volumeCompliance = 0.001f;
    softBody.selfCollisionEnabled = false;
    softBody.supportsSuturing = true;
    softBody.material.contact.friction = 0.48f;
    softBody.material.contact.staticFriction = 0.58f;
    if (!world.setSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author soft body.\n");
        return 1;
    }

    const float needleParticleRadius = 0.1f;
    const std::vector<Diligent::float3> needleProxyParticles =
        makeArcProxyParticles(0.95f, -0.1f * kPi, -0.9f * kPi, 10u);
    constexpr float kNeedleInverseMass = 1.4f;
    const ProxyMassProperties needleMassProperties =
        computeProxyMassProperties(needleProxyParticles, needleParticleRadius, kNeedleInverseMass);
    const std::uint32_t tipProxyIndex = 0u;
    const std::uint32_t tailProxyIndex =
        static_cast<std::uint32_t>(needleMassProperties.centeredPoints.size() - 1u);

    const auto needleEntity = world.createEntity();
    const Diligent::float3 tipTangentLocal =
        needleProxyParticles.size() >= 2u
            ? Diligent::normalize(needleProxyParticles[tipProxyIndex] -
                                  needleProxyParticles[tipProxyIndex + 1u])
            : Diligent::float3{1.0f, 0.0f, 0.0f};
    const Diligent::float3 desiredTipTangentWorld = Diligent::normalize(
        Diligent::float3{1.0f, 0.12f, 0.0f});

    const float kBaseNeedleAngle = signedAngleXY(tipTangentLocal, desiredTipTangentWorld);
    constexpr float kRotationStageAngleDelta = 3.35f;
    auto computeNeedleRotation = [](float angle)
    {
        return Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, angle);
    };

    const Diligent::float3 arcCenterLocal = -needleMassProperties.centerOfMass;
    const Diligent::float3 tipFromArcCenterLocal = needleProxyParticles[tipProxyIndex];

    auto computeNeedleBodyPositionFromArcCenter = [&](const Diligent::float3 &arcCenterPosition,
                                                      const Diligent::QuaternionF &needleRotation)
    {
        return arcCenterPosition - needleRotation.RotateVector(arcCenterLocal);
    };

    const Diligent::QuaternionF needleRotation = computeNeedleRotation(kBaseNeedleAngle);
    const Diligent::float3 startTipPosition{-1.6f, -0.24f, 0.0f};
    const Diligent::float3 startArcCenterPosition =
        startTipPosition - needleRotation.RotateVector(tipFromArcCenterLocal);

    TransformComponent needleTransform{};
    needleTransform.worldTransform.rotation = needleRotation;
    needleTransform.worldTransform.position =
        computeNeedleBodyPositionFromArcCenter(startArcCenterPosition, needleRotation);
    world.setTransform(needleEntity, needleTransform);

    ParticleContactMaterialDesc needleContactMaterial{};
    needleContactMaterial.friction = 0.32f;
    needleContactMaterial.staticFriction = 0.4f;
    needleContactMaterial.damping = 0.02f;

    RigidBodyComponent needleBody{};
    needleBody.simulated = true;
    needleBody.bodyType = RigidBodyType::Kinematic;
    needleBody.inverseMass = kNeedleInverseMass;
    needleBody.inverseInertiaLocal = needleMassProperties.inverseInertiaLocal;
    needleBody.proxyParticleLocalPositions = needleMassProperties.centeredPoints;
    needleBody.proxyParticleMaterial = needleContactMaterial;
    needleBody.proxyParticleRadius = needleParticleRadius;
    needleBody.proxyCollisionLayer = 0x4u;
    needleBody.proxyCollisionMask = 0x1u;
    needleBody.suturingEnabled = false;
    needleBody.needleTipProxyIndex = tipProxyIndex;
    needleBody.kinematicTargetEnabled = true;
    needleBody.kinematicTargetRotation = needleRotation;
    needleBody.kinematicTargetPosition = needleTransform.worldTransform.position;
    world.setRigidBody(needleEntity, needleBody);

    const auto strandEntity = world.createEntity();
    StrandComponent strand{};
    strand.particleMass = 0.12f;
    strand.particleRadius = 0.08f;
    strand.distanceCompliance = 0.000001f;
    strand.bendCompliance = 0.03f;
    strand.selfCollisionEnabled = false;
    strand.suturingEnabled = false;
    strand.needleTipKinematic = false;
    strand.collisionLayer = 0x2u;
    strand.collisionMask = 0x1u;

    const Diligent::float3 tailWorldPosition =
        needleTransform.worldTransform.position +
        needleRotation.RotateVector(needleMassProperties.centeredPoints[tailProxyIndex]);
    const Diligent::float3 tailDirectionWorld{1.0f, 0.0f, 0.0f};
    const float strandSpacing = softBody.source.regularGrid.targetParticleSpacing;
    for (std::uint32_t i = 0u; i < 18u; ++i)
    {
        const float offset = strandSpacing * static_cast<float>(i);
        strand.restPositions.push_back(tailWorldPosition - tailDirectionWorld * offset);
    }

    if (!world.setStrand(strandEntity, strand))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author attached strand.\n");
        return 1;
    }

    AuthoredParticleDistanceConstraintState needleThreadAttachment{};
    needleThreadAttachment.particleA.entityId = needleEntity;
    needleThreadAttachment.particleA.type = AuthoredParticleReferenceType::RigidProxyParticle;
    needleThreadAttachment.particleA.localParticleIndex = tailProxyIndex;
    needleThreadAttachment.particleB.entityId = strandEntity;
    needleThreadAttachment.particleB.type = AuthoredParticleReferenceType::StrandParticle;
    needleThreadAttachment.particleB.localParticleIndex = 0u;
    needleThreadAttachment.restLength = strandSpacing;
    needleThreadAttachment.compliance = 0.0f;
    world.upsertParticleDistanceConstraint(needleThreadAttachment);

    AuthoredSuturingSequenceState suturingSequence{};
    suturingSequence.pathNodeSpacing = strandSpacing;
    for (std::uint32_t proxyIndex = 0u;
         proxyIndex < static_cast<std::uint32_t>(needleProxyParticles.size()); ++proxyIndex)
    {
        suturingSequence.entries.push_back(
            {needleEntity, AuthoredParticleReferenceType::RigidProxyParticle, proxyIndex});
    }
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(strand.restPositions.size()); ++particleIndex)
    {
        suturingSequence.entries.push_back(
            {strandEntity, AuthoredParticleReferenceType::StrandParticle, particleIndex});
    }
    world.upsertSuturingSequence(suturingSequence);

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick =
        [needleEntity, computeNeedleBodyPositionFromArcCenter, computeNeedleRotation,
         startArcCenterPosition, kBaseNeedleAngle](const FrameContext &frame, Runtime &cbRuntime)
    {
        auto needleBody = cbRuntime.getWorld().tryGetRigidBody(needleEntity);
        if (!needleBody.has_value())
        {
            return;
        }

        const float t = static_cast<float>(frame.timeSeconds);
        const float cycleDuration = 16.0f;
        const float horizontalPhase = 0.34f;
        const float rotationPhase = 0.46f;
        const float pullUpPhaseStart = horizontalPhase + rotationPhase;
        const float cycle = std::fmod(std::max(t, 0.0f), cycleDuration) / cycleDuration;

        Diligent::float3 arcCenterPosition = startArcCenterPosition;
        if (cycle <= horizontalPhase)
        {
            const float u = cycle / horizontalPhase;
            arcCenterPosition.x = startArcCenterPosition.x + 1.2f * u;
        }
        else if (cycle <= pullUpPhaseStart)
        {
            arcCenterPosition.x = startArcCenterPosition.x + 1.2f;
        }
        else
        {
            arcCenterPosition.x = startArcCenterPosition.x + 1.2f;
            const float u = (cycle - pullUpPhaseStart) / (1.0f - pullUpPhaseStart);
            arcCenterPosition.y = startArcCenterPosition.y + 5.4f * u;
        }

        float needleAngle = kBaseNeedleAngle;
        if (cycle <= horizontalPhase)
        {
            needleAngle = kBaseNeedleAngle;
        }
        else if (cycle <= pullUpPhaseStart)
        {
            const float u = (cycle - horizontalPhase) / rotationPhase;
            needleAngle += kRotationStageAngleDelta * u;
        }
        else
        {
            needleAngle += kRotationStageAngleDelta;
        }
        const Diligent::QuaternionF needleRotation = computeNeedleRotation(needleAngle);

        needleBody->bodyType = RigidBodyType::Kinematic;
        needleBody->kinematicTargetEnabled = true;
        needleBody->kinematicTargetRotation = needleRotation;
        needleBody->kinematicTargetPosition =
            computeNeedleBodyPositionFromArcCenter(arcCenterPosition, needleRotation);
        cbRuntime.getWorld().setRigidBody(needleEntity, *needleBody);
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
