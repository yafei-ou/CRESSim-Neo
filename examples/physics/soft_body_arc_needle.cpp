#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ParticleContactMaterialDesc;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName, "  Debug particle rendering is enabled by default.\n", false);
}

std::vector<Diligent::float3> makeArcProxyParticles(float arcRadius, float startAngleRadians,
                                                    float endAngleRadians,
                                                    std::uint32_t sampleCount)
{
    const std::uint32_t count = std::max(sampleCount, 2u);
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
    config.physicsDesc.softContactIterations = 10u;
    config.physicsDesc.rigidRigidContactIterations = 10u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Soft Body Arc Needle";
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
        cressim::neo::examples::helpers::makePlaneMesh(12.0f, "SoftBodyArcNeedle.GroundMesh"));

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "SoftBodyArcNeedle.GroundMaterial";
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
    softTransform.worldTransform.position = {0.0f, 0.55f, 0.0f};
    softTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.2f);
    world.setTransform(softEntity, softTransform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {2.0f, 1.0f, 1.2f};
    softBody.source.regularGrid.targetParticleSpacing = 0.24f;
    softBody.particleMass = 0.08f;
    softBody.particleRadius = 0.12f;
    softBody.edgeCompliance = 0.0f;
    softBody.volumeCompliance = 0.0007f;
    softBody.selfCollisionEnabled = true;
    softBody.material.contact.friction = 0.48f;
    softBody.material.contact.staticFriction = 0.58f;
    if (!world.setSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author soft body.\n");
        return 1;
    }

    const float needleParticleRadius = 0.11f;
    const std::vector<Diligent::float3> needleProxyParticles =
        makeArcProxyParticles(0.95f, -0.9f * kPi, -0.1f * kPi, 18u);
    constexpr float kNeedleInverseMass = 1.4f;
    const ProxyMassProperties needleMassProperties =
        computeProxyMassProperties(needleProxyParticles, needleParticleRadius, kNeedleInverseMass);

    const auto needleEntity = world.createEntity();
    TransformComponent needleTransform{};
    needleTransform.worldTransform.position = {0.0f, 3.2f, 0.0f};
    needleTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, 0.3f);
    needleTransform.worldTransform.position +=
        needleTransform.worldTransform.rotation.RotateVector(needleMassProperties.centerOfMass);
    world.setTransform(needleEntity, needleTransform);

    ParticleContactMaterialDesc needleContactMaterial{};
    needleContactMaterial.friction = 0.32f;
    needleContactMaterial.staticFriction = 0.4f;
    needleContactMaterial.damping = 0.02f;

    RigidBodyComponent needleBody{};
    needleBody.simulated = true;
    needleBody.bodyType = RigidBodyType::Dynamic;
    needleBody.inverseMass = kNeedleInverseMass;
    needleBody.inverseInertiaLocal = needleMassProperties.inverseInertiaLocal;
    needleBody.proxyParticleLocalPositions = needleMassProperties.centeredPoints;
    needleBody.proxyParticleMaterial = needleContactMaterial;
    needleBody.proxyParticleRadius = needleParticleRadius;
    needleBody.proxyCollisionLayer = 0x1u;
    needleBody.proxyCollisionMask = 0xffffffffu;
    world.setRigidBody(needleEntity, needleBody);

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
