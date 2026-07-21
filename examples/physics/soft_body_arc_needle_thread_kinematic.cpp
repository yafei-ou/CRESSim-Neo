#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
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
using cressim::neo::engine::ProceduralDeformableCurveRenderComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::EnvironmentIblBakeOptions;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::AuthoredParticleDistanceConstraintState;
using cressim::neo::physics::AuthoredParticleCollisionFilterState;
using cressim::neo::physics::AuthoredParticleSequenceState;
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
constexpr const char *kSuturingSkyboxCrossPath =
    "examples/cubemaps/Cubemap/Cubemap_Sky_23-512x512.png";
constexpr float kGroundCenterY = -1.1f;
constexpr float kGroundHalfHeight = 0.08f;
constexpr Diligent::float3 kSoftBodySize = {2.0f, 1.0f, 1.8f};
constexpr float kSoftBodyParticleRadius = 0.09f;

EnvironmentIblDesc loadSuturingSkyboxIbl(cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path crossPath =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
        kSuturingSkyboxCrossPath;

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 0.18f;
    options.backgroundIntensity = 1.00f;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
}

enum class SuturingExampleMode
{
    NeedleOnly,
    NeedleThread,
    StrandOnly,
};

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

SuturingExampleMode parseMode(const std::string &value)
{
    if (value == "needle")
    {
        return SuturingExampleMode::NeedleOnly;
    }
    if (value == "thread" || value == "needle-thread" || value == "needle_thread")
    {
        return SuturingExampleMode::NeedleThread;
    }
    if (value == "strand" || value == "leader" || value == "strand-only" ||
        value == "strand_only")
    {
        return SuturingExampleMode::StrandOnly;
    }

    throw std::invalid_argument("Unsupported mode: " + value +
                                ". Expected needle, thread, or strand.");
}

const char *modeWindowTitle(SuturingExampleMode mode)
{
    switch (mode)
    {
    case SuturingExampleMode::NeedleOnly:
        return "CRESSim Neo Kinematic Arc Needle Suturing";
    case SuturingExampleMode::NeedleThread:
        return "CRESSim Neo Kinematic Arc Needle With Strand";
    case SuturingExampleMode::StrandOnly:
        return "CRESSim Neo Kinematic Suturing Strand Driver";
    }

    return "CRESSim Neo Suturing Example";
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        "  --mode needle|thread|strand  Select suturing demo variant (default: thread).\n"
        "  --debug-particles           Show debug particles instead of mesh presentation.\n",
        false);
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
    SuturingExampleMode mode = SuturingExampleMode::NeedleThread;
    bool debugParticles = false;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--mode")
            {
                if (i + 1 >= argc)
                {
                    throw std::invalid_argument("--mode requires a value.");
                }
                mode = parseMode(argv[++i]);
                continue;
            }
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
            {
                continue;
            }

            if (arg == "--debug-particles")
            {
                debugParticles = true;
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
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.physicsDesc.substeps = 1u;
    config.physicsDesc.defaultIterations = 100u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = modeWindowTitle(mode);
    viewerDefaults.showStats = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.enableDebugParticles = debugParticles;

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
    camera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(Diligent::float3{-0.4f, -1.0f, 0.3f});
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 4.8f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    if (!world.setEnvironmentIbl(0u, loadSuturingSkyboxIbl(resources)))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to assign suturing skybox IBL.\n");
        return 1;
    }

    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(12.0f, "KinematicArcNeedleThread.GroundMesh"));

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "KinematicArcNeedleThread.GroundMaterial";
    groundMaterialDesc.baseColor = {0.76f, 0.79f, 0.82f};
    groundMaterialDesc.roughness = 0.94f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    MaterialResourceDesc threadMaterialDesc{};
    threadMaterialDesc.debugName = "KinematicArcNeedleThread.ThreadMaterial";
    threadMaterialDesc.baseColor = {0.16f, 0.52f, 0.22f};
    threadMaterialDesc.roughness = 0.72f;
    threadMaterialDesc.metallic = 0.0f;
    const auto threadMaterial = resources.registerMaterial(threadMaterialDesc);

    MaterialResourceDesc softMaterialDesc{};
    softMaterialDesc.debugName = "KinematicArcNeedleThread.SoftBodyMaterial";
    softMaterialDesc.baseColor = {0.84f, 0.57f, 0.49f};
    softMaterialDesc.roughness = 0.82f;
    softMaterialDesc.metallic = 0.0f;
    const auto softMaterial = resources.registerMaterial(softMaterialDesc);

    MaterialResourceDesc needleMaterialDesc{};
    needleMaterialDesc.debugName = "KinematicArcNeedleThread.NeedleMaterial";
    needleMaterialDesc.baseColor = {0.78f, 0.78f, 0.82f};
    needleMaterialDesc.roughness = 0.22f;
    needleMaterialDesc.metallic = 0.88f;
    const auto needleMaterial = resources.registerMaterial(needleMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, kGroundCenterY, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{groundMesh, groundMaterial, true});

    RigidBodyComponent groundBody{};
    groundBody.bodyType = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);

    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {12.0f, kGroundHalfHeight, 12.0f, 0.0f};
    groundCollider.friction = 0.55f;
    groundCollider.staticFriction = 0.7f;
    world.addCollider(groundEntity, groundCollider);

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    const float groundTopY = kGroundCenterY + kGroundHalfHeight;
    softTransform.worldTransform.position = {
        0.0f, groundTopY + kSoftBodyParticleRadius + 0.5f * kSoftBodySize.y, 0.0f};
    world.setTransform(softEntity, softTransform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = kSoftBodySize;
    softBody.source.regularGrid.targetParticleSpacing = 0.24f;
    softBody.source.regularGrid.staticParticleIndices = makeRightSideStaticIndices(
        softBody.source.regularGrid.size, softBody.source.regularGrid.targetParticleSpacing);
    softBody.particleMass = 0.08f;
    softBody.particleRadius = kSoftBodyParticleRadius;
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
    if (!debugParticles)
    {
        const auto softMesh = resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
            {softBody.source.regularGrid.size.x * 0.5f, softBody.source.regularGrid.size.y * 0.5f,
             softBody.source.regularGrid.size.z * 0.5f},
            "KinematicArcNeedleThread.SoftBodyMesh"));
        world.setMeshRenderer(softEntity, MeshRendererComponent{softMesh, softMaterial, true});
    }

    const float strandSpacing = softBody.source.regularGrid.targetParticleSpacing;
    const bool strandEnabled = mode != SuturingExampleMode::NeedleOnly;
    const bool useArcNeedle = mode != SuturingExampleMode::StrandOnly;

    const auto driverEntity = world.createEntity();
    TransformComponent driverTransform{};

    ParticleContactMaterialDesc driverContactMaterial{};
    driverContactMaterial.friction = 0.32f;
    driverContactMaterial.staticFriction = 0.4f;
    driverContactMaterial.damping = 0.02f;

    RigidBodyComponent driverBody{};
    driverBody.bodyType = RigidBodyType::Kinematic;
    driverBody.proxyCollisionLayer = 0x4u;
    driverBody.proxyCollisionMask = 0x1u | 0x2u;
    driverBody.kinematicTargetEnabled = true;

    std::uint32_t tipProxyIndex = 0u;
    std::uint32_t tailProxyIndex = 0u;
    float baseNeedleAngle = 0.0f;
    Diligent::float3 startArcCenterPosition{0.0f, 0.0f, 0.0f};
    auto computeNeedleRotation = [](float angle)
    {
        return Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, angle);
    };
    std::function<Diligent::float3(const Diligent::float3 &, const Diligent::QuaternionF &)>
        computeNeedleBodyPositionFromArcCenter =
        [](const Diligent::float3 &arcCenterPosition, const Diligent::QuaternionF &)
    {
        return arcCenterPosition;
    };
    Diligent::QuaternionF initialDriverRotation{};
    Diligent::float3 strandAnchorWorldPosition{0.0f, 0.0f, 0.0f};

    if (useArcNeedle)
    {
        const float needleParticleRadius = 0.1f;
        const std::vector<Diligent::float3> needleProxyParticles =
            makeArcProxyParticles(0.95f, -0.1f * kPi, -0.9f * kPi, 10u);
        constexpr float kNeedleInverseMass = 1.4f;
        const ProxyMassProperties needleMassProperties = computeProxyMassProperties(
            needleProxyParticles, needleParticleRadius, kNeedleInverseMass);
        tipProxyIndex = 0u;
        tailProxyIndex =
            static_cast<std::uint32_t>(needleMassProperties.centeredPoints.size() - 1u);

        const Diligent::float3 tipTangentLocal =
            needleProxyParticles.size() >= 2u
                ? Diligent::normalize(needleProxyParticles[tipProxyIndex] -
                                      needleProxyParticles[tipProxyIndex + 1u])
                : Diligent::float3{1.0f, 0.0f, 0.0f};
        const Diligent::float3 desiredTipTangentWorld =
            Diligent::normalize(Diligent::float3{1.0f, 0.12f, 0.0f});

        baseNeedleAngle = signedAngleXY(tipTangentLocal, desiredTipTangentWorld);
        const Diligent::float3 arcCenterLocal = -needleMassProperties.centerOfMass;
        const Diligent::float3 tipFromArcCenterLocal = needleProxyParticles[tipProxyIndex];
        computeNeedleBodyPositionFromArcCenter =
            [arcCenterLocal](const Diligent::float3 &arcCenterPosition,
                             const Diligent::QuaternionF &needleRotation)
        {
            return arcCenterPosition - needleRotation.RotateVector(arcCenterLocal);
        };

        initialDriverRotation = computeNeedleRotation(baseNeedleAngle);
        const Diligent::float3 startTipPosition{-1.6f, -0.24f, 0.0f};
        startArcCenterPosition =
            startTipPosition - initialDriverRotation.RotateVector(tipFromArcCenterLocal);
        driverTransform.worldTransform.rotation = initialDriverRotation;
        driverTransform.worldTransform.position =
            computeNeedleBodyPositionFromArcCenter(startArcCenterPosition, initialDriverRotation);

        driverBody.inverseMass = kNeedleInverseMass;
        driverBody.inverseInertiaLocal = needleMassProperties.inverseInertiaLocal;
        driverBody.proxyParticleLocalPositions = needleMassProperties.centeredPoints;
        driverBody.proxyParticleMaterial = driverContactMaterial;
        driverBody.proxyParticleRadius = needleParticleRadius;
        driverBody.needleTipProxyIndex = tipProxyIndex;
        driverBody.kinematicTargetRotation = initialDriverRotation;
        driverBody.kinematicTargetPosition = driverTransform.worldTransform.position;
        strandAnchorWorldPosition =
            driverTransform.worldTransform.position +
            initialDriverRotation.RotateVector(needleMassProperties.centeredPoints[tailProxyIndex]);

        if (!debugParticles)
        {
            const auto needleMesh = resources.registerMesh(
                cressim::neo::examples::helpers::makePolylineTubeMesh(
                    needleMassProperties.centeredPoints, 12u, 0.085f,
                    "KinematicArcNeedleThread.NeedleMesh", 3.5f, 0.18f, 0.78f));
            world.setMeshRenderer(driverEntity,
                                  MeshRendererComponent{needleMesh, needleMaterial, true});
        }
    }
    else
    {
        constexpr float kStrandOnlyBaseY = -0.5f;
        tipProxyIndex = 0u;
        tailProxyIndex = 0u;
        driverTransform.worldTransform.position = {-1.8f, kStrandOnlyBaseY, 0.0f};
        driverTransform.worldTransform.rotation = computeNeedleRotation(0.0f);
        driverBody.inverseMass = 1.0f;
        driverBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
        driverBody.proxyParticleLocalPositions = {{0.0f, 0.0f, 0.0f}};
        driverBody.proxyParticleMaterial = driverContactMaterial;
        driverBody.proxyParticleRadius = 0.1f;
        driverBody.needleTipProxyIndex = 0u;
        driverBody.kinematicTargetRotation = driverTransform.worldTransform.rotation;
        driverBody.kinematicTargetPosition = driverTransform.worldTransform.position;
        strandAnchorWorldPosition = driverTransform.worldTransform.position;

        if (!debugParticles)
        {
            const auto leaderMesh = resources.registerMesh(
                cressim::neo::examples::helpers::makeCapsuleMesh(
                    0.085f, 0.12f, 16u, 8u, 2u, "KinematicArcNeedleThread.LeaderMesh"));
            world.setMeshRenderer(driverEntity,
                                  MeshRendererComponent{leaderMesh, needleMaterial, true});
        }
    }

    world.setTransform(driverEntity, driverTransform);
    world.setRigidBody(driverEntity, driverBody);

    const float driverAttachmentRestLength = 0.0f;
    std::uint32_t strandParticleCount = 0u;
    cressim::neo::common::EntityId strandEntity = cressim::neo::common::kInvalidEntityId;
    if (strandEnabled)
    {
        strandEntity = world.createEntity();
        world.setTransform(strandEntity, TransformComponent{});
        StrandComponent strand{};
        strand.particleMass = 0.12f;
        strand.particleRadius = 0.1f;
        strand.stretchShearCompliance = 0.000001f;
        strand.bendCompliance = 0.005f;
        strand.selfCollisionEnabled = false;
        strand.suturingEnabled = false;
        strand.collisionLayer = 0x2u;
        strand.collisionMask = 0x1u | 0x4u;

        const Diligent::float3 strandDirectionWorld{1.0f, 0.0f, 0.0f};
        const std::uint32_t count = 18u;
        for (std::uint32_t i = 0u; i < count; ++i)
        {
            const float offset =
                driverAttachmentRestLength + strandSpacing * static_cast<float>(i);
            strand.restPositions.push_back(strandAnchorWorldPosition - strandDirectionWorld * offset);
        }
        strandParticleCount = static_cast<std::uint32_t>(strand.restPositions.size());

        if (!world.setStrand(strandEntity, strand))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Failed to author attached strand.\n");
            return 1;
        }

        AuthoredParticleDistanceConstraintState driverAttachment{};
        driverAttachment.particleA.entityId = driverEntity;
        driverAttachment.particleA.type = AuthoredParticleReferenceType::RigidProxyParticle;
        driverAttachment.particleA.localParticleIndex = tailProxyIndex;
        driverAttachment.particleB.entityId = strandEntity;
        driverAttachment.particleB.type = AuthoredParticleReferenceType::StrandParticle;
        driverAttachment.particleB.localParticleIndex = 0u;
        driverAttachment.restLength = driverAttachmentRestLength;
        driverAttachment.compliance = 0.0f;
        world.upsertParticleDistanceConstraint(driverAttachment);

        AuthoredParticleCollisionFilterState tailAnchorFilter{};
        tailAnchorFilter.particle.entityId = driverEntity;
        tailAnchorFilter.particle.type = AuthoredParticleReferenceType::RigidProxyParticle;
        tailAnchorFilter.particle.localParticleIndex = tailProxyIndex;
        tailAnchorFilter.collisionLayer = driverBody.proxyCollisionLayer;
        tailAnchorFilter.collisionMask = 0u;
        world.upsertParticleCollisionFilter(tailAnchorFilter);

        AuthoredSuturingSequenceState suturingSequence{};
        suturingSequence.pathNodeSpacing = strandSpacing;
        for (std::uint32_t proxyIndex = 0u;
             proxyIndex < static_cast<std::uint32_t>(driverBody.proxyParticleLocalPositions.size());
             ++proxyIndex)
        {
            if (proxyIndex == tailProxyIndex)
            {
                continue;
            }
            suturingSequence.entries.push_back(
                {driverEntity, AuthoredParticleReferenceType::RigidProxyParticle, proxyIndex});
        }
        for (std::uint32_t particleIndex = 0u; particleIndex < strandParticleCount; ++particleIndex)
        {
            suturingSequence.entries.push_back(
                {strandEntity, AuthoredParticleReferenceType::StrandParticle, particleIndex});
        }
        world.upsertSuturingSequence(suturingSequence);

        AuthoredParticleSequenceState renderSequence{};
        for (std::uint32_t particleIndex = 0u; particleIndex < strandParticleCount; ++particleIndex)
        {
            renderSequence.entries.push_back(
                {strandEntity, AuthoredParticleReferenceType::StrandParticle, particleIndex});
        }
        auto &authoredRenderSequence = world.upsertParticleSequence(renderSequence);

        if (!debugParticles)
        {
            constexpr std::uint32_t kThreadRadialResolution = 10u;
            const auto threadMesh = resources.registerMesh(
                cressim::neo::examples::helpers::makeCanonicalCurveTubeMesh(
                    strandParticleCount, kThreadRadialResolution,
                    "KinematicArcNeedleThread.ThreadCurveMesh", 4.0f));
            world.setMeshRenderer(strandEntity,
                                  MeshRendererComponent{threadMesh, threadMaterial, true});
            world.setProceduralDeformableCurveRender(
                strandEntity,
                ProceduralDeformableCurveRenderComponent{
                    authoredRenderSequence.sequenceId,
                    0.05f,
                    kThreadRadialResolution,
                    true,
                });
        }
    }
    else
    {
        AuthoredSuturingSequenceState suturingSequence{};
        suturingSequence.pathNodeSpacing = strandSpacing;
        for (std::uint32_t proxyIndex = 0u;
             proxyIndex < static_cast<std::uint32_t>(driverBody.proxyParticleLocalPositions.size());
             ++proxyIndex)
        {
            suturingSequence.entries.push_back(
                {driverEntity, AuthoredParticleReferenceType::RigidProxyParticle, proxyIndex});
        }
        world.upsertSuturingSequence(suturingSequence);
    }

    DebugViewerCallbacks callbacks{};
    if (useArcNeedle)
    {
        callbacks.beforeTick =
            [driverEntity, computeNeedleBodyPositionFromArcCenter, computeNeedleRotation,
             startArcCenterPosition, baseNeedleAngle](const FrameContext &frame, Runtime &cbRuntime)
        {
            auto driverBody = cbRuntime.getWorld().tryGetRigidBody(driverEntity);
            if (!driverBody.has_value())
            {
                return;
            }

            const float t = static_cast<float>(frame.timeSeconds);
            const float horizontalDuration = 5.44f;
            const float rotationDuration = 7.36f;
            const float pullUpDuration = 6.0f;
            const float cycleDuration = horizontalDuration + rotationDuration + pullUpDuration;
            const float cycleTime = std::fmod(std::max(t, 0.0f), cycleDuration);
            const float rotationPhaseStart = horizontalDuration;
            const float pullUpPhaseStart = horizontalDuration + rotationDuration;

            Diligent::float3 arcCenterPosition = startArcCenterPosition;
            if (cycleTime <= rotationPhaseStart)
            {
                const float u = cycleTime / horizontalDuration;
                arcCenterPosition.x = startArcCenterPosition.x + 1.2f * u;
            }
            else if (cycleTime <= pullUpPhaseStart)
            {
                arcCenterPosition.x = startArcCenterPosition.x + 1.2f;
            }
            else
            {
                arcCenterPosition.x = startArcCenterPosition.x + 1.2f;
                const float u = (cycleTime - pullUpPhaseStart) / pullUpDuration;
                arcCenterPosition.y = startArcCenterPosition.y + 5.4f * u;
            }

            float needleAngle = baseNeedleAngle;
            if (cycleTime <= rotationPhaseStart)
            {
                needleAngle = baseNeedleAngle;
            }
            else if (cycleTime <= pullUpPhaseStart)
            {
                const float u = (cycleTime - rotationPhaseStart) / rotationDuration;
                needleAngle += 3.35f * u;
            }
            else
            {
                needleAngle += 3.35f;
            }
            const Diligent::QuaternionF needleRotation = computeNeedleRotation(needleAngle);

            driverBody->bodyType = RigidBodyType::Kinematic;
            driverBody->kinematicTargetEnabled = true;
            driverBody->kinematicTargetRotation = needleRotation;
            driverBody->kinematicTargetPosition =
                computeNeedleBodyPositionFromArcCenter(arcCenterPosition, needleRotation);
            cbRuntime.getWorld().setRigidBody(driverEntity, *driverBody);
        };
    }
    else
    {
        callbacks.beforeTick = [driverEntity](const FrameContext &frame, Runtime &cbRuntime)
        {
            auto driverBody = cbRuntime.getWorld().tryGetRigidBody(driverEntity);
            if (!driverBody.has_value())
            {
                return;
            }

            const float t = static_cast<float>(frame.timeSeconds);
            const float cycleDuration = 12.0f;
            const float horizontalPhase = 0.35f;
            const float liftPhase = 0.30f;
            const float leftPullPhase = 1.0f - horizontalPhase - liftPhase;
            const float startX = -1.8f;
            const float exitX = -0.1f;
            const float endX = -2.1f;
            const float baseY = -0.5f;
            const float liftedY = 0.90f;
            const float cycle = std::fmod(std::max(t, 0.0f), cycleDuration) / cycleDuration;
            float x = startX;
            float y = baseY;
            if (cycle <= horizontalPhase)
            {
                const float u = cycle / horizontalPhase;
                x = startX + (exitX - startX) * u;
            }
            else if (cycle <= horizontalPhase + liftPhase)
            {
                const float u = (cycle - horizontalPhase) / liftPhase;
                const float easedU = u * u * (3.0f - 2.0f * u);
                x = exitX;
                y = baseY + (liftedY - baseY) * easedU;
            }
            else
            {
                const float u = (cycle - horizontalPhase - liftPhase) / leftPullPhase;
                const float easedU = u * u * (3.0f - 2.0f * u);
                x = exitX + (endX - exitX) * easedU;
                y = liftedY;
            }

            driverBody->bodyType = RigidBodyType::Kinematic;
            driverBody->kinematicTargetEnabled = true;
            driverBody->kinematicTargetRotation =
                Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, 0.0f);
            driverBody->kinematicTargetPosition = Diligent::float3{x, y, 0.0f};
            cbRuntime.getWorld().setRigidBody(driverEntity, *driverBody);
        };
    }

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
