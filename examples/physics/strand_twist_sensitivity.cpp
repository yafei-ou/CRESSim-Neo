#include "common/frame_context.h"
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
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::ProceduralDeformableCurveRenderComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::AuthoredParticleReference;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredParticleSequenceState;
using cressim::neo::physics::AuthoredStrandRigidAttachmentConstraintState;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;

struct ExampleOptions
{
    std::uint32_t particleCount    = 14u;
    float particleSpacing          = 0.42f;
    float drivePeriodSeconds       = 6.5f;
    float twistAmplitudeDegrees    = 320.0f;
    float stiffTwistCompliance     = 2.0e-6f;
    float softTwistCompliance      = 2.5e-1f;
    float bendCompliance           = 5.0e-5f;
    float stretchShearCompliance   = 2.0e-6f;
};

struct RodRig
{
    EntityId strandEntity                     = 0u;
    EntityId tipHandleEntity                  = 0u;
    Diligent::float3 tipRest{};
    Diligent::QuaternionF restSegmentRotation{};
    float phaseOffset                         = 0.0f;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--particles N] [--spacing VALUE] [--period VALUE] [--twist-amplitude VALUE]"
        " [--stiff-twist VALUE] [--soft-twist VALUE]",
        false);
}

std::uint32_t parseUIntOption(const std::string &value, const char *optionName)
{
    const char *begin = value.c_str();
    char *end         = nullptr;
    const auto parsed = std::strtoul(begin, &end, 10);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parseFloatOption(const std::string &value, const char *optionName)
{
    const char *begin  = value.c_str();
    char *end          = nullptr;
    const float parsed = std::strtof(begin, &end);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return parsed;
}

bool tryParseSceneArgument(int argc, char **argv, int &index, ExampleOptions &options)
{
    const std::string arg = argv[index];
    if (arg == "--particles")
    {
        options.particleCount = parseUIntOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--particles"),
            "particle count");
        if (options.particleCount < 4u)
        {
            throw std::invalid_argument("--particles must be at least 4.");
        }
        return true;
    }
    if (arg == "--spacing")
    {
        options.particleSpacing = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--spacing"),
            "particle spacing");
        if (options.particleSpacing <= 0.0f)
        {
            throw std::invalid_argument("--spacing must be greater than zero.");
        }
        return true;
    }
    if (arg == "--period")
    {
        options.drivePeriodSeconds = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--period"),
            "drive period");
        if (options.drivePeriodSeconds <= 0.0f)
        {
            throw std::invalid_argument("--period must be greater than zero.");
        }
        return true;
    }
    if (arg == "--twist-amplitude")
    {
        options.twistAmplitudeDegrees = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index,
                                                                 "--twist-amplitude"),
            "twist amplitude");
        if (options.twistAmplitudeDegrees < 0.0f)
        {
            throw std::invalid_argument("--twist-amplitude must be non-negative.");
        }
        return true;
    }
    if (arg == "--stiff-twist")
    {
        options.stiffTwistCompliance = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index,
                                                                 "--stiff-twist"),
            "stiff twist compliance");
        if (options.stiffTwistCompliance < 0.0f)
        {
            throw std::invalid_argument("--stiff-twist must be non-negative.");
        }
        return true;
    }
    if (arg == "--soft-twist")
    {
        options.softTwistCompliance = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index,
                                                                 "--soft-twist"),
            "soft twist compliance");
        if (options.softTwistCompliance < 0.0f)
        {
            throw std::invalid_argument("--soft-twist must be non-negative.");
        }
        return true;
    }

    return false;
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic  = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

float degreesToRadians(float degrees) noexcept
{
    return degrees * (kPi / 180.0f);
}

Diligent::float3 safeNormalize(const Diligent::float3 &value,
                               const Diligent::float3 &fallback) noexcept
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= 1.0e-8f)
    {
        return fallback;
    }
    return value * (1.0f / std::sqrt(lengthSq));
}

Diligent::QuaternionF normalizeQuaternion(const Diligent::QuaternionF &value) noexcept
{
    const float lengthSq = Diligent::dot(value.q, value.q);
    if (lengthSq <= 1.0e-8f)
    {
        return Diligent::QuaternionF{};
    }
    return Diligent::normalize(value);
}

void setKinematicHandle(Runtime &runtime, EntityId entity, const Diligent::float3 &position,
                        const Diligent::QuaternionF &rotation, MaterialHandle material,
                        cressim::neo::graphics::MeshHandle mesh, const Diligent::float3 &scale)
{
    auto &world = runtime.getWorld();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation = rotation;
    transform.worldTransform.scale    = scale;
    world.setTransform(entity, transform);
    world.setMeshRenderer(entity, MeshRendererComponent{mesh, material, true});

    RigidBodyComponent body{};
    body.bodyType                = RigidBodyType::Kinematic;
    body.inverseMass             = 0.0f;
    body.kinematicTargetEnabled  = true;
    body.kinematicTargetPosition = position;
    body.kinematicTargetRotation = rotation;
    world.setRigidBody(entity, body);
}

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &xAxis,
                                          const Diligent::float3 &yAxis,
                                          const Diligent::float3 &zAxis) noexcept
{
    const float m00 = xAxis.x;
    const float m01 = yAxis.x;
    const float m02 = zAxis.x;
    const float m10 = xAxis.y;
    const float m11 = yAxis.y;
    const float m12 = zAxis.y;
    const float m20 = xAxis.z;
    const float m21 = yAxis.z;
    const float m22 = zAxis.z;
    const float trace = m00 + m11 + m22;

    Diligent::QuaternionF result{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        result.q.w    = 0.25f * s;
        result.q.x    = (m21 - m12) / s;
        result.q.y    = (m02 - m20) / s;
        result.q.z    = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        result.q.w    = (m21 - m12) / s;
        result.q.x    = 0.25f * s;
        result.q.y    = (m01 + m10) / s;
        result.q.z    = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        result.q.w    = (m02 - m20) / s;
        result.q.x    = (m01 + m10) / s;
        result.q.y    = 0.25f * s;
        result.q.z    = (m12 + m21) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        result.q.w    = (m10 - m01) / s;
        result.q.x    = (m02 + m20) / s;
        result.q.y    = (m12 + m21) / s;
        result.q.z    = 0.25f * s;
    }

    return normalizeQuaternion(result);
}

Diligent::float3 defaultRootMaterialNormal(const Diligent::float3 &segmentDirection,
                                           const Diligent::float3 &authoredNormal) noexcept
{
    const Diligent::float3 direction = safeNormalize(segmentDirection, {1.0f, 0.0f, 0.0f});
    Diligent::float3 normal          = authoredNormal;
    if (Diligent::dot(normal, normal) <= 1.0e-8f)
    {
        normal = {0.0f, 1.0f, 0.0f};
    }

    normal = normal - direction * Diligent::dot(normal, direction);
    if (Diligent::dot(normal, normal) <= 1.0e-8f)
    {
        Diligent::float3 fallback{0.0f, 1.0f, 0.0f};
        if (std::abs(Diligent::dot(direction, fallback)) > 0.9f)
        {
            fallback = {1.0f, 0.0f, 0.0f};
        }
        normal = fallback - direction * Diligent::dot(fallback, direction);
    }

    return safeNormalize(normal, {0.0f, 1.0f, 0.0f});
}

Diligent::QuaternionF segmentOrientationFromRest(const Diligent::float3 &segmentStart,
                                                 const Diligent::float3 &segmentEnd,
                                                 const Diligent::float3 &rootMaterialNormal) noexcept
{
    const Diligent::float3 tangent =
        safeNormalize(segmentEnd - segmentStart, Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 normal = defaultRootMaterialNormal(tangent, rootMaterialNormal);
    const Diligent::float3 yAxis =
        safeNormalize(normal - tangent * Diligent::dot(normal, tangent),
                      defaultRootMaterialNormal(tangent, normal));
    const Diligent::float3 zAxis =
        safeNormalize(Diligent::cross(tangent, yAxis), Diligent::float3{0.0f, 0.0f, 1.0f});
    return quaternionFromBasis(tangent, yAxis, zAxis);
}

// This helper uses the strand rigid-attachment path as a rigid-driven follower:
// the handle drives the strand segment pose, but the strand does not push back
// physically onto the handle.
void authorRigidDrivenSegmentFollow(Runtime &runtime, EntityId strandEntity,
                                    std::uint32_t localSegmentIndex, float segmentT,
                                    EntityId rigidEntity, float translationCompliance,
                                    float rotationCompliance)
{
    AuthoredStrandRigidAttachmentConstraintState attachment{};
    attachment.strandEntityId         = strandEntity;
    attachment.localSegmentIndex      = localSegmentIndex;
    attachment.segmentT               = segmentT;
    attachment.rigidBodyEntityId      = rigidEntity;
    attachment.translationCompliance  = translationCompliance;
    attachment.rotationCompliance     = rotationCompliance;
    attachment.enabled                = true;
    runtime.getWorld().upsertStrandRigidAttachmentConstraint(attachment);
}

void setKinematicHandle(Runtime &runtime, EntityId entity, const Diligent::float3 &position,
                        MaterialHandle material, cressim::neo::graphics::MeshHandle mesh,
                        const Diligent::float3 &scale)
{
    setKinematicHandle(runtime, entity, position, Diligent::QuaternionF{}, material, mesh, scale);
}

RodRig authorRodRig(Runtime &runtime, const ExampleOptions &options, float xOffset,
                    float twistCompliance, float phaseOffset, MaterialHandle rodMaterial,
                    MaterialHandle tipHandleMaterial,
                    cressim::neo::graphics::MeshHandle handleMesh,
                    cressim::neo::graphics::MeshHandle strandMesh)
{
    auto &world = runtime.getWorld();

    RodRig rig{};
    rig.strandEntity    = world.createEntity();
    rig.tipHandleEntity = world.createEntity();
    rig.phaseOffset     = phaseOffset;

    StrandComponent strand{};
    strand.restPositions.reserve(options.particleCount);
    strand.staticParticleIndices = {0u};
    strand.particleMass          = 0.18f;
    strand.particleRadius        = 0.065f;
    strand.stretchShearCompliance = options.stretchShearCompliance;
    strand.bendCompliance         = options.bendCompliance;
    strand.twistCompliance        = twistCompliance;
    strand.distanceCompliance     = 0.0f;
    strand.rootMaterialNormal     = {0.0f, 0.0f, 1.0f};
    strand.selfCollisionEnabled   = false;
    strand.collisionLayer         = 0u;
    strand.collisionMask          = 0u;

    const Diligent::float3 base{xOffset, -2.2f, 0.0f};
    for (std::uint32_t i = 0u; i < options.particleCount; ++i)
    {
        strand.restPositions.push_back(
            Diligent::float3{xOffset, base.y + static_cast<float>(i) * options.particleSpacing, 0.0f});
    }

    rig.tipRest = strand.restPositions[options.particleCount - 1u];
    rig.restSegmentRotation =
        segmentOrientationFromRest(strand.restPositions[options.particleCount - 2u],
                                   strand.restPositions[options.particleCount - 1u],
                                   strand.rootMaterialNormal);

    if (!world.setStrand(rig.strandEntity, strand))
    {
        throw std::runtime_error("Failed to author strand twist sensitivity rig.");
    }

    AuthoredParticleSequenceState sequence{};
    sequence.entries.reserve(options.particleCount);
    for (std::uint32_t i = 0u; i < options.particleCount; ++i)
    {
        sequence.entries.push_back(
            AuthoredParticleReference{rig.strandEntity, AuthoredParticleReferenceType::StrandParticle,
                                      i});
    }
    sequence = world.upsertParticleSequence(sequence);
    world.setMeshRenderer(rig.strandEntity, MeshRendererComponent{strandMesh, rodMaterial, true});
    world.setProceduralDeformableCurveRender(
        rig.strandEntity, ProceduralDeformableCurveRenderComponent{sequence.sequenceId, 0.085f, 12u,
                                                                   true});

    setKinematicHandle(runtime, rig.tipHandleEntity, rig.tipRest, rig.restSegmentRotation,
                       tipHandleMaterial, handleMesh, {0.22f, 0.045f, 0.11f});
    authorRigidDrivenSegmentFollow(runtime, rig.strandEntity, options.particleCount - 2u, 1.0f,
                                   rig.tipHandleEntity, 0.0f, 0.0f);

    return rig;
}

Diligent::QuaternionF drivenTipRotation(const RodRig &rig, float timeSeconds,
                                        const ExampleOptions &options) noexcept
{
    const float settle = std::clamp((timeSeconds - 0.6f) / 1.2f, 0.0f, 1.0f);
    const float omega  = (2.0f * kPi) / std::max(options.drivePeriodSeconds, 0.1f);
    const float phase  = omega * timeSeconds + rig.phaseOffset;
    const float angleRadians = degreesToRadians(options.twistAmplitudeDegrees) * settle *
                               std::sin(phase);
    return rig.restSegmentRotation *
           Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, angleRadians);
}

void updateHandle(Runtime &runtime, EntityId entity, const Diligent::float3 &position,
                  const Diligent::QuaternionF &rotation, const Diligent::float3 &scale)
{
    auto &world = runtime.getWorld();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation = rotation;
    transform.worldTransform.scale    = scale;
    world.setTransform(entity, transform);

    const std::optional<RigidBodyComponent> current = world.tryGetRigidBody(entity);
    if (!current.has_value())
    {
        return;
    }

    RigidBodyComponent updated = *current;
    updated.bodyType                = RigidBodyType::Kinematic;
    updated.kinematicTargetEnabled  = true;
    updated.kinematicTargetPosition = position;
    updated.kinematicTargetRotation = rotation;
    world.setRigidBody(entity, updated);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    ExampleOptions sceneOptions{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options,
                                                                        false))
            {
                continue;
            }
            if (tryParseSceneArgument(argc, argv, i, sceneOptions))
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
    config.physicsDesc.defaultIterations = 48u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Strand Twist Sensitivity";
    viewerDefaults.width       = 1360u;
    viewerDefaults.height      = 820u;
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);

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

    auto &world     = runtime.getWorld();
    auto &resources = runtime.getResources();

    const auto handleMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh(
            {1.0f, 1.0f, 1.0f}, "PhysicsStrandTwistSensitivity.HandleMesh"));
    const auto strandMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCanonicalCurveTubeMesh(
            sceneOptions.particleCount, 12u, "PhysicsStrandTwistSensitivity.StrandMesh", 4.0f));

    const auto stiffRodMaterial =
        registerMaterial(resources, "PhysicsStrandTwistSensitivity.StiffRod", {0.16f, 0.76f, 0.44f},
                         0.34f);
    const auto softRodMaterial =
        registerMaterial(resources, "PhysicsStrandTwistSensitivity.SoftRod", {0.92f, 0.54f, 0.20f},
                         0.36f);
    const auto stiffTipMaterial =
        registerMaterial(resources, "PhysicsStrandTwistSensitivity.StiffTip", {0.90f, 0.98f, 0.94f},
                         0.22f);
    const auto softTipMaterial =
        registerMaterial(resources, "PhysicsStrandTwistSensitivity.SoftTip", {0.99f, 0.94f, 0.84f},
                         0.22f);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 1.25f, -10.8f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.4f, -1.0f, 0.25f};
    light.color     = {1.0f, 0.98f, 0.95f};
    light.intensity = 8.5f;
    world.setDirectionalLight(lightEntity, light);

    const RodRig stiffRig =
        authorRodRig(runtime, sceneOptions, -1.8f, sceneOptions.stiffTwistCompliance, 0.0f,
                     stiffRodMaterial, stiffTipMaterial, handleMesh, strandMesh);
    const RodRig softRig =
        authorRodRig(runtime, sceneOptions, 1.8f, sceneOptions.softTwistCompliance, 0.0f,
                     softRodMaterial, softTipMaterial, handleMesh, strandMesh);

    auto renderOptions                         = runtime.renderFrameOptions();
    renderOptions.debugStrandFrames.enabled    = true;
    renderOptions.debugStrandFrames.axisLength = 0.24f;
    renderOptions.debugStrandFrames.thickness  = 0.018f;
    renderOptions.debugStrandFrames.opacity    = 0.95f;
    runtime.setRenderFrameOptions(renderOptions);

    CRESSIM_LOG_INFO(
        "Twist sensitivity example loaded.\n"
        "Left rod: stiff twist compliance=", sceneOptions.stiffTwistCompliance,
        " Right rod: soft twist compliance=", sceneOptions.softTwistCompliance, "\n"
        "Expected behavior: the bright top handle twists the tip segment directly, and the debug "
        "RGB frame axes show the actual GPU segment orientations along each strand. The left rod "
        "should keep the colored frames rotating more coherently down the rod, while the right rod "
        "should localize more of the twist near the driven tip.\n");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick =
        [sceneOptions, stiffRig, softRig](const FrameContext &frame, Runtime &cbRuntime)
    {
        const float t = static_cast<float>(frame.timeSeconds);
        updateHandle(cbRuntime, stiffRig.tipHandleEntity, stiffRig.tipRest,
                     drivenTipRotation(stiffRig, t, sceneOptions), {0.22f, 0.045f, 0.11f});
        updateHandle(cbRuntime, softRig.tipHandleEntity, softRig.tipRest,
                     drivenTipRotation(softRig, t, sceneOptions), {0.22f, 0.045f, 0.11f});

        if (frame.frameIndex > 0u && frame.frameIndex % 300u == 0u)
        {
            CRESSIM_LOG_INFO("Frame ", frame.frameIndex,
                             ": compare left (stiff twist) vs right (soft twist).");
        }
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk     = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();
    return runOk ? 0 : 1;
}
