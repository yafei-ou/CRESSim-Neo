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

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
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
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::AuthoredParticleReference;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredParticleSequenceState;
using cressim::neo::physics::AuthoredRigidParticleAttachmentConstraintState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

// GLFW-compatible key codes. The viewer deliberately keeps GLFW out of its public API.
constexpr int kKeyA          = 65;
constexpr int kKeyD          = 68;
constexpr int kKeyE          = 69;
constexpr int kKeyQ          = 81;
constexpr int kKeyS          = 83;
constexpr int kKeyW          = 87;
constexpr int kKeyRight      = 262;
constexpr int kKeyLeft       = 263;
constexpr int kKeyDown       = 264;
constexpr int kKeyUp         = 265;
constexpr int kKeyPageUp     = 266;
constexpr int kKeyPageDown   = 267;
constexpr std::uint32_t kParticleCount = 33u;
constexpr float kParticleSpacing       = 0.14f;
constexpr float kParticleRadius        = 0.055f;
constexpr float kGapDemoParticleRadius = 0.025f;
constexpr float kHandleHeight          = 0.075f;
constexpr float kHandleSpeed           = 1.5f;
constexpr float kPlaneLimit            = 3.65f;
constexpr float kHeightLimit           = 3.75f;

struct InteractiveState
{
    EntityId leftHandle  = cressim::neo::common::kInvalidEntityId;
    EntityId rightHandle = cressim::neo::common::kInvalidEntityId;
    Diligent::float3 leftPosition{};
    Diligent::float3 rightPosition{};
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--capsule-self-collision] [--show-particle-gaps]"
        " [--particle-radius R] [--particle-spacing S]"
        " [--contact-slop S]",
        false);
}

float parseNonNegativeFloat(const char *value, const char *optionName, bool allowZero)
{
    char *end         = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) ||
        (allowZero ? parsed < 0.0f : parsed <= 0.0f))
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return parsed;
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

EntityId authorKinematicHandle(Runtime &runtime, const Diligent::float3 &position,
                               MeshHandle mesh, MaterialHandle material)
{
    auto &world          = runtime.getWorld();
    const EntityId entity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);
    world.setMeshRenderer(entity, MeshRendererComponent{mesh, material, true});

    RigidBodyComponent body{};
    body.bodyType                = RigidBodyType::Kinematic;
    body.inverseMass             = 0.0f;
    body.inverseInertiaLocal     = {0.0f, 0.0f, 0.0f};
    body.kinematicTargetEnabled  = true;
    body.kinematicTargetPosition = position;
    world.setRigidBody(entity, body);
    return entity;
}

void updateKinematicHandle(Runtime &runtime, EntityId entity, const Diligent::float3 &position)
{
    auto &world = runtime.getWorld();

    TransformComponent transform = world.tryGetTransform(entity).value_or(TransformComponent{});
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    RigidBodyComponent body = world.tryGetRigidBody(entity).value_or(RigidBodyComponent{});
    body.bodyType                = RigidBodyType::Kinematic;
    body.kinematicTargetEnabled  = true;
    body.kinematicTargetPosition = position;
    body.kinematicTargetRotation = transform.worldTransform.rotation;
    world.setRigidBody(entity, body);
}

void attachParticle(Runtime &runtime, EntityId strandEntity, std::uint32_t particleIndex,
                    EntityId handleEntity)
{
    AuthoredRigidParticleAttachmentConstraintState attachment{};
    attachment.particle.entityId           = strandEntity;
    attachment.particle.type               = AuthoredParticleReferenceType::StrandParticle;
    attachment.particle.localParticleIndex = particleIndex;
    attachment.rigidBodyEntityId            = handleEntity;
    attachment.compliance                   = 0.0f;
    attachment.enabled                      = true;
    if (!runtime.getWorld().upsertRigidParticleAttachmentConstraint(attachment))
    {
        throw std::runtime_error("Failed to attach a rope endpoint to its handle.");
    }
}

void authorGround(Runtime &runtime, MeshHandle mesh, MaterialHandle material)
{
    auto &world           = runtime.getWorld();
    const EntityId entity = world.createEntity();

    TransformComponent transform{};
    world.setTransform(entity, transform);
    world.setMeshRenderer(entity, MeshRendererComponent{mesh, material, true});

    RigidBodyComponent body{};
    body.bodyType            = RigidBodyType::Static;
    body.inverseMass         = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType      = ColliderShapeType::Box;
    collider.shapeParams    = {4.0f, 0.08f, 4.0f, 0.0f};
    collider.localPosition  = {0.0f, -0.08f, 0.0f};
    collider.friction       = 0.75f;
    collider.staticFriction = 0.95f;
    world.addCollider(entity, collider);
}

InteractiveState authorRope(Runtime &runtime, MeshHandle strandMesh, MeshHandle handleMesh,
                            MaterialHandle strandMaterial, MaterialHandle leftHandleMaterial,
                            MaterialHandle rightHandleMaterial, bool showParticleGaps,
                            bool capsuleSelfCollision, float particleSpacing,
                            float particleRadius)
{
    auto &world                 = runtime.getWorld();
    const EntityId strandEntity = world.createEntity();

    StrandComponent strand{};
    strand.restPositions.reserve(kParticleCount);
    const float halfLength = 0.5f * static_cast<float>(kParticleCount - 1u) * particleSpacing;
    for (std::uint32_t i = 0u; i < kParticleCount; ++i)
    {
        strand.restPositions.push_back(
            {-halfLength + static_cast<float>(i) * particleSpacing, kHandleHeight, 0.0f});
    }
    strand.particleMass           = 0.025f;
    strand.particleRadius         = particleRadius;
    strand.stretchShearCompliance = 2.0e-6f;
    strand.bendCompliance         = 2.0e-4f;
    strand.twistCompliance        = 1.0e-3f;
    strand.distanceCompliance     = 0.0f;
    strand.rootMaterialNormal     = {0.0f, 1.0f, 0.0f};
    strand.material.contact.friction       = 0.72f;
    strand.material.contact.staticFriction = 0.92f;
    strand.material.contact.damping        = 0.08f;
    // The experimental capsule pass replaces node-sphere self-collision to avoid double solving.
    strand.selfCollisionEnabled = !capsuleSelfCollision;

    if (!world.setStrand(strandEntity, strand))
    {
        throw std::runtime_error("Failed to author the keyboard-controlled rope.");
    }

    if (!showParticleGaps)
    {
        AuthoredParticleSequenceState sequence{};
        sequence.entries.reserve(kParticleCount);
        for (std::uint32_t i = 0u; i < kParticleCount; ++i)
        {
            sequence.entries.push_back(
                {strandEntity, AuthoredParticleReferenceType::StrandParticle, i});
        }
        sequence = world.upsertParticleSequence(sequence);
        world.setMeshRenderer(strandEntity,
                              MeshRendererComponent{strandMesh, strandMaterial, true});
        world.setProceduralDeformableCurveRender(
            strandEntity,
            ProceduralDeformableCurveRenderComponent{sequence.sequenceId, 0.065f, 12u, true});
    }

    InteractiveState state{};
    state.leftPosition  = strand.restPositions.front();
    state.rightPosition = strand.restPositions.back();
    state.leftHandle = authorKinematicHandle(runtime, state.leftPosition, handleMesh,
                                              leftHandleMaterial);
    state.rightHandle = authorKinematicHandle(runtime, state.rightPosition, handleMesh,
                                               rightHandleMaterial);
    attachParticle(runtime, strandEntity, 0u, state.leftHandle);
    attachParticle(runtime, strandEntity, kParticleCount - 1u, state.rightHandle);
    return state;
}

Diligent::float3 spatialInput(const DebugViewerApp &viewer, int forward, int backward, int left,
                              int right, int up, int down)
{
    Diligent::float3 direction{};
    direction.z += viewer.isKeyDown(forward) ? 1.0f : 0.0f;
    direction.z -= viewer.isKeyDown(backward) ? 1.0f : 0.0f;
    direction.x -= viewer.isKeyDown(left) ? 1.0f : 0.0f;
    direction.x += viewer.isKeyDown(right) ? 1.0f : 0.0f;
    direction.y += viewer.isKeyDown(up) ? 1.0f : 0.0f;
    direction.y -= viewer.isKeyDown(down) ? 1.0f : 0.0f;

    const float lengthSq = Diligent::dot(direction, direction);
    return lengthSq > 1.0f ? direction * (1.0f / std::sqrt(lengthSq)) : direction;
}

void advanceHandle(Diligent::float3 &position, const Diligent::float3 &direction,
                   float deltaSeconds)
{
    position += direction * (kHandleSpeed * deltaSeconds);
    position.x = std::clamp(position.x, -kPlaneLimit, kPlaneLimit);
    position.z = std::clamp(position.z, -kPlaneLimit, kPlaneLimit);
    position.y = std::clamp(position.y, kHandleHeight, kHeightLimit);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    bool capsuleSelfCollision = false;
    bool showParticleGaps     = false;
    bool particleRadiusWasSet = false;
    float particleRadius      = kParticleRadius;
    float particleSpacing     = kParticleSpacing;
    float contactSlop         = 1.0e-3f;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h")
            {
                printUsage(argv[0]);
                return 0;
            }
            if (arg == "--show-particle-gaps")
            {
                showParticleGaps = true;
                continue;
            }
            if (arg == "--capsule-self-collision")
            {
                capsuleSelfCollision = true;
                continue;
            }
            if (arg == "--particle-radius")
            {
                particleRadius = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--particle-radius"),
                    "particle radius", false);
                particleRadiusWasSet = true;
                continue;
            }
            if (arg == "--particle-spacing")
            {
                particleSpacing = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--particle-spacing"),
                    "particle spacing", false);
                continue;
            }
            if (arg == "--contact-slop")
            {
                contactSlop = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i,
                                                                        "--contact-slop"),
                    "contact slop", true);
                continue;
            }
            if (!cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options,
                                                                          false))
            {
                printUsage(argv[0]);
                return 2;
            }
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    if (showParticleGaps && !particleRadiusWasSet)
    {
        particleRadius = kGapDemoParticleRadius;
    }
    const float halfLength =
        0.5f * static_cast<float>(kParticleCount - 1u) * particleSpacing;
    if (halfLength > kPlaneLimit)
    {
        CRESSIM_LOG_ERROR("Particle spacing places the rope outside the controllable plane. "
                          "Use --particle-spacing ",
                          2.0f * kPlaneLimit / static_cast<float>(kParticleCount - 1u),
                          " or smaller.\n");
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.physicsDesc.substeps          = 6u;
    config.physicsDesc.defaultIterations = 32u;
    config.physicsDesc.contact.slop       = contactSlop;
    config.physicsDesc.enableExperimentalStrandCapsuleSelfCollision = capsuleSelfCollision;

    DebugViewerApp viewer;
    ViewerExampleDefaults defaults{};
    defaults.windowTitle = "CRESSim Neo - Two-End Rope Keyboard Control";
    defaults.width       = 1280u;
    defaults.height      = 820u;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, defaults);
    viewerDesc.useFixedTimestep     = true;
    viewerDesc.enableDebugParticles = showParticleGaps;
    // The movement keys belong to the rope handles in this example, not the fly camera.
    viewerDesc.keymap.moveForward  = -1;
    viewerDesc.keymap.moveBackward = -1;
    viewerDesc.keymap.moveLeft     = -1;
    viewerDesc.keymap.moveRight    = -1;
    viewerDesc.keymap.moveUp       = -1;
    viewerDesc.keymap.moveDown     = -1;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Rope keyboard-control viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rope keyboard-control runtime initialization failed.\n");
        return 1;
    }

    try
    {
        auto &world = runtime.getWorld();

        const EntityId cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 4.8f, -6.2f};
        cameraTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.62f);
        world.setTransform(cameraEntity, cameraTransform);
        world.setCamera(cameraEntity, CameraComponent{});

        const EntityId lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = {-0.35f, -1.0f, 0.25f};
        light.color     = {1.0f, 0.98f, 0.94f};
        light.intensity = 8.0f;
        world.setDirectionalLight(lightEntity, light);

        auto &resources = runtime.getResources();
        const MeshHandle planeMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(4.0f, "RopeKeyboard.Ground", 8.0f));
        const MeshHandle handleMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeSphereMesh(0.12f, 24u, 12u,
                                                            "RopeKeyboard.Handle"));
        const MeshHandle strandMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeCanonicalCurveTubeMesh(
                kParticleCount, 12u, "RopeKeyboard.Strand", 4.0f));

        const MaterialHandle groundMaterial = registerMaterial(
            resources, "RopeKeyboard.GroundMaterial", {0.32f, 0.36f, 0.42f}, 0.92f);
        const MaterialHandle strandMaterial = registerMaterial(
            resources, "RopeKeyboard.StrandMaterial", {0.92f, 0.64f, 0.18f}, 0.58f);
        const MaterialHandle leftHandleMaterial = registerMaterial(
            resources, "RopeKeyboard.LeftHandleMaterial", {0.12f, 0.72f, 0.96f}, 0.28f);
        const MaterialHandle rightHandleMaterial = registerMaterial(
            resources, "RopeKeyboard.RightHandleMaterial", {0.96f, 0.25f, 0.20f}, 0.28f);

        authorGround(runtime, planeMesh, groundMaterial);
        InteractiveState state = authorRope(runtime, strandMesh, handleMesh, strandMaterial,
                                            leftHandleMaterial, rightHandleMaterial,
                                            showParticleGaps, capsuleSelfCollision,
                                            particleSpacing, particleRadius);

        if (showParticleGaps)
        {
            auto renderOptions                                      = runtime.renderFrameOptions();
            renderOptions.debugParticles.enabled                    = true;
            renderOptions.debugParticles.useParticleRadii           = true;
            renderOptions.debugParticles.drawConstraintEdges        = true;
            renderOptions.debugParticles.highlightStaticParticles = false;
            renderOptions.debugParticles.color     = {0.12f, 0.72f, 0.96f, 1.0f};
            renderOptions.debugParticles.edgeColor = {1.0f, 0.82f, 0.18f, 1.0f};
            runtime.setRenderFrameOptions(renderOptions);
        }

        CRESSIM_LOG_INFO(
            "Two-end rope control loaded.\n"
            "Blue/left endpoint: W/S move forward/back, A/D move left/right, E/Q move "
            "up/down.\n"
            "Red/right endpoint: Arrow Up/Down move forward/back, Arrow Left/Right move "
            "left/right, Page Up/Page Down move up/down.\n"
            "Use --show-particle-gaps to display the real sparse collision particles.\n"
            "Press Escape to quit; hold the right mouse button to look around.\n");
        CRESSIM_LOG_INFO("Self-collision mode: ",
                         capsuleSelfCollision ? "experimental swept segment capsules"
                                              : "particle spheres",
                         ".\n");
        const float surfaceGap = particleSpacing - 2.0f * particleRadius;
        CRESSIM_LOG_INFO("Collision geometry: spacing=", particleSpacing,
                         ", radius=", particleRadius, ", surface gap=", surfaceGap,
                         ", contact slop=", contactSlop, ".\n");

        DebugViewerCallbacks callbacks{};
        callbacks.beforeTick = [&viewer, &state](const FrameContext &frame, Runtime &cbRuntime)
        {
            const Diligent::float3 leftDirection =
                spatialInput(viewer, kKeyW, kKeyS, kKeyA, kKeyD, kKeyE, kKeyQ);
            const Diligent::float3 rightDirection =
                spatialInput(viewer, kKeyUp, kKeyDown, kKeyLeft, kKeyRight, kKeyPageUp,
                             kKeyPageDown);
            advanceHandle(state.leftPosition, leftDirection, frame.deltaSeconds);
            advanceHandle(state.rightPosition, rightDirection, frame.deltaSeconds);
            updateKinematicHandle(cbRuntime, state.leftHandle, state.leftPosition);
            updateKinematicHandle(cbRuntime, state.rightHandle, state.rightPosition);
        };

        DebugViewerCameraBinding binding{};
        binding.cameraEntity = cameraEntity;
        const bool runOk      = viewer.run(runtime, binding, callbacks);

        runtime.shutdown();
        viewer.shutdown();
        return runOk ? 0 : 1;
    }
    catch (const std::runtime_error &error)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rope keyboard-control setup failed: ", error.what(), "\n");
        return 1;
    }
}
