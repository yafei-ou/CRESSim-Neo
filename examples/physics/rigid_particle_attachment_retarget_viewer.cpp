#include "common/logger.h"
#include "engine/components.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::CustomComputePassDesc;
using cressim::neo::engine::CustomComputePassHandle;
using cressim::neo::engine::CustomComputeResourceAccess;
using cressim::neo::engine::CustomComputeResourceBindingDesc;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredRigidParticleAttachmentConstraintState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr const char *kRetargetShader = R"(
#include "structured_buffer_compat.hlsli"
#include "physics/core/physics_math.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"

cbuffer AttachmentRetargetConstants
{
    uint enabled;
    uint particleIndex;
    float targetX;
    float targetY;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_KinematicTargetFlags);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidParticleAttachmentConstraint, g_Constraints);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuRigidParticleAttachmentConstraint constraint = CRESSIM_SB_LOAD(g_Constraints, 0u);
    constraint.enabled = enabled;
    constraint.particleIndex = particleIndex;
    CRESSIM_SB_STORE(g_Constraints, 0u, constraint);

    float4 targetPosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, 0u);
    targetPosition.x = targetX;
    targetPosition.y = targetY;
    CRESSIM_SB_STORE(g_KinematicTargetPositions, 0u, targetPosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, 0u,
                     QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, 0u)));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, 0u, kKinematicTargetEnabled);
}
)";

constexpr std::uint32_t kAttachmentSlotParticleA = 2u;
constexpr std::uint32_t kAttachmentSlotParticleB = 5u;
constexpr std::uint64_t kPhaseLengthFrames = 180u;

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

std::vector<std::uint8_t> makeRetargetConstants(std::uint32_t enabled, std::uint32_t particleIndex,
                                                float targetX, float targetY)
{
    struct Constants
    {
        std::uint32_t enabled = 0u;
        std::uint32_t particleIndex = 0u;
        float targetX = 0.0f;
        float targetY = 0.0f;
    };

    const Constants constants{enabled, particleIndex, targetX, targetY};
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(&constants);
    return std::vector<std::uint8_t>(bytes, bytes + sizeof(constants));
}

CustomComputePassHandle createRetargetPass(Runtime &runtime)
{
    std::vector<CustomComputeResourceBindingDesc> bindings(6u);
    bindings[0].shaderVariableName = "g_RigidBodyPositionsInvMass";
    bindings[0].resourceKey = "rigid.positions";
    bindings[0].access = CustomComputeResourceAccess::ReadOnly;
    bindings[1].shaderVariableName = "g_RigidBodyOrientations";
    bindings[1].resourceKey = "rigid.orientations";
    bindings[1].access = CustomComputeResourceAccess::ReadOnly;
    bindings[2].shaderVariableName = "g_KinematicTargetPositions";
    bindings[2].resourceKey = "rigid.kinematic_target_positions";
    bindings[2].access = CustomComputeResourceAccess::ReadWrite;
    bindings[3].shaderVariableName = "g_KinematicTargetOrientations";
    bindings[3].resourceKey = "rigid.kinematic_target_orientations";
    bindings[3].access = CustomComputeResourceAccess::ReadWrite;
    bindings[4].shaderVariableName = "g_KinematicTargetFlags";
    bindings[4].resourceKey = "rigid.kinematic_target_flags";
    bindings[4].access = CustomComputeResourceAccess::ReadWrite;
    bindings[5].shaderVariableName = "g_Constraints";
    bindings[5].resourceKey = "constraint.rigid_particle_attachments";
    bindings[5].access = CustomComputeResourceAccess::ReadWrite;

    CustomComputePassDesc passDesc{};
    passDesc.debugName = "RigidParticleAttachmentRetargetViewer";
    passDesc.shaderSource = kRetargetShader;
    passDesc.threadGroupSizeX = 1u;
    passDesc.resourceBindings = bindings;
    passDesc.dispatch.mode = cressim::neo::engine::CustomComputeDispatchMode::ExplicitGroupCount;
    passDesc.dispatch.groupCountX = 1u;
    passDesc.constantBufferVariableName = "AttachmentRetargetConstants";
    passDesc.constantBufferSizeBytes = 16u;
    return runtime.createCustomComputePass(passDesc);
}

void authorGround(Runtime &runtime, MeshHandle mesh, MaterialHandle material)
{
    auto &world = runtime.getWorld();
    const EntityId groundEntity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = {0.0f, -0.65f, 0.0f};
    world.setTransform(groundEntity, transform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{mesh, material, true});

    RigidBodyComponent body{};
    body.bodyType = RigidBodyType::Static;
    world.setRigidBody(groundEntity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {4.0f, 0.1f, 4.0f, 0.0f};
    collider.localPosition = {0.0f, -0.1f, 0.0f};
    collider.friction = 0.9f;
    collider.staticFriction = 1.1f;
    world.addCollider(groundEntity, collider);
}

void authorScene(Runtime &runtime, MeshHandle handleMesh, MeshHandle groundMesh,
                 MaterialHandle handleMaterial, MaterialHandle groundMaterial)
{
    auto &world = runtime.getWorld();

    const EntityId handleEntity = world.createEntity();
    TransformComponent handleTransform{};
    handleTransform.worldTransform.position = {0.0f, 1.0f, 0.0f};
    world.setTransform(handleEntity, handleTransform);
    world.setMeshRenderer(handleEntity, MeshRendererComponent{handleMesh, handleMaterial, true});

    RigidBodyComponent handleBody{};
    handleBody.bodyType = RigidBodyType::Kinematic;
    handleBody.inverseMass = 1.0f;
    handleBody.simulated = true;
    handleBody.kinematicTargetEnabled = true;
    handleBody.kinematicTargetPosition = handleTransform.worldTransform.position;
    world.setRigidBody(handleEntity, handleBody);

    const EntityId strandAEntity = world.createEntity();
    cressim::neo::engine::StrandComponent strandA{};
    strandA.restPositions = {{-0.92f, 1.0f, 0.0f}, {-0.74f, 0.72f, 0.0f}, {-0.54f, 0.44f, 0.0f}};
    strandA.staticParticleIndices = {0u};
    strandA.particleRadius = 0.045f;
    strandA.particleMass = 0.08f;
    strandA.simulated = true;
    if (!world.setStrand(strandAEntity, strandA))
    {
        throw std::runtime_error("Failed to author strand A.");
    }

    const EntityId strandBEntity = world.createEntity();
    cressim::neo::engine::StrandComponent strandB{};
    strandB.restPositions = {{0.92f, 1.0f, 0.0f}, {0.74f, 0.72f, 0.0f}, {0.54f, 0.44f, 0.0f}};
    strandB.staticParticleIndices = {0u};
    strandB.particleRadius = 0.045f;
    strandB.particleMass = 0.08f;
    strandB.simulated = true;
    if (!world.setStrand(strandBEntity, strandB))
    {
        throw std::runtime_error("Failed to author strand B.");
    }

    AuthoredRigidParticleAttachmentConstraintState attachment{};
    attachment.constraintId = 1u;
    attachment.rigidBodyEntityId = handleEntity;
    attachment.particle.entityId = strandAEntity;
    attachment.particle.type = AuthoredParticleReferenceType::StrandParticle;
    attachment.particle.localParticleIndex = 2u;
    attachment.enabled = true;
    if (!world.upsertRigidParticleAttachmentConstraint(attachment))
    {
        throw std::runtime_error("Failed to author rigid-particle attachment.");
    }

    authorGround(runtime, groundMesh, groundMaterial);
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }

        if (!cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options, false))
        {
            printUsage(argv[0]);
            return 1;
        }
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.physicsDesc.substeps = 4u;
    config.physicsDesc.defaultIterations = 24u;

    DebugViewerApp viewer;
    ViewerExampleDefaults defaults{};
    defaults.windowTitle = "CRESSim Neo Rigid Particle Attachment Retarget";
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, defaults);
    viewerDesc.useFixedTimestep = true;
    viewerDesc.enableDebugParticles = true;
    viewerDesc.showStats = true;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Rigid-particle attachment retarget viewer initialization failed.");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rigid-particle attachment retarget runtime initialization failed.");
        return 1;
    }

    try
    {
        auto &world = runtime.getWorld();
        const EntityId cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 1.7f, -4.8f};
        cameraTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, -0.22f);
        world.setTransform(cameraEntity, cameraTransform);
        world.setCamera(cameraEntity, CameraComponent{});

        const EntityId lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = {-0.3f, -1.0f, 0.15f};
        light.color = {1.0f, 0.98f, 0.95f};
        light.intensity = 7.5f;
        world.setDirectionalLight(lightEntity, light);

        auto &resources = runtime.getResources();
        const MeshHandle handleMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeBoxMesh({0.12f, 0.12f, 0.12f},
                                                         "AttachmentRetarget.Handle"));
        const MeshHandle groundMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(4.0f, "AttachmentRetarget.Ground", 4.0f));
        const MaterialHandle handleMaterial =
            registerMaterial(resources, "AttachmentRetarget.HandleMaterial",
                             {0.12f, 0.62f, 0.88f}, 0.28f);
        const MaterialHandle groundMaterial =
            registerMaterial(resources, "AttachmentRetarget.GroundMaterial",
                             {0.76f, 0.78f, 0.82f}, 0.95f);

        authorScene(runtime, handleMesh, groundMesh, handleMaterial, groundMaterial);

        runtime.prepare();
        if (!runtime.uploadWorld())
        {
            throw std::runtime_error("uploadWorld() failed.");
        }

        const CustomComputePassHandle pass = createRetargetPass(runtime);
        if (!pass.isValid())
        {
            throw std::runtime_error("Failed to create retarget custom compute pass.");
        }

        struct ViewerState
        {
            CustomComputePassHandle pass{};
            std::uint64_t currentPhase = ~0ull;
        };

        ViewerState state{};
        state.pass = pass;

        CRESSIM_LOG_INFO(
            "Expected behavior: the strands start slanted inward. Phase A pulls the left tip inward/up "
            "to the blue handle, phase D disables the rigid-particle attachment completely and parks the "
            "handle away from both strands, and phase B pulls the right tip inward/up. During phase D, "
            "neither free tip should keep tracking the box.");

        DebugViewerCallbacks callbacks{};
        callbacks.beforeTick = [&state](const FrameContext &frame, Runtime &cbRuntime) {
            const std::uint64_t phase = (frame.frameIndex / kPhaseLengthFrames) % 3u;
            if (phase == state.currentPhase)
            {
                return;
            }

            state.currentPhase = phase;
            const std::uint32_t enabled = phase == 1u ? 0u : 1u;
            const std::uint32_t particleIndex =
                phase == 2u ? kAttachmentSlotParticleB : kAttachmentSlotParticleA;
            const float targetX = phase == 0u ? -0.22f : (phase == 1u ? 0.0f : 0.22f);
            const float targetY = phase == 0u ? 1.55f : (phase == 1u ? 1.95f : 1.55f);
            if (!cbRuntime.updateCustomComputePassConstants(
                    state.pass, makeRetargetConstants(enabled, particleIndex, targetX, targetY)) ||
                !cbRuntime.executeCustomComputePass(state.pass))
            {
                CRESSIM_LOG_ERROR("Failed to update rigid-particle attachment retarget pass.");
            }

            const char *phaseName = phase == 0u ? "A" : (phase == 1u ? "D" : "B");
            CRESSIM_LOG_INFO("Attachment phase ", phaseName, ": enabled=", enabled,
                             ", particleIndex=", particleIndex, ", targetX=", targetX,
                             ", targetY=", targetY, ".");
        };

        DebugViewerCameraBinding binding{};
        binding.cameraEntity = cameraEntity;
        const bool runOk = viewer.run(runtime, binding, callbacks);

        runtime.shutdown();
        viewer.shutdown();

        if (!runOk)
        {
            CRESSIM_LOG_ERROR("Rigid-particle attachment retarget viewer run failed.");
            return 1;
        }
    }
    catch (const std::runtime_error &error)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rigid-particle attachment retarget viewer setup failed: ", error.what());
        return 1;
    }

    return 0;
}
