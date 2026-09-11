#include "common/frame_context.h"
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

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ClothComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr std::uint32_t kColumns = 25u;
constexpr std::uint32_t kRows    = 19u;
constexpr float kWidth           = 4.8f;
constexpr float kHeight          = 3.6f;

MeshResourceDesc makeClothMesh(ClothComponent &cloth)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ClothHangingPatch.Surface";
    mesh.vertices.reserve(kColumns * kRows);
    mesh.indices.reserve((kColumns - 1u) * (kRows - 1u) * 6u);
    cloth.source.objectSpaceRestPositions.reserve(kColumns * kRows);
    cloth.source.triangleVertexIndices.reserve(mesh.indices.capacity());

    for (std::uint32_t row = 0u; row < kRows; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(kRows - 1u);
        for (std::uint32_t column = 0u; column < kColumns; ++column)
        {
            const float u = static_cast<float>(column) / static_cast<float>(kColumns - 1u);
            const Diligent::float3 position{(u - 0.5f) * kWidth, -v * kHeight, 0.0f};
            cloth.source.objectSpaceRestPositions.push_back(position);
            mesh.vertices.push_back(
                {position, {0.0f, 0.0f, 1.0f}, u, v, {1.0f, 0.0f, 0.0f, -1.0f}});
        }
    }

    for (std::uint32_t row = 0u; row + 1u < kRows; ++row)
    {
        for (std::uint32_t column = 0u; column + 1u < kColumns; ++column)
        {
            const std::uint32_t a = row * kColumns + column;
            const std::uint32_t b = a + 1u;
            const std::uint32_t c = a + kColumns;
            const std::uint32_t d = c + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }
    cloth.source.triangleVertexIndices = mesh.indices;

    // A sewn top edge: every top-row particle is fixed, while the remaining surface is free.
    for (std::uint32_t column = 0u; column < kColumns; ++column)
        cloth.source.staticParticleIndices.push_back(column);
    return mesh;
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options,
                                                                        false))
                continue;
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
    config.physicsDesc.softInternalIterations = 32u;
    config.physicsDesc.softContactIterations  = 24u;

    DebugViewerApp viewer;
    ViewerExampleDefaults defaults{};
    defaults.windowTitle  = "CRESSim Neo XPBD Cloth - Hanging Patch";
    defaults.width        = 960u;
    defaults.height       = 720u;
    defaults.showStats    = true;
    defaults.vSync        = true;
    const auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, defaults);
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

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.15f, -7.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 46.0f;
    camera.clearColorValue    = {0.025f, 0.035f, 0.055f, 1.0f};
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.55f};
    light.color     = {1.0f, 0.97f, 0.92f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    ClothComponent cloth{};
    const auto clothMesh = resources.registerMesh(makeClothMesh(cloth));
    MaterialResourceDesc clothMaterialDesc{};
    clothMaterialDesc.debugName             = "ClothHangingPatch.ClothMaterial";
    clothMaterialDesc.baseColor             = {0.12f, 0.38f, 0.92f};
    clothMaterialDesc.roughness             = 0.72f;
    clothMaterialDesc.pipeline.featureFlags = MaterialFeatureFlags::DoubleSided;
    const auto clothMaterial                = resources.registerMaterial(clothMaterialDesc);

    const auto clothEntity = world.createEntity();
    TransformComponent clothTransform{};
    clothTransform.worldTransform.position = {0.0f, 4.15f, 0.0f};
    world.setTransform(clothEntity, clothTransform);
    MeshRendererComponent clothRenderer{};
    clothRenderer.mesh     = clothMesh;
    clothRenderer.material = clothMaterial;
    clothRenderer.visible  = true;
    world.setMeshRenderer(clothEntity, clothRenderer);
    cloth.particleMass         = 0.035f;
    cloth.particleRadius       = 0.065f;
    cloth.structuralCompliance = 1.0e-7f;
    cloth.bendCompliance       = 2.5e-3f;
    cloth.selfCollisionEnabled = true;
    if (!world.setCloth(clothEntity, cloth))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author the cloth patch.\n");
        return 1;
    }

    const float sphereRadius = 0.72f;
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        sphereRadius, 32u, 20u, "ClothHangingPatch.PusherMesh"));
    MaterialResourceDesc sphereMaterialDesc{};
    sphereMaterialDesc.debugName = "ClothHangingPatch.PusherMaterial";
    sphereMaterialDesc.baseColor = {0.96f, 0.33f, 0.12f};
    sphereMaterialDesc.metallic  = 0.08f;
    sphereMaterialDesc.roughness = 0.34f;
    const auto sphereMaterial    = resources.registerMaterial(sphereMaterialDesc);

    const auto sphereEntity = world.createEntity();
    const Diligent::float3 sphereBase{0.0f, 2.15f, -1.35f};
    TransformComponent sphereTransform{};
    sphereTransform.worldTransform.position = sphereBase;
    world.setTransform(sphereEntity, sphereTransform);
    MeshRendererComponent sphereRenderer{};
    sphereRenderer.mesh     = sphereMesh;
    sphereRenderer.material = sphereMaterial;
    sphereRenderer.visible  = true;
    world.setMeshRenderer(sphereEntity, sphereRenderer);
    RigidBodyComponent sphereBody{};
    sphereBody.bodyType                = RigidBodyType::Kinematic;
    sphereBody.inverseMass             = 0.0f;
    sphereBody.inverseInertiaLocal     = {0.0f, 0.0f, 0.0f};
    sphereBody.kinematicTargetEnabled  = true;
    sphereBody.kinematicTargetPosition = sphereBase;
    sphereBody.kinematicTargetRotation = {};
    world.setRigidBody(sphereEntity, sphereBody);
    ColliderComponent sphereCollider{};
    sphereCollider.shapeType   = ColliderShapeType::Sphere;
    sphereCollider.shapeParams = {sphereRadius, 0.0f, 0.0f, 0.0f};
    world.addCollider(sphereEntity, sphereCollider);

    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(6.0f, "ClothHangingPatch.GroundMesh"));
    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "ClothHangingPatch.GroundMaterial";
    groundMaterialDesc.baseColor = {0.34f, 0.37f, 0.42f};
    groundMaterialDesc.roughness = 0.9f;
    const auto groundMaterial    = resources.registerMaterial(groundMaterialDesc);
    const auto groundEntity      = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent groundRenderer{};
    groundRenderer.mesh     = groundMesh;
    groundRenderer.material = groundMaterial;
    groundRenderer.visible  = true;
    world.setMeshRenderer(groundEntity, groundRenderer);
    RigidBodyComponent groundBody{};
    groundBody.bodyType            = RigidBodyType::Static;
    groundBody.inverseMass         = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {6.0f, 0.05f, 6.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    world.physicsWorld().ensureDerivedStateUpToDate();
    const auto *clothState = world.physicsWorld().tryGetCloth(clothEntity);
    CRESSIM_LOG_INFO("Cloth patch: ", clothState != nullptr ? clothState->particleCount : 0u,
                     " particles, ",
                     clothState != nullptr ? clothState->structuralConstraintCount : 0u,
                     " structural constraints, ",
                     clothState != nullptr ? clothState->dihedralConstraintCount : 0u,
                     " dihedral constraints.\n");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick =
        [sphereEntity, sphereBase](const FrameContext &frame, Runtime &callbackRuntime)
    {
        auto body = callbackRuntime.getWorld().tryGetRigidBody(sphereEntity);
        if (!body.has_value()) return;
        const float time                = static_cast<float>(frame.timeSeconds);
        const float approach            = 0.5f * (1.0f - std::cos(time * 0.7f));
        body->bodyType                  = RigidBodyType::Kinematic;
        body->kinematicTargetEnabled    = true;
        body->kinematicTargetPosition   = sphereBase;
        body->kinematicTargetPosition.x = 0.42f * std::sin(time * 0.65f);
        body->kinematicTargetPosition.z = sphereBase.z + 1.1f * approach;
        callbackRuntime.getWorld().setRigidBody(sphereEntity, *body);
    };

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{cameraEntity}, callbacks);
    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
