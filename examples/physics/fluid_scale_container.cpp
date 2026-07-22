#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kBaseParticleRadius  = 0.09f;
constexpr float kBaseParticleSpacing = 0.18f;
constexpr float kBaseContainerHalfXz = 2.0f;
constexpr float kBaseContainerHeight = 4.0f;
constexpr float kBaseWallThickness   = 0.15f;

struct ScaleOptions
{
    CommonExampleOptions common{};
    float scale                      = 1.0f;
    float pressureRelaxation         = 0.7f;
    float maxUnderDensityRatio       = 0.0f;
    float viscosity                  = 0.0f;
    float cohesion                   = 1.0f;
    float surfaceTension             = 2.0f;
    float vorticityConfinement       = 1.0f;
    std::uint32_t substeps           = 4u;
    std::uint32_t fluidIterations   = 10u;
    bool debugParticles              = false;
};

float parsePositiveFloat(const std::string &value, const char *optionName)
{
    char *end          = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || !(parsed > 0.0f))
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ".");
    }
    return parsed;
}

float parseNonNegativeFloat(const std::string &value, const char *optionName)
{
    char *end          = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || parsed < 0.0f)
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ".");
    }
    return parsed;
}

std::uint32_t parsePositiveUint(const std::string &value, const char *optionName)
{
    char *end                      = nullptr;
    const unsigned long parsed     = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed == 0u ||
        parsed > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ".");
    }
    return static_cast<std::uint32_t>(parsed);
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--scale S] [--substeps N] [--fluid-iterations N] [--pressure-relaxation R]"
        " [--max-under-density-ratio R]"
        " [--viscosity V] [--cohesion C] [--surface-tension T] [--vorticity V]"
        " [--debug-particles]",
        false);
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName  = name;
    desc.baseColor  = baseColor;
    desc.roughness  = roughness;
    return resources.registerMaterial(desc);
}

void spawnStaticBox(cressim::neo::engine::World &world, MeshHandle mesh, MaterialHandle material,
                    const Diligent::float3 &position, const Diligent::float3 &halfExtents,
                    bool visible)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = halfExtents;
    world.setTransform(entity, transform);

    if (visible)
    {
        world.setMeshRenderer(entity, MeshRendererComponent{mesh, material, true});
    }

    RigidBodyComponent body{};
    body.bodyType             = RigidBodyType::Static;
    body.inverseMass          = 0.0f;
    body.inverseInertiaLocal  = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType   = ColliderShapeType::Box;
    collider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(entity, collider);
}

bool spawnFluid(cressim::neo::engine::World &world, float scale,
                const cressim::neo::physics::FluidMaterialDesc &material)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = {0.0f, 1.55f * scale, 0.0f};
    world.setTransform(entity, transform);

    FluidComponent fluid{};
    fluid.source.kind                               = FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size                   = {2.7f * scale, 2.5f * scale, 2.7f * scale};
    fluid.source.regularGrid.targetParticleSpacing  = kBaseParticleSpacing * scale;
    fluid.particleRadius                            = kBaseParticleRadius * scale;
    fluid.particleMass = 1000.0f * 8.0f * fluid.particleRadius * fluid.particleRadius *
                         fluid.particleRadius;
    fluid.material     = material;
    fluid.visualColor  = {0.18f, 0.66f, 0.95f, 0.72f};
    return world.setFluid(entity, fluid);
}

} // namespace

int main(int argc, char **argv)
{
    ScaleOptions options{};
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options.common, false))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--scale")
            {
                options.scale = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i, "--scale"),
                    "scale");
            }
            else if (arg == "--substeps")
            {
                options.substeps = parsePositiveUint(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i, "--substeps"),
                    "substeps");
            }
            else if (arg == "--fluid-iterations")
            {
                options.fluidIterations = parsePositiveUint(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--fluid-iterations"),
                    "fluid iterations");
            }
            else if (arg == "--pressure-relaxation")
            {
                options.pressureRelaxation = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--pressure-relaxation"),
                    "pressure relaxation");
            }
            else if (arg == "--max-under-density-ratio")
            {
                options.maxUnderDensityRatio = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--max-under-density-ratio"),
                    "max under-density ratio");
            }
            else if (arg == "--viscosity")
            {
                options.viscosity = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--viscosity"),
                    "viscosity");
            }
            else if (arg == "--cohesion")
            {
                options.cohesion = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i,
                                                                          "--cohesion"),
                    "cohesion");
            }
            else if (arg == "--surface-tension")
            {
                options.surfaceTension = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--surface-tension"),
                    "surface tension");
            }
            else if (arg == "--vorticity")
            {
                options.vorticityConfinement = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i,
                                                                          "--vorticity"),
                    "vorticity");
            }
            else if (arg == "--debug-particles")
            {
                options.debugParticles = true;
            }
            else
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.physicsDesc.gravity.y *= options.scale;
    config.physicsDesc.substeps          = options.substeps;
    config.physicsDesc.defaultIterations = options.fluidIterations;
    config.physicsDesc.fluidIterations   = options.fluidIterations;
    config.physicsDesc.fluid.constraintRelaxation = options.pressureRelaxation;
    config.physicsDesc.fluid.maxUnderDensityRatio = options.maxUnderDensityRatio;
    config.physicsDesc.softContactIterations = 0u;
    config.physicsDesc.softInternalIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults defaults{};
    defaults.windowTitle = "CRESSim Neo Fluid Scale Container";
    defaults.showStats   = true;
    defaults.vSync       = false;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options.common, defaults);
    // This example is used to compare scale-dependent solver behavior. Decouple its
    // physics step from first-frame shader compilation and presentation timing.
    viewerDesc.useFixedTimestep = true;
    viewerDesc.enableDebugParticles = options.debugParticles;
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
    cameraTransform.worldTransform.position = {0.0f, 4.0f * options.scale, -10.5f * options.scale};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.24f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 38.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color     = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "FluidScaleContainer.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "FluidScaleContainer.Floor", {0.76f, 0.79f, 0.84f}, 0.92f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "FluidScaleContainer.Wall", {0.12f, 0.34f, 0.72f}, 0.58f);

    const float s         = options.scale;
    const float halfXz    = kBaseContainerHalfXz * s;
    const float height    = kBaseContainerHeight * s;
    const float thickness = kBaseWallThickness * s;
    spawnStaticBox(world, boxMesh, floorMaterial, {0.0f, -thickness, 0.0f},
                   {halfXz + thickness, thickness, halfXz + thickness}, true);
    spawnStaticBox(world, boxMesh, wallMaterial, {-halfXz - thickness, 0.5f * height, 0.0f},
                   {thickness, 0.5f * height + thickness, halfXz + thickness}, true);
    spawnStaticBox(world, boxMesh, wallMaterial, {halfXz + thickness, 0.5f * height, 0.0f},
                   {thickness, 0.5f * height + thickness, halfXz + thickness}, true);
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.5f * height, halfXz + thickness},
                   {halfXz + thickness, 0.5f * height + thickness, thickness}, true);
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.5f * height, -halfXz - thickness},
                   {halfXz + thickness, 0.5f * height + thickness, thickness}, false);
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, height + thickness, 0.0f},
                   {halfXz + thickness, thickness, halfXz + thickness}, false);

    cressim::neo::physics::FluidMaterialDesc material{};
    material.contact.friction       = 0.04f;
    material.contact.staticFriction = 0.06f;
    material.contact.damping        = 0.2f;
    material.viscosity              = options.viscosity;
    material.cohesion               = options.cohesion;
    material.gravityScale           = 0.5f;
    material.cflCoefficient         = 1.0f;
    material.vorticityConfinement   = options.vorticityConfinement;
    material.surfaceTension         = options.surfaceTension;
    if (!spawnFluid(world, s, material))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author fluid scale container.\n");
        return 1;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    CRESSIM_LOG_INFO("Fluid scale container authored ", world.physicsWorld().particles().size(),
                     " particles; scale=", s, ", radius=", kBaseParticleRadius * s,
                     ", spacing=", kBaseParticleSpacing * s, ", substeps=", options.substeps,
                     ", fluidIterations=", options.fluidIterations,
                     ", pressureRelaxation=", options.pressureRelaxation,
                     ", maxUnderDensityRatio=",
                     options.maxUnderDensityRatio, ".\n");

    const bool runOk = viewer.run(runtime, DebugViewerCameraBinding{cameraEntity}, {});
    runtime.shutdown();
    viewer.shutdown();
    return runOk ? 0 : 1;
}
