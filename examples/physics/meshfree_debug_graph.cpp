#include "common/logger.h"
#include "helpers/asset_paths.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "physics/load_particle_cloud.h"
#include "physics/load_surface_mesh.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::MeshfreeSoftBodyComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SurfaceMeshData;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

struct MeshfreeDebugOptions
{
    CommonExampleOptions common{};
    bool drawConstraintEdges = false;
    bool drawShapeClusters = false;
    bool debugParticlesExplicit = false;
    bool showDebugParticles = false;
    bool pinToGround = true;
    bool vSync = false;
    std::filesystem::path cloudPath =
        cressim::neo::examples::helpers::assetPath("physics/fixtures/gallbladder_particles.bin");
    std::filesystem::path surfacePath =
        cressim::neo::examples::helpers::assetPath("physics/fixtures/Gallbladder.obj");
    std::uint32_t neighbourCount = 14u;
    std::uint32_t substeps = 0u;
    std::uint32_t softInternalIterations = 0u;
    std::uint32_t softContactIterations = 0u;
    cressim::neo::physics::SoftBodyShapeMatchingDesc shapeMatching{};
    float cloudScale = 0.035f;
    float particleRadius = 0.035f;
    float particleMass = 0.0f;
    float compliance = -1.0f;
    float damping = -1.0f;
    float pinBand = 0.01f;
    float shapeCorrectionDebugScale = 40.0f;
    Diligent::float3 rotationDegrees{0.0f, 0.0f, 0.0f};
};

struct ParticleBounds
{
    Diligent::float3 min{};
    Diligent::float3 max{};
    Diligent::float3 center{};
    Diligent::float3 extent{};
};

struct ShapeMatchingTopologyStats
{
    std::uint32_t clusterCount                  = 0u;
    std::uint32_t totalMemberships              = 0u;
    std::uint32_t maximumMembershipsPerParticle = 0u;
    std::uint32_t minimumClusterSize            = 0u;
    std::uint32_t maximumClusterSize            = 0u;
    std::uint32_t inactiveClusterCount          = 0u;
    std::uint32_t degenerateClusterCount        = 0u;
    float meanMembershipsPerParticle            = 0.0f;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--cloud PATH] [--cloud-scale S] [--neighbours N] [--particle-radius R]"
        " [--surface PATH]"
        " [--particle-mass M] [--compliance C] [--damping D] [--substeps N]"
        " [--soft-iterations N] [--contact-iterations N] [--rotation-degrees X Y Z]"
        " [--rotate-x DEG] [--rotate-y DEG] [--rotate-z DEG] [--pin-band B] [--drop]"
        " [--enable-shape-matching] [--shape-cluster-size N] [--shape-memberships N]"
        " [--shape-iterations N] [--shape-stiffness S] [--draw-shape-clusters]"
        " [--shape-correction-debug-scale S] [--disable-cut-aware-clusters]"
        " [--show-particles] [--hide-particles] [--draw-edges] [--vsync]",
        false);
}

float parseFloat(const char *value, const char *optionName)
{
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0')
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return parsed;
}

std::uint32_t parsePositiveUint32(const char *value, const char *optionName)
{
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0ul ||
        parsed > static_cast<unsigned long>(UINT32_MAX))
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parsePositiveFloat(const char *value, const char *optionName)
{
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !(parsed > 0.0f))
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return parsed;
}

Diligent::float3 parseFloat3(int argc, char **argv, int &index, const char *optionName)
{
    const float x = parseFloat(
        cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, optionName),
        optionName);
    const float y = parseFloat(
        cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, optionName),
        optionName);
    const float z = parseFloat(
        cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, optionName),
        optionName);
    return {x, y, z};
}

Diligent::QuaternionF makeEulerRotationDegrees(const Diligent::float3 &degrees)
{
    const float radiansPerDegree = Diligent::PI_F / 180.0f;
    const Diligent::QuaternionF rotationX =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f},
                                                     degrees.x * radiansPerDegree);
    const Diligent::QuaternionF rotationY =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f},
                                                     degrees.y * radiansPerDegree);
    const Diligent::QuaternionF rotationZ =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f},
                                                     degrees.z * radiansPerDegree);
    return rotationZ * rotationY * rotationX;
}

float parseNonNegativeFloat(const char *value, const char *optionName)
{
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || parsed < 0.0f)
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

MaterialHandle registerSurfaceShellMaterial(
    cressim::neo::graphics::RenderResourceManager &resources)
{
    MaterialResourceDesc desc{};
    desc.debugName             = "MeshfreeDebug.CubeSurfaceShell";
    desc.baseColor             = {0.82f, 0.28f, 0.34f};
    desc.roughness             = 0.68f;
    desc.renderMode            = MaterialRenderMode::Opaque;
    desc.opacity               = 1.0f;
    desc.castsShadows          = true;
    desc.pipeline.featureFlags = MaterialFeatureFlags::DoubleSided;
    return resources.registerMaterial(desc);
}

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

ParticleBounds computeParticleBounds(const std::vector<Diligent::float3> &particles)
{
    ParticleBounds bounds{};
    if (particles.empty())
    {
        return bounds;
    }

    bounds.min = particles.front();
    bounds.max = particles.front();
    for (const Diligent::float3 &particle : particles)
    {
        bounds.min.x = std::min(bounds.min.x, particle.x);
        bounds.min.y = std::min(bounds.min.y, particle.y);
        bounds.min.z = std::min(bounds.min.z, particle.z);
        bounds.max.x = std::max(bounds.max.x, particle.x);
        bounds.max.y = std::max(bounds.max.y, particle.y);
        bounds.max.z = std::max(bounds.max.z, particle.z);
    }

    bounds.center = {(bounds.min.x + bounds.max.x) * 0.5f,
                     (bounds.min.y + bounds.max.y) * 0.5f,
                     (bounds.min.z + bounds.max.z) * 0.5f};
    bounds.extent = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                     bounds.max.z - bounds.min.z};
    return bounds;
}

void centerAndScaleParticles(std::vector<Diligent::float3> &particles, const float scale)
{
    const ParticleBounds bounds = computeParticleBounds(particles);
    for (Diligent::float3 &particle : particles)
    {
        particle = (particle - bounds.center) * scale;
    }
}

void centerAndScaleSurface(SurfaceMeshData &surface, const Diligent::float3 &center,
                           const float scale)
{
    for (Diligent::float3 &position : surface.surfaceRestPositions)
    {
        position = (position - center) * scale;
    }
}

MeshResourceDesc makeSurfaceMeshResource(const SurfaceMeshData &surface, const char *debugName)
{
    MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    mesh.vertices.reserve(surface.surfaceRestPositions.size());
    for (std::size_t i = 0u; i < surface.surfaceRestPositions.size(); ++i)
    {
        MeshResourceDesc::Vertex vertex{};
        vertex.position = surface.surfaceRestPositions[i];
        if (i < surface.surfaceNormals.size())
        {
            vertex.normal = surface.surfaceNormals[i];
        }
        mesh.vertices.push_back(vertex);
    }

    mesh.indices.reserve(surface.surfaceTriangles.size() * 3u);
    for (const Diligent::uint3 &triangle : surface.surfaceTriangles)
    {
        mesh.indices.push_back(triangle.x);
        mesh.indices.push_back(triangle.y);
        mesh.indices.push_back(triangle.z);
    }
    return mesh;
}

ParticleBounds computeRotatedParticleBounds(const std::vector<Diligent::float3> &particles,
                                            const Diligent::QuaternionF &rotation)
{
    ParticleBounds bounds{};
    if (particles.empty())
    {
        return bounds;
    }

    std::vector<Diligent::float3> rotatedParticles;
    rotatedParticles.reserve(particles.size());
    for (const Diligent::float3 &particle : particles)
    {
        rotatedParticles.push_back(rotation.RotateVector(particle));
    }
    return computeParticleBounds(rotatedParticles);
}

std::vector<std::uint32_t> selectGroundPinParticles(
    const std::vector<Diligent::float3> &particles, const Diligent::QuaternionF &rotation,
    const float pinBand)
{
    std::vector<std::uint32_t> staticParticleIndices;
    if (particles.empty())
    {
        return staticParticleIndices;
    }

    float minY = rotation.RotateVector(particles.front()).y;
    for (const Diligent::float3 &particle : particles)
    {
        minY = std::min(minY, rotation.RotateVector(particle).y);
    }

    const float maxPinnedY = minY + std::max(pinBand, 0.0f);
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(particles.size()); ++particleIndex)
    {
        if (rotation.RotateVector(particles[particleIndex]).y <= maxPinnedY)
        {
            staticParticleIndices.push_back(particleIndex);
        }
    }

    if (staticParticleIndices.empty())
    {
        staticParticleIndices.push_back(0u);
    }
    return staticParticleIndices;
}

std::vector<Diligent::float3> loadConfiguredParticles(const MeshfreeDebugOptions &options,
                                                      ParticleBounds &outSourceBounds)
{
    std::vector<Diligent::float3> particles;
    std::string errorMessage;
    if (!cressim::neo::physics::readParticleCloudBin(options.cloudPath, particles, errorMessage))
    {
        CRESSIM_LOG_ERROR(errorMessage, "\n");
        return {};
    }

    outSourceBounds = computeParticleBounds(particles);
    centerAndScaleParticles(particles, options.cloudScale);
    return particles;
}

void applyDebugParticleGraphOptions(Runtime &runtime, const bool enabled,
                                    const bool drawConstraintEdges,
                                    const float particleRadius,
                                    const std::uint32_t shapeMaxMembershipCount,
                                    const float shapeCorrectionDebugScale)
{
    cressim::neo::graphics::RenderFrameOptions renderOptions = runtime.renderFrameOptions();
    renderOptions.debugParticles.enabled                  = enabled;
    renderOptions.debugParticles.drawConstraintEdges      = drawConstraintEdges;
    renderOptions.debugParticles.highlightStaticParticles = true;
    renderOptions.debugParticles.useParticleRadii         = true;
    renderOptions.debugParticles.shapeMaxMembershipCount  = shapeMaxMembershipCount;
    renderOptions.debugParticles.shapeCorrectionScale     = shapeCorrectionDebugScale;
    renderOptions.debugParticles.color                    = {0.18f, 0.74f, 1.0f, 1.0f};
    renderOptions.debugParticles.staticColor              = {1.0f, 0.22f, 0.12f, 1.0f};
    renderOptions.debugParticles.edgeColor                = {1.0f, 0.86f, 0.18f, 1.0f};
    renderOptions.debugParticles.fallbackRadius           = particleRadius;
    runtime.setRenderFrameOptions(renderOptions);
}

ShapeMatchingTopologyStats computeShapeMatchingTopologyStats(
    const cressim::neo::physics::ShapeMatchingDataHost &data,
    const std::uint32_t particleCount)
{
    ShapeMatchingTopologyStats stats{};
    stats.clusterCount     = static_cast<std::uint32_t>(data.clusters.size());
    stats.totalMemberships = static_cast<std::uint32_t>(data.members.size());
    stats.meanMembershipsPerParticle =
        particleCount > 0u ? static_cast<float>(data.members.size()) / static_cast<float>(particleCount)
                           : 0.0f;
    for (const cressim::neo::physics::ParticleShapeMembershipRangeGPU &range :
         data.particleMembershipRanges)
    {
        stats.maximumMembershipsPerParticle =
            std::max(stats.maximumMembershipsPerParticle, range.count);
    }
    for (const cressim::neo::physics::ShapeClusterGPU &cluster : data.clusters)
    {
        stats.minimumClusterSize =
            stats.minimumClusterSize == 0u
                ? cluster.memberCount
                : std::min(stats.minimumClusterSize, cluster.memberCount);
        stats.maximumClusterSize = std::max(stats.maximumClusterSize, cluster.memberCount);
        if ((cluster.flags & cressim::neo::physics::ShapeCluster_Active) == 0u)
        {
            ++stats.inactiveClusterCount;
        }
        if ((cluster.flags & cressim::neo::physics::ShapeCluster_Degenerate) != 0u)
        {
            ++stats.degenerateClusterCount;
        }
    }
    return stats;
}

} // namespace

int main(int argc, char **argv)
{
    MeshfreeDebugOptions options{};
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
            if (arg == "--cloud")
            {
                options.cloudPath = cressim::neo::examples::helpers::requireOptionValue(
                    argc, argv, i, "--cloud");
                continue;
            }
            if (arg == "--surface")
            {
                options.surfacePath = cressim::neo::examples::helpers::requireOptionValue(
                    argc, argv, i, "--surface");
                continue;
            }
            if (arg == "--cloud-scale")
            {
                options.cloudScale = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--cloud-scale"),
                    "--cloud-scale");
                continue;
            }
            if (arg == "--neighbours")
            {
                options.neighbourCount = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--neighbours"),
                    "--neighbours");
                continue;
            }
            if (arg == "--particle-radius")
            {
                options.particleRadius = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--particle-radius"),
                    "--particle-radius");
                continue;
            }
            if (arg == "--particle-mass")
            {
                options.particleMass = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--particle-mass"),
                    "--particle-mass");
                continue;
            }
            if (arg == "--compliance")
            {
                options.compliance = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--compliance"),
                    "--compliance");
                continue;
            }
            if (arg == "--damping")
            {
                options.damping = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--damping"),
                    "--damping");
                continue;
            }
            if (arg == "--substeps")
            {
                options.substeps = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--substeps"),
                    "--substeps");
                continue;
            }
            if (arg == "--soft-iterations")
            {
                options.softInternalIterations = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--soft-iterations"),
                    "--soft-iterations");
                continue;
            }
            if (arg == "--contact-iterations")
            {
                options.softContactIterations = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--contact-iterations"),
                    "--contact-iterations");
                continue;
            }
            if (arg == "--enable-shape-matching")
            {
                options.shapeMatching.enabled = true;
                continue;
            }
            if (arg == "--shape-cluster-size")
            {
                options.shapeMatching.targetClusterSize = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--shape-cluster-size"),
                    "--shape-cluster-size");
                continue;
            }
            if (arg == "--shape-memberships")
            {
                options.shapeMatching.minimumMembershipsPerParticle = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--shape-memberships"),
                    "--shape-memberships");
                continue;
            }
            if (arg == "--shape-iterations")
            {
                options.shapeMatching.solverIterations = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--shape-iterations"),
                    "--shape-iterations");
                continue;
            }
            if (arg == "--shape-stiffness")
            {
                options.shapeMatching.stiffnessPerPass = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--shape-stiffness"),
                    "--shape-stiffness");
                continue;
            }
            if (arg == "--shape-correction-debug-scale")
            {
                options.shapeCorrectionDebugScale = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--shape-correction-debug-scale"),
                    "--shape-correction-debug-scale");
                continue;
            }
            if (arg == "--disable-cut-aware-clusters")
            {
                options.shapeMatching.cutAware = false;
                continue;
            }
            if (arg == "--rotation-degrees")
            {
                options.rotationDegrees = parseFloat3(argc, argv, i, "--rotation-degrees");
                continue;
            }
            if (arg == "--rotate-x")
            {
                options.rotationDegrees.x = parseFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--rotate-x"),
                    "--rotate-x");
                continue;
            }
            if (arg == "--rotate-y")
            {
                options.rotationDegrees.y = parseFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--rotate-y"),
                    "--rotate-y");
                continue;
            }
            if (arg == "--rotate-z")
            {
                options.rotationDegrees.z = parseFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--rotate-z"),
                    "--rotate-z");
                continue;
            }
            if (arg == "--pin-band")
            {
                options.pinBand = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--pin-band"),
                    "--pin-band");
                continue;
            }
            if (arg == "--pin-to-ground")
            {
                options.pinToGround = true;
                continue;
            }
            if (arg == "--drop")
            {
                options.pinToGround = false;
                continue;
            }
            if (arg == "--draw-edges")
            {
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--show-particles")
            {
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--hide-particles")
            {
                options.showDebugParticles = false;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--vsync")
            {
                options.vSync = true;
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.physicsDesc.substeps                    = 4u;
    config.physicsDesc.defaultIterations           = 16u;
    config.physicsDesc.softInternalIterations      = 32u;
    config.physicsDesc.softContactIterations       = 12u;
    config.physicsDesc.rigidRigidContactIterations = 0u;
    if (options.substeps != 0u)
    {
        config.physicsDesc.substeps = options.substeps;
    }
    if (options.softInternalIterations != 0u)
    {
        config.physicsDesc.softInternalIterations = options.softInternalIterations;
    }
    if (options.softContactIterations != 0u)
    {
        config.physicsDesc.softContactIterations = options.softContactIterations;
    }

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Meshfree Debug Graph";
    viewerDefaults.showStats   = true;
    viewerDefaults.vSync       = options.vSync;
    auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);
    viewerDesc.enableDebugParticles = options.showDebugParticles;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Meshfree debug graph viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Meshfree debug graph runtime initialization failed.\n");
        return 1;
    }

    auto &world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.85f, -4.2f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.10f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 42.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color     = {1.0f, 0.98f, 0.94f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(4.0f, "MeshfreeDebug.GroundMesh"));
    const MaterialHandle groundMaterial =
        registerMaterial(resources, "MeshfreeDebug.Ground", {0.18f, 0.20f, 0.22f}, 0.86f);
    const MaterialHandle surfaceShellMaterial = registerSurfaceShellMaterial(resources);

    constexpr float kGroundSurfaceY       = -0.72f;
    constexpr float kGroundColliderHalfY  = 0.35f;
    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, kGroundSurfaceY, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity,
                          MeshRendererComponent{groundMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType            = RigidBodyType::Static;
    groundBody.inverseMass         = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType      = ColliderShapeType::Box;
    groundCollider.shapeParams    = {4.0f, kGroundColliderHalfY, 4.0f, 0.0f};
    groundCollider.localPosition  = {0.0f, -kGroundColliderHalfY, 0.0f};
    groundCollider.friction       = 0.55f;
    groundCollider.staticFriction = 0.75f;
    groundCollider.collisionLayer = 0x1u;
    groundCollider.collisionMask  = 0x2u;
    world.addCollider(groundEntity, groundCollider);

    ParticleBounds sourceParticleBounds{};
    std::vector<Diligent::float3> particles =
        loadConfiguredParticles(options, sourceParticleBounds);
    if (particles.empty())
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("No meshfree particles were loaded.\n");
        return 1;
    }
    const ParticleBounds particleBounds = computeParticleBounds(particles);
    CRESSIM_LOG_INFO("Meshfree debug source: ", particles.size(), " particles, bounds min=(",
                     particleBounds.min.x, ", ", particleBounds.min.y, ", ",
                     particleBounds.min.z, "), max=(", particleBounds.max.x, ", ",
                     particleBounds.max.y, ", ", particleBounds.max.z, ").\n");

    SurfaceMeshData surfaceMesh;
    std::string surfaceErrorMessage;
    if (!cressim::neo::physics::readObjSurfaceMesh(options.surfacePath, surfaceMesh,
                                                   surfaceErrorMessage))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR(surfaceErrorMessage, "\n");
        return 1;
    }
    centerAndScaleSurface(surfaceMesh, sourceParticleBounds.center, options.cloudScale);
    CRESSIM_LOG_INFO("Meshfree surface shell: ", surfaceMesh.surfaceRestPositions.size(),
                     " vertices, ", surfaceMesh.surfaceTriangles.size(), " triangles from ",
                     options.surfacePath.string(), ".\n");

    const float configuredParticleRadius = options.particleRadius;
    const Diligent::QuaternionF softRotation = makeEulerRotationDegrees(options.rotationDegrees);
    const ParticleBounds rotatedParticleBounds =
        computeRotatedParticleBounds(particles, softRotation);
    const float groundParticleCenterY = kGroundSurfaceY + configuredParticleRadius;
    const float initialSoftY =
        options.pinToGround ? groundParticleCenterY - rotatedParticleBounds.min.y : 1.05f;

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, initialSoftY, 0.0f};
    softTransform.worldTransform.rotation = softRotation;
    world.setTransform(softEntity, softTransform);

    MeshfreeSoftBodyComponent softBody{};
    softBody.particles                       = std::move(particles);
    softBody.surfaceRestPositions            = surfaceMesh.surfaceRestPositions;
    softBody.surfaceNormals                  = surfaceMesh.surfaceNormals;
    softBody.surfaceTriangles                = surfaceMesh.surfaceTriangles;
    if (options.pinToGround)
    {
        softBody.staticParticleIndices =
            selectGroundPinParticles(softBody.particles, softRotation, options.pinBand);
    }
    softBody.neighbourCount                  = options.neighbourCount;
    softBody.particleRadius                  = configuredParticleRadius;
    softBody.particleMass                    = options.particleMass > 0.0f
                                                   ? options.particleMass
                                                   : 0.0002f;
    softBody.compliance                      = options.compliance >= 0.0f
                                                   ? options.compliance
                                                   : 5.0e-3f;
    softBody.shapeMatching                   = options.shapeMatching;
    softBody.material.contact.friction       = 0.45f;
    softBody.material.contact.staticFriction = 0.60f;
    softBody.material.contact.damping        = options.damping >= 0.0f
                                                   ? options.damping
                                                   : 4.20f;
    softBody.selfCollisionEnabled            = false;
    softBody.collisionLayer                  = 0x2u;
    softBody.collisionMask                   = 0x1u;
    if (!world.setMeshfreeSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author meshfree debug soft body.\n");
        return 1;
    }

    if (!options.drawConstraintEdges && !surfaceMesh.surfaceRestPositions.empty())
    {
        const MeshHandle surfaceShellMesh = resources.registerMesh(
            makeSurfaceMeshResource(surfaceMesh, "MeshfreeDebug.CubeSurfaceShellMesh"));
        world.setMeshRenderer(softEntity,
                              MeshRendererComponent{surfaceShellMesh, surfaceShellMaterial, true});
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const cressim::neo::physics::ShapeMatchingDataHost &shapeMatchingData =
        world.physicsWorld().shapeMatchingData();
    const ShapeMatchingTopologyStats shapeStats = computeShapeMatchingTopologyStats(
        shapeMatchingData, static_cast<std::uint32_t>(softBody.particles.size()));
    if (const cressim::neo::physics::SoftBodyState *softState =
            world.physicsWorld().tryGetSoftBody(softEntity))
    {
        const float averageDegree =
            softState->particleCount > 0u
                ? (2.0f * static_cast<float>(softState->edgeCount)) /
                      static_cast<float>(softState->particleCount)
                : 0.0f;
        CRESSIM_LOG_INFO("Meshfree XPBD graph: ", softState->edgeCount, " distance constraints",
                         options.drawConstraintEdges ? " (edge debug draw enabled).\n"
                                                     : " (edge debug draw disabled).\n");
        CRESSIM_LOG_INFO("Meshfree XPBD tuning: neighbours=", softBody.neighbourCount,
                         ", average degree=", averageDegree,
                         ", substeps=", config.physicsDesc.substeps,
                         ", soft iterations=", config.physicsDesc.softInternalIterations,
                         ", contact iterations=", config.physicsDesc.softContactIterations,
                         ", compliance=", softBody.compliance,
                         ", damping=", softBody.material.contact.damping,
                         ", particle mass=", softBody.particleMass,
                         ", rotation degrees=(", options.rotationDegrees.x, ", ",
                         options.rotationDegrees.y, ", ", options.rotationDegrees.z, ").\n");
        CRESSIM_LOG_INFO("Meshfree shape matching: ",
                         softBody.shapeMatching.enabled ? "enabled" : "disabled",
                         ", cluster size=", softBody.shapeMatching.targetClusterSize,
                         ", memberships=",
                         softBody.shapeMatching.minimumMembershipsPerParticle,
                         ", iterations=", softBody.shapeMatching.solverIterations,
                         ", stiffness=", softBody.shapeMatching.stiffnessPerPass,
                         ", cut-aware=",
                         softBody.shapeMatching.cutAware ? "true" : "false",
                         options.drawShapeClusters
                             ? " (cluster debug requested; particle graph overlay enabled).\n"
                             : ".\n");
        CRESSIM_LOG_INFO("Meshfree shape-matching debug stats: cluster count=",
                         shapeStats.clusterCount,
                         ", total memberships=", shapeStats.totalMemberships,
                         ", mean memberships per particle=",
                         shapeStats.meanMembershipsPerParticle,
                         ", maximum memberships per particle=",
                         shapeStats.maximumMembershipsPerParticle,
                         ", minimum cluster size=", shapeStats.minimumClusterSize,
                         ", maximum cluster size=", shapeStats.maximumClusterSize,
                         ", inactive cluster count=", shapeStats.inactiveClusterCount,
                         ", degenerate rotation count=", shapeStats.degenerateClusterCount,
                         ".\n");
        CRESSIM_LOG_INFO("Meshfree XPBD pinning: ",
                         options.pinToGround ? "enabled" : "disabled",
                         ", static particles=", softBody.staticParticleIndices.size(),
                         ", pin band=", options.pinBand,
                         ", initial center y=", initialSoftY, ".\n");
    }

    applyDebugParticleGraphOptions(runtime, options.showDebugParticles,
                                   options.drawConstraintEdges,
                                   options.particleRadius,
                                   shapeStats.maximumMembershipsPerParticle,
                                   options.shapeCorrectionDebugScale);

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&options, shapeStats](const cressim::neo::common::FrameContext &,
                                                  Runtime &callbackRuntime)
    {
        applyDebugParticleGraphOptions(callbackRuntime, options.showDebugParticles,
                                       options.drawConstraintEdges,
                                       options.particleRadius,
                                       shapeStats.maximumMembershipsPerParticle,
                                       options.shapeCorrectionDebugScale);
    };
    callbacks.afterTick = [&viewerDesc](const cressim::neo::common::FrameContext &frame,
                                        Runtime &callbackRuntime)
    {
        if (viewerDesc.statsIntervalFrames == 0u ||
            (frame.frameIndex > 1u &&
             frame.frameIndex % viewerDesc.statsIntervalFrames != 0u))
        {
            return;
        }
        const cressim::neo::physics::PhysicsSolver *solver =
            callbackRuntime.getPhysicsSolver();
        if (solver == nullptr)
        {
            return;
        }
        const cressim::neo::physics::ShapeMatchingSolverStats stats =
            solver->lastShapeMatchingStats();
        CRESSIM_LOG_INFO("Meshfree shape-matching runtime stats: cluster count=",
                         stats.clusterCount,
                         ", total memberships=", stats.totalMemberships,
                         ", mean memberships per particle=",
                         stats.meanMembershipsPerParticle,
                         ", maximum memberships per particle=",
                         stats.maximumMembershipsPerParticle,
                         ", inactive cluster count=", stats.inactiveClusterCount,
                         ", degenerate rotation count=", stats.degenerateRotationCount,
                         ", maximum shape correction=",
                         stats.maximumShapeCorrection,
                         ", average shape correction=",
                         stats.averageShapeCorrection,
                         ".\n");
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk     = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Meshfree debug graph viewer run failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Meshfree debug graph viewer finished.\n");
    return 0;
}
