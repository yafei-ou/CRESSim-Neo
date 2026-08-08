#include "common/logger.h"
#include "common/id.h"
#include "helpers/asset_paths.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "physics/load_particle_cloud.h"
#include "physics/load_surface_mesh.h"
#include "physics/physics_world.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
using cressim::neo::physics::CuttingToolGPU;
using cressim::neo::physics::CuttingToolShape;
using cressim::neo::physics::ElectrocauteryToolGPU;
using cressim::neo::physics::ElectrocauteryToolMode;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftEdge;
using cressim::neo::physics::SoftEdgeToolCounters;
using cressim::neo::physics::SoftRenderDataHost;
using cressim::neo::physics::SoftThermalMaterialDesc;
using cressim::neo::physics::SurfaceMeshData;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

enum class ThermalDebugMode
{
    None,
    ElectrosurgicalCut,
    ElectrosurgicalCoagulation,
    ElectrosurgicalBlend,
};

enum class ThermalVisualizationMode
{
    PbrSurface,
    ParticleHeatmap,
};

enum class ElectrocauteryDemoPhase
{
    Entry,
    Dwell,
    Plunge,
    Press,
    Retract,
    Complete,
};

struct MeshfreeDebugOptions
{
    CommonExampleOptions common{};
    bool drawConstraintEdges = false;
    bool drawShapeClusters = false;
    bool debugParticlesExplicit = false;
    bool showDebugParticles = false;
    bool pinToGround = true;
    bool disableEdgeTest = false;
    bool disableEdgeRegion = false;
    bool enableFracture = false;
    bool enableCuttingTool = false;
    bool instantCut = false;
    bool showCutEdges = false;
    bool showStrain = false;
    bool showDamage = false;
    ThermalDebugMode thermalDebugMode = ThermalDebugMode::None;
    ThermalVisualizationMode thermalVisualizationMode = ThermalVisualizationMode::PbrSurface;
    CuttingToolShape toolShape = CuttingToolShape::Blade;
    std::filesystem::path cloudPath =
        cressim::neo::examples::helpers::assetPath("physics/fixtures/cube_particles.bin");
    std::filesystem::path surfacePath =
        cressim::neo::examples::helpers::assetPath("physics/fixtures/Cube.obj");
    std::uint32_t neighbourCount = 14u;
    std::uint32_t substeps = 0u;
    std::uint32_t softInternalIterations = 0u;
    std::uint32_t softContactIterations = 0u;
    cressim::neo::physics::SoftBodyShapeMatchingDesc shapeMatching{};
    float cloudScale = 0.35f;
    float particleRadius = 0.035f;
    float particleMass = 0.0f;
    float compliance = -1.0f;
    float damping = -1.0f;
    float pinBand = 0.01f;
    float shapeCorrectionDebugScale = 40.0f;
    float fractureThreshold = 0.35f;
    float toolRadius = 0.003f;
    float toolStrength = 5.0f;
    float bladeLength = 0.0f;
    float bladeDepth = 0.0f;
    float bladeThickness = 0.012f;
    float bladeCutBandDepth = 0.0f;
    float disableEdgeRegionRadius = 0.0f;
    Diligent::float3 disableEdgeRegionCenter{0.0f, 0.0f, 0.0f};
    Diligent::float3 rotationDegrees{0.0f, 0.0f, 0.0f};
};

struct ElectrocauteryDemoState
{
    ElectrocauteryDemoPhase phase = ElectrocauteryDemoPhase::Entry;
    double phaseStartTimeSeconds = 0.0;
    Diligent::float3 entryTip{0.0f, 0.0f, 0.0f};
    Diligent::float3 contactTip{0.0f, 0.0f, 0.0f};
    Diligent::float3 surfaceNormal{0.0f, 0.0f, -1.0f};
    Diligent::float3 trocarShaftAxis{0.0f, 1.0f, 0.0f};
    Diligent::float3 frozenTip{0.0f, 0.0f, 0.0f};
    Diligent::float3 frozenShaftAxis{0.0f, 1.0f, 0.0f};
    float pressDepth = 0.08f;
    float groundTipY = -0.72f;
    float retractDistance = 0.70f;
    float capsuleRadius = 0.025f;
    float capsuleHalfHeight = 0.18f;
    float cutPunctureDamageThreshold = 0.55f;
    std::uint32_t thermallyCutEdgesAtPlungeStart = 0u;
    std::uint32_t lastThermallyCutEdges = 0u;
    std::uint32_t stableThermalCutReadbacks = 0u;
    bool tipReachedGround = false;
};

struct ElectrocauteryToolPose
{
    Diligent::float3 proximalTip{0.0f, 0.0f, 0.0f};
    Diligent::float3 distalTip{0.0f, 0.0f, 0.0f};
    Diligent::float3 shaftAxis{0.0f, 1.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    float capsuleRadius = 0.025f;
    float capsuleHalfHeight = 0.18f;
    bool heatActive = false;
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

struct EdgeDebugStats
{
    std::uint32_t activeCount = 0u;
    std::uint32_t disabledCount = 0u;
    std::uint32_t cutCount = 0u;
    std::uint32_t fracturedCount = 0u;
    float maxStrain = 0.0f;
    float averageStrain = 0.0f;
    float maxDamage = 0.0f;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--cloud-scale S] [--neighbours N] [--particle-radius R]"
        " [--particle-mass M] [--compliance C] [--damping D] [--substeps N]"
        " [--soft-iterations N] [--contact-iterations N] [--rotation-degrees X Y Z]"
        " [--rotate-x DEG] [--rotate-y DEG] [--rotate-z DEG] [--pin-band B] [--drop]"
        " [--enable-shape-matching] [--shape-cluster-size N] [--shape-memberships N]"
        " [--shape-iterations N] [--shape-stiffness S] [--draw-shape-clusters]"
        " [--shape-correction-debug-scale S] [--disable-cut-aware-clusters]"
        " [--show-particles] [--hide-particles] [--draw-edges] [--disable-edge-test]"
        " [--disable-edge-region X Y Z R] [--enable-fracture] [--fracture-threshold S]"
        " [--enable-cutting-tool] [--tool-shape blade] [--tool-radius R]"
        " [--tool-strength S] [--instant-cut] [--blade-length L] [--blade-depth D]"
        " [--blade-thickness T] [--blade-cut-band-depth D]"
        " [--show-cut-edges] [--show-strain] [--show-damage]"
        " [--thermal-debug electrosurgical_cut|electrosurgical_coagulation]"
        " [--thermal-visual pbr|heatmap]",
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

CuttingToolShape parseToolShape(const char *value)
{
    const std::string parsed = value != nullptr ? value : "";
    if (parsed == "blade")
    {
        return CuttingToolShape::Blade;
    }
    throw std::invalid_argument("Invalid --tool-shape: " + parsed);
}

ThermalDebugMode parseThermalDebugMode(const char *value)
{
    const std::string parsed = value != nullptr ? value : "";
    if (parsed == "electrosurgical_cut")
    {
        return ThermalDebugMode::ElectrosurgicalCut;
    }
    if (parsed == "electrosurgical_coagulation")
    {
        return ThermalDebugMode::ElectrosurgicalCoagulation;
    }
    if (parsed == "electrosurgical_blend")
    {
        return ThermalDebugMode::ElectrosurgicalBlend;
    }
    throw std::invalid_argument("Invalid --thermal-debug: " + parsed);
}

ThermalVisualizationMode parseThermalVisualizationMode(const char *value)
{
    const std::string parsed = value != nullptr ? value : "";
    if (parsed == "pbr")
    {
        return ThermalVisualizationMode::PbrSurface;
    }
    if (parsed == "heatmap")
    {
        return ThermalVisualizationMode::ParticleHeatmap;
    }
    throw std::invalid_argument("Invalid --thermal-visual: " + parsed);
}

bool thermalDebugEnabled(const MeshfreeDebugOptions &options)
{
    return options.thermalDebugMode != ThermalDebugMode::None;
}

bool thermalHeatmapEnabled(const MeshfreeDebugOptions &options)
{
    return thermalDebugEnabled(options) &&
           options.thermalVisualizationMode == ThermalVisualizationMode::ParticleHeatmap;
}

bool thermalPbrSurfaceEnabled(const MeshfreeDebugOptions &options)
{
    return !thermalDebugEnabled(options) ||
           options.thermalVisualizationMode == ThermalVisualizationMode::PbrSurface;
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
    desc.baseColor             = {0.33f, 0.42f, 0.18f}; //{0.82f, 0.28f, 0.34f};
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

std::uint64_t particlePairKey(std::uint32_t a, std::uint32_t b)
{
    if (b < a)
    {
        std::swap(a, b);
    }
    return (static_cast<std::uint64_t>(a) << 32u) | static_cast<std::uint64_t>(b);
}

std::unordered_set<std::uint64_t> buildThermalCutParticlePairs(
    const std::vector<SoftEdge> &softEdges)
{
    std::unordered_set<std::uint64_t> pairs;
    for (const SoftEdge &edge : softEdges)
    {
        if ((edge.flags & cressim::neo::physics::Edge_ThermalCut) == 0u)
        {
            continue;
        }
        pairs.insert(particlePairKey(edge.particleA, edge.particleB));
    }
    return pairs;
}

bool triangleTouchesThermalCutEdge(
    std::uint32_t i0,
    std::uint32_t i1,
    std::uint32_t i2,
    const SoftRenderDataHost &softRenderData,
    const std::unordered_set<std::uint64_t> &thermalCutParticlePairs)
{
    if (thermalCutParticlePairs.empty())
    {
        return false;
    }

    auto pairWasThermallyCut = [&](std::uint32_t a, std::uint32_t b)
    {
        return thermalCutParticlePairs.find(particlePairKey(a, b)) !=
               thermalCutParticlePairs.end();
    };

    std::uint32_t boundParticles[12u]{};
    std::uint32_t boundParticleCount = 0u;
    auto appendBindingParticles = [&](std::uint32_t vertexIndex)
    {
        if (vertexIndex >= softRenderData.vertexBindings.size())
        {
            return;
        }

        const auto &binding = softRenderData.vertexBindings[vertexIndex];
        const std::uint32_t indices[4u] = {
            binding.particleIndices.x,
            binding.particleIndices.y,
            binding.particleIndices.z,
            binding.particleIndices.w};
        const float weights[4u] = {
            binding.weights.x,
            binding.weights.y,
            binding.weights.z,
            binding.weights.w};
        for (std::uint32_t slot = 0u; slot < 4u; ++slot)
        {
            if (weights[slot] <= 1.0e-4f)
            {
                continue;
            }
            boundParticles[boundParticleCount++] = indices[slot];
        }
    };

    appendBindingParticles(i0);
    appendBindingParticles(i1);
    appendBindingParticles(i2);
    for (std::uint32_t a = 0u; a < boundParticleCount; ++a)
    {
        for (std::uint32_t b = a + 1u; b < boundParticleCount; ++b)
        {
            if (pairWasThermallyCut(boundParticles[a], boundParticles[b]))
            {
                return true;
            }
        }
    }
    return false;
}

std::uint32_t filterSurfaceMeshTrianglesByTopology(
    MeshResourceDesc &surfaceMesh, const std::vector<std::uint32_t> &originalIndices,
    const std::vector<std::uint32_t> &vertexComponents,
    const SoftRenderDataHost *softRenderData,
    const std::unordered_set<std::uint64_t> *thermalCutParticlePairs)
{
    if (originalIndices.empty() || vertexComponents.empty())
    {
        return 0u;
    }

    constexpr std::uint32_t kInvalidComponent = UINT32_MAX;
    std::vector<std::uint32_t> filteredIndices;
    filteredIndices.reserve(originalIndices.size());
    std::uint32_t culledTriangleCount = 0u;

    for (std::size_t triangle = 0u; triangle + 2u < originalIndices.size(); triangle += 3u)
    {
        const std::uint32_t i0 = originalIndices[triangle + 0u];
        const std::uint32_t i1 = originalIndices[triangle + 1u];
        const std::uint32_t i2 = originalIndices[triangle + 2u];
        if (i0 >= vertexComponents.size() || i1 >= vertexComponents.size() ||
            i2 >= vertexComponents.size())
        {
            filteredIndices.push_back(i0);
            filteredIndices.push_back(i1);
            filteredIndices.push_back(i2);
            continue;
        }

        const std::uint32_t c0 = vertexComponents[i0];
        const std::uint32_t c1 = vertexComponents[i1];
        const std::uint32_t c2 = vertexComponents[i2];
        const bool crossesDisconnectedComponents =
            c0 != kInvalidComponent && c1 != kInvalidComponent &&
            c2 != kInvalidComponent && !(c0 == c1 && c1 == c2);
        const bool touchesThermalCutEdge =
            softRenderData != nullptr && thermalCutParticlePairs != nullptr &&
            triangleTouchesThermalCutEdge(i0, i1, i2, *softRenderData,
                                          *thermalCutParticlePairs);
        if (!crossesDisconnectedComponents && !touchesThermalCutEdge)
        {
            filteredIndices.push_back(i0);
            filteredIndices.push_back(i1);
            filteredIndices.push_back(i2);
            continue;
        }

        ++culledTriangleCount;
    }

    if (surfaceMesh.indices != filteredIndices)
    {
        surfaceMesh.indices = std::move(filteredIndices);
    }
    return culledTriangleCount;
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

float length3(const Diligent::float3 &value)
{
    return std::sqrt(Diligent::dot(value, value));
}

Diligent::float3 choosePerpendicular(const Diligent::float3 &axis)
{
    const Diligent::float3 reference = std::abs(axis.y) < 0.99f
                                           ? Diligent::float3{0.0f, 1.0f, 0.0f}
                                           : Diligent::float3{1.0f, 0.0f, 0.0f};
    return Diligent::normalize(Diligent::cross(reference, axis));
}

float segmentDistanceToPoint(const Diligent::float3 &a, const Diligent::float3 &b,
                             const Diligent::float3 &point)
{
    const Diligent::float3 ab = b - a;
    const float abLengthSq = Diligent::dot(ab, ab);
    if (abLengthSq <= 1.0e-12f)
    {
        return length3(point - a);
    }

    const float t = std::clamp(Diligent::dot(point - a, ab) / abLengthSq, 0.0f, 1.0f);
    return length3((a + ab * t) - point);
}

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &x,
                                          const Diligent::float3 &y,
                                          const Diligent::float3 &z)
{
    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;
    const float trace = m00 + m11 + m22;

    Diligent::QuaternionF q{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.q.w = 0.25f * s;
        q.q.x = (m21 - m12) / s;
        q.q.y = (m02 - m20) / s;
        q.q.z = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.q.w = (m21 - m12) / s;
        q.q.x = 0.25f * s;
        q.q.y = (m01 + m10) / s;
        q.q.z = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.q.w = (m02 - m20) / s;
        q.q.x = (m01 + m10) / s;
        q.q.y = 0.25f * s;
        q.q.z = (m12 + m21) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.q.w = (m10 - m01) / s;
        q.q.x = (m02 + m20) / s;
        q.q.y = (m12 + m21) / s;
        q.q.z = 0.25f * s;
    }

    const float lengthSq = Diligent::dot(q.q, q.q);
    return lengthSq > 1.0e-12f ? Diligent::normalize(q)
                               : Diligent::QuaternionF{0.0f, 0.0f, 0.0f, 1.0f};
}

struct BladeToolDimensions
{
    float length = 0.0f;
    float visualHalfDepth = 0.0f;
    float visualHalfThickness = 0.0f;
    float cuttingBandHalfDepth = 0.0f;
    float cuttingHalfThickness = 0.0f;
};

BladeToolDimensions resolveBladeToolDimensions(const MeshfreeDebugOptions &options,
                                               const ParticleBounds &rotatedParticleBounds)
{
    BladeToolDimensions dimensions{};
    const float minimumReach = std::max(options.toolRadius * 8.0f, 0.01f);
    dimensions.length =
        options.bladeLength > 0.0f
            ? options.bladeLength
            : std::max(rotatedParticleBounds.extent.x * 0.70f, minimumReach);

    const float defaultDepth =
        std::max({options.toolRadius * 8.0f, rotatedParticleBounds.extent.y * 0.16f, 0.025f});
    const float visualDepth = options.bladeDepth > 0.0f ? options.bladeDepth : defaultDepth;
    dimensions.visualHalfDepth = visualDepth * 0.5f;
    dimensions.visualHalfThickness = std::max(options.bladeThickness * 0.5f, 1.0e-5f);

    const float defaultBandDepth = std::max(options.toolRadius, visualDepth * 0.12f);
    const float bandDepth =
        options.bladeCutBandDepth > 0.0f ? options.bladeCutBandDepth : defaultBandDepth;
    dimensions.cuttingBandHalfDepth =
        std::min(std::max(bandDepth * 0.5f, 1.0e-5f), dimensions.visualHalfDepth);
    dimensions.cuttingHalfThickness =
        std::max(dimensions.visualHalfThickness, options.toolRadius);
    return dimensions;
}

EdgeDebugStats computeEdgeDebugStats(const cressim::neo::physics::PhysicsWorld &physicsWorld)
{
    EdgeDebugStats stats{};
    const auto &edges = physicsWorld.softEdges();
    float strainSum = 0.0f;

    for (const cressim::neo::physics::SoftEdge &edge : edges)
    {
        const bool active = (edge.flags & cressim::neo::physics::Edge_Active) != 0u;
        const bool disabled = (edge.flags & cressim::neo::physics::Edge_Disabled) != 0u;
        stats.activeCount += active && !disabled ? 1u : 0u;
        stats.disabledCount += disabled || !active ? 1u : 0u;
        stats.cutCount += (edge.flags & cressim::neo::physics::Edge_Cut) != 0u ? 1u : 0u;
        stats.fracturedCount +=
            (edge.flags & cressim::neo::physics::Edge_Fractured) != 0u ? 1u : 0u;
        stats.maxDamage = std::max(stats.maxDamage, edge.damage);
        const float strain = std::abs(edge.strain);
        stats.maxStrain = std::max(stats.maxStrain, strain);
        strainSum += strain;
    }

    stats.averageStrain =
        !edges.empty() ? strainSum / static_cast<float>(edges.size()) : 0.0f;
    return stats;
}

void logEdgeDebugStats(const char *label, const EdgeDebugStats &stats)
{
    CRESSIM_LOG_INFO(label, ": active edges=", stats.activeCount,
                     ", disabled edges=", stats.disabledCount,
                     ", cut edges=", stats.cutCount,
                     ", fractured edges=", stats.fracturedCount,
                     ", max strain=", stats.maxStrain,
                     ", average strain=", stats.averageStrain,
                     ", max damage=", stats.maxDamage, ".\n");
}

void logSoftEdgeToolCounters(const SoftEdgeToolCounters &counters)
{
    CRESSIM_LOG_INFO("Cutting tool counters: candidates=", counters.numToolEdgeCandidates,
                     ", newly cut=", counters.numNewlyCutEdges,
                     ", already disabled=", counters.numAlreadyDisabledEdges,
                     ", active after cut=", counters.numActiveEdgesAfterCut, ".\n");
    CRESSIM_LOG_INFO("Thermal particle stats: temperature min/avg/max=",
                     counters.minimumTemperatureC, " / ", counters.averageTemperatureC,
                     " / ", counters.maximumTemperatureC,
                     " C, particles damaged=", counters.particlesAboveDamageThreshold,
                     ", damage avg/max=", counters.averageThermalDamage, " / ",
                     counters.maximumThermalDamage, ", water min/avg=",
                     counters.minimumWaterFraction, " / ", counters.averageWaterFraction,
                     ", char avg/max=", counters.averageCharFraction, " / ",
                     counters.maximumCharFraction, ".\n");
    CRESSIM_LOG_INFO("Thermal edge stats: modified=", counters.thermallyModifiedEdges,
                     ", thermally cut=", counters.thermallyCutEdges,
                     ", shrink ratio min/avg=", counters.minimumShrinkRatio, " / ",
                     counters.averageShrinkRatio, ", failure threshold min/avg=",
                     counters.minimumEffectiveFailureThreshold, " / ",
                     counters.averageEffectiveFailureThreshold, ".\n");
}

CuttingToolGPU makeCuttingTool(const MeshfreeDebugOptions &options,
                               const Diligent::float3 &bodyCenter,
                               const ParticleBounds &rotatedParticleBounds,
                               float timeSeconds)
{
    (void)timeSeconds;
    CuttingToolGPU tool{};
    if (!options.enableCuttingTool || thermalDebugEnabled(options) ||
        options.toolShape != CuttingToolShape::Blade)
    {
        return tool;
    }

    const float minimumReach = std::max(options.toolRadius * 8.0f, 0.01f);

    tool.shape = static_cast<std::uint32_t>(CuttingToolShape::Blade);
    tool.enabled = 1u;
    tool.instantCut = options.instantCut ? 1u : 0u;
    tool.strength = options.instantCut ? 1.0e6f : options.toolStrength;
    tool.cutResistanceScale = 1.0f;

    tool.tipA = bodyCenter;
    tool.tipB = {bodyCenter.x, bodyCenter.y + minimumReach, bodyCenter.z};
    tool.radius = options.toolRadius;

    const BladeToolDimensions blade = resolveBladeToolDimensions(options, rotatedParticleBounds);
    const float bladeSweepHalfHeight =
        std::max(rotatedParticleBounds.extent.y * 0.65f, options.toolRadius * 10.0f);
    const float bladeY = bodyCenter.y + std::sin(timeSeconds * 0.65f) * bladeSweepHalfHeight;
    const Diligent::float3 bladeAxisU{1.0f, 0.0f, 0.0f};
    const Diligent::float3 bladeAxisV{0.0f, -1.0f, 0.0f};
    const Diligent::float3 bladeNormal{0.0f, 0.0f, 1.0f};
    const Diligent::float3 bladeVisualCenter{
        bodyCenter.x,
        bladeY + (blade.visualHalfDepth - blade.cuttingBandHalfDepth),
        bodyCenter.z};
    tool.bladeCenter = bladeVisualCenter +
                       bladeAxisV * (blade.visualHalfDepth - blade.cuttingBandHalfDepth);
    tool.bladeHalfLength = blade.length * 0.5f;
    tool.bladeAxisU = bladeAxisU;
    tool.bladeHalfDepth = blade.cuttingBandHalfDepth;
    tool.bladeAxisV = bladeAxisV;
    tool.bladeHalfThickness = blade.cuttingHalfThickness;
    tool.bladeNormal = bladeNormal;
    return tool;
}

Diligent::float3 lerp3(const Diligent::float3 &a, const Diligent::float3 &b, float t)
{
    return a + (b - a) * std::clamp(t, 0.0f, 1.0f);
}

float safeExtent(float value)
{
    return std::max(std::abs(value), 1.0e-5f);
}

float smooth01(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

Diligent::float3 selectElectrocauteryContactTip(
    const SurfaceMeshData &surfaceMesh,
    const Diligent::float3 &softBodyPosition,
    const Diligent::QuaternionF &softBodyRotation,
    const Diligent::float3 &bodyCenter,
    const ParticleBounds &rotatedParticleBounds,
    float capsuleRadius)
{
    const Diligent::float3 fallbackTarget{
        bodyCenter.x + rotatedParticleBounds.extent.x * 0.16f,
        softBodyPosition.y + rotatedParticleBounds.max.y -
            rotatedParticleBounds.extent.y * 0.18f,
        softBodyPosition.z + rotatedParticleBounds.min.z +
            rotatedParticleBounds.extent.z * 0.18f};
    if (surfaceMesh.surfaceRestPositions.empty())
    {
        return fallbackTarget;
    }

    Diligent::float3 bestPoint = fallbackTarget;
    float bestScore = 0.0f;
    bool foundPoint = false;
    for (const Diligent::float3 &surfacePosition : surfaceMesh.surfaceRestPositions)
    {
        const Diligent::float3 worldPosition =
            softBodyPosition + softBodyRotation.RotateVector(surfacePosition);
        const float dx = (worldPosition.x - fallbackTarget.x) /
                         safeExtent(rotatedParticleBounds.extent.x);
        const float dy = (worldPosition.y - fallbackTarget.y) /
                         safeExtent(rotatedParticleBounds.extent.y);
        const float dz = (worldPosition.z - fallbackTarget.z) /
                         safeExtent(rotatedParticleBounds.extent.z);
        const float score = dx * dx + dy * dy * 1.35f + dz * dz * 0.85f;
        if (!foundPoint || score < bestScore)
        {
            bestPoint = worldPosition;
            bestScore = score;
            foundPoint = true;
        }
    }

    return bestPoint;
}

ElectrocauteryDemoState makeElectrocauteryDemoState(
    const MeshfreeDebugOptions &options,
    const Diligent::float3 &bodyCenter,
    const Diligent::float3 &softBodyPosition,
    const Diligent::QuaternionF &softBodyRotation,
    const ParticleBounds &rotatedParticleBounds,
    const SurfaceMeshData &surfaceMesh,
    const Diligent::float3 &cameraPosition,
    const Diligent::QuaternionF &cameraRotation,
    float cameraVerticalFovDegrees,
    float cameraNearClip,
    float presentationAspect,
    float groundSurfaceY)
{
    ElectrocauteryDemoState state{};
    state.capsuleRadius = std::max(options.toolRadius, 1.0e-5f);
    state.pressDepth = std::max(rotatedParticleBounds.extent.z * 0.18f, state.capsuleRadius * 2.0f);
    state.groundTipY = groundSurfaceY + state.capsuleRadius * 0.25f;
    state.retractDistance = std::max(rotatedParticleBounds.extent.z * 0.85f, state.capsuleRadius * 10.0f);

    // 1. Find the initial visual surface point
    state.contactTip = selectElectrocauteryContactTip(surfaceMesh, softBodyPosition, softBodyRotation, bodyCenter, rotatedParticleBounds, state.capsuleRadius);

    // 2. Calculate the Entry Tip FIRST to establish the tool's physical trajectory from the right
    const Diligent::float3 cameraForward = Diligent::normalize(cameraRotation.RotateVector({0.0f, 0.0f, 1.0f}));
    const Diligent::float3 cameraRight = Diligent::normalize(cameraRotation.RotateVector({1.0f, 0.0f, 0.0f}));
    const Diligent::float3 cameraUp = Diligent::normalize(cameraRotation.RotateVector({0.0f, 1.0f, 0.0f}));

    const float contactDepth = std::max(Diligent::dot(state.contactTip - cameraPosition, cameraForward), cameraNearClip + state.capsuleRadius * 4.0f);
    const float entryDepth = std::max(contactDepth * 0.70f, cameraNearClip + state.capsuleRadius * 4.0f);
    const float halfNearHeight = std::tan(cameraVerticalFovDegrees * (Diligent::PI_F / 180.0f) * 0.5f) * entryDepth;
    const float halfNearWidth = halfNearHeight * std::max(presentationAspect, 1.0e-3f);

    state.entryTip = cameraPosition + cameraForward * entryDepth + cameraRight * (halfNearWidth * 1.12f) + cameraUp * (halfNearHeight * 0.86f);

    // 3. Define the normal pointing backward along the tool's path, NOT at the camera
    state.surfaceNormal = Diligent::normalize(state.entryTip - state.contactTip);
    const Diligent::float3 inward = -state.surfaceNormal;

    // 4. SINK IT DEEP: Push the tip past the visual surface so it pierces the large XPBD particles
    const float deepSink = std::max(rotatedParticleBounds.extent.x * 0.2f, options.particleRadius * 0.5f);
    state.contactTip = state.contactTip + inward * deepSink;

    state.trocarShaftAxis = Diligent::normalize(state.entryTip - state.contactTip);
    state.groundTipY = std::min(state.groundTipY, state.contactTip.y - state.capsuleRadius);

    const float plungeTravel = std::max(state.contactTip.y - state.groundTipY, 0.0f);
    const float shaftLength = length3(state.entryTip - state.contactTip) + plungeTravel + state.retractDistance + 1.50f;
    state.capsuleHalfHeight = std::max(shaftLength * 0.5f - state.capsuleRadius, state.capsuleRadius * 8.0f);
    state.frozenTip = state.entryTip;
    state.frozenShaftAxis = state.trocarShaftAxis;

    return state;
}

ElectrocauteryToolPose makeElectrocauteryToolPose(
    const MeshfreeDebugOptions &options,
    ElectrocauteryDemoState &state,
    double timeSeconds)
{
    constexpr double kEntryDurationSeconds = 2.4;
    constexpr double kRetractDurationSeconds = 1.35;
    constexpr double kCutPlungeDurationSeconds = 3.2;
    constexpr double kCoagulationPressDurationCapSeconds = 8.0;

    if (!thermalDebugEnabled(options))
    {
        return {};
    }

    auto phaseElapsed = [&]() -> double
    {
        return std::max(0.0, timeSeconds - state.phaseStartTimeSeconds);
    };

    if (state.phase == ElectrocauteryDemoPhase::Entry &&
        phaseElapsed() >= kEntryDurationSeconds)
    {
        state.phase = options.thermalDebugMode == ThermalDebugMode::ElectrosurgicalCut
                          ? ElectrocauteryDemoPhase::Dwell
                          : ElectrocauteryDemoPhase::Press;
        state.phaseStartTimeSeconds = timeSeconds;
    }

    const Diligent::float3 inward = -state.surfaceNormal;
    Diligent::float3 distalTip = state.contactTip;
    Diligent::float3 shaftAxis = state.trocarShaftAxis;
    bool heatActive = false;
    state.tipReachedGround = false;

    switch (state.phase)
    {
    case ElectrocauteryDemoPhase::Entry:
    {
        const float t = smooth01(static_cast<float>(phaseElapsed() / kEntryDurationSeconds));
        distalTip = lerp3(state.entryTip, state.contactTip, t);
        shaftAxis = state.trocarShaftAxis;
        break;
    }
    case ElectrocauteryDemoPhase::Dwell:
    {
        distalTip = state.contactTip;
        shaftAxis = state.trocarShaftAxis;
        heatActive = true;
        break;
    }
    case ElectrocauteryDemoPhase::Plunge:
    {
        const float t = smooth01(static_cast<float>(phaseElapsed() / kCutPlungeDurationSeconds));
        
        // Sweep downwards AND pull deeply inward through the volume
        Diligent::float3 plungeOffset = {0.2f, 0.1f, 0.0f}; 
        Diligent::float3 plungeTarget = state.contactTip;
        plungeTarget.y = state.groundTipY; 
        plungeTarget = plungeTarget + plungeOffset - inward * (state.capsuleRadius * 8.0f); // Add a horizontal slicing sweep
        
        distalTip = lerp3(state.contactTip, plungeTarget, t);
        
        if (t >= 1.0f)
        {
            distalTip = plungeTarget;
            state.tipReachedGround = true;
        }
        shaftAxis = state.trocarShaftAxis;
        heatActive = true;
        break;
    }
    case ElectrocauteryDemoPhase::Press:
    {
        const double pressDurationCap = kCoagulationPressDurationCapSeconds;
        const float t = smooth01(static_cast<float>(phaseElapsed() / pressDurationCap));
        distalTip = state.contactTip + inward * (state.pressDepth * t);
        shaftAxis = state.trocarShaftAxis;
        heatActive = true;
        break;
    }
    case ElectrocauteryDemoPhase::Retract:
    {
        const float t = smooth01(static_cast<float>(phaseElapsed() / kRetractDurationSeconds));
        const Diligent::float3 retractedTip =
            state.contactTip + state.surfaceNormal * state.retractDistance;
        distalTip = lerp3(state.frozenTip, retractedTip, t);
        shaftAxis = state.frozenShaftAxis;
        if (t >= 1.0f)
        {
            state.phase = ElectrocauteryDemoPhase::Complete;
            state.phaseStartTimeSeconds = timeSeconds;
            state.frozenTip = distalTip;
        }
        break;
    }
    case ElectrocauteryDemoPhase::Complete:
        distalTip = state.frozenTip;
        shaftAxis = state.frozenShaftAxis;
        break;
    }

    const float shaftLength =
        std::max((state.capsuleHalfHeight + state.capsuleRadius) * 2.0f,
                 state.capsuleRadius * 8.0f);
    shaftAxis = Diligent::normalize(shaftAxis);

    ElectrocauteryToolPose pose{};
    pose.distalTip = distalTip;
    pose.proximalTip = distalTip + shaftAxis * shaftLength;
    pose.shaftAxis = shaftAxis;
    pose.capsuleRadius = state.capsuleRadius;
    pose.capsuleHalfHeight = state.capsuleHalfHeight;
    pose.heatActive = heatActive;

    const Diligent::float3 yAxis = shaftAxis;
    const Diligent::float3 xAxis = choosePerpendicular(yAxis);
    const Diligent::float3 zAxis = Diligent::normalize(Diligent::cross(xAxis, yAxis));
    pose.rotation = quaternionFromBasis(xAxis, yAxis, zAxis);
    state.frozenTip = distalTip;
    state.frozenShaftAxis = shaftAxis;
    return pose;
}

void requestElectrocauteryPlunge(ElectrocauteryDemoState &state, double timeSeconds,
                                 std::uint32_t thermallyCutEdges)
{
    if (state.phase != ElectrocauteryDemoPhase::Dwell)
    {
        return;
    }

    state.phase = ElectrocauteryDemoPhase::Plunge;
    state.phaseStartTimeSeconds = timeSeconds;
    state.thermallyCutEdgesAtPlungeStart = thermallyCutEdges;
    state.lastThermallyCutEdges = thermallyCutEdges;
    state.stableThermalCutReadbacks = 0u;
    state.tipReachedGround = false;
}

bool updateElectrocauteryCutColumnCompletion(ElectrocauteryDemoState &state,
                                             std::uint32_t thermallyCutEdges)
{
    if (!state.tipReachedGround ||
        thermallyCutEdges <= state.thermallyCutEdgesAtPlungeStart)
    {
        state.lastThermallyCutEdges = thermallyCutEdges;
        state.stableThermalCutReadbacks = 0u;
        return false;
    }

    if (thermallyCutEdges > state.lastThermallyCutEdges)
    {
        state.lastThermallyCutEdges = thermallyCutEdges;
        state.stableThermalCutReadbacks = 0u;
        return false;
    }

    ++state.stableThermalCutReadbacks;
    return state.stableThermalCutReadbacks >= 2u;
}

void requestElectrocauteryRetraction(ElectrocauteryDemoState &state, double timeSeconds)
{
    if (state.phase != ElectrocauteryDemoPhase::Press &&
        state.phase != ElectrocauteryDemoPhase::Plunge)
    {
        return;
    }

    state.phase = ElectrocauteryDemoPhase::Retract;
    state.phaseStartTimeSeconds = timeSeconds;
}

ElectrocauteryToolGPU makeElectrocauteryTool(const MeshfreeDebugOptions &options,
                                             const ElectrocauteryToolPose &pose)
{
    ElectrocauteryToolGPU tool{};
    if (!thermalDebugEnabled(options) || !pose.heatActive)
    {
        return tool;
    }

    const float meanParticleSpacing = std::max(options.particleRadius * 2.0f, 1.0e-5f);
    tool.query.shape = static_cast<std::uint32_t>(CuttingToolShape::Capsule);
    tool.query.enabled = 1u;
    tool.query.tipA = pose.proximalTip;
    tool.query.tipB = pose.distalTip;
    tool.query.radius = pose.capsuleRadius;

    switch (options.thermalDebugMode)
    {
    case ThermalDebugMode::ElectrosurgicalCut:
        tool.mode = static_cast<std::uint32_t>(ElectrocauteryToolMode::ElectrosurgicalCut);
        tool.activeTipLength = std::max(1.4f * meanParticleSpacing, pose.capsuleRadius * 2.0f);
        tool.heatRadius = std::max(1.65f * meanParticleSpacing, pose.capsuleRadius * 1.7f);
        tool.ablationRadius = std::max(0.55f * meanParticleSpacing, pose.capsuleRadius * 0.65f);
        tool.heatingRateCPerSecond = 420.0f;
        tool.ablationInfluenceThreshold = 0.22f;
        tool.cutMode.powerDensity = 1.6e9f;
        tool.cutMode.heatingRadius = std::max(tool.heatRadius, tool.ablationRadius * 2.2f);
        tool.cutMode.falloffExponent = 2.4f;
        tool.cutMode.thermalCutEnabled = 1u;
        tool.cutMode.shrinkageScale = 0.30f;
        tool.cutMode.charScale = 0.90f;
        tool.cutMode.waterLossScale = 3.2f;
        break;
    case ThermalDebugMode::ElectrosurgicalCoagulation:
        tool.mode =
            static_cast<std::uint32_t>(ElectrocauteryToolMode::ElectrosurgicalCoagulation);
        tool.activeTipLength = std::max(2.0f * meanParticleSpacing, pose.capsuleRadius * 2.5f);
        tool.heatRadius = std::max(2.8f * meanParticleSpacing, pose.capsuleRadius * 2.8f);
        tool.ablationRadius = 0.0f;
        tool.heatingRateCPerSecond = 185.0f;
        tool.ablationInfluenceThreshold = 1.0f;
        tool.coagulationMode.powerDensity = 7.0e8f;
        tool.coagulationMode.heatingRadius = tool.heatRadius;
        tool.coagulationMode.falloffExponent = 1.2f;
        tool.coagulationMode.thermalCutEnabled = 0u;
        tool.coagulationMode.shrinkageScale = 1.35f;
        tool.coagulationMode.charScale = 1.8f;
        tool.coagulationMode.waterLossScale = 1.45f;
        break;
    case ThermalDebugMode::ElectrosurgicalBlend:
        tool.mode = static_cast<std::uint32_t>(ElectrocauteryToolMode::ElectrosurgicalBlend);
        tool.activeTipLength = std::max(1.8f * meanParticleSpacing, pose.capsuleRadius * 2.2f);
        tool.heatRadius = std::max(2.2f * meanParticleSpacing, pose.capsuleRadius * 2.2f);
        tool.ablationRadius = std::max(0.65f * meanParticleSpacing, pose.capsuleRadius * 0.75f);
        tool.heatingRateCPerSecond = 185.0f;
        tool.ablationInfluenceThreshold = 0.5f;
        tool.blendMode.powerDensity = 8.2e8f;
        tool.blendMode.heatingRadius = std::max(tool.heatRadius, tool.ablationRadius * 2.0f);
        tool.blendMode.falloffExponent = 1.6f;
        tool.blendMode.thermalCutEnabled = 1u;
        tool.blendMode.shrinkageScale = 0.65f;
        tool.blendMode.charScale = 1.1f;
        tool.blendMode.waterLossScale = 2.0f;
        break;
    case ThermalDebugMode::None:
        break;
    }

    return tool;
}

TransformComponent makeElectrocauteryToolTransform(const ElectrocauteryToolPose &pose)
{
    TransformComponent transform{};
    transform.worldTransform.position =
        pose.distalTip + pose.shaftAxis * (pose.capsuleHalfHeight + pose.capsuleRadius);
    transform.worldTransform.rotation = pose.rotation;
    return transform;
}

void configureThermalMaterialForDebug(const MeshfreeDebugOptions &options,
                                      SoftThermalMaterialDesc &material)
{
    if (!thermalDebugEnabled(options))
    {
        return;
    }

    material.bodyTemperatureC = 37.0f;
    material.maximumTemperatureC = 250.0f;
    material.diffusionRate = 4.5f;
    material.coolingRate = 0.25f;
    material.damageStartTemperatureC = 60.0f;
    material.damageFullTemperatureC = 100.0f;
    material.damageRate = 2.0f;
    material.evaporationStartTemperatureC = 90.0f;
    material.evaporationTransitionWidthC = 15.0f;
    material.evaporationRate = 2.0f;
    material.charStartTemperatureC = 140.0f;
    material.charFullTemperatureC = 220.0f;
    material.charRate = 1.0f;
    material.shrinkageRate = 2.0f;
    material.shrinkDamageStart = 0.15f;
    material.shrinkDamageFull = 0.85f;
    material.minimumFailureThresholdScale = 0.25f;
    material.minimumCutResistanceScale = 0.20f;
    material.thermalCutDamageThreshold = 0.90f;
    material.thermalCutWaterThreshold = 0.18f;
    material.maximumComplianceMultiplier = 4.0f;

    switch (options.thermalDebugMode)
    {
    case ThermalDebugMode::ElectrosurgicalCut:
        material.damageRate = 6.0f;
        material.evaporationRate = 0.35f;
        material.charStartTemperatureC = 220.0f;
        material.charFullTemperatureC = 250.0f;
        material.charRate = 0.12f;
        material.maximumShrinkage = 0.06f;
        material.minimumFailureThresholdScale = 0.20f;
        material.minimumCutResistanceScale = 0.12f;
        material.thermalCutDamageThreshold = 0.55f;
        material.thermalCutWaterThreshold = 0.98f;
        break;
    case ThermalDebugMode::ElectrosurgicalCoagulation:
        material.maximumShrinkage = 0.3f;
        material.damageRate = 4.0f;
        material.evaporationRate = 2.5f;
        material.charStartTemperatureC = 125.0f;
        material.charFullTemperatureC = 205.0f;
        material.charRate = 2.0f;
        material.minimumFailureThresholdScale = 0.35f;
        material.thermalCutDamageThreshold = 0.95f;
        break;
    case ThermalDebugMode::ElectrosurgicalBlend:
        material.maximumShrinkage = 0.15f;
        material.damageRate = 2.4f;
        material.charRate = 1.0f;
        break;
    case ThermalDebugMode::None:
        break;
    }
}

TransformComponent makeCuttingToolTransform(const MeshfreeDebugOptions &options,
                                            const CuttingToolGPU &tool,
                                            const ParticleBounds &rotatedParticleBounds)
{
    TransformComponent transform{};
    if (tool.shape == static_cast<std::uint32_t>(CuttingToolShape::Blade))
    {
        const BladeToolDimensions blade = resolveBladeToolDimensions(options, rotatedParticleBounds);
        transform.worldTransform.position =
            tool.bladeCenter -
            tool.bladeAxisV * (blade.visualHalfDepth - blade.cuttingBandHalfDepth);
        transform.worldTransform.rotation =
            quaternionFromBasis(tool.bladeAxisU, tool.bladeAxisV, tool.bladeNormal);
        return transform;
    }

    return transform;
}

void logCuttingToolDebug(const CuttingToolGPU &tool)
{
    if (tool.enabled == 0u)
    {
        return;
    }

    if (tool.shape == static_cast<std::uint32_t>(CuttingToolShape::Blade))
    {
        CRESSIM_LOG_INFO("Cutting tool blade: center=(", tool.bladeCenter.x, ", ",
                         tool.bladeCenter.y, ", ", tool.bladeCenter.z,
                         "), halfExtents=(", tool.bladeHalfLength, ", ",
                         tool.bladeHalfDepth, ", ", tool.bladeHalfThickness,
                         "), strength=", tool.strength, ".\n");
        return;
    }

}

bool edgeIntersectsDisableSelection(const MeshfreeDebugOptions &options,
                                    const Diligent::float3 &a,
                                    const Diligent::float3 &b,
                                    float planeX)
{
    if (options.disableEdgeTest && ((a.x <= planeX && b.x >= planeX) ||
                                    (b.x <= planeX && a.x >= planeX)))
    {
        return true;
    }

    if (!options.disableEdgeRegion)
    {
        return false;
    }

    return segmentDistanceToPoint(a, b, options.disableEdgeRegionCenter) <=
           options.disableEdgeRegionRadius;
}

std::uint32_t applyManualEdgeDisabling(Runtime &runtime, cressim::neo::common::EntityId softEntity,
                                       const MeshfreeDebugOptions &options, float planeX)
{
    if (!options.disableEdgeTest && !options.disableEdgeRegion)
    {
        return 0u;
    }

    auto &physicsWorld = runtime.getWorld().physicsWorld();
    physicsWorld.ensureDerivedStateUpToDate();
    const cressim::neo::physics::SoftBodyState *softState =
        physicsWorld.tryGetSoftBody(softEntity);
    if (softState == nullptr)
    {
        return 0u;
    }

    const auto &particles = physicsWorld.particles();
    const auto &edges = physicsWorld.softEdges();
    const std::uint32_t edgeEnd =
        std::min(softState->edgeOffset + softState->edgeCount,
                 static_cast<std::uint32_t>(edges.size()));
    std::uint32_t selectedEdgeCount = 0u;

    for (std::uint32_t edgeIndex = softState->edgeOffset; edgeIndex < edgeEnd; ++edgeIndex)
    {
        const cressim::neo::physics::SoftEdge &edge = edges[edgeIndex];
        if (edge.particleA >= particles.positionsInvMass.size() ||
            edge.particleB >= particles.positionsInvMass.size())
        {
            continue;
        }

        const Diligent::float4 &a4 = particles.positionsInvMass[edge.particleA];
        const Diligent::float4 &b4 = particles.positionsInvMass[edge.particleB];
        const Diligent::float3 a{a4.x, a4.y, a4.z};
        const Diligent::float3 b{b4.x, b4.y, b4.z};
        const bool selected =
            edgeIntersectsDisableSelection(options, a, b, planeX) ||
            (options.enableCuttingTool &&
             segmentDistanceToPoint(a, b, options.disableEdgeRegionCenter) <= options.toolRadius);
        if (!selected)
        {
            continue;
        }

        cressim::neo::physics::SoftEdge updated = edge;
        updated.flags |= cressim::neo::physics::Edge_Cut;
        if (options.instantCut || options.disableEdgeTest || options.disableEdgeRegion)
        {
            updated.damage = 1.0f;
        }
        else
        {
            updated.damage = std::clamp(
                updated.damage + options.toolStrength / std::max(updated.cutResistance, 1.0e-6f),
                0.0f, 1.0f);
        }
        if (updated.damage >= 1.0f)
        {
            updated.flags |= cressim::neo::physics::Edge_Disabled;
            updated.flags &= ~cressim::neo::physics::Edge_Active;
        }
        ++selectedEdgeCount;
        physicsWorld.setSoftEdgeState(edgeIndex, updated);
    }

    return selectedEdgeCount;
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

void applyDebugParticleGraphOptions(Runtime &runtime, const MeshfreeDebugOptions &options,
                                    const std::uint32_t shapeMaxMembershipCount)
{
    cressim::neo::graphics::RenderFrameOptions renderOptions = runtime.renderFrameOptions();
    renderOptions.debugParticles.enabled                  = options.showDebugParticles;
    renderOptions.debugParticles.drawConstraintEdges      = options.drawConstraintEdges;
    renderOptions.debugParticles.highlightStaticParticles = true;
    renderOptions.debugParticles.useParticleRadii         = true;
    renderOptions.debugParticles.shapeMatchingModes =
        options.drawShapeClusters
            ? (cressim::neo::graphics::DebugShapeMatching_MembershipCount |
               cressim::neo::graphics::DebugShapeMatching_CorrectionMagnitude)
            : 0u;
    renderOptions.debugParticles.shapeMaxMembershipCount  = shapeMaxMembershipCount;
    renderOptions.debugParticles.shapeCorrectionScale     = options.shapeCorrectionDebugScale;
    renderOptions.debugParticles.color                    = {0.18f, 0.74f, 1.0f, 1.0f};
    renderOptions.debugParticles.staticColor              = {1.0f, 0.22f, 0.12f, 1.0f};
    renderOptions.debugParticles.edgeColor                = {1.0f, 1.0f, 1.0f, 1.0f};
    renderOptions.debugParticles.edgeHighStrainColor      = {1.0f, 0.08f, 0.04f, 1.0f};
    renderOptions.debugParticles.edgeDamagedColor         = {1.0f, 0.48f, 0.04f, 1.0f};
    renderOptions.debugParticles.edgeDisabledColor        = {0.0f, 0.0f, 0.0f, 1.0f};
    renderOptions.debugParticles.showCutEdges             = options.showCutEdges;
    renderOptions.debugParticles.showStrain               = options.showStrain;
    renderOptions.debugParticles.showDamage               = options.showDamage;
    renderOptions.debugParticles.showThermalHeatmap       = thermalHeatmapEnabled(options);
    renderOptions.debugParticles.highStrainThreshold      = options.fractureThreshold;
    renderOptions.debugParticles.fallbackRadius           = options.particleRadius;
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
            if (arg == "--draw-shape-clusters")
            {
                options.drawShapeClusters = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
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
            if (arg == "--disable-edge-test")
            {
                options.disableEdgeTest = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--disable-edge-region")
            {
                options.disableEdgeRegionCenter =
                    parseFloat3(argc, argv, i, "--disable-edge-region");
                options.disableEdgeRegionRadius = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--disable-edge-region"),
                    "--disable-edge-region");
                options.disableEdgeRegion = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--enable-fracture")
            {
                options.enableFracture = true;
                options.showDamage = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--fracture-threshold")
            {
                options.fractureThreshold = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--fracture-threshold"),
                    "--fracture-threshold");
                continue;
            }
            if (arg == "--enable-cutting-tool")
            {
                options.enableCuttingTool = true;
                continue;
            }
            if (arg == "--tool-shape")
            {
                options.toolShape = parseToolShape(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--tool-shape"));
                continue;
            }
            if (arg == "--tool-radius")
            {
                options.toolRadius = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--tool-radius"),
                    "--tool-radius");
                continue;
            }
            if (arg == "--tool-strength")
            {
                options.toolStrength = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--tool-strength"),
                    "--tool-strength");
                continue;
            }
            if (arg == "--instant-cut")
            {
                options.instantCut = true;
                options.enableCuttingTool = true;
                continue;
            }
            if (arg == "--blade-length")
            {
                options.bladeLength = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--blade-length"),
                    "--blade-length");
                continue;
            }
            if (arg == "--blade-depth")
            {
                options.bladeDepth = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--blade-depth"),
                    "--blade-depth");
                continue;
            }
            if (arg == "--blade-thickness")
            {
                options.bladeThickness = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--blade-thickness"),
                    "--blade-thickness");
                continue;
            }
            if (arg == "--blade-cut-band-depth")
            {
                options.bladeCutBandDepth = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--blade-cut-band-depth"),
                    "--blade-cut-band-depth");
                continue;
            }
            if (arg == "--show-cut-edges")
            {
                options.showCutEdges = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--show-strain")
            {
                options.showStrain = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--show-damage")
            {
                options.showDamage = true;
                options.drawConstraintEdges = true;
                options.showDebugParticles = true;
                options.debugParticlesExplicit = true;
                continue;
            }
            if (arg == "--thermal-debug")
            {
                options.thermalDebugMode = parseThermalDebugMode(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--thermal-debug"));
                options.enableCuttingTool = true;
                continue;
            }
            if (arg == "--thermal-visual")
            {
                options.thermalVisualizationMode = parseThermalVisualizationMode(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--thermal-visual"));
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

    if (thermalHeatmapEnabled(options) && !options.debugParticlesExplicit)
    {
        options.showDebugParticles = true;
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
    const MaterialHandle cuttingToolMaterial =
        registerMaterial(resources, "MeshfreeDebug.CuttingTool", {0.10f, 0.82f, 0.92f}, 0.28f);

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
    const Diligent::float3 cuttingToolBodyCenter =
        softTransform.worldTransform.position + rotatedParticleBounds.center;

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
    softBody.edgeFailureThreshold            = options.enableFracture
                                                   ? options.fractureThreshold
                                                   : 1.0e6f;
    softBody.edgeCutResistance               = options.instantCut
                                                   ? 1.0e-6f
                                                   : 1.0f;
    softBody.material.contact.friction       = 0.45f;
    softBody.material.contact.staticFriction = 0.60f;
    softBody.material.contact.damping        = options.damping >= 0.0f
                                                   ? options.damping
                                                   : 6.00f;
    configureThermalMaterialForDebug(options, softBody.material.thermal);
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

    CuttingToolGPU currentCuttingTool =
        makeCuttingTool(options, cuttingToolBodyCenter, rotatedParticleBounds, 0.0f);
    world.physicsWorld().setCuttingTool(currentCuttingTool);
    ElectrocauteryDemoState electrocauteryDemoState =
        makeElectrocauteryDemoState(options, cuttingToolBodyCenter,
                                    softTransform.worldTransform.position, softRotation,
                                    rotatedParticleBounds, surfaceMesh,
                                    cameraTransform.worldTransform.position,
                                    cameraTransform.worldTransform.rotation,
                                    camera.verticalFovDegrees, camera.nearClip,
                                    static_cast<float>(viewerDesc.width) /
                                        static_cast<float>(std::max(viewerDesc.height, 1u)),
                                    kGroundSurfaceY);
    ElectrocauteryToolPose currentElectrocauteryPose =
        makeElectrocauteryToolPose(options, electrocauteryDemoState, 0.0);
    world.physicsWorld().setElectrocauteryTool(
        makeElectrocauteryTool(options, currentElectrocauteryPose));
    cressim::neo::common::EntityId cuttingToolEntity =
        cressim::neo::common::kInvalidEntityId;
    if (thermalDebugEnabled(options))
    {
        MeshHandle cuttingToolMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeCapsuleMesh(
                currentElectrocauteryPose.capsuleRadius,
                currentElectrocauteryPose.capsuleHalfHeight,
                16u, 4u, 1u, "MeshfreeDebug.ElectrocauteryCapsule"));
        cuttingToolEntity = world.createEntity();
        world.setTransform(cuttingToolEntity,
                           makeElectrocauteryToolTransform(currentElectrocauteryPose));
        world.setMeshRenderer(cuttingToolEntity,
                              MeshRendererComponent{cuttingToolMesh, cuttingToolMaterial, true});
    }
    else if (options.enableCuttingTool && options.toolShape == CuttingToolShape::Blade)
    {
        const BladeToolDimensions blade =
            resolveBladeToolDimensions(options, rotatedParticleBounds);
        MeshHandle cuttingToolMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeBoxMesh(
                Diligent::float3{blade.length * 0.5f, blade.visualHalfDepth,
                                 blade.visualHalfThickness},
                "MeshfreeDebug.CuttingToolBlade"));
        cuttingToolEntity = world.createEntity();
        world.setTransform(cuttingToolEntity,
                           makeCuttingToolTransform(options, currentCuttingTool,
                                                    rotatedParticleBounds));
        world.setMeshRenderer(cuttingToolEntity,
                              MeshRendererComponent{cuttingToolMesh, cuttingToolMaterial, true});
        logCuttingToolDebug(currentCuttingTool);
    }

    const std::uint32_t disabledEdgeCount = applyManualEdgeDisabling(
        runtime, softEntity, options, softTransform.worldTransform.position.x);
    if (disabledEdgeCount > 0u)
    {
        CRESSIM_LOG_INFO("Manual edge disabling selected ", disabledEdgeCount,
                         " edges. Disabled/cut edges are skipped by the XPBD distance solve.\n");
    }

    MeshHandle surfaceShellMesh{};
    MeshResourceDesc surfaceShellMeshResource{};
    std::vector<std::uint32_t> surfaceShellOriginalIndices;
    const bool surfaceShellActive =
        thermalPbrSurfaceEnabled(options) && !options.drawConstraintEdges &&
        !surfaceMesh.surfaceRestPositions.empty();
    if (surfaceShellActive)
    {
        surfaceShellMeshResource =
            makeSurfaceMeshResource(surfaceMesh, "MeshfreeDebug.CubeSurfaceShellMesh");
        surfaceShellOriginalIndices = surfaceShellMeshResource.indices;
        surfaceShellMesh = resources.registerMesh(surfaceShellMeshResource);
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
        logEdgeDebugStats("Meshfree XPBD edge state",
                          computeEdgeDebugStats(world.physicsWorld()));
    }

    applyDebugParticleGraphOptions(runtime, options,
                                   shapeStats.maximumMembershipsPerParticle);

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&options, shapeStats, cuttingToolBodyCenter, rotatedParticleBounds,
                            cuttingToolEntity, &electrocauteryDemoState](
                               const cressim::neo::common::FrameContext &frame,
                               Runtime &callbackRuntime)
    {
        applyDebugParticleGraphOptions(callbackRuntime, options,
                                       shapeStats.maximumMembershipsPerParticle);
        CuttingToolGPU tool{};
        if (thermalDebugEnabled(options))
        {
            ElectrocauteryToolPose pose =
                makeElectrocauteryToolPose(options, electrocauteryDemoState,
                                           frame.timeSeconds);
            callbackRuntime.getWorld().physicsWorld().setCuttingTool(tool);
            callbackRuntime.getWorld().physicsWorld().setElectrocauteryTool(
                makeElectrocauteryTool(options, pose));
            if (cuttingToolEntity != cressim::neo::common::kInvalidEntityId)
            {
                callbackRuntime.getWorld().setTransform(
                    cuttingToolEntity, makeElectrocauteryToolTransform(pose));
            }
            return;
        }

        tool = makeCuttingTool(options, cuttingToolBodyCenter, rotatedParticleBounds,
                               static_cast<float>(frame.timeSeconds));
        callbackRuntime.getWorld().physicsWorld().setCuttingTool(tool);
        callbackRuntime.getWorld().physicsWorld().setElectrocauteryTool({});
        if (cuttingToolEntity != cressim::neo::common::kInvalidEntityId)
        {
            callbackRuntime.getWorld().setTransform(
                cuttingToolEntity,
                makeCuttingToolTransform(options, tool, rotatedParticleBounds));
        }
    };
    callbacks.afterTick = [&options, &viewerDesc, &resources, softEntity, &surfaceShellMesh,
                           surfaceShellActive, &surfaceShellMeshResource,
                           &surfaceShellOriginalIndices,
                           &electrocauteryDemoState](
                              const cressim::neo::common::FrameContext &frame,
                              Runtime &callbackRuntime)
    {
        const bool hasMutableEdges =
            options.disableEdgeTest || options.disableEdgeRegion || options.enableFracture ||
            options.enableCuttingTool || thermalDebugEnabled(options);
        const bool shouldLog = frame.frameIndex == 1u || (frame.frameIndex % 120u) == 0u;
        const bool shouldRefreshCutSurface = hasMutableEdges && (frame.frameIndex % 4u) == 0u;
        if ((options.drawConstraintEdges || hasMutableEdges) &&
            (shouldLog || shouldRefreshCutSurface))
        {
            auto &physicsWorld = callbackRuntime.getWorld().physicsWorld();
            SoftEdgeToolCounters counters{};
            auto *solver = callbackRuntime.getPhysicsSolver();
            if (solver == nullptr ||
                !solver->readbackSoftEdgeDebugStateBlocking(physicsWorld, counters))
            {
                CRESSIM_LOG_WARNING("Soft-edge GPU diagnostics readback failed.\n");
                return;
            }

            if (thermalDebugEnabled(options) &&
                options.thermalDebugMode == ThermalDebugMode::ElectrosurgicalCut)
            {
                if (electrocauteryDemoState.phase == ElectrocauteryDemoPhase::Dwell &&
                    counters.maximumThermalDamage >=
                        electrocauteryDemoState.cutPunctureDamageThreshold)
                {
                    requestElectrocauteryPlunge(electrocauteryDemoState,
                                                frame.timeSeconds,
                                                counters.thermallyCutEdges);
                }
                else if (electrocauteryDemoState.phase == ElectrocauteryDemoPhase::Plunge &&
                         updateElectrocauteryCutColumnCompletion(
                             electrocauteryDemoState,
                             counters.thermallyCutEdges))
                {
                    requestElectrocauteryRetraction(electrocauteryDemoState,
                                                    frame.timeSeconds);
                }
            }
            else if (thermalDebugEnabled(options) &&
                     electrocauteryDemoState.phase == ElectrocauteryDemoPhase::Press)
            {
                const bool coagulationComplete =
                    options.thermalDebugMode == ThermalDebugMode::ElectrosurgicalCoagulation &&
                    counters.maximumCharFraction >= 0.98f;
                if (coagulationComplete)
                {
                    requestElectrocauteryRetraction(electrocauteryDemoState,
                                                    frame.timeSeconds);
                }
            }

            std::vector<std::uint32_t> surfaceVertexComponents;
            const std::uint32_t rejectedSurfaceWeights =
                hasMutableEdges
                    ? physicsWorld.validateSoftRenderSkinningAgainstActiveEdges(
                          surfaceShellActive ? &surfaceVertexComponents : nullptr)
                    : 0u;
            std::uint32_t culledSurfaceTriangles = 0u;
            if (surfaceShellActive && !surfaceVertexComponents.empty() &&
                !surfaceShellOriginalIndices.empty())
            {
                const std::vector<std::uint32_t> beforeIndices = surfaceShellMeshResource.indices;
                const bool thermalCutSurfaceCulling =
                    thermalDebugEnabled(options) &&
                    options.thermalDebugMode == ThermalDebugMode::ElectrosurgicalCut &&
                    counters.thermallyCutEdges > 0u;
                const std::unordered_set<std::uint64_t> thermalCutParticlePairs =
                    thermalCutSurfaceCulling
                        ? buildThermalCutParticlePairs(physicsWorld.softEdges())
                        : std::unordered_set<std::uint64_t>{};
                culledSurfaceTriangles = filterSurfaceMeshTrianglesByTopology(
                    surfaceShellMeshResource, surfaceShellOriginalIndices,
                    surfaceVertexComponents, thermalCutSurfaceCulling
                                                 ? &physicsWorld.softRenderData()
                                                 : nullptr,
                    thermalCutSurfaceCulling ? &thermalCutParticlePairs : nullptr);
                if (surfaceShellMeshResource.indices != beforeIndices)
                {
                    MeshHandle filteredSurfaceMesh = resources.registerMesh(surfaceShellMeshResource);
                    if (callbackRuntime.getWorld().setRenderableMeshResource(
                            softEntity, filteredSurfaceMesh))
                    {
                        surfaceShellMesh = filteredSurfaceMesh;
                    }
                    else
                    {
                        CRESSIM_LOG_WARNING("Failed to switch cut surface mesh resource.\n");
                    }
                }
            }
            if (shouldLog)
            {
                logCuttingToolDebug(physicsWorld.cuttingTool());
                logEdgeDebugStats("Meshfree XPBD GPU edge state",
                                  computeEdgeDebugStats(physicsWorld));
                logSoftEdgeToolCounters(counters);
                if (rejectedSurfaceWeights > 0u)
                {
                    CRESSIM_LOG_INFO("Topology-aware skinning rejected ",
                                     rejectedSurfaceWeights,
                                     " cross-component surface weights.\n");
                }
                if (culledSurfaceTriangles > 0u)
                {
                    CRESSIM_LOG_INFO("Topology-aware surface culling removed ",
                                     culledSurfaceTriangles,
                                     " cross-component triangles.\n");
                }
            }
        }

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
