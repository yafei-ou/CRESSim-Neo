#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/asset_paths.h"
#include "helpers/viewer_example.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
using cressim::neo::graphics::EnvironmentIblBakeOptions;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kEnvSpacing              = 96.0f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr const char *kToroidSkyboxCrossPath =
    "environments/cubemaps/Cubemap/Cubemap_Sky_05-512x512.png";

struct SceneMaterials
{
    MaterialHandle ground{};
    MaterialHandle obstacle{};
    MaterialHandle softBody{};
};

struct EnvironmentTuning
{
    Diligent::float3 softBodyColor{0.86f, 0.33f, 0.26f};
    float groundFriction         = 0.55f;
    float obstacleFriction       = 0.45f;
    float softFriction           = 0.35f;
    float softRestitution        = 0.02f;
    float softDamping            = 0.6f;
    float particleMass           = 0.02f;
    float particleRadius         = 0.35f;
    float edgeCompliance         = 0.00035f;
    float volumeCompliance       = 0.0018f;
};

struct ToroidVariantTuning
{
    float friction         = 0.35f;
    float restitution      = 0.02f;
    float damping          = 0.6f;
    float particleMass     = 0.02f;
    float particleRadius   = 0.35f;
    float edgeCompliance   = 0.00035f;
    float volumeCompliance = 0.0018f;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, " [--toroids N]", true);
}

std::int32_t resolveObjIndex(std::int32_t index, std::size_t count)
{
    if (index > 0)
    {
        return index - 1;
    }
    if (index < 0)
    {
        return static_cast<std::int32_t>(count) + index;
    }
    return -1;
}

bool parseObjFaceVertex(const std::string &token, std::int32_t &positionIndex,
                        std::int32_t &texcoordIndex, std::int32_t &normalIndex)
{
    positionIndex = -1;
    texcoordIndex = -1;
    normalIndex   = -1;
    const std::size_t firstSlash = token.find('/');
    if (firstSlash == std::string::npos)
    {
        positionIndex = std::stoi(token);
        return true;
    }

    const std::size_t secondSlash = token.find('/', firstSlash + 1u);
    positionIndex                 = std::stoi(token.substr(0u, firstSlash));
    if (secondSlash == std::string::npos)
    {
        const std::string texcoordToken = token.substr(firstSlash + 1u);
        if (!texcoordToken.empty())
        {
            texcoordIndex = std::stoi(texcoordToken);
        }
        return true;
    }

    if (secondSlash > firstSlash + 1u)
    {
        texcoordIndex = std::stoi(token.substr(firstSlash + 1u, secondSlash - firstSlash - 1u));
    }
    if (secondSlash + 1u < token.size())
    {
        normalIndex = std::stoi(token.substr(secondSlash + 1u));
    }
    return true;
}

MeshResourceDesc loadObjMesh(const std::filesystem::path &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open OBJ file: " + path.string());
    }

    MeshResourceDesc mesh{};
    mesh.debugName = "SoftParticleToroidMultiEnv.SurfaceOBJ";

    std::vector<Diligent::float3> positions;
    std::vector<Diligent::float3> normals;
    std::vector<Diligent::float2> texcoords;
    std::unordered_map<std::string, std::uint32_t> vertexCache;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;
        if (keyword == "v")
        {
            Diligent::float3 position{};
            lineStream >> position.x >> position.y >> position.z;
            positions.push_back(position);
        }
        else if (keyword == "vn")
        {
            Diligent::float3 normal{};
            lineStream >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (keyword == "vt")
        {
            Diligent::float2 texcoord{};
            lineStream >> texcoord.x >> texcoord.y;
            texcoords.push_back(texcoord);
        }
        else if (keyword == "f")
        {
            std::vector<std::string> faceVertices;
            std::string token;
            while (lineStream >> token)
            {
                faceVertices.push_back(token);
            }
            if (faceVertices.size() < 3u)
            {
                continue;
            }

            const auto emitVertex = [&](const std::string &faceToken) -> std::uint32_t {
                if (const auto it = vertexCache.find(faceToken); it != vertexCache.end())
                {
                    return it->second;
                }

                std::int32_t positionIndex = -1;
                std::int32_t texcoordIndex = -1;
                std::int32_t normalIndex   = -1;
                parseObjFaceVertex(faceToken, positionIndex, texcoordIndex, normalIndex);
                positionIndex = resolveObjIndex(positionIndex, positions.size());
                texcoordIndex = resolveObjIndex(texcoordIndex, texcoords.size());
                normalIndex   = resolveObjIndex(normalIndex, normals.size());
                if (positionIndex < 0 ||
                    static_cast<std::size_t>(positionIndex) >= positions.size())
                {
                    throw std::runtime_error("OBJ face references an invalid position index: " +
                                             path.string());
                }

                MeshResourceDesc::Vertex vertex{};
                vertex.position = positions[static_cast<std::size_t>(positionIndex)];
                if (texcoordIndex >= 0 && static_cast<std::size_t>(texcoordIndex) < texcoords.size())
                {
                    const Diligent::float2 texcoord =
                        texcoords[static_cast<std::size_t>(texcoordIndex)];
                    vertex.texCoordU = texcoord.x;
                    vertex.texCoordV = texcoord.y;
                }
                if (normalIndex >= 0 && static_cast<std::size_t>(normalIndex) < normals.size())
                {
                    vertex.normal = normals[static_cast<std::size_t>(normalIndex)];
                }

                const std::uint32_t index = static_cast<std::uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vertex);
                vertexCache.emplace(faceToken, index);
                return index;
            };

            const std::uint32_t i0 = emitVertex(faceVertices[0]);
            for (std::size_t i = 1u; i + 1u < faceVertices.size(); ++i)
            {
                const std::uint32_t i1 = emitVertex(faceVertices[i]);
                const std::uint32_t i2 = emitVertex(faceVertices[i + 1u]);
                mesh.indices.insert(mesh.indices.end(), {i0, i1, i2});
            }
        }
    }

    for (std::size_t triangleBase = 0u; triangleBase + 2u < mesh.indices.size();
         triangleBase += 3u)
    {
        std::swap(mesh.indices[triangleBase + 1u], mesh.indices[triangleBase + 2u]);
    }

    for (auto &vertex : mesh.vertices)
    {
        vertex.normal = {0.0f, 0.0f, 0.0f};
    }

    for (std::size_t triangleBase = 0u; triangleBase + 2u < mesh.indices.size();
         triangleBase += 3u)
    {
        const std::uint32_t i0 = mesh.indices[triangleBase + 0u];
        const std::uint32_t i1 = mesh.indices[triangleBase + 1u];
        const std::uint32_t i2 = mesh.indices[triangleBase + 2u];
        const Diligent::float3 e0 = mesh.vertices[i1].position - mesh.vertices[i0].position;
        const Diligent::float3 e1 = mesh.vertices[i2].position - mesh.vertices[i0].position;
        const Diligent::float3 faceNormal = Diligent::normalize(Diligent::cross(e1, e0));
        mesh.vertices[i0].normal += faceNormal;
        mesh.vertices[i1].normal += faceNormal;
        mesh.vertices[i2].normal += faceNormal;
    }

    for (auto &vertex : mesh.vertices)
    {
        if (Diligent::dot(vertex.normal, vertex.normal) > 1.0e-8f)
        {
            vertex.normal = Diligent::normalize(vertex.normal);
        }
        else
        {
            vertex.normal = {0.0f, 1.0f, 0.0f};
        }
    }

    if (mesh.vertices.empty() || mesh.indices.size() < 3u)
    {
        throw std::runtime_error("OBJ mesh is empty or invalid: " + path.string());
    }

    return mesh;
}

std::filesystem::path fixturePath(const char *name)
{
    return std::filesystem::path(__FILE__).parent_path() / "fixtures" / name;
}

Diligent::float3 envOrigin(std::uint32_t envIndex, std::uint32_t envCount)
{
    const std::uint32_t cols = std::max(
        1u, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<float>(envCount)))));
    const std::uint32_t rows = std::max(1u, (envCount + cols - 1u) / cols);
    const std::uint32_t col  = envIndex % cols;
    const std::uint32_t row  = envIndex / cols;
    const float xCenter      = (static_cast<float>(cols) - 1.0f) * 0.5f;
    const float zCenter      = (static_cast<float>(rows) - 1.0f) * 0.5f;
    return {(static_cast<float>(col) - xCenter) * kEnvSpacing, 0.0f,
            (static_cast<float>(row) - zCenter) * kEnvSpacing};
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

EnvironmentIblDesc loadToroidSkyboxIbl(cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path crossPath =
        cressim::neo::examples::helpers::assetPath(kToroidSkyboxCrossPath);

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 0.20f;
    options.backgroundIntensity = 1.00f;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
}

EnvironmentTuning tuningForEnvironment(std::uint32_t envIndex)
{
    const float phase = static_cast<float>(envIndex) * 0.79f;
    EnvironmentTuning tuning{};
    tuning.softBodyColor = Diligent::float3{
        0.55f + 0.25f * (0.5f + 0.5f * std::sin(phase + 0.2f)),
        0.25f + 0.35f * (0.5f + 0.5f * std::sin(phase + 2.1f)),
        0.20f + 0.45f * (0.5f + 0.5f * std::sin(phase + 4.0f))};
    tuning.groundFriction   = 0.60f + 0.25f * (0.5f + 0.5f * std::sin(phase + 0.5f));
    tuning.obstacleFriction = 0.25f + 0.35f * (0.5f + 0.5f * std::cos(phase + 0.8f));
    tuning.softFriction     = 0.30f + 0.75f * (0.5f + 0.5f * std::sin(phase + 1.3f));
    tuning.softRestitution  = 0.02f + 0.28f * (0.5f + 0.5f * std::cos(phase + 0.1f));
    tuning.softDamping      = 0.05f + 0.35f * (0.5f + 0.5f * std::sin(phase + 2.7f));
    tuning.particleMass     = 0.014f + 0.018f * (0.5f + 0.5f * std::cos(phase + 1.8f));
    tuning.particleRadius   = 0.28f + 0.12f * (0.5f + 0.5f * std::sin(phase + 3.4f));
    tuning.edgeCompliance   = 0.00024f + 0.00012f * static_cast<float>(envIndex % 4u);
    tuning.volumeCompliance = 0.0011f + 0.00028f * static_cast<float>((envIndex + 1u) % 5u);
    return tuning;
}

ToroidVariantTuning tuningForToroidVariant(const EnvironmentTuning &base,
                                           std::uint32_t envIndex,
                                           std::uint32_t toroidIndex,
                                           std::uint32_t toroidCount)
{
    const float envPhase = static_cast<float>(envIndex) * 0.37f;
    const float normalizedIndex =
        toroidCount > 1u ? static_cast<float>(toroidIndex) / static_cast<float>(toroidCount - 1u)
                         : 0.0f;
    const float variantPhase = envPhase + static_cast<float>(toroidIndex) * 1.41f;

    ToroidVariantTuning tuning{};
    tuning.friction =
        std::clamp(base.softFriction + (normalizedIndex - 0.5f) * 0.30f, 0.02f, 1.2f);
    tuning.restitution =
        std::clamp(base.softRestitution + 0.10f * std::sin(variantPhase), 0.0f, 0.95f);
    tuning.damping =
        std::max(0.0f, base.softDamping + (normalizedIndex - 0.5f) * 0.90f +
                           0.18f * std::cos(variantPhase * 0.7f));
    tuning.particleMass =
        std::max(0.002f, base.particleMass * (0.82f + 0.36f * normalizedIndex));
    tuning.particleRadius =
        std::max(0.12f, base.particleRadius * (0.92f + 0.18f * std::sin(variantPhase + 0.5f)));
    tuning.edgeCompliance =
        std::max(0.0f, base.edgeCompliance * (0.6f + 1.6f * normalizedIndex));
    tuning.volumeCompliance =
        std::max(0.0f, base.volumeCompliance * (0.7f + 1.4f * (1.0f - normalizedIndex)));
    return tuning;
}

void authorEnvironment(Runtime &runtime, std::uint32_t envIndex, std::uint32_t envCount,
                       MeshHandle boxMesh, MeshHandle planeMesh, MeshHandle toroidMesh,
                       const SceneMaterials &materials,
                       cressim::neo::graphics::RenderResourceManager &resources,
                       const std::filesystem::path &nodeFile,
                       const std::filesystem::path &eleFile, std::uint32_t toroidCount,
                       cressim::neo::common::EntityId &outCameraEntity)
{
    auto &world                   = runtime.getWorld();
    const Diligent::float3 origin = envOrigin(envIndex, envCount);
    const float phase             = static_cast<float>(envIndex) * 0.61f;
    const EnvironmentTuning tuning = tuningForEnvironment(envIndex);
    const MaterialHandle softBodyMaterial = registerMaterial(
        resources, ("SoftParticleToroidMultiEnv.SoftBody.Env" + std::to_string(envIndex)).c_str(),
        tuning.softBodyColor, 0.42f);

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = origin + Diligent::float3{0.0f, 18.0f, -42.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.34f);
    world.setTransform(outCameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.renderOrder = envIndex;
    camera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(
        Diligent::float3{-0.35f + 0.07f * std::sin(phase), -1.0f, 0.25f + 0.06f * std::cos(phase)});
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = origin + Diligent::float3{0.0f, -13.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, materials.ground, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {40.0f, 0.05f, 40.0f, 0.0f};
    groundCollider.friction    = tuning.groundFriction;
    world.addCollider(groundEntity, groundCollider);

    const auto obstacleEntity = world.createEntity(envIndex);
    TransformComponent obstacleTransform{};
    obstacleTransform.worldTransform.position =
        origin + Diligent::float3{12.0f + 2.0f * std::sin(phase), 2.0f, 0.5f * std::cos(phase)};
    obstacleTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.2f * phase);
    world.setTransform(obstacleEntity, obstacleTransform);
    world.setMeshRenderer(obstacleEntity, MeshRendererComponent{boxMesh, materials.obstacle, true});
    RigidBodyComponent obstacleBody{};
    obstacleBody.bodyType            = RigidBodyType::Dynamic;
    obstacleBody.inverseMass         = 0.05f;
    obstacleBody.inverseInertiaLocal = cressim::neo::examples::helpers::computeBoxInverseInertia(
        {2.5f, 2.5f, 2.5f}, obstacleBody.inverseMass);
    world.setRigidBody(obstacleEntity, obstacleBody);
    ColliderComponent obstacleCollider{};
    obstacleCollider.shapeType   = ColliderShapeType::Box;
    obstacleCollider.shapeParams = {2.5f, 2.5f, 2.5f, 0.0f};
    obstacleCollider.friction    = tuning.obstacleFriction;
    world.addCollider(obstacleEntity, obstacleCollider);

    const auto staticObstacleEntity = world.createEntity(envIndex);
    TransformComponent staticObstacleTransform{};
    staticObstacleTransform.worldTransform.position =
        origin + Diligent::float3{-10.0f, 1.5f, 1.2f * std::sin(phase * 0.5f)};
    world.setTransform(staticObstacleEntity, staticObstacleTransform);
    world.setMeshRenderer(staticObstacleEntity,
                          MeshRendererComponent{boxMesh, materials.obstacle, true});
    RigidBodyComponent staticObstacleBody{};
    staticObstacleBody.bodyType    = RigidBodyType::Static;
    staticObstacleBody.inverseMass = 0.0f;
    world.setRigidBody(staticObstacleEntity, staticObstacleBody);
    ColliderComponent staticObstacleCollider{};
    staticObstacleCollider.shapeType   = ColliderShapeType::Box;
    staticObstacleCollider.shapeParams = {2.5f, 2.5f, 2.5f, 0.0f};
    staticObstacleCollider.friction    = tuning.obstacleFriction;
    world.addCollider(staticObstacleEntity, staticObstacleCollider);

    TransformComponent softTransform{};
    softTransform.worldTransform.position =
        origin + Diligent::float3{-2.0f + 1.5f * std::cos(phase), 10.0f + 0.6f * std::sin(phase),
                                  1.0f * std::sin(phase * 0.7f)};
    softTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.18f * std::sin(phase));
    softTransform.worldTransform.scale = {5.0f, 5.0f, 5.0f};

    SoftBodyComponent softBody{};
    softBody.source.kind            = SoftBodySourceKind::TetGenFiles;
    softBody.source.tetGen.nodeFile = nodeFile.string();
    softBody.source.tetGen.eleFile  = eleFile.string();
    softBody.selfCollisionEnabled   = true;
    softBody.collisionLayer         = 0x1u;
    softBody.collisionMask          = 0xffffffffu;

    for (std::uint32_t toroidIndex = 0u; toroidIndex < toroidCount; ++toroidIndex)
    {
        const ToroidVariantTuning toroidTuning =
            tuningForToroidVariant(tuning, envIndex, toroidIndex, toroidCount);
        const auto softEntity = world.createEntity(envIndex);
        TransformComponent toroidTransform = softTransform;
        toroidTransform.worldTransform.position.y += 16.0f * static_cast<float>(toroidIndex);
        toroidTransform.worldTransform.position.x +=
            (static_cast<float>(toroidIndex % 2u) - 0.5f) * 2.5f;
        toroidTransform.worldTransform.position.z +=
            (static_cast<float>(toroidIndex) - static_cast<float>(toroidCount - 1u) * 0.5f) * 1.2f;
        toroidTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle(
                {0.0f, 1.0f, 0.0f}, 0.18f * std::sin(phase) + 0.21f * static_cast<float>(toroidIndex));
        world.setTransform(softEntity, toroidTransform);
        world.setMeshRenderer(softEntity,
                              MeshRendererComponent{toroidMesh, softBodyMaterial, true});

        SoftBodyComponent softBodyVariant = softBody;
        softBodyVariant.material.contact.friction    = toroidTuning.friction;
        softBodyVariant.material.contact.restitution = toroidTuning.restitution;
        softBodyVariant.material.contact.damping     = toroidTuning.damping;
        softBodyVariant.particleMass         = toroidTuning.particleMass;
        softBodyVariant.particleRadius       = toroidTuning.particleRadius;
        softBodyVariant.edgeCompliance       = toroidTuning.edgeCompliance;
        softBodyVariant.volumeCompliance     = toroidTuning.volumeCompliance;

        if (!world.setSoftBody(softEntity, softBodyVariant))
        {
            throw std::runtime_error("Failed to author toroid soft body for multi-env viewer.");
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;
    std::uint32_t toroidCount = 1u;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, true))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--toroids")
            {
                const char* value = cressim::neo::examples::helpers::requireOptionValue(
                    argc, argv, i, "--toroids");
                toroidCount = cressim::neo::examples::helpers::parseEnvCount(value);
                continue;
            }

            printUsage(argv[0]);
            return 2;
        }
    }
    catch (const std::invalid_argument& error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.physicsDesc.softContactIterations  = 20;
    config.physicsDesc.softInternalIterations = 20;
    config.sceneLayout.envCount = options.envCount;

    const std::filesystem::path nodeFile = fixturePath("toroid.node");
    const std::filesystem::path eleFile  = fixturePath("toroid.ele");
    const std::filesystem::path surfaceObjFile = fixturePath("toroid_surface.obj");
    if (!std::filesystem::exists(nodeFile) || !std::filesystem::exists(eleFile) ||
        !std::filesystem::exists(surfaceObjFile))
    {
        CRESSIM_LOG_ERROR("Toroid soft-body fixtures are missing.");
        return 1;
    }

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Debug Viewer";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = false;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.statsIntervalFrames     = 60u;
    viewerDesc.enableDebugParticles    = false;

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

    auto &resources            = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(cressim::neo::examples::helpers::makeCubeMesh(
        2.5f, "SoftParticleToroidMultiEnv.CubeMesh"));
    const MeshHandle planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(
            40.0f, "SoftParticleToroidMultiEnv.PlaneMesh"));
    const MeshHandle toroidMesh = resources.registerMesh(loadObjMesh(surfaceObjFile));

    SceneMaterials materials{};
    materials.ground =
        registerMaterial(resources, "SoftParticleToroidMultiEnv.Ground",
                         {0.72f, 0.75f, 0.79f}, 0.90f);
    materials.obstacle =
        registerMaterial(resources, "SoftParticleToroidMultiEnv.Obstacle",
                         {0.16f, 0.43f, 0.86f}, 0.55f);
    auto &world = runtime.getWorld();
    const auto sharedIbl = loadToroidSkyboxIbl(resources);
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        if (!world.setEnvironmentIbl(envIndex, sharedIbl))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Failed to assign toroid skybox IBL.\n");
            return 1;
        }
    }

    cressim::neo::common::EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(runtime, envIndex, options.envCount, boxMesh, planeMesh, toroidMesh,
                          materials, resources, nodeFile, eleFile, toroidCount, cameraEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
    }

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{primaryCamera});

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
