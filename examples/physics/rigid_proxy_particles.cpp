#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/inertia.h"
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
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ParticleContactMaterialDesc;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName, "  Proxy particle debug visualization is enabled by default.\n", false);
}

std::vector<Diligent::float3> makeBoxSurfaceProxy(const Diligent::float3 &halfExtents,
                                                  std::uint32_t samplesPerAxis)
{
    const std::uint32_t count = std::max(samplesPerAxis, 2u);
    const auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    std::vector<Diligent::float3> points;
    points.reserve(static_cast<std::size_t>(count) * static_cast<std::size_t>(count) * 6u);

    for (std::uint32_t iy = 0u; iy < count; ++iy)
    {
        const float ty = static_cast<float>(iy) / static_cast<float>(count - 1u);
        const float y = lerp(-halfExtents.y, halfExtents.y, ty);
        for (std::uint32_t iz = 0u; iz < count; ++iz)
        {
            const float tz = static_cast<float>(iz) / static_cast<float>(count - 1u);
            const float z = lerp(-halfExtents.z, halfExtents.z, tz);
            points.push_back({-halfExtents.x, y, z});
            points.push_back({halfExtents.x, y, z});
        }
    }

    for (std::uint32_t ix = 0u; ix < count; ++ix)
    {
        const float tx = static_cast<float>(ix) / static_cast<float>(count - 1u);
        const float x = lerp(-halfExtents.x, halfExtents.x, tx);
        for (std::uint32_t iz = 0u; iz < count; ++iz)
        {
            const float tz = static_cast<float>(iz) / static_cast<float>(count - 1u);
            const float z = lerp(-halfExtents.z, halfExtents.z, tz);
            points.push_back({x, -halfExtents.y, z});
            points.push_back({x, halfExtents.y, z});
        }
    }

    for (std::uint32_t ix = 0u; ix < count; ++ix)
    {
        const float tx = static_cast<float>(ix) / static_cast<float>(count - 1u);
        const float x = lerp(-halfExtents.x, halfExtents.x, tx);
        for (std::uint32_t iy = 0u; iy < count; ++iy)
        {
            const float ty = static_cast<float>(iy) / static_cast<float>(count - 1u);
            const float y = lerp(-halfExtents.y, halfExtents.y, ty);
            points.push_back({x, y, -halfExtents.z});
            points.push_back({x, y, halfExtents.z});
        }
    }

    return points;
}

std::vector<Diligent::float3> makeSphereSurfaceProxy(float radius, std::uint32_t slices,
                                                     std::uint32_t stacks)
{
    constexpr float kPi = 3.14159265358979323846f;
    const std::uint32_t safeSlices = std::max(slices, 3u);
    const std::uint32_t safeStacks = std::max(stacks, 2u);
    std::vector<Diligent::float3> points;
    points.reserve(static_cast<std::size_t>(safeSlices + 1u) *
                   static_cast<std::size_t>(safeStacks + 1u));

    for (std::uint32_t stack = 0u; stack <= safeStacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(safeStacks);
        const float phi = v * kPi;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);
        for (std::uint32_t slice = 0u; slice < safeSlices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(safeSlices);
            const float theta = u * (2.0f * kPi);
            points.push_back(
                {ringRadius * std::cos(theta) * radius, y * radius,
                 ringRadius * std::sin(theta) * radius});
        }
    }

    return points;
}

Diligent::float3 computeProxyHalfExtents(const std::vector<Diligent::float3> &points,
                                         float radius)
{
    if (points.empty())
    {
        return {radius, radius, radius};
    }

    Diligent::float3 minPoint = points.front();
    Diligent::float3 maxPoint = points.front();
    for (const Diligent::float3 &point : points)
    {
        minPoint.x = std::min(minPoint.x, point.x);
        minPoint.y = std::min(minPoint.y, point.y);
        minPoint.z = std::min(minPoint.z, point.z);
        maxPoint.x = std::max(maxPoint.x, point.x);
        maxPoint.y = std::max(maxPoint.y, point.y);
        maxPoint.z = std::max(maxPoint.z, point.z);
    }

    return {(maxPoint.x - minPoint.x) * 0.5f + radius, (maxPoint.y - minPoint.y) * 0.5f + radius,
            (maxPoint.z - minPoint.z) * 0.5f + radius};
}

void addBoxRigidBody(cressim::neo::engine::World &world, Runtime &runtime,
                     const Diligent::float3 &position, const Diligent::float3 &halfExtents,
                     cressim::neo::graphics::MaterialHandle material,
                     RigidBodyType bodyType = RigidBodyType::Dynamic)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    auto &resources = runtime.getResources();
    const auto mesh = resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
        halfExtents, "RigidProxyParticles.BoxMesh"));
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);

    RigidBodyComponent body{};
    body.bodyType = bodyType;
    body.inverseMass = bodyType == RigidBodyType::Static ? 0.0f : 1.0f;
    body.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(halfExtents, body.inverseMass);
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {halfExtents.x, halfExtents.y, halfExtents.z, 0.0f};
    world.addCollider(entity, collider);
}

void addSphereRigidBody(cressim::neo::engine::World &world, Runtime &runtime,
                        const Diligent::float3 &position, float radius,
                        cressim::neo::graphics::MaterialHandle material,
                        RigidBodyType bodyType = RigidBodyType::Dynamic)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    auto &resources = runtime.getResources();
    const auto mesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        radius, 20u, 12u, "RigidProxyParticles.SphereMesh"));
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);

    RigidBodyComponent body{};
    body.bodyType = bodyType;
    body.inverseMass = bodyType == RigidBodyType::Static ? 0.0f : 1.0f;
    body.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeSphereInverseInertia(radius, body.inverseMass);
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Sphere;
    collider.shapeParams = {radius, 0.0f, 0.0f, 0.0f};
    world.addCollider(entity, collider);
}

void addProxyBoxRigidBody(cressim::neo::engine::World &world, Runtime &runtime,
                          const Diligent::float3 &position, const Diligent::float3 &halfExtents,
                          const std::vector<Diligent::float3> &proxyParticles,
                          float proxyParticleRadius,
                          const cressim::neo::physics::ParticleContactMaterialDesc &proxyMaterial,
                          cressim::neo::graphics::MaterialHandle material,
                          RigidBodyType bodyType = RigidBodyType::Dynamic)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    auto &resources = runtime.getResources();
    const auto mesh = resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
        halfExtents, "RigidProxyParticles.ProxyBodyMesh"));
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);

    RigidBodyComponent body{};
    body.bodyType = bodyType;
    body.inverseMass = bodyType == RigidBodyType::Static ? 0.0f : 1.0f;
    body.inverseInertiaLocal = bodyType == RigidBodyType::Static
                                   ? Diligent::float3{0.0f, 0.0f, 0.0f}
                                   : cressim::neo::examples::helpers::computeBoxInverseInertia(
                                         halfExtents, body.inverseMass);
    body.proxyParticleMaterial = proxyMaterial;
    body.proxyParticleRadius = proxyParticleRadius;
    body.proxyParticleLocalPositions = proxyParticles;
    world.setRigidBody(entity, body);
}

void addProxySphereRigidBody(cressim::neo::engine::World &world, Runtime &runtime,
                             const Diligent::float3 &position, float radius,
                             const std::vector<Diligent::float3> &proxyParticles,
                             float proxyParticleRadius,
                             const cressim::neo::physics::ParticleContactMaterialDesc &proxyMaterial,
                             cressim::neo::graphics::MaterialHandle material,
                             RigidBodyType bodyType = RigidBodyType::Dynamic)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    auto &resources = runtime.getResources();
    const auto mesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        radius, 20u, 12u, "RigidProxyParticles.ProxySphereMesh"));
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);

    RigidBodyComponent body{};
    body.bodyType = bodyType;
    body.inverseMass = bodyType == RigidBodyType::Static ? 0.0f : 1.0f;
    body.inverseInertiaLocal =
        bodyType == RigidBodyType::Static
            ? Diligent::float3{0.0f, 0.0f, 0.0f}
            : cressim::neo::examples::helpers::computeSphereInverseInertia(radius, body.inverseMass);
    body.proxyParticleMaterial = proxyMaterial;
    body.proxyParticleRadius = proxyParticleRadius;
    body.proxyParticleLocalPositions = proxyParticles;
    world.setRigidBody(entity, body);
}

void addSoftBody(cressim::neo::engine::World &world, const Diligent::float3 &position,
                 const Diligent::float3 &size, float spacing, float particleRadius)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.35f);
    world.setTransform(entity, transform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = size;
    softBody.source.regularGrid.targetParticleSpacing = spacing;
    softBody.particleMass = 0.12f;
    softBody.particleRadius = particleRadius;
    softBody.edgeCompliance = 0.0f;
    softBody.volumeCompliance = 0.0008f;
    softBody.selfCollisionEnabled = true;
    softBody.material.contact.friction = 0.45f;
    softBody.material.contact.staticFriction = 0.55f;
    world.setSoftBody(entity, softBody);
}

void addFluid(cressim::neo::engine::World &world, const Diligent::float3 &position,
              const Diligent::float3 &size, float spacing, float particleRadius)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    FluidComponent fluid{};
    fluid.source.kind = FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = size;
    fluid.source.regularGrid.targetParticleSpacing = spacing;
    fluid.particleRadius = particleRadius;
    const float diameter = 2.0f * particleRadius;
    fluid.particleMass = diameter * diameter * diameter * 1000.0f;
    fluid.material.viscosity = 0.08f;
    fluid.material.cohesion = 0.01f;
    fluid.material.surfaceTension = 0.003f;
    fluid.visualColor = {0.26f, 0.66f, 0.97f, 0.72f};
    world.setFluid(entity, fluid);
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
    config.physicsDesc.substeps = 4u;
    config.physicsDesc.defaultIterations = 12u;
    config.physicsDesc.rigidRigidContactIterations = 12u;
    config.physicsDesc.softContactIterations = 8u;
    config.physicsDesc.fluidIterations = 4u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Rigid Proxy Particles";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = true;
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
    cameraTransform.worldTransform.position = {0.0f, 3.2f, -10.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.18f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 40.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 6.5f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(14.0f, "RigidProxyParticles.GroundMesh"));

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "RigidProxyParticles.GroundMaterial";
    groundMaterialDesc.baseColor = {0.77f, 0.80f, 0.83f};
    groundMaterialDesc.roughness = 0.92f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    MaterialResourceDesc obstacleMaterialDesc{};
    obstacleMaterialDesc.debugName = "RigidProxyParticles.ObstacleMaterial";
    obstacleMaterialDesc.baseColor = {0.19f, 0.48f, 0.82f};
    obstacleMaterialDesc.roughness = 0.56f;
    const auto obstacleMaterial = resources.registerMaterial(obstacleMaterialDesc);

    MaterialResourceDesc primitiveBoxMaterialDesc{};
    primitiveBoxMaterialDesc.debugName = "RigidProxyParticles.PrimitiveBoxMaterial";
    primitiveBoxMaterialDesc.baseColor = {0.93f, 0.33f, 0.21f};
    primitiveBoxMaterialDesc.roughness = 0.48f;
    const auto primitiveBoxMaterial = resources.registerMaterial(primitiveBoxMaterialDesc);

    MaterialResourceDesc proxyBoxMaterialDesc{};
    proxyBoxMaterialDesc.debugName = "RigidProxyParticles.ProxyBoxMaterial";
    proxyBoxMaterialDesc.baseColor = {0.18f, 0.77f, 0.44f};
    proxyBoxMaterialDesc.roughness = 0.44f;
    const auto proxyBoxMaterial = resources.registerMaterial(proxyBoxMaterialDesc);

    MaterialResourceDesc primitiveSphereMaterialDesc{};
    primitiveSphereMaterialDesc.debugName = "RigidProxyParticles.PrimitiveSphereMaterial";
    primitiveSphereMaterialDesc.baseColor = {0.84f, 0.72f, 0.22f};
    primitiveSphereMaterialDesc.roughness = 0.52f;
    const auto primitiveSphereMaterial = resources.registerMaterial(primitiveSphereMaterialDesc);

    MaterialResourceDesc proxySphereMaterialDesc{};
    proxySphereMaterialDesc.debugName = "RigidProxyParticles.ProxySphereMaterial";
    proxySphereMaterialDesc.baseColor = {0.63f, 0.42f, 0.88f};
    proxySphereMaterialDesc.roughness = 0.42f;
    const auto proxySphereMaterial = resources.registerMaterial(proxySphereMaterialDesc);

    MaterialResourceDesc proxyStaticMaterialDesc{};
    proxyStaticMaterialDesc.debugName = "RigidProxyParticles.ProxyStaticMaterial";
    proxyStaticMaterialDesc.baseColor = {0.13f, 0.64f, 0.69f};
    proxyStaticMaterialDesc.roughness = 0.72f;
    const auto proxyStaticMaterial = resources.registerMaterial(proxyStaticMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.1f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent groundRenderer{};
    groundRenderer.mesh = planeMesh;
    groundRenderer.material = groundMaterial;
    groundRenderer.visible = true;
    world.setMeshRenderer(groundEntity, groundRenderer);

    RigidBodyComponent groundBody{};
    groundBody.bodyType = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);

    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {14.0f, 0.08f, 14.0f, 0.0f};
    groundCollider.friction = 0.45f;
    groundCollider.staticFriction = 0.6f;
    world.addCollider(groundEntity, groundCollider);

    addBoxRigidBody(world, runtime, {-5.7f, -0.25f, 0.6f}, {1.45f, 0.28f, 1.0f}, obstacleMaterial,
                    RigidBodyType::Static);
    addBoxRigidBody(world, runtime, {-0.6f, -0.25f, 0.6f}, {1.45f, 0.28f, 1.0f}, obstacleMaterial,
                    RigidBodyType::Static);
    addBoxRigidBody(world, runtime, {5.4f, -0.25f, 0.6f}, {1.55f, 0.28f, 1.35f}, obstacleMaterial,
                    RigidBodyType::Static);

    const Diligent::float3 boxHalfExtents{0.55f, 0.55f, 0.55f};
    const float sphereRadius = 0.58f;
    const float proxyParticleRadius = 0.14f;
    const std::vector<Diligent::float3> boxProxyPoints =
        makeBoxSurfaceProxy(boxHalfExtents, 4u);
    const std::vector<Diligent::float3> sphereProxyPoints =
        makeSphereSurfaceProxy(sphereRadius, 16u, 8u);
    ParticleContactMaterialDesc proxyBoxContactMaterial{};
    proxyBoxContactMaterial.friction = 0.42f;
    proxyBoxContactMaterial.staticFriction = 0.56f;
    ParticleContactMaterialDesc proxySphereContactMaterial{};
    proxySphereContactMaterial.friction = 0.28f;
    proxySphereContactMaterial.staticFriction = 0.34f;
    proxySphereContactMaterial.restitution = 0.06f;
    ParticleContactMaterialDesc proxyStaticContactMaterial{};
    proxyStaticContactMaterial.friction = 0.48f;
    proxyStaticContactMaterial.staticFriction = 0.62f;

    addBoxRigidBody(world, runtime, {-6.4f, 4.2f, -0.7f}, boxHalfExtents, primitiveBoxMaterial);
    addProxyBoxRigidBody(world, runtime, {-3.8f, 4.2f, -0.7f}, boxHalfExtents, boxProxyPoints,
                         proxyParticleRadius, proxyBoxContactMaterial, proxyBoxMaterial);

    addSphereRigidBody(world, runtime, {-1.2f, 5.7f, -0.3f}, sphereRadius, primitiveSphereMaterial);
    addProxySphereRigidBody(world, runtime, {1.4f, 5.7f, -0.3f}, sphereRadius, sphereProxyPoints,
                            proxyParticleRadius, proxySphereContactMaterial, proxySphereMaterial);

    addProxyBoxRigidBody(world, runtime, {0.0f, 0.15f, -0.45f}, {0.72f, 0.32f, 0.72f},
                         boxProxyPoints, proxyParticleRadius, proxyStaticContactMaterial,
                         proxyStaticMaterial,
                         RigidBodyType::Static);
    addProxySphereRigidBody(world, runtime, {0.0f, 4.9f, -0.45f}, sphereRadius, sphereProxyPoints,
                            proxyParticleRadius, proxySphereContactMaterial, proxySphereMaterial);

    addProxyBoxRigidBody(world, runtime, {5.0f, 0.15f, -0.35f}, {0.82f, 0.32f, 0.82f},
                         boxProxyPoints, proxyParticleRadius, proxyStaticContactMaterial,
                         proxyStaticMaterial,
                         RigidBodyType::Static);
    addSoftBody(world, {4.15f, 4.75f, -0.35f}, {0.95f, 0.95f, 0.95f}, 0.30f, 0.14f);

    addProxySphereRigidBody(world, runtime, {6.75f, 0.9f, -0.1f}, 0.72f,
                            makeSphereSurfaceProxy(0.72f, 16u, 8u), 0.13f,
                            proxyStaticContactMaterial,
                            proxyStaticMaterial, RigidBodyType::Static);
    addFluid(world, {6.75f, 3.35f, -0.1f}, {1.35f, 1.35f, 1.35f}, 0.22f, 0.10f);

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
