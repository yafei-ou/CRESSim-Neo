#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "viewer/debug_viewer_app.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kCompositeChildHalfExtent = 0.60f;
constexpr float kCompositeHalfExtent = 1.0f;
constexpr float kCompositeChildOffset = kCompositeHalfExtent - kCompositeChildHalfExtent;

GpuBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return GpuBackend::Null;
    }
    if (value == "vulkan")
    {
        return GpuBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--backend vulkan|null] [--frames N]\n");
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "MixedShapeViewer.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3& normal, const Diligent::float3& v0,
                             const Diligent::float3& v1, const Diligent::float3& v2,
                             const Diligent::float3& v3) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 3u);
        mesh.indices.push_back(base + 2u);
    };

    const float h = halfExtent;
    addFace({0.0f, 0.0f, 1.0f}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h});
    addFace({0.0f, 0.0f, -1.0f}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h});
    addFace({-1.0f, 0.0f, 0.0f}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h});
    addFace({1.0f, 0.0f, 0.0f}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h});
    addFace({0.0f, 1.0f, 0.0f}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h});
    addFace({0.0f, -1.0f, 0.0f}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h});
    return mesh;
}

MeshResourceDesc makePlaneMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "MixedShapeViewer.PlaneMesh";
    const float h = halfExtent;
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f}};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

MeshResourceDesc makeSphereMesh(float radius, std::uint32_t slices, std::uint32_t stacks)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "MixedShapeViewer.SphereMesh";
    mesh.vertices.reserve((stacks + 1u) * (slices + 1u));
    mesh.indices.reserve(stacks * slices * 6u);

    for (std::uint32_t stack = 0u; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * kPi;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float x = ringRadius * std::cos(theta);
            const float z = ringRadius * std::sin(theta);
            const Diligent::float3 normal{x, y, z};
            mesh.vertices.push_back({normal * radius, normal, u, v});
        }
    }

    const std::uint32_t ring = slices + 1u;
    for (std::uint32_t stack = 0u; stack < stacks; ++stack)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = stack * ring + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ring;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

MeshResourceDesc makeCapsuleMesh(float radius, float halfHeight, std::uint32_t slices,
                                 std::uint32_t hemisphereRings,
                                 std::uint32_t bodyRings)
{
    struct Ring
    {
        float y = 0.0f;
        float r = 0.0f;
    };

    MeshResourceDesc mesh{};
    mesh.debugName = "MixedShapeViewer.CapsuleMesh";
    std::vector<Ring> rings;
    rings.reserve(2u * hemisphereRings + bodyRings + 2u);

    rings.push_back({halfHeight + radius, 0.0f});
    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (kPi * 0.5f);
        rings.push_back({halfHeight + radius * std::cos(angle), radius * std::sin(angle)});
    }

    for (std::uint32_t i = 1u; i <= bodyRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(bodyRings + 1u);
        rings.push_back({halfHeight * (1.0f - 2.0f * t), radius});
    }

    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (kPi * 0.5f);
        rings.push_back({-halfHeight - radius * std::sin(angle), radius * std::cos(angle)});
    }

    mesh.vertices.reserve(static_cast<std::size_t>(rings.size()) * (slices + 1u));
    mesh.indices.reserve((static_cast<std::uint32_t>(rings.size()) - 1u) * slices * 6u);

    for (std::size_t ringIndex = 0u; ringIndex < rings.size(); ++ringIndex)
    {
        const float y = rings[ringIndex].y;
        const float rr = rings[ringIndex].r;

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float x = rr * std::cos(theta);
            const float z = rr * std::sin(theta);

            Diligent::float3 normal{};
            if (y > halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y - halfHeight, z});
            }
            else if (y < -halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y + halfHeight, z});
            }
            else
            {
                normal = rr > 0.0f ? Diligent::normalize(Diligent::float3{x, 0.0f, z})
                                   : Diligent::float3{0.0f, 1.0f, 0.0f};
            }

            const float v = static_cast<float>(ringIndex) /
                            static_cast<float>(std::max<std::size_t>(1u, rings.size() - 1u));
            mesh.vertices.push_back({{x, y, z}, normal, u, v});
        }
    }

    const std::uint32_t ringStride = slices + 1u;
    for (std::uint32_t ringIndex = 0u;
         ringIndex + 1u < static_cast<std::uint32_t>(rings.size()); ++ringIndex)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = ringIndex * ringStride + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ringStride;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

Diligent::float3 computeBoxInverseInertia(const Diligent::float3& halfExtents, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix = mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy = mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz = mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;

    return {ix > 0.0f ? 1.0f / ix : 0.0f,
            iy > 0.0f ? 1.0f / iy : 0.0f,
            iz > 0.0f ? 1.0f / iz : 0.0f};
}

Diligent::float3 computeSphereInverseInertia(float radius, float inverseMass)
{
    if (inverseMass <= 0.0f || radius <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float inertia = 0.4f * mass * radius * radius;
    const float inverseInertia = inertia > 0.0f ? 1.0f / inertia : 0.0f;
    return {inverseInertia, inverseInertia, inverseInertia};
}

Diligent::float4 colliderParamsForShape(ColliderShapeType shape)
{
    switch (shape)
    {
        case ColliderShapeType::Sphere:
            return {0.6f, 0.0f, 0.0f, 0.0f};
        case ColliderShapeType::Box:
            return {0.55f, 0.45f, 0.5f, 0.0f};
        case ColliderShapeType::Capsule:
            return {0.35f, 0.55f, 0.0f, 0.0f};
    }

    return {0.55f, 0.45f, 0.5f, 0.0f};
}

Diligent::float3 inverseInertiaForShape(ColliderShapeType shape,
                                        const Diligent::float4& colliderParams,
                                        float inverseMass)
{
    switch (shape)
    {
        case ColliderShapeType::Sphere:
            return computeSphereInverseInertia(colliderParams.x, inverseMass);
        case ColliderShapeType::Box:
            return computeBoxInverseInertia(
                {colliderParams.x, colliderParams.y, colliderParams.z}, inverseMass);
        case ColliderShapeType::Capsule:
            return computeBoxInverseInertia(
                {colliderParams.x, colliderParams.y + colliderParams.x, colliderParams.x},
                inverseMass);
    }

    return {0.0f, 0.0f, 0.0f};
}

struct BodySpawn
{
    ColliderShapeType shape = ColliderShapeType::Box;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    cressim::neo::graphics::MaterialHandle material{};
};

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    std::uint64_t numFrames = 0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            config.gpuDeviceDesc.preferredBackend = parseBackend(argv[++i]);
            continue;
        }
        if (arg == "--frames")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            numFrames = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = true;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = false;
    viewerDesc.width = 960;
    viewerDesc.height = 540;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR( "Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.6f, -6.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.5f));
    const auto largeCubeMesh = resources.registerMesh(makeCubeMesh(kCompositeHalfExtent));
    const auto planeMesh = resources.registerMesh(makePlaneMesh(8.0f));

    MaterialResourceDesc compositeMaterialDesc{};
    compositeMaterialDesc.debugName = "MixedShapeViewer.CompositeBodyMaterial";
    compositeMaterialDesc.baseColor = {0.16f, 0.62f, 0.96f};
    compositeMaterialDesc.metallic = 0.0f;
    compositeMaterialDesc.roughness = 0.40f;
    const auto compositeMaterial = resources.registerMaterial(compositeMaterialDesc);

    MaterialResourceDesc probeMaterialDesc{};
    probeMaterialDesc.debugName = "MixedShapeViewer.ProbeBodyMaterial";
    probeMaterialDesc.baseColor = {0.92f, 0.26f, 0.18f};
    probeMaterialDesc.metallic = 0.0f;
    probeMaterialDesc.roughness = 0.42f;
    const auto probeMaterial = resources.registerMaterial(probeMaterialDesc);

    MaterialResourceDesc planeMaterialDesc{};
    planeMaterialDesc.debugName = "MixedShapeViewer.GroundMaterial";
    planeMaterialDesc.baseColor = {0.72f, 0.74f, 0.77f};
    planeMaterialDesc.metallic = 0.0f;
    planeMaterialDesc.roughness = 0.85f;
    const auto planeMaterial = resources.registerMaterial(planeMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent groundMesh{};
    groundMesh.mesh = planeMesh;
    groundMesh.material = planeMaterial;
    groundMesh.visible = true;
    world.setMeshRenderer(groundEntity, groundMesh);
    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto compositeEntity = world.createEntity();
    TransformComponent compositeTransform{};
    compositeTransform.worldTransform.position = {0.0f, 1.85f, 2.0f};
    world.setTransform(compositeEntity, compositeTransform);

    MeshRendererComponent compositeMesh{};
    compositeMesh.mesh = largeCubeMesh;
    compositeMesh.material = compositeMaterial;
    compositeMesh.visible = true;
    world.setMeshRenderer(compositeEntity, compositeMesh);

    RigidBodyComponent compositeBody{};
    compositeBody.simulated = true;
    compositeBody.inverseMass = 1.0f;
    compositeBody.inverseInertiaLocal =
        computeBoxInverseInertia({kCompositeHalfExtent, kCompositeHalfExtent, kCompositeHalfExtent},
                                 compositeBody.inverseMass);
    world.setRigidBody(compositeEntity, compositeBody);

    const std::vector<Diligent::float3> colliderOffsets = {
        {-kCompositeChildOffset, -kCompositeChildOffset, -kCompositeChildOffset},
        {kCompositeChildOffset, -kCompositeChildOffset, -kCompositeChildOffset},
        {-kCompositeChildOffset, kCompositeChildOffset, -kCompositeChildOffset},
        {kCompositeChildOffset, kCompositeChildOffset, -kCompositeChildOffset},
        {-kCompositeChildOffset, -kCompositeChildOffset, kCompositeChildOffset},
        {kCompositeChildOffset, -kCompositeChildOffset, kCompositeChildOffset},
        {-kCompositeChildOffset, kCompositeChildOffset, kCompositeChildOffset},
        {kCompositeChildOffset, kCompositeChildOffset, kCompositeChildOffset},
    };

    for (const Diligent::float3& offset : colliderOffsets)
    {
        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Box;
        collider.shapeParams = {kCompositeChildHalfExtent, kCompositeChildHalfExtent,
                                kCompositeChildHalfExtent, 0.0f};
        collider.localPosition = offset;
        world.addCollider(compositeEntity, collider);
    }

    const auto singleColliderEntity = world.createEntity();
    TransformComponent singleColliderTransform{};
    singleColliderTransform.worldTransform.position = {3.6f, 1.85f, 2.0f};
    world.setTransform(singleColliderEntity, singleColliderTransform);

    MeshRendererComponent singleColliderMesh{};
    singleColliderMesh.mesh = largeCubeMesh;
    singleColliderMesh.material = probeMaterial;
    singleColliderMesh.visible = true;
    world.setMeshRenderer(singleColliderEntity, singleColliderMesh);

    RigidBodyComponent singleColliderBody{};
    singleColliderBody.simulated = true;
    singleColliderBody.inverseMass = 1.0f;
    singleColliderBody.inverseInertiaLocal =
        computeBoxInverseInertia({kCompositeHalfExtent, kCompositeHalfExtent, kCompositeHalfExtent},
                                 singleColliderBody.inverseMass);
    world.setRigidBody(singleColliderEntity, singleColliderBody);

    ColliderComponent singleCollider{};
    singleCollider.shapeType = ColliderShapeType::Box;
    singleCollider.shapeParams = {kCompositeHalfExtent, kCompositeHalfExtent,
                                  kCompositeHalfExtent, 0.0f};
    world.addCollider(singleColliderEntity, singleCollider);

    // const auto probeEntity = world.createEntity();
    // TransformComponent probeTransform{};
    // probeTransform.worldTransform.position = {1.55f, 1.15f, 2.0f};
    // probeTransform.worldTransform.scale = {0.9f, 0.9f, 0.9f};
    // world.setTransform(probeEntity, probeTransform);

    // MeshRendererComponent probeMesh{};
    // probeMesh.mesh = cubeMesh;
    // probeMesh.material = probeMaterial;
    // probeMesh.visible = true;
    // world.setMeshRenderer(probeEntity, probeMesh);

    // RigidBodyComponent probeBody{};
    // probeBody.simulated = true;
    // probeBody.inverseMass = 1.0f;
    // probeBody.inverseInertiaLocal = computeBoxInverseInertia({0.45f, 0.45f, 0.45f},
    //                                                          probeBody.inverseMass);
    // world.setRigidBody(probeEntity, probeBody);

    // ColliderComponent probeCollider{};
    // probeCollider.shapeType = ColliderShapeType::Box;
    // probeCollider.shapeParams = {0.45f, 0.45f, 0.45f, 0.0f};
    // world.addCollider(probeEntity, probeCollider);

    std::uint64_t beforeCalls = 0;
    std::uint64_t afterCalls = 0;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext&, Runtime&) { ++beforeCalls; };
    callbacks.afterTick = [&](const FrameContext&, Runtime&) { ++afterCalls; };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR( "Viewer run failed.\n");
        return 1;
    }
    if (viewerDesc.maxFrames > 0 &&
        (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
    {
        CRESSIM_LOG_ERROR( "Unexpected callback counts. before=" , beforeCalls , " after=" , afterCalls
                  , " expected=" , viewerDesc.maxFrames , '\n');
        return 1;
    }

    CRESSIM_LOG_INFO( "Physics mixed-shape viewer passed. Frames=" , viewerDesc.maxFrames , '\n');
    return 0;
}
