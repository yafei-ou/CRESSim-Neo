#include "common/frame_context.h"
#include "common/math_types.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::common::Transform;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::gpu::RenderOutputBinding;
using cressim::neo::gpu::RenderOutputMode;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RenderResourceManager;

MeshResourceDesc makeCubeMesh(const float halfExtent, const std::string &debugName)
{
    MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3)
    {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 2u);
        mesh.indices.push_back(baseIndex + 1u);
        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 3u);
        mesh.indices.push_back(baseIndex + 2u);
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

py::object tryGetRenderTargetReadback(Runtime &runtime, const GpuRenderTargetReadbackRequest request)
{
    auto *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        return py::none();
    }

    GpuRenderTargetReadbackEvent event{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(request, event))
    {
        return py::none();
    }

    return py::cast(event);
}

} // namespace

PYBIND11_MODULE(_cressim_neo, m)
{
    m.doc() = "Minimal Python bindings for CRESSim-Neo runtime authoring and frame readback.";

    py::class_<Diligent::float3>(m, "Float3")
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f)
        .def_readwrite("x", &Diligent::float3::x)
        .def_readwrite("y", &Diligent::float3::y)
        .def_readwrite("z", &Diligent::float3::z);

    py::class_<Diligent::float4>(m, "Float4")
        .def(py::init<float, float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f, py::arg("w") = 0.0f)
        .def_readwrite("x", &Diligent::float4::x)
        .def_readwrite("y", &Diligent::float4::y)
        .def_readwrite("z", &Diligent::float4::z)
        .def_readwrite("w", &Diligent::float4::w);

    py::class_<Diligent::QuaternionF>(m, "Quaternion")
        .def(py::init<float, float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f, py::arg("w") = 1.0f)
        .def_property(
            "x", [](const Diligent::QuaternionF &q) { return q.q.x; },
            [](Diligent::QuaternionF &q, const float value) { q.q.x = value; })
        .def_property(
            "y", [](const Diligent::QuaternionF &q) { return q.q.y; },
            [](Diligent::QuaternionF &q, const float value) { q.q.y = value; })
        .def_property(
            "z", [](const Diligent::QuaternionF &q) { return q.q.z; },
            [](Diligent::QuaternionF &q, const float value) { q.q.z = value; })
        .def_property(
            "w", [](const Diligent::QuaternionF &q) { return q.q.w; },
            [](Diligent::QuaternionF &q, const float value) { q.q.w = value; });

    py::enum_<GpuBackend>(m, "GpuBackend")
        .value("Null", GpuBackend::Null)
        .value("D3D12", GpuBackend::D3D12)
        .value("Vulkan", GpuBackend::Vulkan);

    py::enum_<RenderOutputMode>(m, "RenderOutputMode")
        .value("ManagedPrimary", RenderOutputMode::ManagedPrimary)
        .value("ExplicitSurface", RenderOutputMode::ExplicitSurface);

    py::enum_<Diligent::TEXTURE_FORMAT>(m, "TextureFormat")
        .value("Unknown", Diligent::TEX_FORMAT_UNKNOWN)
        .value("RGBA8Unorm", Diligent::TEX_FORMAT_RGBA8_UNORM)
        .value("RGBA8UnormSrgb", Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB)
        .value("BGRA8Unorm", Diligent::TEX_FORMAT_BGRA8_UNORM)
        .value("BGRA8UnormSrgb", Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB)
        .value("RGBA16Float", Diligent::TEX_FORMAT_RGBA16_FLOAT)
        .value("R32Float", Diligent::TEX_FORMAT_R32_FLOAT)
        .value("R32Uint", Diligent::TEX_FORMAT_R32_UINT)
        .value("D32Float", Diligent::TEX_FORMAT_D32_FLOAT);

    py::enum_<CameraComponent::Product>(m, "CameraProduct")
        .value("ColorDepth", CameraComponent::Product::ColorDepth)
        .value("Depth", CameraComponent::Product::Depth)
        .value("SegmentationDepth", CameraComponent::Product::SegmentationDepth);

    py::enum_<cressim::neo::physics::RigidBodyType>(m, "RigidBodyType")
        .value("Static", cressim::neo::physics::RigidBodyType::Static)
        .value("Kinematic", cressim::neo::physics::RigidBodyType::Kinematic)
        .value("Dynamic", cressim::neo::physics::RigidBodyType::Dynamic);

    py::enum_<cressim::neo::physics::ColliderShapeType>(m, "ColliderShapeType")
        .value("Sphere", cressim::neo::physics::ColliderShapeType::Sphere)
        .value("Box", cressim::neo::physics::ColliderShapeType::Box)
        .value("Capsule", cressim::neo::physics::ColliderShapeType::Capsule);

    py::class_<FrameContext>(m, "FrameContext")
        .def(py::init<>())
        .def_readwrite("frame_index", &FrameContext::frameIndex)
        .def_readwrite("time_seconds", &FrameContext::timeSeconds)
        .def_readwrite("delta_seconds", &FrameContext::deltaSeconds);

    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def_readwrite("position", &Transform::position)
        .def_readwrite("rotation", &Transform::rotation)
        .def_readwrite("scale", &Transform::scale);

    py::class_<TransformComponent>(m, "TransformComponent")
        .def(py::init<>())
        .def_readwrite("world_transform", &TransformComponent::worldTransform);

    py::class_<GpuRenderTargetHandle>(m, "GpuRenderTargetHandle")
        .def(py::init<>())
        .def_readwrite("id", &GpuRenderTargetHandle::id);

    py::class_<GpuRenderTargetBinding>(m, "GpuRenderTargetBinding")
        .def(py::init<>())
        .def_readwrite("target", &GpuRenderTargetBinding::target)
        .def_readwrite("first_layer", &GpuRenderTargetBinding::firstLayer)
        .def_readwrite("layer_count", &GpuRenderTargetBinding::layerCount);

    py::class_<RenderOutputBinding>(m, "RenderOutputBinding")
        .def(py::init<>())
        .def_readwrite("mode", &RenderOutputBinding::mode)
        .def_readwrite("binding", &RenderOutputBinding::binding);

    py::class_<GpuRenderTargetDesc>(m, "GpuRenderTargetDesc")
        .def(py::init<>())
        .def_readwrite("width", &GpuRenderTargetDesc::width)
        .def_readwrite("height", &GpuRenderTargetDesc::height)
        .def_readwrite("array_size", &GpuRenderTargetDesc::arraySize)
        .def_readwrite("color", &GpuRenderTargetDesc::color)
        .def_readwrite("depth", &GpuRenderTargetDesc::depth)
        .def_readwrite("color_format", &GpuRenderTargetDesc::colorFormat)
        .def_readwrite("depth_format", &GpuRenderTargetDesc::depthFormat)
        .def_readwrite("layered_rendering", &GpuRenderTargetDesc::layeredRendering)
        .def_readwrite("shader_readable", &GpuRenderTargetDesc::shaderReadable)
        .def_readwrite("unordered_access", &GpuRenderTargetDesc::unorderedAccess)
        .def_readwrite("debug_name", &GpuRenderTargetDesc::debugName);

    py::class_<GpuRenderTargetReadbackRequest>(m, "GpuRenderTargetReadbackRequest")
        .def(py::init<>())
        .def_readwrite("id", &GpuRenderTargetReadbackRequest::id);

    py::class_<GpuRenderTargetReadbackEvent>(m, "GpuRenderTargetReadbackEvent")
        .def(py::init<>())
        .def_readwrite("frame_index", &GpuRenderTargetReadbackEvent::frameIndex)
        .def_readwrite("color_format", &GpuRenderTargetReadbackEvent::colorFormat)
        .def_readwrite("width", &GpuRenderTargetReadbackEvent::width)
        .def_readwrite("height", &GpuRenderTargetReadbackEvent::height)
        .def_readwrite("row_stride_bytes", &GpuRenderTargetReadbackEvent::rowStrideBytes)
        .def_readwrite("color_width", &GpuRenderTargetReadbackEvent::colorWidth)
        .def_readwrite("color_height", &GpuRenderTargetReadbackEvent::colorHeight)
        .def_readwrite("color_row_stride_bytes", &GpuRenderTargetReadbackEvent::colorRowStrideBytes)
        .def_readwrite("color_bytes", &GpuRenderTargetReadbackEvent::colorBytes)
        .def_readwrite("depth_format", &GpuRenderTargetReadbackEvent::depthFormat)
        .def_readwrite("depth_width", &GpuRenderTargetReadbackEvent::depthWidth)
        .def_readwrite("depth_height", &GpuRenderTargetReadbackEvent::depthHeight)
        .def_readwrite("depth_row_stride_bytes", &GpuRenderTargetReadbackEvent::depthRowStrideBytes)
        .def_readwrite("depth_bytes", &GpuRenderTargetReadbackEvent::depthBytes);

    py::class_<cressim::neo::common::SceneLayoutDesc>(m, "SceneLayoutDesc")
        .def(py::init<>())
        .def_readwrite("env_count", &cressim::neo::common::SceneLayoutDesc::envCount)
        .def_readwrite("max_renderable_objects_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxRenderableObjectsPerEnv)
        .def_readwrite("max_lights_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxLightsPerEnv)
        .def_readwrite("max_cameras_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxCamerasPerEnv);

    py::class_<cressim::neo::gpu::GpuDeviceDesc>(m, "GpuDeviceDesc")
        .def(py::init<>())
        .def_readwrite("preferred_backend", &cressim::neo::gpu::GpuDeviceDesc::preferredBackend)
        .def_readwrite("enable_validation", &cressim::neo::gpu::GpuDeviceDesc::enableValidation)
        .def_readwrite("shader_directory", &cressim::neo::gpu::GpuDeviceDesc::shaderDirectory);

    py::class_<RuntimeConfig>(m, "RuntimeConfig")
        .def(py::init<>())
        .def_readwrite("gpu_device_desc", &RuntimeConfig::gpuDeviceDesc)
        .def_readwrite("scene_layout", &RuntimeConfig::sceneLayout);

    py::class_<MeshHandle>(m, "MeshHandle")
        .def(py::init<>())
        .def_readwrite("id", &MeshHandle::id);

    py::class_<MaterialHandle>(m, "MaterialHandle")
        .def(py::init<>())
        .def_readwrite("id", &MaterialHandle::id);

    py::class_<MeshResourceDesc::Vertex>(m, "MeshVertex")
        .def(py::init<>())
        .def_readwrite("position", &MeshResourceDesc::Vertex::position)
        .def_readwrite("normal", &MeshResourceDesc::Vertex::normal)
        .def_readwrite("tex_coord_u", &MeshResourceDesc::Vertex::texCoordU)
        .def_readwrite("tex_coord_v", &MeshResourceDesc::Vertex::texCoordV)
        .def_readwrite("tangent", &MeshResourceDesc::Vertex::tangent);

    py::class_<MeshResourceDesc>(m, "MeshResourceDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &MeshResourceDesc::debugName)
        .def_readwrite("vertices", &MeshResourceDesc::vertices)
        .def_readwrite("indices", &MeshResourceDesc::indices);

    py::class_<MaterialResourceDesc>(m, "MaterialResourceDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &MaterialResourceDesc::debugName)
        .def_readwrite("base_color", &MaterialResourceDesc::baseColor)
        .def_readwrite("metallic", &MaterialResourceDesc::metallic)
        .def_readwrite("roughness", &MaterialResourceDesc::roughness)
        .def_readwrite("opacity", &MaterialResourceDesc::opacity);

    py::class_<MeshRendererComponent>(m, "MeshRendererComponent")
        .def(py::init<>())
        .def_readwrite("mesh", &MeshRendererComponent::mesh)
        .def_readwrite("material", &MeshRendererComponent::material)
        .def_readwrite("segmentation_id", &MeshRendererComponent::segmentationId)
        .def_readwrite("visible", &MeshRendererComponent::visible);

    py::class_<CameraComponent>(m, "CameraComponent")
        .def(py::init<>())
        .def_readwrite("vertical_fov_degrees", &CameraComponent::verticalFovDegrees)
        .def_readwrite("near_clip", &CameraComponent::nearClip)
        .def_readwrite("far_clip", &CameraComponent::farClip)
        .def_readwrite("product", &CameraComponent::product)
        .def_readwrite("output", &CameraComponent::output)
        .def_readwrite("output_width", &CameraComponent::outputWidth)
        .def_readwrite("output_height", &CameraComponent::outputHeight)
        .def_readwrite("clear_color", &CameraComponent::clearColor)
        .def_readwrite("clear_depth", &CameraComponent::clearDepth)
        .def_readwrite("clear_color_value", &CameraComponent::clearColorValue)
        .def_readwrite("clear_depth_value", &CameraComponent::clearDepthValue)
        .def_readwrite("render_order", &CameraComponent::renderOrder);

    py::class_<DirectionalLightComponent>(m, "DirectionalLightComponent")
        .def(py::init<>())
        .def_readwrite("direction", &DirectionalLightComponent::direction)
        .def_readwrite("color", &DirectionalLightComponent::color)
        .def_readwrite("intensity", &DirectionalLightComponent::intensity)
        .def_readwrite("range", &DirectionalLightComponent::range)
        .def_readwrite("shadow_distance", &DirectionalLightComponent::shadowDistance)
        .def_readwrite("shadow_fade_distance", &DirectionalLightComponent::shadowFadeDistance)
        .def_readwrite("shadow_bias", &DirectionalLightComponent::shadowBias)
        .def_readwrite("casts_shadows", &DirectionalLightComponent::castsShadows);

    py::class_<RigidBodyComponent>(m, "RigidBodyComponent")
        .def(py::init<>())
        .def_readwrite("linear_velocity", &RigidBodyComponent::linearVelocity)
        .def_readwrite("angular_velocity", &RigidBodyComponent::angularVelocity)
        .def_readwrite("body_type", &RigidBodyComponent::bodyType)
        .def_readwrite("inverse_mass", &RigidBodyComponent::inverseMass)
        .def_readwrite("kinematic_target_position", &RigidBodyComponent::kinematicTargetPosition)
        .def_readwrite("kinematic_target_rotation", &RigidBodyComponent::kinematicTargetRotation)
        .def_readwrite("kinematic_target_enabled", &RigidBodyComponent::kinematicTargetEnabled)
        .def_readwrite("simulated", &RigidBodyComponent::simulated);

    py::class_<ColliderComponent>(m, "ColliderComponent")
        .def(py::init<>())
        .def_readwrite("shape_type", &ColliderComponent::shapeType)
        .def_readwrite("shape_params", &ColliderComponent::shapeParams)
        .def_readwrite("local_position", &ColliderComponent::localPosition)
        .def_readwrite("local_rotation", &ColliderComponent::localRotation)
        .def_readwrite("enabled", &ColliderComponent::enabled)
        .def_readwrite("friction", &ColliderComponent::friction)
        .def_readwrite("static_friction", &ColliderComponent::staticFriction)
        .def_readwrite("restitution", &ColliderComponent::restitution)
        .def_readwrite("collision_layer", &ColliderComponent::collisionLayer)
        .def_readwrite("collision_mask", &ColliderComponent::collisionMask);

    py::class_<cressim::neo::engine::ColliderHandle>(m, "ColliderHandle")
        .def(py::init<>())
        .def_readwrite("id", &cressim::neo::engine::ColliderHandle::id);

    py::class_<RenderResourceManager>(m, "RenderResourceManager")
        .def("register_mesh", &RenderResourceManager::registerMesh)
        .def("register_material", &RenderResourceManager::registerMaterial);

    py::class_<World>(m, "World")
        .def("create_entity", &World::createEntity, py::arg("env_index") = 0u)
        .def("set_transform", &World::setTransform)
        .def("set_mesh_renderer", &World::setMeshRenderer)
        .def("set_camera", &World::setCamera)
        .def("set_directional_light", &World::setDirectionalLight)
        .def("set_rigid_body", &World::setRigidBody)
        .def("add_collider", &World::addCollider);

    py::class_<Runtime>(m, "Runtime")
        .def(py::init<>())
        .def(
            "initialize",
            [](Runtime &runtime, py::object configObject)
            {
                RuntimeConfig config{};
                if (!configObject.is_none())
                {
                    config = configObject.cast<RuntimeConfig>();
                }
                return runtime.initialize(config);
            },
            py::arg("config") = py::none())
        .def("shutdown", &Runtime::shutdown)
        .def("prepare", &Runtime::prepare)
        .def("step_physics", &Runtime::stepPhysics)
        .def("step_simulation_sensors", &Runtime::stepSimulationSensors)
        .def("step_visual_sensors", &Runtime::stepVisualSensors)
        .def("end_frame", &Runtime::endFrame)
        .def("world", [](Runtime &runtime) -> World & { return runtime.getWorld(); },
             py::return_value_policy::reference_internal)
        .def("resources",
             [](Runtime &runtime) -> RenderResourceManager & { return runtime.getResources(); },
             py::return_value_policy::reference_internal)
        .def("create_render_target",
             [](Runtime &runtime, const GpuRenderTargetDesc &desc)
             {
                 auto *device = runtime.getGpuDevice();
                 if (device == nullptr)
                 {
                     throw std::runtime_error("Runtime GPU device is unavailable.");
                 }
                 return device->renderTargetSystem().createRenderTarget(desc);
             })
        .def("is_valid_render_target",
             [](Runtime &runtime, const GpuRenderTargetHandle target)
             {
                 auto *device = runtime.getGpuDevice();
                 return device != nullptr && device->renderTargetSystem().isValidRenderTarget(target);
             })
        .def("request_render_target_readback",
             [](Runtime &runtime, const GpuRenderTargetBinding &binding)
             {
                 auto *device = runtime.getGpuDevice();
                 if (device == nullptr)
                 {
                     throw std::runtime_error("Runtime GPU device is unavailable.");
                 }
                 return device->renderTargetSystem().requestRenderTargetReadback(binding);
             })
        .def("try_get_render_target_readback", &tryGetRenderTargetReadback);

    m.def("make_cube_mesh", &makeCubeMesh, py::arg("half_extent"),
          py::arg("debug_name") = "Python.CubeMesh");
}
