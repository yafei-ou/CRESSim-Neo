#include "common/frame_context.h"
#include "common/math_types.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "engine/runtime_internal.h"
#include "examples/helpers/shape_meshes.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace
{

enum DLDeviceType
{
    kDLCUDA = 2,
};

enum DLDataTypeCode
{
    kDLInt   = 0,
    kDLUInt  = 1,
    kDLFloat = 2,
    kDLBool  = 6,
};

struct DLDevice
{
    int device_type;
    int device_id;
};

struct DLDataType
{
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLTensor
{
    void *data;
    DLDevice device;
    int ndim;
    DLDataType dtype;
    std::int64_t *shape;
    std::int64_t *strides;
    std::uint64_t byte_offset;
};

struct DLManagedTensor
{
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(DLManagedTensor *self);
};

struct ExportedSharedBufferTensorContext
{
    std::shared_ptr<void> sharedBufferLease;
};

using cressim::neo::common::FrameContext;
using cressim::neo::common::Transform;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::CustomComputeDispatchDesc;
using cressim::neo::engine::CustomComputeDispatchMode;
using cressim::neo::engine::CustomComputePassDesc;
using cressim::neo::engine::CustomComputePassHandle;
using cressim::neo::engine::CustomComputeResourceAccess;
using cressim::neo::engine::CustomComputeResourceBindingDesc;
using cressim::neo::engine::CustomComputeResourceDesc;
using cressim::neo::engine::CustomComputeResourceKind;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::SharedBufferAccess;
using cressim::neo::engine::SharedBufferBindFlags;
using cressim::neo::engine::SharedBufferCudaView;
using cressim::neo::engine::SharedBufferDesc;
using cressim::neo::engine::SharedBufferHandle;
using cressim::neo::engine::SharedBufferInfo;
using cressim::neo::engine::SharedBufferTensorDesc;
using cressim::neo::engine::SharedBufferTensorDTypeCode;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDeviceDesc;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::gpu::RenderOutputBinding;
using cressim::neo::gpu::RenderOutputMode;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialPipelineDesc;
using cressim::neo::graphics::MaterialProgramFamily;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RendererDesc;
using cressim::neo::graphics::RenderResourceManager;
using cressim::neo::graphics::RenderStats;
using cressim::neo::graphics::TextureHandle;
using cressim::neo::physics::ParticleContactMaterialDesc;
using cressim::neo::physics::PhysicsSolverDesc;

py::object tryGetRenderTargetReadback(Runtime &runtime,
                                      const GpuRenderTargetReadbackRequest request)
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

void deleteManagedTensor(DLManagedTensor *managedTensor)
{
    if (managedTensor == nullptr)
    {
        return;
    }

    delete static_cast<ExportedSharedBufferTensorContext *>(managedTensor->manager_ctx);
    delete[] managedTensor->dl_tensor.shape;
    delete[] managedTensor->dl_tensor.strides;
    delete managedTensor;
}

std::uint8_t toDlpackDTypeCode(const SharedBufferTensorDTypeCode code)
{
    switch (code)
    {
    case SharedBufferTensorDTypeCode::Int:
        return kDLInt;
    case SharedBufferTensorDTypeCode::UInt:
        return kDLUInt;
    case SharedBufferTensorDTypeCode::Float:
        return kDLFloat;
    case SharedBufferTensorDTypeCode::Bool:
        return kDLBool;
    }
    return kDLUInt;
}

std::vector<std::int64_t> makeCompactStrides(const std::vector<std::int64_t> &shape)
{
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i)
    {
        strides[i - 2u] = strides[i - 1u] * shape[i - 1u];
    }
    return strides;
}

py::capsule exportSharedBufferToDLPack(Runtime &runtime, const SharedBufferHandle handle,
                                       const SharedBufferTensorDesc &desc)
{
    SharedBufferCudaView view{};
    if (!runtime.tryGetSharedBufferCudaView(handle, view) || !view.isValid())
    {
        throw std::runtime_error("Shared buffer CUDA view is unavailable.");
    }
    std::shared_ptr<void> sharedBufferLease =
        cressim::neo::engine::RuntimeInternalAccess::retainSharedBufferLease(runtime, handle);
    if (!sharedBufferLease)
    {
        throw std::runtime_error("Shared buffer lease is unavailable.");
    }
    if (desc.shape.empty())
    {
        throw std::runtime_error("Shared buffer tensor export requires a non-empty shape.");
    }
    if (desc.dtypeBits == 0u || (desc.dtypeBits % 8u) != 0u || desc.dtypeLanes == 0u)
    {
        throw std::runtime_error("Shared buffer tensor export requires byte-aligned dtype bits "
                                 "and non-zero lanes.");
    }

    for (const std::int64_t dim : desc.shape)
    {
        if (dim <= 0)
        {
            throw std::runtime_error("Shared buffer tensor shape dimensions must be positive.");
        }
    }

    std::vector<std::int64_t> strides =
        desc.strides.empty() ? makeCompactStrides(desc.shape) : desc.strides;
    if (strides.size() != desc.shape.size())
    {
        throw std::runtime_error("Shared buffer tensor strides must match shape rank.");
    }
    for (const std::int64_t stride : strides)
    {
        if (stride <= 0)
        {
            throw std::runtime_error("Shared buffer tensor strides must be positive.");
        }
    }

    const std::uint64_t dtypeBytes  = (static_cast<std::uint64_t>(desc.dtypeBits) / 8u) *
                                      static_cast<std::uint64_t>(desc.dtypeLanes);
    std::uint64_t lastElementOffset = 0u;
    for (std::size_t i = 0; i < desc.shape.size(); ++i)
    {
        lastElementOffset +=
            static_cast<std::uint64_t>(strides[i]) * static_cast<std::uint64_t>(desc.shape[i] - 1);
    }
    const std::uint64_t requiredBytes = desc.byteOffset + (lastElementOffset + 1u) * dtypeBytes;
    if (requiredBytes > view.sizeBytes)
    {
        throw std::runtime_error("Shared buffer tensor view exceeds shared buffer bounds.");
    }

    auto *managedTensor    = new DLManagedTensor{};
    managedTensor->deleter = &deleteManagedTensor;
    managedTensor->manager_ctx =
        new ExportedSharedBufferTensorContext{std::move(sharedBufferLease)};
    managedTensor->dl_tensor.data =
        static_cast<void *>(static_cast<std::uint8_t *>(view.devicePointer) + desc.byteOffset);
    managedTensor->dl_tensor.device  = {kDLCUDA, view.deviceOrdinal};
    managedTensor->dl_tensor.ndim    = static_cast<int>(desc.shape.size());
    managedTensor->dl_tensor.dtype   = {toDlpackDTypeCode(desc.dtypeCode), desc.dtypeBits,
                                        desc.dtypeLanes};
    managedTensor->dl_tensor.shape   = new std::int64_t[desc.shape.size()];
    managedTensor->dl_tensor.strides = new std::int64_t[strides.size()];
    std::memcpy(managedTensor->dl_tensor.shape, desc.shape.data(),
                desc.shape.size() * sizeof(std::int64_t));
    std::memcpy(managedTensor->dl_tensor.strides, strides.data(),
                strides.size() * sizeof(std::int64_t));
    managedTensor->dl_tensor.byte_offset = 0u;

    return py::capsule(managedTensor, "dltensor",
                       [](PyObject *capsule)
                       {
                           if (capsule == nullptr)
                           {
                               return;
                           }

                           const bool isLiveCapsule = PyCapsule_IsValid(capsule, "dltensor") != 0;
                           if (!isLiveCapsule)
                           {
                               PyErr_Clear();
                               return;
                           }

                           auto *managed = static_cast<DLManagedTensor *>(
                               PyCapsule_GetPointer(capsule, "dltensor"));
                           if (managed == nullptr)
                           {
                               PyErr_Clear();
                               return;
                           }
                           if (managed->deleter != nullptr)
                           {
                               managed->deleter(managed);
                           }
                       });
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

    py::enum_<CustomComputeResourceKind>(m, "CustomComputeResourceKind")
        .value("Buffer", CustomComputeResourceKind::Buffer);

    py::enum_<CustomComputeResourceAccess>(m, "CustomComputeResourceAccess")
        .value("ReadOnly", CustomComputeResourceAccess::ReadOnly)
        .value("WriteOnly", CustomComputeResourceAccess::WriteOnly)
        .value("ReadWrite", CustomComputeResourceAccess::ReadWrite);

    py::enum_<CustomComputeDispatchMode>(m, "CustomComputeDispatchMode")
        .value("ExplicitGroupCount", CustomComputeDispatchMode::ExplicitGroupCount)
        .value("ResourceElementCount", CustomComputeDispatchMode::ResourceElementCount);

    py::enum_<SharedBufferAccess>(m, "SharedBufferAccess")
        .value("ReadOnly", SharedBufferAccess::ReadOnly)
        .value("WriteOnly", SharedBufferAccess::WriteOnly)
        .value("ReadWrite", SharedBufferAccess::ReadWrite);

    py::enum_<SharedBufferBindFlags>(m, "SharedBufferBindFlags", py::arithmetic())
        .value("None", SharedBufferBindFlags::None)
        .value("ShaderResource", SharedBufferBindFlags::ShaderResource)
        .value("UnorderedAccess", SharedBufferBindFlags::UnorderedAccess)
        .def("__or__", [](const SharedBufferBindFlags lhs, const SharedBufferBindFlags rhs)
             { return lhs | rhs; })
        .def("__and__", [](const SharedBufferBindFlags lhs, const SharedBufferBindFlags rhs)
             { return lhs & rhs; });

    py::enum_<SharedBufferTensorDTypeCode>(m, "SharedBufferTensorDTypeCode")
        .value("Int", SharedBufferTensorDTypeCode::Int)
        .value("UInt", SharedBufferTensorDTypeCode::UInt)
        .value("Float", SharedBufferTensorDTypeCode::Float)
        .value("Bool", SharedBufferTensorDTypeCode::Bool);

    py::class_<CustomComputePassHandle>(m, "CustomComputePassHandle")
        .def(py::init<>())
        .def_readwrite("id", &CustomComputePassHandle::id)
        .def("is_valid", &CustomComputePassHandle::isValid);

    py::class_<SharedBufferHandle>(m, "SharedBufferHandle")
        .def(py::init<>())
        .def_readwrite("id", &SharedBufferHandle::id)
        .def("is_valid", &SharedBufferHandle::isValid);

    py::class_<SharedBufferDesc>(m, "SharedBufferDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &SharedBufferDesc::debugName)
        .def_readwrite("element_stride_bytes", &SharedBufferDesc::elementStrideBytes)
        .def_readwrite("element_count", &SharedBufferDesc::elementCount)
        .def_readwrite("minimum_capacity", &SharedBufferDesc::minimumCapacity)
        .def_readwrite("access", &SharedBufferDesc::access)
        .def_readwrite("bind_flags", &SharedBufferDesc::bindFlags);

    py::class_<SharedBufferInfo>(m, "SharedBufferInfo")
        .def(py::init<>())
        .def_readwrite("handle", &SharedBufferInfo::handle)
        .def_readwrite("debug_name", &SharedBufferInfo::debugName)
        .def_readwrite("element_stride_bytes", &SharedBufferInfo::elementStrideBytes)
        .def_readwrite("element_count", &SharedBufferInfo::elementCount)
        .def_readwrite("capacity", &SharedBufferInfo::capacity)
        .def_readwrite("size_bytes", &SharedBufferInfo::sizeBytes)
        .def_readwrite("access", &SharedBufferInfo::access)
        .def_readwrite("bind_flags", &SharedBufferInfo::bindFlags)
        .def_readwrite("exportable", &SharedBufferInfo::exportable)
        .def_readwrite("imported_into_cuda", &SharedBufferInfo::importedIntoCuda);

    py::class_<SharedBufferCudaView>(m, "SharedBufferCudaView")
        .def(py::init<>())
        .def_property_readonly("device_pointer", [](const SharedBufferCudaView &view)
                               { return reinterpret_cast<std::uintptr_t>(view.devicePointer); })
        .def_readwrite("size_bytes", &SharedBufferCudaView::sizeBytes)
        .def_readwrite("device_ordinal", &SharedBufferCudaView::deviceOrdinal)
        .def("is_valid", &SharedBufferCudaView::isValid);

    py::class_<SharedBufferTensorDesc>(m, "SharedBufferTensorDesc")
        .def(py::init<>())
        .def_readwrite("shape", &SharedBufferTensorDesc::shape)
        .def_readwrite("strides", &SharedBufferTensorDesc::strides)
        .def_readwrite("dtype_code", &SharedBufferTensorDesc::dtypeCode)
        .def_readwrite("dtype_bits", &SharedBufferTensorDesc::dtypeBits)
        .def_readwrite("dtype_lanes", &SharedBufferTensorDesc::dtypeLanes)
        .def_readwrite("byte_offset", &SharedBufferTensorDesc::byteOffset);

    py::class_<CustomComputeResourceDesc>(m, "CustomComputeResourceDesc")
        .def(py::init<>())
        .def_readwrite("key", &CustomComputeResourceDesc::key)
        .def_readwrite("kind", &CustomComputeResourceDesc::kind)
        .def_readwrite("access", &CustomComputeResourceDesc::access)
        .def_readwrite("element_count", &CustomComputeResourceDesc::elementCount)
        .def_readwrite("element_stride_bytes", &CustomComputeResourceDesc::elementStrideBytes)
        .def_readwrite("binding_generation", &CustomComputeResourceDesc::bindingGeneration);

    py::class_<CustomComputeResourceBindingDesc>(m, "CustomComputeResourceBindingDesc")
        .def(py::init<>())
        .def_readwrite("shader_variable_name",
                       &CustomComputeResourceBindingDesc::shaderVariableName)
        .def_readwrite("resource_key", &CustomComputeResourceBindingDesc::resourceKey)
        .def_readwrite("shared_buffer_handle",
                       &CustomComputeResourceBindingDesc::sharedBufferHandle)
        .def_readwrite("access", &CustomComputeResourceBindingDesc::access);

    py::class_<CustomComputeDispatchDesc>(m, "CustomComputeDispatchDesc")
        .def(py::init<>())
        .def_readwrite("mode", &CustomComputeDispatchDesc::mode)
        .def_readwrite("group_count_x", &CustomComputeDispatchDesc::groupCountX)
        .def_readwrite("group_count_y", &CustomComputeDispatchDesc::groupCountY)
        .def_readwrite("group_count_z", &CustomComputeDispatchDesc::groupCountZ)
        .def_readwrite("count_resource_key", &CustomComputeDispatchDesc::countResourceKey);

    py::class_<CustomComputePassDesc>(m, "CustomComputePassDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &CustomComputePassDesc::debugName)
        .def_readwrite("shader_path", &CustomComputePassDesc::shaderPath)
        .def_readwrite("shader_source", &CustomComputePassDesc::shaderSource)
        .def_readwrite("entry_point", &CustomComputePassDesc::entryPoint)
        .def_readwrite("thread_group_size_x", &CustomComputePassDesc::threadGroupSizeX)
        .def_readwrite("thread_group_size_y", &CustomComputePassDesc::threadGroupSizeY)
        .def_readwrite("thread_group_size_z", &CustomComputePassDesc::threadGroupSizeZ)
        .def_readwrite("resource_bindings", &CustomComputePassDesc::resourceBindings)
        .def_readwrite("constant_buffer_variable_name",
                       &CustomComputePassDesc::constantBufferVariableName)
        .def_readwrite("constant_buffer_size_bytes",
                       &CustomComputePassDesc::constantBufferSizeBytes)
        .def_readwrite("constant_data", &CustomComputePassDesc::constantData)
        .def_readwrite("dispatch", &CustomComputePassDesc::dispatch);

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

    py::enum_<CameraComponent::BackgroundMode>(m, "CameraBackgroundMode")
        .value("ClearColor", CameraComponent::BackgroundMode::ClearColor)
        .value("EnvironmentCubemap", CameraComponent::BackgroundMode::EnvironmentCubemap);

    py::enum_<cressim::neo::physics::RigidBodyType>(m, "RigidBodyType")
        .value("Static", cressim::neo::physics::RigidBodyType::Static)
        .value("Kinematic", cressim::neo::physics::RigidBodyType::Kinematic)
        .value("Dynamic", cressim::neo::physics::RigidBodyType::Dynamic);

    py::enum_<cressim::neo::physics::ColliderShapeType>(m, "ColliderShapeType")
        .value("Sphere", cressim::neo::physics::ColliderShapeType::Sphere)
        .value("Box", cressim::neo::physics::ColliderShapeType::Box)
        .value("Capsule", cressim::neo::physics::ColliderShapeType::Capsule);

    py::enum_<cressim::neo::gpu::VulkanShaderCompilerMode>(m, "VulkanShaderCompilerMode")
        .value("Auto", cressim::neo::gpu::VulkanShaderCompilerMode::Auto)
        .value("ForceDefault", cressim::neo::gpu::VulkanShaderCompilerMode::ForceDefault)
        .value("ForceDXC", cressim::neo::gpu::VulkanShaderCompilerMode::ForceDXC);

    py::enum_<MaterialProgramFamily>(m, "MaterialProgramFamily")
        .value("StandardLit", MaterialProgramFamily::StandardLit)
        .value("SoftBodyLit", MaterialProgramFamily::SoftBodyLit)
        .value("CurveLit", MaterialProgramFamily::CurveLit);

    py::enum_<cressim::neo::graphics::MaterialFeatureFlags>(m, "MaterialFeatureFlags",
                                                            py::arithmetic())
        .value("None", cressim::neo::graphics::MaterialFeatureFlags::None)
        .value("AlphaTest", cressim::neo::graphics::MaterialFeatureFlags::AlphaTest)
        .value("NormalMap", cressim::neo::graphics::MaterialFeatureFlags::NormalMap)
        .value("ClearCoat", cressim::neo::graphics::MaterialFeatureFlags::ClearCoat)
        .value("DoubleSided", cressim::neo::graphics::MaterialFeatureFlags::DoubleSided);

    py::enum_<MaterialRenderMode>(m, "MaterialRenderMode")
        .value("Opaque", MaterialRenderMode::Opaque)
        .value("Cutout", MaterialRenderMode::Cutout)
        .value("Transparent", MaterialRenderMode::Transparent);

    py::enum_<IblQualityTier>(m, "IblQualityTier")
        .value("Off", IblQualityTier::Off)
        .value("DiffuseOnly", IblQualityTier::DiffuseOnly)
        .value("Full", IblQualityTier::Full);

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
        .def_readwrite("layer_count", &GpuRenderTargetBinding::layerCount)
        .def("is_valid", &GpuRenderTargetBinding::isValid)
        .def(py::self == py::self);

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
        .def_readwrite("binding", &GpuRenderTargetReadbackEvent::binding)
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
                       &cressim::neo::common::SceneLayoutDesc::maxCamerasPerEnv)
        .def("total_renderable_object_capacity",
             &cressim::neo::common::SceneLayoutDesc::totalRenderableObjectCapacity)
        .def("total_light_capacity", &cressim::neo::common::SceneLayoutDesc::totalLightCapacity)
        .def("total_camera_capacity", &cressim::neo::common::SceneLayoutDesc::totalCameraCapacity);

    py::class_<cressim::neo::gpu::GpuRenderViewport>(m, "GpuRenderViewport")
        .def(py::init<>())
        .def_readwrite("x", &cressim::neo::gpu::GpuRenderViewport::x)
        .def_readwrite("y", &cressim::neo::gpu::GpuRenderViewport::y)
        .def_readwrite("width", &cressim::neo::gpu::GpuRenderViewport::width)
        .def_readwrite("height", &cressim::neo::gpu::GpuRenderViewport::height);

    py::class_<GpuDeviceDesc::PresentationDesc>(m, "GpuPresentationDesc")
        .def(py::init<>())
        .def_readwrite("enabled", &GpuDeviceDesc::PresentationDesc::enabled)
        .def_readwrite("sync_interval", &GpuDeviceDesc::PresentationDesc::syncInterval)
        .def_readwrite("preferred_color_format",
                       &GpuDeviceDesc::PresentationDesc::preferredColorFormat)
        .def_property(
            "native_window", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeWindow); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeWindow = reinterpret_cast<void *>(value); })
        .def_readwrite("native_window_id", &GpuDeviceDesc::PresentationDesc::nativeWindowId)
        .def_property(
            "native_display", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeDisplay); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeDisplay = reinterpret_cast<void *>(value); })
        .def_property(
            "native_connection", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeConnection); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeConnection = reinterpret_cast<void *>(value); });

    py::class_<GpuDeviceDesc>(m, "GpuDeviceDesc")
        .def(py::init<>())
        .def_readwrite("preferred_backend", &GpuDeviceDesc::preferredBackend)
        .def_readwrite("enable_validation", &GpuDeviceDesc::enableValidation)
        .def_readwrite("default_render_target_desc", &GpuDeviceDesc::defaultRenderTargetDesc)
        .def_readwrite("presentation", &GpuDeviceDesc::presentation)
        .def_readwrite("vulkan_shader_compiler_mode", &GpuDeviceDesc::vulkanShaderCompilerMode)
        .def_readwrite("shader_directory", &GpuDeviceDesc::shaderDirectory);

    py::class_<RendererDesc>(m, "RendererDesc")
        .def(py::init<>())
        .def_readwrite("ibl_quality_tier", &RendererDesc::iblQualityTier);

    py::class_<PhysicsSolverDesc>(m, "PhysicsSolverDesc")
        .def(py::init<>())
        .def_readwrite("substeps", &PhysicsSolverDesc::substeps)
        .def_readwrite("default_iterations", &PhysicsSolverDesc::defaultIterations)
        .def_readwrite("fluid_iterations", &PhysicsSolverDesc::fluidIterations)
        .def_readwrite("soft_internal_iterations", &PhysicsSolverDesc::softInternalIterations)
        .def_readwrite("soft_contact_iterations", &PhysicsSolverDesc::softContactIterations)
        .def_readwrite("rigid_joint_iterations", &PhysicsSolverDesc::rigidJointIterations)
        .def_readwrite("rigid_rigid_contact_iterations",
                       &PhysicsSolverDesc::rigidRigidContactIterations)
        .def_readwrite("enable_blocking_readback", &PhysicsSolverDesc::enableBlockingReadback);

    py::class_<RuntimeConfig>(m, "RuntimeConfig")
        .def(py::init<>())
        .def_readwrite("gpu_device_desc", &RuntimeConfig::gpuDeviceDesc)
        .def_readwrite("scene_layout", &RuntimeConfig::sceneLayout)
        .def_readwrite("renderer_desc", &RuntimeConfig::rendererDesc)
        .def_readwrite("physics_desc", &RuntimeConfig::physicsDesc);

    py::class_<MeshHandle>(m, "MeshHandle").def(py::init<>()).def_readwrite("id", &MeshHandle::id);

    py::class_<MaterialHandle>(m, "MaterialHandle")
        .def(py::init<>())
        .def_readwrite("id", &MaterialHandle::id);

    py::class_<TextureHandle>(m, "TextureHandle")
        .def(py::init<>())
        .def_readwrite("id", &TextureHandle::id);

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

    py::class_<MaterialPipelineDesc>(m, "MaterialPipelineDesc")
        .def(py::init<>())
        .def_readwrite("program_family", &MaterialPipelineDesc::programFamily)
        .def_readwrite("feature_flags", &MaterialPipelineDesc::featureFlags)
        .def_readwrite("alpha_cutoff", &MaterialPipelineDesc::alphaCutoff);

    py::class_<MaterialResourceDesc>(m, "MaterialResourceDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &MaterialResourceDesc::debugName)
        .def_readwrite("base_color", &MaterialResourceDesc::baseColor)
        .def_readwrite("metallic", &MaterialResourceDesc::metallic)
        .def_readwrite("roughness", &MaterialResourceDesc::roughness)
        .def_readwrite("emissive_factor", &MaterialResourceDesc::emissiveFactor)
        .def_readwrite("base_color_texture", &MaterialResourceDesc::baseColorTexture)
        .def_readwrite("normal_texture", &MaterialResourceDesc::normalTexture)
        .def_readwrite("metallic_roughness_texture",
                       &MaterialResourceDesc::metallicRoughnessTexture)
        .def_readwrite("emissive_texture", &MaterialResourceDesc::emissiveTexture)
        .def_readwrite("ao_texture", &MaterialResourceDesc::aoTexture)
        .def_readwrite("pipeline", &MaterialResourceDesc::pipeline)
        .def_readwrite("render_mode", &MaterialResourceDesc::renderMode)
        .def_readwrite("render_order", &MaterialResourceDesc::renderOrder)
        .def_readwrite("opacity", &MaterialResourceDesc::opacity)
        .def_readwrite("casts_shadows", &MaterialResourceDesc::castsShadows)
        .def_readwrite("receives_shadows", &MaterialResourceDesc::receivesShadows);

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
        .def_readwrite("viewport", &CameraComponent::viewport)
        .def_readwrite("clear_color", &CameraComponent::clearColor)
        .def_readwrite("clear_depth", &CameraComponent::clearDepth)
        .def_readwrite("clear_color_value", &CameraComponent::clearColorValue)
        .def_readwrite("clear_depth_value", &CameraComponent::clearDepthValue)
        .def_readwrite("background_mode", &CameraComponent::backgroundMode)
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

    py::class_<ParticleContactMaterialDesc>(m, "ParticleContactMaterialDesc")
        .def(py::init<>())
        .def_readwrite("friction", &ParticleContactMaterialDesc::friction)
        .def_readwrite("restitution", &ParticleContactMaterialDesc::restitution)
        .def_readwrite("damping", &ParticleContactMaterialDesc::damping)
        .def_readwrite("static_friction", &ParticleContactMaterialDesc::staticFriction);

    py::class_<RigidBodyComponent>(m, "RigidBodyComponent")
        .def(py::init<>())
        .def_readwrite("linear_velocity", &RigidBodyComponent::linearVelocity)
        .def_readwrite("angular_velocity", &RigidBodyComponent::angularVelocity)
        .def_readwrite("inverse_inertia_local", &RigidBodyComponent::inverseInertiaLocal)
        .def_readwrite("proxy_particle_local_positions",
                       &RigidBodyComponent::proxyParticleLocalPositions)
        .def_readwrite("proxy_particle_material", &RigidBodyComponent::proxyParticleMaterial)
        .def_readwrite("body_type", &RigidBodyComponent::bodyType)
        .def_readwrite("inverse_mass", &RigidBodyComponent::inverseMass)
        .def_readwrite("proxy_particle_radius", &RigidBodyComponent::proxyParticleRadius)
        .def_readwrite("proxy_collision_layer", &RigidBodyComponent::proxyCollisionLayer)
        .def_readwrite("proxy_collision_mask", &RigidBodyComponent::proxyCollisionMask)
        .def_readwrite("suturing_enabled", &RigidBodyComponent::suturingEnabled)
        .def_readwrite("needle_tip_proxy_index", &RigidBodyComponent::needleTipProxyIndex)
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
        .def_readwrite("id", &cressim::neo::engine::ColliderHandle::id)
        .def("is_valid", &cressim::neo::engine::ColliderHandle::isValid);

    py::class_<RenderStats>(m, "RenderStats")
        .def(py::init<>())
        .def_readwrite("draw_calls", &RenderStats::drawCalls)
        .def_readwrite("opaque_draw_calls", &RenderStats::opaqueDrawCalls)
        .def_readwrite("transparent_draw_calls", &RenderStats::transparentDrawCalls)
        .def_readwrite("shadow_draw_calls", &RenderStats::shadowDrawCalls)
        .def_readwrite("renderable_count", &RenderStats::renderableCount)
        .def_readwrite("light_count", &RenderStats::lightCount)
        .def_readwrite("rendered_camera_count", &RenderStats::renderedCameraCount)
        .def_readwrite("render_target_resize_requests", &RenderStats::renderTargetResizeRequests)
        .def_readwrite("render_target_resize_no_ops", &RenderStats::renderTargetResizeNoOps)
        .def_readwrite("render_target_recreate_count", &RenderStats::renderTargetRecreateCount)
        .def_readwrite("render_target_resize_conflicts", &RenderStats::renderTargetResizeConflicts);

    py::class_<RenderResourceManager>(m, "RenderResourceManager")
        .def("register_mesh", &RenderResourceManager::registerMesh)
        .def("register_material", &RenderResourceManager::registerMaterial)
        .def("is_valid_mesh",
             py::overload_cast<MeshHandle>(&RenderResourceManager::isValid, py::const_))
        .def("is_valid_material",
             py::overload_cast<MaterialHandle>(&RenderResourceManager::isValid, py::const_))
        .def("try_get_mesh",
             [](const RenderResourceManager &resources, const MeshHandle mesh) -> py::object
             {
                 if (const auto *desc = resources.tryGetMesh(mesh))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             })
        .def("try_get_material",
             [](const RenderResourceManager &resources, const MaterialHandle material) -> py::object
             {
                 if (const auto *desc = resources.tryGetMaterial(material))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             })
        .def("try_get_mesh_local_bounds",
             [](const RenderResourceManager &resources, const MeshHandle mesh) -> py::object
             {
                 Diligent::float3 boundsMin{};
                 Diligent::float3 boundsMax{};
                 if (!resources.tryGetMeshLocalBounds(mesh, boundsMin, boundsMax))
                 {
                     return py::none();
                 }
                 return py::make_tuple(boundsMin, boundsMax);
             })
        .def("mesh_version", &RenderResourceManager::meshVersion);

    py::class_<World>(m, "World")
        .def("create_entity", &World::createEntity, py::arg("env_index") = 0u)
        .def("destroy_entity", &World::destroyEntity)
        .def("set_scene_layout", &World::setSceneLayout)
        .def("scene_layout", &World::sceneLayout, py::return_value_policy::reference_internal)
        .def("set_entity_environment", &World::setEntityEnvironment)
        .def("entity_environment", &World::entityEnvironment)
        .def("is_alive", &World::isAlive)
        .def("entities", &World::entities, py::return_value_policy::reference_internal)
        .def("set_transform", &World::setTransform)
        .def("remove_transform", &World::removeTransform)
        .def("try_get_transform", &World::tryGetTransform)
        .def("set_mesh_renderer", &World::setMeshRenderer)
        .def("remove_mesh_renderer", &World::removeMeshRenderer)
        .def("try_get_mesh_renderer", &World::tryGetMeshRenderer)
        .def("set_camera", &World::setCamera)
        .def("remove_camera", &World::removeCamera)
        .def("try_get_camera", &World::tryGetCamera)
        .def("set_directional_light", &World::setDirectionalLight)
        .def("remove_directional_light", &World::removeDirectionalLight)
        .def("try_get_directional_light", &World::tryGetDirectionalLight)
        .def("set_rigid_body", &World::setRigidBody)
        .def("remove_rigid_body", &World::removeRigidBody)
        .def("try_get_rigid_body", &World::tryGetRigidBody)
        .def("add_collider", &World::addCollider)
        .def("update_collider", &World::updateCollider)
        .def("remove_collider", &World::removeCollider)
        .def("try_get_collider", &World::tryGetCollider)
        .def("collider_handles", &World::colliderHandles,
             py::return_value_policy::reference_internal);

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
        .def("upload_world", &Runtime::uploadWorld)
        .def("create_shared_buffer", &Runtime::createSharedBuffer)
        .def("destroy_shared_buffer", &Runtime::destroySharedBuffer)
        .def("list_shared_buffers", &Runtime::listSharedBuffers)
        .def("try_get_shared_buffer_info",
             [](Runtime &runtime, const SharedBufferHandle handle) -> py::object
             {
                 SharedBufferInfo info{};
                 if (!runtime.tryGetSharedBufferInfo(handle, info))
                 {
                     return py::none();
                 }
                 return py::cast(info);
             })
        .def("try_get_shared_buffer_cuda_view",
             [](Runtime &runtime, const SharedBufferHandle handle) -> py::object
             {
                 SharedBufferCudaView view{};
                 if (!runtime.tryGetSharedBufferCudaView(handle, view))
                 {
                     return py::none();
                 }
                 return py::cast(view);
             })
        .def("sync_shared_buffer_to_cuda", &Runtime::syncSharedBufferToCuda)
        .def("sync_shared_buffer_from_cuda", &Runtime::syncSharedBufferFromCuda)
        .def("shared_buffer_to_dlpack", &exportSharedBufferToDLPack)
        .def("step_physics", &Runtime::stepPhysics)
        .def("step_simulation_sensors", &Runtime::stepSimulationSensors)
        .def("step_visual_sensors", &Runtime::stepVisualSensors)
        .def("end_frame", &Runtime::endFrame)
        .def("list_custom_compute_resources", &Runtime::listCustomComputeResources)
        .def("create_custom_compute_pass", &Runtime::createCustomComputePass)
        .def("update_custom_compute_pass_constants",
             [](Runtime &runtime, const CustomComputePassHandle handle, py::bytes data)
             {
                 std::string bytes = data;
                 return runtime.updateCustomComputePassConstants(
                     handle, std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
             })
        .def("execute_custom_compute_pass", &Runtime::executeCustomComputePass)
        .def("destroy_custom_compute_pass", &Runtime::destroyCustomComputePass)
        .def("last_render_stats", &Runtime::lastRenderStats,
             py::return_value_policy::reference_internal)
        .def(
            "world", [](Runtime &runtime) -> World & { return runtime.getWorld(); },
            py::return_value_policy::reference_internal)
        .def(
            "resources", [](Runtime &runtime) -> RenderResourceManager &
            { return runtime.getResources(); }, py::return_value_policy::reference_internal)
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
                 return device != nullptr &&
                        device->renderTargetSystem().isValidRenderTarget(target);
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

    m.def("make_cube_mesh", &cressim::neo::examples::helpers::makeCubeMesh, py::arg("half_extent"),
          py::arg("debug_name") = "Python.CubeMesh", py::arg("uv_scale") = 1.0f);
    m.def("make_box_mesh", &cressim::neo::examples::helpers::makeBoxMesh, py::arg("half_extents"),
          py::arg("debug_name"), py::arg("uv_scale_u") = 1.0f, py::arg("uv_scale_v") = 1.0f);
    m.def("make_plane_mesh",
          py::overload_cast<float, const std::string &, float>(
              &cressim::neo::examples::helpers::makePlaneMesh),
          py::arg("half_extent"), py::arg("debug_name"), py::arg("uv_scale") = 1.0f);
    m.def("make_sphere_mesh", &cressim::neo::examples::helpers::makeSphereMesh, py::arg("radius"),
          py::arg("slices"), py::arg("stacks"), py::arg("debug_name"));
    m.def("make_capsule_mesh", &cressim::neo::examples::helpers::makeCapsuleMesh, py::arg("radius"),
          py::arg("half_height"), py::arg("slices"), py::arg("hemisphere_rings"),
          py::arg("body_rings"), py::arg("debug_name"));
}
