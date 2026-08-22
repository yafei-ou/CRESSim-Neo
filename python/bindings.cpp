#include "common/frame_context.h"
#include "common/math_types.h"
#include "engine/components.h"
#include "engine/constraint_layout_mapping.h"
#include "engine/joint_layout_mapping.h"
#include "engine/particle_layout_mapping.h"
#include "engine/rigid_layout_mapping.h"
#include "engine/runtime.h"
#include "examples/helpers/shape_meshes.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"
#ifdef CRESSIM_NEO_PYTHON_HAS_VIEWER
#include "viewer/debug_viewer_app.h"
#endif

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
    cressim::neo::engine::SharedBufferLease sharedBufferLease;
};

template <typename JointState>
JointState remapJointBodiesToEntityIds(const cressim::neo::engine::World &world,
                                       const JointState &state)
{
    JointState remapped        = state;
    const auto &rigidBodies    = world.physicsWorld().rigidBodySoA();
    const auto resolveEntityId = [&](const cressim::neo::physics::RigidBodyId bodyId)
    {
        for (std::size_t i = 0; i < rigidBodies.rigidBodyIds.size(); ++i)
        {
            if (rigidBodies.rigidBodyIds[i] == bodyId && i < rigidBodies.entityIds.size())
            {
                return rigidBodies.entityIds[i];
            }
        }
        return bodyId;
    };
    remapped.bodyA = resolveEntityId(state.bodyA);
    remapped.bodyB = resolveEntityId(state.bodyB);
    return remapped;
}

using cressim::neo::common::FrameContext;
using cressim::neo::common::Transform;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::ConstraintLayoutMapping;
using cressim::neo::engine::CustomComputeDispatchDesc;
using cressim::neo::engine::CustomComputeDispatchMode;
using cressim::neo::engine::CustomComputePassDesc;
using cressim::neo::engine::CustomComputePassHandle;
using cressim::neo::engine::CustomComputeResourceAccess;
using cressim::neo::engine::CustomComputeResourceBindingDesc;
using cressim::neo::engine::CustomComputeResourceDesc;
using cressim::neo::engine::CustomComputeResourceKind;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::JointLayoutMapping;
using cressim::neo::engine::MeshfreeSoftBodyComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::ParticleLayoutMapping;
using cressim::neo::engine::PointLightComponent;
using cressim::neo::engine::ProceduralDeformableCurveRenderComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::RigidDistanceConstraintLayoutMapping;
using cressim::neo::engine::RigidLayoutMapping;
using cressim::neo::engine::RigidParticleAttachmentConstraintLayoutMapping;
using cressim::neo::engine::RoutedCableConstraintLayoutMapping;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::RuntimeInfo;
using cressim::neo::engine::SharedBufferAccess;
using cressim::neo::engine::SharedBufferBindFlags;
using cressim::neo::engine::SharedBufferCudaView;
using cressim::neo::engine::SharedBufferDesc;
using cressim::neo::engine::SharedBufferHandle;
using cressim::neo::engine::SharedBufferInfo;
using cressim::neo::engine::SharedBufferTensorDesc;
using cressim::neo::engine::SharedBufferTensorDTypeCode;
using cressim::neo::engine::SoftBodyAuthoringParticles;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::SpotLightComponent;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::UltrasoundAmplitudeRange;
using cressim::neo::engine::UltrasoundProbeComponent;
using cressim::neo::engine::UltrasoundProbeLayout;
using cressim::neo::engine::UltrasoundProbeResult;
using cressim::neo::engine::UltrasoundRendererComponent;
using cressim::neo::engine::UltrasoundScattererSourceComponent;
using cressim::neo::engine::World;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDeviceDesc;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::gpu::GpuRenderTargetTexturePlane;
using cressim::neo::gpu::RenderOutputBinding;
using cressim::neo::gpu::RenderOutputMode;
using cressim::neo::graphics::EnvironmentFluidDesc;
using cressim::neo::graphics::EnvironmentIblDesc;
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
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureDimension;
using cressim::neo::graphics::TextureHandle;
using cressim::neo::graphics::TextureMipPolicy;
using cressim::neo::graphics::TexturePixelFormat;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::physics::AuthoredParticleCollisionFilterState;
using cressim::neo::physics::AuthoredParticleDistanceConstraintState;
using cressim::neo::physics::AuthoredParticleReference;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredParticleSequenceState;
using cressim::neo::physics::AuthoredRigidDistanceConstraintState;
using cressim::neo::physics::AuthoredRigidParticleAttachmentConstraintState;
using cressim::neo::physics::AuthoredRoutedCableConstraintState;
using cressim::neo::physics::AuthoredRoutedCableRoutePoint;
using cressim::neo::physics::AuthoredStrandRigidAttachmentConstraintState;
using cressim::neo::physics::AuthoredSuturingSequenceState;
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::FluidMaterialDesc;
using cressim::neo::physics::FluidRegularGridSource;
using cressim::neo::physics::FluidSourceDesc;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::ParticleContactMaterialDesc;
using cressim::neo::physics::ParticleKind;
using cressim::neo::physics::ParticleOwnerType;
using cressim::neo::physics::ParticleStrandRole;
using cressim::neo::physics::PhysicsSolverDesc;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::physics::SliderJointState;
using cressim::neo::physics::SoftBodyMaterialDesc;
using cressim::neo::physics::SoftBodyMeshfreeParticleSource;
using cressim::neo::physics::SoftBodyRegularGridSource;
using cressim::neo::physics::SoftBodySourceDesc;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::physics::SoftBodyTetGenSource;
using cressim::neo::physics::SoftBodyTetMeshSource;
using cressim::neo::physics::SphericalJointState;
using cressim::neo::physics::StrandMaterialDesc;
#ifdef CRESSIM_NEO_PYTHON_HAS_VIEWER
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;
#endif

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

py::object exportSharedBufferToDLPack(Runtime &runtime, const SharedBufferHandle handle,
                                      const SharedBufferTensorDesc &desc)
{
    SharedBufferCudaView view{};
    if (!runtime.tryGetSharedBufferCudaView(handle, view) || !view.isValid())
    {
        throw std::runtime_error(
            "Shared buffer CUDA view is unavailable. CUDA interop may be disabled in this build "
            "or this shared buffer may not be imported into CUDA.");
    }
    auto sharedBufferLease = runtime.retainSharedBufferLease(handle);
    if (!sharedBufferLease.isValid())
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

    py::class_<Diligent::float3>(m, "Float3", "Three-component single-precision vector.")
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f, "Initializes the vector components.")
        .def_readwrite("x", &Diligent::float3::x, "X component.")
        .def_readwrite("y", &Diligent::float3::y, "Y component.")
        .def_readwrite("z", &Diligent::float3::z, "Z component.");

    py::class_<Diligent::float4>(m, "Float4", "Four-component single-precision vector.")
        .def(py::init<float, float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f, py::arg("w") = 0.0f, "Initializes the vector components.")
        .def_readwrite("x", &Diligent::float4::x, "X component.")
        .def_readwrite("y", &Diligent::float4::y, "Y component.")
        .def_readwrite("z", &Diligent::float4::z, "Z component.")
        .def_readwrite("w", &Diligent::float4::w, "W component.");

    py::class_<Diligent::uint3>(m, "UInt3", "Three-component unsigned 32-bit integer vector.")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t>(), py::arg("x") = 0u,
             py::arg("y") = 0u, py::arg("z") = 0u, "Initializes the vector components.")
        .def_readwrite("x", &Diligent::uint3::x, "X component.")
        .def_readwrite("y", &Diligent::uint3::y, "Y component.")
        .def_readwrite("z", &Diligent::uint3::z, "Z component.");

    py::class_<Diligent::uint4>(m, "UInt4", "Four-component unsigned 32-bit integer vector.")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>(),
             py::arg("x") = 0u, py::arg("y") = 0u, py::arg("z") = 0u, py::arg("w") = 0u,
             "Initializes the vector components.")
        .def_readwrite("x", &Diligent::uint4::x, "X component.")
        .def_readwrite("y", &Diligent::uint4::y, "Y component.")
        .def_readwrite("z", &Diligent::uint4::z, "Z component.")
        .def_readwrite("w", &Diligent::uint4::w, "W component.");

    py::class_<Diligent::QuaternionF>(m, "Quaternion",
                                       "Single-precision quaternion used for 3D rotations.")
        .def(py::init<float, float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f,
             py::arg("z") = 0.0f, py::arg("w") = 1.0f,
             "Initializes quaternion components; the default is the identity rotation.")
        .def_property(
            "x", [](const Diligent::QuaternionF &q) { return q.q.x; },
            [](Diligent::QuaternionF &q, const float value) { q.q.x = value; },
            "X component.")
        .def_property(
            "y", [](const Diligent::QuaternionF &q) { return q.q.y; },
            [](Diligent::QuaternionF &q, const float value) { q.q.y = value; },
            "Y component.")
        .def_property(
            "z", [](const Diligent::QuaternionF &q) { return q.q.z; },
            [](Diligent::QuaternionF &q, const float value) { q.q.z = value; },
            "Z component.")
        .def_property(
            "w", [](const Diligent::QuaternionF &q) { return q.q.w; },
            [](Diligent::QuaternionF &q, const float value) { q.q.w = value; },
            "W component.");

    py::enum_<GpuBackend>(m, "GpuBackend", "GPU execution backend.")
        .value("Null", GpuBackend::Null, "Backend without GPU execution.")
        .value("D3D12", GpuBackend::D3D12, "Direct3D 12 backend.")
        .value("Vulkan", GpuBackend::Vulkan, "Vulkan backend.");

    py::enum_<RenderOutputMode>(m, "RenderOutputMode", "Destination selection mode for rendered output.")
        .value("ManagedPrimary", RenderOutputMode::ManagedPrimary,
               "Use the runtime-managed primary output.")
        .value("ExplicitSurface", RenderOutputMode::ExplicitSurface,
               "Use the explicitly specified render-target binding.");

    py::enum_<CustomComputeResourceKind>(m, "CustomComputeResourceKind",
                                         "Resource type for custom compute shader bindings.")
        .value("Buffer", CustomComputeResourceKind::Buffer,
               "Structured or raw GPU buffer resource.")
        .value("Texture", CustomComputeResourceKind::Texture, "Texture image resource.");

    py::enum_<CustomComputeResourceAccess>(
        m, "CustomComputeResourceAccess", "Access permission mode for custom compute shader resources.")
        .value("ReadOnly", CustomComputeResourceAccess::ReadOnly, "Read-only shader resource.")
        .value("WriteOnly", CustomComputeResourceAccess::WriteOnly,
               "Write-only unordered access resource.")
        .value("ReadWrite", CustomComputeResourceAccess::ReadWrite,
               "Read-write unordered access resource.");

    py::enum_<GpuRenderTargetTexturePlane>(m, "GpuRenderTargetTexturePlane",
                                            "Texture attachment selected from a render target.")
        .value("Color", GpuRenderTargetTexturePlane::Color, "Color attachment.")
        .value("Depth", GpuRenderTargetTexturePlane::Depth, "Depth attachment.");

    py::enum_<CustomComputeDispatchMode>(m, "CustomComputeDispatchMode",
                                         "Compute dispatch mode for executing custom compute passes.")
        .value("ExplicitGroupCount", CustomComputeDispatchMode::ExplicitGroupCount,
               "Dispatch using explicit 3D thread group counts.")
        .value("ResourceElementCount", CustomComputeDispatchMode::ResourceElementCount,
               "Compute thread group count dynamically from resource element count.");

    py::enum_<SharedBufferAccess>(m, "SharedBufferAccess",
                                  "Access permission flags for shared engine GPU buffers.")
        .value("ReadOnly", SharedBufferAccess::ReadOnly,
               "Buffer is read-only by shaders/compute.")
        .value("WriteOnly", SharedBufferAccess::WriteOnly,
               "Buffer is write-only by shaders/compute.")
        .value("ReadWrite", SharedBufferAccess::ReadWrite,
               "Buffer supports read and write access.");

    py::enum_<SharedBufferBindFlags>(m, "SharedBufferBindFlags",
                                     "Binding usage flags for shared GPU buffers.", py::arithmetic())
        .value("None", SharedBufferBindFlags::None, "No binding flags specified.")
        .value("ShaderResource", SharedBufferBindFlags::ShaderResource,
               "Bindable as a Shader Resource View (SRV).")
        .value("UnorderedAccess", SharedBufferBindFlags::UnorderedAccess,
               "Bindable as an Unordered Access View (UAV).")
        .def("__or__", [](const SharedBufferBindFlags lhs, const SharedBufferBindFlags rhs)
             { return lhs | rhs; }, "Bitwise OR operator for SharedBufferBindFlags.")
        .def("__and__", [](const SharedBufferBindFlags lhs, const SharedBufferBindFlags rhs)
             { return lhs & rhs; }, "Bitwise AND operator for SharedBufferBindFlags.");

    py::enum_<SharedBufferTensorDTypeCode>(
        m, "SharedBufferTensorDTypeCode",
        "DLPack tensor data type codes for shared buffer tensor view export.")
        .value("Int", SharedBufferTensorDTypeCode::Int, "Signed integer.")
        .value("UInt", SharedBufferTensorDTypeCode::UInt, "Unsigned integer.")
        .value("Float", SharedBufferTensorDTypeCode::Float, "Floating point.")
        .value("Bool", SharedBufferTensorDTypeCode::Bool, "Boolean.");

    py::class_<CustomComputePassHandle>(
        m, "CustomComputePassHandle", "Handle wrapper identifying a registered custom compute shader pass.")
        .def(py::init<>(), "Initializes an invalid custom compute pass handle.")
        .def_readwrite("id", &CustomComputePassHandle::id, "Unique handle identifier.")
        .def("is_valid", &CustomComputePassHandle::isValid,
             "Checks if the custom compute pass handle is valid.\n\n"
             "Returns:\n"
             "    True if valid, false otherwise.");

    py::class_<SharedBufferHandle>(m, "SharedBufferHandle",
                                   "Handle wrapper identifying an engine-owned shared GPU buffer.")
        .def(py::init<>(), "Initializes an invalid shared-buffer handle.")
        .def_readwrite("id", &SharedBufferHandle::id, "Unique handle identifier.")
        .def("is_valid", &SharedBufferHandle::isValid,
             "Checks if the shared buffer handle is valid.");

    py::class_<SharedBufferDesc>(m, "SharedBufferDesc",
                                 "Descriptor for creating an engine-owned shared GPU buffer.")
        .def(py::init<>(), "Initializes the descriptor with its default access and binding flags.")
        .def_readwrite("debug_name", &SharedBufferDesc::debugName,
                       "Debug label for GPU diagnostic tools.")
        .def_readwrite("element_stride_bytes", &SharedBufferDesc::elementStrideBytes,
                       "Stride per element in bytes.")
        .def_readwrite("element_count", &SharedBufferDesc::elementCount, "Initial element count.")
        .def_readwrite("minimum_capacity", &SharedBufferDesc::minimumCapacity,
                       "Minimum allocation capacity.")
        .def_readwrite("access", &SharedBufferDesc::access, "Access mode.")
        .def_readwrite("bind_flags", &SharedBufferDesc::bindFlags, "GPU binding usage flags.");

    py::class_<SharedBufferInfo>(m, "SharedBufferInfo",
                                 "Detailed information and runtime state for a shared GPU buffer.")
        .def(py::init<>(), "Initializes empty shared-buffer information.")
        .def_readwrite("handle", &SharedBufferInfo::handle, "Buffer handle.")
        .def_readwrite("debug_name", &SharedBufferInfo::debugName, "Debug label.")
        .def_readwrite("element_stride_bytes", &SharedBufferInfo::elementStrideBytes,
                       "Element stride in bytes.")
        .def_readwrite("element_count", &SharedBufferInfo::elementCount,
                       "Current element count.")
        .def_readwrite("capacity", &SharedBufferInfo::capacity, "Total element capacity.")
        .def_readwrite("size_bytes", &SharedBufferInfo::sizeBytes,
                       "Total allocated size in bytes.")
        .def_readwrite("access", &SharedBufferInfo::access, "Access mode.")
        .def_readwrite("bind_flags", &SharedBufferInfo::bindFlags, "Binding flags.")
        .def_readwrite("exportable", &SharedBufferInfo::exportable,
                       "True if buffer is exportable for CUDA/DLPack interop.")
        .def_readwrite("imported_into_cuda", &SharedBufferInfo::importedIntoCuda,
                       "True if buffer is currently imported into CUDA.");

    py::class_<SharedBufferCudaView>(m, "SharedBufferCudaView",
                                     "View descriptor exposing CUDA device memory pointer and size for interop.")
        .def(py::init<>(), "Initializes an invalid CUDA device-memory view.")
        .def_property_readonly("device_pointer", [](const SharedBufferCudaView &view)
                               { return reinterpret_cast<std::uintptr_t>(view.devicePointer); },
                               "Raw CUDA device pointer. Exposed to Python as an integer address.")
        .def_readwrite("size_bytes", &SharedBufferCudaView::sizeBytes,
                       "Size of accessible device memory in bytes.")
        .def_readwrite("device_ordinal", &SharedBufferCudaView::deviceOrdinal,
                       "CUDA device ordinal index.")
        .def("is_valid", &SharedBufferCudaView::isValid,
             "Checks if the CUDA device view is valid.");

    py::class_<SharedBufferTensorDesc>(
        m, "SharedBufferTensorDesc",
        "Tensor metadata descriptor for exporting a shared buffer to DLPack / PyTorch.")
        .def(py::init<>(), "Initializes metadata for a scalar float32 tensor at the buffer start.")
        .def_readwrite("shape", &SharedBufferTensorDesc::shape, "Tensor dimensions shape array.")
        .def_readwrite("strides", &SharedBufferTensorDesc::strides, "Tensor strides array.")
        .def_readwrite("dtype_code", &SharedBufferTensorDesc::dtypeCode,
                       "Element data type code.")
        .def_readwrite("dtype_bits", &SharedBufferTensorDesc::dtypeBits,
                       "Element bit width (e.g. 32 for float32).")
        .def_readwrite("dtype_lanes", &SharedBufferTensorDesc::dtypeLanes,
                       "Vector lanes count per element.")
        .def_readwrite("byte_offset", &SharedBufferTensorDesc::byteOffset,
                       "Byte offset from buffer start.");

    py::class_<GpuRenderTargetHandle>(m, "GpuRenderTargetHandle",
                                      "Opaque per-device handle for an offscreen render target.")
        .def(py::init<>(), "Initializes an invalid render-target handle.")
        .def_readwrite("id", &GpuRenderTargetHandle::id, "Opaque render-target identifier.");

    py::class_<GpuRenderTargetBinding>(m, "GpuRenderTargetBinding",
                                       "Selection of one or more array layers in an offscreen render target.")
        .def(py::init<>(), "Initializes a binding with an invalid target.")
        .def_readwrite("target", &GpuRenderTargetBinding::target, "Target to bind.")
        .def_readwrite("first_layer", &GpuRenderTargetBinding::firstLayer,
                       "First target array layer included in the binding.")
        .def_readwrite("layer_count", &GpuRenderTargetBinding::layerCount,
                       "Number of consecutive target array layers included in the binding.")
        .def("is_valid", &GpuRenderTargetBinding::isValid,
             "Returns whether the binding has a valid target identifier and a nonzero layer count.")
        .def(py::self == py::self, "Returns whether two bindings select the same target layers.");

    py::class_<RigidLayoutMapping>(
        m, "RigidLayoutMapping",
        "Prepared host-side slot layout for rigid bodies and colliders.")
        .def(py::init<>(), "Initializes an empty rigid-body and collider layout mapping.")
        .def_readwrite("rigid_body_count", &RigidLayoutMapping::rigidBodyCount,
                       "Number of rigid-body slots in the prepared layout.")
        .def_readwrite("collider_count", &RigidLayoutMapping::colliderCount,
                       "Number of collider slots in the prepared layout.")
        .def_readwrite(
            "layout_revision", &RigidLayoutMapping::layoutRevision,
            "Prepared rigid/collider slot-layout invalidation key produced by ``Runtime.prepare()``.")
        .def_readwrite("rigid_body_ids", &RigidLayoutMapping::rigidBodyIds,
                       "Physics rigid-body IDs, indexed by rigid-body slot.")
        .def_readwrite("rigid_body_entity_ids", &RigidLayoutMapping::rigidBodyEntityIds,
                       "Entity IDs owning the rigid bodies, indexed by rigid-body slot.")
        .def_readwrite("rigid_body_environment_indices",
                       &RigidLayoutMapping::rigidBodyEnvironmentIndices,
                       "Environment indices of rigid bodies, indexed by rigid-body slot.")
        .def_readwrite("collider_ids", &RigidLayoutMapping::colliderIds,
                       "Physics collider IDs, indexed by collider slot.")
        .def_readwrite("collider_entity_ids", &RigidLayoutMapping::colliderEntityIds,
                       "Entity IDs owning the colliders, indexed by collider slot.")
        .def_readwrite("collider_owner_body_ids", &RigidLayoutMapping::colliderOwnerBodyIds,
                       "Physics IDs of collider owner bodies, indexed by collider slot.")
        .def_readwrite("collider_owner_body_indices", &RigidLayoutMapping::colliderOwnerBodyIndices,
                       "Rigid-body slots owning colliders, indexed by collider slot.")
        .def_readwrite("collider_environment_indices",
                       &RigidLayoutMapping::colliderEnvironmentIndices,
                       "Environment indices of colliders, indexed by collider slot.")
        .def_readwrite("collider_shape_types", &RigidLayoutMapping::colliderShapeTypes,
                       "Encoded collider shape types, indexed by collider slot.")
        .def_readwrite("collider_enabled", &RigidLayoutMapping::colliderEnabledFlags,
                       "Collider enabled flags, indexed by collider slot.")
        .def_readwrite("collider_collision_layers", &RigidLayoutMapping::colliderCollisionLayers,
                       "Collider collision-layer bitmasks, indexed by collider slot.")
        .def_readwrite("collider_collision_masks", &RigidLayoutMapping::colliderCollisionMasks,
                       "Collider collision-filter bitmasks, indexed by collider slot.")
        .def_readwrite("collider_local_positions", &RigidLayoutMapping::colliderLocalPositions,
                       "Collider local positions, indexed by collider slot.")
        .def_readwrite("collider_local_rotations", &RigidLayoutMapping::colliderLocalRotations,
                       "Collider local rotations, indexed by collider slot.")
        .def_readwrite("collider_shape_params", &RigidLayoutMapping::colliderShapeParams,
                       "Collider shape parameters, indexed by collider slot.")
        .def_readwrite("body_collider_offsets", &RigidLayoutMapping::bodyColliderOffsets,
                       "Offsets into ``body_collider_indices``, indexed by rigid-body slot.")
        .def_readwrite("body_collider_counts", &RigidLayoutMapping::bodyColliderCounts,
                       "Collider counts per rigid-body slot.")
        .def_readwrite("body_collider_indices", &RigidLayoutMapping::bodyColliderIndices,
                       "Flattened collider slots, grouped by owning rigid-body slot.");

    py::class_<RigidParticleAttachmentConstraintLayoutMapping>(
        m, "RigidParticleAttachmentConstraintLayoutMapping",
        "Prepared host-side layout for rigid-particle attachment constraints.")
        .def(py::init<>(), "Initializes an empty rigid-particle attachment constraint layout.")
        .def_readwrite("count", &RigidParticleAttachmentConstraintLayoutMapping::count,
                       "Number of rigid-particle attachment constraint slots.")
        .def_readwrite("constraint_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::constraintIds,
                       "Constraint IDs for the attachment constraint slots.")
        .def_readwrite("environment_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::environmentIndices,
                       "Environment indices of the referenced rigid bodies.")
        .def_readwrite("rigid_body_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::rigidBodyIds,
                       "Rigid-body IDs referenced by the attachment constraints.")
        .def_readwrite("rigid_body_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::rigidBodyIndices,
                       "Prepared rigid-body slots referenced by the attachment constraints.")
        .def_readwrite("particle_entity_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleEntityIds,
                       "Entity IDs that own the referenced particles.")
        .def_readwrite("particle_reference_types",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleReferenceTypes,
                       "Numeric ``AuthoredParticleReferenceType`` values for the referenced particles.")
        .def_readwrite("particle_local_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleLocalIndices,
                       "Particle indices local to their owning entities.")
        .def_readwrite("enabled", &RigidParticleAttachmentConstraintLayoutMapping::enabledFlags,
                       "Whether each attachment constraint is enabled, as ``0`` or ``1``.");

    py::class_<RigidDistanceConstraintLayoutMapping>(
        m, "RigidDistanceConstraintLayoutMapping",
        "Prepared host-side layout for rigid distance constraints.")
        .def(py::init<>(), "Initializes an empty rigid distance constraint layout.")
        .def_readwrite("count", &RigidDistanceConstraintLayoutMapping::count,
                       "Number of rigid distance constraint slots.")
        .def_readwrite("constraint_ids", &RigidDistanceConstraintLayoutMapping::constraintIds,
                       "Constraint IDs for the rigid distance constraint slots.")
        .def_readwrite("environment_indices",
                       &RigidDistanceConstraintLayoutMapping::environmentIndices,
                       "Environment indices of endpoint A's referenced rigid bodies.")
        .def_readwrite("rigid_body_ids_a", &RigidDistanceConstraintLayoutMapping::rigidBodyIdsA,
                       "Rigid-body IDs for endpoint A.")
        .def_readwrite("rigid_body_ids_b", &RigidDistanceConstraintLayoutMapping::rigidBodyIdsB,
                       "Rigid-body IDs for endpoint B.")
        .def_readwrite("rigid_body_indices_a",
                       &RigidDistanceConstraintLayoutMapping::rigidBodyIndicesA,
                       "Prepared rigid-body slots for endpoint A.")
        .def_readwrite("rigid_body_indices_b",
                       &RigidDistanceConstraintLayoutMapping::rigidBodyIndicesB,
                       "Prepared rigid-body slots for endpoint B.")
        .def_readwrite("enabled", &RigidDistanceConstraintLayoutMapping::enabledFlags,
                       "Whether each rigid distance constraint is enabled, as ``0`` or ``1``.");

    py::class_<RoutedCableConstraintLayoutMapping>(
        m, "RoutedCableConstraintLayoutMapping",
        "Prepared host-side layout for routed cable constraints and their route points.")
        .def(py::init<>(), "Initializes an empty routed cable constraint layout.")
        .def_readwrite("count", &RoutedCableConstraintLayoutMapping::count,
                       "Number of routed cable constraint slots.")
        .def_readwrite("constraint_ids", &RoutedCableConstraintLayoutMapping::constraintIds,
                       "Constraint IDs for the routed cable constraint slots.")
        .def_readwrite("environment_indices",
                       &RoutedCableConstraintLayoutMapping::environmentIndices,
                       "Environment indices resolved from each cable's first valid route point.")
        .def_readwrite("route_point_offsets",
                       &RoutedCableConstraintLayoutMapping::routePointOffsets,
                       "Offsets into the flattened route-point arrays for each cable.")
        .def_readwrite("route_point_counts", &RoutedCableConstraintLayoutMapping::routePointCounts,
                       "Route-point counts for each cable.")
        .def_readwrite("enabled", &RoutedCableConstraintLayoutMapping::enabledFlags,
                       "Whether each routed cable constraint is enabled, as ``0`` or ``1``.")
        .def_readwrite("route_point_rigid_body_ids",
                       &RoutedCableConstraintLayoutMapping::routePointRigidBodyIds,
                       "Rigid-body IDs for flattened route points.")
        .def_readwrite("route_point_rigid_body_indices",
                       &RoutedCableConstraintLayoutMapping::routePointRigidBodyIndices,
                       "Prepared rigid-body slots for flattened route points.")
        .def_readwrite("route_point_local_guide_offsets",
                       &RoutedCableConstraintLayoutMapping::routePointLocalGuideOffsets,
                       "Local guide offsets for flattened route points.");

    py::class_<ConstraintLayoutMapping>(
        m, "ConstraintLayoutMapping",
        "Prepared host-side slot layout for rigid-adjacent constraints.")
        .def(py::init<>(), "Initializes an empty constraint layout mapping.")
        .def_readwrite("layout_revision", &ConstraintLayoutMapping::layoutRevision,
                       "Prepared constraint slot-layout invalidation key produced by ``Runtime.prepare()``.")
        .def_readwrite("rigid_particle_attachments",
                       &ConstraintLayoutMapping::rigidParticleAttachments,
                       "Prepared layout for rigid-particle attachment constraints.")
        .def_readwrite("rigid_distance_constraints",
                       &ConstraintLayoutMapping::rigidDistanceConstraints,
                       "Prepared layout for rigid distance constraints.")
        .def_readwrite("routed_cables", &ConstraintLayoutMapping::routedCables,
                       "Prepared layout for routed cable constraints.");

    py::class_<JointLayoutMapping>(
        m, "JointLayoutMapping", "Prepared host-side slot layout for rigid joints.")
        .def(py::init<>(), "Initializes an empty rigid-joint layout mapping.")
        .def_readwrite("ball_joint_count", &JointLayoutMapping::ballJointCount,
                       "Number of ball joint slots.")
        .def_readwrite("hinge_joint_count", &JointLayoutMapping::hingeJointCount,
                       "Number of hinge joint slots.")
        .def_readwrite("spherical_joint_count", &JointLayoutMapping::sphericalJointCount,
                       "Number of spherical joint slots.")
        .def_readwrite("slider_joint_count", &JointLayoutMapping::sliderJointCount,
                       "Number of slider joint slots.")
        .def_readwrite("layout_revision", &JointLayoutMapping::layoutRevision,
                       "Prepared rigid-joint slot-layout invalidation key produced by ``Runtime.prepare()``.")
        .def_readwrite("ball_joint_ids", &JointLayoutMapping::ballJointIds,
                       "Ball joint IDs for the ball joint slots.")
        .def_readwrite("ball_environment_indices", &JointLayoutMapping::ballEnvironmentIndices,
                       "Environment indices of ball joint endpoint A.")
        .def_readwrite("ball_body_ids_a", &JointLayoutMapping::ballBodyIdsA,
                       "Rigid-body IDs for ball joint endpoint A.")
        .def_readwrite("ball_body_ids_b", &JointLayoutMapping::ballBodyIdsB,
                       "Rigid-body IDs for ball joint endpoint B.")
        .def_readwrite("ball_body_indices_a", &JointLayoutMapping::ballBodyIndicesA,
                       "Prepared rigid-body slots for ball joint endpoint A.")
        .def_readwrite("ball_body_indices_b", &JointLayoutMapping::ballBodyIndicesB,
                       "Prepared rigid-body slots for ball joint endpoint B.")
        .def_readwrite("hinge_joint_ids", &JointLayoutMapping::hingeJointIds,
                       "Hinge joint IDs for the hinge joint slots.")
        .def_readwrite("hinge_environment_indices", &JointLayoutMapping::hingeEnvironmentIndices,
                       "Environment indices of hinge joint endpoint A.")
        .def_readwrite("hinge_body_ids_a", &JointLayoutMapping::hingeBodyIdsA,
                       "Rigid-body IDs for hinge joint endpoint A.")
        .def_readwrite("hinge_body_ids_b", &JointLayoutMapping::hingeBodyIdsB,
                       "Rigid-body IDs for hinge joint endpoint B.")
        .def_readwrite("hinge_body_indices_a", &JointLayoutMapping::hingeBodyIndicesA,
                       "Prepared rigid-body slots for hinge joint endpoint A.")
        .def_readwrite("hinge_body_indices_b", &JointLayoutMapping::hingeBodyIndicesB,
                       "Prepared rigid-body slots for hinge joint endpoint B.")
        .def_readwrite("spherical_joint_ids", &JointLayoutMapping::sphericalJointIds,
                       "Spherical joint IDs for the spherical joint slots.")
        .def_readwrite("spherical_environment_indices",
                       &JointLayoutMapping::sphericalEnvironmentIndices,
                       "Environment indices of spherical joint endpoint A.")
        .def_readwrite("spherical_body_ids_a", &JointLayoutMapping::sphericalBodyIdsA,
                       "Rigid-body IDs for spherical joint endpoint A.")
        .def_readwrite("spherical_body_ids_b", &JointLayoutMapping::sphericalBodyIdsB,
                       "Rigid-body IDs for spherical joint endpoint B.")
        .def_readwrite("spherical_body_indices_a", &JointLayoutMapping::sphericalBodyIndicesA,
                       "Prepared rigid-body slots for spherical joint endpoint A.")
        .def_readwrite("spherical_body_indices_b", &JointLayoutMapping::sphericalBodyIndicesB,
                       "Prepared rigid-body slots for spherical joint endpoint B.")
        .def_readwrite("slider_joint_ids", &JointLayoutMapping::sliderJointIds,
                       "Slider joint IDs for the slider joint slots.")
        .def_readwrite("slider_environment_indices", &JointLayoutMapping::sliderEnvironmentIndices,
                       "Environment indices of slider joint endpoint A.")
        .def_readwrite("slider_body_ids_a", &JointLayoutMapping::sliderBodyIdsA,
                       "Rigid-body IDs for slider joint endpoint A.")
        .def_readwrite("slider_body_ids_b", &JointLayoutMapping::sliderBodyIdsB,
                       "Rigid-body IDs for slider joint endpoint B.")
        .def_readwrite("slider_body_indices_a", &JointLayoutMapping::sliderBodyIndicesA,
                       "Prepared rigid-body slots for slider joint endpoint A.")
        .def_readwrite("slider_body_indices_b", &JointLayoutMapping::sliderBodyIndicesB,
                       "Prepared rigid-body slots for slider joint endpoint B.");

    py::class_<ParticleLayoutMapping>(
        m, "ParticleLayoutMapping",
        "Prepared host-side slot layout for particles and deformable entities.")
        .def(py::init<>(), "Initializes an empty particle layout mapping.")
        .def_readwrite("particle_count", &ParticleLayoutMapping::particleCount,
                       "Number of prepared particle slots.")
        .def_readwrite("soft_body_count", &ParticleLayoutMapping::softBodyCount,
                       "Number of prepared soft-body slots.")
        .def_readwrite("fluid_count", &ParticleLayoutMapping::fluidCount,
                       "Number of prepared fluid slots.")
        .def_readwrite("strand_count", &ParticleLayoutMapping::strandCount,
                       "Number of prepared strand slots.")
        .def_readwrite("layout_revision", &ParticleLayoutMapping::layoutRevision,
                       "Prepared particle/deformable slot-layout invalidation key produced by ``Runtime.prepare()``.")
        .def_readwrite("environment_indices", &ParticleLayoutMapping::environmentIndices,
                       "Environment indices by prepared particle slot.")
        .def_readwrite("particle_kinds", &ParticleLayoutMapping::particleKinds,
                       "Numeric ``ParticleKind`` values by prepared particle slot.")
        .def_readwrite("owner_types", &ParticleLayoutMapping::ownerTypes,
                       "Numeric ``ParticleOwnerType`` values by prepared particle slot.")
        .def_readwrite("owner_indices", &ParticleLayoutMapping::ownerIndices,
                       "Owner slots, interpreted according to ``owner_types``.")
        .def_readwrite("strand_ids", &ParticleLayoutMapping::strandIds,
                       "Strand or suturing group IDs by prepared particle slot.")
        .def_readwrite("strand_orders", &ParticleLayoutMapping::strandOrders,
                       "Particle order within its strand or suturing group.")
        .def_readwrite("strand_roles", &ParticleLayoutMapping::strandRoles,
                       "Numeric ``ParticleStrandRole`` values by prepared particle slot.")
        .def_readwrite("owning_soft_body_indices", &ParticleLayoutMapping::owningSoftBodyIndices,
                       "Owning soft-body slots by prepared particle slot.")
        .def_readwrite("particle_material_indices", &ParticleLayoutMapping::particleMaterialIndices,
                       "Contact material indices by prepared particle slot.")
        .def_readwrite("fluid_material_indices", &ParticleLayoutMapping::fluidMaterialIndices,
                       "Fluid material indices by prepared particle slot.")
        .def_readwrite("phases", &ParticleLayoutMapping::phases,
                       "Packed particle phase values by prepared particle slot.")
        .def_readwrite("collision_layers", &ParticleLayoutMapping::collisionLayers,
                       "Collision layers by prepared particle slot.")
        .def_readwrite("collision_masks", &ParticleLayoutMapping::collisionMasks,
                       "Collision masks by prepared particle slot.")
        .def_readwrite("adjacency_offsets", &ParticleLayoutMapping::adjacencyOffsets,
                       "Offsets into the internal flattened particle adjacency list.")
        .def_readwrite("adjacency_counts", &ParticleLayoutMapping::adjacencyCounts,
                       "Adjacent-particle counts by prepared particle slot.")
        .def_readwrite("soft_body_entity_ids", &ParticleLayoutMapping::softBodyEntityIds,
                       "Entity IDs for the prepared soft-body slots.")
        .def_readwrite("soft_body_environment_indices",
                       &ParticleLayoutMapping::softBodyEnvironmentIndices,
                       "Environment indices for the prepared soft-body slots.")
        .def_readwrite("soft_body_particle_offsets",
                       &ParticleLayoutMapping::softBodyParticleOffsets,
                       "First prepared particle slot for each soft body.")
        .def_readwrite("soft_body_particle_counts", &ParticleLayoutMapping::softBodyParticleCounts,
                       "Particle counts for the prepared soft-body slots.")
        .def_readwrite("fluid_entity_ids", &ParticleLayoutMapping::fluidEntityIds,
                       "Entity IDs for the prepared fluid slots.")
        .def_readwrite("fluid_environment_indices", &ParticleLayoutMapping::fluidEnvironmentIndices,
                       "Environment indices for the prepared fluid slots.")
        .def_readwrite("fluid_particle_offsets", &ParticleLayoutMapping::fluidParticleOffsets,
                       "First prepared particle slot for each fluid.")
        .def_readwrite("fluid_particle_counts", &ParticleLayoutMapping::fluidParticleCounts,
                       "Particle counts for the prepared fluid slots.")
        .def_readwrite("strand_entity_ids", &ParticleLayoutMapping::strandEntityIds,
                       "Entity IDs for the prepared strand slots.")
        .def_readwrite("strand_environment_indices",
                       &ParticleLayoutMapping::strandEnvironmentIndices,
                       "Environment indices for the prepared strand slots.")
        .def_readwrite("strand_particle_offsets", &ParticleLayoutMapping::strandParticleOffsets,
                       "First prepared particle slot for each strand.")
        .def_readwrite("strand_particle_counts", &ParticleLayoutMapping::strandParticleCounts,
                       "Particle counts for the prepared strand slots.");

    py::class_<CustomComputeResourceDesc>(
        m, "CustomComputeResourceDesc", "Metadata descriptor for a custom compute GPU resource.")
        .def(py::init<>(), "Initializes a custom compute GPU resource descriptor.")
        .def_readwrite("key", &CustomComputeResourceDesc::key,
                       "Unique resource string identifier key.")
        .def_readwrite("kind", &CustomComputeResourceDesc::kind,
                       "Resource type kind (Buffer, Texture).")
        .def_readwrite("access", &CustomComputeResourceDesc::access, "Access mode.")
        .def_readwrite("element_count", &CustomComputeResourceDesc::elementCount,
                       "Number of elements in resource.")
        .def_readwrite("element_stride_bytes", &CustomComputeResourceDesc::elementStrideBytes,
                       "Stride per element in bytes.")
        .def_readwrite("binding_generation", &CustomComputeResourceDesc::bindingGeneration,
                       "Resource binding invalidation generation key.");

    py::class_<CustomComputeResourceBindingDesc>(
        m, "CustomComputeResourceBindingDesc",
        "Binding descriptor mapping a GPU resource to a compute shader variable name.")
        .def(py::init<>(), "Initializes a custom compute resource binding descriptor.")
        .def_readwrite("shader_variable_name",
                       &CustomComputeResourceBindingDesc::shaderVariableName,
                       "HLSL shader variable name.")
        .def_readwrite("resource_key", &CustomComputeResourceBindingDesc::resourceKey,
                       "Registered engine resource key.")
        .def_readwrite("shared_buffer_handle",
                       &CustomComputeResourceBindingDesc::sharedBufferHandle,
                       "Optional shared engine GPU buffer handle.")
        .def_readwrite("render_target_binding",
                       &CustomComputeResourceBindingDesc::renderTargetBinding,
                       "Optional render target binding.")
        .def_readwrite("render_target_texture_plane",
                       &CustomComputeResourceBindingDesc::renderTargetTexturePlane,
                       "Texture plane selection (Color, Depth).")
        .def_readwrite("access", &CustomComputeResourceBindingDesc::access,
                       "Resource access mode.");

    py::class_<CustomComputeDispatchDesc>(
        m, "CustomComputeDispatchDesc", "Dispatch execution parameters for custom compute passes.")
        .def(py::init<>(), "Initializes custom compute dispatch parameters.")
        .def_readwrite("mode", &CustomComputeDispatchDesc::mode, "Dispatch mode.")
        .def_readwrite("group_count_x", &CustomComputeDispatchDesc::groupCountX,
                       "Dispatch thread group count along X dimension.")
        .def_readwrite("group_count_y", &CustomComputeDispatchDesc::groupCountY,
                       "Dispatch thread group count along Y dimension.")
        .def_readwrite("group_count_z", &CustomComputeDispatchDesc::groupCountZ,
                       "Dispatch thread group count along Z dimension.")
        .def_readwrite("count_resource_key", &CustomComputeDispatchDesc::countResourceKey,
                       "Resource key used for element count dispatch mode.");

    py::class_<CustomComputePassDesc>(
        m, "CustomComputePassDesc",
        "Complete descriptor for compiling and instantiating a custom compute shader pass.")
        .def(py::init<>(), "Initializes a custom compute shader pass descriptor.")
        .def_readwrite("debug_name", &CustomComputePassDesc::debugName,
                       "Debug label for diagnostics.")
        // Expose filesystem paths as Python strings.  Binding std::filesystem::path
        // directly leaks the platform's C++ implementation type into pybind11
        // docstrings (for example, std::filesystem::__cxx11::path on libstdc++),
        // which cannot be represented in generated Python stubs.
        .def_property(
            "shader_directory",
            [](const CustomComputePassDesc &desc) { return desc.shaderDirectory.string(); },
            [](CustomComputePassDesc &desc, const std::string &value)
            { desc.shaderDirectory = value; },
            "Root directory for shader source files.")
        .def_readwrite("shader_path", &CustomComputePassDesc::shaderPath,
                       "Shader source path relative to shaderDirectory.")
        .def_readwrite("shader_source", &CustomComputePassDesc::shaderSource,
                       "Optional raw HLSL shader source code string.")
        .def_property(
            "include_directories",
            [](const CustomComputePassDesc &desc)
            {
                std::vector<std::string> directories;
                directories.reserve(desc.includeDirectories.size());
                for (const auto &directory : desc.includeDirectories)
                {
                    directories.push_back(directory.string());
                }
                return directories;
            },
            [](CustomComputePassDesc &desc, const std::vector<std::string> &values)
            {
                desc.includeDirectories.clear();
                desc.includeDirectories.reserve(values.size());
                for (const auto &value : values)
                {
                    desc.includeDirectories.emplace_back(value);
                }
            },
            "Custom shader search paths for HLSL includes.")
        .def_readwrite("entry_point", &CustomComputePassDesc::entryPoint,
                       "Shader entry point function name.")
        .def_readwrite("thread_group_size_x", &CustomComputePassDesc::threadGroupSizeX,
                       "Workgroup size X dimension.")
        .def_readwrite("thread_group_size_y", &CustomComputePassDesc::threadGroupSizeY,
                       "Workgroup size Y dimension.")
        .def_readwrite("thread_group_size_z", &CustomComputePassDesc::threadGroupSizeZ,
                       "Workgroup size Z dimension.")
        .def_readwrite("resource_bindings", &CustomComputePassDesc::resourceBindings,
                       "Resource bindings array.")
        .def_readwrite("constant_buffer_variable_name",
                       &CustomComputePassDesc::constantBufferVariableName,
                       "Constant buffer HLSL variable name.")
        .def_readwrite("constant_buffer_size_bytes",
                       &CustomComputePassDesc::constantBufferSizeBytes,
                       "Constant buffer size in bytes.")
        .def_readwrite("constant_data", &CustomComputePassDesc::constantData,
                       "Initial constant buffer binary data payload.")
        .def_readwrite("dispatch", &CustomComputePassDesc::dispatch, "Dispatch parameters.");

    py::enum_<Diligent::TEXTURE_FORMAT>(m, "TextureFormat",
                                         "Texture formats supported by the Python binding.")
        .value("Unknown", Diligent::TEX_FORMAT_UNKNOWN,
               "Unspecified format; uses an API-specific default where supported.")
        .value("RGBA8Unorm", Diligent::TEX_FORMAT_RGBA8_UNORM,
               "Four-channel 8-bit unsigned normalized RGBA format.")
        .value("RGBA8UnormSrgb", Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB,
               "Four-channel 8-bit unsigned normalized sRGB RGBA format.")
        .value("BGRA8Unorm", Diligent::TEX_FORMAT_BGRA8_UNORM,
               "Four-channel 8-bit unsigned normalized BGRA format.")
        .value("BGRA8UnormSrgb", Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB,
               "Four-channel 8-bit unsigned normalized sRGB BGRA format.")
        .value("RGBA16Float", Diligent::TEX_FORMAT_RGBA16_FLOAT,
               "Four-channel 16-bit floating-point RGBA format.")
        .value("R32Float", Diligent::TEX_FORMAT_R32_FLOAT,
               "Single-channel 32-bit floating-point format.")
        .value("R32Uint", Diligent::TEX_FORMAT_R32_UINT,
               "Single-channel 32-bit unsigned integer format.")
        .value("D32Float", Diligent::TEX_FORMAT_D32_FLOAT,
               "32-bit floating-point depth format.");

    py::enum_<CameraComponent::Product>(m, "CameraProduct",
                                        "Output product modes rendered by the camera.")
        .value("ColorDepth", CameraComponent::Product::ColorDepth,
               "Standard RGBA color and depth output.")
        .value("Depth", CameraComponent::Product::Depth, "Single-channel depth map output.")
        .value("SegmentationDepth", CameraComponent::Product::SegmentationDepth,
               "Semantic segmentation mask and depth output.");

    py::enum_<UltrasoundProbeComponent::Geometry>(
        m, "UltrasoundProbeGeometry", "Ultrasound transducer probe array geometry type.")
        .value("Linear", UltrasoundProbeComponent::Geometry::Linear, "Linear array transducer.")
        .value("Curvilinear", UltrasoundProbeComponent::Geometry::Curvilinear,
               "Convex / curvilinear array transducer.");

    py::enum_<CameraComponent::BackgroundMode>(m, "CameraBackgroundMode",
                                               "Background clear modes for camera rendering.")
        .value("ClearColor", CameraComponent::BackgroundMode::ClearColor,
               "Solid clear color background.")
        .value("EnvironmentCubemap", CameraComponent::BackgroundMode::EnvironmentCubemap,
               "Skybox / Image-Based Lighting cubemap background.");

    py::enum_<cressim::neo::physics::RigidBodyType>(m, "RigidBodyType",
                                                     "Rigid body simulation types.")
        .value("Static", cressim::neo::physics::RigidBodyType::Static,
               "Static rigid body.")
        .value("Kinematic", cressim::neo::physics::RigidBodyType::Kinematic,
               "Kinematic rigid body.")
        .value("Dynamic", cressim::neo::physics::RigidBodyType::Dynamic,
               "Dynamically simulated rigid body.");

    py::enum_<cressim::neo::physics::ColliderShapeType>(m, "ColliderShapeType",
                                                         "Collider primitive geometry types.")
        .value("Sphere", cressim::neo::physics::ColliderShapeType::Sphere,
               "Spherical collider primitive.")
        .value("Box", cressim::neo::physics::ColliderShapeType::Box,
               "Box collider primitive.")
        .value("Capsule", cressim::neo::physics::ColliderShapeType::Capsule,
               "Capsule collider primitive.");

    py::enum_<AuthoredParticleReferenceType>(m, "AuthoredParticleReferenceType")
        .value("SoftBodyParticle", AuthoredParticleReferenceType::SoftBodyParticle)
        .value("StrandParticle", AuthoredParticleReferenceType::StrandParticle)
        .value("RigidProxyParticle", AuthoredParticleReferenceType::RigidProxyParticle);

    py::enum_<SoftBodySourceKind>(m, "SoftBodySourceKind",
                                  "Selects which member of a soft-body source description is used.")
        .value("RegularGrid", SoftBodySourceKind::RegularGrid,
               "Generate a tetrahedral soft body from a regular grid.")
        .value("TetMesh", SoftBodySourceKind::TetMesh,
               "Use supplied rest positions and tetrahedron indices.")
        .value("TetGenFiles", SoftBodySourceKind::TetGenFiles,
               "Load rest positions and tetrahedra from TetGen node and element files.")
        .value("MeshfreeParticles", SoftBodySourceKind::MeshfreeParticles,
               "Use supplied particles and construct a k-nearest-neighbour graph.");

    py::enum_<FluidSourceKind>(m, "FluidSourceKind",
                               "Selects which member of a fluid source description is used.")
        .value("RegularGrid", FluidSourceKind::RegularGrid,
               "Generate fluid particles from a regular grid.");

    py::enum_<ParticleKind>(m, "ParticleKind")
        .value("SoftSolid", ParticleKind::SoftSolid)
        .value("Fluid", ParticleKind::Fluid);

    py::enum_<ParticleOwnerType>(m, "ParticleOwnerType")
        .value("None", ParticleOwnerType::None)
        .value("SoftBody", ParticleOwnerType::SoftBody)
        .value("FluidBody", ParticleOwnerType::FluidBody)
        .value("Strand", ParticleOwnerType::Strand)
        .value("RigidBody", ParticleOwnerType::RigidBody);

    py::enum_<ParticleStrandRole>(m, "ParticleStrandRole",
                                  "Suturing roles assigned to particles.")
        .value("None", ParticleStrandRole::None, "No suturing role.")
        .value("NeedleTip", ParticleStrandRole::NeedleTip,
               "Tip particle of a suturing sequence.")
        .value("NeedleBody", ParticleStrandRole::NeedleBody,
               "Non-tip needle particle of a suturing sequence.")
        .value("Thread", ParticleStrandRole::Thread,
               "Thread particle of a suturing sequence.");

    py::enum_<RigidJointDriveMode>(m, "RigidJointDriveMode",
                                   "Rigid-joint drive control modes; availability depends on joint type.")
        .value("None", RigidJointDriveMode::None, "Disable joint drive control.")
        .value("TargetPosition", RigidJointDriveMode::TargetPosition,
               "Position-drive mode for hinge and slider joints.")
        .value("TargetVelocity", RigidJointDriveMode::TargetVelocity,
               "Velocity-drive mode for hinge and slider joints.")
        .value("TargetOrientation", RigidJointDriveMode::TargetOrientation,
               "Orientation-drive mode for spherical joints.");

    py::enum_<cressim::neo::gpu::VulkanShaderCompilerMode>(
        m, "VulkanShaderCompilerMode", "Shader compiler selection policy for the Vulkan backend.")
        .value("Auto", cressim::neo::gpu::VulkanShaderCompilerMode::Auto,
               "Select DXC when available; otherwise use the default compiler.")
        .value("ForceDefault", cressim::neo::gpu::VulkanShaderCompilerMode::ForceDefault,
               "Always use the default compiler.")
        .value("ForceDXC", cressim::neo::gpu::VulkanShaderCompilerMode::ForceDXC,
               "Require the DirectX Shader Compiler (DXC).");

    py::enum_<MaterialProgramFamily>(m, "MaterialProgramFamily",
                                     "Selects the material shader program family.")
        .value("StandardLit", MaterialProgramFamily::StandardLit,
               "Material program for standard mesh geometry.")
        .value("SoftBodyLit", MaterialProgramFamily::SoftBodyLit,
               "Material program for soft-body render geometry.")
        .value("CurveLit", MaterialProgramFamily::CurveLit,
               "Material program for curve render geometry.");

    py::enum_<cressim::neo::graphics::MaterialFeatureFlags>(m, "MaterialFeatureFlags",
                                                            py::arithmetic(),
                                                            "Bit flags enabling material shader features.")
        .value("None", cressim::neo::graphics::MaterialFeatureFlags::None,
               "Enable no optional features.")
        .value("AlphaTest", cressim::neo::graphics::MaterialFeatureFlags::AlphaTest,
               "Enable alpha testing.")
        .value("NormalMap", cressim::neo::graphics::MaterialFeatureFlags::NormalMap,
               "Enable normal-map shading.")
        .value("ClearCoat", cressim::neo::graphics::MaterialFeatureFlags::ClearCoat,
               "Enable clear-coat shading.")
        .value("DoubleSided", cressim::neo::graphics::MaterialFeatureFlags::DoubleSided,
               "Disable back-face culling.");

    py::enum_<MaterialRenderMode>(m, "MaterialRenderMode",
                                  "Selects the render pass and alpha behavior of a material.")
        .value("Opaque", MaterialRenderMode::Opaque, "Render in the opaque pass.")
        .value("Cutout", MaterialRenderMode::Cutout,
               "Render with alpha testing enabled.")
        .value("Transparent", MaterialRenderMode::Transparent,
               "Render in the transparent pass.");

    py::enum_<TextureColorSpace>(m, "TextureColorSpace", "Specifies a texture color space.")
        .value("Linear", TextureColorSpace::Linear, "Use linear color space.")
        .value("Srgb", TextureColorSpace::Srgb, "Use sRGB color space for RGBA8 textures.");

    py::enum_<TexturePixelFormat>(m, "TexturePixelFormat", "Specifies a texture pixel format.")
        .value("RGBA8", TexturePixelFormat::RGBA8, "Four 8-bit RGBA channels.")
        .value("RGBA16F", TexturePixelFormat::RGBA16F, "Four 16-bit floating-point RGBA channels.");

    py::enum_<TextureMipPolicy>(m, "TextureMipPolicy", "Specifies a stored texture mipmap policy.")
        .value("Disabled", TextureMipPolicy::Disabled, "Mipmap generation is disabled.")
        .value("Generate", TextureMipPolicy::Generate,
               "Mipmap generation is requested; current resource registration does not generate mipmaps.");

    py::enum_<TextureDimension>(m, "TextureDimension", "Specifies a texture dimension.")
        .value("Texture2D", TextureDimension::Texture2D, "A two-dimensional texture with one layer.")
        .value("TextureCube", TextureDimension::TextureCube, "A cubemap texture with six layers.");

    py::enum_<IblQualityTier>(m, "IblQualityTier",
                              "Selects the image-based-lighting quality tier.")
        .value("Off", IblQualityTier::Off, "Disable image-based lighting.")
        .value("DiffuseOnly", IblQualityTier::DiffuseOnly,
               "Use diffuse image-based lighting only.")
        .value("Full", IblQualityTier::Full,
               "Use diffuse and prefiltered specular image-based lighting.");

    py::class_<FrameContext>(m, "FrameContext",
                             "Timing and index values associated with an execution frame.")
        .def(py::init<>(), "Initializes a frame context with zero-valued fields.")
        .def_readwrite("frame_index", &FrameContext::frameIndex,
                       "Index associated with the frame being executed.")
        .def_readwrite("time_seconds", &FrameContext::timeSeconds,
                       "Time associated with the frame, in seconds.")
        .def_readwrite("delta_seconds", &FrameContext::deltaSeconds,
                       "Duration of the frame, in seconds.");

    py::class_<Transform>(m, "Transform",
                          "Position, orientation, and per-axis scale of an object in world space.")
        .def(py::init<>(), "Initializes an identity transform.")
        .def_readwrite("position", &Transform::position, "World-space translation.")
        .def_readwrite("rotation", &Transform::rotation, "World-space orientation quaternion.")
        .def_readwrite("scale", &Transform::scale, "Per-axis scale.");

    py::class_<TransformComponent>(
        m, "TransformComponent",
        "World transform component for spatial positioning, orientation, and scaling.")
        .def(py::init<>(), "Initializes the component with an identity transform.")
        .def_readwrite("world_transform", &TransformComponent::worldTransform,
                       "3D world transform matrix/pose.");

    py::class_<RenderOutputBinding>(m, "RenderOutputBinding",
                                    "Destination selection for rendered output.")
        .def(py::init<>(), "Initializes output for the runtime-managed primary target.")
        .def_readwrite("mode", &RenderOutputBinding::mode, "Output destination selection mode.")
        .def_readwrite("binding", &RenderOutputBinding::binding,
                       "Explicit target binding used when ``mode`` is ``ExplicitSurface``.");

    py::class_<GpuRenderTargetDesc>(m, "GpuRenderTargetDesc",
                                    "Descriptor for creating an offscreen GPU render target.")
        .def(py::init<>(), "Initializes the descriptor with its default attachment settings.")
        .def_readwrite("width", &GpuRenderTargetDesc::width,
                       "Target width in pixels; zero selects the current presentation width or 1280.")
        .def_readwrite("height", &GpuRenderTargetDesc::height,
                       "Target height in pixels; zero selects the current presentation height or 720.")
        .def_readwrite("array_size", &GpuRenderTargetDesc::arraySize,
                       "Number of array layers in each attachment.")
        .def_readwrite("color", &GpuRenderTargetDesc::color, "Whether to create a color attachment.")
        .def_readwrite("depth", &GpuRenderTargetDesc::depth, "Whether to create a depth attachment.")
        .def_readwrite("color_format", &GpuRenderTargetDesc::colorFormat,
                       "Color attachment format; ``TextureFormat.Unknown`` selects an automatic format.")
        .def_readwrite("depth_format", &GpuRenderTargetDesc::depthFormat,
                       "Depth attachment format; ``TextureFormat.Unknown`` selects an automatic format.")
        .def_readwrite("layered_rendering", &GpuRenderTargetDesc::layeredRendering,
                       "Whether attachment textures support layered rendering.")
        .def_readwrite("shader_readable", &GpuRenderTargetDesc::shaderReadable,
                       "Whether later shader passes may sample the attachments.")
        .def_readwrite("unordered_access", &GpuRenderTargetDesc::unorderedAccess,
                       "Whether the color attachment supports unordered-access views for compute passes.")
        .def_readwrite("debug_name", &GpuRenderTargetDesc::debugName,
                       "Diagnostic label for the render target.");

    py::class_<GpuRenderTargetReadbackRequest>(m, "GpuRenderTargetReadbackRequest",
                                               "Opaque per-device handle for a queued render-target readback.")
        .def(py::init<>(), "Initializes an invalid readback request.")
        .def_readwrite("id", &GpuRenderTargetReadbackRequest::id, "Opaque request identifier.");

    py::class_<GpuRenderTargetReadbackEvent>(m, "GpuRenderTargetReadbackEvent",
                                             "Completed CPU-side readback of an offscreen render target.")
        .def(py::init<>(), "Initializes an empty readback event.")
        .def_readwrite("binding", &GpuRenderTargetReadbackEvent::binding,
                       "Render-target layers read back by this event.")
        .def_readwrite("frame_index", &GpuRenderTargetReadbackEvent::frameIndex,
                       "Index of the frame that produced the readback.")
        .def_readwrite("color_format", &GpuRenderTargetReadbackEvent::colorFormat,
                       "Format of the color payload, when present.")
        .def_readwrite("width", &GpuRenderTargetReadbackEvent::width,
                       "Legacy alias for ``color_width``.")
        .def_readwrite("height", &GpuRenderTargetReadbackEvent::height,
                       "Legacy alias for ``color_height``.")
        .def_readwrite("row_stride_bytes", &GpuRenderTargetReadbackEvent::rowStrideBytes,
                       "Legacy alias for ``color_row_stride_bytes``.")
        .def_readwrite("color_width", &GpuRenderTargetReadbackEvent::colorWidth,
                       "Width of the color payload in pixels.")
        .def_readwrite("color_height", &GpuRenderTargetReadbackEvent::colorHeight,
                       "Height of the color payload in pixels.")
        .def_readwrite("color_row_stride_bytes", &GpuRenderTargetReadbackEvent::colorRowStrideBytes,
                       "Packed row size of the color payload in bytes.")
        .def_readwrite("color_bytes", &GpuRenderTargetReadbackEvent::colorBytes,
                       "Packed color pixel data in row-major order.")
        .def_readwrite("depth_format", &GpuRenderTargetReadbackEvent::depthFormat,
                       "Format of the depth payload, when present.")
        .def_readwrite("depth_width", &GpuRenderTargetReadbackEvent::depthWidth,
                       "Width of the depth payload in pixels.")
        .def_readwrite("depth_height", &GpuRenderTargetReadbackEvent::depthHeight,
                       "Height of the depth payload in pixels.")
        .def_readwrite("depth_row_stride_bytes", &GpuRenderTargetReadbackEvent::depthRowStrideBytes,
                       "Packed row size of the depth payload in bytes.")
        .def_readwrite("depth_bytes", &GpuRenderTargetReadbackEvent::depthBytes,
                       "Packed depth pixel data in row-major order.");

    py::class_<cressim::neo::common::SceneLayoutDesc>(
        m, "SceneLayoutDesc",
        "Capacity configuration for the environments, renderable objects, lights, and cameras in a scene.")
        .def(py::init<>(), "Initializes the descriptor with its default capacities.")
        .def_readwrite("env_count", &cressim::neo::common::SceneLayoutDesc::envCount,
                       "Number of simulation and rendering environments.")
        .def_readwrite("max_renderable_objects_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxRenderableObjectsPerEnv,
                       "Maximum number of renderable objects in each environment.")
        .def_readwrite("max_lights_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxLightsPerEnv,
                       "Maximum number of lights in each environment.")
        .def_readwrite("max_cameras_per_env",
                       &cressim::neo::common::SceneLayoutDesc::maxCamerasPerEnv,
                       "Maximum number of cameras in each environment.")
        .def("total_renderable_object_capacity",
             &cressim::neo::common::SceneLayoutDesc::totalRenderableObjectCapacity,
             "Returns the total renderable-object capacity across all environments.")
        .def("total_light_capacity", &cressim::neo::common::SceneLayoutDesc::totalLightCapacity,
             "Returns the total light capacity across all environments.")
        .def("total_camera_capacity", &cressim::neo::common::SceneLayoutDesc::totalCameraCapacity,
             "Returns the total camera capacity across all environments.");

    py::class_<cressim::neo::gpu::GpuRenderViewport>(m, "GpuRenderViewport",
                                                      "Normalized viewport rectangle relative to a bound render target.")
        .def(py::init<>(), "Initializes a viewport covering the complete render target.")
        .def_readwrite("x", &cressim::neo::gpu::GpuRenderViewport::x,
                       "Normalized left coordinate in the range [0, 1].")
        .def_readwrite("y", &cressim::neo::gpu::GpuRenderViewport::y,
                       "Normalized top coordinate in the range [0, 1].")
        .def_readwrite("width", &cressim::neo::gpu::GpuRenderViewport::width,
                       "Normalized viewport width in the range [0, 1].")
        .def_readwrite("height", &cressim::neo::gpu::GpuRenderViewport::height,
                       "Normalized viewport height in the range [0, 1].");

    py::class_<GpuDeviceDesc::PresentationDesc>(
        m, "GpuPresentationDesc", "Platform presentation and swap-chain configuration.")
        .def(py::init<>(), "Initializes presentation as disabled.")
        .def_readwrite("enabled", &GpuDeviceDesc::PresentationDesc::enabled,
                       "Whether to create a presentation swap chain.")
        .def_readwrite("sync_interval", &GpuDeviceDesc::PresentationDesc::syncInterval,
                       "Swap-chain presentation interval; one enables v-sync and zero disables it.")
        .def_readwrite("preferred_color_format",
                       &GpuDeviceDesc::PresentationDesc::preferredColorFormat,
                       "Preferred swap-chain color format; ``TextureFormat.Unknown`` lets the backend choose.")
        .def_property(
            "native_window", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeWindow); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeWindow = reinterpret_cast<void *>(value); },
            "Platform-native window pointer represented as an integer (Win32 HWND or macOS NSView).")
        .def_readwrite("native_window_id", &GpuDeviceDesc::PresentationDesc::nativeWindowId,
                       "Platform-native window ID (X11 Window or XCB window handle on Linux).")
        .def_property(
            "native_display", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeDisplay); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeDisplay = reinterpret_cast<void *>(value); },
            "X11 display pointer represented as an integer.")
        .def_property(
            "native_connection", [](const GpuDeviceDesc::PresentationDesc &desc)
            { return reinterpret_cast<std::uintptr_t>(desc.nativeConnection); },
            [](GpuDeviceDesc::PresentationDesc &desc, const std::uintptr_t value)
            { desc.nativeConnection = reinterpret_cast<void *>(value); },
            "XCB connection pointer represented as an integer.");

    py::class_<GpuDeviceDesc>(m, "GpuDeviceDesc",
                              "Configuration for initializing the GPU device and its shader source search paths.")
        .def(py::init<>(), "Initializes the descriptor with a Vulkan backend preference and validation enabled.")
        .def_readwrite("preferred_backend", &GpuDeviceDesc::preferredBackend,
                       "Preferred GPU execution backend.")
        .def_readwrite("enable_validation", &GpuDeviceDesc::enableValidation,
                       "Whether to enable graphics-backend validation.")
        .def_readwrite("default_render_target_desc", &GpuDeviceDesc::defaultRenderTargetDesc,
                       "Descriptor used for the runtime's default render target.")
        .def_readwrite("presentation", &GpuDeviceDesc::presentation,
                       "Presentation and swap-chain configuration.")
        .def_readwrite("vulkan_shader_compiler_mode", &GpuDeviceDesc::vulkanShaderCompilerMode,
                       "Shader compiler selection policy for the Vulkan backend.")
        .def_readwrite("shader_directory", &GpuDeviceDesc::shaderDirectory,
                       "Optional override for the engine shader package root.")
        .def_property(
            "shader_include_directories",
            [](const GpuDeviceDesc &desc)
            {
                std::vector<std::string> directories;
                directories.reserve(desc.shaderIncludeDirectories.size());
                for (const auto &directory : desc.shaderIncludeDirectories)
                {
                    directories.push_back(directory.string());
                }
                return directories;
            },
            [](GpuDeviceDesc &desc, const std::vector<std::string> &values)
            {
                desc.shaderIncludeDirectories.clear();
                desc.shaderIncludeDirectories.reserve(values.size());
                for (const auto &value : values)
                {
                    desc.shaderIncludeDirectories.emplace_back(value);
                }
            },
            "Ordered application shader-header search roots. Engine headers remain available under ``shader_directory/include``.");

    py::class_<RendererDesc>(m, "RendererDesc", "Configuration for the graphics renderer.")
        .def(py::init<>(), "Initializes the renderer with image-based lighting disabled.")
        .def_readwrite("ibl_quality_tier", &RendererDesc::iblQualityTier,
                       "Image-based-lighting quality tier.");

    py::class_<PhysicsSolverDesc>(m, "PhysicsSolverDesc")
        .def(py::init<>())
        .def_readwrite("gravity", &PhysicsSolverDesc::gravity)
        .def_readwrite("substeps", &PhysicsSolverDesc::substeps)
        .def_readwrite("default_iterations", &PhysicsSolverDesc::defaultIterations)
        .def_readwrite("fluid_iterations", &PhysicsSolverDesc::fluidIterations)
        .def_readwrite("soft_internal_iterations", &PhysicsSolverDesc::softInternalIterations)
        .def_readwrite("soft_contact_iterations", &PhysicsSolverDesc::softContactIterations)
        .def_readwrite("rigid_joint_iterations", &PhysicsSolverDesc::rigidJointIterations)
        .def_readwrite("rigid_rigid_contact_iterations",
                       &PhysicsSolverDesc::rigidRigidContactIterations)
        .def_readwrite("enable_blocking_readback", &PhysicsSolverDesc::enableBlockingReadback);

    py::class_<RuntimeConfig>(m, "RuntimeConfig",
                              "Configuration descriptor for initializing the CRESSim-Neo engine runtime.")
        .def(py::init<>(), "Initializes the default runtime configuration.")
        .def_readwrite("gpu_device_desc", &RuntimeConfig::gpuDeviceDesc,
                       "Desired GPU device configuration.")
        .def_readwrite("scene_layout", &RuntimeConfig::sceneLayout,
                       "Scene layout capacity settings.")
        .def_readwrite("renderer_desc", &RuntimeConfig::rendererDesc,
                       "Graphics renderer parameters.")
        .def_readwrite("physics_desc", &RuntimeConfig::physicsDesc,
                       "Physics solver parameters.");

#ifdef CRESSIM_NEO_PYTHON_HAS_VIEWER
    py::class_<DebugViewerAppDesc>(m, "DebugViewerAppDesc",
                                   "Configuration for the debug viewer application.")
        .def(py::init<>(), "Initializes the default debug viewer configuration.")
        .def_readwrite("window_title", &DebugViewerAppDesc::windowTitle,
                       "Window title.")
        .def_readwrite("width", &DebugViewerAppDesc::width,
                       "Requested render target and window width in pixels.")
        .def_readwrite("height", &DebugViewerAppDesc::height,
                       "Requested render target and window height in pixels.")
        .def_readwrite("window_enabled", &DebugViewerAppDesc::windowEnabled,
                       "Whether to create a native viewer window.")
        .def_readwrite("window_visible", &DebugViewerAppDesc::windowVisible,
                       "Whether a created viewer window starts visible.")
        .def_readwrite("start_fullscreen", &DebugViewerAppDesc::startFullscreen,
                       "Whether to start in exclusive fullscreen mode.")
        .def_readwrite("start_fullscreen_windowed", &DebugViewerAppDesc::startFullscreenWindowed,
                       "Whether to start in borderless fullscreen-windowed mode.")
        .def_readwrite("v_sync", &DebugViewerAppDesc::vSync,
                       "Whether presentation uses vertical synchronization.")
        .def_readwrite("input_sensitivity", &DebugViewerAppDesc::inputSensitivity,
                       "Mouse-look input sensitivity.")
        .def_readwrite("move_speed", &DebugViewerAppDesc::moveSpeed,
                       "Default free-camera movement speed.")
        .def_readwrite("speed_boost_scale", &DebugViewerAppDesc::speedBoostScale,
                       "Multiplier applied while boost movement is active.")
        .def_readwrite("speed_slow_scale", &DebugViewerAppDesc::speedSlowScale,
                       "Multiplier applied while slow movement is active.")
        .def_readwrite("wheel_speed_scale", &DebugViewerAppDesc::wheelSpeedScale,
                       "Mouse-wheel adjustment scale for movement speed.")
        .def_readwrite("min_move_speed", &DebugViewerAppDesc::minMoveSpeed,
                       "Lower bound for free-camera movement speed.")
        .def_readwrite("max_move_speed", &DebugViewerAppDesc::maxMoveSpeed,
                       "Upper bound for free-camera movement speed.")
        .def_readwrite("fixed_delta_seconds", &DebugViewerAppDesc::fixedDeltaSeconds,
                       "Fixed frame delta in seconds.")
        .def_readwrite("use_fixed_timestep", &DebugViewerAppDesc::useFixedTimestep,
                       "Whether viewer ticks use ``fixed_delta_seconds`` instead of wall-clock time.")
        .def_readwrite("step_simulation", &DebugViewerAppDesc::stepSimulation,
                       "Whether each viewer tick advances simulation physics and sensors.")
        .def_readwrite("max_frames", &DebugViewerAppDesc::maxFrames,
                       "Maximum number of viewer frames; zero leaves the frame count unbounded.")
        .def_readwrite("show_stats", &DebugViewerAppDesc::showStats,
                       "Whether to display viewer statistics.")
        .def_readwrite("enable_debug_particles", &DebugViewerAppDesc::enableDebugParticles,
                       "Whether to enable debug-particle rendering.")
        .def_readwrite("stats_interval_frames", &DebugViewerAppDesc::statsIntervalFrames,
                       "Frame interval used to update displayed statistics.");

    py::class_<DebugViewerCameraBinding>(
        m, "DebugViewerCameraBinding", "Viewer camera selection and optional input overrides.")
        .def(py::init<>(), "Initializes an empty debug viewer camera binding.")
        .def_readwrite("camera_entity", &DebugViewerCameraBinding::cameraEntity,
                       "Entity ID of the camera controlled and presented by the viewer.")
        .def_readwrite("move_speed", &DebugViewerCameraBinding::moveSpeed,
                       "Free-camera movement speed override; zero uses the application setting.")
        .def_readwrite("input_sensitivity", &DebugViewerCameraBinding::inputSensitivity,
                       "Mouse-look sensitivity override; zero uses the application setting.")
        .def_readwrite("speed_boost_scale", &DebugViewerCameraBinding::speedBoostScale,
                       "Boost movement scale override; zero uses the application setting.")
        .def_readwrite("speed_slow_scale", &DebugViewerCameraBinding::speedSlowScale,
                       "Slow movement scale override; zero uses the application setting.");

    py::class_<DebugViewerCallbacks>(m, "DebugViewerCallbacks",
                                     "Optional Python callbacks invoked around each viewer tick.",
                                     py::dynamic_attr())
        .def(py::init<>(), "Initializes empty debug viewer callbacks.")
        .def_property(
            "before_tick",
            [](py::object self) -> py::object
            {
                return py::hasattr(self, "_before_tick_py") ? self.attr("_before_tick_py")
                                                            : py::none();
            },
            [](py::object self, py::object value)
            {
                DebugViewerCallbacks &callbacks = self.cast<DebugViewerCallbacks &>();
                if (value.is_none())
                {
                    callbacks.beforeTick         = {};
                    self.attr("_before_tick_py") = py::none();
                    return;
                }

                py::function fn              = value.cast<py::function>();
                self.attr("_before_tick_py") = fn;
                callbacks.beforeTick         = [fn](const FrameContext &frame, Runtime &runtime)
                {
                    py::gil_scoped_acquire gil;
                    fn(frame, py::cast(&runtime, py::return_value_policy::reference));
                };
            },
            "Callable invoked before each viewer tick as ``callback(frame_context, runtime)``; set to ``None`` to clear it.")
        .def_property(
            "after_tick",
            [](py::object self) -> py::object
            {
                return py::hasattr(self, "_after_tick_py") ? self.attr("_after_tick_py")
                                                           : py::none();
            },
            [](py::object self, py::object value)
            {
                DebugViewerCallbacks &callbacks = self.cast<DebugViewerCallbacks &>();
                if (value.is_none())
                {
                    callbacks.afterTick         = {};
                    self.attr("_after_tick_py") = py::none();
                    return;
                }

                py::function fn             = value.cast<py::function>();
                self.attr("_after_tick_py") = fn;
                callbacks.afterTick         = [fn](const FrameContext &frame, Runtime &runtime)
                {
                    py::gil_scoped_acquire gil;
                    fn(frame, py::cast(&runtime, py::return_value_policy::reference));
                };
            },
            "Callable invoked after each viewer tick as ``callback(frame_context, runtime)``; set to ``None`` to clear it.");

    py::class_<DebugViewerApp>(m, "DebugViewerApp", "Interactive debug viewer application.")
        .def(py::init<>(), "Constructs a debug viewer application.")
        .def("initialize", [](DebugViewerApp &viewer, const DebugViewerAppDesc &desc,
                              RuntimeConfig &config) { return viewer.initialize(desc, config); },
             "Initializes the viewer and updates ``config`` with presentation settings. Returns ``False`` if the viewer cannot be initialized.",
             py::arg("desc"), py::arg("config"))
        .def(
            "run",
            [](DebugViewerApp &viewer, py::object runtime_obj, py::object binding_obj,
               py::object callbacks_obj) -> bool
            {
                Runtime &runtime = runtime_obj.cast<Runtime &>();
                const DebugViewerCameraBinding &binding =
                    binding_obj.cast<const DebugViewerCameraBinding &>();
                if (callbacks_obj.is_none())
                {
                    return viewer.run(runtime, binding);
                }
                const DebugViewerCallbacks &callbacks =
                    callbacks_obj.cast<const DebugViewerCallbacks &>();
                return viewer.run(runtime, binding, callbacks);
            },
            "Runs the viewer loop using a valid camera entity and initialized runtime. Optional callbacks receive ``(FrameContext, Runtime)``. Returns ``False`` if setup or execution fails.",
            py::arg("runtime"), py::arg("binding"), py::arg("callbacks") = py::none())
        .def("request_exit", &DebugViewerApp::requestExit,
             "Requests that the active viewer loop exit.")
        .def("shutdown", &DebugViewerApp::shutdown,
             "Shuts down the viewer and releases its window resources.");
#endif

    py::class_<RuntimeInfo>(m, "RuntimeInfo",
                            "Information structure holding engine version and optional feature support flags.")
        .def(py::init<>(), "Initializes empty runtime information.")
        .def_readwrite("engine_version", &RuntimeInfo::engineVersion,
                       "Full semver engine version string.")
        .def_readwrite("engine_version_major", &RuntimeInfo::engineVersionMajor,
                       "Major version number.")
        .def_readwrite("engine_version_minor", &RuntimeInfo::engineVersionMinor,
                       "Minor version number.")
        .def_readwrite("engine_version_patch", &RuntimeInfo::engineVersionPatch,
                       "Patch version number.")
        .def_readwrite("cuda_interop_supported", &RuntimeInfo::cudaInteropSupported,
                       "True if CUDA interop is enabled and available.")
        .def_readwrite("ultrasound_supported", &RuntimeInfo::ultrasoundSupported,
                       "True if CRESSim-Ultrasound integration is available.");

    py::class_<MeshHandle>(m, "MeshHandle", "Handle identifying a registered mesh resource.")
        .def(py::init<>(), "Initializes an invalid mesh handle.")
        .def_readwrite("id", &MeshHandle::id, "Resource identifier.");

    py::class_<MaterialHandle>(m, "MaterialHandle", "Handle identifying a registered material resource.")
        .def(py::init<>(), "Initializes an invalid material handle.")
        .def_readwrite("id", &MaterialHandle::id, "Resource identifier.");

    py::class_<TextureHandle>(m, "TextureHandle", "Handle identifying a registered texture resource.")
        .def(py::init<>(), "Initializes an invalid texture handle.")
        .def_readwrite("id", &TextureHandle::id, "Resource identifier.");

    py::class_<MeshResourceDesc::Vertex>(m, "MeshVertex", "Vertex data for a mesh resource.")
        .def(py::init<>(), "Initializes a vertex with default position, normal, texture coordinates, and tangent.")
        .def_readwrite("position", &MeshResourceDesc::Vertex::position, "Vertex position.")
        .def_readwrite("normal", &MeshResourceDesc::Vertex::normal, "Vertex normal.")
        .def_readwrite("tex_coord_u", &MeshResourceDesc::Vertex::texCoordU,
                       "U texture coordinate.")
        .def_readwrite("tex_coord_v", &MeshResourceDesc::Vertex::texCoordV,
                       "V texture coordinate.")
        .def_readwrite("tangent", &MeshResourceDesc::Vertex::tangent,
                       "Tangent vector with handedness in its w component.");

    py::class_<MeshResourceDesc>(m, "MeshResourceDesc", "Describes mesh data to register.")
        .def(py::init<>(), "Initializes an empty mesh description.")
        .def_readwrite("debug_name", &MeshResourceDesc::debugName, "Debug name for the mesh.")
        .def_readwrite("vertices", &MeshResourceDesc::vertices,
                       "Mesh vertices. Registration normalizes normals and generates or repairs tangents.")
        .def_readwrite("indices", &MeshResourceDesc::indices,
                       "Triangle index buffer.");

    py::class_<MaterialPipelineDesc>(m, "MaterialPipelineDesc",
                                     "Describes the shader program and features of a material.")
        .def(py::init<>(), "Initializes the standard lit pipeline with no optional features.")
        .def_readwrite("program_family", &MaterialPipelineDesc::programFamily,
                       "Material shader program family.")
        .def_readwrite("feature_flags", &MaterialPipelineDesc::featureFlags,
                       "Optional material shader features.")
        .def_readwrite("alpha_cutoff", &MaterialPipelineDesc::alphaCutoff,
                       "Alpha threshold used by alpha-test shading.");

    py::class_<MaterialResourceDesc>(m, "MaterialResourceDesc",
                                     "Describes a material resource to register.")
        .def(py::init<>(), "Initializes a default opaque material.")
        .def_readwrite("debug_name", &MaterialResourceDesc::debugName, "Debug name for the material.")
        .def_readwrite("base_color", &MaterialResourceDesc::baseColor, "Base color factor.")
        .def_readwrite("metallic", &MaterialResourceDesc::metallic, "Metallic factor.")
        .def_readwrite("roughness", &MaterialResourceDesc::roughness, "Roughness factor.")
        .def_readwrite("emissive_factor", &MaterialResourceDesc::emissiveFactor,
                       "Emissive color factor.")
        .def_readwrite("base_color_texture", &MaterialResourceDesc::baseColorTexture,
                       "Base-color texture handle.")
        .def_readwrite("normal_texture", &MaterialResourceDesc::normalTexture,
                       "Normal-map texture handle. Registration enables the NormalMap feature when its identifier is not invalid.")
        .def_readwrite("metallic_roughness_texture",
                       &MaterialResourceDesc::metallicRoughnessTexture,
                       "Metallic-roughness texture handle.")
        .def_readwrite("emissive_texture", &MaterialResourceDesc::emissiveTexture,
                       "Emissive texture handle.")
        .def_readwrite("ao_texture", &MaterialResourceDesc::aoTexture,
                       "Ambient-occlusion texture handle.")
        .def_readwrite("pipeline", &MaterialResourceDesc::pipeline,
                       "Material pipeline description.")
        .def_readwrite("render_mode", &MaterialResourceDesc::renderMode,
                       "Material render mode. Registration enables AlphaTest exactly for Cutout materials.")
        .def_readwrite("render_order", &MaterialResourceDesc::renderOrder,
                       "Orders materials only within the same render mode; lower values draw earlier.")
        .def_readwrite("opacity", &MaterialResourceDesc::opacity, "Base-color alpha factor.")
        .def_readwrite("casts_shadows", &MaterialResourceDesc::castsShadows,
                       "Whether the material casts shadows.")
        .def_readwrite("receives_shadows", &MaterialResourceDesc::receivesShadows,
                       "Whether the material receives shadows.");

    py::class_<TextureResourceDesc::SubresourceDesc>(m, "TextureSubresourceDesc",
                                                      "Pixel data for one texture mip level and layer.")
        .def(py::init<>(), "Initializes an empty subresource description.")
        .def_readwrite("pixel_data", &TextureResourceDesc::SubresourceDesc::pixelData,
                       "Raw pixel bytes for the subresource.");

    py::class_<TextureResourceDesc>(m, "TextureResourceDesc",
                                    "Describes texture data to register.")
        .def(py::init<>(), "Initializes a one-pixel, one-mip 2D RGBA8 linear texture description.")
        .def_readwrite("debug_name", &TextureResourceDesc::debugName, "Debug name for the texture.")
        .def_readwrite("width", &TextureResourceDesc::width,
                       "Base-level texture width in pixels. Registration clamps it to at least one.")
        .def_readwrite("height", &TextureResourceDesc::height,
                       "Base-level texture height in pixels. Registration clamps it to at least one.")
        .def_readwrite("mip_level_count", &TextureResourceDesc::mipLevelCount,
                       "Number of mip levels. Registration clamps it to at least one.")
        .def_readwrite("dimension", &TextureResourceDesc::dimension, "Texture dimension.")
        .def_readwrite("pixel_format", &TextureResourceDesc::pixelFormat, "Texture pixel format.")
        .def_readwrite("color_space", &TextureResourceDesc::colorSpace, "Texture color space.")
        .def_readwrite("mip_policy", &TextureResourceDesc::mipPolicy,
                       "Requested mipmap policy retained with the resource description.")
        .def_readwrite("subresources", &TextureResourceDesc::subresources,
                       "Mip-major pixel data: one entry per mip and layer; cube textures have six layers.")
        .def_readwrite("pixel_data", &TextureResourceDesc::pixelData,
                       "Convenience base-level pixel data used at registration only when subresources is empty.");

    py::class_<MeshRendererComponent>(
        m, "MeshRendererComponent",
        "Mesh renderer component binding a 3D mesh and material for visual rendering.")
        .def(py::init<>(), "Initializes a mesh renderer with invalid resource handles and visible set to true.")
        .def_readwrite("mesh", &MeshRendererComponent::mesh, "Handle to the geometry mesh resource.")
        .def_readwrite("material", &MeshRendererComponent::material,
                       "Handle to the visual material resource.")
        .def_readwrite("segmentation_id", &MeshRendererComponent::segmentationId,
                       "ID for semantic image segmentation passes.")
        .def_readwrite("visible", &MeshRendererComponent::visible,
                       "Visibility flag for camera rendering.");

    py::class_<CameraComponent>(m, "CameraComponent",
                                "Camera component defining projection, view targets, and rendering modes.")
        .def(py::init<>(), "Initializes a camera with the default projection, output, and clear settings.")
        .def_readwrite("vertical_fov_degrees", &CameraComponent::verticalFovDegrees,
                       "Vertical Field-of-View in degrees.")
        .def_readwrite("near_clip", &CameraComponent::nearClip, "Near clipping plane distance.")
        .def_readwrite("far_clip", &CameraComponent::farClip, "Far clipping plane distance.")
        .def_readwrite("product", &CameraComponent::product, "Rendered camera output product type.")
        .def_readwrite("output", &CameraComponent::output, "Target render output binding descriptor.")
        .def_readwrite("output_width", &CameraComponent::outputWidth,
                       "Optional explicit target width (0 for default).")
        .def_readwrite("output_height", &CameraComponent::outputHeight,
                       "Optional explicit target height (0 for default).")
        .def_readwrite("viewport", &CameraComponent::viewport,
                       "Viewport rectangle on the output target.")
        .def_readwrite("clear_color", &CameraComponent::clearColor,
                       "Whether to clear target color buffer before rendering.")
        .def_readwrite("clear_depth", &CameraComponent::clearDepth,
                       "Whether to clear target depth buffer before rendering.")
        .def_readwrite("clear_color_value", &CameraComponent::clearColorValue,
                       "Clear color RGBA values.")
        .def_readwrite("clear_depth_value", &CameraComponent::clearDepthValue,
                       "Clear depth value.")
        .def_readwrite("background_mode", &CameraComponent::backgroundMode,
                       "Camera background rendering mode.")
        .def_readwrite("render_order", &CameraComponent::renderOrder,
                       "Rendering priority order (ascending).");

    py::class_<DirectionalLightComponent>(
        m, "DirectionalLightComponent",
        "Directional light source for global scene illumination and shadow mapping.")
        .def(py::init<>(), "Initializes a downward white directional light that casts shadows.")
        .def_readwrite("direction", &DirectionalLightComponent::direction, "Light direction vector.")
        .def_readwrite("color", &DirectionalLightComponent::color, "Light color RGB values.")
        .def_readwrite("intensity", &DirectionalLightComponent::intensity,
                       "Illumination intensity multiplier.")
        .def_readwrite("range", &DirectionalLightComponent::range,
                       "Maximum light range (0 for infinite).")
        .def_readwrite("shadow_distance", &DirectionalLightComponent::shadowDistance,
                       "Maximum shadow rendering distance.")
        .def_readwrite("shadow_fade_distance", &DirectionalLightComponent::shadowFadeDistance,
                       "Distance over which shadows fade out.")
        .def_readwrite("shadow_bias", &DirectionalLightComponent::shadowBias,
                       "Shadow depth comparison bias.")
        .def_readwrite("casts_shadows", &DirectionalLightComponent::castsShadows,
                       "Enable shadow map generation.");

    py::class_<PointLightComponent>(m, "PointLightComponent",
                                    "Point light source emitting light uniformly in all directions.")
        .def(py::init<>(), "Initializes a white point light that does not cast shadows.")
        .def_readwrite("color", &PointLightComponent::color, "Light color RGB values.")
        .def_readwrite("intensity", &PointLightComponent::intensity,
                       "Illumination intensity multiplier.")
        .def_readwrite("range", &PointLightComponent::range, "Attenuation distance range.")
        .def_readwrite("shadow_bias", &PointLightComponent::shadowBias, "Shadow depth bias.")
        .def_readwrite("casts_shadows", &PointLightComponent::castsShadows,
                       "Enable shadow map generation.");

    py::class_<SpotLightComponent>(m, "SpotLightComponent",
                                   "Spot light source emitting a cone of light in a specified direction.")
        .def(py::init<>(), "Initializes a downward white spot light that does not cast shadows.")
        .def_readwrite("direction", &SpotLightComponent::direction,
                       "Spot light emission direction.")
        .def_readwrite("color", &SpotLightComponent::color, "Light color RGB values.")
        .def_readwrite("intensity", &SpotLightComponent::intensity,
                       "Illumination intensity multiplier.")
        .def_readwrite("range", &SpotLightComponent::range, "Attenuation distance range.")
        .def_readwrite("inner_cone_angle", &SpotLightComponent::innerConeAngle,
                       "Inner full-intensity cone angle (degrees).")
        .def_readwrite("outer_cone_angle", &SpotLightComponent::outerConeAngle,
                       "Outer zero-intensity cone angle (degrees).")
        .def_readwrite("shadow_bias", &SpotLightComponent::shadowBias, "Shadow depth bias.")
        .def_readwrite("casts_shadows", &SpotLightComponent::castsShadows,
                       "Enable shadow map generation.");

    py::class_<EnvironmentIblDesc>(m, "EnvironmentIblDesc",
                                   "Describes image-based lighting and skybox textures for an environment.")
        .def(py::init<>(), "Initializes an environment IBL description with invalid texture handles.")
        .def_readwrite("background_cubemap", &EnvironmentIblDesc::backgroundCubemap,
                       "Cubemap used for the environment background.")
        .def_readwrite("irradiance_cubemap", &EnvironmentIblDesc::irradianceCubemap,
                       "Cubemap used for diffuse image-based lighting.")
        .def_readwrite("prefiltered_specular_cubemap",
                       &EnvironmentIblDesc::prefilteredSpecularCubemap,
                       "Cubemap used for prefiltered specular image-based lighting.")
        .def_readwrite("intensity", &EnvironmentIblDesc::intensity,
                       "Image-based-lighting intensity multiplier.")
        .def_readwrite("background_intensity", &EnvironmentIblDesc::backgroundIntensity,
                       "Environment background intensity multiplier.")
        .def("enabled", &EnvironmentIblDesc::enabled,
             "Returns false for Off; for DiffuseOnly, checks that the irradiance handle ID is not "
             "invalid; for Full, checks the irradiance and prefiltered specular handle IDs.");

    py::class_<EnvironmentFluidDesc>(m, "EnvironmentFluidDesc",
                                     "Configures fluid surface filtering and compositing for an environment.")
        .def(py::init<>(), "Initializes the default fluid rendering configuration.")
        .def_readwrite("smoothness", &EnvironmentFluidDesc::smoothness,
                       "Fluid surface smoothness passed to the composite shader.")
        .def_readwrite("specular", &EnvironmentFluidDesc::specular,
                       "RGB specular factor passed to the composite shader.")
        .def_readwrite("fresnel", &EnvironmentFluidDesc::fresnel,
                       "Fresnel factor passed to the composite shader.")
        .def_readwrite("depth_edge_threshold", &EnvironmentFluidDesc::depthEdgeThreshold,
                       "Depth discontinuity threshold used for fluid normal reconstruction.")
        .def_readwrite("filter_radius_pixels", &EnvironmentFluidDesc::filterRadiusPixels,
                       "Maximum fluid depth-filter radius in pixels; values below one are clamped to one.")
        .def_readwrite("filter_world_radius", &EnvironmentFluidDesc::filterWorldRadius,
                       "World-space fluid depth-filter radius; values at or below 1e-4 use 0.18.")
        .def_readwrite("filter_depth_threshold", &EnvironmentFluidDesc::filterDepthThreshold,
                       "Depth threshold for fluid depth filtering; values at or below 1e-4 use 1e-4.")
        .def_readwrite("enable_background_refraction",
                       &EnvironmentFluidDesc::enableBackgroundRefraction,
                       "Whether fluid compositing refracts the background color.")
        .def_readwrite("refraction_ior", &EnvironmentFluidDesc::refractionIor,
                       "Index of refraction passed to the fluid composite shader.")
        .def_readwrite("refraction_view_thickness", &EnvironmentFluidDesc::refractionViewThickness,
                       "View-space thickness used for fluid refraction.");

    py::class_<ParticleContactMaterialDesc>(m, "ParticleContactMaterialDesc",
                                            "Contact material parameters for particles.")
        .def(py::init<>(), "Initializes a contact material with zero friction, restitution, and damping.")
        .def_readwrite("friction", &ParticleContactMaterialDesc::friction,
                       "Dynamic friction coefficient.")
        .def_readwrite("restitution", &ParticleContactMaterialDesc::restitution,
                       "Restitution coefficient.")
        .def_readwrite("damping", &ParticleContactMaterialDesc::damping,
                       "Contact damping coefficient.")
        .def_readwrite("static_friction", &ParticleContactMaterialDesc::staticFriction,
                       "Static friction coefficient; a negative value uses friction when the material is normalized.");

    py::class_<SoftBodyRegularGridSource>(m, "SoftBodyRegularGridSource",
                                          "Parameters for generating a tetrahedral soft body from a regular grid.")
        .def(py::init<>(), "Initializes a one-unit grid with target particle spacing 0.25.")
        .def_readwrite("size", &SoftBodyRegularGridSource::size,
                       "Grid extent in object space.")
        .def_readwrite("target_particle_spacing", &SoftBodyRegularGridSource::targetParticleSpacing,
                       "Target distance between generated grid particles.")
        .def_readwrite("static_particle_indices",
                       &SoftBodyRegularGridSource::staticParticleIndices,
                       "Generated particle indices to make static.");

    py::class_<SoftBodyTetMeshSource>(m, "SoftBodyTetMeshSource",
                                      "Tetrahedral-mesh source data for a soft body.")
        .def(py::init<>(), "Initializes an empty tetrahedral-mesh source.")
        .def_readwrite("object_space_rest_positions",
                       &SoftBodyTetMeshSource::objectSpaceRestPositions,
                       "Object-space rest positions of tetrahedral-mesh vertices.")
        .def_readwrite("tet_vertex_indices", &SoftBodyTetMeshSource::tetVertexIndices,
                       "Flat tetrahedron index buffer; each group of four indices defines one tetrahedron.")
        .def_readwrite("static_particle_indices", &SoftBodyTetMeshSource::staticParticleIndices,
                       "Particle indices to make static.");

    py::class_<SoftBodyTetGenSource>(m, "SoftBodyTetGenSource",
                                     "TetGen-file source data for a soft body.")
        .def(py::init<>(), "Initializes an empty TetGen-file source.")
        .def_readwrite("node_file", &SoftBodyTetGenSource::nodeFile,
                       "Path to the TetGen .node file containing rest positions.")
        .def_readwrite("ele_file", &SoftBodyTetGenSource::eleFile,
                       "Path to the TetGen .ele file containing tetrahedra.")
        .def_readwrite("static_particle_indices", &SoftBodyTetGenSource::staticParticleIndices,
                       "Particle indices to make static.");

    py::class_<SoftBodyMeshfreeParticleSource>(m, "SoftBodyMeshfreeParticleSource",
                                               "Particle source data for a meshfree soft body.")
        .def(py::init<>(), "Initializes an empty meshfree source with 12 neighbours per particle.")
        .def_readwrite("particle_rest_positions",
                       &SoftBodyMeshfreeParticleSource::particleRestPositions,
                       "Rest positions of the simulated particles.")
        .def_readwrite("surface_rest_positions",
                       &SoftBodyMeshfreeParticleSource::surfaceRestPositions,
                       "Rest positions of the render surface vertices.")
        .def_readwrite("surface_normals", &SoftBodyMeshfreeParticleSource::surfaceNormals,
                       "Rest normals of the render surface vertices.")
        .def_readwrite("surface_triangles", &SoftBodyMeshfreeParticleSource::surfaceTriangles,
                       "Triangle indices for the render surface.")
        .def_readwrite("static_particle_indices",
                       &SoftBodyMeshfreeParticleSource::staticParticleIndices,
                       "Particle indices to make static.")
        .def_readwrite("neighbour_count", &SoftBodyMeshfreeParticleSource::neighbourCount,
                       "Number of nearest neighbours used to build the particle graph.");

    py::class_<SoftBodySourceDesc>(m, "SoftBodySourceDesc",
                                   "Selects and stores source data for a soft body.")
        .def(py::init<>(), "Initializes a regular-grid soft-body source.")
        .def_readwrite("kind", &SoftBodySourceDesc::kind,
                       "Selects which source-data member is used.")
        .def_readwrite("regular_grid", &SoftBodySourceDesc::regularGrid,
                       "Source data used when kind is RegularGrid.")
        .def_readwrite("tet_mesh", &SoftBodySourceDesc::tetMesh,
                       "Source data used when kind is TetMesh.")
        .def_readwrite("tet_gen", &SoftBodySourceDesc::tetGen,
                       "Source data used when kind is TetGenFiles.")
        .def_readwrite("meshfree_particles", &SoftBodySourceDesc::meshfreeParticles,
                       "Source data used when kind is MeshfreeParticles.");

    py::class_<FluidRegularGridSource>(m, "FluidRegularGridSource",
                                       "Parameters for generating fluid particles from a regular grid.")
        .def(py::init<>(), "Initializes a one-unit grid with target particle spacing 0.25.")
        .def_readwrite("size", &FluidRegularGridSource::size, "Grid extent in object space.")
        .def_readwrite("target_particle_spacing", &FluidRegularGridSource::targetParticleSpacing,
                       "Target distance between generated fluid particles.");

    py::class_<FluidSourceDesc>(m, "FluidSourceDesc", "Selects and stores source data for a fluid body.")
        .def(py::init<>(), "Initializes a regular-grid fluid source.")
        .def_readwrite("kind", &FluidSourceDesc::kind,
                       "Selects which source-data member is used.")
        .def_readwrite("regular_grid", &FluidSourceDesc::regularGrid,
                       "Source data used when kind is RegularGrid.");

    py::class_<SoftBodyMaterialDesc>(m, "SoftBodyMaterialDesc",
                                     "Material parameters for a soft body.")
        .def(py::init<>(), "Initializes the default soft-body material.")
        .def_readwrite("contact", &SoftBodyMaterialDesc::contact,
                       "Particle contact material parameters.");

    py::class_<StrandMaterialDesc>(m, "StrandMaterialDesc", "Material parameters for a strand.")
        .def(py::init<>(), "Initializes the default strand material.")
        .def_readwrite("contact", &StrandMaterialDesc::contact,
                       "Particle contact material parameters.");

    py::class_<FluidMaterialDesc>(m, "FluidMaterialDesc", "Material parameters for a fluid body.")
        .def(py::init<>(), "Initializes the default fluid material.")
        .def_readwrite("contact", &FluidMaterialDesc::contact,
                       "Particle contact material parameters.")
        .def_readwrite("viscosity", &FluidMaterialDesc::viscosity, "Fluid viscosity coefficient.")
        .def_readwrite("cohesion", &FluidMaterialDesc::cohesion, "Fluid cohesion coefficient.")
        .def_readwrite("surface_tension", &FluidMaterialDesc::surfaceTension,
                       "Fluid surface-tension coefficient.")
        .def_readwrite("vorticity_confinement", &FluidMaterialDesc::vorticityConfinement,
                       "Fluid vorticity-confinement coefficient.")
        .def_readwrite("gravity_scale", &FluidMaterialDesc::gravityScale,
                       "Multiplier applied to gravity for this fluid.")
        .def_readwrite("cfl_coefficient", &FluidMaterialDesc::cflCoefficient,
                       "Coefficient used to derive the fluid CFL radius.");

    py::class_<RigidBodyComponent>(
        m, "RigidBodyComponent",
        "Rigid body component defining linear/angular velocity, mass properties, and kinematic targets.")
        .def(py::init<>(), "Initializes the default rigid body component.")
        .def_readwrite("linear_velocity", &RigidBodyComponent::linearVelocity,
                       "Linear velocity vector.")
        .def_readwrite("angular_velocity", &RigidBodyComponent::angularVelocity,
                       "Angular velocity vector.")
        .def_readwrite("inverse_inertia_local", &RigidBodyComponent::inverseInertiaLocal,
                       "Local inverse inertia tensor diagonal.")
        .def_readwrite("proxy_particle_local_positions",
                       &RigidBodyComponent::proxyParticleLocalPositions,
                       "Local positions of proxy particles.")
        .def_readwrite("proxy_particle_material", &RigidBodyComponent::proxyParticleMaterial,
                       "Proxy particle contact material properties.")
        .def_readwrite("body_type", &RigidBodyComponent::bodyType,
                       "Rigid body type (Dynamic, Static, Kinematic).")
        .def_readwrite("inverse_mass", &RigidBodyComponent::inverseMass, "Inverse mass (1/kg).")
        .def_readwrite("proxy_particle_radius", &RigidBodyComponent::proxyParticleRadius,
                       "Radius of proxy collision particles.")
        .def_readwrite("proxy_collision_layer", &RigidBodyComponent::proxyCollisionLayer,
                       "Bitmask collision layer for proxy particles.")
        .def_readwrite("proxy_collision_mask", &RigidBodyComponent::proxyCollisionMask,
                       "Bitmask collision mask for proxy particles.")
        .def_readwrite("suturing_enabled", &RigidBodyComponent::suturingEnabled,
                       "Enable surgical suturing needle proxy interactions.")
        .def_readwrite("needle_tip_proxy_index", &RigidBodyComponent::needleTipProxyIndex,
                       "Proxy particle index representing needle tip.")
        .def_readwrite("kinematic_target_position", &RigidBodyComponent::kinematicTargetPosition,
                       "Kinematic target position for interpolation.")
        .def_readwrite("kinematic_target_rotation", &RigidBodyComponent::kinematicTargetRotation,
                       "Kinematic target orientation quaternion.")
        .def_readwrite("kinematic_target_enabled", &RigidBodyComponent::kinematicTargetEnabled,
                       "Enable kinematic target positioning.");

    py::class_<SoftBodyComponent>(
        m, "SoftBodyComponent",
        "Soft body component supporting tetrahedral and meshfree particle sources.")
        .def(py::init<>(), "Initializes the default soft body component.")
        .def_readwrite("source", &SoftBodyComponent::source,
                       "Source mesh file or tetrahedral asset descriptor.")
        .def_readwrite("material", &SoftBodyComponent::material,
                       "Particle contact material parameters.")
        .def_readwrite("particle_mass", &SoftBodyComponent::particleMass, "Mass per node particle.")
        .def_readwrite("particle_radius", &SoftBodyComponent::particleRadius,
                       "Collision radius per particle.")
        .def_readwrite("edge_compliance", &SoftBodyComponent::edgeCompliance,
                       "Extended Position Based Dynamics (XPBD) edge constraint compliance.")
        .def_readwrite("volume_compliance", &SoftBodyComponent::volumeCompliance,
                       "XPBD volume conservation constraint compliance.")
        .def_readwrite("self_collision_enabled", &SoftBodyComponent::selfCollisionEnabled,
                       "Enable internal self-collision handling.")
        .def_readwrite("supports_suturing", &SoftBodyComponent::supportsSuturing,
                       "Enable surgical thread suturing insertion.")
        .def_readwrite("collision_layer", &SoftBodyComponent::collisionLayer,
                       "Collision bitmask layer.")
        .def_readwrite("collision_mask", &SoftBodyComponent::collisionMask,
                       "Collision bitmask filter.");

    py::class_<MeshfreeSoftBodyComponent>(
        m, "MeshfreeSoftBodyComponent",
        "Meshfree / particle-based soft body component for point cloud elastic simulation.")
        .def(py::init<>(), "Initializes the default meshfree soft body component.")
        .def_readwrite("particles", &MeshfreeSoftBodyComponent::particles,
                       "Particle rest position list.")
        .def_readwrite("surface_rest_positions", &MeshfreeSoftBodyComponent::surfaceRestPositions,
                       "Surface mesh rest position coordinates.")
        .def_readwrite("surface_normals", &MeshfreeSoftBodyComponent::surfaceNormals,
                       "Surface mesh normal vectors.")
        .def_readwrite("surface_triangles", &MeshfreeSoftBodyComponent::surfaceTriangles,
                       "Surface mesh triangle indices.")
        .def_readwrite("static_particle_indices", &MeshfreeSoftBodyComponent::staticParticleIndices,
                       "Particle indices fixed in space.")
        .def_readwrite("material", &MeshfreeSoftBodyComponent::material,
                       "Material property descriptor.")
        .def_readwrite("particle_radius", &MeshfreeSoftBodyComponent::particleRadius,
                       "Particle radius.")
        .def_readwrite("particle_mass", &MeshfreeSoftBodyComponent::particleMass,
                       "Mass per particle.")
        .def_readwrite("neighbour_count", &MeshfreeSoftBodyComponent::neighbourCount,
                       "Particle neighbor interaction count.")
        .def_readwrite("compliance", &MeshfreeSoftBodyComponent::compliance,
                       "Constraint compliance parameter.")
        .def_readwrite("self_collision_enabled", &MeshfreeSoftBodyComponent::selfCollisionEnabled,
                       "Enable self-collision.")
        .def_readwrite("collision_layer", &MeshfreeSoftBodyComponent::collisionLayer,
                       "Collision layer.")
        .def_readwrite("collision_mask", &MeshfreeSoftBodyComponent::collisionMask,
                       "Collision mask.");

    py::class_<HingeJointState>(m, "HingeJointState",
                                "State for a hinge joint between two rigid bodies.")
        .def(py::init<>(), "Initializes the default hinge-joint state.")
        .def_readwrite("joint_id", &HingeJointState::jointId,
                       "Joint identifier; an invalid ID creates a joint on upsert.")
        .def_readwrite("enabled", &HingeJointState::enabled, "Enable this joint.")
        .def_readwrite("suppress_connected_body_collisions",
                       &HingeJointState::suppressConnectedBodyCollisions,
                       "Suppress collisions between the connected bodies.")
        .def_readwrite("drive_mode", &HingeJointState::driveMode, "Hinge drive control mode.")
        .def_readwrite("limit_enabled", &HingeJointState::limitEnabled, "Enable hinge angle limits.")
        .def_readwrite("body_a", &HingeJointState::bodyA,
                       "First connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("body_b", &HingeJointState::bodyB,
                       "Second connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("local_anchor_a", &HingeJointState::localAnchorA,
                       "Joint anchor in body A local coordinates.")
        .def_readwrite("local_anchor_b", &HingeJointState::localAnchorB,
                       "Joint anchor in body B local coordinates.")
        .def_readwrite("local_rotation_a", &HingeJointState::localRotationA,
                       "Joint frame orientation in body A local coordinates.")
        .def_readwrite("local_rotation_b", &HingeJointState::localRotationB,
                       "Joint frame orientation in body B local coordinates.")
        .def_readwrite("limit_min", &HingeJointState::limitMin, "Minimum hinge angle limit.")
        .def_readwrite("limit_max", &HingeJointState::limitMax, "Maximum hinge angle limit.")
        .def_readwrite("constraint_compliance", &HingeJointState::constraintCompliance,
                       "Hinge constraint compliance.")
        .def_readwrite("drive_compliance", &HingeJointState::driveCompliance,
                       "Hinge drive compliance.")
        .def_readwrite("drive_target_angle", &HingeJointState::driveTargetAngle,
                       "Target angle for position drive control.")
        .def_readwrite("drive_damping", &HingeJointState::driveDamping,
                       "Damping applied by the hinge drive.")
        .def_readwrite("drive_max_angular_velocity", &HingeJointState::driveMaxAngularVelocity,
                       "Maximum angular velocity applied by the hinge drive.")
        .def_readwrite("drive_target_angular_velocity",
                       &HingeJointState::driveTargetAngularVelocity,
                       "Target angular velocity for velocity drive control.");

    py::class_<BallJointState>(m, "BallJointState",
                               "State for a ball joint that constrains two local anchors together.")
        .def(py::init<>(), "Initializes the default ball-joint state.")
        .def_readwrite("joint_id", &BallJointState::jointId,
                       "Joint identifier; an invalid ID creates a joint on upsert.")
        .def_readwrite("enabled", &BallJointState::enabled, "Enable this joint.")
        .def_readwrite("suppress_connected_body_collisions",
                       &BallJointState::suppressConnectedBodyCollisions,
                       "Suppress collisions between the connected bodies.")
        .def_readwrite("body_a", &BallJointState::bodyA,
                       "First connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("body_b", &BallJointState::bodyB,
                       "Second connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("local_anchor_a", &BallJointState::localAnchorA,
                       "Joint anchor in body A local coordinates.")
        .def_readwrite("local_anchor_b", &BallJointState::localAnchorB,
                       "Joint anchor in body B local coordinates.");

    py::class_<SphericalJointState>(
        m, "SphericalJointState",
        "State for a spherical joint with optional swing, twist, and orientation-drive constraints.")
        .def(py::init<>(), "Initializes the default spherical-joint state.")
        .def_readwrite("joint_id", &SphericalJointState::jointId,
                       "Joint identifier; an invalid ID creates a joint on upsert.")
        .def_readwrite("enabled", &SphericalJointState::enabled, "Enable this joint.")
        .def_readwrite("suppress_connected_body_collisions",
                       &SphericalJointState::suppressConnectedBodyCollisions,
                       "Suppress collisions between the connected bodies.")
        .def_readwrite("drive_mode", &SphericalJointState::driveMode,
                       "Spherical-joint drive control mode.")
        .def_readwrite("limit_enabled", &SphericalJointState::limitEnabled,
                       "Enable swing and twist limits.")
        .def_readwrite("body_a", &SphericalJointState::bodyA,
                       "First connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("body_b", &SphericalJointState::bodyB,
                       "Second connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("local_anchor_a", &SphericalJointState::localAnchorA,
                       "Joint anchor in body A local coordinates.")
        .def_readwrite("local_anchor_b", &SphericalJointState::localAnchorB,
                       "Joint anchor in body B local coordinates.")
        .def_readwrite("local_rotation_a", &SphericalJointState::localRotationA,
                       "Joint frame orientation in body A local coordinates.")
        .def_readwrite("local_rotation_b", &SphericalJointState::localRotationB,
                       "Joint frame orientation in body B local coordinates.")
        .def_readwrite("swing_limit_y", &SphericalJointState::swingLimitY,
                       "Maximum swing limit about the joint Y axis.")
        .def_readwrite("swing_limit_z", &SphericalJointState::swingLimitZ,
                       "Maximum swing limit about the joint Z axis.")
        .def_readwrite("twist_limit_min", &SphericalJointState::twistLimitMin,
                       "Minimum twist angle limit.")
        .def_readwrite("twist_limit_max", &SphericalJointState::twistLimitMax,
                       "Maximum twist angle limit.")
        .def_readwrite("constraint_compliance", &SphericalJointState::constraintCompliance,
                       "Anchor constraint compliance.")
        .def_readwrite("swing_compliance", &SphericalJointState::swingCompliance,
                       "Swing-limit constraint compliance.")
        .def_readwrite("twist_compliance", &SphericalJointState::twistCompliance,
                       "Twist-limit constraint compliance.")
        .def_readwrite("drive_compliance", &SphericalJointState::driveCompliance,
                       "Orientation-drive compliance.")
        .def_readwrite("drive_target_orientation", &SphericalJointState::driveTargetOrientation,
                       "Target orientation for orientation drive control.");

    py::class_<SliderJointState>(m, "SliderJointState",
                                "State for a slider joint between two rigid bodies.")
        .def(py::init<>(), "Initializes the default slider-joint state.")
        .def_readwrite("joint_id", &SliderJointState::jointId,
                       "Joint identifier; an invalid ID creates a joint on upsert.")
        .def_readwrite("enabled", &SliderJointState::enabled, "Enable this joint.")
        .def_readwrite("suppress_connected_body_collisions",
                       &SliderJointState::suppressConnectedBodyCollisions,
                       "Suppress collisions between the connected bodies.")
        .def_readwrite("drive_mode", &SliderJointState::driveMode, "Slider drive control mode.")
        .def_readwrite("limit_enabled", &SliderJointState::limitEnabled,
                       "Enable slider position limits.")
        .def_readwrite("body_a", &SliderJointState::bodyA,
                       "First connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("body_b", &SliderJointState::bodyB,
                       "Second connected body ID. Use an entity ID when upserting through World; retrieved states contain a rigid-body ID.")
        .def_readwrite("local_anchor_a", &SliderJointState::localAnchorA,
                       "Joint anchor in body A local coordinates.")
        .def_readwrite("local_anchor_b", &SliderJointState::localAnchorB,
                       "Joint anchor in body B local coordinates.")
        .def_readwrite("local_rotation_a", &SliderJointState::localRotationA,
                       "Joint frame orientation in body A local coordinates.")
        .def_readwrite("local_rotation_b", &SliderJointState::localRotationB,
                       "Joint frame orientation in body B local coordinates.")
        .def_readwrite("limit_min", &SliderJointState::limitMin, "Minimum slider position limit.")
        .def_readwrite("limit_max", &SliderJointState::limitMax, "Maximum slider position limit.")
        .def_readwrite("constraint_compliance", &SliderJointState::constraintCompliance,
                       "Slider constraint compliance.")
        .def_readwrite("drive_compliance", &SliderJointState::driveCompliance,
                       "Slider drive compliance.")
        .def_readwrite("drive_damping", &SliderJointState::driveDamping,
                       "Damping applied by the slider drive.")
        .def_readwrite("drive_max_velocity", &SliderJointState::driveMaxVelocity,
                       "Maximum velocity applied by the slider drive.")
        .def_readwrite("drive_target_position", &SliderJointState::driveTargetPosition,
                       "Target position for position drive control.")
        .def_readwrite("drive_target_velocity", &SliderJointState::driveTargetVelocity,
                       "Target velocity for velocity drive control.");

    py::class_<ColliderComponent>(
        m, "ColliderComponent",
        "Collider component attached to entities for physical collision queries and dynamics.")
        .def(py::init<>(), "Initializes the default collider component.")
        .def_readwrite("shape_type", &ColliderComponent::shapeType,
                       "Collision primitive geometry shape type.")
        .def_readwrite("shape_params", &ColliderComponent::shapeParams,
                       "Shape dimensions (e.g. radius, box extents).")
        .def_readwrite("local_position", &ColliderComponent::localPosition,
                       "Local offset position relative to transform.")
        .def_readwrite("local_rotation", &ColliderComponent::localRotation,
                       "Local offset rotation relative to transform.")
        .def_readwrite("enabled", &ColliderComponent::enabled, "Active collision state.")
        .def_readwrite("friction", &ColliderComponent::friction,
                       "Dynamic friction coefficient.")
        .def_readwrite("static_friction", &ColliderComponent::staticFriction,
                       "Static friction coefficient (-1 to reuse dynamic friction).")
        .def_readwrite("restitution", &ColliderComponent::restitution,
                       "Coefficient of restitution (bounciness).")
        .def_readwrite("collision_layer", &ColliderComponent::collisionLayer,
                       "Bitmask collision layer.")
        .def_readwrite("collision_mask", &ColliderComponent::collisionMask,
                       "Bitmask collision filtering mask.");

    py::class_<StrandComponent>(m, "StrandComponent",
                                "1D elastic strand component for surgical threads and sutures.")
        .def(py::init<>(), "Initializes the default strand component.")
        .def_readwrite("material", &StrandComponent::material,
                       "Particle contact material parameters.")
        .def_readwrite("rest_positions", &StrandComponent::restPositions,
                       "Rest positions of strand particles.")
        .def_readwrite("static_particle_indices", &StrandComponent::staticParticleIndices,
                       "Fixed node particle indices.")
        .def_readwrite("particle_mass", &StrandComponent::particleMass,
                       "Mass per strand node particle.")
        .def_readwrite("particle_radius", &StrandComponent::particleRadius,
                       "Collision radius per strand node.")
        .def_readwrite("stretch_shear_compliance", &StrandComponent::stretchShearCompliance,
                       "Stretching and shearing constraint compliance.")
        .def_readwrite("bend_compliance", &StrandComponent::bendCompliance,
                       "Bending constraint compliance.")
        .def_readwrite("twist_compliance", &StrandComponent::twistCompliance,
                       "Torsional twisting compliance.")
        .def_readwrite("distance_compliance", &StrandComponent::distanceCompliance,
                       "Distance constraint compliance.")
        .def_readwrite("root_material_normal", &StrandComponent::rootMaterialNormal,
                       "Normal vector at strand root constraint.")
        .def_readwrite("self_collision_enabled", &StrandComponent::selfCollisionEnabled,
                       "Enable strand self-collision.")
        .def_readwrite("suturing_enabled", &StrandComponent::suturingEnabled,
                       "Enable suturing path tracking.")
        .def_readwrite("path_node_spacing", &StrandComponent::pathNodeSpacing,
                       "Node spacing along suturing path.")
        .def_readwrite("collision_layer", &StrandComponent::collisionLayer,
                       "Collision layer.")
        .def_readwrite("collision_mask", &StrandComponent::collisionMask,
                       "Collision mask.");

    py::class_<FluidComponent>(m, "FluidComponent",
                               "Particle-based fluid simulation component (SPH / Position-Based Fluids).")
        .def(py::init<>(), "Initializes the default fluid component.")
        .def_readwrite("source", &FluidComponent::source,
                       "Fluid particle emitter/initializer source descriptor.")
        .def_readwrite("material", &FluidComponent::material,
                       "Fluid material properties (viscosity, surface tension).")
        .def_readwrite("visual_color", &FluidComponent::visualColor,
                       "Visual RGBA color for fluid rendering.")
        .def_readwrite("particle_mass", &FluidComponent::particleMass, "Fluid particle mass.")
        .def_readwrite("particle_radius", &FluidComponent::particleRadius,
                       "Particle interaction radius.")
        .def_readwrite("collision_layer", &FluidComponent::collisionLayer,
                       "Collision layer.")
        .def_readwrite("collision_mask", &FluidComponent::collisionMask,
                       "Collision mask.");

    py::class_<UltrasoundProbeComponent>(
        m, "UltrasoundProbeComponent",
        "Ultrasound transducer probe component for simulated B-mode ultrasound imaging.")
        .def(py::init<>(), "Initializes the default ultrasound probe component.")
        .def_readwrite("enabled", &UltrasoundProbeComponent::enabled,
                       "Enable ultrasound beam simulation.")
        .def_readwrite("geometry", &UltrasoundProbeComponent::geometry,
                       "Transducer array geometry.")
        .def_readwrite("num_scanlines", &UltrasoundProbeComponent::numScanlines,
                       "Number of acoustic scanlines.")
        .def_readwrite("line_length", &UltrasoundProbeComponent::lineLength,
                       "Scanline penetration depth.")
        .def_readwrite("scanline_spacing", &UltrasoundProbeComponent::scanlineSpacing,
                       "Spacing between adjacent scanlines.")
        .def_readwrite("sector_angle_degrees", &UltrasoundProbeComponent::sectorAngleDegrees,
                       "Sector sweep angle for curvilinear arrays.")
        .def_readwrite("probe_radius", &UltrasoundProbeComponent::probeRadius,
                       "Physical transducer head curvature radius.")
        .def_readwrite("sound_speed", &UltrasoundProbeComponent::soundSpeed,
                       "Acoustic speed of sound in medium (m/s).")
        .def_readwrite("world_units_per_meter", &UltrasoundProbeComponent::worldUnitsPerMeter,
                       "World unit scaling factor per meter.")
        .def_readwrite("noise_amplitude", &UltrasoundProbeComponent::noiseAmplitude,
                       "Thermal and speckle noise amplitude.")
        .def_readwrite("sampling_frequency", &UltrasoundProbeComponent::samplingFrequency,
                       "Acoustic RF signal sampling frequency (Hz).")
        .def_readwrite("demodulation_frequency", &UltrasoundProbeComponent::demodulationFrequency,
                       "RF demodulation carrier frequency (Hz).")
        .def_readwrite("center_frequency", &UltrasoundProbeComponent::centerFrequency,
                       "Transducer center frequency (Hz).")
        .def_readwrite("fractional_bandwidth", &UltrasoundProbeComponent::fractionalBandwidth,
                       "Transducer fractional bandwidth.")
        .def_readwrite("beam_sigma_lateral", &UltrasoundProbeComponent::beamSigmaLateral,
                       "Lateral acoustic beam profile Gaussian sigma.")
        .def_readwrite("beam_sigma_elevational", &UltrasoundProbeComponent::beamSigmaElevational,
                       "Elevational acoustic beam profile Gaussian sigma.")
        .def_readwrite("radial_decimation", &UltrasoundProbeComponent::radialDecimation,
                       "Radial decimation factor for image generation.")
        .def_readwrite("threads_per_block", &UltrasoundProbeComponent::threadsPerBlock,
                       "CUDA compute block thread size.")
        .def_readwrite("cuda_num_streams", &UltrasoundProbeComponent::cudaNumStreams,
                       "Number of concurrent CUDA streams.")
        .def_readwrite("num_time_samples", &UltrasoundProbeComponent::numTimeSamples,
                       "RF time-domain samples per scanline.")
        .def_readwrite("use_arc_projection", &UltrasoundProbeComponent::useArcProjection,
                       "Enable arc projection geometry.")
        .def_readwrite("enable_phase_delay", &UltrasoundProbeComponent::enablePhaseDelay,
                       "Enable phase delay beamforming.");

    py::class_<UltrasoundRendererComponent>(m, "UltrasoundRendererComponent")
        .def(py::init<>())
        .def_readwrite("enabled", &UltrasoundRendererComponent::enabled)
        .def_readwrite("output", &UltrasoundRendererComponent::output)
        .def_readwrite("output_width", &UltrasoundRendererComponent::outputWidth)
        .def_readwrite("output_height", &UltrasoundRendererComponent::outputHeight)
        .def_readwrite("use_fixed_max_normalization",
                       &UltrasoundRendererComponent::useFixedMaxNormalization)
        .def_readwrite("fixed_max_signal", &UltrasoundRendererComponent::fixedMaxSignal);

    py::class_<UltrasoundProbeLayout>(m, "UltrasoundProbeLayout")
        .def(py::init<>())
        .def_readwrite("num_scanlines", &UltrasoundProbeLayout::numScanlines)
        .def_readwrite("samples_per_scanline", &UltrasoundProbeLayout::samplesPerScanline)
        .def_readwrite("image_width", &UltrasoundProbeLayout::imageWidth)
        .def_readwrite("image_height", &UltrasoundProbeLayout::imageHeight)
        .def_readwrite("color_format", &UltrasoundProbeLayout::colorFormat)
        .def_readwrite("layered_output_supported", &UltrasoundProbeLayout::layeredOutputSupported);

    py::class_<UltrasoundAmplitudeRange>(m, "UltrasoundAmplitudeRange")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("minimum"), py::arg("maximum"))
        .def_readwrite("minimum", &UltrasoundAmplitudeRange::minimum)
        .def_readwrite("maximum", &UltrasoundAmplitudeRange::maximum);

    py::class_<UltrasoundScattererSourceComponent>(m, "UltrasoundScattererSourceComponent")
        .def(py::init<>())
        .def_readwrite("enabled", &UltrasoundScattererSourceComponent::enabled)
        .def_readwrite("density", &UltrasoundScattererSourceComponent::density)
        .def_readwrite("point_distance_override",
                       &UltrasoundScattererSourceComponent::pointDistanceOverride);

    py::class_<UltrasoundProbeResult>(m, "UltrasoundProbeResult")
        .def(py::init<>())
        .def_readwrite("prepared", &UltrasoundProbeResult::prepared)
        .def_readwrite("completed", &UltrasoundProbeResult::completed)
        .def_readwrite("completed_frame_index", &UltrasoundProbeResult::completedFrameIndex)
        .def_readwrite("num_scanlines", &UltrasoundProbeResult::numScanlines)
        .def_readwrite("samples_per_scanline", &UltrasoundProbeResult::samplesPerScanline)
        .def_readwrite("total_scatterer_count", &UltrasoundProbeResult::totalScattererCount)
        .def_readwrite("image_width", &UltrasoundProbeResult::imageWidth)
        .def_readwrite("image_height", &UltrasoundProbeResult::imageHeight)
        .def_readwrite("image_binding", &UltrasoundProbeResult::imageBinding);

    py::class_<ProceduralDeformableCurveRenderComponent>(
        m, "ProceduralDeformableCurveRenderComponent",
        "Procedural render component for generating tube meshes along deformable curves (strands/sutures).")
        .def(py::init<>(), "Initializes the default procedural deformable-curve render component.")
        .def_readwrite("sequence_id", &ProceduralDeformableCurveRenderComponent::sequenceId,
                       "Target particle sequence ID.")
        .def_readwrite("radius", &ProceduralDeformableCurveRenderComponent::radius,
                       "Tube mesh cross-section radius.")
        .def_readwrite("radial_resolution",
                       &ProceduralDeformableCurveRenderComponent::radialResolution,
                       "Number of radial tube segments.")
        .def_readwrite("enabled", &ProceduralDeformableCurveRenderComponent::enabled,
                       "Enable rendering.");

    py::class_<SoftBodyAuthoringParticles>(m, "SoftBodyAuthoringParticles")
        .def(py::init<>())
        .def_readwrite("particle_count", &SoftBodyAuthoringParticles::particleCount)
        .def_readwrite("rest_positions", &SoftBodyAuthoringParticles::restPositions);

    py::class_<AuthoredParticleReference>(m, "AuthoredParticleReference")
        .def(py::init<>())
        .def_readwrite("entity_id", &AuthoredParticleReference::entityId)
        .def_readwrite("type", &AuthoredParticleReference::type)
        .def_readwrite("local_particle_index", &AuthoredParticleReference::localParticleIndex);

    py::class_<AuthoredParticleDistanceConstraintState>(m,
                                                        "AuthoredParticleDistanceConstraintState")
        .def(py::init<>())
        .def_readwrite("constraint_id", &AuthoredParticleDistanceConstraintState::constraintId)
        .def_readwrite("particle_a", &AuthoredParticleDistanceConstraintState::particleA)
        .def_readwrite("particle_b", &AuthoredParticleDistanceConstraintState::particleB)
        .def_readwrite("rest_length", &AuthoredParticleDistanceConstraintState::restLength)
        .def_readwrite("compliance", &AuthoredParticleDistanceConstraintState::compliance)
        .def_readwrite("enabled", &AuthoredParticleDistanceConstraintState::enabled);

    py::class_<AuthoredParticleSequenceState>(m, "AuthoredParticleSequenceState")
        .def(py::init<>())
        .def_readwrite("sequence_id", &AuthoredParticleSequenceState::sequenceId)
        .def_readwrite("entries", &AuthoredParticleSequenceState::entries)
        .def_readwrite("enabled", &AuthoredParticleSequenceState::enabled);

    py::class_<AuthoredParticleCollisionFilterState>(m, "AuthoredParticleCollisionFilterState")
        .def(py::init<>())
        .def_readwrite("filter_id", &AuthoredParticleCollisionFilterState::filterId)
        .def_readwrite("particle", &AuthoredParticleCollisionFilterState::particle)
        .def_readwrite("collision_layer", &AuthoredParticleCollisionFilterState::collisionLayer)
        .def_readwrite("collision_mask", &AuthoredParticleCollisionFilterState::collisionMask)
        .def_readwrite("enabled", &AuthoredParticleCollisionFilterState::enabled);

    py::class_<AuthoredSuturingSequenceState>(m, "AuthoredSuturingSequenceState")
        .def(py::init<>())
        .def_readwrite("sequence_id", &AuthoredSuturingSequenceState::sequenceId)
        .def_readwrite("entries", &AuthoredSuturingSequenceState::entries)
        .def_readwrite("tip_entry_index", &AuthoredSuturingSequenceState::tipEntryIndex)
        .def_readwrite("path_node_spacing", &AuthoredSuturingSequenceState::pathNodeSpacing)
        .def_readwrite("enabled", &AuthoredSuturingSequenceState::enabled);

    py::class_<AuthoredRigidParticleAttachmentConstraintState>(
        m, "AuthoredRigidParticleAttachmentConstraintState")
        .def(py::init<>())
        .def_readwrite("constraint_id",
                       &AuthoredRigidParticleAttachmentConstraintState::constraintId)
        .def_readwrite("particle", &AuthoredRigidParticleAttachmentConstraintState::particle)
        .def_readwrite("rigid_body_entity_id",
                       &AuthoredRigidParticleAttachmentConstraintState::rigidBodyEntityId)
        .def_readwrite("local_anchor", &AuthoredRigidParticleAttachmentConstraintState::localAnchor)
        .def_readwrite("compliance", &AuthoredRigidParticleAttachmentConstraintState::compliance)
        .def_readwrite("enabled", &AuthoredRigidParticleAttachmentConstraintState::enabled);

    py::class_<AuthoredStrandRigidAttachmentConstraintState>(
        m, "AuthoredStrandRigidAttachmentConstraintState")
        .def(py::init<>())
        .def_readwrite("constraint_id", &AuthoredStrandRigidAttachmentConstraintState::constraintId)
        .def_readwrite("strand_entity_id",
                       &AuthoredStrandRigidAttachmentConstraintState::strandEntityId)
        .def_readwrite("local_segment_index",
                       &AuthoredStrandRigidAttachmentConstraintState::localSegmentIndex)
        .def_readwrite("segment_t", &AuthoredStrandRigidAttachmentConstraintState::segmentT)
        .def_readwrite("rigid_body_entity_id",
                       &AuthoredStrandRigidAttachmentConstraintState::rigidBodyEntityId)
        .def_readwrite("local_anchor", &AuthoredStrandRigidAttachmentConstraintState::localAnchor)
        .def_readwrite("local_rotation",
                       &AuthoredStrandRigidAttachmentConstraintState::localRotation)
        .def_readwrite("translation_compliance",
                       &AuthoredStrandRigidAttachmentConstraintState::translationCompliance)
        .def_readwrite("rotation_compliance",
                       &AuthoredStrandRigidAttachmentConstraintState::rotationCompliance)
        .def_readwrite("enabled", &AuthoredStrandRigidAttachmentConstraintState::enabled);

    py::class_<AuthoredRigidDistanceConstraintState>(m, "AuthoredRigidDistanceConstraintState")
        .def(py::init<>())
        .def_readwrite("constraint_id", &AuthoredRigidDistanceConstraintState::constraintId)
        .def_readwrite("entity_a", &AuthoredRigidDistanceConstraintState::entityA)
        .def_readwrite("entity_b", &AuthoredRigidDistanceConstraintState::entityB)
        .def_readwrite("local_anchor_a", &AuthoredRigidDistanceConstraintState::localAnchorA)
        .def_readwrite("local_anchor_b", &AuthoredRigidDistanceConstraintState::localAnchorB)
        .def_readwrite("rest_distance", &AuthoredRigidDistanceConstraintState::restDistance)
        .def_readwrite("compliance", &AuthoredRigidDistanceConstraintState::compliance)
        .def_readwrite("enabled", &AuthoredRigidDistanceConstraintState::enabled);

    py::class_<AuthoredRoutedCableRoutePoint>(m, "AuthoredRoutedCableRoutePoint")
        .def(py::init<>())
        .def_readwrite("entity_id", &AuthoredRoutedCableRoutePoint::entityId)
        .def_readwrite("local_guide_offset", &AuthoredRoutedCableRoutePoint::localGuideOffset);

    py::class_<AuthoredRoutedCableConstraintState>(m, "AuthoredRoutedCableConstraintState")
        .def(py::init<>())
        .def_readwrite("constraint_id", &AuthoredRoutedCableConstraintState::constraintId)
        .def_readwrite("route_points", &AuthoredRoutedCableConstraintState::routePoints)
        .def_readwrite("target_length", &AuthoredRoutedCableConstraintState::targetLength)
        .def_readwrite("compliance", &AuthoredRoutedCableConstraintState::compliance)
        .def_readwrite("tension_only", &AuthoredRoutedCableConstraintState::tensionOnly)
        .def_readwrite("enabled", &AuthoredRoutedCableConstraintState::enabled);

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

    py::class_<RenderResourceManager>(m, "RenderResourceManager",
                                      "Registry of mesh, material, and texture resources.")
        .def("register_mesh", &RenderResourceManager::registerMesh,
             "Registers a mesh and returns its handle. Registration copies the description, "
             "normalizes normals, generates or repairs tangents, and computes local bounds.")
        .def("register_material", &RenderResourceManager::registerMaterial,
             "Registers a material and returns its handle. Registration copies and normalizes "
             "the material feature flags.")
        .def("register_texture", &RenderResourceManager::registerTexture,
             "Registers a texture and returns its handle. Registration copies the description, "
             "clamps dimensions and mip count to at least one, and expands convenience pixel data.")
        .def("is_valid_mesh",
             py::overload_cast<MeshHandle>(&RenderResourceManager::isValid, py::const_),
             "Returns whether a mesh handle identifies a registered mesh.")
        .def("is_valid_material",
             py::overload_cast<MaterialHandle>(&RenderResourceManager::isValid, py::const_),
             "Returns whether a material handle identifies a registered material.")
        .def("is_valid_texture",
             py::overload_cast<TextureHandle>(&RenderResourceManager::isValid, py::const_),
             "Returns whether a texture handle identifies a registered texture.")
        .def("try_get_mesh",
             [](const RenderResourceManager &resources, const MeshHandle mesh) -> py::object
             {
                 if (const auto *desc = resources.tryGetMesh(mesh))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             },
             "Returns a copy of a registered mesh description, or None for an invalid handle.")
        .def("try_get_material",
             [](const RenderResourceManager &resources, const MaterialHandle material) -> py::object
             {
                 if (const auto *desc = resources.tryGetMaterial(material))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             },
             "Returns a copy of a registered material description, or None for an invalid handle.")
        .def("try_get_texture",
             [](const RenderResourceManager &resources, const TextureHandle texture) -> py::object
             {
                 if (const auto *desc = resources.tryGetTexture(texture))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             },
             "Returns a copy of a registered texture description, or None for an invalid handle.")
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
             },
             "Returns (minimum, maximum) local bounds for a non-empty registered mesh, or None.")
        .def("mesh_version", &RenderResourceManager::meshVersion,
             "Returns the mesh version, or 0 for an invalid handle.");

    py::class_<World>(m, "World")
        .def("create_entity", &World::createEntity, py::arg("env_index") = 0u)
        .def("destroy_entity", &World::destroyEntity)
        .def("set_scene_layout", &World::setSceneLayout)
        .def("scene_layout", &World::sceneLayout, py::return_value_policy::reference_internal)
        .def("set_entity_environment", &World::setEntityEnvironment)
        .def("entity_environment", &World::entityEnvironment)
        .def("set_environment_ibl", &World::setEnvironmentIbl)
        .def("set_environment_fluid", &World::setEnvironmentFluid)
        .def("try_get_environment_ibl",
             [](const World &world, const std::uint32_t envIndex) -> py::object
             {
                 if (const auto *desc = world.tryGetEnvironmentIbl(envIndex))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             })
        .def("try_get_environment_fluid",
             [](const World &world, const std::uint32_t envIndex) -> py::object
             {
                 if (const auto *desc = world.tryGetEnvironmentFluid(envIndex))
                 {
                     return py::cast(*desc);
                 }
                 return py::none();
             })
        .def("is_alive", &World::isAlive)
        .def("entities", &World::entities, py::return_value_policy::reference_internal)
        .def("set_transform", &World::setTransform)
        .def("remove_transform", &World::removeTransform)
        .def("try_get_transform", &World::tryGetTransform)
        .def("entity_pose_slot", &World::entityPoseSlot)
        .def("set_mesh_renderer", &World::setMeshRenderer)
        .def("remove_mesh_renderer", &World::removeMeshRenderer)
        .def("try_get_mesh_renderer", &World::tryGetMeshRenderer)
        .def("set_camera", &World::setCamera)
        .def("remove_camera", &World::removeCamera)
        .def("try_get_camera", &World::tryGetCamera)
        .def("set_directional_light", &World::setDirectionalLight)
        .def("remove_directional_light", &World::removeDirectionalLight)
        .def("try_get_directional_light", &World::tryGetDirectionalLight)
        .def("set_point_light", &World::setPointLight)
        .def("remove_point_light", &World::removePointLight)
        .def("try_get_point_light", &World::tryGetPointLight)
        .def("set_spot_light", &World::setSpotLight)
        .def("remove_spot_light", &World::removeSpotLight)
        .def("try_get_spot_light", &World::tryGetSpotLight)
        .def("set_rigid_body", &World::setRigidBody)
        .def("remove_rigid_body", &World::removeRigidBody)
        .def("try_get_rigid_body", &World::tryGetRigidBody)
        .def("set_soft_body", &World::setSoftBody)
        .def("set_meshfree_soft_body", &World::setMeshfreeSoftBody)
        .def("remove_soft_body", &World::removeSoftBody)
        .def("try_get_soft_body", &World::tryGetSoftBody)
        .def("set_strand", &World::setStrand)
        .def("remove_strand", &World::removeStrand)
        .def("try_get_strand", &World::tryGetStrand)
        .def("set_procedural_deformable_curve_render", &World::setProceduralDeformableCurveRender)
        .def("remove_procedural_deformable_curve_render",
             &World::removeProceduralDeformableCurveRender)
        .def("try_get_procedural_deformable_curve_render",
             &World::tryGetProceduralDeformableCurveRender)
        .def("set_fluid", &World::setFluid)
        .def("remove_fluid", &World::removeFluid)
        .def("try_get_fluid", &World::tryGetFluid)
        .def("set_ultrasound_probe", &World::setUltrasoundProbe)
        .def("remove_ultrasound_probe", &World::removeUltrasoundProbe)
        .def("try_get_ultrasound_probe", &World::tryGetUltrasoundProbe)
        .def("set_ultrasound_renderer", &World::setUltrasoundRenderer)
        .def("remove_ultrasound_renderer", &World::removeUltrasoundRenderer)
        .def("try_get_ultrasound_renderer", &World::tryGetUltrasoundRenderer)
        .def("set_ultrasound_scatterer_source", &World::setUltrasoundScattererSource)
        .def("remove_ultrasound_scatterer_source", &World::removeUltrasoundScattererSource)
        .def("try_get_ultrasound_scatterer_source", &World::tryGetUltrasoundScattererSource)
        .def("set_ultrasound_scatterer_amplitude_ranges",
             &World::setUltrasoundScattererAmplitudeRanges)
        .def("clear_ultrasound_scatterer_amplitude_ranges",
             &World::clearUltrasoundScattererAmplitudeRanges)
        .def("try_get_ultrasound_scatterer_amplitude_ranges",
             [](const World &world, const cressim::neo::common::EntityId entityId) -> py::object
             {
                 if (const auto *ranges = world.tryGetUltrasoundScattererAmplitudeRanges(entityId))
                 {
                     return py::cast(*ranges);
                 }
                 return py::none();
             })
        .def("try_get_ultrasound_probe_result",
             [](const World &world, const cressim::neo::common::EntityId entityId) -> py::object
             {
                 if (const auto *result = world.tryGetUltrasoundProbeResult(entityId))
                 {
                     return py::cast(*result);
                 }
                 return py::none();
             })
        .def("upsert_particle_sequence", &World::upsertParticleSequence,
             py::return_value_policy::reference_internal)
        .def("remove_particle_sequence", &World::removeParticleSequence)
        .def("try_get_particle_sequence",
             [](const World &world,
                const cressim::neo::physics::ParticleSequenceId sequenceId) -> py::object
             {
                 if (const auto *state = world.tryGetParticleSequence(sequenceId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_particle_distance_constraint", &World::upsertParticleDistanceConstraint,
             py::return_value_policy::reference_internal)
        .def("remove_particle_distance_constraint", &World::removeParticleDistanceConstraint)
        .def("try_get_particle_distance_constraint",
             [](const World &world,
                const cressim::neo::physics::ParticleConstraintId constraintId) -> py::object
             {
                 if (const auto *state = world.tryGetParticleDistanceConstraint(constraintId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_ball_joint", &World::upsertBallJoint)
        .def("remove_ball_joint", &World::removeBallJoint)
        .def("try_get_ball_joint",
             [](const World &world, const cressim::neo::physics::BallJointId jointId) -> py::object
             {
                 if (const auto *state = world.tryGetBallJoint(jointId))
                 {
                     return py::cast(remapJointBodiesToEntityIds(world, *state));
                 }
                 return py::none();
             })
        .def("upsert_hinge_joint", &World::upsertHingeJoint)
        .def("remove_hinge_joint", &World::removeHingeJoint)
        .def("try_get_hinge_joint",
             [](const World &world, const cressim::neo::physics::HingeJointId jointId) -> py::object
             {
                 if (const auto *state = world.tryGetHingeJoint(jointId))
                 {
                     return py::cast(remapJointBodiesToEntityIds(world, *state));
                 }
                 return py::none();
             })
        .def("upsert_spherical_joint", &World::upsertSphericalJoint)
        .def("remove_spherical_joint", &World::removeSphericalJoint)
        .def("try_get_spherical_joint",
             [](const World &world,
                const cressim::neo::physics::SphericalJointId jointId) -> py::object
             {
                 if (const auto *state = world.tryGetSphericalJoint(jointId))
                 {
                     return py::cast(remapJointBodiesToEntityIds(world, *state));
                 }
                 return py::none();
             })
        .def("upsert_slider_joint", &World::upsertSliderJoint)
        .def("remove_slider_joint", &World::removeSliderJoint)
        .def(
            "try_get_slider_joint",
            [](const World &world, const cressim::neo::physics::SliderJointId jointId) -> py::object
            {
                if (const auto *state = world.tryGetSliderJoint(jointId))
                {
                    return py::cast(remapJointBodiesToEntityIds(world, *state));
                }
                return py::none();
            })
        .def("upsert_rigid_particle_attachment_constraint",
             [](World &world, const AuthoredRigidParticleAttachmentConstraintState &state)
             { return world.upsertRigidParticleAttachmentConstraint(state); })
        .def("remove_rigid_particle_attachment_constraint",
             &World::removeRigidParticleAttachmentConstraint)
        .def("try_get_rigid_particle_attachment_constraint",
             [](const World &world,
                const cressim::neo::physics::RigidParticleAttachmentConstraintId constraintId)
                 -> py::object
             {
                 if (const auto *state =
                         world.tryGetRigidParticleAttachmentConstraint(constraintId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_strand_rigid_attachment_constraint",
             [](World &world, const AuthoredStrandRigidAttachmentConstraintState &state)
             { return world.upsertStrandRigidAttachmentConstraint(state); })
        .def("remove_strand_rigid_attachment_constraint",
             &World::removeStrandRigidAttachmentConstraint)
        .def("try_get_strand_rigid_attachment_constraint",
             [](const World &world,
                const cressim::neo::physics::StrandRigidAttachmentConstraintId constraintId)
                 -> py::object
             {
                 if (const auto *state = world.tryGetStrandRigidAttachmentConstraint(constraintId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_rigid_distance_constraint",
             [](World &world, const AuthoredRigidDistanceConstraintState &state)
             { return world.upsertRigidDistanceConstraint(state); })
        .def("remove_rigid_distance_constraint", &World::removeRigidDistanceConstraint)
        .def("try_get_rigid_distance_constraint",
             [](const World &world,
                const cressim::neo::physics::RigidDistanceConstraintId constraintId) -> py::object
             {
                 if (const auto *state = world.tryGetRigidDistanceConstraint(constraintId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_routed_cable_constraint",
             [](World &world, const AuthoredRoutedCableConstraintState &state)
             { return world.upsertRoutedCableConstraint(state); })
        .def("remove_routed_cable_constraint", &World::removeRoutedCableConstraint)
        .def("try_get_routed_cable_constraint",
             [](const World &world,
                const cressim::neo::physics::RoutedCableConstraintId constraintId) -> py::object
             {
                 if (const auto *state = world.tryGetRoutedCableConstraint(constraintId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_particle_collision_filter", &World::upsertParticleCollisionFilter,
             py::return_value_policy::reference_internal)
        .def("remove_particle_collision_filter", &World::removeParticleCollisionFilter)
        .def("try_get_particle_collision_filter",
             [](const World &world,
                const cressim::neo::physics::ParticleCollisionFilterId filterId) -> py::object
             {
                 if (const auto *state = world.tryGetParticleCollisionFilter(filterId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("upsert_suturing_sequence", &World::upsertSuturingSequence,
             py::return_value_policy::reference_internal)
        .def("remove_suturing_sequence", &World::removeSuturingSequence)
        .def("try_get_suturing_sequence",
             [](const World &world,
                const cressim::neo::physics::SuturingSequenceId sequenceId) -> py::object
             {
                 if (const auto *state = world.tryGetSuturingSequence(sequenceId))
                 {
                     return py::cast(*state);
                 }
                 return py::none();
             })
        .def("add_collider", &World::addCollider)
        .def("update_collider", &World::updateCollider)
        .def("remove_collider", &World::removeCollider)
        .def("replace_colliders", &World::replaceColliders)
        .def("try_get_collider", &World::tryGetCollider)
        .def("collider_handles", &World::colliderHandles,
             py::return_value_policy::reference_internal)
        .def("try_get_soft_body_authoring_particles", &World::tryGetSoftBodyAuthoringParticles);

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
            "Initializes the engine runtime with the specified configuration parameters.\n\n"
            "Args:\n"
            "    config: Runtime configuration options including GPU, physics, and renderer settings. ``None`` uses the default configuration.\n\n"
            "Returns:\n"
            "    True if initialization succeeds; false otherwise.",
            py::arg("config") = py::none())
        .def("shutdown", &Runtime::shutdown)
        .def("get_info", &Runtime::getInfo,
             "Returns engine version and optional feature support information.")
        .def("set_gravity", &Runtime::setGravity, py::arg("gravity"))
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
        .def("get_prepared_rigid_layout_mapping",
             [](Runtime &runtime)
             {
                 RigidLayoutMapping mapping{};
                 if (!runtime.tryGetPreparedRigidLayoutMapping(mapping))
                 {
                     throw std::runtime_error("Prepared rigid layout mapping is unavailable.");
                 }
                 return mapping;
             })
        .def("get_prepared_constraint_layout_mapping",
             [](Runtime &runtime)
             {
                 ConstraintLayoutMapping mapping{};
                 if (!runtime.tryGetPreparedConstraintLayoutMapping(mapping))
                 {
                     throw std::runtime_error("Prepared constraint layout mapping is unavailable.");
                 }
                 return mapping;
             })
        .def("get_prepared_particle_layout_mapping",
             [](Runtime &runtime)
             {
                 ParticleLayoutMapping mapping{};
                 if (!runtime.tryGetPreparedParticleLayoutMapping(mapping))
                 {
                     throw std::runtime_error("Prepared particle layout mapping is unavailable.");
                 }
                 return mapping;
             })
        .def("get_prepared_joint_layout_mapping",
             [](Runtime &runtime)
             {
                 JointLayoutMapping mapping{};
                 if (!runtime.tryGetPreparedJointLayoutMapping(mapping))
                 {
                     throw std::runtime_error("Prepared joint layout mapping is unavailable.");
                 }
                 return mapping;
             })
        .def("shared_buffer_to_dlpack", &exportSharedBufferToDLPack)
        .def("step_physics", &Runtime::stepPhysics)
        .def("step_simulation_sensors", &Runtime::stepSimulationSensors)
        .def("step_visual_sensors", &Runtime::stepVisualSensors)
        .def("end_frame", &Runtime::endFrame)
        .def("list_custom_compute_resources", &Runtime::listCustomComputeResources,
             "Lists custom compute resources registered for the uploaded world. Returns an empty list when unavailable.")
        .def("create_custom_compute_pass", &Runtime::createCustomComputePass,
             "Compiles and registers a custom compute pass for the uploaded world. Returns an invalid handle if creation fails.",
             py::arg("desc"))
        .def("update_custom_compute_pass_constants",
             [](Runtime &runtime, const CustomComputePassHandle handle, py::bytes data)
             {
                 std::string bytes = data;
                 return runtime.updateCustomComputePassConstants(
                     handle, std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
             },
             "Updates a custom compute pass's constant-buffer data. Returns ``False`` for an invalid handle, a pass without constants, or an oversized payload.",
             py::arg("handle"), py::arg("data"))
        .def("execute_custom_compute_pass", &Runtime::executeCustomComputePass,
             "Executes a registered custom compute pass for the uploaded world. Returns ``False`` if the pass or its required resources are unavailable or changed.",
             py::arg("handle"))
        .def("destroy_custom_compute_pass", &Runtime::destroyCustomComputePass,
             "Destroys a registered custom compute pass. Returns ``True`` if the handle was registered.",
             py::arg("handle"))
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
        .def("try_get_render_target_readback", &tryGetRenderTargetReadback)
        .def(
            "compute_ultrasound_probe_layout",
            [](const Runtime &runtime, const UltrasoundProbeComponent &probe,
               const UltrasoundRendererComponent &renderer)
            {
                UltrasoundProbeLayout layout{};
                if (!runtime.computeUltrasoundProbeLayout(probe, renderer, layout))
                {
                    throw std::runtime_error("Failed to compute ultrasound probe layout.");
                }
                return layout;
            },
            py::arg("probe"), py::arg("renderer"));

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
