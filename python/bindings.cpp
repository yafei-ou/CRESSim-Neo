#include "common/frame_context.h"
#include "common/math_types.h"
#include "engine/components.h"
#include "engine/constraint_layout_mapping.h"
#include "engine/joint_layout_mapping.h"
#include "engine/particle_layout_mapping.h"
#include "engine/rigid_layout_mapping.h"
#include "engine/runtime.h"
#include "engine/runtime_internal.h"
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
    std::shared_ptr<void> sharedBufferLease;
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

    py::class_<Diligent::uint3>(m, "UInt3")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t>(), py::arg("x") = 0u,
             py::arg("y") = 0u, py::arg("z") = 0u)
        .def_readwrite("x", &Diligent::uint3::x)
        .def_readwrite("y", &Diligent::uint3::y)
        .def_readwrite("z", &Diligent::uint3::z);

    py::class_<Diligent::uint4>(m, "UInt4")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>(),
             py::arg("x") = 0u, py::arg("y") = 0u, py::arg("z") = 0u, py::arg("w") = 0u)
        .def_readwrite("x", &Diligent::uint4::x)
        .def_readwrite("y", &Diligent::uint4::y)
        .def_readwrite("z", &Diligent::uint4::z)
        .def_readwrite("w", &Diligent::uint4::w);

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
        .value("Buffer", CustomComputeResourceKind::Buffer)
        .value("Texture", CustomComputeResourceKind::Texture);

    py::enum_<CustomComputeResourceAccess>(m, "CustomComputeResourceAccess")
        .value("ReadOnly", CustomComputeResourceAccess::ReadOnly)
        .value("WriteOnly", CustomComputeResourceAccess::WriteOnly)
        .value("ReadWrite", CustomComputeResourceAccess::ReadWrite);

    py::enum_<GpuRenderTargetTexturePlane>(m, "GpuRenderTargetTexturePlane")
        .value("Color", GpuRenderTargetTexturePlane::Color)
        .value("Depth", GpuRenderTargetTexturePlane::Depth);

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

    py::class_<RigidLayoutMapping>(m, "RigidLayoutMapping")
        .def(py::init<>())
        .def_readwrite("rigid_body_count", &RigidLayoutMapping::rigidBodyCount)
        .def_readwrite("collider_count", &RigidLayoutMapping::colliderCount)
        .def_readwrite("layout_revision", &RigidLayoutMapping::layoutRevision)
        .def_readwrite("rigid_body_ids", &RigidLayoutMapping::rigidBodyIds)
        .def_readwrite("rigid_body_entity_ids", &RigidLayoutMapping::rigidBodyEntityIds)
        .def_readwrite("rigid_body_environment_indices",
                       &RigidLayoutMapping::rigidBodyEnvironmentIndices)
        .def_readwrite("collider_ids", &RigidLayoutMapping::colliderIds)
        .def_readwrite("collider_entity_ids", &RigidLayoutMapping::colliderEntityIds)
        .def_readwrite("collider_owner_body_ids", &RigidLayoutMapping::colliderOwnerBodyIds)
        .def_readwrite("collider_owner_body_indices", &RigidLayoutMapping::colliderOwnerBodyIndices)
        .def_readwrite("collider_environment_indices",
                       &RigidLayoutMapping::colliderEnvironmentIndices)
        .def_readwrite("collider_shape_types", &RigidLayoutMapping::colliderShapeTypes)
        .def_readwrite("collider_enabled", &RigidLayoutMapping::colliderEnabledFlags)
        .def_readwrite("collider_collision_layers", &RigidLayoutMapping::colliderCollisionLayers)
        .def_readwrite("collider_collision_masks", &RigidLayoutMapping::colliderCollisionMasks)
        .def_readwrite("collider_local_positions", &RigidLayoutMapping::colliderLocalPositions)
        .def_readwrite("collider_local_rotations", &RigidLayoutMapping::colliderLocalRotations)
        .def_readwrite("collider_shape_params", &RigidLayoutMapping::colliderShapeParams)
        .def_readwrite("body_collider_offsets", &RigidLayoutMapping::bodyColliderOffsets)
        .def_readwrite("body_collider_counts", &RigidLayoutMapping::bodyColliderCounts)
        .def_readwrite("body_collider_indices", &RigidLayoutMapping::bodyColliderIndices);

    py::class_<RigidParticleAttachmentConstraintLayoutMapping>(
        m, "RigidParticleAttachmentConstraintLayoutMapping")
        .def(py::init<>())
        .def_readwrite("count", &RigidParticleAttachmentConstraintLayoutMapping::count)
        .def_readwrite("constraint_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::constraintIds)
        .def_readwrite("environment_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::environmentIndices)
        .def_readwrite("rigid_body_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::rigidBodyIds)
        .def_readwrite("rigid_body_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::rigidBodyIndices)
        .def_readwrite("particle_entity_ids",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleEntityIds)
        .def_readwrite("particle_reference_types",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleReferenceTypes)
        .def_readwrite("particle_local_indices",
                       &RigidParticleAttachmentConstraintLayoutMapping::particleLocalIndices)
        .def_readwrite("enabled", &RigidParticleAttachmentConstraintLayoutMapping::enabledFlags);

    py::class_<RigidDistanceConstraintLayoutMapping>(m, "RigidDistanceConstraintLayoutMapping")
        .def(py::init<>())
        .def_readwrite("count", &RigidDistanceConstraintLayoutMapping::count)
        .def_readwrite("constraint_ids", &RigidDistanceConstraintLayoutMapping::constraintIds)
        .def_readwrite("environment_indices",
                       &RigidDistanceConstraintLayoutMapping::environmentIndices)
        .def_readwrite("rigid_body_ids_a", &RigidDistanceConstraintLayoutMapping::rigidBodyIdsA)
        .def_readwrite("rigid_body_ids_b", &RigidDistanceConstraintLayoutMapping::rigidBodyIdsB)
        .def_readwrite("rigid_body_indices_a",
                       &RigidDistanceConstraintLayoutMapping::rigidBodyIndicesA)
        .def_readwrite("rigid_body_indices_b",
                       &RigidDistanceConstraintLayoutMapping::rigidBodyIndicesB)
        .def_readwrite("enabled", &RigidDistanceConstraintLayoutMapping::enabledFlags);

    py::class_<RoutedCableConstraintLayoutMapping>(m, "RoutedCableConstraintLayoutMapping")
        .def(py::init<>())
        .def_readwrite("count", &RoutedCableConstraintLayoutMapping::count)
        .def_readwrite("constraint_ids", &RoutedCableConstraintLayoutMapping::constraintIds)
        .def_readwrite("environment_indices",
                       &RoutedCableConstraintLayoutMapping::environmentIndices)
        .def_readwrite("route_point_offsets",
                       &RoutedCableConstraintLayoutMapping::routePointOffsets)
        .def_readwrite("route_point_counts", &RoutedCableConstraintLayoutMapping::routePointCounts)
        .def_readwrite("enabled", &RoutedCableConstraintLayoutMapping::enabledFlags)
        .def_readwrite("route_point_rigid_body_ids",
                       &RoutedCableConstraintLayoutMapping::routePointRigidBodyIds)
        .def_readwrite("route_point_rigid_body_indices",
                       &RoutedCableConstraintLayoutMapping::routePointRigidBodyIndices)
        .def_readwrite("route_point_local_guide_offsets",
                       &RoutedCableConstraintLayoutMapping::routePointLocalGuideOffsets);

    py::class_<ConstraintLayoutMapping>(m, "ConstraintLayoutMapping")
        .def(py::init<>())
        .def_readwrite("layout_revision", &ConstraintLayoutMapping::layoutRevision)
        .def_readwrite("rigid_particle_attachments",
                       &ConstraintLayoutMapping::rigidParticleAttachments)
        .def_readwrite("rigid_distance_constraints",
                       &ConstraintLayoutMapping::rigidDistanceConstraints)
        .def_readwrite("routed_cables", &ConstraintLayoutMapping::routedCables);

    py::class_<JointLayoutMapping>(m, "JointLayoutMapping")
        .def(py::init<>())
        .def_readwrite("ball_joint_count", &JointLayoutMapping::ballJointCount)
        .def_readwrite("hinge_joint_count", &JointLayoutMapping::hingeJointCount)
        .def_readwrite("spherical_joint_count", &JointLayoutMapping::sphericalJointCount)
        .def_readwrite("slider_joint_count", &JointLayoutMapping::sliderJointCount)
        .def_readwrite("layout_revision", &JointLayoutMapping::layoutRevision)
        .def_readwrite("ball_joint_ids", &JointLayoutMapping::ballJointIds)
        .def_readwrite("ball_environment_indices", &JointLayoutMapping::ballEnvironmentIndices)
        .def_readwrite("ball_body_ids_a", &JointLayoutMapping::ballBodyIdsA)
        .def_readwrite("ball_body_ids_b", &JointLayoutMapping::ballBodyIdsB)
        .def_readwrite("ball_body_indices_a", &JointLayoutMapping::ballBodyIndicesA)
        .def_readwrite("ball_body_indices_b", &JointLayoutMapping::ballBodyIndicesB)
        .def_readwrite("hinge_joint_ids", &JointLayoutMapping::hingeJointIds)
        .def_readwrite("hinge_environment_indices", &JointLayoutMapping::hingeEnvironmentIndices)
        .def_readwrite("hinge_body_ids_a", &JointLayoutMapping::hingeBodyIdsA)
        .def_readwrite("hinge_body_ids_b", &JointLayoutMapping::hingeBodyIdsB)
        .def_readwrite("hinge_body_indices_a", &JointLayoutMapping::hingeBodyIndicesA)
        .def_readwrite("hinge_body_indices_b", &JointLayoutMapping::hingeBodyIndicesB)
        .def_readwrite("spherical_joint_ids", &JointLayoutMapping::sphericalJointIds)
        .def_readwrite("spherical_environment_indices",
                       &JointLayoutMapping::sphericalEnvironmentIndices)
        .def_readwrite("spherical_body_ids_a", &JointLayoutMapping::sphericalBodyIdsA)
        .def_readwrite("spherical_body_ids_b", &JointLayoutMapping::sphericalBodyIdsB)
        .def_readwrite("spherical_body_indices_a", &JointLayoutMapping::sphericalBodyIndicesA)
        .def_readwrite("spherical_body_indices_b", &JointLayoutMapping::sphericalBodyIndicesB)
        .def_readwrite("slider_joint_ids", &JointLayoutMapping::sliderJointIds)
        .def_readwrite("slider_environment_indices", &JointLayoutMapping::sliderEnvironmentIndices)
        .def_readwrite("slider_body_ids_a", &JointLayoutMapping::sliderBodyIdsA)
        .def_readwrite("slider_body_ids_b", &JointLayoutMapping::sliderBodyIdsB)
        .def_readwrite("slider_body_indices_a", &JointLayoutMapping::sliderBodyIndicesA)
        .def_readwrite("slider_body_indices_b", &JointLayoutMapping::sliderBodyIndicesB);

    py::class_<ParticleLayoutMapping>(m, "ParticleLayoutMapping")
        .def(py::init<>())
        .def_readwrite("particle_count", &ParticleLayoutMapping::particleCount)
        .def_readwrite("soft_body_count", &ParticleLayoutMapping::softBodyCount)
        .def_readwrite("fluid_count", &ParticleLayoutMapping::fluidCount)
        .def_readwrite("strand_count", &ParticleLayoutMapping::strandCount)
        .def_readwrite("layout_revision", &ParticleLayoutMapping::layoutRevision)
        .def_readwrite("environment_indices", &ParticleLayoutMapping::environmentIndices)
        .def_readwrite("particle_kinds", &ParticleLayoutMapping::particleKinds)
        .def_readwrite("owner_types", &ParticleLayoutMapping::ownerTypes)
        .def_readwrite("owner_indices", &ParticleLayoutMapping::ownerIndices)
        .def_readwrite("strand_ids", &ParticleLayoutMapping::strandIds)
        .def_readwrite("strand_orders", &ParticleLayoutMapping::strandOrders)
        .def_readwrite("strand_roles", &ParticleLayoutMapping::strandRoles)
        .def_readwrite("owning_soft_body_indices", &ParticleLayoutMapping::owningSoftBodyIndices)
        .def_readwrite("particle_material_indices", &ParticleLayoutMapping::particleMaterialIndices)
        .def_readwrite("fluid_material_indices", &ParticleLayoutMapping::fluidMaterialIndices)
        .def_readwrite("phases", &ParticleLayoutMapping::phases)
        .def_readwrite("collision_layers", &ParticleLayoutMapping::collisionLayers)
        .def_readwrite("collision_masks", &ParticleLayoutMapping::collisionMasks)
        .def_readwrite("adjacency_offsets", &ParticleLayoutMapping::adjacencyOffsets)
        .def_readwrite("adjacency_counts", &ParticleLayoutMapping::adjacencyCounts)
        .def_readwrite("soft_body_entity_ids", &ParticleLayoutMapping::softBodyEntityIds)
        .def_readwrite("soft_body_environment_indices",
                       &ParticleLayoutMapping::softBodyEnvironmentIndices)
        .def_readwrite("soft_body_particle_offsets",
                       &ParticleLayoutMapping::softBodyParticleOffsets)
        .def_readwrite("soft_body_particle_counts", &ParticleLayoutMapping::softBodyParticleCounts)
        .def_readwrite("fluid_entity_ids", &ParticleLayoutMapping::fluidEntityIds)
        .def_readwrite("fluid_environment_indices", &ParticleLayoutMapping::fluidEnvironmentIndices)
        .def_readwrite("fluid_particle_offsets", &ParticleLayoutMapping::fluidParticleOffsets)
        .def_readwrite("fluid_particle_counts", &ParticleLayoutMapping::fluidParticleCounts)
        .def_readwrite("strand_entity_ids", &ParticleLayoutMapping::strandEntityIds)
        .def_readwrite("strand_environment_indices",
                       &ParticleLayoutMapping::strandEnvironmentIndices)
        .def_readwrite("strand_particle_offsets", &ParticleLayoutMapping::strandParticleOffsets)
        .def_readwrite("strand_particle_counts", &ParticleLayoutMapping::strandParticleCounts);

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
        .def_readwrite("render_target_binding",
                       &CustomComputeResourceBindingDesc::renderTargetBinding)
        .def_readwrite("render_target_texture_plane",
                       &CustomComputeResourceBindingDesc::renderTargetTexturePlane)
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

    py::enum_<UltrasoundProbeComponent::Geometry>(m, "UltrasoundProbeGeometry")
        .value("Linear", UltrasoundProbeComponent::Geometry::Linear)
        .value("Curvilinear", UltrasoundProbeComponent::Geometry::Curvilinear);

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

    py::enum_<AuthoredParticleReferenceType>(m, "AuthoredParticleReferenceType")
        .value("SoftBodyParticle", AuthoredParticleReferenceType::SoftBodyParticle)
        .value("StrandParticle", AuthoredParticleReferenceType::StrandParticle)
        .value("RigidProxyParticle", AuthoredParticleReferenceType::RigidProxyParticle);

    py::enum_<SoftBodySourceKind>(m, "SoftBodySourceKind")
        .value("RegularGrid", SoftBodySourceKind::RegularGrid)
        .value("TetMesh", SoftBodySourceKind::TetMesh)
        .value("TetGenFiles", SoftBodySourceKind::TetGenFiles)
        .value("MeshfreeParticles", SoftBodySourceKind::MeshfreeParticles);

    py::enum_<FluidSourceKind>(m, "FluidSourceKind")
        .value("RegularGrid", FluidSourceKind::RegularGrid);

    py::enum_<ParticleKind>(m, "ParticleKind")
        .value("SoftSolid", ParticleKind::SoftSolid)
        .value("Fluid", ParticleKind::Fluid);

    py::enum_<ParticleOwnerType>(m, "ParticleOwnerType")
        .value("None", ParticleOwnerType::None)
        .value("SoftBody", ParticleOwnerType::SoftBody)
        .value("FluidBody", ParticleOwnerType::FluidBody)
        .value("Strand", ParticleOwnerType::Strand)
        .value("RigidBody", ParticleOwnerType::RigidBody);

    py::enum_<ParticleStrandRole>(m, "ParticleStrandRole")
        .value("None", ParticleStrandRole::None)
        .value("NeedleTip", ParticleStrandRole::NeedleTip)
        .value("NeedleBody", ParticleStrandRole::NeedleBody)
        .value("Thread", ParticleStrandRole::Thread);

    py::enum_<RigidJointDriveMode>(m, "RigidJointDriveMode")
        .value("None", RigidJointDriveMode::None)
        .value("TargetPosition", RigidJointDriveMode::TargetPosition)
        .value("TargetVelocity", RigidJointDriveMode::TargetVelocity)
        .value("TargetOrientation", RigidJointDriveMode::TargetOrientation);

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

    py::enum_<TextureColorSpace>(m, "TextureColorSpace")
        .value("Linear", TextureColorSpace::Linear)
        .value("Srgb", TextureColorSpace::Srgb);

    py::enum_<TexturePixelFormat>(m, "TexturePixelFormat")
        .value("RGBA8", TexturePixelFormat::RGBA8)
        .value("RGBA16F", TexturePixelFormat::RGBA16F);

    py::enum_<TextureMipPolicy>(m, "TextureMipPolicy")
        .value("Disabled", TextureMipPolicy::Disabled)
        .value("Generate", TextureMipPolicy::Generate);

    py::enum_<TextureDimension>(m, "TextureDimension")
        .value("Texture2D", TextureDimension::Texture2D)
        .value("TextureCube", TextureDimension::TextureCube);

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

#ifdef CRESSIM_NEO_PYTHON_HAS_VIEWER
    py::class_<DebugViewerAppDesc>(m, "DebugViewerAppDesc")
        .def(py::init<>())
        .def_readwrite("window_title", &DebugViewerAppDesc::windowTitle)
        .def_readwrite("width", &DebugViewerAppDesc::width)
        .def_readwrite("height", &DebugViewerAppDesc::height)
        .def_readwrite("window_enabled", &DebugViewerAppDesc::windowEnabled)
        .def_readwrite("window_visible", &DebugViewerAppDesc::windowVisible)
        .def_readwrite("start_fullscreen", &DebugViewerAppDesc::startFullscreen)
        .def_readwrite("start_fullscreen_windowed", &DebugViewerAppDesc::startFullscreenWindowed)
        .def_readwrite("v_sync", &DebugViewerAppDesc::vSync)
        .def_readwrite("input_sensitivity", &DebugViewerAppDesc::inputSensitivity)
        .def_readwrite("move_speed", &DebugViewerAppDesc::moveSpeed)
        .def_readwrite("speed_boost_scale", &DebugViewerAppDesc::speedBoostScale)
        .def_readwrite("speed_slow_scale", &DebugViewerAppDesc::speedSlowScale)
        .def_readwrite("wheel_speed_scale", &DebugViewerAppDesc::wheelSpeedScale)
        .def_readwrite("min_move_speed", &DebugViewerAppDesc::minMoveSpeed)
        .def_readwrite("max_move_speed", &DebugViewerAppDesc::maxMoveSpeed)
        .def_readwrite("fixed_delta_seconds", &DebugViewerAppDesc::fixedDeltaSeconds)
        .def_readwrite("use_fixed_timestep", &DebugViewerAppDesc::useFixedTimestep)
        .def_readwrite("step_simulation", &DebugViewerAppDesc::stepSimulation)
        .def_readwrite("max_frames", &DebugViewerAppDesc::maxFrames)
        .def_readwrite("show_stats", &DebugViewerAppDesc::showStats)
        .def_readwrite("enable_debug_particles", &DebugViewerAppDesc::enableDebugParticles)
        .def_readwrite("stats_interval_frames", &DebugViewerAppDesc::statsIntervalFrames);

    py::class_<DebugViewerCameraBinding>(m, "DebugViewerCameraBinding")
        .def(py::init<>())
        .def_readwrite("camera_entity", &DebugViewerCameraBinding::cameraEntity)
        .def_readwrite("move_speed", &DebugViewerCameraBinding::moveSpeed)
        .def_readwrite("input_sensitivity", &DebugViewerCameraBinding::inputSensitivity)
        .def_readwrite("speed_boost_scale", &DebugViewerCameraBinding::speedBoostScale)
        .def_readwrite("speed_slow_scale", &DebugViewerCameraBinding::speedSlowScale);

    py::class_<DebugViewerCallbacks>(m, "DebugViewerCallbacks")
        .def(py::init<>())
        .def_readwrite("before_tick", &DebugViewerCallbacks::beforeTick)
        .def_readwrite("after_tick", &DebugViewerCallbacks::afterTick);

    py::class_<DebugViewerApp>(m, "DebugViewerApp")
        .def(py::init<>())
        .def("initialize", [](DebugViewerApp &viewer, const DebugViewerAppDesc &desc,
                              RuntimeConfig &config) { return viewer.initialize(desc, config); })
        .def("run",
             [](DebugViewerApp &viewer, Runtime &runtime, const DebugViewerCameraBinding &binding)
             { return viewer.run(runtime, binding); })
        .def("run",
             [](DebugViewerApp &viewer, Runtime &runtime, const DebugViewerCameraBinding &binding,
                const DebugViewerCallbacks &callbacks)
             { return viewer.run(runtime, binding, callbacks); })
        .def("request_exit", &DebugViewerApp::requestExit)
        .def("shutdown", &DebugViewerApp::shutdown);
#endif

    py::class_<RuntimeInfo>(m, "RuntimeInfo")
        .def(py::init<>())
        .def_readwrite("engine_version", &RuntimeInfo::engineVersion)
        .def_readwrite("engine_version_major", &RuntimeInfo::engineVersionMajor)
        .def_readwrite("engine_version_minor", &RuntimeInfo::engineVersionMinor)
        .def_readwrite("engine_version_patch", &RuntimeInfo::engineVersionPatch)
        .def_readwrite("cuda_interop_supported", &RuntimeInfo::cudaInteropSupported)
        .def_readwrite("ultrasound_supported", &RuntimeInfo::ultrasoundSupported);

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

    py::class_<TextureResourceDesc::SubresourceDesc>(m, "TextureSubresourceDesc")
        .def(py::init<>())
        .def_readwrite("pixel_data", &TextureResourceDesc::SubresourceDesc::pixelData);

    py::class_<TextureResourceDesc>(m, "TextureResourceDesc")
        .def(py::init<>())
        .def_readwrite("debug_name", &TextureResourceDesc::debugName)
        .def_readwrite("width", &TextureResourceDesc::width)
        .def_readwrite("height", &TextureResourceDesc::height)
        .def_readwrite("mip_level_count", &TextureResourceDesc::mipLevelCount)
        .def_readwrite("dimension", &TextureResourceDesc::dimension)
        .def_readwrite("pixel_format", &TextureResourceDesc::pixelFormat)
        .def_readwrite("color_space", &TextureResourceDesc::colorSpace)
        .def_readwrite("mip_policy", &TextureResourceDesc::mipPolicy)
        .def_readwrite("subresources", &TextureResourceDesc::subresources)
        .def_readwrite("pixel_data", &TextureResourceDesc::pixelData);

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

    py::class_<PointLightComponent>(m, "PointLightComponent")
        .def(py::init<>())
        .def_readwrite("color", &PointLightComponent::color)
        .def_readwrite("intensity", &PointLightComponent::intensity)
        .def_readwrite("range", &PointLightComponent::range)
        .def_readwrite("shadow_bias", &PointLightComponent::shadowBias)
        .def_readwrite("casts_shadows", &PointLightComponent::castsShadows);

    py::class_<SpotLightComponent>(m, "SpotLightComponent")
        .def(py::init<>())
        .def_readwrite("direction", &SpotLightComponent::direction)
        .def_readwrite("color", &SpotLightComponent::color)
        .def_readwrite("intensity", &SpotLightComponent::intensity)
        .def_readwrite("range", &SpotLightComponent::range)
        .def_readwrite("inner_cone_angle", &SpotLightComponent::innerConeAngle)
        .def_readwrite("outer_cone_angle", &SpotLightComponent::outerConeAngle)
        .def_readwrite("shadow_bias", &SpotLightComponent::shadowBias)
        .def_readwrite("casts_shadows", &SpotLightComponent::castsShadows);

    py::class_<EnvironmentIblDesc>(m, "EnvironmentIblDesc")
        .def(py::init<>())
        .def_readwrite("background_cubemap", &EnvironmentIblDesc::backgroundCubemap)
        .def_readwrite("irradiance_cubemap", &EnvironmentIblDesc::irradianceCubemap)
        .def_readwrite("prefiltered_specular_cubemap",
                       &EnvironmentIblDesc::prefilteredSpecularCubemap)
        .def_readwrite("intensity", &EnvironmentIblDesc::intensity)
        .def_readwrite("background_intensity", &EnvironmentIblDesc::backgroundIntensity)
        .def("enabled", &EnvironmentIblDesc::enabled);

    py::class_<EnvironmentFluidDesc>(m, "EnvironmentFluidDesc")
        .def(py::init<>())
        .def_readwrite("smoothness", &EnvironmentFluidDesc::smoothness)
        .def_readwrite("specular", &EnvironmentFluidDesc::specular)
        .def_readwrite("fresnel", &EnvironmentFluidDesc::fresnel)
        .def_readwrite("depth_edge_threshold", &EnvironmentFluidDesc::depthEdgeThreshold)
        .def_readwrite("filter_radius_pixels", &EnvironmentFluidDesc::filterRadiusPixels)
        .def_readwrite("filter_world_radius", &EnvironmentFluidDesc::filterWorldRadius)
        .def_readwrite("filter_depth_threshold", &EnvironmentFluidDesc::filterDepthThreshold)
        .def_readwrite("enable_background_refraction",
                       &EnvironmentFluidDesc::enableBackgroundRefraction)
        .def_readwrite("refraction_ior", &EnvironmentFluidDesc::refractionIor)
        .def_readwrite("refraction_view_thickness", &EnvironmentFluidDesc::refractionViewThickness);

    py::class_<ParticleContactMaterialDesc>(m, "ParticleContactMaterialDesc")
        .def(py::init<>())
        .def_readwrite("friction", &ParticleContactMaterialDesc::friction)
        .def_readwrite("restitution", &ParticleContactMaterialDesc::restitution)
        .def_readwrite("damping", &ParticleContactMaterialDesc::damping)
        .def_readwrite("static_friction", &ParticleContactMaterialDesc::staticFriction);

    py::class_<SoftBodyRegularGridSource>(m, "SoftBodyRegularGridSource")
        .def(py::init<>())
        .def_readwrite("size", &SoftBodyRegularGridSource::size)
        .def_readwrite("target_particle_spacing", &SoftBodyRegularGridSource::targetParticleSpacing)
        .def_readwrite("static_particle_indices",
                       &SoftBodyRegularGridSource::staticParticleIndices);

    py::class_<SoftBodyTetMeshSource>(m, "SoftBodyTetMeshSource")
        .def(py::init<>())
        .def_readwrite("object_space_rest_positions",
                       &SoftBodyTetMeshSource::objectSpaceRestPositions)
        .def_readwrite("tet_vertex_indices", &SoftBodyTetMeshSource::tetVertexIndices)
        .def_readwrite("static_particle_indices", &SoftBodyTetMeshSource::staticParticleIndices);

    py::class_<SoftBodyTetGenSource>(m, "SoftBodyTetGenSource")
        .def(py::init<>())
        .def_readwrite("node_file", &SoftBodyTetGenSource::nodeFile)
        .def_readwrite("ele_file", &SoftBodyTetGenSource::eleFile)
        .def_readwrite("static_particle_indices", &SoftBodyTetGenSource::staticParticleIndices);

    py::class_<SoftBodyMeshfreeParticleSource>(m, "SoftBodyMeshfreeParticleSource")
        .def(py::init<>())
        .def_readwrite("particle_rest_positions",
                       &SoftBodyMeshfreeParticleSource::particleRestPositions)
        .def_readwrite("surface_rest_positions",
                       &SoftBodyMeshfreeParticleSource::surfaceRestPositions)
        .def_readwrite("surface_normals", &SoftBodyMeshfreeParticleSource::surfaceNormals)
        .def_readwrite("surface_triangles", &SoftBodyMeshfreeParticleSource::surfaceTriangles)
        .def_readwrite("static_particle_indices",
                       &SoftBodyMeshfreeParticleSource::staticParticleIndices)
        .def_readwrite("neighbour_count", &SoftBodyMeshfreeParticleSource::neighbourCount);

    py::class_<SoftBodySourceDesc>(m, "SoftBodySourceDesc")
        .def(py::init<>())
        .def_readwrite("kind", &SoftBodySourceDesc::kind)
        .def_readwrite("regular_grid", &SoftBodySourceDesc::regularGrid)
        .def_readwrite("tet_mesh", &SoftBodySourceDesc::tetMesh)
        .def_readwrite("tet_gen", &SoftBodySourceDesc::tetGen)
        .def_readwrite("meshfree_particles", &SoftBodySourceDesc::meshfreeParticles);

    py::class_<FluidRegularGridSource>(m, "FluidRegularGridSource")
        .def(py::init<>())
        .def_readwrite("size", &FluidRegularGridSource::size)
        .def_readwrite("target_particle_spacing", &FluidRegularGridSource::targetParticleSpacing);

    py::class_<FluidSourceDesc>(m, "FluidSourceDesc")
        .def(py::init<>())
        .def_readwrite("kind", &FluidSourceDesc::kind)
        .def_readwrite("regular_grid", &FluidSourceDesc::regularGrid);

    py::class_<SoftBodyMaterialDesc>(m, "SoftBodyMaterialDesc")
        .def(py::init<>())
        .def_readwrite("contact", &SoftBodyMaterialDesc::contact);

    py::class_<StrandMaterialDesc>(m, "StrandMaterialDesc")
        .def(py::init<>())
        .def_readwrite("contact", &StrandMaterialDesc::contact);

    py::class_<FluidMaterialDesc>(m, "FluidMaterialDesc")
        .def(py::init<>())
        .def_readwrite("contact", &FluidMaterialDesc::contact)
        .def_readwrite("viscosity", &FluidMaterialDesc::viscosity)
        .def_readwrite("cohesion", &FluidMaterialDesc::cohesion)
        .def_readwrite("surface_tension", &FluidMaterialDesc::surfaceTension)
        .def_readwrite("vorticity_confinement", &FluidMaterialDesc::vorticityConfinement)
        .def_readwrite("gravity_scale", &FluidMaterialDesc::gravityScale)
        .def_readwrite("cfl_coefficient", &FluidMaterialDesc::cflCoefficient);

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

    py::class_<SoftBodyComponent>(m, "SoftBodyComponent")
        .def(py::init<>())
        .def_readwrite("source", &SoftBodyComponent::source)
        .def_readwrite("material", &SoftBodyComponent::material)
        .def_readwrite("particle_mass", &SoftBodyComponent::particleMass)
        .def_readwrite("particle_radius", &SoftBodyComponent::particleRadius)
        .def_readwrite("edge_compliance", &SoftBodyComponent::edgeCompliance)
        .def_readwrite("volume_compliance", &SoftBodyComponent::volumeCompliance)
        .def_readwrite("simulated", &SoftBodyComponent::simulated)
        .def_readwrite("self_collision_enabled", &SoftBodyComponent::selfCollisionEnabled)
        .def_readwrite("supports_suturing", &SoftBodyComponent::supportsSuturing)
        .def_readwrite("collision_layer", &SoftBodyComponent::collisionLayer)
        .def_readwrite("collision_mask", &SoftBodyComponent::collisionMask);

    py::class_<MeshfreeSoftBodyComponent>(m, "MeshfreeSoftBodyComponent")
        .def(py::init<>())
        .def_readwrite("particles", &MeshfreeSoftBodyComponent::particles)
        .def_readwrite("surface_rest_positions", &MeshfreeSoftBodyComponent::surfaceRestPositions)
        .def_readwrite("surface_normals", &MeshfreeSoftBodyComponent::surfaceNormals)
        .def_readwrite("surface_triangles", &MeshfreeSoftBodyComponent::surfaceTriangles)
        .def_readwrite("static_particle_indices", &MeshfreeSoftBodyComponent::staticParticleIndices)
        .def_readwrite("material", &MeshfreeSoftBodyComponent::material)
        .def_readwrite("particle_radius", &MeshfreeSoftBodyComponent::particleRadius)
        .def_readwrite("particle_mass", &MeshfreeSoftBodyComponent::particleMass)
        .def_readwrite("neighbour_count", &MeshfreeSoftBodyComponent::neighbourCount)
        .def_readwrite("compliance", &MeshfreeSoftBodyComponent::compliance)
        .def_readwrite("simulated", &MeshfreeSoftBodyComponent::simulated)
        .def_readwrite("self_collision_enabled", &MeshfreeSoftBodyComponent::selfCollisionEnabled)
        .def_readwrite("collision_layer", &MeshfreeSoftBodyComponent::collisionLayer)
        .def_readwrite("collision_mask", &MeshfreeSoftBodyComponent::collisionMask);

    py::class_<HingeJointState>(m, "HingeJointState")
        .def(py::init<>())
        .def_readwrite("joint_id", &HingeJointState::jointId)
        .def_readwrite("enabled", &HingeJointState::enabled)
        .def_readwrite("suppress_connected_body_collisions",
                       &HingeJointState::suppressConnectedBodyCollisions)
        .def_readwrite("drive_mode", &HingeJointState::driveMode)
        .def_readwrite("limit_enabled", &HingeJointState::limitEnabled)
        .def_readwrite("body_a", &HingeJointState::bodyA)
        .def_readwrite("body_b", &HingeJointState::bodyB)
        .def_readwrite("local_anchor_a", &HingeJointState::localAnchorA)
        .def_readwrite("local_anchor_b", &HingeJointState::localAnchorB)
        .def_readwrite("local_rotation_a", &HingeJointState::localRotationA)
        .def_readwrite("local_rotation_b", &HingeJointState::localRotationB)
        .def_readwrite("limit_min", &HingeJointState::limitMin)
        .def_readwrite("limit_max", &HingeJointState::limitMax)
        .def_readwrite("constraint_compliance", &HingeJointState::constraintCompliance)
        .def_readwrite("drive_compliance", &HingeJointState::driveCompliance)
        .def_readwrite("drive_target_angle", &HingeJointState::driveTargetAngle)
        .def_readwrite("drive_damping", &HingeJointState::driveDamping)
        .def_readwrite("drive_max_angular_velocity", &HingeJointState::driveMaxAngularVelocity)
        .def_readwrite("drive_target_angular_velocity",
                       &HingeJointState::driveTargetAngularVelocity);

    py::class_<BallJointState>(m, "BallJointState")
        .def(py::init<>())
        .def_readwrite("joint_id", &BallJointState::jointId)
        .def_readwrite("enabled", &BallJointState::enabled)
        .def_readwrite("suppress_connected_body_collisions",
                       &BallJointState::suppressConnectedBodyCollisions)
        .def_readwrite("body_a", &BallJointState::bodyA)
        .def_readwrite("body_b", &BallJointState::bodyB)
        .def_readwrite("local_anchor_a", &BallJointState::localAnchorA)
        .def_readwrite("local_anchor_b", &BallJointState::localAnchorB);

    py::class_<SphericalJointState>(m, "SphericalJointState")
        .def(py::init<>())
        .def_readwrite("joint_id", &SphericalJointState::jointId)
        .def_readwrite("enabled", &SphericalJointState::enabled)
        .def_readwrite("suppress_connected_body_collisions",
                       &SphericalJointState::suppressConnectedBodyCollisions)
        .def_readwrite("drive_mode", &SphericalJointState::driveMode)
        .def_readwrite("limit_enabled", &SphericalJointState::limitEnabled)
        .def_readwrite("body_a", &SphericalJointState::bodyA)
        .def_readwrite("body_b", &SphericalJointState::bodyB)
        .def_readwrite("local_anchor_a", &SphericalJointState::localAnchorA)
        .def_readwrite("local_anchor_b", &SphericalJointState::localAnchorB)
        .def_readwrite("local_rotation_a", &SphericalJointState::localRotationA)
        .def_readwrite("local_rotation_b", &SphericalJointState::localRotationB)
        .def_readwrite("swing_limit_y", &SphericalJointState::swingLimitY)
        .def_readwrite("swing_limit_z", &SphericalJointState::swingLimitZ)
        .def_readwrite("twist_limit_min", &SphericalJointState::twistLimitMin)
        .def_readwrite("twist_limit_max", &SphericalJointState::twistLimitMax)
        .def_readwrite("constraint_compliance", &SphericalJointState::constraintCompliance)
        .def_readwrite("swing_compliance", &SphericalJointState::swingCompliance)
        .def_readwrite("twist_compliance", &SphericalJointState::twistCompliance)
        .def_readwrite("drive_compliance", &SphericalJointState::driveCompliance)
        .def_readwrite("drive_target_orientation", &SphericalJointState::driveTargetOrientation);

    py::class_<SliderJointState>(m, "SliderJointState")
        .def(py::init<>())
        .def_readwrite("joint_id", &SliderJointState::jointId)
        .def_readwrite("enabled", &SliderJointState::enabled)
        .def_readwrite("suppress_connected_body_collisions",
                       &SliderJointState::suppressConnectedBodyCollisions)
        .def_readwrite("drive_mode", &SliderJointState::driveMode)
        .def_readwrite("limit_enabled", &SliderJointState::limitEnabled)
        .def_readwrite("body_a", &SliderJointState::bodyA)
        .def_readwrite("body_b", &SliderJointState::bodyB)
        .def_readwrite("local_anchor_a", &SliderJointState::localAnchorA)
        .def_readwrite("local_anchor_b", &SliderJointState::localAnchorB)
        .def_readwrite("local_rotation_a", &SliderJointState::localRotationA)
        .def_readwrite("local_rotation_b", &SliderJointState::localRotationB)
        .def_readwrite("limit_min", &SliderJointState::limitMin)
        .def_readwrite("limit_max", &SliderJointState::limitMax)
        .def_readwrite("constraint_compliance", &SliderJointState::constraintCompliance)
        .def_readwrite("drive_compliance", &SliderJointState::driveCompliance)
        .def_readwrite("drive_damping", &SliderJointState::driveDamping)
        .def_readwrite("drive_max_velocity", &SliderJointState::driveMaxVelocity)
        .def_readwrite("drive_target_position", &SliderJointState::driveTargetPosition)
        .def_readwrite("drive_target_velocity", &SliderJointState::driveTargetVelocity);

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

    py::class_<StrandComponent>(m, "StrandComponent")
        .def(py::init<>())
        .def_readwrite("material", &StrandComponent::material)
        .def_readwrite("rest_positions", &StrandComponent::restPositions)
        .def_readwrite("static_particle_indices", &StrandComponent::staticParticleIndices)
        .def_readwrite("particle_mass", &StrandComponent::particleMass)
        .def_readwrite("particle_radius", &StrandComponent::particleRadius)
        .def_readwrite("stretch_shear_compliance", &StrandComponent::stretchShearCompliance)
        .def_readwrite("bend_compliance", &StrandComponent::bendCompliance)
        .def_readwrite("twist_compliance", &StrandComponent::twistCompliance)
        .def_readwrite("distance_compliance", &StrandComponent::distanceCompliance)
        .def_readwrite("root_material_normal", &StrandComponent::rootMaterialNormal)
        .def_readwrite("simulated", &StrandComponent::simulated)
        .def_readwrite("self_collision_enabled", &StrandComponent::selfCollisionEnabled)
        .def_readwrite("suturing_enabled", &StrandComponent::suturingEnabled)
        .def_readwrite("path_node_spacing", &StrandComponent::pathNodeSpacing)
        .def_readwrite("collision_layer", &StrandComponent::collisionLayer)
        .def_readwrite("collision_mask", &StrandComponent::collisionMask);

    py::class_<FluidComponent>(m, "FluidComponent")
        .def(py::init<>())
        .def_readwrite("source", &FluidComponent::source)
        .def_readwrite("material", &FluidComponent::material)
        .def_readwrite("visual_color", &FluidComponent::visualColor)
        .def_readwrite("particle_mass", &FluidComponent::particleMass)
        .def_readwrite("particle_radius", &FluidComponent::particleRadius)
        .def_readwrite("simulated", &FluidComponent::simulated)
        .def_readwrite("collision_layer", &FluidComponent::collisionLayer)
        .def_readwrite("collision_mask", &FluidComponent::collisionMask);

    py::class_<UltrasoundProbeComponent>(m, "UltrasoundProbeComponent")
        .def(py::init<>())
        .def_readwrite("enabled", &UltrasoundProbeComponent::enabled)
        .def_readwrite("geometry", &UltrasoundProbeComponent::geometry)
        .def_readwrite("num_scanlines", &UltrasoundProbeComponent::numScanlines)
        .def_readwrite("line_length", &UltrasoundProbeComponent::lineLength)
        .def_readwrite("scanline_spacing", &UltrasoundProbeComponent::scanlineSpacing)
        .def_readwrite("sector_angle_degrees", &UltrasoundProbeComponent::sectorAngleDegrees)
        .def_readwrite("probe_radius", &UltrasoundProbeComponent::probeRadius)
        .def_readwrite("sound_speed", &UltrasoundProbeComponent::soundSpeed)
        .def_readwrite("world_units_per_meter", &UltrasoundProbeComponent::worldUnitsPerMeter)
        .def_readwrite("noise_amplitude", &UltrasoundProbeComponent::noiseAmplitude)
        .def_readwrite("sampling_frequency", &UltrasoundProbeComponent::samplingFrequency)
        .def_readwrite("demodulation_frequency", &UltrasoundProbeComponent::demodulationFrequency)
        .def_readwrite("center_frequency", &UltrasoundProbeComponent::centerFrequency)
        .def_readwrite("fractional_bandwidth", &UltrasoundProbeComponent::fractionalBandwidth)
        .def_readwrite("beam_sigma_lateral", &UltrasoundProbeComponent::beamSigmaLateral)
        .def_readwrite("beam_sigma_elevational", &UltrasoundProbeComponent::beamSigmaElevational)
        .def_readwrite("radial_decimation", &UltrasoundProbeComponent::radialDecimation)
        .def_readwrite("threads_per_block", &UltrasoundProbeComponent::threadsPerBlock)
        .def_readwrite("cuda_num_streams", &UltrasoundProbeComponent::cudaNumStreams)
        .def_readwrite("num_time_samples", &UltrasoundProbeComponent::numTimeSamples)
        .def_readwrite("use_arc_projection", &UltrasoundProbeComponent::useArcProjection)
        .def_readwrite("enable_phase_delay", &UltrasoundProbeComponent::enablePhaseDelay);

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

    py::class_<ProceduralDeformableCurveRenderComponent>(m,
                                                         "ProceduralDeformableCurveRenderComponent")
        .def(py::init<>())
        .def_readwrite("sequence_id", &ProceduralDeformableCurveRenderComponent::sequenceId)
        .def_readwrite("radius", &ProceduralDeformableCurveRenderComponent::radius)
        .def_readwrite("radial_resolution",
                       &ProceduralDeformableCurveRenderComponent::radialResolution)
        .def_readwrite("enabled", &ProceduralDeformableCurveRenderComponent::enabled);

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

    py::class_<RenderResourceManager>(m, "RenderResourceManager")
        .def("register_mesh", &RenderResourceManager::registerMesh)
        .def("register_material", &RenderResourceManager::registerMaterial)
        .def("register_texture", &RenderResourceManager::registerTexture)
        .def("is_valid_mesh",
             py::overload_cast<MeshHandle>(&RenderResourceManager::isValid, py::const_))
        .def("is_valid_material",
             py::overload_cast<MaterialHandle>(&RenderResourceManager::isValid, py::const_))
        .def("is_valid_texture",
             py::overload_cast<TextureHandle>(&RenderResourceManager::isValid, py::const_))
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
        .def("try_get_texture",
             [](const RenderResourceManager &resources, const TextureHandle texture) -> py::object
             {
                 if (const auto *desc = resources.tryGetTexture(texture))
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
            py::arg("config") = py::none())
        .def("shutdown", &Runtime::shutdown)
        .def("get_info", &Runtime::getInfo)
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
