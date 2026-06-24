#ifndef CRESSIM_NEO_SRC_ENGINE_CUSTOM_COMPUTE_SERVICE_H
#define CRESSIM_NEO_SRC_ENGINE_CUSTOM_COMPUTE_SERVICE_H

#include "engine/custom_compute.h"

#include "gpu/gpu_compute_pass.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cressim::neo::gpu
{
class GpuDevice;
}

namespace cressim::neo::physics
{
class PhysicsSolver;
class PhysicsWorld;
} // namespace cressim::neo::physics

namespace cressim::neo::engine
{

class SharedBufferService;

class CustomComputeService
{
public:
    explicit CustomComputeService(gpu::GpuDevice &device);
    ~CustomComputeService();

    std::vector<CustomComputeResourceDesc> listResources(physics::PhysicsSolver &solver,
                                                         physics::PhysicsWorld &world);
    CustomComputePassHandle createPass(physics::PhysicsSolver &solver, physics::PhysicsWorld &world,
                                       const SharedBufferService *sharedBuffers,
                                       const CustomComputePassDesc &desc);
    bool updatePassConstants(CustomComputePassHandle handle, const std::vector<std::uint8_t> &data);
    bool executePass(physics::PhysicsSolver &solver, physics::PhysicsWorld &world,
                     const SharedBufferService *sharedBuffers, CustomComputePassHandle handle);
    bool destroyPass(CustomComputePassHandle handle);
    void clear();

private:
    struct ResourceEntry;
    struct PassState;

    using ResourceMap = std::unordered_map<std::string, ResourceEntry>;

    bool buildResourceRegistry(physics::PhysicsSolver &solver, physics::PhysicsWorld &world,
                               ResourceMap &outResources,
                               std::vector<CustomComputeResourceDesc> *outDescs);
    static bool isAccessCompatible(CustomComputeResourceAccess requested,
                                   CustomComputeResourceAccess allowed) noexcept;
    static Diligent::BUFFER_VIEW_TYPE bufferViewTypeForAccess(
        CustomComputeResourceAccess access) noexcept;
    static std::uint32_t roundUpConstantBufferSize(std::uint32_t sizeBytes) noexcept;
    bool bindPassResources(PassState &pass, const ResourceMap &resources,
                           const SharedBufferService *sharedBuffers);
    bool uploadConstantData(PassState &pass, const std::vector<std::uint8_t> &data);

private:
    gpu::GpuDevice &mDevice;
    std::uint64_t mNextPassId = 1u;
    std::unordered_map<std::uint64_t, std::unique_ptr<PassState>> mPasses;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_SRC_ENGINE_CUSTOM_COMPUTE_SERVICE_H
