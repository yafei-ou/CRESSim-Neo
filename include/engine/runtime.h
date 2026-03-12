#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/export.h"
#include "engine/world.h"
#include "gpu/gpu_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/render_world.h"
#include "graphics/renderer.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <memory>

namespace cressim::neo::engine
{

struct RuntimeConfig
{
    gpu::GpuDeviceDesc gpuDeviceDesc{};
    graphics::RendererDesc rendererDesc{};
    physics::PhysicsSolverDesc physicsDesc{};
};

class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    bool initialize(const RuntimeConfig& config = RuntimeConfig{});
    void shutdown();

    void tick(const common::FrameContext& frameContext);

    World& getWorld() noexcept;
    const World& getWorld() const noexcept;

    gpu::GpuDevice* getGpuDevice() noexcept;
    const gpu::GpuDevice* getGpuDevice() const noexcept;
    physics::PhysicsSolver* getPhysicsSolver() noexcept;
    const physics::PhysicsSolver* getPhysicsSolver() const noexcept;
    const graphics::RenderStats& lastRenderStats() const noexcept;

    graphics::RenderResourceManager& getResources() noexcept;
    const graphics::RenderResourceManager& getResources() const noexcept;

private:
    bool syncWorldToRenderWorld();
    bool syncWorldToPhysicsWorld();
    bool syncPhysicsWorldToWorld();

    bool mInitialized = false;
    std::unique_ptr<gpu::GpuDevice> mGpuDevice;
    std::unique_ptr<physics::PhysicsSolver> mPhysicsSolver;
    std::unique_ptr<graphics::Renderer> mRenderer;
    graphics::RenderStats mLastRenderStats{};
    std::uint64_t mLastSyncedPhysicsWorldRevision = ~0ull;
    std::uint64_t mLastSyncedWorldRevision        = ~0ull;
    World mWorld;
    graphics::RenderResourceManager mResources;
    physics::PhysicsWorld mPhysicsWorld;
    graphics::RenderWorld mRenderWorld;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
