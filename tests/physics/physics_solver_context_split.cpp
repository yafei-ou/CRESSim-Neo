#include "common/frame_context.h"
#include "engine/runtime.h"
#include "common/logger.h"


int main()
{
    using namespace cressim::neo;

    {
        engine::RuntimeConfig config{};
        config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
        config.gpuDeviceDesc.enableValidation = false;

        engine::Runtime runtime;
        if (!runtime.initialize(config))
        {
            CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
            return 1;
        }

        gpu::GpuDevice* device = runtime.getGpuDevice();
        if (device == nullptr)
        {
            CRESSIM_LOG_ERROR( "Runtime returned null GPU device.\n");
            runtime.shutdown();
            return 1;
        }

        gpu::GpuGraphicsBackendContext graphicsContext{};
        gpu::GpuComputeBackendContext physicsContext{};
        if (!device->tryGetGraphicsBackendContext(graphicsContext) ||
            !device->tryGetPhysicsBackendContext(physicsContext))
        {
            CRESSIM_LOG_ERROR( "Failed to retrieve graphics/physics contexts.\n");
            runtime.shutdown();
            return 1;
        }
        if (graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr ||
            physicsContext.renderDevice == nullptr || physicsContext.computeContext == nullptr)
        {
            CRESSIM_LOG_ERROR( "Context retrieval returned null backend pointers.\n");
            runtime.shutdown();
            return 1;
        }
        if (graphicsContext.renderDevice != physicsContext.renderDevice)
        {
            CRESSIM_LOG_ERROR( "Graphics and physics contexts use different devices unexpectedly.\n");
            runtime.shutdown();
            return 1;
        }
        if (graphicsContext.contextId != graphicsContext.graphicsContext->GetDesc().ContextId)
        {
            CRESSIM_LOG_ERROR( "Graphics backend context reported an unexpected context id.\n");
            runtime.shutdown();
            return 1;
        }
        if (physicsContext.contextId != physicsContext.computeContext->GetDesc().ContextId)
        {
            CRESSIM_LOG_ERROR( "Physics backend context reported an unexpected context id.\n");
            runtime.shutdown();
            return 1;
        }
        if (gpu::contextMaskForId(graphicsContext.contextId) == 0u ||
            gpu::contextMaskForId(physicsContext.contextId) == 0u)
        {
            CRESSIM_LOG_ERROR( "Context id to mask conversion produced an invalid mask.\n");
            runtime.shutdown();
            return 1;
        }

        physics::PhysicsSolver transientSolver(*device);
        physics::PhysicsWorld world;
        common::FrameContext frame{};
        frame.deltaSeconds = 1.0f / 60.0f;
        if (transientSolver.step(frame, world))
        {
            CRESSIM_LOG_ERROR( "Solver step unexpectedly succeeded before initialize().\n");
            runtime.shutdown();
            return 1;
        }

        runtime.shutdown();
    }

    CRESSIM_LOG_INFO( "Physics solver context split checks passed.\n");
    return 0;
}
