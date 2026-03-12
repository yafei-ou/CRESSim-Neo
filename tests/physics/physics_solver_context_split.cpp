#include "common/frame_context.h"
#include "engine/runtime.h"

#include <iostream>

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
            std::cerr << "Runtime initialization failed.\n";
            return 1;
        }

        gpu::GpuDevice* device = runtime.getGpuDevice();
        if (device == nullptr)
        {
            std::cerr << "Runtime returned null GPU device.\n";
            runtime.shutdown();
            return 1;
        }

        gpu::GpuBackendContext graphicsContext{};
        gpu::GpuComputeBackendContext physicsContext{};
        if (!device->tryGetGraphicsBackendContext(graphicsContext) ||
            !device->tryGetPhysicsBackendContext(physicsContext))
        {
            std::cerr << "Failed to retrieve graphics/physics contexts.\n";
            runtime.shutdown();
            return 1;
        }
        if (graphicsContext.renderDevice == nullptr || graphicsContext.immediateContext == nullptr ||
            physicsContext.renderDevice == nullptr || physicsContext.computeContext == nullptr)
        {
            std::cerr << "Context retrieval returned null backend pointers.\n";
            runtime.shutdown();
            return 1;
        }
        if (graphicsContext.renderDevice != physicsContext.renderDevice)
        {
            std::cerr << "Graphics and physics contexts use different devices unexpectedly.\n";
            runtime.shutdown();
            return 1;
        }

        physics::PhysicsSolver transientSolver(*device);
        physics::PhysicsWorld world;
        common::FrameContext frame{};
        frame.deltaSeconds = 1.0f / 60.0f;
        if (transientSolver.step(frame, world))
        {
            std::cerr << "Solver step unexpectedly succeeded before initialize().\n";
            runtime.shutdown();
            return 1;
        }

        runtime.shutdown();
    }

    std::cout << "Physics solver context split checks passed.\n";
    return 0;
}
