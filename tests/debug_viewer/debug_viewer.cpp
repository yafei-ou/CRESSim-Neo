#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::common::Quatf;
using cressim::neo::common::Vec3f;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RenderTargetDesc;
using cressim::neo::graphics::RenderTargetHandle;
using cressim::neo::graphics::RenderTargetReadbackEvent;

struct AppConfig
{
    GraphicsBackend backend = GraphicsBackend::Vulkan;
    bool viewerEnabled = true;
    bool validation = false;
    bool readbackEnabled = false;
    bool vSync = true;
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    std::uint64_t maxFrames = 0;
};

struct KeyState
{
    int key = 0;
    bool wasDown = false;
};

GraphicsBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return GraphicsBackend::Null;
    }
    if (value == "vulkan")
    {
        return GraphicsBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

bool parseOnOff(const std::string& value)
{
    if (value == "on")
    {
        return true;
    }
    if (value == "off")
    {
        return false;
    }
    throw std::invalid_argument("Expected on|off, got: " + value);
}

void printUsage(const char* appName)
{
    std::cerr << "Usage: " << appName << " [--backend vulkan|null] [--viewer on|off] [--validation on|off]\n"
              << "       [--readback on|off] [--vsync on|off] [--width N] [--height N] [--frames N]\n";
}

bool consumeKeyPress(GLFWwindow* window, KeyState& state)
{
    const bool isDown = glfwGetKey(window, state.key) == GLFW_PRESS;
    const bool pressed = isDown && !state.wasDown;
    state.wasDown = isDown;
    return pressed;
}

float degreesToRadians(float value)
{
    return value * 0.017453292519943295769f;
}

Quatf quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees, float rollDegrees)
{
    const float pitch = degreesToRadians(pitchDegrees) * 0.5f;
    const float yaw = degreesToRadians(yawDegrees) * 0.5f;
    const float roll = degreesToRadians(rollDegrees) * 0.5f;

    const float sinPitch = std::sin(pitch);
    const float cosPitch = std::cos(pitch);
    const float sinYaw = std::sin(yaw);
    const float cosYaw = std::cos(yaw);
    const float sinRoll = std::sin(roll);
    const float cosRoll = std::cos(roll);

    Quatf q{};
    q.w = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw;
    q.x = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw;
    q.y = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw;
    q.z = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw;
    return q;
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "DebugViewer.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Vec3f& normal, const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 3u);
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

bool parseArguments(int argc, char** argv, AppConfig& outConfig)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto expectValue = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for " + arg);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--backend")
        {
            const std::string value = expectValue();
            outConfig.backend = parseBackend(value);
            continue;
        }
        if (arg == "--viewer")
        {
            const std::string value = expectValue();
            outConfig.viewerEnabled = parseOnOff(value);
            continue;
        }
        if (arg == "--validation")
        {
            const std::string value = expectValue();
            outConfig.validation = parseOnOff(value);
            continue;
        }
        if (arg == "--readback")
        {
            const std::string value = expectValue();
            outConfig.readbackEnabled = parseOnOff(value);
            continue;
        }
        if (arg == "--vsync")
        {
            const std::string value = expectValue();
            outConfig.vSync = parseOnOff(value);
            continue;
        }
        if (arg == "--width")
        {
            const std::string value = expectValue();
            outConfig.width = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
            continue;
        }
        if (arg == "--height")
        {
            const std::string value = expectValue();
            outConfig.height = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
            continue;
        }
        if (arg == "--frames")
        {
            const std::string value = expectValue();
            outConfig.maxFrames = static_cast<std::uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
            continue;
        }

        throw std::invalid_argument("Unknown argument: " + arg);
    }

    outConfig.width = std::max(outConfig.width, 1u);
    outConfig.height = std::max(outConfig.height, 1u);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    AppConfig app{};
    try
    {
        (void)parseArguments(argc, argv, app);
    }
    catch (const std::exception& e)
    {
        printUsage(argv[0]);
        std::cerr << e.what() << '\n';
        return 2;
    }

    if (!app.viewerEnabled && app.maxFrames == 0)
    {
        app.maxFrames = 240;
    }
    if (app.backend == GraphicsBackend::Null && app.viewerEnabled)
    {
        std::cout << "Null backend does not support window presentation; forcing viewer off.\n";
        app.viewerEnabled = false;
    }

    GLFWwindow* window = nullptr;
    if (app.viewerEnabled)
    {
        if (glfwInit() != GLFW_TRUE)
        {
            std::cerr << "Failed to initialize GLFW.\n";
            return 1;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(
            static_cast<int>(app.width),
            static_cast<int>(app.height),
            "CRESSim Neo Debug Viewer",
            nullptr,
            nullptr);
        if (window == nullptr)
        {
            glfwTerminate();
            std::cerr << "Failed to create debug viewer window.\n";
            return 1;
        }
    }

    RuntimeConfig runtimeConfig{};
    runtimeConfig.graphicsDeviceDesc.preferredBackend = app.backend;
    runtimeConfig.graphicsDeviceDesc.enableValidation = app.validation;
    runtimeConfig.graphicsDeviceDesc.initialWidth = app.width;
    runtimeConfig.graphicsDeviceDesc.initialHeight = app.height;
    runtimeConfig.rendererDesc.debugViewer.enabled = app.viewerEnabled;
    runtimeConfig.rendererDesc.debugViewer.syncInterval = app.vSync ? 1u : 0u;

#if defined(_WIN32)
    if (app.viewerEnabled)
    {
        runtimeConfig.renderer.debugViewer.nativeWindow = glfwGetWin32Window(window);
    }
#elif defined(__linux__)
    if (app.viewerEnabled)
    {
        runtimeConfig.rendererDesc.debugViewer.nativeWindowId = static_cast<std::uint64_t>(glfwGetX11Window(window));
        runtimeConfig.rendererDesc.debugViewer.nativeDisplay = glfwGetX11Display();
    }
#endif

    Runtime runtime;
    if (!runtime.initialize(runtimeConfig))
    {
        if (window != nullptr)
        {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    auto* graphicsDevice = runtime.getGraphicsDevice();
    if (graphicsDevice == nullptr)
    {
        runtime.shutdown();
        if (window != nullptr)
        {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
        std::cerr << "Graphics device unavailable.\n";
        return 1;
    }

    RenderTargetHandle offscreenTarget{};
    {
        RenderTargetDesc offscreenDesc{};
        offscreenDesc.width = app.width;
        offscreenDesc.height = app.height;
        offscreenDesc.cpuReadback = true;
        offscreenDesc.debugName = "DebugViewer.Offscreen";
        offscreenTarget = graphicsDevice->createRenderTarget(offscreenDesc);
    }
    const bool hasOffscreenTarget = graphicsDevice->isValidRenderTarget(offscreenTarget);

    auto& world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.2f, 4.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    camera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
    camera.outputWidth = app.width;
    camera.outputHeight = app.height;
    camera.outputTarget = app.viewerEnabled ? RenderTargetHandle{} : offscreenTarget;
    camera.requestReadback = app.readbackEnabled && !app.viewerEnabled && hasOffscreenTarget;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -0.45f, -1.0f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 4.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getScene().resources();
    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));

    MaterialResourceDesc frontMaterialDesc{};
    frontMaterialDesc.debugName = "DebugViewer.FrontMaterial";
    frontMaterialDesc.baseColor = {0.95f, 0.10f, 0.08f};
    frontMaterialDesc.metallic = 0.0f;
    frontMaterialDesc.roughness = 0.45f;
    const auto frontMaterial = resources.registerMaterial(frontMaterialDesc);

    MaterialResourceDesc backMaterialDesc{};
    backMaterialDesc.debugName = "DebugViewer.BackMaterial";
    backMaterialDesc.baseColor = {0.10f, 0.85f, 0.12f};
    backMaterialDesc.metallic = 0.0f;
    backMaterialDesc.roughness = 0.45f;
    const auto backMaterial = resources.registerMaterial(backMaterialDesc);

    const auto frontCubeEntity = world.createEntity();
    TransformComponent frontCubeTransform{};
    frontCubeTransform.worldTransform.position = {0.18f, -0.02f, 0.05f};
    world.setTransform(frontCubeEntity, frontCubeTransform);
    MeshRendererComponent frontCube{};
    frontCube.mesh = cubeMesh;
    frontCube.material = frontMaterial;
    frontCube.visible = true;
    world.setMeshRenderer(frontCubeEntity, frontCube);

    const auto backCubeEntity = world.createEntity();
    TransformComponent backCubeTransform{};
    backCubeTransform.worldTransform.position = {-0.14f, 0.03f, -1.35f};
    backCubeTransform.worldTransform.scale = {1.35f, 1.35f, 1.35f};
    world.setTransform(backCubeEntity, backCubeTransform);
    MeshRendererComponent backCube{};
    backCube.mesh = cubeMesh;
    backCube.material = backMaterial;
    backCube.visible = true;
    world.setMeshRenderer(backCubeEntity, backCube);

    bool renderToWindow = app.viewerEnabled;
    bool animate = true;
    bool showFrontCube = true;
    bool requestReadback = app.readbackEnabled;
    bool showStats = true;

    if (app.viewerEnabled)
    {
        std::cout << "Debug viewer controls: Esc=quit, V=toggle viewer output, 1=toggle front cube, "
                  << "2=toggle animation, R=toggle readback, H=toggle stats\n";
        if (!hasOffscreenTarget)
        {
            std::cout << "Offscreen target unavailable; viewer toggle and readback are limited to window mode.\n";
        }
    }

    KeyState keyQuit{GLFW_KEY_ESCAPE, false};
    KeyState keyToggleViewer{GLFW_KEY_V, false};
    KeyState keyToggleCube{GLFW_KEY_1, false};
    KeyState keyToggleAnimation{GLFW_KEY_2, false};
    KeyState keyToggleReadback{GLFW_KEY_R, false};
    KeyState keyToggleStats{GLFW_KEY_H, false};

    double lastTimeSeconds = app.viewerEnabled ? glfwGetTime() : 0.0;
    float frontYawDegrees = 32.0f;
    float backYawDegrees = -24.0f;

    std::uint64_t readbackEvents = 0;
    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    while (true)
    {
        if (app.viewerEnabled)
        {
            glfwPollEvents();
            if (glfwWindowShouldClose(window))
            {
                break;
            }
            if (consumeKeyPress(window, keyQuit))
            {
                break;
            }
            if (consumeKeyPress(window, keyToggleViewer) && hasOffscreenTarget)
            {
                renderToWindow = !renderToWindow;
            }
            if (consumeKeyPress(window, keyToggleCube))
            {
                showFrontCube = !showFrontCube;
            }
            if (consumeKeyPress(window, keyToggleAnimation))
            {
                animate = !animate;
            }
            if (consumeKeyPress(window, keyToggleReadback))
            {
                requestReadback = !requestReadback;
            }
            if (consumeKeyPress(window, keyToggleStats))
            {
                showStats = !showStats;
            }
        }

        if (app.viewerEnabled)
        {
            const double nowSeconds = glfwGetTime();
            const double dtSeconds = std::max(1.0 / 240.0, std::min(nowSeconds - lastTimeSeconds, 0.1));
            frame.deltaSeconds = static_cast<float>(dtSeconds);
            frame.timeSeconds += dtSeconds;
            lastTimeSeconds = nowSeconds;
        }
        else
        {
            frame.timeSeconds += static_cast<double>(frame.deltaSeconds);
        }

        if (animate)
        {
            frontYawDegrees += frame.deltaSeconds * 25.0f;
            backYawDegrees -= frame.deltaSeconds * 18.0f;
        }

        frontCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(-18.0f, frontYawDegrees, 0.0f);
        backCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(12.0f, backYawDegrees, 0.0f);
        world.setTransform(frontCubeEntity, frontCubeTransform);
        world.setTransform(backCubeEntity, backCubeTransform);

        frontCube.visible = showFrontCube;
        world.setMeshRenderer(frontCubeEntity, frontCube);

        if (renderToWindow)
        {
            int fbWidth = static_cast<int>(app.width);
            int fbHeight = static_cast<int>(app.height);
            if (window != nullptr)
            {
                glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            }
            fbWidth = std::max(fbWidth, 1);
            fbHeight = std::max(fbHeight, 1);

            graphicsDevice->resizeDefaultRenderTarget(static_cast<std::uint32_t>(fbWidth), static_cast<std::uint32_t>(fbHeight));
            camera.outputTarget = {};
            camera.outputWidth = static_cast<std::uint32_t>(fbWidth);
            camera.outputHeight = static_cast<std::uint32_t>(fbHeight);
            camera.requestReadback = false;
        }
        else
        {
            camera.outputTarget = hasOffscreenTarget ? offscreenTarget : RenderTargetHandle{};
            camera.outputWidth = app.width;
            camera.outputHeight = app.height;
            camera.requestReadback = requestReadback && hasOffscreenTarget;
        }
        world.setCamera(cameraEntity, camera);

        frame.frameIndex++;
        runtime.tick(frame);

        RenderTargetReadbackEvent readbackEvent{};
        while (runtime.tryPopReadbackEvent(readbackEvent))
        {
            if (readbackEvent.target.id == offscreenTarget.id)
            {
                ++readbackEvents;
            }
        }

        if (showStats && (frame.frameIndex % 120u == 0u))
        {
            std::cout << "frame=" << frame.frameIndex
                      << " viewer=" << (renderToWindow ? "on" : "off")
                      << " frontCube=" << (showFrontCube ? "on" : "off")
                      << " animate=" << (animate ? "on" : "off")
                      << " readback=" << (camera.requestReadback ? "on" : "off")
                      << " readbackEvents=" << readbackEvents << '\n';
        }

        if (app.maxFrames > 0 && frame.frameIndex >= app.maxFrames)
        {
            break;
        }
    }

    runtime.shutdown();

    if (window != nullptr)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    return 0;
}
