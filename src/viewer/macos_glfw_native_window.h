#ifndef CRESSIM_NEO_VIEWER_MACOS_GLFW_NATIVE_WINDOW_H
#define CRESSIM_NEO_VIEWER_MACOS_GLFW_NATIVE_WINDOW_H

struct GLFWwindow;

namespace cressim::neo::viewer
{

void *prepareMacOSPresentationView(GLFWwindow *window);

} // namespace cressim::neo::viewer

#endif // CRESSIM_NEO_VIEWER_MACOS_GLFW_NATIVE_WINDOW_H
