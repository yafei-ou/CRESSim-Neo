#include "viewer/macos_glfw_native_window.h"

#define GLFW_EXPOSE_NATIVE_COCOA 1
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

namespace cressim::neo::viewer
{

void *prepareMacOSPresentationView(GLFWwindow *window)
{
    if (window == nullptr)
    {
        return nullptr;
    }

    NSWindow *nativeWindow = glfwGetCocoaWindow(window);
    if (nativeWindow == nil)
    {
        return nullptr;
    }

    NSView *view = nativeWindow.contentView;
    if (view == nil)
    {
        return nullptr;
    }

    CAMetalLayer *layer = [view.layer isKindOfClass:[CAMetalLayer class]]
                              ? (CAMetalLayer *)view.layer
                              : [CAMetalLayer layer];
    if (layer == nil)
    {
        return nullptr;
    }

    [view setWantsLayer:YES];
    [view setLayer:layer];
    [layer setAutoresizingMask:kCALayerWidthSizable | kCALayerHeightSizable];
    [layer setFrame:view.bounds];

    const CGSize viewScale = [view convertSizeToBacking:CGSizeMake(1.0, 1.0)];
    [layer setContentsScale:MIN(viewScale.width, viewScale.height)];

    return (__bridge void *)view;
}

} // namespace cressim::neo::viewer
