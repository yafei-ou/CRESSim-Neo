#ifndef CRESSIM_NEO_GRAPHICS_CAMERA_INPUT_HLSLI
#define CRESSIM_NEO_GRAPHICS_CAMERA_INPUT_HLSLI

struct CameraInput
{
    float4 position;
    float4 orientation;
    float4 projectionParams;
    float4 viewportAndOutputSize;
    uint envIndex;
    uint cameraSlot;
    uint active;
    uint entityPoseSlot;
};

#endif // CRESSIM_NEO_GRAPHICS_CAMERA_INPUT_HLSLI
