#include "include/structured_buffer_compat.hlsli"

struct CameraInput
{
    float4 position;
    float4 orientation;
    float4 projectionParams;
    float4 viewportAndOutputSize;
    uint envIndex;
    uint cameraSlot;
    uint active;
    uint reserved;
};

cbuffer GraphicsSkybox
{
    uint g_SkyboxCameraIndex;
    uint g_SkyboxTargetLayer;
    float g_SkyboxViewportAspect;
    float g_SkyboxPadding0;
};

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);

struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint EnvIndex : TEXCOORD1;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
};

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const CameraInput camera = CRESSIM_SB_LOAD(g_CameraInputs, g_SkyboxCameraIndex);
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };
    const float2 texCoords[3] = {
        float2(0.0, 1.0),
        float2(0.0, -1.0),
        float2(2.0, 1.0),
    };

    Out.Position = float4(positions[vertexId], 1.0, 1.0);
    Out.TexCoord = texCoords[vertexId];
    Out.EnvIndex = camera.envIndex;
#if MANUAL_LAYER_EXPORT
    Out.Layer = g_SkyboxTargetLayer;
#endif
}
