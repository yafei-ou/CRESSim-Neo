#include "structured_buffer_compat.hlsli"

struct BatchCamera
{
    uint globalCameraIndex;
    uint envIndex;
    uint mainLightIndex;
    uint colorLayer;
    uint shadowLayer;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

CRESSIM_STRUCTURED_BUFFER(BatchCamera, g_BatchCameras);

struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint CameraIndex : TEXCOORD1;
    nointerpolation uint EnvIndex : TEXCOORD2;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
};

void main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID, out VSOutput Out)
{
    const BatchCamera batchCamera = CRESSIM_SB_LOAD(g_BatchCameras, instanceId);
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
    Out.CameraIndex = batchCamera.globalCameraIndex;
    Out.EnvIndex = batchCamera.envIndex;
#if MANUAL_LAYER_EXPORT
    Out.Layer = batchCamera.colorLayer;
#endif
}
