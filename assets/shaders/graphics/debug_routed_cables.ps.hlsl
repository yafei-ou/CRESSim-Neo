cbuffer GraphicsDebugRoutedCables
{
    uint4 g_DebugRoutedCableParams;
    float4 g_DebugRoutedCableMisc;
};

struct PSInput
{
    float4 Position : SV_Position;
    nointerpolation float4 Color : TEXCOORD0;
};

float4 main(in PSInput In) : SV_Target
{
    return In.Color;
}
