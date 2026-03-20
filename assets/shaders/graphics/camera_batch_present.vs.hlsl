struct VSOutput
{
    float4 Position : SV_Position;
    float2 Uv : TEXCOORD0;
};

VSOutput main(uint vertexId : SV_VertexID)
{
    float2 pos;
    pos.x = (vertexId == 2u) ? 3.0 : -1.0;
    pos.y = (vertexId == 1u) ? -3.0 : 1.0;

    VSOutput outp;
    outp.Position = float4(pos, 0.0, 1.0);
    outp.Uv = float2(0.5, 0.5) + float2(0.5, -0.5) * pos;
    return outp;
}
