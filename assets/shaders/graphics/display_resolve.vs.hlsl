struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
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

    Out.Position = float4(positions[vertexId], 0.0, 1.0);
    Out.TexCoord = texCoords[vertexId];
}
