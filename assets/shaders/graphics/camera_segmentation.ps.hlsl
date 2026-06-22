struct PSInput
{
    float4 Position : SV_Position;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
    nointerpolation uint SegmentationId : TEXCOORD0;
};

uint main(in PSInput In) : SV_Target
{
    return In.SegmentationId;
}
