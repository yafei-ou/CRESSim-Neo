struct IndirectCommandDesc
{
    uint visibleOffset;
    uint maxVisibleCount;
    uint indexCount;
    uint reserved;
};

struct IndirectCandidate
{
    uint objectIndex;
    uint commandIndex;
    uint visibilityMask;
    uint reserved;
};

cbuffer GraphicsIndirectFilterConstants
{
    uint g_CandidateCount;
    uint g_FilterPadding0;
    uint g_FilterPadding1;
    uint g_FilterPadding2;
};

StructuredBuffer<IndirectCommandDesc> g_CommandDescs;
StructuredBuffer<IndirectCandidate> g_Candidates;
StructuredBuffer<uint> g_RenderableVisibilityFlags;
StructuredBuffer<uint> g_RenderableShadowCascadeMasks;
RWStructuredBuffer<uint> g_CommandCountsRW;
RWStructuredBuffer<uint> g_VisibleObjectIndicesRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint candidateIndex = dispatchThreadId.x;
    if (candidateIndex >= g_CandidateCount)
    {
        return;
    }

    const IndirectCandidate candidate = g_Candidates[candidateIndex];
    bool visible = false;
    if (candidate.visibilityMask == 0u)
    {
        visible = g_RenderableVisibilityFlags[candidate.objectIndex] != 0u;
    }
    else
    {
        visible = (g_RenderableShadowCascadeMasks[candidate.objectIndex] &
                   candidate.visibilityMask) != 0u;
    }

    if (!visible)
    {
        return;
    }

    const IndirectCommandDesc desc = g_CommandDescs[candidate.commandIndex];
    uint visibleSlot = 0u;
    InterlockedAdd(g_CommandCountsRW[candidate.commandIndex], 1u, visibleSlot);
    if (visibleSlot < desc.maxVisibleCount)
    {
        g_VisibleObjectIndicesRW[desc.visibleOffset + visibleSlot] = candidate.objectIndex;
    }
}
