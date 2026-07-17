#include "physics/physics_curve_render_dispatch_constants.hlsli"
#include "physics/core/physics_base.hlsli"

struct GpuCurveRenderDescriptor
{
    uint particleIndexStart;
    uint particleCount;
    uint vertexBase;
    uint vertexCount;
    uint radialResolution;
    uint environmentIndex;
    float radius;
};

struct GpuBodyAabb
{
    float4 minBounds;
    float4 maxBounds;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuCurveRenderDescriptor, g_CurveRenderDescriptors);
CRESSIM_STRUCTURED_BUFFER(uint, g_CurveRenderParticleIndices);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_CurveRenderPositionsRW);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_CurveRenderNormalsRW);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyAabb, g_CurveWorldAabbsRW);

static const float kPi = 3.14159265358979323846;

float3 normalizeOrFallback(float3 value, float3 fallback)
{
    const float lenSq = dot(value, value);
    if (lenSq <= 1.0e-12)
    {
        return fallback;
    }
    return value * rsqrt(lenSq);
}

float3 chooseReferenceAxis(float3 tangent)
{
    return abs(tangent.y) < 0.95 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
}

float3 rotateAroundAxis(float3 value, float3 axis, float sinAngle, float cosAngle)
{
    return value * cosAngle + cross(axis, value) * sinAngle +
           axis * dot(axis, value) * (1.0 - cosAngle);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint curveIndex = dispatchThreadID.x;
    if (curveIndex >= curveCount)
    {
        return;
    }

    const GpuCurveRenderDescriptor descriptor =
        CRESSIM_SB_LOAD(g_CurveRenderDescriptors, curveIndex);
    if (descriptor.particleCount < 2u || descriptor.radialResolution < 3u ||
        descriptor.vertexCount == 0u)
    {
        GpuBodyAabb emptyAabb;
        emptyAabb.minBounds = float4(0.0, 0.0, 0.0, 0.0);
        emptyAabb.maxBounds = float4(0.0, 0.0, 0.0, 0.0);
        CRESSIM_SB_STORE(g_CurveWorldAabbsRW, curveIndex, emptyAabb);
        return;
    }

    float3 prevCenter =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass,
                        CRESSIM_SB_LOAD(g_CurveRenderParticleIndices,
                                        descriptor.particleIndexStart)).xyz;
    float3 nextCenter =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass,
                        CRESSIM_SB_LOAD(g_CurveRenderParticleIndices,
                                        descriptor.particleIndexStart + 1u)).xyz;
    float3 prevTangent = normalizeOrFallback(nextCenter - prevCenter, float3(1.0, 0.0, 0.0));
    float3 referenceAxis = chooseReferenceAxis(prevTangent);
    float3 normal =
        normalizeOrFallback(cross(referenceAxis, prevTangent), float3(0.0, 0.0, 1.0));
    float3 binormal = normalizeOrFallback(cross(prevTangent, normal), float3(0.0, 1.0, 0.0));

    float3 boundsMin = prevCenter - descriptor.radius;
    float3 boundsMax = prevCenter + descriptor.radius;

    [loop]
    for (uint sampleIndex = 0u; sampleIndex < descriptor.particleCount; ++sampleIndex)
    {
        const uint particleIndex = CRESSIM_SB_LOAD(g_CurveRenderParticleIndices,
                                                   descriptor.particleIndexStart + sampleIndex);
        const float3 center = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
        float3 tangent = prevTangent;
        if (sampleIndex == 0u)
        {
            const float3 forward =
                CRESSIM_SB_LOAD(g_ParticlePositionsInvMass,
                                CRESSIM_SB_LOAD(g_CurveRenderParticleIndices,
                                                descriptor.particleIndexStart + 1u)).xyz;
            tangent = normalizeOrFallback(forward - center, prevTangent);
        }
        else if (sampleIndex + 1u >= descriptor.particleCount)
        {
            tangent = normalizeOrFallback(center - prevCenter, prevTangent);
        }
        else
        {
            const float3 forward =
                CRESSIM_SB_LOAD(g_ParticlePositionsInvMass,
                                CRESSIM_SB_LOAD(g_CurveRenderParticleIndices,
                                                descriptor.particleIndexStart + sampleIndex + 1u)).xyz;
            tangent = normalizeOrFallback(forward - prevCenter, prevTangent);
        }

        if (sampleIndex > 0u)
        {
            const float3 axis = cross(prevTangent, tangent);
            const float axisLenSq = dot(axis, axis);
            if (axisLenSq > 1.0e-12)
            {
                const float axisLen = sqrt(axisLenSq);
                const float3 unitAxis = axis / axisLen;
                const float cosAngle = clamp(dot(prevTangent, tangent), -1.0, 1.0);
                const float sinAngle = min(axisLen, 1.0);
                normal = rotateAroundAxis(normal, unitAxis, sinAngle, cosAngle);
            }
            normal =
                normalizeOrFallback(normal - tangent * dot(normal, tangent),
                                    normalizeOrFallback(cross(chooseReferenceAxis(tangent), tangent),
                                                        float3(0.0, 0.0, 1.0)));
            binormal = normalizeOrFallback(cross(tangent, normal), binormal);
        }

        boundsMin = min(boundsMin, center - descriptor.radius);
        boundsMax = max(boundsMax, center + descriptor.radius);

        [loop]
        for (uint radialIndex = 0u; radialIndex < descriptor.radialResolution; ++radialIndex)
        {
            const uint vertexIndex =
                descriptor.vertexBase + sampleIndex * descriptor.radialResolution + radialIndex;
            if (vertexIndex >= descriptor.vertexBase + descriptor.vertexCount)
            {
                continue;
            }

            const float angle = (2.0 * kPi * radialIndex) / descriptor.radialResolution;
            const float s = sin(angle);
            const float c = cos(angle);
            const float3 radialDirection = c * normal + s * binormal;
            const float3 vertexPosition = center + descriptor.radius * radialDirection;

            CRESSIM_SB_STORE(g_CurveRenderPositionsRW, vertexIndex, float4(vertexPosition, 1.0));
            CRESSIM_SB_STORE(g_CurveRenderNormalsRW, vertexIndex,
                             float4(normalizeOrFallback(radialDirection, normal), 0.0));
        }

        prevCenter = center;
        prevTangent = tangent;
    }

    GpuBodyAabb aabb;
    aabb.minBounds = float4(boundsMin, 0.0);
    aabb.maxBounds = float4(boundsMax, 0.0);
    CRESSIM_SB_STORE(g_CurveWorldAabbsRW, curveIndex, aabb);
}
