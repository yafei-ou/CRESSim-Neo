#ifndef CRESSIM_NEO_PHYSICS_FLUID_ANISOTROPY_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_ANISOTROPY_COMMON_HLSLI

float FluidAnisotropyCube(float x)
{
    return x * x * x;
}

float FluidAnisotropyWeight(float distance, float inverseRadius)
{
    return 1.0 - FluidAnisotropyCube(distance * inverseRadius);
}

float3 FluidAnisotropyClamp(float3 value, float minValue, float maxValue)
{
    return float3(clamp(value.x, minValue, maxValue),
                  clamp(value.y, minValue, maxValue),
                  clamp(value.z, minValue, maxValue));
}

float3x3 FluidAnisotropyIdentity()
{
    return float3x3(1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
}

void FluidAnisotropyJacobiRotate(inout float3x3 A, inout float3x3 R, uint p, uint q)
{
    if (abs(A[p][q]) <= 1.0e-20)
    {
        return;
    }

    float d = (A[p][p] - A[q][q]) / (2.0 * A[p][q]);
    float t = 1.0 / (abs(d) + sqrt(d * d + 1.0));
    if (d < 0.0)
    {
        t = -t;
    }

    const float c = rsqrt(t * t + 1.0);
    const float s = t * c;

    A[p][p] += t * A[p][q];
    A[q][q] -= t * A[p][q];
    A[p][q] = 0.0;
    A[q][p] = 0.0;

    [unroll]
    for (uint k = 0u; k < 3u; ++k)
    {
        if (k == p || k == q)
        {
            continue;
        }

        const float Akp = c * A[k][p] + s * A[k][q];
        const float Akq = -s * A[k][p] + c * A[k][q];
        A[k][p] = Akp;
        A[p][k] = Akp;
        A[k][q] = Akq;
        A[q][k] = Akq;
    }

    [unroll]
    for (uint k = 0u; k < 3u; ++k)
    {
        const float Rkp = c * R[k][p] + s * R[k][q];
        const float Rkq = -s * R[k][p] + c * R[k][q];
        R[k][p] = Rkp;
        R[k][q] = Rkq;
    }
}

void FluidAnisotropyEigenDecomposition(float3x3 covariance, out float3x3 basis,
                                       out float3 eigenValues)
{
    basis = FluidAnisotropyIdentity();

    [unroll]
    for (uint iteration = 0u; iteration < 4u; ++iteration)
    {
        uint pairIndex = 0u;
        float maxValue = abs(covariance[0][1]);
        const float a02 = abs(covariance[0][2]);
        if (a02 > maxValue)
        {
            pairIndex = 1u;
            maxValue = a02;
        }

        const float a12 = abs(covariance[1][2]);
        if (a12 > maxValue)
        {
            pairIndex = 2u;
            maxValue = a12;
        }

        if (maxValue < 1.0e-15)
        {
            break;
        }

        if (pairIndex == 0u)
        {
            FluidAnisotropyJacobiRotate(covariance, basis, 0u, 1u);
        }
        else if (pairIndex == 1u)
        {
            FluidAnisotropyJacobiRotate(covariance, basis, 0u, 2u);
        }
        else
        {
            FluidAnisotropyJacobiRotate(covariance, basis, 1u, 2u);
        }
    }

    eigenValues = float3(max(covariance[0][0], 0.0),
                         max(covariance[1][1], 0.0),
                         max(covariance[2][2], 0.0));
}

void FluidAnisotropySwap(inout float a, inout float b)
{
    const float tmp = a;
    a = b;
    b = tmp;
}

void FluidAnisotropySwap(inout float3 a, inout float3 b)
{
    const float3 tmp = a;
    a = b;
    b = tmp;
}

void FluidAnisotropySortDescending(inout float3 eigenValues, inout float3 axis1,
                                   inout float3 axis2, inout float3 axis3)
{
    if (eigenValues.x < eigenValues.y)
    {
        FluidAnisotropySwap(eigenValues.x, eigenValues.y);
        FluidAnisotropySwap(axis1, axis2);
    }
    if (eigenValues.x < eigenValues.z)
    {
        FluidAnisotropySwap(eigenValues.x, eigenValues.z);
        FluidAnisotropySwap(axis1, axis3);
    }
    if (eigenValues.y < eigenValues.z)
    {
        FluidAnisotropySwap(eigenValues.y, eigenValues.z);
        FluidAnisotropySwap(axis2, axis3);
    }
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_ANISOTROPY_COMMON_HLSLI
