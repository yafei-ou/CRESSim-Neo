#ifndef CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI
#define CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI

#include "include/byte_address_buffer_compat.hlsli"

// Each logical correction entry is stored as a padded raw float4 record so the
// CAS path keeps 16-byte alignment and can clear the whole entry with one Store4.
#define CRESSIM_ATOMIC_FLOAT_STRIDE 16u
#define CRESSIM_ATOMIC_FLOAT_COMPONENT_SIZE 4u
#define CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, component) \
    ((index) * CRESSIM_ATOMIC_FLOAT_STRIDE + (component) * CRESSIM_ATOMIC_FLOAT_COMPONENT_SIZE)

#define CRESSIM_ATOMIC_ADD_FLOAT_CAS(buffer, byteOffset, deltaValue)                     \
    do                                                                                   \
    {                                                                                    \
        uint cressimAtomicExpected = CRESSIM_BAB_LOAD((buffer), byteOffset);             \
        [allow_uav_condition]                                                            \
        for (;;)                                                                         \
        {                                                                                \
            const uint cressimAtomicDesired =                                            \
                asuint(asfloat(cressimAtomicExpected) + (deltaValue));                   \
            uint cressimAtomicOriginal = 0u;                                             \
            CRESSIM_BAB_INTERLOCKED_COMPARE_EXCHANGE((buffer), byteOffset,               \
                                                     cressimAtomicExpected,              \
                                                     cressimAtomicDesired,               \
                                                     cressimAtomicOriginal);             \
            if (cressimAtomicOriginal == cressimAtomicExpected)                          \
            {                                                                            \
                break;                                                                   \
            }                                                                            \
            cressimAtomicExpected = cressimAtomicOriginal;                               \
        }                                                                                \
    } while (false)

#define CRESSIM_ATOMIC_ADD_FLOAT3_CAS(buffer, index, deltaValue)                                 \
    do                                                                                           \
    {                                                                                            \
        const float3 cressimAtomicDelta = (deltaValue);                                          \
        CRESSIM_ATOMIC_ADD_FLOAT_CAS((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 0u), \
                                     cressimAtomicDelta.x);                                      \
        CRESSIM_ATOMIC_ADD_FLOAT_CAS((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 1u), \
                                     cressimAtomicDelta.y);                                      \
        CRESSIM_ATOMIC_ADD_FLOAT_CAS((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 2u), \
                                     cressimAtomicDelta.z);                                      \
    } while (false)

#define CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(buffer, index)                                           \
    float3(asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 0u))), \
           asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 1u))), \
           asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 2u))))

#define CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(buffer, index) \
    CRESSIM_BAB_STORE4((buffer), (index) * CRESSIM_ATOMIC_FLOAT_STRIDE, uint4(0u, 0u, 0u, 0u))

#endif // CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI
