#ifndef CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI
#define CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI

#ifndef CRESSIM_NATIVE_FLOAT_BUFFER_ATOMICS
#    define CRESSIM_NATIVE_FLOAT_BUFFER_ATOMICS 0
#endif

#if CRESSIM_NATIVE_FLOAT_BUFFER_ATOMICS

#    include "../structured_buffer_compat.hlsli"

// Each logical correction entry stays float4-shaped so the native and CAS paths
// share the same 16-byte record layout and clear one whole entry at a time.
#    define CRESSIM_RW_ATOMIC_FLOAT_BUFFER(name) CRESSIM_RW_STRUCTURED_BUFFER(float4, name)

static const uint kCressimSpirvScopeDevice = 1u;
static const uint kCressimSpirvMemorySemanticsAcquireReleaseUniform = 0x48u;

[[vk::ext_capability(6033)]]
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_instruction(6035)]]
float CressimAtomicFloatAddNative([[vk::ext_reference]] float pointer, uint scope,
                                  uint semantics, float value);

#    define CRESSIM_ATOMIC_ADD_FLOAT3_CAS(buffer, index, deltaValue)                             \
        do                                                                                       \
        {                                                                                        \
            const float3 cressimAtomicDelta = (deltaValue);                                      \
            CressimAtomicFloatAddNative(CRESSIM_SB_REF((buffer), index).x,                       \
                                        kCressimSpirvScopeDevice,                                \
                                        kCressimSpirvMemorySemanticsAcquireReleaseUniform,       \
                                        cressimAtomicDelta.x);                                   \
            CressimAtomicFloatAddNative(CRESSIM_SB_REF((buffer), index).y,                       \
                                        kCressimSpirvScopeDevice,                                \
                                        kCressimSpirvMemorySemanticsAcquireReleaseUniform,       \
                                        cressimAtomicDelta.y);                                   \
            CressimAtomicFloatAddNative(CRESSIM_SB_REF((buffer), index).z,                       \
                                        kCressimSpirvScopeDevice,                                \
                                        kCressimSpirvMemorySemanticsAcquireReleaseUniform,       \
                                        cressimAtomicDelta.z);                                   \
        } while (false)

#    define CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(buffer, index) \
        (CRESSIM_SB_LOAD((buffer), index).xyz)

#    define CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(buffer, index) \
        CRESSIM_SB_STORE((buffer), index, float4(0.0, 0.0, 0.0, 0.0))

#else

#    include "../byte_address_buffer_compat.hlsli"

// Each logical correction entry is stored as a padded raw float4 record so the
// CAS path keeps 16-byte alignment and can clear the whole entry with one Store4.
#    define CRESSIM_RW_ATOMIC_FLOAT_BUFFER(name) CRESSIM_RW_BYTE_ADDRESS_BUFFER(name)
#    define CRESSIM_ATOMIC_FLOAT_STRIDE 16u
#    define CRESSIM_ATOMIC_FLOAT_COMPONENT_SIZE 4u
#    define CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, component) \
        ((index) * CRESSIM_ATOMIC_FLOAT_STRIDE + (component) * CRESSIM_ATOMIC_FLOAT_COMPONENT_SIZE)

#    define CRESSIM_ATOMIC_ADD_FLOAT_CAS(buffer, byteOffset, deltaValue)                    \
        do                                                                                  \
        {                                                                                   \
            uint cressimAtomicExpected = CRESSIM_BAB_LOAD((buffer), byteOffset);            \
            [allow_uav_condition]                                                           \
            for (;;)                                                                        \
            {                                                                               \
                const uint cressimAtomicDesired =                                           \
                    asuint(asfloat(cressimAtomicExpected) + (deltaValue));                  \
                uint cressimAtomicOriginal = 0u;                                            \
                CRESSIM_BAB_INTERLOCKED_COMPARE_EXCHANGE((buffer), byteOffset,              \
                                                         cressimAtomicExpected,             \
                                                         cressimAtomicDesired,              \
                                                         cressimAtomicOriginal);            \
                if (cressimAtomicOriginal == cressimAtomicExpected)                         \
                {                                                                           \
                    break;                                                                  \
                }                                                                           \
                cressimAtomicExpected = cressimAtomicOriginal;                              \
            }                                                                               \
        } while (false)

#    define CRESSIM_ATOMIC_ADD_FLOAT3_CAS(buffer, index, deltaValue)                                 \
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

#    define CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(buffer, index)                                           \
        float3(asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 0u))), \
               asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 1u))), \
               asfloat(CRESSIM_BAB_LOAD((buffer), CRESSIM_ATOMIC_FLOAT_COMPONENT_OFFSET(index, 2u))))

#    define CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(buffer, index) \
        CRESSIM_BAB_STORE4((buffer), (index) * CRESSIM_ATOMIC_FLOAT_STRIDE, uint4(0u, 0u, 0u, 0u))

#endif

#endif // CRESSIM_NEO_PHYSICS_ATOMIC_FLOAT_HLSLI
