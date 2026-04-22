#ifndef CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI
#define CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI

// On Metal, we implement the raw-buffer compatibility layer using structured
// uint buffers to avoid MSL type/name collisions during translation.
// The CPU side still creates these resources as raw buffers, which has worked
// correctly with the current backend.

#ifdef METAL
#    include "structured_buffer_compat.hlsli"

#    define CRESSIM_BYTE_ADDRESS_BUFFER(name) CRESSIM_STRUCTURED_BUFFER(uint, name)
#    define CRESSIM_RW_BYTE_ADDRESS_BUFFER(name) CRESSIM_RW_STRUCTURED_BUFFER(uint, name)
#    define CRESSIM_GLOBALLYCOHERENT_RW_BYTE_ADDRESS_BUFFER(name) \
        CRESSIM_GLOBALLYCOHERENT_RW_STRUCTURED_BUFFER(uint, name)

#    define CRESSIM_BAB_WORD_INDEX(byteOffset) ((byteOffset) / 4u)

#    define CRESSIM_BAB_LOAD(name, byteOffset) \
        (CRESSIM_SB_LOAD((name), CRESSIM_BAB_WORD_INDEX(byteOffset)))

#    define CRESSIM_BAB_STORE(name, byteOffset, value) \
        CRESSIM_SB_STORE((name), CRESSIM_BAB_WORD_INDEX(byteOffset), (value))

#    define CRESSIM_BAB_STORE4(name, byteOffset, value)                                      \
        do                                                                                   \
        {                                                                                    \
            const uint4 cressimBabStore4Value = (value);                                     \
            const uint cressimBabStore4Index = CRESSIM_BAB_WORD_INDEX(byteOffset);           \
            CRESSIM_SB_STORE((name), cressimBabStore4Index + 0u, cressimBabStore4Value.x);   \
            CRESSIM_SB_STORE((name), cressimBabStore4Index + 1u, cressimBabStore4Value.y);   \
            CRESSIM_SB_STORE((name), cressimBabStore4Index + 2u, cressimBabStore4Value.z);   \
            CRESSIM_SB_STORE((name), cressimBabStore4Index + 3u, cressimBabStore4Value.w);   \
        } while (false)

#    define CRESSIM_BAB_INTERLOCKED_COMPARE_EXCHANGE(                                         \
        name, byteOffset, compareValue, value, original)                                      \
        InterlockedCompareExchange(CRESSIM_SB_REF((name), CRESSIM_BAB_WORD_INDEX(byteOffset)), \
                                   compareValue, value, original)
#else
#    define CRESSIM_BYTE_ADDRESS_BUFFER(name) ByteAddressBuffer name
#    define CRESSIM_RW_BYTE_ADDRESS_BUFFER(name) RWByteAddressBuffer name
#    define CRESSIM_GLOBALLYCOHERENT_RW_BYTE_ADDRESS_BUFFER(name) \
        globallycoherent RWByteAddressBuffer name

#    define CRESSIM_BAB_LOAD(name, byteOffset) ((name).Load(byteOffset))
#    define CRESSIM_BAB_STORE(name, byteOffset, value) ((name).Store(byteOffset, value))
#    define CRESSIM_BAB_STORE4(name, byteOffset, value) ((name).Store4(byteOffset, value))
#    define CRESSIM_BAB_INTERLOCKED_COMPARE_EXCHANGE(name, byteOffset, compareValue, value, original) \
        ((name).InterlockedCompareExchange(byteOffset, compareValue, value, original))
#endif

#endif // CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI
