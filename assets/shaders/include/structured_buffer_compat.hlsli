#ifndef CRESSIM_NEO_STRUCTURED_BUFFER_COMPAT_HLSLI
#define CRESSIM_NEO_STRUCTURED_BUFFER_COMPAT_HLSLI

#ifdef METAL
#    define CRESSIM_STRUCTURED_BUFFER(type, name) \
        struct name##_StructuredBufferElement     \
        {                                         \
            type value;                           \
        };                                        \
        StructuredBuffer<name##_StructuredBufferElement> name

#    define CRESSIM_RW_STRUCTURED_BUFFER(type, name) \
        struct name##_RWStructuredBufferElement      \
        {                                            \
            type value;                              \
        };                                           \
        RWStructuredBuffer<name##_RWStructuredBufferElement> name

#    define CRESSIM_SB_REF(name, index) ((name)[index].value)
#else
#    define CRESSIM_STRUCTURED_BUFFER(type, name) StructuredBuffer<type> name
#    define CRESSIM_RW_STRUCTURED_BUFFER(type, name) RWStructuredBuffer<type> name
#    define CRESSIM_SB_REF(name, index) ((name)[index])
#endif

#define CRESSIM_SB_LOAD(name, index) (CRESSIM_SB_REF(name, index))
#define CRESSIM_SB_STORE(name, index, value) (CRESSIM_SB_REF(name, index) = (value))

#endif // CRESSIM_NEO_STRUCTURED_BUFFER_COMPAT_HLSLI
