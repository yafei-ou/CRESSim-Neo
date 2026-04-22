#ifndef CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI
#define CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI

#define CRESSIM_BYTE_ADDRESS_BUFFER(name) ByteAddressBuffer name
#define CRESSIM_RW_BYTE_ADDRESS_BUFFER(name) RWByteAddressBuffer name
#define CRESSIM_GLOBALLYCOHERENT_RW_BYTE_ADDRESS_BUFFER(name) globallycoherent RWByteAddressBuffer name

#define CRESSIM_BAB_LOAD(name, byteOffset) ((name).Load(byteOffset))
#define CRESSIM_BAB_STORE(name, byteOffset, value) ((name).Store(byteOffset, value))
#define CRESSIM_BAB_STORE4(name, byteOffset, value) ((name).Store4(byteOffset, value))
#define CRESSIM_BAB_INTERLOCKED_COMPARE_EXCHANGE(name, byteOffset, compareValue, value, original) \
    ((name).InterlockedCompareExchange(byteOffset, compareValue, value, original))

#endif // CRESSIM_NEO_BYTE_ADDRESS_BUFFER_COMPAT_HLSLI
