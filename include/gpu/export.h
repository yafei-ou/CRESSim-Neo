#ifndef CRESSIM_NEO_GPU_EXPORT_H
#define CRESSIM_NEO_GPU_EXPORT_H

#if defined(_WIN32)

#if defined(CRESSIM_NEO_STATIC)
#define CRESSIM_NEO_GPU_API
#elif defined(CRESSIM_NEO_GPU_EXPORT)
#define CRESSIM_NEO_GPU_API __declspec(dllexport)
#else
#define CRESSIM_NEO_GPU_API __declspec(dllimport)
#endif

#else

#if defined(CRESSIM_NEO_GPU_EXPORT)
#define CRESSIM_NEO_GPU_API __attribute__((visibility("default")))
#else
#define CRESSIM_NEO_GPU_API
#endif

#endif

#endif // CRESSIM_NEO_GPU_EXPORT_H
