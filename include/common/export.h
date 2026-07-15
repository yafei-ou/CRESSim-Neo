#ifndef CRESSIM_NEO_COMMON_EXPORT_H
#define CRESSIM_NEO_COMMON_EXPORT_H

#if defined(CRESSIM_NEO_STATIC)
#define CRESSIM_NEO_COMMON_API
#elif defined(_WIN32)
#if defined(CRESSIM_NEO_COMMON_EXPORT)
#define CRESSIM_NEO_COMMON_API __declspec(dllexport)
#else
#define CRESSIM_NEO_COMMON_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define CRESSIM_NEO_COMMON_API __attribute__((visibility("default")))
#else
#define CRESSIM_NEO_COMMON_API
#endif

#endif // CRESSIM_NEO_COMMON_EXPORT_H
