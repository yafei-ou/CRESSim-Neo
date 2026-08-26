#ifndef CRESSIM_NEO_COMMON_EXPORT_H
#define CRESSIM_NEO_COMMON_EXPORT_H

#include "cressim_neo/visibility.h"

/// @file export.h
/// @brief DLL export and import symbol visibility macros for the Common module.

#if defined(CRESSIM_NEO_STATIC)
/// @brief Symbol export macro for the Common module in static or shared builds.
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
