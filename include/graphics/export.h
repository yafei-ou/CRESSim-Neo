#ifndef CRESSIM_NEO_GRAPHICS_EXPORT_H
#define CRESSIM_NEO_GRAPHICS_EXPORT_H

#include "cressim_neo/visibility.h"

/// @file export.h
/// @brief Symbol export macros for the CRESSim-Neo graphics rendering engine library.

#if defined(CRESSIM_NEO_STATIC)
/// @brief Macro for exporting/importing symbols from the CRESSim-Neo graphics shared library.
#define CRESSIM_NEO_GRAPHICS_API
#elif defined(_WIN32)
#if defined(CRESSIM_NEO_GRAPHICS_EXPORT)
#define CRESSIM_NEO_GRAPHICS_API __declspec(dllexport)
#else
#define CRESSIM_NEO_GRAPHICS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define CRESSIM_NEO_GRAPHICS_API __attribute__((visibility("default")))
#else
#define CRESSIM_NEO_GRAPHICS_API
#endif

#endif // CRESSIM_NEO_GRAPHICS_EXPORT_H
