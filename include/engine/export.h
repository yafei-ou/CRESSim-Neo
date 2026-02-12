#ifndef CRESSIM_NEO_ENGINE_EXPORT_H
#define CRESSIM_NEO_ENGINE_EXPORT_H

#if defined(_WIN32)
#if defined(CRESSIM_NEO_STATIC)
#define CRESSIM_NEO_ENGINE_API
#elif defined(CRESSIM_NEO_ENGINE_EXPORT)
#define CRESSIM_NEO_ENGINE_API __declspec(dllexport)
#else
#define CRESSIM_NEO_ENGINE_API __declspec(dllimport)
#endif
#else
#define CRESSIM_NEO_ENGINE_API
#endif

#endif // CRESSIM_NEO_ENGINE_EXPORT_H
