#ifndef CRESSIM_NEO_VISIBILITY_H
#define CRESSIM_NEO_VISIBILITY_H

#if defined(__GNUC__) || defined(__clang__)
#define CRESSIM_NEO_LOCAL __attribute__((visibility("hidden")))
#else
#define CRESSIM_NEO_LOCAL
#endif

#endif // CRESSIM_NEO_VISIBILITY_H
