#ifndef CRESSIM_NEO_SRC_ENGINE_RUNTIME_INTERNAL_H
#define CRESSIM_NEO_SRC_ENGINE_RUNTIME_INTERNAL_H

#include "engine/shared_buffer.h"

#include <memory>

namespace cressim::neo::engine
{

class Runtime;

class RuntimeInternalAccess
{
public:
    static std::shared_ptr<void> retainSharedBufferLease(const Runtime &runtime,
                                                         SharedBufferHandle handle);
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_SRC_ENGINE_RUNTIME_INTERNAL_H
