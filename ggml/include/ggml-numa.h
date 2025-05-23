#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// numa strategies
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED   = 0,
    GGML_NUMA_STRATEGY_SMT        = 1,
    GGML_NUMA_STRATEGY_PHYSICAL   = 2,
    GGML_NUMA_STRATEGY_COUNT
};

#define GGML_BACKEND_NUMA_NAME "NUMA"

// backend API
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_numa_reg(ggml_backend_reg_t reg_cpu);

GGML_BACKEND_API bool ggml_backend_numa_enable(ggml_backend_reg_t reg, ggml_numa_strategy numa_strategy);

#ifdef  __cplusplus
}
#endif
