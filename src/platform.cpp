/**
 * @file platform.cpp
 * @brief CPU thread simulation globals
 */

#include "platform.h"

#ifdef USE_CPU

namespace cpu_thread_sim {
thread_local dim3 blockIdx(0, 0, 0);
thread_local dim3 blockDim(1, 1, 1);
thread_local dim3 threadIdx(0, 0, 0);
thread_local dim3 gridDim(1, 1, 1);
} // namespace cpu_thread_sim

#endif
