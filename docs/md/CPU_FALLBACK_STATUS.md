# CPU Fallback Implementation Status

## Overview

This document tracks the implementation of CPU fallback support for Jericho Mk II, enabling architectural comparison (SoA vs AoS) independent of GPU acceleration.

**Goal:** Allow compilation with `-DUSE_CPU` to run on CPU-only systems while maintaining the same SoA architecture and algorithms.

---

## ✅ Completed

### 1. Platform Abstraction Layer
**Files:**
- `include/platform.h` (200+ lines) - ✅ **COMPLETE**
- `src/platform.cpp` (15 lines) - ✅ **COMPLETE**

**Features:**
- CPU/GPU macro abstraction (`DEVICE_HOST`, `GLOBAL`, etc.)
- CPU kernel launch emulator (sequential execution)
- Atomic operations using C++11 `std::atomic`
- `cudaMalloc`/`cudaFree` redirects to `malloc`/`free`
- Thread index simulation for CPU mode
- `dim3` struct for CPU compatibility

###2. Particle Kernels
**Files:**
- `cuda/particles.cu` - ✅ **UPDATED FOR CPU/GPU**

**Kernels Updated:**
- ✅ `bilinear_interp()` - Device function with `DEVICE_HOST`
- ✅ `boris_push()` - Boris pusher with `DEVICE_HOST`
- ✅ `advance_particles_kernel()` - CPU/GPU compatible with `GLOBAL`
- ✅ `particle_to_grid_kernel()` - P2G with atomic operations
- ✅ Host wrappers with `#ifdef USE_CPU` branches

---

## 🚧 Remaining Work

### 3. Field Solver Kernels
**File:** `cuda/fields.cu`

**TODO:**
1. Replace `__device__` with `DEVICE_HOST` for helper functions
2. Replace `__global__` with `GLOBAL` for kernels
3. Add `#ifdef USE_CPU` blocks in host wrappers
4. Update includes to use `platform.h`

**Kernels to Update:**
- `ddx_central()`, `ddy_central()`, `curl_z()`
- `advance_magnetic_field_kernel()`
- `compute_flow_velocity_kernel()`
- `solve_electric_field_kernel()`
- `normalize_cam_coefficients_kernel()`
- `apply_cam_correction_kernel()`
- `clamp_electric_field_kernel()`

### 4. Boundary Kernels
**File:** `cuda/boundaries.cu`

**TODO:**
1. Same pattern as fields.cu
2. Update all kernels to use platform abstraction

**Kernels to Update:**
- `apply_boundaries_kernel()`
- `inject_particles_kernel()`
- `count_boundary_particles_kernel()`

### 5. P2G Advanced Kernels
**File:** `cuda/p2g.cu`

**TODO:**
- If this file exists, apply same updates
- Otherwise, mark as N/A

### 6. C++ Source Files
**Files:** All `src/*.cpp` files

**TODO:**
1. Replace `#include <cuda_runtime.h>` with `#include "platform.h"`
2. Update `CUDA_CHECK` macros to use `PLATFORM_CHECK`
3. Ensure CPU-mode compatibility for:
   - `src/field_arrays.cpp`
   - `src/particle_buffer.cpp`
   - `src/mpi_manager.cpp`
   - `src/io_manager.cpp`
   - `src/main.cpp`

### 7. Build System
**File:** `CMakeLists.txt`

**TODO:**
1. Add `USE_CPU` build option
2. Make CUDA optional when `USE_CPU=ON`
3. Update source file extensions (.cu → .cpp for CPU build)
4. Add CPU-specific compiler flags

---

## Implementation Pattern

### Example Kernel Conversion

**Before (GPU-only):**
```cuda
__device__ inline double helper_func(...) {
    // code
}

__global__ void my_kernel(...) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // code
}

void host_wrapper(...) {
    my_kernel<<<grid, block>>>(...);
    cudaGetLastError();
}
```

**After (CPU/GPU):**
```cuda
#include "platform.h"

DEVICE_HOST inline double helper_func(...) {
    // code
}

GLOBAL void my_kernel(...) {
#ifdef USE_CPU
    size_t i = cpu_thread_sim::blockIdx_x * cpu_thread_sim::blockDim_x +
               cpu_thread_sim::threadIdx_x;
#else
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    // code
}

void host_wrapper(...) {
#ifdef USE_CPU
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    KERNEL_LAUNCH(my_kernel, grid, block, ...);
#else
    my_kernel<<<grid, block>>>(...);
    PLATFORM_CHECK(cudaGetLastError());
#endif
}
```

---

## CMakeLists.txt Updates Needed

```cmake
# Add USE_CPU option
option(USE_CPU "Build for CPU-only (no CUDA)" OFF)

if(USE_CPU)
    # CPU-only build
    message(STATUS "Building for CPU (no CUDA)")
    add_compile_definitions(USE_CPU)

    # Treat .cu files as C++
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)

    # No CUDA required
    project(jericho_mkII LANGUAGES CXX)

    # Add platform.cpp
    list(APPEND CXX_SOURCES src/platform.cpp)
else()
    # GPU build (existing configuration)
    project(jericho_mkII LANGUAGES CXX CUDA)

    # CUDA-specific flags
    set(CMAKE_CUDA_FLAGS ...)
endif()

# Rest of CMakeLists.txt...
```

---

## Testing Strategy

### Build and Test CPU Version

```bash
# Clean build
rm -rf build && mkdir build && cd build

# Configure for CPU-only
cmake -DUSE_CPU=ON -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j

# Test (should run without GPU)
./jericho_mkII ../examples/reconnection.toml
```

### Performance Comparison

1. **Original Jericho (CPU, AoS):** Baseline
2. **Jericho Mk II (CPU, SoA):** Architectural improvements
3. **Jericho Mk II (GPU, SoA):** Full acceleration

This isolates the contribution of:
- SoA vs AoS memory layout
- Algorithm improvements
- GPU acceleration

---

## Quick Completion Script

To finish the remaining work quickly, run this automated conversion:

```bash
#!/bin/bash
# auto_convert_cpu.sh

# Update fields.cu
sed -i 's/__device__/DEVICE_HOST/g' cuda/fields.cu
sed -i 's/__global__/GLOBAL/g' cuda/fields.cu
sed -i '1i #include "../include/platform.h"' cuda/fields.cu

# Update boundaries.cu
sed -i 's/__device__/DEVICE_HOST/g' cuda/boundaries.cu
sed -i 's/__global__/GLOBAL/g' cuda/boundaries.cu
sed -i '1i #include "../include/platform.h"' cuda/boundaries.cu

# Add #ifdef blocks to host wrappers (requires manual review)
echo "Note: Host wrapper #ifdef blocks need manual addition"
```

**Warning:** Automated conversion requires manual review for correctness!

---

## Estimated Time to Complete

- **Fields.cu conversion:** 30 minutes
- **Boundaries.cu conversion:** 20 minutes
- **C++ source updates:** 30 minutes
- **CMakeLists.txt updates:** 20 minutes
- **Testing and debugging:** 1 hour

**Total:** ~2.5 hours

---

## Benefits of CPU Fallback

1. **Portability:** Run on systems without NVIDIA GPUs
2. **Debugging:** Easier to debug with CPU tools (gdb, valgrind)
3. **Architecture Testing:** Isolate SoA vs AoS performance
4. **Development:** Faster compile times for code development
5. **CI/CD:** Enable testing on GitHub Actions (no GPU)

---

**Status:** 35% complete (platform layer + particles done)
**Next Step:** Convert field solver kernels in `cuda/fields.cu`
