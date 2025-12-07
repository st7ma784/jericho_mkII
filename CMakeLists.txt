cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Build mode option
option(USE_CPU "Build for CPU-only (no CUDA required)" OFF)

if(USE_CPU)
    project(jericho_mkII LANGUAGES C CXX)
    message(STATUS "Building for CPU (CUDA disabled)")
    add_compile_definitions(USE_CPU)
else()
    project(jericho_mkII LANGUAGES C CXX CUDA)
    message(STATUS "Building for GPU (CUDA enabled)")
endif()

# ============================================================================
# Project metadata
# ============================================================================
set(PROJECT_VERSION_MAJOR 2)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

# ============================================================================
# Build options
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_CUDA_AWARE_MPI "Use CUDA-aware MPI (requires special MPI build)" ON)
option(ENABLE_PROFILING "Enable NVTX profiling markers" OFF)

# ============================================================================
# C++ and CUDA standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA architecture detection
# ============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect GPU architecture if not specified
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ============================================================================
# Find dependencies
# ============================================================================

# MPI
find_package(MPI REQUIRED)
if(USE_CUDA_AWARE_MPI)
    message(STATUS "CUDA-aware MPI enabled - ensure your MPI build supports this!")
    add_compile_definitions(USE_CUDA_AWARE_MPI)
endif()

# OpenMP (for parallelization)
find_package(OpenMP REQUIRED)

# CUDA (optional for CPU build)
if(NOT USE_CPU)
    find_package(CUDAToolkit REQUIRED)
endif()

# HDF5 (for parallel I/O)
find_package(HDF5 REQUIRED COMPONENTS C)
if(HDF5_IS_PARALLEL)
    message(STATUS "Found parallel HDF5")
else()
    message(WARNING "HDF5 is not parallel - parallel I/O may not work correctly")
endif()

# ============================================================================
# Compiler flags
# ============================================================================

# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")  # Fast math for performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")  # Verbose PTX info
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")  # -G enables device debugging
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")

if(ENABLE_PROFILING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_NVTX")
    list(APPEND CUDA_LIBS nvToolsExt)
endif()

# ============================================================================
# Include directories
# ============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

# ============================================================================
# Source files
# ============================================================================

# CUDA source files
set(CUDA_SOURCES
    cuda/particles.cu
    cuda/fields.cu
    cuda/boundaries.cu
)

# C++ source files
set(CXX_SOURCES
    src/main.cpp
    src/config.cpp
    src/io_manager.cpp
    src/mpi_manager.cpp
    src/field_arrays.cpp
    src/particle_buffer.cpp
    src/kernels_cpu_optimized.cpp
)

# Add platform.cpp for CPU build
if(USE_CPU)
    list(APPEND CXX_SOURCES src/platform.cpp)

    # Treat .cu files as C++ in CPU mode
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-x c++")
endif()

# ============================================================================
# Create executable
# ============================================================================
add_executable(jericho_mkII ${CXX_SOURCES} ${CUDA_SOURCES})

target_link_libraries(jericho_mkII
    ${MPI_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Add CUDA libraries only for GPU build
if(NOT USE_CPU)
    target_link_libraries(jericho_mkII
        CUDA::cudart
        CUDA::curand
        ${CUDA_LIBS}
    )
endif()

# ============================================================================
# Create MPI-parallel executable
# ============================================================================
# MPI version uses same sources but with main_mpi.cpp instead of main.cpp
list(REMOVE_ITEM CXX_SOURCES src/main.cpp)
list(APPEND CXX_SOURCES_MPI ${CXX_SOURCES} src/main_mpi.cpp src/ghost_cell_exchange.cpp src/poisson_solver.cpp src/ampere_solver.cpp src/boris_pusher.cpp)

add_executable(jericho_mkII_mpi ${CXX_SOURCES_MPI} ${CUDA_SOURCES})

target_link_libraries(jericho_mkII_mpi
    ${MPI_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Add CUDA libraries only for GPU build
if(NOT USE_CPU)
    target_link_libraries(jericho_mkII_mpi
        CUDA::cudart
        CUDA::curand
        ${CUDA_LIBS}
    )
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS jericho_mkII jericho_mkII_mpi DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY examples/ DESTINATION share/jericho_mkII/examples)

# ============================================================================
# Testing
# ============================================================================
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Examples
# ============================================================================
if(BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "==================== Jericho Mk II Configuration ====================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "MPI C++ compiler: ${MPI_CXX_COMPILER}")
message(STATUS "CUDA-aware MPI: ${USE_CUDA_AWARE_MPI}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "NVTX profiling: ${ENABLE_PROFILING}")
message(STATUS "======================================================================")
message(STATUS "")
