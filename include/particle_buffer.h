/**
 * @file particle_buffer.h
 * @brief GPU-optimized Structure of Arrays (SoA) particle buffer
 * @author Jericho Mk II Development Team
 *
 * This file implements a modern SoA particle buffer designed for optimal
 * GPU performance. All particle data is stored in separate contiguous arrays
 * to enable coalesced memory access and SIMD vectorization.
 *
 * Key features:
 * - Coalesced memory access on GPU (10-50x faster than AoS)
 * - Dynamic particle management (inflow/outflow boundaries)
 * - CUDA unified memory support
 * - Efficient particle insertion/removal
 * - Cache-friendly CPU access patterns
 */

#pragma once

#include "platform.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jericho {

/**
 * @brief Particle buffer using Structure of Arrays layout
 *
 * @details This class manages particle data in a GPU-friendly SoA format.
 * Instead of an array of particle structs (AoS), we store each particle
 * property in a separate array. This enables:
 *
 * 1. **Coalesced memory access**: Adjacent GPU threads access adjacent memory
 * 2. **SIMD vectorization**: CPU can process 4-8 particles simultaneously
 * 3. **Cache efficiency**: Only load data you need (e.g., just positions)
 * 4. **GPU kernel fusion**: Easy to combine operations
 *
 * Memory layout comparison:
 * ```
 * AoS (bad for GPU):  [x0,y0,vx0,vy0][x1,y1,vx1,vy1][x2,y2,vx2,vy2]...
 * SoA (good for GPU): [x0,x1,x2,...][y0,y1,y2,...][vx0,vx1,vx2,...]...
 * ```
 *
 * Thread 0 accessing x0, Thread 1 accessing x1, etc. results in a single
 * coalesced memory transaction with SoA, but scattered transactions with AoS.
 */
class ParticleBuffer {
  public:
    // =========================================================================
    // Data members (stored on GPU device memory)
    // =========================================================================

    double* x;      ///< X positions [m] - GPU device memory
    double* y;      ///< Y positions [m] - GPU device memory
    double* vx;     ///< X velocities [m/s] - GPU device memory
    double* vy;     ///< Y velocities [m/s] - GPU device memory
    double* weight; ///< Statistical weights (macroparticle multiplicity)
    uint8_t* type;  ///< Species type index (0 = electrons, 1+ = ions)
    bool* active;   ///< Active flag for dynamic particle management

    size_t capacity; ///< Allocated capacity (total slots)
    size_t count;    ///< Current active particle count
    int device_id;   ///< CUDA device ID where buffer resides

    // Free slot management for efficient insert/remove
    size_t* free_slots; ///< Stack of free indices (GPU device memory)
    size_t free_count;  ///< Number of free slots available

    // =========================================================================
    // Host-side metadata (stored on CPU)
    // =========================================================================

    struct Metadata {
        size_t particles_added;     ///< Total particles added (inflow)
        size_t particles_removed;   ///< Total particles removed (outflow)
        size_t particles_reflected; ///< Total particles reflected
        size_t max_particles_seen;  ///< Peak particle count
    } stats;

    // =========================================================================
    // Constructors / Destructors
    // =========================================================================

    /**
     * @brief Construct particle buffer on specified GPU
     *
     * @param initial_capacity Initial number of particle slots to allocate
     * @param device_id CUDA device ID (default: 0)
     *
     * @note Allocates GPU memory. Must call destroy() or use RAII wrapper.
     */
    ParticleBuffer(size_t initial_capacity, int device_id = 0);

    /**
     * @brief Destructor - frees GPU memory
     */
    ~ParticleBuffer();

    // Prevent copying (expensive on GPU), allow moving
    ParticleBuffer(const ParticleBuffer&) = delete;
    ParticleBuffer& operator=(const ParticleBuffer&) = delete;
    ParticleBuffer(ParticleBuffer&& other) noexcept;
    ParticleBuffer& operator=(ParticleBuffer&& other) noexcept;

    // =========================================================================
    // Memory management
    // =========================================================================

    /**
     * @brief Allocate GPU memory for particle arrays
     *
     * @param capacity Number of particle slots to allocate
     */
    void allocate(size_t capacity);

    /**
     * @brief Free all GPU memory
     */
    void destroy();

    /**
     * @brief Resize buffer (may trigger reallocation and copy)
     *
     * @param new_capacity New capacity (must be >= count)
     *
     * @note If new_capacity > capacity, allocates new memory and copies data.
     *       If new_capacity < count, throws exception (would lose particles).
     */
    void resize(size_t new_capacity);

    /**
     * @brief Compact buffer by removing inactive particles
     *
     * @details Moves active particles to fill gaps left by removed particles.
     * This is expensive but reduces memory fragmentation. Call periodically
     * when free_count is large.
     *
     * @note This is a GPU kernel operation, synchronizes device.
     */
    void compact();

    // =========================================================================
    // Particle insertion / removal (dynamic boundary conditions)
    // =========================================================================

    /**
     * @brief Add a single particle (inflow boundary)
     *
     * @param px,py Position [m]
     * @param pvx,pvy Velocity [m/s]
     * @param pweight Statistical weight
     * @param ptype Species type index
     *
     * @return Index where particle was inserted, or -1 if buffer full
     *
     * @note If buffer is full, automatically resizes with 1.5x growth factor
     */
    size_t add_particle(double px, double py, double pvx, double pvy, double pweight,
                        uint8_t ptype);

    /**
     * @brief Remove particle at index (outflow boundary)
     *
     * @param idx Index of particle to remove
     *
     * @note Marks particle as inactive and adds index to free_slots stack.
     *       Actual memory is not freed until compact() is called.
     */
    void remove_particle(size_t idx);

    /**
     * @brief Batch add particles (efficient for inflow boundaries)
     *
     * @param positions Host array of [x0,y0,x1,y1,...] (size: 2*n_particles)
     * @param velocities Host array of [vx0,vy0,vx1,vy1,...] (size: 2*n_particles)
     * @param weights Host array of weights (size: n_particles)
     * @param types Host array of type indices (size: n_particles)
     * @param n_particles Number of particles to add
     *
     * @note Much more efficient than calling add_particle() in a loop.
     *       Uses batched GPU memory transfers.
     */
    void add_particles_batch(const double* positions, const double* velocities,
                             const double* weights, const uint8_t* types, size_t n_particles);

    // =========================================================================
    // Data transfer (CPU <-> GPU)
    // =========================================================================

    /**
     * @brief Copy particle data from host to device
     *
     * @param h_x,h_y,h_vx,h_vy,h_weight,h_type Host arrays
     * @param n Number of particles to copy
     * @param offset Offset in device arrays to start copying to
     *
     * @note Uses asynchronous CUDA memcpy for better performance
     */
    void copy_to_device(const double* h_x, const double* h_y, const double* h_vx,
                        const double* h_vy, const double* h_weight, const uint8_t* h_type, size_t n,
                        size_t offset = 0);

    /**
     * @brief Copy particle data from device to host
     *
     * @param h_x,h_y,h_vx,h_vy,h_weight,h_type Host arrays (pre-allocated)
     * @param n Number of particles to copy
     * @param offset Offset in device arrays to start copying from
     */
    void copy_to_host(double* h_x, double* h_y, double* h_vx, double* h_vy, double* h_weight,
                      uint8_t* h_type, size_t n, size_t offset = 0) const;

    /**
     * @brief Get particle count (active particles only)
     */
    size_t get_count() const {
        return count;
    }

    /**
     * @brief Get buffer capacity
     */
    size_t get_capacity() const {
        return capacity;
    }

    /**
     * @brief Get memory usage in bytes
     */
    size_t get_memory_bytes() const {
        return capacity * (sizeof(double) * 5 + sizeof(uint8_t) + sizeof(bool));
    }

    /**
     * @brief Print buffer statistics
     */
    void print_stats() const;
};

// =============================================================================
// Helper functions for particle initialization
// =============================================================================

/**
 * @brief Initialize particles uniformly in a rectangular domain
 *
 * @param buffer Particle buffer to fill
 * @param x_min,x_max,y_min,y_max Domain bounds [m]
 * @param particles_per_cell Number of particles per grid cell
 * @param nx,ny Number of grid cells in x,y
 * @param type Species type index
 * @param weight Particle statistical weight
 * @param vx_mean,vy_mean Mean velocity [m/s]
 * @param v_thermal Thermal velocity for Maxwellian distribution [m/s]
 * @param seed Random seed
 */
void initialize_uniform(ParticleBuffer& buffer, double x_min, double x_max, double y_min,
                        double y_max, int particles_per_cell, int nx, int ny, uint8_t type,
                        double weight, double vx_mean, double vy_mean, double v_thermal,
                        unsigned int seed = 42);

/**
 * @brief Initialize particles with Maxwellian velocity distribution
 *
 * @param buffer Particle buffer to modify
 * @param v_thermal Thermal velocity [m/s]
 * @param vx_drift,vy_drift Drift velocity [m/s]
 * @param seed Random seed
 */
void initialize_maxwellian_velocities(ParticleBuffer& buffer, double v_thermal, double vx_drift,
                                      double vy_drift, unsigned int seed = 42);

} // namespace jericho
