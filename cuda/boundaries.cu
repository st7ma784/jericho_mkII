/**
 * @file boundaries.cu
 * @brief CUDA kernels for boundary condition handling
 * @author Jericho Mk II Development Team
 *
 * This file implements boundary conditions for particles in a GPU-efficient manner:
 * - Periodic: Wrap particles around domain
 * - Outflow: Remove particles leaving domain
 * - Inflow: Inject new particles at boundaries
 * - Reflecting: Reverse velocity at walls
 *
 * All kernels handle particles in parallel using atomic operations for
 * dynamic insertion/removal.
 */

#include "../include/particle_buffer.h"
#include "../include/platform.h"
#include <ctime>
#include <cstdlib>
#include <cmath>


namespace jericho {
namespace cuda {

// =============================================================================
// Boundary condition types
// =============================================================================

enum class BoundaryType : uint8_t {
    PERIODIC = 0,   ///< Wrap around to opposite boundary
    OUTFLOW = 1,    ///< Remove particles leaving domain
    INFLOW = 2,     ///< Inject particles (handled separately)
    REFLECTING = 3  ///< Reflect particles elastically
};

/**
 * @brief Boundary configuration for each domain edge
 */
struct BoundaryConfig {
    BoundaryType x_min;  ///< Left boundary
    BoundaryType x_max;  ///< Right boundary
    BoundaryType y_min;  ///< Bottom boundary
    BoundaryType y_max;  ///< Top boundary
};

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel: Apply boundary conditions to particles
 *
 * @details Each thread checks one particle for boundary violations.
 * Depending on boundary type:
 * - Periodic: Wrap position to opposite side
 * - Outflow: Mark particle as inactive (removal)
 * - Reflecting: Reverse velocity component and adjust position
 *
 * @param particles Particle buffer
 * @param x_min,x_max,y_min,y_max Domain bounds [m]
 * @param bc Boundary condition configuration
 * @param removal_flags Output: array marking particles for removal (device)
 * @param n_removed Output: counter for removed particles (device)
 * @param n_particles Number of active particles
 */
GLOBAL void apply_boundaries_kernel(
    double*  x,
    double*  y,
    double*  vx,
    double*  vy,
    bool*  active,
    double x_min, double x_max,
    double y_min, double y_max,
    BoundaryConfig bc,
    bool* removal_flags,
    size_t* n_removed,
    size_t n_particles)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles || !active[i]) return;

    double px = x[i];
    double py = y[i];
    double pvx = vx[i];
    double pvy = vy[i];
    bool remove = false;

    // Domain dimensions
    double Lx = x_max - x_min;
    double Ly = y_max - y_min;

    // Check X boundaries
    if (px < x_min) {
        switch (bc.x_min) {
            case BoundaryType::PERIODIC:
                px += Lx;
                break;
            case BoundaryType::OUTFLOW:
                remove = true;
                break;
            case BoundaryType::REFLECTING:
                px = 2.0 * x_min - px;  // Reflect position
                pvx = -pvx;              // Reverse velocity
                break;
        }
    } else if (px > x_max) {
        switch (bc.x_max) {
            case BoundaryType::PERIODIC:
                px -= Lx;
                break;
            case BoundaryType::OUTFLOW:
                remove = true;
                break;
            case BoundaryType::REFLECTING:
                px = 2.0 * x_max - px;
                pvx = -pvx;
                break;
        }
    }

    // Check Y boundaries
    if (py < y_min) {
        switch (bc.y_min) {
            case BoundaryType::PERIODIC:
                py += Ly;
                break;
            case BoundaryType::OUTFLOW:
                remove = true;
                break;
            case BoundaryType::REFLECTING:
                py = 2.0 * y_min - py;
                pvy = -pvy;
                break;
        }
    } else if (py > y_max) {
        switch (bc.y_max) {
            case BoundaryType::PERIODIC:
                py -= Ly;
                break;
            case BoundaryType::OUTFLOW:
                remove = true;
                break;
            case BoundaryType::REFLECTING:
                py = 2.0 * y_max - py;
                pvy = -pvy;
                break;
        }
    }

    // Update particle or mark for removal
    if (remove) {
        active[i] = false;
        removal_flags[i] = true;
        atomicAdd(n_removed, 1);
    } else {
        x[i] = px;
        y[i] = py;
        vx[i] = pvx;
        vy[i] = pvy;
        removal_flags[i] = false;
    }
}

/**
 * @brief Kernel: Inject particles at inflow boundaries
 *
 * @details Creates new particles at domain boundaries with specified
 * velocity distribution. Uses Maxwellian velocity distribution.
 *
 * This kernel is launched once per inflow boundary with threads creating
 * particles along that boundary.
 *
 * @param particles Particle buffer
 * @param boundary Which boundary (0=x_min, 1=x_max, 2=y_min, 3=y_max)
 * @param n_inject Number of particles to inject
 * @param x_min,x_max,y_min,y_max Domain bounds
 * @param v_mean Mean velocity [m/s]
 * @param v_thermal Thermal velocity [m/s]
 * @param weight Particle weight
 * @param type Species type
 * @param seed Random seed
 */
GLOBAL void inject_particles_kernel(
    double* x,
    double* y,
    double* vx,
    double* vy,
    double* weight,
    uint8_t* type,
    bool* active,
    size_t* insert_index,
    int boundary,
    size_t n_inject,
    double x_min, double x_max,
    double y_min, double y_max,
    double vx_mean, double vy_mean,
    double v_thermal,
    double particle_weight,
    uint8_t particle_type,
    unsigned int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_inject) return;

    // Get insertion index atomically
    size_t idx = atomicAdd(insert_index, 1);

    // Random number generation (simple LCG for demo)
    unsigned int rng_state = seed + i * 1103515245;
    auto rand_uniform = [&]() -> double {
        rng_state = rng_state * 1103515245 + 12345;
        return static_cast<double>(rng_state) / 4294967296.0;
    };

    // Box-Muller for Gaussian
    auto rand_gaussian = [&]() -> double {
        double u1 = rand_uniform();
        double u2 = rand_uniform();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    };

    double px, py, pvx, pvy;

    // Set position based on boundary
    double Lx = x_max - x_min;
    double Ly = y_max - y_min;

    switch (boundary) {
        case 0:  // X minimum (left edge)
            px = x_min;
            py = y_min + rand_uniform() * Ly;
            break;
        case 1:  // X maximum (right edge)
            px = x_max;
            py = y_min + rand_uniform() * Ly;
            break;
        case 2:  // Y minimum (bottom edge)
            px = x_min + rand_uniform() * Lx;
            py = y_min;
            break;
        case 3:  // Y maximum (top edge)
            px = x_min + rand_uniform() * Lx;
            py = y_max;
            break;
    }

    // Maxwellian velocity distribution
    pvx = vx_mean + v_thermal * rand_gaussian();
    pvy = vy_mean + v_thermal * rand_gaussian();

    // Write particle data
    x[idx] = px;
    y[idx] = py;
    vx[idx] = pvx;
    vy[idx] = pvy;
    weight[idx] = particle_weight;
    type[idx] = particle_type;
    active[idx] = true;
}

// =============================================================================
// Host wrapper functions
// =============================================================================

/**
 * @brief Apply boundary conditions to all particles
 *
 * @param buffer Particle buffer on device
 * @param x_min,x_max,y_min,y_max Domain bounds
 * @param bc Boundary configuration
 * @return Number of particles removed
 */
size_t apply_boundaries(ParticleBuffer& buffer,
                       double x_min, double x_max,
                       double y_min, double y_max,
                       const BoundaryConfig& bc)
{
    // Allocate temporary arrays on device
    bool* removal_flags;
    size_t* d_n_removed;
    size_t h_n_removed = 0;

    cudaMalloc(&removal_flags, buffer.capacity * sizeof(bool));
    cudaMalloc(&d_n_removed, sizeof(size_t));
    cudaMemset(d_n_removed, 0, sizeof(size_t));

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (buffer.count + threads_per_block - 1) / threads_per_block;

#ifdef USE_CPU
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    KERNEL_LAUNCH(apply_boundaries_kernel, grid, block,
        buffer.x, buffer.y, buffer.vx, buffer.vy, buffer.active,
        x_min, x_max, y_min, y_max,
        bc, removal_flags, d_n_removed, buffer.count);
#else
    apply_boundaries_kernel<<<num_blocks, threads_per_block>>>(
        buffer.x, buffer.y, buffer.vx, buffer.vy, buffer.active,
        x_min, x_max, y_min, y_max,
        bc, removal_flags, d_n_removed, buffer.count
    );
    PLATFORM_CHECK(cudaGetLastError());
#endif

    // Get number of removed particles
    cudaMemcpy(&h_n_removed, d_n_removed, sizeof(size_t), cudaMemcpyDeviceToHost);

    // Update particle count
    buffer.count -= h_n_removed;
    buffer.stats.particles_removed += h_n_removed;

    // Cleanup
    cudaFree(removal_flags);
    cudaFree(d_n_removed);

    return h_n_removed;
}

/**
 * @brief Inject particles at inflow boundary
 *
 * @param buffer Particle buffer
 * @param boundary Which boundary (0-3)
 * @param n_inject Number of particles to inject
 * @param domain_bounds [x_min, x_max, y_min, y_max]
 * @param v_mean Mean velocity
 * @param v_thermal Thermal velocity
 * @param weight Particle weight
 * @param type Species type
 */
void inject_particles(ParticleBuffer& buffer,
                     int boundary,
                     size_t n_inject,
                     const double* domain_bounds,
                     double vx_mean, double vy_mean,
                     double v_thermal,
                     double weight,
                     uint8_t type)
{
    // Ensure buffer has space
    if (buffer.count + n_inject > buffer.capacity) {
        buffer.resize(buffer.capacity * 2);
    }

    // Device counter for insertion index
    size_t* d_insert_index;
    cudaMalloc(&d_insert_index, sizeof(size_t));
    cudaMemcpy(d_insert_index, &buffer.count, sizeof(size_t), cudaMemcpyHostToDevice);

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (n_inject + threads_per_block - 1) / threads_per_block;

#ifdef USE_CPU
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    KERNEL_LAUNCH(inject_particles_kernel, grid, block,
        buffer.x, buffer.y, buffer.vx, buffer.vy, buffer.weight, buffer.type, buffer.active,
        d_insert_index,
        boundary, n_inject,
        domain_bounds[0], domain_bounds[1], domain_bounds[2], domain_bounds[3],
        vx_mean, vy_mean, v_thermal, weight, type,
        static_cast<unsigned int>(time(nullptr)));
#else
    inject_particles_kernel<<<num_blocks, threads_per_block>>>(
        buffer.x, buffer.y, buffer.vx, buffer.vy, buffer.weight, buffer.type, buffer.active,
        d_insert_index,
        boundary, n_inject,
        domain_bounds[0], domain_bounds[1], domain_bounds[2], domain_bounds[3],
        vx_mean, vy_mean, v_thermal, weight, type,
        static_cast<unsigned int>(time(nullptr))
    );
    PLATFORM_CHECK(cudaGetLastError());
#endif

    // Update count
    buffer.count += n_inject;
    buffer.stats.particles_added += n_inject;

    cudaFree(d_insert_index);
}

} // namespace cuda
} // namespace jericho
