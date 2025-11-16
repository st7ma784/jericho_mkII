/**
 * @file particles.cu
 * @brief Particle operation kernels (CPU/GPU compatible)
 * @author Jericho Mk II Development Team
 *
 * This file contains kernels for particle push, field interpolation,
 * and particle-to-grid operations. Supports both CPU and GPU execution.
 */

#include "../include/particle_buffer.h"
#include "../include/platform.h"
#include <cmath>

namespace jericho {
namespace cuda {

// =============================================================================
// Device helper functions (work on both CPU and GPU)
// =============================================================================

/**
 * @brief Bilinear interpolation from grid to particle position
 */
DEVICE_HOST inline double bilinear_interp(const double* field, int nx,
                                          double px, double py,
                                          double dx, double dy,
                                          double x_min, double y_min) {
    // Convert particle position to grid coordinates
    double gx = (px - x_min) / dx;
    double gy = (py - y_min) / dy;

    // Get lower-left grid cell indices
    int ix = static_cast<int>(floor(gx));
    int iy = static_cast<int>(floor(gy));

    // Get interpolation weights
    double fx = gx - ix;  // Fractional x
    double fy = gy - iy;  // Fractional y

    // Bilinear interpolation coefficients
    double w00 = (1.0 - fx) * (1.0 - fy);  // Lower-left
    double w10 = fx * (1.0 - fy);          // Lower-right
    double w01 = (1.0 - fx) * fy;          // Upper-left
    double w11 = fx * fy;                  // Upper-right

    // Interpolate (assumes periodic or ghost cells handle boundaries)
    return w00 * field[iy * nx + ix] +
           w10 * field[iy * nx + (ix + 1)] +
           w01 * field[(iy + 1) * nx + ix] +
           w11 * field[(iy + 1) * nx + (ix + 1)];
}

/**
 * @brief Boris particle pusher (velocity update)
 */
DEVICE_HOST inline void boris_push(double& vx, double& vy,
                                   double Ex, double Ey, double Bz,
                                   double q_over_m, double dt) {
    // Constants
    const double half_dt = 0.5 * dt;
    const double qm_half_dt = q_over_m * half_dt;

    // Step 1: Half acceleration by E
    double vx_minus = vx + qm_half_dt * Ex;
    double vy_minus = vy + qm_half_dt * Ey;

    // Step 2: Rotation by B (Boris rotation)
    double t = qm_half_dt * Bz;  // Rotation parameter
    double t2 = t * t;
    double s = 2.0 * t / (1.0 + t2);  // Rotation factor

    double vx_prime = vx_minus + vy_minus * t;
    double vy_prime = vy_minus - vx_minus * t;

    double vx_plus = vx_minus + vy_prime * s;
    double vy_plus = vy_minus - vx_prime * s;

    // Step 3: Half acceleration by E
    vx = vx_plus + qm_half_dt * Ex;
    vy = vy_plus + qm_half_dt * Ey;
}

// =============================================================================
// Kernels (CPU/GPU compatible)
// =============================================================================

/**
 * @brief Kernel: Advance particle positions and velocities
 */
GLOBAL void advance_particles_kernel(
    double* x,
    double* y,
    double* vx,
    double* vy,
    const double* weight,
    const uint8_t* type,
    const bool* active,
    const double* Ex,
    const double* Ey,
    const double* Bz,
    int nx, int ny,
    double dx, double dy,
    double x_min, double y_min,
    double dt,
    const double* q_over_m_by_type,
    size_t n_particles)
{
    // Thread index = particle index
#ifdef USE_CPU
    size_t i = blockIdx.x * blockDim.x +
               threadIdx.x;
#else
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    if (i >= n_particles || !active[i]) return;

    // Get particle data (coalesced reads!)
    double px = x[i];
    double py = y[i];
    double pvx = vx[i];
    double pvy = vy[i];
    uint8_t ptype = type[i];

    // Interpolate fields to particle position
    double E_x = bilinear_interp(Ex, nx, px, py, dx, dy, x_min, y_min);
    double E_y = bilinear_interp(Ey, nx, px, py, dx, dy, x_min, y_min);
    double B_z = bilinear_interp(Bz, nx, px, py, dx, dy, x_min, y_min);

    // Get charge-to-mass ratio for this species
    double qm = q_over_m_by_type[ptype];

    // Boris push (updates pvx, pvy)
    boris_push(pvx, pvy, E_x, E_y, B_z, qm, dt);

    // Advance position
    px += pvx * dt;
    py += pvy * dt;

    // Write back (coalesced writes!)
    x[i] = px;
    y[i] = py;
    vx[i] = pvx;
    vy[i] = pvy;
}

/**
 * @brief Kernel: Particle-to-grid (P2G) interpolation
 */
GLOBAL void particle_to_grid_kernel(
    const double* x,
    const double* y,
    const double* vx,
    const double* vy,
    const double* weight,
    const uint8_t* type,
    const bool* active,
    double* charge_density,
    double* current_x,
    double* current_y,
    int nx, int ny,
    double dx, double dy,
    double x_min, double y_min,
    const double* q_by_type,
    size_t n_particles)
{
#ifdef USE_CPU
    size_t i = blockIdx.x * blockDim.x +
               threadIdx.x;
#else
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    if (i >= n_particles || !active[i]) return;

    // Get particle data
    double px = x[i];
    double py = y[i];
    double pvx = vx[i];
    double pvy = vy[i];
    double pw = weight[i];
    uint8_t ptype = type[i];

    // Convert to grid coordinates
    double gx = (px - x_min) / dx;
    double gy = (py - y_min) / dy;

    int ix = static_cast<int>(floor(gx));
    int iy = static_cast<int>(floor(gy));

    double fx = gx - ix;
    double fy = gy - iy;

    // Bilinear weights
    double w00 = (1.0 - fx) * (1.0 - fy);
    double w10 = fx * (1.0 - fy);
    double w01 = (1.0 - fx) * fy;
    double w11 = fx * fy;

    // Particle charge
    double q = q_by_type[ptype] * pw;

    // Scatter charge density to 4 grid points (atomic for thread safety)
    atomicAdd(&charge_density[iy * nx + ix], q * w00);
    atomicAdd(&charge_density[iy * nx + (ix + 1)], q * w10);
    atomicAdd(&charge_density[(iy + 1) * nx + ix], q * w01);
    atomicAdd(&charge_density[(iy + 1) * nx + (ix + 1)], q * w11);

    // Scatter current density (J = q*v for each component)
    double jx = q * pvx;
    double jy = q * pvy;

    atomicAdd(&current_x[iy * nx + ix], jx * w00);
    atomicAdd(&current_x[iy * nx + (ix + 1)], jx * w10);
    atomicAdd(&current_x[(iy + 1) * nx + ix], jx * w01);
    atomicAdd(&current_x[(iy + 1) * nx + (ix + 1)], jx * w11);

    atomicAdd(&current_y[iy * nx + ix], jy * w00);
    atomicAdd(&current_y[iy * nx + (ix + 1)], jy * w10);
    atomicAdd(&current_y[(iy + 1) * nx + ix], jy * w01);
    atomicAdd(&current_y[(iy + 1) * nx + (ix + 1)], jy * w11);
}

// =============================================================================
// Host wrapper functions (callable from CPU)
// =============================================================================

/**
 * @brief Advance all particles by one timestep
 */
void advance_particles(ParticleBuffer& buffer,
                      const double* Ex, const double* Ey, const double* Bz,
                      int nx, int ny, double dx, double dy,
                      double x_min, double y_min, double dt,
                      const double* q_over_m_by_type)
{
    // Launch configuration
    const int threads_per_block = 256;
    const int num_blocks = (buffer.count + threads_per_block - 1) / threads_per_block;

#ifdef USE_CPU
    // CPU execution
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    KERNEL_LAUNCH(advance_particles_kernel, grid, block,
        buffer.x, buffer.y, buffer.vx, buffer.vy,
        buffer.weight, buffer.type, buffer.active,
        Ex, Ey, Bz,
        nx, ny, dx, dy, x_min, y_min, dt,
        q_over_m_by_type, buffer.count
    );
#else
    // GPU execution
    advance_particles_kernel<<<num_blocks, threads_per_block>>>(
        buffer.x, buffer.y, buffer.vx, buffer.vy,
        buffer.weight, buffer.type, buffer.active,
        Ex, Ey, Bz,
        nx, ny, dx, dy, x_min, y_min, dt,
        q_over_m_by_type, buffer.count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in advance_particles: %s\n",
                cudaGetErrorString(err));
    }
#endif
}

/**
 * @brief Interpolate particles to grid (P2G operation)
 */
void particle_to_grid(const ParticleBuffer& buffer,
                     double* charge_density, double* current_x, double* current_y,
                     int nx, int ny, double dx, double dy,
                     double x_min, double y_min,
                     const double* q_by_type)
{
    const int threads_per_block = 256;
    const int num_blocks = (buffer.count + threads_per_block - 1) / threads_per_block;

#ifdef USE_CPU
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    KERNEL_LAUNCH(particle_to_grid_kernel, grid, block,
        buffer.x, buffer.y, buffer.vx, buffer.vy,
        buffer.weight, buffer.type, buffer.active,
        charge_density, current_x, current_y,
        nx, ny, dx, dy, x_min, y_min,
        q_by_type, buffer.count
    );
#else
    particle_to_grid_kernel<<<num_blocks, threads_per_block>>>(
        buffer.x, buffer.y, buffer.vx, buffer.vy,
        buffer.weight, buffer.type, buffer.active,
        charge_density, current_x, current_y,
        nx, ny, dx, dy, x_min, y_min,
        q_by_type, buffer.count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in particle_to_grid: %s\n",
                cudaGetErrorString(err));
    }
#endif
}

} // namespace cuda
} // namespace jericho
