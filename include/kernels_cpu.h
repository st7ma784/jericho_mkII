/**
 * @file kernels_cpu.h
 * @brief CPU kernel function declarations
 */

#ifndef JERICHO_KERNELS_CPU_H
#define JERICHO_KERNELS_CPU_H

namespace jericho {
namespace cuda {

// Boris pusher for particle advancement
void advance_particles_cpu(double* x, double* y,
                          double* vx, double* vy,
                          const double* Ex, const double* Ey, const double* Bz,
                          int nx, int ny,
                          double dx, double dy,
                          double x_min, double y_min,
                          double dt,
                          const double* qm_by_type,
                          int n_particles,
                          int species_id,
                          bool periodic_x = true,
                          bool periodic_y = true);

// Particle-to-grid deposition (Cloud-in-Cell)
void particle_to_grid_cpu(const double* x, const double* y,
                         const double* vx, const double* vy,
                         double* rho, double* Jx, double* Jy,
                         int nx, int ny,
                         double dx, double dy,
                         double x_min, double y_min,
                         double q,
                         double weight,
                         int n_particles);

// Field solver utilities
void compute_flow_velocity(int nx, int ny,
                          const double* rho,
                          const double* Jx, const double* Jy,
                          double* vx_flow, double* vy_flow);

void solve_electric_field(int nx, int ny,
                         double dx, double dy,
                         const double* rho,
                         const double* vx_flow, const double* vy_flow,
                         const double* Bz,
                         double* Ex, double* Ey,
                         bool use_hall = true);

void apply_cam_correction(int nx, int ny,
                         double* Ex, double* Ey,
                         double dt);

void clamp_electric_field(int nx, int ny,
                         const double* rho,
                         double* Ex, double* Ey,
                         double rho_threshold = 1e-5);

void advance_magnetic_field(int nx, int ny,
                           double dx, double dy,
                           double* Bz,
                           const double* Ex, const double* Ey,
                           double dt);

} // namespace cuda
} // namespace jericho

#endif // JERICHO_KERNELS_CPU_H
