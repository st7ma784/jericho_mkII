/**
 * @file kernels_cpu.cpp
 * @brief CPU implementations of core PIC-MHD kernels
 * @details Implements Boris pusher, P2G deposition, and field solvers
 *          optimized for SoA particle memory layout
 */

#include <algorithm>
#include <cmath>
#include <iostream>

namespace jericho {
namespace cuda {

// Physical constants
constexpr double EPSILON_0 = 8.854187817e-12; // F/m
constexpr double MU_0 = 1.25663706212e-6;     // H/m
constexpr double CHARGE_E = 1.602176634e-19;  // C

// =============================================================================
// Boris Pusher - Core particle acceleration/velocity update
// =============================================================================
/**
 * Boris pusher algorithm for particle advancement
 * Advances particles using E and B fields with second-order accuracy
 *
 * References:
 * - Boris, J. P. (1970), "Relativistic plasma simulation-optimization of a collision less model"
 * - Birdsall & Langdon (1985), "Plasma Physics via Computer Simulation"
 */
inline void boris_step(double& vx, double& vy, double Ex, double Ey, double Bz,
                       double qm, // charge/mass ratio
                       double dt) {
    // Half-step acceleration by E field
    double ax = qm * Ex * dt / 2.0;
    double ay = qm * Ey * dt / 2.0;

    vx += ax;
    vy += ay;

    // Magnetic field rotation
    double Omegaz = qm * Bz * dt / 2.0; // Larmor frequency * dt/2
    double tan_half = std::tan(Omegaz);
    double cos_omega = 1.0 / std::sqrt(1.0 + tan_half * tan_half);
    double sin_omega = tan_half * cos_omega;

    // Rotate velocity by magnetic field (second order)
    double vx_rot = vx * cos_omega + vy * sin_omega;
    double vy_rot = -vx * sin_omega + vy * cos_omega;

    vx = vx_rot;
    vy = vy_rot;

    // Second half-step acceleration by E field
    vx += ax;
    vy += ay;
}

/**
 * Advance particles using Boris algorithm
 *
 * @param[in,out] x, y             Particle positions (modified)
 * @param[in,out] vx, vy           Particle velocities (modified)
 * @param[in] Ex, Ey, Bz           Electric and magnetic fields (grid)
 * @param[in] nx, ny               Grid dimensions
 * @param[in] dx, dy               Grid spacing
 * @param[in] x_min, y_min         Domain minimum coordinates
 * @param[in] dt                   Timestep
 * @param[in] qm_by_type           Charge-to-mass ratios for each species
 * @param[in] n_particles          Number of particles
 * @param[in] species_id           Species index for qm lookup
 * @param[in] periodic_x, periodic_y  Boundary conditions
 */
void advance_particles_cpu(double* x, double* y, double* vx, double* vy, const double* Ex,
                           const double* Ey, const double* Bz, int nx, int ny, double dx, double dy,
                           double x_min, double y_min, double dt, const double* qm_by_type,
                           int n_particles, int species_id, bool periodic_x = true,
                           bool periodic_y = true) {

    double qm = qm_by_type[species_id];
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double x_max = x_min + nx * dx;
    double y_max = y_min + ny * dy;
    double Lx = x_max - x_min;
    double Ly = y_max - y_min;

#pragma omp parallel for collapse(1)
    for (int p = 0; p < n_particles; p++) {
        // Get particle position
        double xp = x[p];
        double yp = y[p];

        // Find grid cell indices (linear interpolation)
        double xi = (xp - x_min) * inv_dx;
        double yi = (yp - y_min) * inv_dy;

        int ix = static_cast<int>(xi);
        int iy = static_cast<int>(yi);

        // Clamp to valid range for field lookup
        ix = std::min(std::max(ix, 0), nx - 2);
        iy = std::min(std::max(iy, 0), ny - 2);

        // Linear interpolation weights
        double fx = xi - ix;
        double fy = yi - iy;

        // Get field values at particle location by bilinear interpolation
        auto get_field_interp = [&](const double* field, int i, int j) -> double {
            double f00 = field[j * nx + i];
            double f10 = field[j * nx + (i + 1)];
            double f01 = field[(j + 1) * nx + i];
            double f11 = field[(j + 1) * nx + (i + 1)];

            return (1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10 + (1 - fx) * fy * f01 +
                   fx * fy * f11;
        };

        double Ex_p = get_field_interp(Ex, ix, iy);
        double Ey_p = get_field_interp(Ey, ix, iy);
        double Bz_p = get_field_interp(Bz, ix, iy);

        // Boris step
        boris_step(vx[p], vy[p], Ex_p, Ey_p, Bz_p, qm, dt);

        // Update position
        xp += vx[p] * dt;
        yp += vy[p] * dt;

        // Apply periodic boundary conditions
        if (periodic_x) {
            while (xp < x_min)
                xp += Lx;
            while (xp >= x_max)
                xp -= Lx;
        } else {
            // Absorbing boundary
            if (xp < x_min || xp >= x_max) {
                x[p] = -1e10; // Mark as inactive
                continue;
            }
        }

        if (periodic_y) {
            while (yp < y_min)
                yp += Ly;
            while (yp >= y_max)
                yp -= Ly;
        } else {
            // Absorbing boundary
            if (yp < y_min || yp >= y_max) {
                x[p] = -1e10; // Mark as inactive
                continue;
            }
        }

        x[p] = xp;
        y[p] = yp;
    }
}

// =============================================================================
// Particle-to-Grid (P2G) Deposition
// =============================================================================
/**
 * Deposit particle charge and current to grid using Cloud-in-Cell (CIC)
 * Linear weighting for momentum conservation
 *
 * @param[in] x, y                 Particle positions
 * @param[in] vx, vy               Particle velocities
 * @param[in,out] rho              Charge density on grid (accumulate)
 * @param[in,out] Jx, Jy           Current density on grid (accumulate)
 * @param[in] nx, ny               Grid dimensions
 * @param[in] dx, dy               Grid spacing
 * @param[in] x_min, y_min         Domain minimum
 * @param[in] q                    Particle charge
 * @param[in] weight               Particle weight (number of particles per macroparticle)
 * @param[in] n_particles          Number of particles
 */
void particle_to_grid_cpu(const double* x, const double* y, const double* vx, const double* vy,
                          double* rho, double* Jx, double* Jy, int nx, int ny, double dx, double dy,
                          double x_min, double y_min, double q, double weight, int n_particles) {

    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double inv_cell_vol = inv_dx * inv_dy;

#pragma omp parallel for collapse(1)
    for (int p = 0; p < n_particles; p++) {
        double xp = x[p];
        double yp = y[p];

        // Skip inactive particles (marked with x = -1e10)
        if (xp < -1e9)
            continue;

        // Find grid cell
        double xi = (xp - x_min) * inv_dx;
        double yi = (yp - y_min) * inv_dy;

        int ix = static_cast<int>(xi);
        int iy = static_cast<int>(yi);

        // Clamp to valid range
        if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1)
            continue;

        // Fractional position within cell
        double fx = xi - ix;
        double fy = yi - iy;

        // CIC weights for 4 nearest grid points
        double w00 = (1.0 - fx) * (1.0 - fy);
        double w10 = fx * (1.0 - fy);
        double w01 = (1.0 - fx) * fy;
        double w11 = fx * fy;

        double q_w = q * weight;
        double qvx_w = q_w * vx[p];
        double qvy_w = q_w * vy[p];

        // Deposit to 4 grid cells
        int idx00 = iy * nx + ix;
        int idx10 = iy * nx + (ix + 1);
        int idx01 = (iy + 1) * nx + ix;
        int idx11 = (iy + 1) * nx + (ix + 1);

// Atomic operations to prevent race conditions
// In single-threaded: just direct assignment
// In OpenMP: use critical section or atomic add
#pragma omp critical
        {
            rho[idx00] += w00 * q_w;
            rho[idx10] += w10 * q_w;
            rho[idx01] += w01 * q_w;
            rho[idx11] += w11 * q_w;

            Jx[idx00] += w00 * qvx_w;
            Jx[idx10] += w10 * qvx_w;
            Jx[idx01] += w01 * qvx_w;
            Jx[idx11] += w11 * qvx_w;

            Jy[idx00] += w00 * qvy_w;
            Jy[idx10] += w10 * qvy_w;
            Jy[idx01] += w01 * qvy_w;
            Jy[idx11] += w11 * qvy_w;
        }
    }
}

// =============================================================================
// Field Solver Stubs (can be expanded later)
// =============================================================================

void compute_flow_velocity(int nx, int ny, const double* rho, const double* Jx, const double* Jy,
                           double* vx_flow, double* vy_flow) {
    // Stub: v = J / rho, with protection for small density
    for (int i = 0; i < nx * ny; i++) {
        if (std::abs(rho[i]) > 1e-10) {
            vx_flow[i] = Jx[i] / rho[i];
            vy_flow[i] = Jy[i] / rho[i];
        } else {
            vx_flow[i] = 0.0;
            vy_flow[i] = 0.0;
        }
    }
}

void solve_electric_field(int nx, int ny, double dx, double dy, const double* rho,
                          const double* vx_flow, const double* vy_flow, const double* Bz,
                          double* Ex, double* Ey, bool use_hall) {
    // Stub: Simplified Ohm's law E = -v x B + J/(sigma)
    for (int i = 0; i < nx * ny; i++) {
        // E = -v x B term
        Ex[i] = vx_flow[i] * Bz[i];
        Ey[i] = -vy_flow[i] * Bz[i];
    }
}

void apply_cam_correction(int nx, int ny, double* Ex, double* Ey, double dt) {
    // Stub: CAM (CAM) correction for stability
    // In full implementation, would add smoothing filter
}

void clamp_electric_field(int nx, int ny, const double* rho, double* Ex, double* Ey,
                          double rho_threshold = 1e-5) {
    // Stub: Clamp E field in low-density regions
    for (int i = 0; i < nx * ny; i++) {
        if (std::abs(rho[i]) < rho_threshold) {
            Ex[i] = 0.0;
            Ey[i] = 0.0;
        }
    }
}

void advance_magnetic_field(int nx, int ny, double dx, double dy, double* Bz, const double* Ex,
                            const double* Ey, double dt) {
    // Stub: Faraday's law dB/dt = -∇ x E
    // In 2D: dBz/dt = (dEy/dx - dEx/dy)
}

} // namespace cuda
} // namespace jericho
