/**
 * @file ohms_law_solver.cpp
 * @brief Implementation of generalized Ohm's law electric field solver
 * @author Jericho Mk II Development Team
 */

#include "ohms_law_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace jericho {

// ============================================================================
// Flow Velocity Computation
// ============================================================================

void compute_flow_velocity(const double* Jx, const double* Jy, const double* charge_density,
                           double* Ux, double* Uy, int nx, int ny, const OhmsLawConfig& config) {
    // U = J/q with charge density floor to prevent division by zero
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;

            // Apply charge floor (essential for stability)
            double q = std::fabs(charge_density[idx]);
            double q_safe = std::fmax(q, config.q_min);

            // Compute flow velocity
            Ux[idx] = Jx[idx] / q_safe;
            Uy[idx] = Jy[idx] / q_safe;
        }
    }
}

// ============================================================================
// Charge Density Smoothing
// ============================================================================

void smooth_charge_density(const double* q_in, double* q_out, int nx, int ny,
                           double smoothing_factor) {
    // Apply 4-point stencil smoothing
    // q_smooth = (1-α)·q + α·avg(neighbors)
    const double alpha = smoothing_factor;
    const double one_minus_alpha = 1.0 - alpha;

    // Copy to output first (handles boundary cells)
    std::memcpy(q_out, q_in, nx * ny * sizeof(double));

    // Smooth interior cells
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;
            int idx_xp = iy * nx + (ix + 1);
            int idx_xm = iy * nx + (ix - 1);
            int idx_yp = (iy + 1) * nx + ix;
            int idx_ym = (iy - 1) * nx + ix;

            // 4-neighbor average
            double avg = 0.25 * (q_in[idx_xp] + q_in[idx_xm] + q_in[idx_yp] + q_in[idx_ym]);

            // Blend
            q_out[idx] = one_minus_alpha * q_in[idx] + alpha * avg;
        }
    }
}

// ============================================================================
// Hall Term Computation
// ============================================================================

void compute_hall_term(const double* Bz, const double* charge_density, double* Ex_hall,
                       double* Ey_hall, int nx, int ny, double dx, double dy,
                       const OhmsLawConfig& config) {
    // Hall term: -∇×B/(μ₀·q)
    // For 2D with Bz: E_hall = -∇Bz/(μ₀·q)

    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_mu0 = 1.0 / config.mu_0;

    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;
            int idx_xp = iy * nx + (ix + 1);
            int idx_xm = iy * nx + (ix - 1);
            int idx_yp = (iy + 1) * nx + ix;
            int idx_ym = (iy - 1) * nx + ix;

            // Apply charge floor
            double q = std::fabs(charge_density[idx]);
            double q_safe = std::fmax(q, config.q_min);

            // Central differences
            double dBz_dx = (Bz[idx_xp] - Bz[idx_xm]) * inv_2dx;
            double dBz_dy = (Bz[idx_yp] - Bz[idx_ym]) * inv_2dy;

            // Hall term (note: negative sign!)
            Ex_hall[idx] = -dBz_dx * inv_mu0 / q_safe;
            Ey_hall[idx] = -dBz_dy * inv_mu0 / q_safe;
        }
    }
}

// ============================================================================
// Hall Term Boundary Tapering
// ============================================================================

void apply_hall_term_tapering(double* Ex_hall, double* Ey_hall, int nx, int ny, int taper_width) {
    // Linear taper from 0 (boundary) to 1 (interior)
    // Taper function: f(d) = min(d / width, 1.0)
    // where d is distance from nearest boundary

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;

            // Distance from each boundary
            int dist_xmin = ix;
            int dist_xmax = nx - 1 - ix;
            int dist_ymin = iy;
            int dist_ymax = ny - 1 - iy;

            // Minimum distance to any boundary
            int min_dist = std::min({dist_xmin, dist_xmax, dist_ymin, dist_ymax});

            // Taper factor: 0 at boundary, 1 at interior
            double taper_factor = (min_dist < taper_width) ? static_cast<double>(min_dist) /
                                                                 static_cast<double>(taper_width)
                                                           : 1.0;

            // Apply tapering
            Ex_hall[idx] *= taper_factor;
            Ey_hall[idx] *= taper_factor;
        }
    }
}

// ============================================================================
// Main Ohm's Law Solver
// ============================================================================

void solve_electric_field_ohms_law(const double* Ux, const double* Uy, const double* Bz,
                                   const double* charge_density, double* Ex, double* Ey, int nx,
                                   int ny, double dx, double dy, const OhmsLawConfig& config) {
    // Step 1: Compute convective term: E = -U×B
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;

            // Convective electric field (cross product for 2D)
            // E = -U×B, with B = (0, 0, Bz)
            // Ex = -Uy·Bz
            // Ey =  Ux·Bz
            Ex[idx] = -Uy[idx] * Bz[idx];
            Ey[idx] = Ux[idx] * Bz[idx];
        }
    }

    // Step 2: Add Hall term if enabled
    if (config.use_hall_term) {
        std::vector<double> Ex_hall(nx * ny, 0.0);
        std::vector<double> Ey_hall(nx * ny, 0.0);

        // Compute Hall term
        compute_hall_term(Bz, charge_density, Ex_hall.data(), Ey_hall.data(), nx, ny, dx, dy,
                          config);

        // Apply tapering near boundaries if enabled
        if (config.use_tapering) {
            apply_hall_term_tapering(Ex_hall.data(), Ey_hall.data(), nx, ny, config.taper_width);
        }

        // Add to electric field (SUBTRACTIVE - see compute_hall_term for sign)
        for (int i = 0; i < nx * ny; ++i) {
            Ex[i] += Ex_hall[i];
            Ey[i] += Ey_hall[i];
        }
    }
}

// ============================================================================
// Full Pipeline Solver
// ============================================================================

void solve_ohms_law_full(const double* Jx, const double* Jy, const double* Bz,
                         const double* charge_density, double* Ex, double* Ey, int nx, int ny,
                         double dx, double dy, const OhmsLawConfig& config) {
    // Allocate temporary arrays
    std::vector<double> Ux(nx * ny);
    std::vector<double> Uy(nx * ny);
    std::vector<double> q_work(nx * ny);

    // Optional: Smooth charge density
    const double* q_to_use = charge_density;
    if (config.use_smoothing) {
        smooth_charge_density(charge_density, q_work.data(), nx, ny, 0.5);
        q_to_use = q_work.data();
    } else {
        q_to_use = charge_density;
    }

    // Step 1: Compute flow velocity U = J/q
    compute_flow_velocity(Jx, Jy, q_to_use, Ux.data(), Uy.data(), nx, ny, config);

    // Step 2: Solve E-field using Ohm's law
    solve_electric_field_ohms_law(Ux.data(), Uy.data(), Bz, q_to_use, Ex, Ey, nx, ny, dx, dy,
                                  config);
}

// ============================================================================
// Diagnostics
// ============================================================================

double compute_hall_parameter(const double* charge_density, const double* Bz, double dx, int nx,
                              int ny) {
    // Hall parameter: β = d_i / L
    // where d_i = c/ω_pi is ion inertial length
    // ω_pi = sqrt(n e² / (ε₀ m_i))
    //
    // Simplified: β ≈ (characteristic ion inertial length) / dx

    const double e = 1.60217663e-19;   // Elementary charge [C]
    const double m_i = 1.67262192e-27; // Proton mass [kg]
    const double eps0 = 8.8541878e-12; // Permittivity [F/m]
    const double c = 2.99792458e8;     // Speed of light [m/s]

    double sum_beta = 0.0;
    int count = 0;

    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;

            // Number density from charge density
            double q = std::fabs(charge_density[idx]);
            if (q < 1.0e-20)
                continue; // Skip empty cells

            double n = q / e; // Number density [1/m³]

            // Ion plasma frequency
            double omega_pi = std::sqrt(n * e * e / (eps0 * m_i));

            // Ion inertial length
            double d_i = c / omega_pi;

            // Hall parameter
            double beta = d_i / dx;

            sum_beta += beta;
            count++;
        }
    }

    return (count > 0) ? sum_beta / count : 0.0;
}

} // namespace jericho
