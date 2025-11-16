/**
 * @file boris_pusher.cpp
 * @brief Implementation of Boris algorithm for particle advancement
 */

#include "boris_pusher.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace jericho {

/**
 * @brief Interpolate field value at particle position using bilinear interpolation
 */
double interpolate_field(const double* field, double x, double y, double dx, double dy, int nx,
                         int ny) {

    if (!field)
        return 0.0;

    // Normalize coordinates
    double xi = x / dx;
    double yi = y / dy;

    // Get cell indices (with bounds checking)
    int i = (int)xi;
    int j = (int)yi;

    // Periodic boundary conditions
    i = i % nx;
    if (i < 0)
        i += nx;
    j = j % ny;
    if (j < 0)
        j += ny;

    int i1 = (i + 1) % nx;
    int j1 = (j + 1) % ny;

    // Fractional coordinates
    double fx = xi - std::floor(xi);
    double fy = yi - std::floor(yi);

    // Bilinear interpolation
    double f00 = field[j * nx + i];
    double f10 = field[j * nx + i1];
    double f01 = field[j1 * nx + i];
    double f11 = field[j1 * nx + i1];

    double f0 = f00 * (1.0 - fx) + f10 * fx;
    double f1 = f01 * (1.0 - fx) + f11 * fx;

    return f0 * (1.0 - fy) + f1 * fy;
}

/**
 * @brief Advance particles using Boris algorithm
 *
 * Algorithm:
 * 1. Half-step electric field acceleration: v^- = v^n + (q/2m)E*dt
 * 2. Magnetic field rotation (using angle-doubling formula)
 * 3. Half-step electric field acceleration: v^(n+1) = v^+ + (q/2m)E*dt
 */
void advance_particles_boris(ParticleBuffer& particles, const double* Ex, const double* Ey,
                             const double* Bz, int nx, int ny, double dx, double dy, double dt,
                             const BorisConfig& config, int species_type) {

    // Get charge-to-mass ratio for this species
    double qm = constants::get_qm_ratio(species_type);
    if (qm == 0.0)
        return; // No motion without q/m

    double half_qm_dt = 0.5 * qm * dt;

    // Process each particle
    for (size_t i = 0; i < particles.count; ++i) {
        // Get particle position
        double x = particles.x[i];
        double y = particles.y[i];

        // Get particle velocity
        double vx = particles.vx[i];
        double vy = particles.vy[i];

        // Interpolate fields at particle position
        double ex = interpolate_field(Ex, x, y, dx, dy, nx, ny);
        double ey = interpolate_field(Ey, x, y, dx, dy, nx, ny);
        double bz = interpolate_field(Bz, x, y, dx, dy, nx, ny);

        // --- STEP 1: Half-step electric field acceleration ---
        double vx_minus = vx + half_qm_dt * ex;
        double vy_minus = vy + half_qm_dt * ey;

        // --- STEP 2: Magnetic field rotation (Boris algorithm) ---
        // Rotation angle: ω = (q/m)*B*dt
        double omega_z = qm * bz * dt; // Full rotation angle for this timestep

        // Compute rotation parameters using angle-doubling formula
        // For rotation by angle ω around z-axis:
        // v_new = v_old + (2*tan(ω/2)) / (1 + tan²(ω/2)) * (v_old × ω_hat)
        //
        // Simplified for 2D (Bz only):
        // tan(ω/2) computed using std::tan for numerical stability
        double half_omega = omega_z / 2.0;
        double tan_half_omega = std::tan(half_omega);

        // Perpendicular component of velocity gets rotated
        // v_perp' = v_perp + 2*tan(ω/2) * (v_perp × ω_hat)
        // For 2D with B = (0, 0, Bz): cross product (vx, vy, 0) × (0, 0, 1) = (vy, -vx, 0)
        double vx_rot = vx_minus + 2.0 * tan_half_omega * vy_minus;
        double vy_rot = vy_minus - 2.0 * tan_half_omega * vx_minus;

        // --- STEP 3: Half-step electric field acceleration ---
        particles.vx[i] = vx_rot + half_qm_dt * ex;
        particles.vy[i] = vy_rot + half_qm_dt * ey;
    }
}

/**
 * @brief Compute total kinetic energy
 */
double compute_kinetic_energy(const ParticleBuffer& particles, int species_type) {

    double mass = constants::get_mass(species_type);
    if (mass <= 0.0)
        return 0.0;

    if (!particles.vx || !particles.vy || particles.count == 0) {
        return 0.0;
    }

    double ke = 0.0;
    size_t nan_count = 0;
    size_t inf_count = 0;

    for (size_t i = 0; i < particles.count; ++i) {
        // Get particle velocities
        double vx = particles.vx[i];
        double vy = particles.vy[i];

        // Check for valid values (avoid NaN)
        if (std::isnan(vx) || std::isnan(vy)) {
            nan_count++;
            continue;
        }

        // Check for inf velocities
        if (std::isinf(vx) || std::isinf(vy)) {
            inf_count++;
            continue;
        }

        double v_sq = vx * vx + vy * vy;

        // Check if v_sq is inf or NaN
        if (std::isnan(v_sq) || std::isinf(v_sq)) {
            inf_count++;
            continue;
        }

        // Account for macroparticle weight (if available)
        double weight = 1.0; // Default weight
        if (particles.weight) {
            weight = particles.weight[i];
            if (std::isnan(weight) || weight <= 0.0) {
                weight = 1.0;
            }
        }

        double contribution = 0.5 * mass * v_sq * weight;

        // Check if contribution is inf/nan
        if (std::isnan(contribution) || std::isinf(contribution)) {
            inf_count++;
            continue;
        }

        ke += contribution;
    }

    // Final check - if ke became inf/nan somehow, warn
    if (std::isinf(ke) || std::isnan(ke)) {
        std::cerr << "WARNING: compute_kinetic_energy returning inf/nan. "
                  << "Species: " << species_type << ", " << "NaN particles: " << nan_count << ", "
                  << "Inf particles: " << inf_count << ", "
                  << "Total particles: " << particles.count << std::endl;
    }

    return ke;
}

/**
 * @brief Compute energy conservation error
 */
double EnergyHistory::conservation_error(int step) const {
    if (step < 0 || step >= (int)total_energy.size())
        return 0.0;
    if (step == 0)
        return 0.0;

    double e0 = total_energy[0];
    if (e0 == 0.0)
        return 0.0;

    double de = std::abs(total_energy[step] - e0);
    return de / std::abs(e0);
}

/**
 * @brief Get average relative error
 */
double EnergyHistory::average_error(int start_step, int end_step) const {
    if (end_step < 0 || end_step >= (int)total_energy.size()) {
        end_step = total_energy.size() - 1;
    }
    if (start_step >= end_step)
        return 0.0;

    double total_error = 0.0;
    int count = 0;

    for (int i = start_step + 1; i <= end_step; ++i) {
        total_error += conservation_error(i);
        count++;
    }

    return (count > 0) ? total_error / count : 0.0;
}

/**
 * @brief Validate energy conservation
 */
bool validate_energy_conservation(const EnergyHistory& history, double tolerance, int start_step,
                                  int end_step) {

    if (history.total_energy.empty())
        return false;

    if (end_step < 0) {
        end_step = history.total_energy.size() - 1;
    }

    double avg_error = history.average_error(start_step, end_step);

    return avg_error <= tolerance;
}

} // namespace jericho
