/**
 * @file boris_pusher.h
 * @brief Boris algorithm for accurate particle advancement in electromagnetic fields
 * @author Jericho Mk II Development Team
 *
 * The Boris algorithm advances particle velocities using:
 *   v^(n+1/2) = v^n + (q/m)*E*dt/2
 *   v^* = rotate(v^(n+1/2), B, dt)  [Lorentz rotation]
 *   v^(n+1) = v^(n+1/2) + (q/m)*E*dt/2
 *
 * Key advantages:
 * - Second-order accurate in time
 * - Preserves energy exactly for E=0, B=const
 * - Handles strong magnetic fields without stability issues
 * - Simple and efficient on GPU (8 FLOPs per particle per component)
 *
 * References:
 * - Birdsall & Langdon (1985), Plasma Physics via Computer Simulation
 * - Qiang et al. (2000), Physical Review E 64, 016502
 */

#pragma once

#include "particle_buffer.h"

#include <cmath>
#include <vector>

namespace jericho {

/**
 * @brief Configuration for Boris pusher
 */
struct BorisConfig {
    bool use_subcycling = false;    ///< Subcycle for strong B fields
    int max_subcycles = 10;         ///< Max subcycles if enabled
    double max_rotation = M_PI / 4; ///< Max rotation per subcycle [rad]
    bool use_device = false;        ///< Use GPU if available

    // Rotating frame physics (NEW)
    bool enable_rotating_frame = false; ///< Enable rotating frame
    double Omega = 0.0;                 ///< Angular velocity [rad/s]
    bool enable_coriolis = true;        ///< Enable Coriolis force
    bool enable_centrifugal = true;     ///< Enable centrifugal force
};

/**
 * @brief Advance particles using Boris algorithm
 *
 * Updates particle velocities using the Boris algorithm for accurate
 * integration of the Lorentz force: dv/dt = (q/m)(E + v×B)
 *
 * @param[in,out] particles Particle buffer (velocities modified in-place)
 * @param[in] Ex, Ey Electric field on grid [V/m]
 * @param[in] Bz Magnetic field on grid [T]
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing [m]
 * @param[in] dt Time step [s]
 * @param[in] config Boris configuration
 * @param[in] species_type 0=electrons, 1=ions (for q/m ratio)
 */
void advance_particles_boris(ParticleBuffer& particles, const double* Ex, const double* Ey,
                             const double* Bz, int nx, int ny, double dx, double dy, double dt,
                             const BorisConfig& config, int species_type);

/**
 * @brief Compute total kinetic energy of particles
 *
 * Integrates kinetic energy over all particles:
 *   KE = Σ_i (1/2) * m_i * |v_i|²
 *
 * @param[in] particles Particle buffer with velocities
 * @param[in] species_type 0=electrons, 1=ions (determines mass)
 * @return Total kinetic energy [J]
 */
double compute_kinetic_energy(const ParticleBuffer& particles, int species_type);

/**
 * @brief Track energy conservation
 *
 * Monitors total energy and energy change rate
 */
struct EnergyHistory {
    std::vector<double> em_energy;     ///< EM energy per step [J]
    std::vector<double> ke_ions;       ///< Ion KE per step [J]
    std::vector<double> ke_electrons;  ///< Electron KE per step [J]
    std::vector<double> total_energy;  ///< Total energy per step [J]
    std::vector<double> energy_change; ///< dE/dt per step [J/s]

    /**
     * @brief Compute energy conservation error
     * @return Relative error: |ΔE|/E₀ per step
     */
    double conservation_error(int step) const;

    /**
     * @brief Get average relative error over range
     */
    double average_error(int start_step, int end_step) const;
};

/**
 * @brief Validate energy conservation
 *
 * Checks that total energy change satisfies:
 *   |d(EM + KE)/dt| / E_total < tolerance
 *
 * @param[in] history Energy history structure
 * @param[in] tolerance Relative tolerance (e.g., 0.01 for ±1%)
 * @param[in] start_step First step to check
 * @param[in] end_step Last step to check
 * @return true if within tolerance, false otherwise
 */
bool validate_energy_conservation(const EnergyHistory& history, double tolerance,
                                  int start_step = 0, int end_step = -1);

/**
 * @brief Helper: Interpolate field value at particle position
 *
 * Uses bilinear interpolation to get field value at (x, y)
 *
 * @param[in] field Grid field array
 * @param[in] x, y Position [m]
 * @param[in] dx, dy Grid spacing [m]
 * @param[in] nx, ny Grid dimensions
 * @return Interpolated field value
 */
double interpolate_field(const double* field, double x, double y, double dx, double dy, int nx,
                         int ny);

/**
 * @brief Physical constants
 */
namespace constants {
constexpr double PROTON_MASS = 1.67262192e-27;        // kg
constexpr double ELECTRON_MASS = 9.1093837e-31;       // kg
constexpr double ELEMENTARY_CHARGE = 1.60217663e-19;  // C
constexpr double VACUUM_PERMITTIVITY = 8.8541878e-12; // F/m
constexpr double SPEED_OF_LIGHT = 2.99792458e8;       // m/s

/**
 * @brief Get mass for species type
 * @param species_type 0=electrons, 1+=ions
 */
inline double get_mass(int species_type) {
    return (species_type == 0) ? ELECTRON_MASS : PROTON_MASS;
}

/**
 * @brief Get charge for species type
 * @param species_type 0=electrons, 1+=ions
 */
inline double get_charge(int species_type) {
    return (species_type == 0) ? -ELEMENTARY_CHARGE : ELEMENTARY_CHARGE;
}

/**
 * @brief Get charge-to-mass ratio
 */
inline double get_qm_ratio(int species_type) {
    double m = get_mass(species_type);
    double q = get_charge(species_type);
    return (m > 0) ? q / m : 0.0;
}
} // namespace constants

} // namespace jericho
