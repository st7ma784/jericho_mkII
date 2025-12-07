/**
 * @file ohms_law_solver.h
 * @brief Generalized Ohm's Law electric field solver for MHD plasma simulations
 * @author Jericho Mk II Development Team
 *
 * Implements generalized Ohm's law for calculating electric fields in magnetized plasmas:
 *
 *   E = -U×B - (∇×B)/(μ₀·q·n) - ∇p/(q·n)
 *       [Convective] [Hall term] [Pressure - optional]
 *
 * where:
 *   U = J/q  is the bulk flow velocity
 *   J = current density from particles
 *   q = charge density from particles
 *   n = number density
 *
 * This is an alternative to the pure Maxwell (Ampere) solver, appropriate for
 * magnetospheric and MHD-scale plasma simulations where ion inertia effects matter.
 *
 * Physics Context:
 * - Convective term (-U×B): Dominant at MHD scales
 * - Hall term: Important at ion gyroradius scales (~10-100 km in magnetosphere)
 * - Captures dispersive waves: whistlers, kinetic Alfvén waves
 *
 * References:
 * - Winske & Omidi (1993), J. Geophys. Res.
 * - Karimabadi et al. (2014), Phys. Plasmas
 */

#pragma once

#include "field_arrays.h"
#include "mpi_manager.h"

#include <vector>

namespace jericho {

/**
 * @brief Configuration for Ohm's Law solver
 */
struct OhmsLawConfig {
    bool use_hall_term = true;     ///< Include Hall term (ion-scale physics)
    double q_min = 1.0e-15;         ///< Charge density floor [C/m³]
    bool use_smoothing = false;     ///< Apply charge density smoothing
    bool use_tapering = false;      ///< Taper Hall term near boundaries
    int taper_width = 5;            ///< Boundary taper width [cells]

    // Physical constants (can override for normalized units)
    double mu_0 = 1.25663706e-6;    ///< Permeability of free space [H/m]
    double epsilon_0 = 8.8541878e-12; ///< Permittivity of free space [F/m]
};

/**
 * @brief Compute bulk flow velocity from current and charge density
 *
 * Computes U = J/q at each grid point with charge density floor to prevent
 * division by zero in depleted regions.
 *
 * @param[in] Jx, Jy Current density components [A/m²]
 * @param[in] charge_density Charge density [C/m³]
 * @param[out] Ux, Uy Flow velocity components [m/s]
 * @param[in] nx, ny Grid dimensions
 * @param[in] config Solver configuration (for q_min)
 */
void compute_flow_velocity(
    const double* Jx, const double* Jy,
    const double* charge_density,
    double* Ux, double* Uy,
    int nx, int ny,
    const OhmsLawConfig& config
);

/**
 * @brief Solve electric field using generalized Ohm's law
 *
 * Computes electric field from:
 *   E = -U×B + E_hall
 *
 * Where E_hall = -∇×B/(μ₀·q·n) if use_hall_term is enabled.
 *
 * For 2.5D geometry (Bz only):
 *   Ex = -Uy·Bz - ∂Bz/∂x / (μ₀·q)
 *   Ey =  Ux·Bz - ∂Bz/∂y / (μ₀·q)
 *
 * @param[in] Ux, Uy Flow velocity [m/s]
 * @param[in] Bz Magnetic field [T]
 * @param[in] charge_density Charge density [C/m³]
 * @param[out] Ex, Ey Electric field components [V/m]
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing [m]
 * @param[in] config Solver configuration
 */
void solve_electric_field_ohms_law(
    const double* Ux, const double* Uy,
    const double* Bz,
    const double* charge_density,
    double* Ex, double* Ey,
    int nx, int ny,
    double dx, double dy,
    const OhmsLawConfig& config
);

/**
 * @brief Apply charge density smoothing (optional)
 *
 * Smooths charge density to prevent sharp gradients that can cause
 * numerical issues in Hall term calculation: ∇(B²)/q
 *
 * Uses 4-point stencil (von Neumann neighborhood):
 *   q_smooth = (1-α)·q + α·avg(q_neighbors)
 *
 * @param[in] q_in Input charge density
 * @param[out] q_out Smoothed charge density
 * @param[in] nx, ny Grid dimensions
 * @param[in] smoothing_factor Blend factor α ∈ [0,1]
 */
void smooth_charge_density(
    const double* q_in,
    double* q_out,
    int nx, int ny,
    double smoothing_factor = 0.5
);

/**
 * @brief Compute Hall term: -∇×B/(μ₀·q)
 *
 * For 2D with Bz only:
 *   E_hall_x = -∂Bz/∂x / (μ₀·q)
 *   E_hall_y = -∂Bz/∂y / (μ₀·q)
 *
 * Alternative formulation (used in tidy_jeri):
 *   E_hall = -∇(B²)/(2μ₀·q)
 *
 * @param[in] Bz Magnetic field [T]
 * @param[in] charge_density Charge density [C/m³]
 * @param[out] Ex_hall, Ey_hall Hall electric field [V/m]
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing [m]
 * @param[in] config Solver configuration
 */
void compute_hall_term(
    const double* Bz,
    const double* charge_density,
    double* Ex_hall, double* Ey_hall,
    int nx, int ny,
    double dx, double dy,
    const OhmsLawConfig& config
);

/**
 * @brief Apply boundary tapering to Hall term
 *
 * Linearly tapers Hall term from full strength (interior) to zero (boundaries)
 * over a buffer zone of specified width. Prevents NaN at inflow/outflow
 * boundaries where particles are lost and charge density → 0.
 *
 * Taper function:
 *   hall_factor(d) = min(d / width, 1.0)
 *
 * where d is distance from nearest boundary.
 *
 * @param[in,out] Ex_hall, Ey_hall Hall term (modified in place)
 * @param[in] nx, ny Grid dimensions
 * @param[in] taper_width Width of taper zone [cells]
 */
void apply_hall_term_tapering(
    double* Ex_hall, double* Ey_hall,
    int nx, int ny,
    int taper_width
);

/**
 * @brief Full Ohm's Law solver pipeline
 *
 * Complete solver that:
 * 1. Computes flow velocity U = J/q
 * 2. Computes convective term -U×B
 * 3. Optionally computes Hall term -∇×B/(μ₀·q)
 * 4. Optionally applies boundary tapering
 *
 * This is the main entry point for using Ohm's Law instead of Ampere-Maxwell.
 *
 * @param[in] Jx, Jy Current density [A/m²]
 * @param[in] Bz Magnetic field [T]
 * @param[in] charge_density Charge density [C/m³]
 * @param[out] Ex, Ey Electric field [V/m]
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing [m]
 * @param[in] config Solver configuration
 */
void solve_ohms_law_full(
    const double* Jx, const double* Jy,
    const double* Bz,
    const double* charge_density,
    double* Ex, double* Ey,
    int nx, int ny,
    double dx, double dy,
    const OhmsLawConfig& config
);

/**
 * @brief Diagnostics: Compute Hall parameter β_Hall = d_i / L
 *
 * Measures importance of Hall physics:
 *   β_Hall << 1: MHD regime (Hall term negligible)
 *   β_Hall ~ 1:  Hall MHD regime (Hall term important)
 *   β_Hall >> 1: Kinetic regime (fluid approximation breaks down)
 *
 * where d_i = c/(ω_pi) is the ion inertial length
 *
 * @param[in] charge_density Charge density [C/m³]
 * @param[in] Bz Magnetic field [T]
 * @param[in] dx Grid spacing [m]
 * @param[in] nx, ny Grid dimensions
 * @return Average Hall parameter
 */
double compute_hall_parameter(
    const double* charge_density,
    const double* Bz,
    double dx,
    int nx, int ny
);

} // namespace jericho
