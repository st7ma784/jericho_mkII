/**
 * @file ampere_solver.h
 * @brief Ampere's Law and magnetic field solver for MPI PIC simulations
 * @author Jericho Mk II Development Team
 *
 * Phase 11.3: Magnetic field evolution and Ampere's Law integration
 *
 * Implements:
 * - Faraday's law: ∂B/∂t = -∇×E  (electric field → magnetic evolution)
 * - Ampere's law: ∂E/∂t = (1/μ₀ε₀)∇×B - μ₀J  (magnetic field + current → electric evolution)
 * - Current density computation from particles
 * - Global current collection via MPI AllReduce
 * - Energy conservation verification
 */

#pragma once

#include "field_arrays.h"
#include "mpi_domain_state.h"
#include <vector>

namespace jericho {

// ============================================================================
// Ampere/Faraday Solver Configuration
// ============================================================================

/**
 * @struct AmpereConfig
 * @brief Configuration for Ampere's Law and magnetic field solver
 */
struct AmpereConfig {
    /// Permeability of free space [H/m]
    static constexpr double mu_0 = 1.25663706212e-6;
    
    /// Permittivity of free space [F/m]
    static constexpr double epsilon_0 = 8.854187817e-12;
    
    /// Speed of light [m/s]
    static constexpr double c = 299792458.0;
    
    /// Time integration scheme: "euler" or "predictor-corrector"
    bool use_pc_corrector = true;
    
    /// Apply energy conservation correction
    bool apply_energy_correction = true;
    
    /// Compute current density from particles
    bool compute_current = true;
    
    /// Number of substeps for field evolution (for accuracy)
    int num_substeps = 1;
};

// ============================================================================
// Current Density Computation (from particles)
// ============================================================================

/**
 * @brief Compute current density from particle velocities
 *
 * Accumulates current from particles on grid:
 * J = Σ_p q_p * v_p * δ(r - r_p)
 *
 * Uses area-weighting (scatter from particles to grid cells).
 *
 * @param[in] particles Particle buffer with positions and velocities
 * @param[out] Jx, Jy Current density components (output)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void compute_current_density(const double* particle_x, const double* particle_y,
                             const double* particle_vx, const double* particle_vy,
                             const double* particle_q, int num_particles,
                             double* Jx, double* Jy,
                             int nx, int ny,
                             double dx, double dy,
                             bool use_device = true);

/**
 * @brief Zero out current density arrays
 *
 * @param[out] Jx, Jy Current density arrays
 * @param[in] nx, ny Grid dimensions
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void zero_current_density(double* Jx, double* Jy,
                         int nx, int ny,
                         bool use_device = true);

// ============================================================================
// Global Current Collection (MPI)
// ============================================================================

/**
 * @brief Collect global current density from all MPI ranks
 *
 * Performs MPI AllReduce to sum local current densities across all ranks.
 * Creates global current distribution for diagnostics or global solve.
 *
 * @param[in] Jx_local, Jy_local Local current density (per rank)
 * @param[out] Jx_global, Jy_global Global current density (output)
 * @param[in] nx, ny Grid dimensions
 * @param[in] mpi_state MPI domain state with communicator
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Total integrated current (for diagnostics)
 */
double collect_global_current(const double* Jx_local, const double* Jy_local,
                              std::vector<double>& Jx_global, std::vector<double>& Jy_global,
                              int nx, int ny,
                              const MPIDomainState& mpi_state,
                              bool use_device = true);

/**
 * @brief Compute total current magnitude (with AllReduce)
 *
 * @param[in] Jx, Jy Current density
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] mpi_state MPI domain state
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Total integrated current magnitude
 */
double compute_global_current_magnitude(const double* Jx, const double* Jy,
                                        int nx, int ny,
                                        double dx, double dy,
                                        const MPIDomainState& mpi_state,
                                        bool use_device = true);

// ============================================================================
// Curl Operator (for Maxwell's Equations)
// ============================================================================

/**
 * @brief Compute curl of vector field in 2D: ∇×F = (∂Fy/∂x - ∂Fx/∂y)ẑ
 *
 * @param[in] Fx, Fy Field components
 * @param[out] curl_z Output curl (z-component)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void compute_curl_2d(const double* Fx, const double* Fy,
                     double* curl_z,
                     int nx, int ny,
                     double dx, double dy,
                     bool use_device = true);

// ============================================================================
// Faraday's Law: ∂B/∂t = -∇×E
// ============================================================================

/**
 * @brief Advance magnetic field using Faraday's Law
 *
 * Faraday's law: ∂B/∂t = -∇×E
 *
 * Time discretization (Leapfrog):
 *   B^{n+1/2} = B^{n-1/2} - dt·∇×E^n
 *
 * Or Predictor-Corrector for improved stability:
 *   B* = B^n - dt·∇×E^n          (predictor)
 *   B^{n+1} = B^n - (dt/2)·(∇×E^n + ∇×E*)  (corrector)
 *
 * @param[in,out] Bz_old Magnetic field (input and output)
 * @param[in] Ex, Ey Electric field
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] dt Timestep
 * @param[in] use_pc_corrector If true, use predictor-corrector; else Euler
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Maximum field change (for stability monitoring)
 */
double advance_magnetic_field_faraday(double* Bz,
                                      const double* Ex, const double* Ey,
                                      int nx, int ny,
                                      double dx, double dy,
                                      double dt,
                                      bool use_pc_corrector = true,
                                      bool use_device = true);

/**
 * @brief Advance magnetic field - Euler method (simple but less accurate)
 *
 * @param[in,out] Bz_old Magnetic field
 * @param[in] Ex, Ey Electric field
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] dt Timestep
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void advance_magnetic_field_euler(double* Bz,
                                  const double* Ex, const double* Ey,
                                  int nx, int ny,
                                  double dx, double dy,
                                  double dt,
                                  bool use_device = true);

/**
 * @brief Advance magnetic field - Predictor-Corrector method (more accurate)
 *
 * B* = B^n - dt·∇×E^n          (predictor: evaluate at half-step)
 * B^{n+1} = B^n - (dt/2)·(∇×E^n + ∇×E*)  (corrector: average)
 *
 * @param[in,out] Bz Magnetic field
 * @param[in] Ex, Ey Electric field
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] dt Timestep
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void advance_magnetic_field_pc(double* Bz,
                               const double* Ex, const double* Ey,
                               int nx, int ny,
                               double dx, double dy,
                               double dt,
                               bool use_device = true);

// ============================================================================
// Ampere-Maxwell Law (Full Ampere's Law with Displacement Current)
// ============================================================================

/**
 * @brief Advance electric field using Ampere-Maxwell Law
 *
 * ∂E/∂t = (1/μ₀ε₀)∇×B - (1/ε₀)J
 *       = c²∇×B - (1/ε₀)J
 *
 * Where:
 * - First term is from magnetic field curl (displacement current)
 * - Second term is source term from plasma current density
 *
 * @param[in,out] Ex, Ey Electric field (updated)
 * @param[in] Bz Magnetic field
 * @param[in] Jx, Jy Current density (source)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] dt Timestep
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void advance_electric_field_ampere(double* Ex, double* Ey,
                                   const double* Bz,
                                   const double* Jx, const double* Jy,
                                   int nx, int ny,
                                   double dx, double dy,
                                   double dt,
                                   bool use_device = true);

// ============================================================================
// Energy Conservation & Monitoring
// ============================================================================

/**
 * @brief Compute electromagnetic energy density
 *
 * Energy density: u = (1/2)·(ε₀E² + B²/μ₀)
 * Total energy: U = ∫ u dV
 *
 * @param[in] Ex, Ey, Bz Field components
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] mpi_state MPI domain state (for global reduction)
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Total electromagnetic energy (integrated over domain)
 */
double compute_electromagnetic_energy(const double* Ex, const double* Ey, const double* Bz,
                                      int nx, int ny,
                                      double dx, double dy,
                                      const MPIDomainState& mpi_state,
                                      bool use_device = true);

/**
 * @brief Compute Poynting vector (energy flux)
 *
 * S = (1/μ₀)·E×B  (energy flux density)
 *
 * In 2D with Bz: Sx = (1/μ₀)·Ey·Bz, Sy = -(1/μ₀)·Ex·Bz
 *
 * @param[in] Ex, Ey, Bz Field components
 * @param[out] Sx, Sy Poynting vector components
 * @param[in] nx, ny Grid dimensions
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void compute_poynting_vector(const double* Ex, const double* Ey, const double* Bz,
                             double* Sx, double* Sy,
                             int nx, int ny,
                             bool use_device = true);

/**
 * @brief Verify energy conservation
 *
 * Check that dU/dt + ∇·S = -J·E (power dissipation)
 * Returns relative error in energy balance.
 *
 * @param[in] Ex, Ey, Bz Field components
 * @param[in] Jx, Jy Current density
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] dt Timestep
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Relative energy balance error
 */
double verify_energy_conservation(const double* Ex, const double* Ey, const double* Bz,
                                  const double* Jx, const double* Jy,
                                  int nx, int ny,
                                  double dx, double dy,
                                  double dt,
                                  bool use_device = true);

// ============================================================================
// Diagnostics & Output
// ============================================================================

/**
 * @brief Print Ampere solver statistics
 *
 * @param[in] fields Field arrays (for diagnostics)
 * @param[in] mpi_state MPI domain state
 * @param[in] iteration_count Current iteration number
 * @param[in] total_energy Electromagnetic energy
 * @param[in] energy_conservation_error Relative error in energy balance
 */
void print_ampere_stats(const FieldArrays& fields,
                       const MPIDomainState& mpi_state,
                       int iteration_count,
                       double total_energy,
                       double energy_conservation_error = -1.0);

} // namespace jericho

