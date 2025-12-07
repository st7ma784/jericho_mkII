/**
 * @file poisson_solver.h
 * @brief Poisson solver for electric field computation in MPI simulations
 * @author Jericho Mk II Development Team
 *
 * Phase 11.2: Solves Poisson equation: ∇²Φ = -ρ/ε₀
 *
 * Computes electric potential from charge density, then derives E field.
 * Supports both local (per-rank) and global charge density collection.
 */

#pragma once

#include "field_arrays.h"
#include "mpi_domain_state.h"

#include <vector>

namespace jericho {

// ============================================================================
// Poisson Solver Configuration
// ============================================================================

/**
 * @struct PoissonConfig
 * @brief Configuration parameters for Poisson solver
 */
struct PoissonConfig {
    /// Permittivity of free space [F/m]
    static constexpr double epsilon_0 = 8.854187817e-12;

    /// Maximum iterations for iterative solver (SOR/Jacobi)
    int max_iterations = 1000;

    /// Convergence tolerance for iterative solver
    double convergence_tol = 1e-6;

    /// SOR (Successive Over-Relaxation) parameter (1.0 = Jacobi, ~1.8 optimal)
    double omega = 1.8;

    /// Use FFT-based solver (true) or iterative (false)
    bool use_fft = true;

    /// Apply boundary conditions (Dirichlet/Neumann)
    bool apply_bc = true;

    /// Periodicity: true for periodic BC, false for open BC
    bool periodic_bc = false;
};

// ============================================================================
// Global Charge Density Collection (Phase 11.2 - MPI)
// ============================================================================

/**
 * @brief Collect global charge density from all ranks via MPI AllReduce
 *
 * Sums local charge densities across all ranks to create global charge
 * distribution. Used for global Poisson solve or diagnostics.
 *
 * @param[in] fields Local field arrays (contains local charge_density)
 * @param[out] global_charge_density Output array (pre-allocated on host)
 * @param[in] mpi_state MPI domain state with topology info
 * @param[in] use_device If true, charge_density is on device (GPU)
 *
 * @return Total integrated charge (for diagnostics)
 */
double collect_global_charge_density(const FieldArrays& fields,
                                     std::vector<double>& global_charge_density,
                                     const MPIDomainState& mpi_state, bool use_device = true);

/**
 * @brief Compute total charge integrated over domain (with AllReduce)
 *
 * @param[in] charge_density Charge density array (local domain)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] mpi_state MPI domain state
 * @param[in] use_device If true, array is on device (GPU)
 *
 * @return Total integrated charge
 */
double compute_global_charge(const double* charge_density, int nx, int ny, double dx, double dy,
                             const MPIDomainState& mpi_state, bool use_device = true);

// ============================================================================
// Poisson Equation Solver
// ============================================================================

/**
 * @brief Solve Poisson equation: ∇²Φ = -ρ/ε₀
 *
 * Computes electric potential from charge density using specified method.
 * For iterative solvers: uses SOR with configurable omega.
 * For FFT solvers: assumes periodic boundary conditions.
 *
 * @param[in] charge_density Source term (-ρ/ε₀)
 * @param[out] potential Electric potential (output)
 * @param[in] nx, ny Grid dimensions (including ghost cells)
 * @param[in] dx, dy Grid spacing
 * @param[in] config Solver configuration
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Number of iterations (iterative) or 1 (FFT)
 */
int solve_poisson_equation(const double* charge_density, double* potential, int nx, int ny,
                           double dx, double dy, const PoissonConfig& config,
                           bool use_device = true);

/**
 * @brief Solve Poisson equation using SOR (Successive Over-Relaxation)
 *
 * Iterative method: stable, guaranteed convergence, but slow.
 * Good for small domains or validation.
 *
 * @param[in] rhs Right-hand side (-ρ/ε₀)
 * @param[in,out] solution Electric potential (initial + final)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] omega SOR parameter (1.0 = Jacobi, ~1.8 optimal)
 * @param[in] max_iter Maximum iterations
 * @param[in] tol Convergence tolerance
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return Number of iterations to convergence
 */
int solve_poisson_sor(const double* rhs, double* solution, int nx, int ny, double dx, double dy,
                      double omega = 1.8, int max_iter = 1000, double tol = 1e-6,
                      bool use_device = true);

/**
 * @brief Solve Poisson equation using FFT (Fast Fourier Transform)
 *
 * Fast spectral method: O(N log N) complexity.
 * Requires periodic boundary conditions.
 * Note: Currently placeholder - full implementation in Phase 11.2.
 *
 * @param[in] rhs Right-hand side (-ρ/ε₀)
 * @param[out] solution Electric potential
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return 1 (FFT completes in single pass)
 */
int solve_poisson_fft(const double* rhs, double* solution, int nx, int ny, double dx, double dy,
                      bool use_device = true);

// ============================================================================
// Electric Field Computation from Potential
// ============================================================================

/**
 * @brief Compute electric field from potential: E = -∇Φ
 *
 * Uses central differences:
 * Ex = -(Φ[i+1,j] - Φ[i-1,j]) / (2·dx)
 * Ey = -(Φ[i,j+1] - Φ[i,j-1]) / (2·dy)
 *
 * @param[in] potential Electric potential (from Poisson solver)
 * @param[out] Ex, Ey Electric field components (output)
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void compute_electric_field(const double* potential, double* Ex, double* Ey, int nx, int ny,
                            double dx, double dy, bool use_device = true);

/**
 * @brief Apply boundary conditions to electric field
 *
 * For periodic BC: no action needed (handled by solver).
 * For open BC: enforces E→0 at edges.
 * For Dirichlet BC: enforces E from background field.
 *
 * @param[in,out] Ex, Ey Electric field components
 * @param[in] Ex_bg, Ey_bg Background electric field (for Dirichlet)
 * @param[in] nx, ny Grid dimensions
 * @param[in] bc_type "periodic", "open", or "dirichlet"
 * @param[in] use_device If true, arrays are on device (GPU)
 */
void apply_field_boundary_conditions(double* Ex, double* Ey, const double* Ex_bg,
                                     const double* Ey_bg, int nx, int ny,
                                     const char* bc_type = "periodic", bool use_device = true);

// ============================================================================
// Validation & Diagnostics
// ============================================================================

/**
 * @brief Verify Poisson solution: compute ∇²Φ and compare to source
 *
 * Validates that computed potential satisfies Poisson equation.
 * Useful for debugging and convergence checking.
 *
 * @param[in] potential Computed electric potential
 * @param[in] charge_density Source charge density
 * @param[in] nx, ny Grid dimensions
 * @param[in] dx, dy Grid spacing
 * @param[in] use_device If true, arrays are on device (GPU)
 *
 * @return RMS error in Poisson equation: ||∇²Φ + ρ/ε₀||
 */
double verify_poisson_solution(const double* potential, const double* charge_density, int nx,
                               int ny, double dx, double dy, bool use_device = true);

/**
 * @brief Print Poisson solver statistics
 *
 * @param[in] fields Field arrays (for diagnostics)
 * @param[in] mpi_state MPI domain state
 * @param[in] iteration_count Number of iterations (if iterative solver)
 * @param[in] convergence_error Final error (if iterative solver)
 */
void print_poisson_stats(const FieldArrays& fields, const MPIDomainState& mpi_state,
                         int iteration_count = -1, double convergence_error = -1.0);

} // namespace jericho
