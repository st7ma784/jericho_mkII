/**
 * @file poisson_solver.cpp
 * @brief Poisson solver implementation for electric field computation
 * @author Jericho Mk II Development Team
 *
 * Phase 11.2: Electric field from charge density
 * 
 * Implements:
 * - Global charge density collection via MPI AllReduce
 * - Poisson equation solver (SOR iterative or FFT)
 * - Electric field computation: E = -∇Φ
 * - Boundary condition application
 * - Solution validation and diagnostics
 */

#include "poisson_solver.h"
#include "platform.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <iostream>
#include <iomanip>

namespace jericho {

// ============================================================================
// Global Charge Density Collection (MPI AllReduce)
// ============================================================================

double collect_global_charge_density(const FieldArrays& fields,
                                     std::vector<double>& global_charge_density,
                                     const MPIDomainState& mpi_state,
                                     bool use_device) {
    int total_points = fields.nx * fields.ny;
    global_charge_density.resize(total_points, 0.0);
    
    // Allocate temporary host buffers
    std::vector<double> local_charge(total_points);
    std::vector<double> global_charge_sum(total_points);
    
    // Copy local charge density to host if needed
    if (use_device) {
        // In GPU mode: device_memcpy(dest, src, size, cudaMemcpyDeviceToHost)
        // For CPU-only: direct memcpy
        std::memcpy(local_charge.data(), fields.charge_density,
                   total_points * sizeof(double));
    } else {
        std::memcpy(local_charge.data(), fields.charge_density,
                   total_points * sizeof(double));
    }
    
    // Global reduction: sum all local charge densities
    MPI_Allreduce(local_charge.data(), global_charge_sum.data(), total_points,
                  MPI_DOUBLE, MPI_SUM, mpi_state.cart_comm);
    
    global_charge_density = global_charge_sum;
    
    // Compute total integrated charge
    double total_charge = 0.0;
    double dx = fields.dx;
    double dy = fields.dy;
    
    for (int iy = 1; iy < fields.ny - 1; ++iy) {
        for (int ix = 1; ix < fields.nx - 1; ++ix) {
            int idx = iy * fields.nx + ix;
            total_charge += global_charge_density[idx] * dx * dy;
        }
    }
    
    return total_charge;
}

double compute_global_charge(const double* charge_density,
                             int nx, int ny,
                             double dx, double dy,
                             const MPIDomainState& mpi_state,
                             bool use_device) {
    int total_points = nx * ny;
    
    // Local charge integration
    double local_charge = 0.0;
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;
            local_charge += charge_density[idx] * dx * dy;
        }
    }
    
    // Global reduction
    double global_charge = 0.0;
    MPI_Allreduce(&local_charge, &global_charge, 1, MPI_DOUBLE, MPI_SUM,
                  mpi_state.cart_comm);
    
    return global_charge;
}

// ============================================================================
// Poisson Equation Solver (Main Entry Point)
// ============================================================================

int solve_poisson_equation(const double* charge_density,
                           double* potential,
                           int nx, int ny,
                           double dx, double dy,
                           const PoissonConfig& config,
                           bool use_device) {
    
    if (config.use_fft) {
        // FFT-based solver (fast, requires periodic BC)
        return solve_poisson_fft(charge_density, potential, nx, ny, dx, dy, use_device);
    } else {
        // Iterative SOR solver (slower, but flexible)
        return solve_poisson_sor(charge_density, potential, nx, ny, dx, dy,
                                config.omega, config.max_iterations,
                                config.convergence_tol, use_device);
    }
}

// ============================================================================
// SOR (Successive Over-Relaxation) Iterative Solver
// ============================================================================

int solve_poisson_sor(const double* rhs,
                      double* solution,
                      int nx, int ny,
                      double dx, double dy,
                      double omega,
                      int max_iter,
                      double tol,
                      bool use_device) {
    
    // Precompute coefficients
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coeff = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2));
    double coeff_x = coeff / dx2;
    double coeff_y = coeff / dy2;
    
    // Allocate residual array
    std::vector<double> residual(nx * ny, 0.0);
    std::vector<double> solution_old(nx * ny, 0.0);
    
    int iteration = 0;
    double error = 1e10;
    
    // Main SOR loop
    for (iteration = 0; iteration < max_iter; ++iteration) {
        error = 0.0;
        
        // Red-black ordering for better cache locality
        // (checkerboard pattern: do red points, then black points)
        for (int iy = 1; iy < ny - 1; ++iy) {
            for (int ix = 1; ix < nx - 1; ++ix) {
                // Skip if not red point (ix + iy) % 2 == 0
                if ((ix + iy) % 2 != 0) continue;
                
                int idx = iy * nx + ix;
                int idx_xp = iy * nx + (ix + 1);
                int idx_xm = iy * nx + (ix - 1);
                int idx_yp = (iy + 1) * nx + ix;
                int idx_ym = (iy - 1) * nx + ix;
                
                double phi_new = coeff_x * (solution[idx_xp] + solution[idx_xm]) +
                                coeff_y * (solution[idx_yp] + solution[idx_ym]) -
                                coeff * rhs[idx];
                
                solution[idx] = (1.0 - omega) * solution[idx] + omega * phi_new;
            }
        }
        
        // Black points
        for (int iy = 1; iy < ny - 1; ++iy) {
            for (int ix = 1; ix < nx - 1; ++ix) {
                // Skip if not black point (ix + iy) % 2 != 0
                if ((ix + iy) % 2 == 0) continue;
                
                int idx = iy * nx + ix;
                int idx_xp = iy * nx + (ix + 1);
                int idx_xm = iy * nx + (ix - 1);
                int idx_yp = (iy + 1) * nx + ix;
                int idx_ym = (iy - 1) * nx + ix;
                
                double phi_new = coeff_x * (solution[idx_xp] + solution[idx_xm]) +
                                coeff_y * (solution[idx_yp] + solution[idx_ym]) -
                                coeff * rhs[idx];
                
                solution[idx] = (1.0 - omega) * solution[idx] + omega * phi_new;
            }
        }
        
        // Compute residual and error
        for (int iy = 1; iy < ny - 1; ++iy) {
            for (int ix = 1; ix < nx - 1; ++ix) {
                int idx = iy * nx + ix;
                int idx_xp = iy * nx + (ix + 1);
                int idx_xm = iy * nx + (ix - 1);
                int idx_yp = (iy + 1) * nx + ix;
                int idx_ym = (iy - 1) * nx + ix;
                
                double laplacian = (solution[idx_xp] + solution[idx_xm] - 2.0 * solution[idx]) / dx2 +
                                  (solution[idx_yp] + solution[idx_ym] - 2.0 * solution[idx]) / dy2;
                
                double res = rhs[idx] + laplacian;
                residual[idx] = res;
                error += res * res;
            }
        }
        
        error = std::sqrt(error / ((nx - 2) * (ny - 2)));
        
        // Check convergence
        if (error < tol) {
            break;
        }
    }
    
    return iteration;
}

// ============================================================================
// FFT-Based Poisson Solver (Placeholder)
// ============================================================================

int solve_poisson_fft(const double* rhs,
                      double* solution,
                      int nx, int ny,
                      double dx, double dy,
                      bool use_device) {
    
    // Phase 11.2 Placeholder: Full FFT implementation with cuFFT/FFTPACK
    // 
    // Algorithm (for reference):
    // 1. Forward FFT: ρ̂ = FFT(ρ)
    // 2. Spectral multiply: Φ̂ = ρ̂ / (kx² + ky²)
    // 3. Inverse FFT: Φ = IFFT(Φ̂)
    // 4. Handle k=0 mode (constant offset)
    //
    // For now, fallback to SOR with tighter tolerance
    
    PoissonConfig config;
    config.use_fft = false;
    config.max_iterations = 5000;
    config.convergence_tol = 1e-8;
    
    return solve_poisson_sor(rhs, solution, nx, ny, dx, dy,
                            1.9, 5000, 1e-8, use_device);
}

// ============================================================================
// Electric Field Computation
// ============================================================================

void compute_electric_field(const double* potential,
                           double* Ex, double* Ey,
                           int nx, int ny,
                           double dx, double dy,
                           bool use_device) {
    
    // E = -∇Φ using central differences
    double inv_2dx = -1.0 / (2.0 * dx);
    double inv_2dy = -1.0 / (2.0 * dy);
    
    // Compute interior points
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;
            int idx_xp = iy * nx + (ix + 1);
            int idx_xm = iy * nx + (ix - 1);
            int idx_yp = (iy + 1) * nx + ix;
            int idx_ym = (iy - 1) * nx + ix;
            
            // Ex = -∂Φ/∂x
            Ex[idx] = inv_2dx * (potential[idx_xp] - potential[idx_xm]);
            
            // Ey = -∂Φ/∂y
            Ey[idx] = inv_2dy * (potential[idx_yp] - potential[idx_ym]);
        }
    }
    
    // Handle boundaries (set to boundary value or 0)
    for (int iy = 0; iy < ny; ++iy) {
        Ex[iy * nx + 0] = 0.0;
        Ex[iy * nx + (nx - 1)] = 0.0;
        Ey[iy * nx + 0] = 0.0;
        Ey[iy * nx + (nx - 1)] = 0.0;
    }
    for (int ix = 0; ix < nx; ++ix) {
        Ex[0 * nx + ix] = 0.0;
        Ex[(ny - 1) * nx + ix] = 0.0;
        Ey[0 * nx + ix] = 0.0;
        Ey[(ny - 1) * nx + ix] = 0.0;
    }
}

void apply_field_boundary_conditions(double* Ex, double* Ey,
                                     const double* Ex_bg, const double* Ey_bg,
                                     int nx, int ny,
                                     const char* bc_type,
                                     bool use_device) {
    
    if (std::strcmp(bc_type, "periodic") == 0) {
        // Periodic BC: no special handling needed (FFT handles it)
        return;
    }
    else if (std::strcmp(bc_type, "open") == 0) {
        // Open BC: E → 0 at boundaries
        for (int iy = 0; iy < ny; ++iy) {
            Ex[iy * nx + 0] = 0.0;
            Ex[iy * nx + (nx - 1)] = 0.0;
            Ey[iy * nx + 0] = 0.0;
            Ey[iy * nx + (nx - 1)] = 0.0;
        }
        for (int ix = 0; ix < nx; ++ix) {
            Ex[0 * nx + ix] = 0.0;
            Ex[(ny - 1) * nx + ix] = 0.0;
            Ey[0 * nx + ix] = 0.0;
            Ey[(ny - 1) * nx + ix] = 0.0;
        }
    }
    else if (std::strcmp(bc_type, "dirichlet") == 0) {
        // Dirichlet BC: E = E_background
        for (int iy = 0; iy < ny; ++iy) {
            int idx_0 = iy * nx + 0;
            int idx_n = iy * nx + (nx - 1);
            Ex[idx_0] = Ex_bg[idx_0];
            Ex[idx_n] = Ex_bg[idx_n];
            Ey[idx_0] = Ey_bg[idx_0];
            Ey[idx_n] = Ey_bg[idx_n];
        }
        for (int ix = 0; ix < nx; ++ix) {
            int idx_0 = 0 * nx + ix;
            int idx_n = (ny - 1) * nx + ix;
            Ex[idx_0] = Ex_bg[idx_0];
            Ex[idx_n] = Ex_bg[idx_n];
            Ey[idx_0] = Ey_bg[idx_0];
            Ey[idx_n] = Ey_bg[idx_n];
        }
    }
}

// ============================================================================
// Solution Validation
// ============================================================================

double verify_poisson_solution(const double* potential,
                              const double* charge_density,
                              int nx, int ny,
                              double dx, double dy,
                              bool use_device) {
    
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double inv_eps0 = 1.0 / PoissonConfig::epsilon_0;
    
    double error_sum = 0.0;
    int count = 0;
    
    // Compute ∇²Φ at each interior point
    for (int iy = 1; iy < ny - 1; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            int idx = iy * nx + ix;
            int idx_xp = iy * nx + (ix + 1);
            int idx_xm = iy * nx + (ix - 1);
            int idx_yp = (iy + 1) * nx + ix;
            int idx_ym = (iy - 1) * nx + ix;
            
            double laplacian = (potential[idx_xp] + potential[idx_xm] - 2.0 * potential[idx]) / dx2 +
                              (potential[idx_yp] + potential[idx_ym] - 2.0 * potential[idx]) / dy2;
            
            // Poisson equation: ∇²Φ = -ρ/ε₀
            double expected = -charge_density[idx] * inv_eps0;
            double error = laplacian - expected;
            
            error_sum += error * error;
            count++;
        }
    }
    
    return std::sqrt(error_sum / count);
}

// ============================================================================
// Diagnostics
// ============================================================================

void print_poisson_stats(const FieldArrays& fields,
                        const MPIDomainState& mpi_state,
                        int iteration_count,
                        double convergence_error) {
    
    if (mpi_state.rank != 0) return;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "POISSON SOLVER STATISTICS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    if (iteration_count >= 0) {
        std::cout << "  Iterations to convergence: " << iteration_count << std::endl;
    }
    if (convergence_error >= 0.0) {
        std::cout << "  Final convergence error:   " << std::scientific << std::setprecision(3)
                 << convergence_error << std::endl;
    }
    
    std::cout << "  Grid size (local):         " << fields.nx_local << " × " << fields.ny_local
             << std::endl;
    std::cout << "  Grid spacing:              dx=" << fields.dx << " m, dy=" << fields.dy << " m"
             << std::endl;
    std::cout << "  Ranks: " << mpi_state.rank << "/" << (mpi_state.neighbors[0] + mpi_state.neighbors[1] +
                            mpi_state.neighbors[2] + mpi_state.neighbors[3]) << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

} // namespace jericho
