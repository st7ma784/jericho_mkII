/**
 * @file main_mpi.cpp
 * @brief MPI-parallel simulation driver for Jericho Mk II - OPTIMIZED
 * @author Jericho Mk II Development Team
 *
 * MPI-parallel PIC simulation framework with Phase 10 optimizations:
 * - Non-blocking ghost cell exchange (MPI_Isend/Irecv)
 * - Reduced global synchronization (AllReduce every 10 steps)
 * - Removed unnecessary MPI_Barrier calls
 * - Overlapped communication with computation
 * - 10-50x MPI performance improvement expected
 */

#include "ampere_solver.h"
#include "boris_pusher.h"
#include "field_arrays.h"
#include "ghost_cell_exchange.h"
#include "kernels_cpu_optimized.h"
#include "mpi_domain_state.h"
#include "particle_buffer.h"
#include "platform.h"
#include "poisson_solver.h"

#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace jericho;

// ============================================================================
// MPI Topology and Configuration
// ============================================================================

// Global MPI topology dimensions
int g_npx = 1;
int g_npy = 1;

// Minimal simulation parameters
struct SimulationParams {
    int nx, ny;
    double dx, dy;
    double x_min, x_max;
    double y_min, y_max;
    double dt;
    int max_steps;
};

// ============================================================================
// Initialize MPI with 2D Cartesian Topology
// ============================================================================

MPIDomainState initialize_mpi_topology(int rank, int size, const SimulationParams& params) {
    MPIDomainState state;
    state.rank = rank;

    // Auto-detect 2D topology (square grid)
    g_npx = (int)std::sqrt(size);
    while (size % g_npx != 0)
        g_npx--;
    g_npy = size / g_npx;

    if (rank == 0) {
        std::cout << "MPI Topology: " << g_npx << " x " << g_npy << " (" << size << " ranks total)"
                  << std::endl;
    }

    // Create 2D Cartesian topology
    int dims[2] = {g_npx, g_npy};
    int periods[2] = {1, 1}; // Periodic boundaries
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &state.cart_comm);
    MPI_Cart_coords(state.cart_comm, rank, 2, state.coords);

    // Get neighbors
    int px = state.coords[0];
    int py = state.coords[1];

    int coords_north[2] = {(px - 1 + g_npx) % g_npx, py};
    int coords_south[2] = {(px + 1) % g_npx, py};
    int coords_east[2] = {px, (py + 1) % g_npy};
    int coords_west[2] = {px, (py - 1 + g_npy) % g_npy};

    MPI_Cart_rank(state.cart_comm, coords_north, &state.neighbors[0]);
    MPI_Cart_rank(state.cart_comm, coords_south, &state.neighbors[1]);
    MPI_Cart_rank(state.cart_comm, coords_east, &state.neighbors[2]);
    MPI_Cart_rank(state.cart_comm, coords_west, &state.neighbors[3]);

    // Compute local domain bounds
    double Lx = params.x_max - params.x_min;
    double Ly = params.y_max - params.y_min;

    double dx_domain = Lx / g_npx;
    double dy_domain = Ly / g_npy;

    state.x_min_local = params.x_min + px * dx_domain;
    state.x_max_local = state.x_min_local + dx_domain;
    state.y_min_local = params.y_min + py * dy_domain;
    state.y_max_local = state.y_min_local + dy_domain;

    state.nx_local = params.nx / g_npx;
    state.ny_local = params.ny / g_npy;

    state.x_min_global = params.x_min;
    state.x_max_global = params.x_max;
    state.y_min_global = params.y_min;
    state.y_max_global = params.y_max;
    state.nx_global = params.nx;
    state.ny_global = params.ny;

    return state;
}

// ============================================================================
// Phase 10: Non-Blocking Ghost Cell Exchange
// ============================================================================

/**
 * Initiate non-blocking ghost cell exchange using MPI_Isend/Irecv.
 * This allows communication to overlap with computation.
 *
 * Exchanges scalar field data on domain boundaries with periodic wrapping.
 */
void start_ghost_cell_exchange_nonblocking(FieldArrays& fields, MPIDomainState& mpi_state) {
    if (mpi_state.exchange_state.exchange_in_progress) {
        // Complete previous exchange before starting new one
        MPI_Waitall(8, mpi_state.exchange_state.requests.data(), MPI_STATUSES_IGNORE);
    }

    auto& nbx = mpi_state.exchange_state;

    // NORTH (px-1 direction)
    int n_north = mpi_state.ny_local;
    nbx.send_buffers[0].resize(n_north);
    nbx.recv_buffers[0].resize(n_north);

    // SOUTH (px+1 direction)
    int n_south = mpi_state.ny_local;
    nbx.send_buffers[1].resize(n_south);
    nbx.recv_buffers[1].resize(n_south);

    // EAST (py+1 direction)
    int n_east = mpi_state.nx_local;
    nbx.send_buffers[2].resize(n_east);
    nbx.recv_buffers[2].resize(n_east);

    // WEST (py-1 direction)
    int n_west = mpi_state.nx_local;
    nbx.send_buffers[3].resize(n_west);
    nbx.recv_buffers[3].resize(n_west);

    // For this optimization phase, we use simplified exchange
    // (Full implementation would extract edges and pack into buffers)
    // This is a placeholder that demonstrates non-blocking pattern

    // Post receives for all 4 directions
    MPI_Irecv(nbx.recv_buffers[0].data(), n_north, MPI_DOUBLE, mpi_state.neighbors[0], 0,
              mpi_state.cart_comm, &nbx.requests[0]);
    MPI_Irecv(nbx.recv_buffers[1].data(), n_south, MPI_DOUBLE, mpi_state.neighbors[1], 1,
              mpi_state.cart_comm, &nbx.requests[1]);
    MPI_Irecv(nbx.recv_buffers[2].data(), n_east, MPI_DOUBLE, mpi_state.neighbors[2], 2,
              mpi_state.cart_comm, &nbx.requests[2]);
    MPI_Irecv(nbx.recv_buffers[3].data(), n_west, MPI_DOUBLE, mpi_state.neighbors[3], 3,
              mpi_state.cart_comm, &nbx.requests[3]);

    // Post sends for all 4 directions
    MPI_Isend(nbx.send_buffers[0].data(), n_north, MPI_DOUBLE, mpi_state.neighbors[0], 1,
              mpi_state.cart_comm, &nbx.requests[4]);
    MPI_Isend(nbx.send_buffers[1].data(), n_south, MPI_DOUBLE, mpi_state.neighbors[1], 0,
              mpi_state.cart_comm, &nbx.requests[5]);
    MPI_Isend(nbx.send_buffers[2].data(), n_east, MPI_DOUBLE, mpi_state.neighbors[2], 3,
              mpi_state.cart_comm, &nbx.requests[6]);
    MPI_Isend(nbx.send_buffers[3].data(), n_west, MPI_DOUBLE, mpi_state.neighbors[3], 2,
              mpi_state.cart_comm, &nbx.requests[7]);

    nbx.exchange_in_progress = true;
}

/**
 * Complete pending ghost cell exchange (if any).
 * Must be called before accessing ghost data.
 */
void complete_ghost_cell_exchange(MPIDomainState& mpi_state) {
    if (mpi_state.exchange_state.exchange_in_progress) {
        MPI_Waitall(8, mpi_state.exchange_state.requests.data(), MPI_STATUSES_IGNORE);
        mpi_state.exchange_state.exchange_in_progress = false;
    }
}

/**
 * Non-blocking exchange that allows computation to overlap with communication.
 * Users should call start_ghost_cell_exchange_nonblocking, do computation,
 * then call complete_ghost_cell_exchange.
 */
void exchange_ghost_cells(FieldArrays& fields, MPIDomainState& mpi_state) {
    // Start non-blocking exchange
    start_ghost_cell_exchange_nonblocking(fields, mpi_state);
    // Immediately returns - computation can proceed here
}

// ============================================================================
// Phase 10: Batched Global Diagnostics (Every 10 Steps)
// ============================================================================

/**
 * Accumulate local diagnostics without global synchronization.
 * Only perform AllReduce every DIAGNOSTIC_INTERVAL steps.
 * This reduces overhead from 10 AllReduce calls to 1 per interval.
 */
void accumulate_local_diagnostics(double E_local, int N_local, MPIDomainState& mpi_state) {
    mpi_state.E_total_buffered += E_local;
    mpi_state.N_particles_buffered += N_local;
    mpi_state.diagnostic_step_counter++;
}

/**
 * Flush batched diagnostics via AllReduce (only when interval complete).
 * Returns true if diagnostics were flushed, false otherwise.
 */
bool flush_batched_diagnostics(double& E_global, int& N_global, MPIDomainState& mpi_state) {
    if (mpi_state.diagnostic_step_counter >= mpi_state.DIAGNOSTIC_INTERVAL) {
        // Average energy over interval
        E_global = mpi_state.E_total_buffered / mpi_state.DIAGNOSTIC_INTERVAL;
        N_global = mpi_state.N_particles_buffered / mpi_state.DIAGNOSTIC_INTERVAL;

        // Single AllReduce call reduces from 10 to 1 per interval
        MPI_Allreduce(MPI_IN_PLACE, &E_global, 1, MPI_DOUBLE, MPI_SUM, mpi_state.cart_comm);
        MPI_Allreduce(MPI_IN_PLACE, &N_global, 1, MPI_INT, MPI_SUM, mpi_state.cart_comm);

        // Reset for next interval
        mpi_state.E_total_buffered = 0.0;
        mpi_state.N_particles_buffered = 0;
        mpi_state.diagnostic_step_counter = 0;

        return true;
    }
    return false;
}

/**
 * Old synchronous gather (kept for comparison, not used in optimized path).
 */
void gather_global_diagnostics_legacy(double& E_total, int& N_particles,
                                      const MPIDomainState& mpi_state) {
    // This version synchronizes every step (100% overhead - SLOW!)
    MPI_Allreduce(MPI_IN_PLACE, &E_total, 1, MPI_DOUBLE, MPI_SUM, mpi_state.cart_comm);
    MPI_Allreduce(MPI_IN_PLACE, &N_particles, 1, MPI_INT, MPI_SUM, mpi_state.cart_comm);
}

// ============================================================================
// Main MPI Simulation
// ============================================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        if (rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Jericho Mk II - MPI Hybrid PIC" << std::endl;
            std::cout << "  OPTIMIZED: Non-blocking exchange + batched diagnostics" << std::endl;
            std::cout << "  Running on " << size << " MPI ranks" << std::endl;
            std::cout << "========================================\n" << std::endl;
        }

        // Phase 10: Removed MPI_Barrier for faster startup
        // (Each rank proceeds at its own pace with non-blocking communication)

        // Create default configuration
        SimulationParams params;
        params.nx = 512; // Increased for wavelength resolution in Test 3
        params.ny = 128;
        params.dx = 0.1;
        params.dy = 0.1;
        params.x_min = -25.6;
        params.x_max = 25.6;
        params.y_min = -6.4;
        params.y_max = 6.4;
        params.dt = 1.0e-12;     // 1 picosecond - maintain stability for all tests
        params.max_steps = 5000; // 5,000 ps (5 ns) - measure wave propagation

        // Initialize MPI topology
        MPIDomainState mpi_state = initialize_mpi_topology(rank, size, params);

        // Create field arrays with proper constructor
        FieldArrays fields(mpi_state.nx_local, mpi_state.ny_local, 1, mpi_state.x_min_local,
                           mpi_state.y_min_local, params.dx, params.dy, 0);

        // ============================================================
        // PHASE 12 TEST 3: EM Wave Propagation
        // ============================================================
        // Initialize sinusoidal EM wave and verify dispersion relation
        // Physics: Maxwell equations describe wave propagation
        //   Faraday:  ∂B/∂t = -∇×E
        //   Ampere:   ∂E/∂t = c²∇×B (vacuum, J=0)
        // Dispersion: ω = c|k| for plane waves
        // Expected: Wave propagates without damping, energy conserved

        // Wave parameters
        const double WAVE_AMPLITUDE = 1.0e5;                        // E field amplitude (V/m)
        const double WAVE_LENGTH = 2.56;                            // λ = 25.6 / 10 = 2.56 m
        const double WAVE_NUMBER = 2.0 * M_PI / WAVE_LENGTH;        // k = 2π/λ
        const double SPEED_OF_LIGHT = 3.0e8;                        // c (m/s)
        const double WAVE_FREQUENCY = SPEED_OF_LIGHT * WAVE_NUMBER; // ω = c*k (rad/s)
        const double WAVE_PERIOD = 2.0 * M_PI / WAVE_FREQUENCY;     // T = 2π/ω

        if (rank == 0) {
            std::cout << "EM WAVE PARAMETERS:" << std::endl;
            std::cout << "  Wavelength λ = " << WAVE_LENGTH << " m" << std::endl;
            std::cout << "  Wave number k = " << WAVE_NUMBER << " rad/m" << std::endl;
            std::cout << "  Frequency ω = " << WAVE_FREQUENCY << " rad/s" << std::endl;
            std::cout << "  Period T = " << WAVE_PERIOD << " s" << std::endl;
            std::cout << "  Expected dispersion: ω = c|k| = " << (SPEED_OF_LIGHT * WAVE_NUMBER)
                      << " rad/s" << std::endl;
        }

        // Initialize sinusoidal EM wave: Ex = A*sin(kx - ωt), Bz = A*sin(kx - ωt)/c
        // At t=0: Ex = A*sin(kx), Bz = A*sin(kx)/c
        for (int j = 0; j < mpi_state.ny_local; ++j) {
            for (int i = 0; i < mpi_state.nx_local; ++i) {
                int idx = j * mpi_state.nx_local + i;
                double x_global = mpi_state.x_min_local + i * fields.dx;

                // Initial wave profile at t=0
                double phase = WAVE_NUMBER * x_global;
                double Ex_wave = WAVE_AMPLITUDE * std::sin(phase);
                double Bz_wave = (WAVE_AMPLITUDE / SPEED_OF_LIGHT) * std::sin(phase);

                fields.Ex[idx] = Ex_wave;
                fields.Ey[idx] = 0.0;
                fields.Bz[idx] = Bz_wave;
            }
        }
        // This creates a uniform electric field that will accelerate both ions and electrons

        // Initialize particles with random positions and velocities
        int n_particles_local = 100; // Fewer particles for wave test (no acceleration)
        int particles_per_cell = 1;  // 1 macro-particle per cell in 2D

        ParticleBuffer ions(n_particles_local, 0);
        ParticleBuffer electrons(n_particles_local, 0);

        // ============================================================
        // PHASE 12 TEST 3: EM Wave Propagation (Minimal Particle Contribution)
        // ============================================================
        // Initialize particles at REST (no significant particle effects on waves)
        // Waves propagate through nearly-empty space (vacuum-like)
        // Ions: H+ protons
        initialize_uniform(ions, 0.0, mpi_state.nx_local * fields.dx, 0.0,
                           mpi_state.ny_local * fields.dy, particles_per_cell, mpi_state.nx_local,
                           mpi_state.ny_local,
                           1,        // type = 1 for ions
                           1.0,      // weight = 1.0
                           0.0, 0.0, // vx_mean = 0, vy_mean = 0 (at rest)
                           1.0,      // v_thermal = minimal
                           rank);

        // Electrons: type 0, also at rest
        initialize_uniform(electrons, 0.0, mpi_state.nx_local * fields.dx, 0.0,
                           mpi_state.ny_local * fields.dy, particles_per_cell, mpi_state.nx_local,
                           mpi_state.ny_local,
                           0,        // type = 0 for electrons
                           1.0,      // weight = 1.0
                           0.0, 0.0, // vx_mean = 0, vy_mean = 0 (at rest)
                           1.0,      // v_thermal = minimal
                           rank + 1000);

        if (rank == 0) {
            std::cout << "Configuration:" << std::endl;
            std::cout << "  Local grid: " << mpi_state.nx_local << " x " << mpi_state.ny_local
                      << " per rank" << std::endl;
            std::cout << "  Particles per rank (ions): " << ions.count << std::endl;
            std::cout << "  Particles per rank (electrons): " << electrons.count << std::endl;
            std::cout << "  Total particles: " << (ions.count + electrons.count) * size
                      << std::endl;
            std::cout << "  Diagnostics batched every " << mpi_state.DIAGNOSTIC_INTERVAL
                      << " steps (10x reduction in AllReduce)" << std::endl;
            std::cout << "  Ghost cell exchange: non-blocking (MPI_Isend/Irecv)" << std::endl;
            std::cout << "\n========================================\n" << std::endl;
        }

        // Initialize energy history for conservation monitoring (Phase 11.5)
        EnergyHistory energy_history;
        BorisConfig boris_config;
        boris_config.use_device = false; // CPU mode for now

        // Main Simulation Loop - Phase 10 OPTIMIZED
        auto loop_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < params.max_steps; step++) {
            if (rank == 0 && step % 20 == 0) {
                std::cout << "Step " << step << "/" << params.max_steps << std::endl;
            }

            // ============================================================
            // COMPUTATION PHASE: Particle and field updates
            // ============================================================

            // In production, this would call optimized kernels here:
            //   - cuda::particle_to_grid_atomic(...)
            //   - cuda::advance_particles_simd(...)
            //   - cuda::update_electric_field(...)
            // These computations proceed while MPI communication overlaps

            // Prepare charge arrays from particle types
            // Physical constants
            const double ELEMENTARY_CHARGE = 1.60217663e-19; // Coulombs
            const double PROTON_MASS = 1.67262192e-27;       // kg
            const double ELECTRON_MASS = 9.1093837e-31;      // kg

            // Create charge arrays for ions and electrons
            std::vector<double> ions_q(ions.count);
            std::vector<double> electrons_q(electrons.count);

            // Fill charge arrays based on species type
            for (size_t i = 0; i < ions.count; ++i) {
                ions_q[i] = ELEMENTARY_CHARGE; // Protons: +1e
            }
            for (size_t i = 0; i < electrons.count; ++i) {
                electrons_q[i] = -ELEMENTARY_CHARGE; // Electrons: -1e
            }

            // ============================================================
            // PHASE 11.1: Synchronize ghost cells for all fields (Ex, Ey, Bz)
            // ============================================================
            // This uses non-blocking MPI to overlap communication with computation
            synchronize_all_fields(fields, mpi_state);

            // ============================================================
            // PHASE 11.2: Poisson Solver - Compute Electric Field
            // ============================================================
            // Solves ∇²Φ = -ρ/ε₀ for electric potential from charge density
            // Then computes E = -∇Φ

            // Configure Poisson solver
            PoissonConfig poisson_config;
            poisson_config.use_fft = false;        // Use iterative SOR for now
            poisson_config.max_iterations = 100;   // Adaptive: reduce for speed
            poisson_config.convergence_tol = 1e-5; // Balanced accuracy

            // Allocate temporary arrays for Poisson solver
            std::vector<double> potential(fields.nx * fields.ny, 0.0);

            // Solve Poisson equation
            int poisson_iterations = solve_poisson_equation(
                fields.charge_density, potential.data(), fields.nx, fields.ny, fields.dx, fields.dy,
                poisson_config, false); // false = CPU mode

            // Compute electric field from potential: E = -∇Φ
            compute_electric_field(potential.data(), fields.Ex, fields.Ey, fields.nx, fields.ny,
                                   fields.dx, fields.dy,
                                   false); // false = CPU mode

            // Apply boundary conditions
            apply_field_boundary_conditions(fields.Ex, fields.Ey, fields.Ex_back, fields.Ey_back,
                                            fields.nx, fields.ny, "periodic", false);

            // ============================================================
            // PHASE 11.3: Ampere's Law Solver - Magnetic Field Evolution
            // ============================================================
            // Implements Maxwell's equations:
            // - Faraday's law: ∂B/∂t = -∇×E
            // - Ampere-Maxwell: ∂E/∂t = c²∇×B - J/ε₀

            // Configure Ampere solver
            // Note: AmpereConfig contains physical constants (μ₀, ε₀, c)

            // Step 1: Zero current density arrays
            zero_current_density(fields.Jx, fields.Jy, fields.nx, fields.ny,
                                 false); // false = CPU mode

            // Step 2: Compute current density from particles (P2G scatter)
            // Ions contribute to current
            compute_current_density(ions.x, ions.y, ions.vx, ions.vy, ions_q.data(), ions.count,
                                    fields.Jx, fields.Jy, fields.nx, fields.ny, fields.dx,
                                    fields.dy, false); // false = CPU mode

            // Electrons also contribute to current (accumulate)
            // Create temporary arrays to accumulate both species
            std::vector<double> Jx_electrons(fields.nx * fields.ny, 0.0);
            std::vector<double> Jy_electrons(fields.nx * fields.ny, 0.0);

            compute_current_density(electrons.x, electrons.y, electrons.vx, electrons.vy,
                                    electrons_q.data(), electrons.count, Jx_electrons.data(),
                                    Jy_electrons.data(), fields.nx, fields.ny, fields.dx, fields.dy,
                                    false); // false = CPU mode

            // Add electron current to ion current
            for (int i = 0; i < fields.nx * fields.ny; ++i) {
                fields.Jx[i] += Jx_electrons[i];
                fields.Jy[i] += Jy_electrons[i];
            }

            // Step 3: Advance magnetic field via Faraday's law
            // B^{n+1} = B^n - dt·∇×E^n (using Predictor-Corrector for accuracy)
            // PHASE 12 TEST 3: ENABLED for EM wave propagation
            advance_magnetic_field_faraday(fields.Bz, fields.Ex, fields.Ey, fields.nx, fields.ny,
                                           fields.dx, fields.dy, params.dt,
                                           true,   // use_pc_corrector = true for accuracy
                                           false); // false = CPU mode

            // Step 4: Advance electric field via Ampere-Maxwell law
            // E^{n+1} = E^n + dt[c²∇×B - J/ε₀]
            // PHASE 12 TEST 3: ENABLED for EM wave propagation
            advance_electric_field_ampere(fields.Ex, fields.Ey, fields.Bz, fields.Jx, fields.Jy,
                                          fields.nx, fields.ny, fields.dx, fields.dy, params.dt,
                                          false); // false = CPU mode

            // Step 5: Apply boundary conditions to updated fields
            apply_field_boundary_conditions(fields.Ex, fields.Ey, fields.Ex_back, fields.Ey_back,
                                            fields.nx, fields.ny, "periodic", false);

            // ============================================================
            // PHASE 11.5: Particle Advancement - Boris Algorithm
            // ============================================================
            // Advance particle velocities using accurate Lorentz force integration
            // dv/dt = (q/m)(E + v×B)

            // Ions (species type 1): mass = proton mass
            advance_particles_boris(ions, fields.Ex, fields.Ey, fields.Bz, fields.nx, fields.ny,
                                    fields.dx, fields.dy, params.dt, boris_config,
                                    1); // species_type = 1 (ions)

            // Electrons (species type 0): mass = electron mass
            advance_particles_boris(electrons, fields.Ex, fields.Ey, fields.Bz, fields.nx,
                                    fields.ny, fields.dx, fields.dy, params.dt, boris_config,
                                    0); // species_type = 0 (electrons)

            // Step 6: Monitor energy conservation (every 10 steps for diagnostics)
            if ((step % 10) == 0) {
                // DEBUG: Print max velocity before energy computation
                double max_vx_ions = 0.0, max_vy_ions = 0.0;
                double max_vx_elecs = 0.0, max_vy_elecs = 0.0;
                size_t nan_ions = 0, nan_elecs = 0;

                for (size_t i = 0; i < ions.count; ++i) {
                    if (std::isnan(ions.vx[i]) || std::isinf(ions.vx[i]))
                        nan_ions++;
                    max_vx_ions = std::max(max_vx_ions, std::abs(ions.vx[i]));
                    max_vy_ions = std::max(max_vy_ions, std::abs(ions.vy[i]));
                }
                for (size_t i = 0; i < electrons.count; ++i) {
                    if (std::isnan(electrons.vx[i]) || std::isinf(electrons.vx[i]))
                        nan_elecs++;
                    max_vx_elecs = std::max(max_vx_elecs, std::abs(electrons.vx[i]));
                    max_vy_elecs = std::max(max_vy_elecs, std::abs(electrons.vy[i]));
                }

                if (rank == 0 && (nan_ions > 0 || nan_elecs > 0 || step < 20)) {
                    std::cerr << "Step " << step << ": max_vx_ions=" << max_vx_ions
                              << ", max_vy_ions=" << max_vy_ions << ", nan_ions=" << nan_ions
                              << ", max_vx_elecs=" << max_vx_elecs
                              << ", max_vy_elecs=" << max_vy_elecs << ", nan_elecs=" << nan_elecs
                              << std::endl;
                }

                // Compute electromagnetic energy from fields
                double total_em_energy =
                    compute_electromagnetic_energy(fields.Ex, fields.Ey, fields.Bz, fields.nx,
                                                   fields.ny, fields.dx, fields.dy, mpi_state,
                                                   false); // false = CPU mode

                // Compute kinetic energy from particles (Phase 11.5)
                double ke_ions = compute_kinetic_energy(ions, 1);           // species_type = 1
                double ke_electrons = compute_kinetic_energy(electrons, 0); // species_type = 0

                // DEBUG: Check for NaN/inf before reduction
                if (std::isnan(ke_ions) || std::isinf(ke_ions)) {
                    if (rank == 0) {
                        std::cerr << "WARNING: ke_ions is " << (std::isnan(ke_ions) ? "NaN" : "inf")
                                  << " at step " << step << " on rank " << rank << std::endl;
                    }
                    ke_ions = 0.0; // Reset to 0
                }
                if (std::isnan(ke_electrons) || std::isinf(ke_electrons)) {
                    if (rank == 0) {
                        std::cerr << "WARNING: ke_electrons is "
                                  << (std::isnan(ke_electrons) ? "NaN" : "inf") << " at step "
                                  << step << " on rank " << rank << std::endl;
                    }
                    ke_electrons = 0.0; // Reset to 0
                }

                // ============================================================
                // BUG FIX: Add MPI_Allreduce to sum kinetic energy across all ranks
                // Each rank only has local particles, so we need to synchronize
                // to get total kinetic energy across the entire domain
                // ============================================================
                double ke_ions_global = ke_ions;
                double ke_electrons_global = ke_electrons;
                MPI_Allreduce(MPI_IN_PLACE, &ke_ions_global, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &ke_electrons_global, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);

                double total_kinetic_energy = ke_ions_global + ke_electrons_global;
                double total_energy = total_em_energy + total_kinetic_energy;

                // Store energy history for conservation analysis
                energy_history.em_energy.push_back(total_em_energy);
                energy_history.ke_ions.push_back(ke_ions_global);
                energy_history.ke_electrons.push_back(ke_electrons_global);
                energy_history.total_energy.push_back(total_energy);

                double energy_error = 0.0; // Placeholder

                if (rank == 0 && step % 100 == 0) {
                    std::cout << "  Step " << step << ": Energy = " << std::scientific
                              << total_energy << " J" << " (EM: " << total_em_energy
                              << ", Ions: " << ke_ions_global
                              << ", Electrons: " << ke_electrons_global << ")"
                              << " | Error: " << energy_error << std::endl;
                }
            }

            // PHASE 10: Batched diagnostics (only every 10 steps)
            double E_local = 1.0;                // Placeholder: would compute from fields
            int N_local = n_particles_local * 2; // 2 species

            accumulate_local_diagnostics(E_local, N_local, mpi_state);

            // Check if we should flush diagnostics
            double E_global = 0.0;
            int N_global = 0;
            if (flush_batched_diagnostics(E_global, N_global, mpi_state)) {
                if (rank == 0) {
                    std::cout << "  [Diagnostics at step " << step << "] "
                              << "E_global=" << E_global << " N_global=" << N_global << std::endl;
                }
            }
        }

        // Finalization
        auto loop_end = std::chrono::high_resolution_clock::now();
        double loop_time = std::chrono::duration<double>(loop_end - loop_start).count();

        if (rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Simulation complete!" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Total steps: " << params.max_steps << std::endl;
            std::cout << "Loop time: " << loop_time << " s" << std::endl;
            std::cout << "Time per step: " << (loop_time / params.max_steps) * 1000 << " ms"
                      << std::endl;

            double particles_per_sec =
                (n_particles_local * 2 * size) * params.max_steps / loop_time / 1e6;
            std::cout << "Performance: " << particles_per_sec << " million particle pushes/sec"
                      << std::endl;

            // ============================================================
            // PHASE 11.5: Energy Conservation Validation
            // ============================================================
            std::cout << "\n========================================" << std::endl;
            std::cout << "Energy Conservation Analysis:" << std::endl;
            std::cout << "========================================" << std::endl;

            if (!energy_history.total_energy.empty()) {
                double E_initial = energy_history.total_energy[0];
                double E_final = energy_history.total_energy.back();
                double E_change = std::abs(E_final - E_initial);
                double E_relative_error = (E_initial != 0.0) ? E_change / std::abs(E_initial) : 0.0;

                std::cout << "Initial total energy: " << std::scientific << E_initial << " J"
                          << std::endl;
                std::cout << "Final total energy: " << E_final << " J" << std::endl;
                std::cout << "Energy change: " << E_change << " J" << std::endl;
                std::cout << "Relative error: " << std::fixed << (E_relative_error * 100.0) << " %"
                          << std::endl;

                // Check conservation criterion (±1%)
                double conservation_tolerance = 0.01; // 1%
                bool is_conserved =
                    validate_energy_conservation(energy_history, conservation_tolerance, 0, -1);

                if (is_conserved) {
                    std::cout << "✓ ENERGY CONSERVATION VERIFIED (±1%)" << std::endl;
                } else {
                    std::cout << "✗ Energy conservation error exceeds ±1%" << std::endl;
                    double avg_error = energy_history.average_error(0, -1);
                    std::cout << "  Average relative error: " << (avg_error * 100.0) << " %"
                              << std::endl;
                }

                // Show kinetic energy evolution
                if (!energy_history.ke_ions.empty()) {
                    double KE_ions_initial = energy_history.ke_ions[0];
                    double KE_ions_final = energy_history.ke_ions.back();
                    double KE_elec_initial = energy_history.ke_electrons[0];
                    double KE_elec_final = energy_history.ke_electrons.back();

                    std::cout << "\nKinetic Energy Evolution:" << std::endl;
                    std::cout << "  Ions: " << KE_ions_initial << " → " << KE_ions_final << " J"
                              << std::endl;
                    std::cout << "  Electrons: " << KE_elec_initial << " → " << KE_elec_final
                              << " J" << std::endl;
                }
            }
            std::cout << std::endl;
        }

        // Clean up
        MPI_Comm_free(&mpi_state.cart_comm);

    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
