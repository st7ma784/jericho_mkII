/**
 * @file main.cpp
 * @brief Main simulation driver for Jericho Mk II
 * @author Jericho Mk II Development Team
 *
 * This is the main entry point for the 2.5D hybrid PIC-MHD simulation.
 * It orchestrates:
 * - MPI initialization and domain decomposition
 * - Configuration loading
 * - Particle and field initialization
 * - Main timestepping loop
 * - I/O and diagnostics
 */

#include "config.h"
#include "field_arrays.h"
#include "io_manager.h"
#include "kernels_cpu.h"
#include "kernels_cpu_optimized.h"
#include "mpi_manager.h"
#include "particle_buffer.h"
#include "platform.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace jericho;

// CUDA kernel declarations (from cuda/*.cu files)
namespace jericho {
namespace cuda {
// Particle kernels (from cuda/particles.cu)
void advance_particles(ParticleBuffer& particles, const double* Ex, const double* Ey,
                       const double* Bz, int nx, int ny, double dx, double dy, double x_min,
                       double y_min, double dt, const double* q_over_m_by_type);

void particle_to_grid(const ParticleBuffer& particles, double* charge_density, double* current_x,
                      double* current_y, int nx, int ny, double dx, double dy, double x_min,
                      double y_min, const double* q_by_type);

// Field solver kernels (from cuda/fields.cu)
void advance_magnetic_field(FieldArrays& fields, double dt);
void compute_flow_velocity(FieldArrays& fields);
void solve_electric_field(FieldArrays& fields, bool use_hall);
void apply_cam_correction(FieldArrays& fields, double dt);
void clamp_electric_field(FieldArrays& fields);

} // namespace cuda
} // namespace jericho

// Physical constants (using the constants namespace from field_arrays.h)
using jericho::constants::c;
using jericho::constants::e;
using jericho::constants::epsilon_0;
using jericho::constants::m_p;
using jericho::constants::mu_0;

// Additional constant not in field_arrays.h
constexpr double m_e = 9.1093837015e-31; // Electron mass [kg]

// =============================================================================
// Simulation parameters
// =============================================================================

struct SimulationParams {
    // Domain (reduced for CPU testing)
    double Lx = 10.0;   // Domain size x [ion inertial lengths]
    double Ly = 10.0;   // Domain size y
    int nx_global = 32; // Grid points x
    int ny_global = 32; // Grid points y
    int nghost = 2;     // Ghost cell layers

    // MPI decomposition
    int npx = 1; // MPI ranks in x
    int npy = 1; // MPI ranks in y

    // Time
    double t_max = 50.0;  // Max time [ion gyro periods]
    double dt = 0.02;     // Timestep
    int max_steps = 2500; // Max iterations

    // Plasma
    double n0 = 1.0e20; // Background density [m^-3]
    double Ti = 100.0;  // Ion temperature [eV]
    double Te = 10.0;   // Electron temperature [eV]
    double B0 = 0.1;    // Magnetic field [T]

    // Particles
    int particles_per_cell = 1; // Minimal for CPU testing/debugging

    // I/O
    int field_cadence = 10;
    int particle_cadence = 100;
    int diag_cadence = 1;
    std::string output_dir = "output";
};

// =============================================================================
// Helper functions
// =============================================================================

void initialize_harris_sheet(SimulationParams& params, FieldArrays& fields, ParticleBuffer& ions,
                             ParticleBuffer& electrons, double x_min, double y_min) {

    // Harris sheet parameters
    double L = 5.0; // Current sheet thickness [ion inertial lengths]

    // Normalize to physical units
    double di =
        c / sqrt(params.n0 * constants::e * constants::e / (constants::epsilon_0 * constants::m_p));
    double L_phys = L * di;

    // Initialize magnetic field: Bz = B0 * tanh(y/L)
    fields.initialize_harris_sheet(params.B0, L_phys);

    std::cout << "Initialized Harris sheet with L = " << L << " di" << std::endl;
}

void compute_diagnostics(const FieldArrays& fields, const ParticleBuffer& ions,
                         const ParticleBuffer& electrons, std::map<std::string, double>& diag) {

    // Field energy
    double field_energy = 0.0;
    size_t n_points = fields.get_total_points();

    for (size_t i = 0; i < n_points; i++) {
        double Ex = fields.Ex[i];
        double Ey = fields.Ey[i];
        double Bz = fields.Bz[i];

        field_energy += (Ex * Ex + Ey * Ey) * constants::epsilon_0;
        field_energy += (Bz * Bz) / constants::mu_0;
    }
    field_energy *= fields.dx * fields.dy;

    // Particle energy
    double particle_energy = 0.0;
    for (size_t i = 0; i < ions.count; i++) {
        if (ions.active[i]) {
            double v2 = ions.vx[i] * ions.vx[i] + ions.vy[i] * ions.vy[i];
            particle_energy += 0.5 * constants::m_p * v2 * ions.weight[i];
        }
    }
    for (size_t i = 0; i < electrons.count; i++) {
        if (electrons.active[i]) {
            double v2 = electrons.vx[i] * electrons.vx[i] + electrons.vy[i] * electrons.vy[i];
            particle_energy += 0.5 * m_e * v2 * electrons.weight[i];
        }
    }

    diag["field_energy"] = field_energy;
    diag["particle_energy"] = particle_energy;
    diag["total_energy"] = field_energy + particle_energy;
    diag["num_ions"] = static_cast<double>(ions.count);
    diag["num_electrons"] = static_cast<double>(electrons.count);
    diag["num_particles"] = static_cast<double>(ions.count + electrons.count);
}

// =============================================================================
// Main simulation loop
// =============================================================================

int main(int argc, char** argv) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // 1. Initialize MPI and load configuration
    // =========================================================================

    // Note: MPI will be initialized by MPIManager constructor
    int rank = 0, size = 1;

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Jericho Mk II - Hybrid PIC-MHD Code\n";
        std::cout << "  CPU Build (Testing SoA architecture)\n";
        std::cout << "========================================\n";
        std::cout << "MPI ranks: " << size << std::endl;
#ifdef USE_CPU
        std::cout << "Mode: CPU (no CUDA)" << std::endl;
#else
        std::cout << "Mode: GPU (CUDA enabled)" << std::endl;
#endif
        std::cout << "\n";
    }

    SimulationParams params;

    // Parse configuration file from command line
    // Usage: jericho_mkII config.toml [max_steps]
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "ERROR: Configuration file required\n";
            std::cerr << "Usage: " << argv[0] << " <config.toml> [max_steps]\n";
        }
        return 1;
    }

    std::string config_file = argv[1];

    // Load and parse configuration
    try {
        Config config;
        config.load(config_file);

        // Copy configuration to SimulationParams
        params.Lx = config.x_max - config.x_min;
        params.Ly = config.y_max - config.y_min;
        params.nx_global = config.nx_global;
        params.ny_global = config.ny_global;
        params.n0 = 1.0e20; // Default density if not in config
        params.Ti = config.species.empty() ? 100.0 : config.species[0].temperature;
        params.Te = config.species.size() > 1 ? config.species[1].temperature : 10.0;
        params.B0 = config.B0;
        params.max_steps = config.n_steps;
        params.dt = config.dt;
        params.t_max = config.n_steps * config.dt;
        params.npx = config.npx;
        params.npy = config.npy;
        params.output_dir = config.output_dir;
        params.field_cadence = config.field_cadence;
        params.particle_cadence = config.particle_cadence;
        params.particles_per_cell =
            config.species.empty() ? 1 : config.species[0].particles_per_cell;

        if (rank == 0) {
            std::cout << "Configuration loaded from: " << config_file << "\n";
            config.print();
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "ERROR loading configuration: " << e.what() << "\n";
        }
        return 1;
    }

    // Allow command-line override of max_steps
    if (argc > 2) {
        params.max_steps = std::atoi(argv[2]);
    }

    // Determine MPI decomposition (from config)
    // Default to single rank per dimension
    if (params.npx <= 0)
        params.npx = 1;
    if (params.npy <= 0)
        params.npy = 1;

    // =========================================================================
    // 2. Initialize MPI manager and domain decomposition
    // =========================================================================

    // Normalize to physical units using ion inertial length
    // di = c / sqrt(n0 * e^2 / (epsilon_0 * m_p))
    double di = constants::c / sqrt(params.n0 * constants::e * constants::e /
                                    (constants::epsilon_0 * constants::m_p));

    double Lx_phys = params.Lx * di;
    double Ly_phys = params.Ly * di;

    if (rank == 0) {
        std::cout << "Physical domain size: " << Lx_phys / 1e6 << " x " << Ly_phys / 1e6 << " Mm\n";
        std::cout << "Ion inertial length (di): " << di / 1e3 << " km\n";
    }

    MPIManager mpi(&argc, &argv, params.npx, params.npy, params.nx_global, params.ny_global, 0.0,
                   Lx_phys, 0.0, Ly_phys,
                   false); // No CUDA-aware MPI in CPU mode

    // Get rank and size from MPI manager
    rank = mpi.rank;
    size = mpi.size;

    if (rank == 0) {
        std::cout << "Domain decomposition:\n";
        std::cout << "  Global: " << params.nx_global << " x " << params.ny_global << "\n";
        std::cout << "  Local:  " << mpi.nx_local << " x " << mpi.ny_local << "\n";
        std::cout << "  MPI grid: " << params.npx << " x " << params.npy << "\n";
        std::cout << "\n";
    }

    // =========================================================================
    // 3. Initialize field arrays
    // =========================================================================

    double dx = Lx_phys / params.nx_global;
    double dy = Ly_phys / params.ny_global;

    FieldArrays fields(mpi.nx_local, mpi.ny_local, params.nghost, dx, dy, mpi.x_min_local,
                       mpi.y_min_local,
                       0); // Device ID

    if (rank == 0) {
        std::cout << "Grid spacing: dx = " << dx / di << " di, dy = " << dy / di << " di\n";
        std::cout << "Field memory: " << fields.get_memory_bytes() / 1e6 << " MB\n";
        std::cout << "\n";
    }

    // =========================================================================
    // 4. Initialize particle buffers
    // =========================================================================

    size_t n_particles_local = mpi.nx_local * mpi.ny_local * params.particles_per_cell;

    ParticleBuffer ions(n_particles_local * 2, 0); // 2x capacity for dynamic particles
    ParticleBuffer electrons(n_particles_local * 2, 0);

    if (rank == 0) {
        std::cout << "Particles per rank: " << n_particles_local << " (ions + electrons)\n";
        size_t total_mem = ions.get_memory_bytes() + electrons.get_memory_bytes();
        std::cout << "Particle memory: " << total_mem / 1e6 << " MB\n";
        std::cout << "\n";
    }

    // =========================================================================
    // 5. Initialize fields and particles for Harris sheet
    // =========================================================================

    initialize_harris_sheet(params, fields, ions, electrons, mpi.x_min_local, mpi.y_min_local);

    // Initialize particles uniformly
    if (rank == 0)
        std::cout << "Initializing particles..." << std::endl;

    double v_ti = sqrt(2.0 * params.Ti * constants::e / constants::m_p);
    double v_te = sqrt(2.0 * params.Te * constants::e / m_e);
    double weight = params.n0 * dx * dy / params.particles_per_cell;

    if (rank == 0)
        std::cout << "  v_ti = " << v_ti << ", v_te = " << v_te << ", weight = " << weight
                  << std::endl;
    if (rank == 0)
        std::cout << "  Initializing ions..." << std::endl;

    initialize_uniform(ions, mpi.x_min_local, mpi.x_max_local, mpi.y_min_local, mpi.y_max_local,
                       params.particles_per_cell, mpi.nx_local, mpi.ny_local, 0, weight, 0.0, 0.0,
                       v_ti, 42 + rank);

    if (rank == 0)
        std::cout << "  Ions initialized: " << ions.count << " particles" << std::endl;
    if (rank == 0)
        std::cout << "  Initializing electrons..." << std::endl;

    initialize_uniform(electrons, mpi.x_min_local, mpi.x_max_local, mpi.y_min_local,
                       mpi.y_max_local, params.particles_per_cell, mpi.nx_local, mpi.ny_local, 1,
                       weight, 0.0, 0.0, v_te, 1337 + rank);

    if (rank == 0) {
        std::cout << "Initialized " << ions.count << " ions\n";
        std::cout << "Initialized " << electrons.count << " electrons\n";
        std::cout << "\n";
    }

    // =========================================================================
    // 6. Initialize I/O manager
    // =========================================================================

    IOManager io(params.output_dir, &mpi);
    io.field_cadence = params.field_cadence;
    io.particle_cadence = params.particle_cadence;

    // =========================================================================
    // 7. Prepare species arrays
    // =========================================================================

    std::vector<double> q_by_type = {constants::e, -constants::e}; // ions, electrons
    std::vector<double> m_by_type = {constants::m_p, m_e};
    std::vector<double> qm_by_type(2);

    for (size_t i = 0; i < 2; i++) {
        qm_by_type[i] = q_by_type[i] / m_by_type[i];
    }

    // =========================================================================
    // 8. Main timestepping loop
    // =========================================================================

    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "Starting main loop\n";
        std::cout << "  Max steps: " << params.max_steps << "\n";
        std::cout << "  dt: " << params.dt << " (ion gyro periods)\n";
        std::cout << "========================================\n\n";
    }

    double t = 0.0;
    double omega_ci = constants::e * params.B0 / constants::m_p;
    double dt_phys = params.dt / omega_ci; // Convert to physical time

    auto loop_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < params.max_steps; step++) {

        if (rank == 0 && step % 100 == 0) {
            std::cout << "\n--- Starting step " << step << " ---" << std::endl;
        }

        // ---------------------------------------------------------------------
        // Step 1: Particle push (Boris algorithm)
        // ---------------------------------------------------------------------

        if (rank == 0 && step == 1)
            std::cout << "  Advancing ions..." << std::endl;
        cuda::advance_particles_cpu(ions.x, ions.y, ions.vx, ions.vy, fields.Ex, fields.Ey,
                                    fields.Bz, fields.nx, fields.ny, fields.dx, fields.dy,
                                    fields.x_min, fields.y_min, dt_phys, qm_by_type.data(),
                                    ions.count, 0); // species_id=0 for ions

        if (rank == 0 && step == 1)
            std::cout << "  Advancing electrons..." << std::endl;
        cuda::advance_particles_cpu(
            electrons.x, electrons.y, electrons.vx, electrons.vy, fields.Ex, fields.Ey, fields.Bz,
            fields.nx, fields.ny, fields.dx, fields.dy, fields.x_min, fields.y_min, dt_phys,
            qm_by_type.data(), electrons.count, 1); // species_id=1 for electrons

        // ---------------------------------------------------------------------
        // Step 2: Particle-to-grid (deposit charge and current)
        // ---------------------------------------------------------------------

        if (rank == 0 && step == 1)
            std::cout << "  Particle-to-grid..." << std::endl;
        fields.zero_particle_quantities();

        // Deposit ions
        cuda::particle_to_grid_cpu(ions.x, ions.y, ions.vx, ions.vy, fields.charge_density,
                                   fields.Jx, fields.Jy, fields.nx, fields.ny, fields.dx, fields.dy,
                                   fields.x_min, fields.y_min, constants::e,
                                   ions.weight[0], // charge and weight
                                   ions.count);

        // Deposit electrons
        cuda::particle_to_grid_cpu(electrons.x, electrons.y, electrons.vx, electrons.vy,
                                   fields.charge_density, fields.Jx, fields.Jy, fields.nx,
                                   fields.ny, fields.dx, fields.dy, fields.x_min, fields.y_min,
                                   -constants::e, electrons.weight[0], // charge and weight
                                   electrons.count);

        // ---------------------------------------------------------------------
        // Step 3: Field solve
        // ---------------------------------------------------------------------

        // Compute ion flow velocity
        // STUB: compute_flow_velocity disabled

        // Solve electric field from Ohm's law
        // STUB: solve_electric_field disabled  // Use Hall term

        // Apply CAM correction for stability
        // STUB: apply_cam_correction disabled

        // Clamp E in low-density regions
        // STUB: clamp_electric_field disabled

        // Advance magnetic field (Faraday's law)
        // STUB: advance_magnetic_field disabled

        // ---------------------------------------------------------------------
        // Step 4: Diagnostics and I/O
        // -----  ------------------------------------------

        if (params.diag_cadence > 0 && step % params.diag_cadence == 0) {
            std::map<std::string, double> diag;
            compute_diagnostics(fields, ions, electrons, diag);

            // Reduce across MPI ranks
            double total_energy = mpi.global_sum(diag["total_energy"]);
            double total_particles = mpi.global_sum(diag["num_particles"]);

            if (rank == 0) {
                diag["total_energy"] = total_energy;
                diag["num_particles"] = total_particles;

                io.write_diagnostics(step, t, diag);

                if (step % 100 == 0) {
                    std::cout << "Step " << step << " / " << params.max_steps << "  |  t = " << t
                              << "  |  E_total = " << total_energy
                              << "  |  N_part = " << static_cast<size_t>(total_particles)
                              << std::endl;
                }
            }
        }

        if (params.field_cadence > 0 && step % params.field_cadence == 0) {
            io.write_fields(step, t, fields);
        }

        if (params.particle_cadence > 0 && step % params.particle_cadence == 0) {
            io.write_particles(step, t, ions);
            io.write_particles(step, t, electrons);
        }

        t += params.dt;
    }

    auto loop_end = std::chrono::high_resolution_clock::now();
    auto loop_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);

    // =========================================================================
    // 9. Final statistics
    // =========================================================================

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Simulation complete!\n";
        std::cout << "========================================\n";
        std::cout << "Total steps: " << params.max_steps << "\n";
        std::cout << "Loop time: " << loop_duration.count() / 1000.0 << " s\n";
        std::cout << "Time per step: " << loop_duration.count() / params.max_steps << " ms\n";
        std::cout << "\n";

        // Calculate particles per second
        double total_particles = (ions.count + electrons.count) * size;
        double particle_pushes = total_particles * params.max_steps;
        double mpps = particle_pushes / (loop_duration.count() * 1000.0); // Millions per second

        std::cout << "Performance:\n";
        std::cout << "  " << mpps << " million particle pushes/sec\n";
        std::cout << "\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    if (rank == 0) {
        std::cout << "Total runtime: " << total_duration.count() << " seconds\n";
        std::cout << "\n";
    }

    // MPI_Finalize is called by MPIManager destructor
    return 0;
}
