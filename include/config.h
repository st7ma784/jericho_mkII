/**
 * @file config.h
 * @brief Configuration parser for TOML-based simulation parameters
 * @author Jericho Mk II Development Team
 *
 * This file provides a configuration system for parsing TOML files
 * containing simulation parameters, grid setup, particle species,
 * boundary conditions, and I/O settings.
 */

#pragma once

#include <string>
#include <vector>
#include <map>

namespace jericho {

/**
 * @brief Particle species configuration
 */
struct SpeciesConfig {
    std::string name;           ///< Species name (e.g., "proton", "electron")
    double charge;              ///< Charge [C]
    double mass;                ///< Mass [kg]
    double density;             ///< Number density [m^-3]
    double temperature;         ///< Temperature [eV]
    double drift_vx;            ///< Drift velocity x [m/s]
    double drift_vy;            ///< Drift velocity y [m/s]
    size_t particles_per_cell;  ///< Macroparticles per cell
};

/**
 * @brief Boundary condition configuration
 */
struct BoundaryConfig {
    std::string type;           ///< "periodic", "outflow", "inflow", "reflecting"

    // Inflow parameters
    double inflow_density;      ///< Inflow density [m^-3]
    double inflow_temperature;  ///< Inflow temperature [eV]
    double inflow_vx;           ///< Inflow velocity x [m/s]
    double inflow_vy;           ///< Inflow velocity y [m/s]
};

/**
 * @brief Main simulation configuration
 *
 * @details Reads TOML configuration file with structure:
 * ```toml
 * [simulation]
 * dt = 1e-9
 * n_steps = 10000
 *
 * [grid]
 * nx = 512
 * ny = 512
 * x_min = 0.0
 * x_max = 1.0
 * y_min = -0.5
 * y_max = 0.5
 *
 * [mpi]
 * npx = 2
 * npy = 2
 * cuda_aware = true
 *
 * [[species]]
 * name = "proton"
 * charge = 1.602e-19
 * mass = 1.673e-27
 * ...
 *
 * [boundaries]
 * left = "periodic"
 * right = "periodic"
 * bottom = "outflow"
 * top = "inflow"
 * ...
 *
 * [fields]
 * magnetic_field_type = "harris_sheet"
 * B0 = 1e-8
 * L = 10.0
 *
 * [output]
 * output_dir = "./output"
 * field_cadence = 10
 * particle_cadence = 100
 * ...
 * ```
 */
class Config {
public:
    // =========================================================================
    // Simulation parameters
    // =========================================================================

    double dt;                  ///< Timestep [s]
    int n_steps;                ///< Number of timesteps
    bool use_cam;               ///< Use Current Advance Method
    double cam_alpha;           ///< CAM mixing parameter (0-1)

    // =========================================================================
    // Grid parameters
    // =========================================================================

    int nx_global;              ///< Global grid points in x
    int ny_global;              ///< Global grid points in y
    double x_min, x_max;        ///< Physical domain bounds x [m]
    double y_min, y_max;        ///< Physical domain bounds y [m]
    int nghost;                 ///< Number of ghost cell layers

    // =========================================================================
    // MPI parameters
    // =========================================================================

    int npx;                    ///< MPI processes in x
    int npy;                    ///< MPI processes in y
    bool cuda_aware_mpi;        ///< Use CUDA-aware MPI

    // =========================================================================
    // Particle species
    // =========================================================================

    std::vector<SpeciesConfig> species;  ///< List of particle species

    // =========================================================================
    // Boundary conditions
    // =========================================================================

    BoundaryConfig bc_left;
    BoundaryConfig bc_right;
    BoundaryConfig bc_bottom;
    BoundaryConfig bc_top;

    // =========================================================================
    // Initial fields
    // =========================================================================

    std::string magnetic_field_type;  ///< "uniform", "harris_sheet"
    double B0;                        ///< Magnetic field strength [T]
    double L;                         ///< Harris sheet thickness [grid units]
    double Ex0, Ey0, Bz0;             ///< Background fields

    // =========================================================================
    // Output parameters
    // =========================================================================

    std::string output_dir;
    int field_cadence;
    int particle_cadence;
    int checkpoint_cadence;
    bool compress_output;
    bool output_particles;
    bool output_Ex, output_Ey, output_Bz;
    bool output_charge, output_current;
    bool output_flow_velocity;

    // =========================================================================
    // Methods
    // =========================================================================

    /**
     * @brief Default constructor with sensible defaults
     */
    Config();

    /**
     * @brief Load configuration from TOML file
     *
     * @param filename Path to TOML configuration file
     */
    void load(const std::string& filename);

    /**
     * @brief Print configuration summary
     */
    void print() const;

    /**
     * @brief Validate configuration parameters
     *
     * @throw std::runtime_error if invalid parameters detected
     */
    void validate() const;

    /**
     * @brief Get grid spacing
     */
    double get_dx() const { return (x_max - x_min) / nx_global; }
    double get_dy() const { return (y_max - y_min) / ny_global; }

    /**
     * @brief Get total number of MPI processes
     */
    int get_mpi_size() const { return npx * npy; }
};

} // namespace jericho
