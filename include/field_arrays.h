/**
 * @file field_arrays.h
 * @brief GPU-resident electromagnetic field arrays for 2.5D hybrid PIC-MHD
 * @author Jericho Mk II Development Team
 *
 * This file defines data structures for electromagnetic fields on the GPU.
 * Fields are stored as 2D arrays in row-major format for optimal GPU access.
 */

#pragma once

#include "platform.h"

#include <cstddef>

namespace jericho {

/**
 * @brief 2.5D electromagnetic field arrays on GPU
 *
 * @details Stores electric and magnetic fields on a 2D grid.
 * The "2.5D" geometry means:
 * - Spatial variation in x,y only
 * - All three vector components present (Ex, Ey, Bz)
 * - Assumes ∂/∂z = 0
 *
 * Memory layout:
 * - Row-major: field[iy * nx + ix]
 * - Ghost cells included for MPI boundaries
 * - All arrays allocated on GPU device memory
 */
class FieldArrays {
  public:
    // Grid dimensions
    int nx;       ///< Grid points in x (including ghost cells)
    int ny;       ///< Grid points in y (including ghost cells)
    int nx_local; ///< Local grid points in x (excluding ghosts)
    int ny_local; ///< Local grid points in y (excluding ghosts)
    int nghost;   ///< Number of ghost cell layers (typically 2)

    // Spatial parameters
    double dx;    ///< Grid spacing in x [m]
    double dy;    ///< Grid spacing in y [m]
    double x_min; ///< Domain minimum x [m]
    double y_min; ///< Domain minimum y [m]

    // Electromagnetic fields (device pointers)
    // Electric field
    double* Ex; ///< Electric field x-component [V/m]
    double* Ey; ///< Electric field y-component [V/m]

    // Magnetic field
    double* Bz; ///< Magnetic field z-component [T]

    // Plasma quantities (from particle-to-grid)
    double* charge_density; ///< Charge density [C/m³]
    double* Jx;             ///< Current density x [A/m²]
    double* Jy;             ///< Current density y [A/m²]
    double* Ux;             ///< Ion flow velocity x [m/s]
    double* Uy;             ///< Ion flow velocity y [m/s]

    // CAM (Current Advance Method) coefficients
    double* Lambda;  ///< CAM Lambda coefficient
    double* Gamma_x; ///< CAM Gamma_x coefficient
    double* Gamma_y; ///< CAM Gamma_y coefficient

    // Background/initial fields (constant)
    double* Ex_back; ///< Background electric field x
    double* Ey_back; ///< Background electric field y
    double* Bz_back; ///< Background magnetic field z

    // Temporary arrays for field solver
    double* tmp1; ///< Scratch array 1
    double* tmp2; ///< Scratch array 2
    double* tmp3; ///< Scratch array 3

    int device_id; ///< CUDA device where arrays reside

    // ==========================================================================
    // Constructors / Destructors
    // ==========================================================================

    /**
     * @brief Construct field arrays on GPU
     *
     * @param nx_local Local grid points in x
     * @param ny_local Local grid points in y
     * @param nghost Number of ghost cell layers
     * @param dx Grid spacing in x
     * @param dy Grid spacing in y
     * @param x_min Domain minimum x
     * @param y_min Domain minimum y
     * @param device_id CUDA device ID
     */
    FieldArrays(int nx_local, int ny_local, int nghost, double dx, double dy, double x_min,
                double y_min, int device_id = 0);

    /**
     * @brief Destructor - free GPU memory
     */
    ~FieldArrays();

    // Prevent copying, allow moving
    FieldArrays(const FieldArrays&) = delete;
    FieldArrays& operator=(const FieldArrays&) = delete;
    FieldArrays(FieldArrays&& other) noexcept;
    FieldArrays& operator=(FieldArrays&& other) noexcept;

    // ==========================================================================
    // Memory management
    // ==========================================================================

    /**
     * @brief Allocate all field arrays on GPU
     */
    void allocate();

    /**
     * @brief Free all GPU memory
     */
    void destroy();

    /**
     * @brief Zero all field arrays
     */
    void zero_fields();

    /**
     * @brief Zero only particle-derived quantities (charge, current)
     */
    void zero_particle_quantities();

    // ==========================================================================
    // Data access
    // ==========================================================================

    /**
     * @brief Get total number of grid points (including ghosts)
     */
    size_t get_total_points() const {
        return nx * ny;
    }

    /**
     * @brief Get memory usage in bytes
     */
    size_t get_memory_bytes() const;

    /**
     * @brief Copy field data to host
     *
     * @param h_Ex,h_Ey,h_Bz Host arrays (pre-allocated)
     */
    void copy_to_host(double* h_Ex, double* h_Ey, double* h_Bz) const;

    /**
     * @brief Copy field data from host
     *
     * @param h_Ex,h_Ey,h_Bz Host arrays
     */
    void copy_from_host(const double* h_Ex, const double* h_Ey, const double* h_Bz);

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /**
     * @brief Initialize with Harris current sheet magnetic field
     *
     * @param B0 Asymptotic field strength [T]
     * @param L Sheet thickness [grid units]
     */
    void initialize_harris_sheet(double B0, double L);

    /**
     * @brief Initialize with uniform magnetic field
     *
     * @param Bz0 Uniform field strength [T]
     */
    void initialize_uniform_field(double Bz0);

    /**
     * @brief Set background fields
     *
     * @param Ex0,Ey0,Bz0 Background field values
     */
    void set_background_fields(double Ex0, double Ey0, double Bz0);
};

// ==============================================================================
// Physical constants (for field solvers)
// ==============================================================================

namespace constants {
constexpr double mu_0 = 1.25663706212e-6;      ///< Permeability of free space [H/m]
constexpr double epsilon_0 = 8.8541878128e-12; ///< Permittivity of free space [F/m]
constexpr double c = 299792458.0;              ///< Speed of light [m/s]
constexpr double e = 1.602176634e-19;          ///< Elementary charge [C]
constexpr double m_p = 1.67262192369e-27;      ///< Proton mass [kg]
constexpr double k_B = 1.380649e-23;           ///< Boltzmann constant [J/K]
} // namespace constants

} // namespace jericho
