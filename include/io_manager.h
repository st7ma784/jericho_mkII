/**
 * @file io_manager.h
 * @brief HDF5-based I/O manager for fields and particles
 * @author Jericho Mk II Development Team
 *
 * This file implements parallel I/O using HDF5 for:
 * - Field output (Ex, Ey, Bz, charge, current, etc.)
 * - Particle output (positions, velocities, species)
 * - Checkpointing and restart
 * - Metadata and diagnostics
 */

#pragma once

#include "field_arrays.h"
#include "particle_buffer.h"
#include "mpi_manager.h"
#include <hdf5.h>
#include <string>
#include <vector>
#include <map>

namespace jericho {

/**
 * @brief HDF5-based parallel I/O manager
 *
 * @details Handles all file I/O operations:
 * - Parallel HDF5 with MPI-IO
 * - Field output (2D arrays)
 * - Particle output (1D arrays)
 * - Checkpoint/restart
 * - Metadata (simulation parameters, timestamps)
 *
 * File structure:
 * ```
 * output/
 * ├── fields_step_0000.h5
 * ├── fields_step_0100.h5
 * ├── particles_step_0000.h5
 * ├── particles_step_0100.h5
 * └── checkpoint_step_1000.h5
 * ```
 *
 * HDF5 file structure:
 * ```
 * fields_step_0000.h5:
 *   /Ex              - 2D dataset [ny_global, nx_global]
 *   /Ey
 *   /Bz
 *   /charge_density
 *   /current_x
 *   /current_y
 *   /metadata        - Attributes (time, step, etc.)
 * ```
 */
class IOManager {
public:
    // Output configuration
    std::string output_dir;         ///< Output directory path
    std::string checkpoint_dir;     ///< Checkpoint directory path
    int field_cadence;              ///< Output fields every N steps
    int particle_cadence;           ///< Output particles every N steps
    int checkpoint_cadence;         ///< Checkpoint every N steps
    bool compress_output;           ///< Enable HDF5 compression
    bool output_particles;          ///< Enable particle output

    // Field selection (which fields to output)
    bool output_Ex, output_Ey, output_Bz;
    bool output_charge, output_current;
    bool output_flow_velocity;

    // MPI reference (for parallel I/O)
    const MPIManager* mpi;

    // ==========================================================================
    // Constructors
    // ==========================================================================

    /**
     * @brief Construct I/O manager
     *
     * @param output_dir Output directory
     * @param mpi MPI manager reference
     */
    IOManager(const std::string& output_dir, const MPIManager* mpi);

    /**
     * @brief Destructor
     */
    ~IOManager();

    // ==========================================================================
    // Field output
    // ==========================================================================

    /**
     * @brief Write all fields to HDF5 file
     *
     * @param step Timestep number
     * @param time Physical time [s]
     * @param fields Field arrays (on device)
     *
     * @note This is a collective operation - all ranks must call it
     */
    void write_fields(int step, double time, const FieldArrays& fields);

    /**
     * @brief Write single field to HDF5 file
     *
     * @param filename Output filename
     * @param dataset_name HDF5 dataset name (e.g., "Ex")
     * @param field Field array on device
     * @param nx_local,ny_local Local grid dimensions
     * @param nx_global,ny_global Global grid dimensions
     * @param offset_x,offset_y Offset in global array
     */
    void write_field(const std::string& filename,
                    const std::string& dataset_name,
                    const double* field,
                    int nx_local, int ny_local,
                    int nx_global, int ny_global,
                    int offset_x, int offset_y);

    // ==========================================================================
    // Particle output
    // ==========================================================================

    /**
     * @brief Write particles to HDF5 file
     *
     * @param step Timestep number
     * @param time Physical time [s]
     * @param particles Particle buffer (on device)
     *
     * @note Uses parallel I/O - each rank writes its particles to different
     * regions of the file
     */
    void write_particles(int step, double time, const ParticleBuffer& particles);

    // ==========================================================================
    // Checkpointing
    // ==========================================================================

    /**
     * @brief Write checkpoint file (full simulation state)
     *
     * @param step Timestep number
     * @param time Physical time
     * @param fields Field arrays
     * @param particles Particle buffer
     *
     * @note Checkpoint includes everything needed to restart:
     * - All fields (Ex, Ey, Bz, charge, current, etc.)
     * - All particles (x, y, vx, vy, weight, type)
     * - Simulation metadata (step, time, parameters)
     */
    void write_checkpoint(int step, double time,
                         const FieldArrays& fields,
                         const ParticleBuffer& particles);

    /**
     * @brief Read checkpoint file and restore simulation state
     *
     * @param filename Checkpoint filename
     * @param step Output: timestep number
     * @param time Output: physical time
     * @param fields Output: field arrays (allocated and filled)
     * @param particles Output: particle buffer (allocated and filled)
     */
    void read_checkpoint(const std::string& filename,
                        int& step, double& time,
                        FieldArrays& fields,
                        ParticleBuffer& particles);

    // ==========================================================================
    // Metadata and diagnostics
    // ==========================================================================

    /**
     * @brief Write simulation metadata to HDF5 file
     *
     * @param file_id HDF5 file handle
     * @param step Timestep number
     * @param time Physical time
     */
    void write_metadata(hid_t file_id, int step, double time);

    /**
     * @brief Write diagnostics (energies, etc.) to CSV file
     *
     * @param step Timestep
     * @param time Physical time
     * @param diagnostics Map of diagnostic quantities
     *
     * @note Only master rank writes diagnostics file
     */
    void write_diagnostics(int step, double time,
                          const std::map<std::string, double>& diagnostics);

    // ==========================================================================
    // Utilities
    // ==========================================================================

    /**
     * @brief Create output directories if they don't exist
     */
    void create_directories();

    /**
     * @brief Generate filename for given step
     *
     * @param prefix Filename prefix (e.g., "fields")
     * @param step Timestep number
     * @return Full path: output_dir/prefix_step_XXXX.h5
     */
    std::string get_filename(const std::string& prefix, int step) const;

    /**
     * @brief List all checkpoint files in checkpoint directory
     *
     * @return Vector of checkpoint filenames, sorted by step number
     */
    std::vector<std::string> list_checkpoints() const;

    /**
     * @brief Find latest checkpoint file
     *
     * @return Path to latest checkpoint, or empty string if none found
     */
    std::string find_latest_checkpoint() const;

private:
    /**
     * @brief Create HDF5 file with parallel access
     *
     * @param filename Output filename
     * @return HDF5 file handle
     */
    hid_t create_parallel_file(const std::string& filename);

    /**
     * @brief Open HDF5 file with parallel access
     */
    hid_t open_parallel_file(const std::string& filename);

    /**
     * @brief Copy field data from device to host
     *
     * @param d_field Device pointer
     * @param h_field Host buffer (allocated by caller)
     * @param nx,ny Grid dimensions
     */
    void device_to_host(const double* d_field, double* h_field, int nx, int ny);

    /**
     * @brief Copy field data from host to device
     */
    void host_to_device(const double* h_field, double* d_field, int nx, int ny);

    // Diagnostics file handle (CSV format)
    FILE* diag_file;
};

} // namespace jericho
