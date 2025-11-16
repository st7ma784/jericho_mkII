/**
 * @file mpi_manager.h
 * @brief MPI+CUDA hybrid parallelism manager
 * @author Jericho Mk II Development Team
 *
 * This file implements domain decomposition and inter-process communication
 * for multi-GPU simulations using MPI. Supports CUDA-aware MPI for efficient
 * GPU-to-GPU data transfer.
 */

#pragma once

#include "field_arrays.h"
#include "particle_buffer.h"
#include <mpi.h>
#include <vector>

namespace jericho {

/**
 * @brief MPI+CUDA manager for multi-GPU parallelism
 *
 * @details Handles:
 * - 2D domain decomposition
 * - Ghost cell exchange (fields)
 * - Particle migration between ranks
 * - Collective reductions (diagnostics)
 * - CUDA-aware MPI optimization
 *
 * Domain decomposition:
 * ```
 * ┌─────────┬─────────┬─────────┐
 * │ Rank 0  │ Rank 1  │ Rank 2  │  GPU 0, 1, 2
 * ├─────────┼─────────┼─────────┤
 * │ Rank 3  │ Rank 4  │ Rank 5  │  GPU 0, 1, 2
 * └─────────┴─────────┴─────────┘
 * ```
 */
class MPIManager {
public:
    // MPI topology
    int rank;           ///< This process's rank
    int size;           ///< Total number of MPI processes
    int npx;            ///< Processes in x direction
    int npy;            ///< Processes in y direction
    int rank_x;         ///< This rank's x coordinate in process grid
    int rank_y;         ///< This rank's y coordinate in process grid

    // Neighbor ranks (for ghost cell exchange)
    int neighbor_left;    ///< Rank of left neighbor (-1 if boundary)
    int neighbor_right;   ///< Rank of right neighbor
    int neighbor_bottom;  ///< Rank of bottom neighbor
    int neighbor_top;     ///< Rank of top neighbor

    // Domain decomposition
    double x_min_global, x_max_global;  ///< Global domain bounds
    double y_min_global, y_max_global;
    double x_min_local, x_max_local;    ///< Local subdomain bounds
    double y_min_local, y_max_local;

    int nx_global, ny_global;  ///< Global grid dimensions
    int nx_local, ny_local;    ///< Local grid dimensions (excluding ghosts)

    // CUDA-aware MPI flag
    bool cuda_aware_mpi;

    // Communication buffers
    struct CommBuffers {
        // Host buffers (for non-CUDA-aware MPI)
        std::vector<double> h_send_left, h_send_right;
        std::vector<double> h_send_bottom, h_send_top;
        std::vector<double> h_recv_left, h_recv_right;
        std::vector<double> h_recv_bottom, h_recv_top;

        // Device buffers (for CUDA-aware MPI)
        double* d_send_left = nullptr;
        double* d_send_right = nullptr;
        double* d_send_bottom = nullptr;
        double* d_send_top = nullptr;
        double* d_recv_left = nullptr;
        double* d_recv_right = nullptr;
        double* d_recv_bottom = nullptr;
        double* d_recv_top = nullptr;

        void allocate(int nx, int ny, int nghost, bool use_device);
        void deallocate();
    } comm_buffers;

    // ==========================================================================
    // Constructors / Destructors
    // ==========================================================================

    /**
     * @brief Initialize MPI manager
     *
     * @param argc,argv Command line arguments (for MPI_Init)
     * @param npx,npy Process grid dimensions
     * @param nx_global,ny_global Global grid dimensions
     * @param x_min,x_max,y_min,y_max Global domain bounds
     * @param cuda_aware Whether to use CUDA-aware MPI
     */
    MPIManager(int* argc, char*** argv,
              int npx, int npy,
              int nx_global, int ny_global,
              double x_min, double x_max,
              double y_min, double y_max,
              bool cuda_aware = false);

    /**
     * @brief Destructor - finalize MPI
     */
    ~MPIManager();

    // ==========================================================================
    // Domain decomposition
    // ==========================================================================

    /**
     * @brief Compute local subdomain bounds for this rank
     */
    void compute_local_domain();

    /**
     * @brief Get rank from 2D coordinates
     *
     * @param rx,ry Rank coordinates
     * @return MPI rank
     */
    int get_rank(int rx, int ry) const {
        return ry * npx + rx;
    }

    /**
     * @brief Get 2D coordinates from rank
     *
     * @param rank MPI rank
     * @param rx,ry Output: rank coordinates
     */
    void get_coords(int rank, int& rx, int& ry) const {
        rx = rank % npx;
        ry = rank / npx;
    }

    // ==========================================================================
    // Ghost cell exchange (fields)
    // ==========================================================================

    /**
     * @brief Exchange ghost cells for a single field
     *
     * @param field Field array on device
     * @param nx,ny Grid dimensions (including ghosts)
     * @param nghost Number of ghost layers
     *
     * @note Uses non-blocking MPI (Isend/Irecv) for better performance
     */
    void exchange_ghost_cells(double* field, int nx, int ny, int nghost);

    /**
     * @brief Exchange ghost cells for all electromagnetic fields
     *
     * @param fields Field arrays structure
     */
    void exchange_all_fields(FieldArrays& fields);

    // ==========================================================================
    // Particle migration
    // ==========================================================================

    /**
     * @brief Migrate particles that have left local domain
     *
     * @details Particles that cross subdomain boundaries are sent to
     * neighboring ranks. This maintains load balance and correctness.
     *
     * Steps:
     * 1. Identify particles outside local domain
     * 2. Pack particle data for each neighbor
     * 3. Exchange particle counts
     * 4. Exchange particle data
     * 5. Unpack received particles
     * 6. Remove migrated particles
     *
     * @param particles Particle buffer
     */
    void migrate_particles(ParticleBuffer& particles);

    // ==========================================================================
    // Collective operations
    // ==========================================================================

    /**
     * @brief Global sum (reduction)
     *
     * @param local_value Value on this rank
     * @return Global sum across all ranks
     */
    double global_sum(double local_value);

    /**
     * @brief Global maximum
     */
    double global_max(double local_value);

    /**
     * @brief Global minimum
     */
    double global_min(double local_value);

    /**
     * @brief Barrier synchronization
     */
    void barrier() const {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ==========================================================================
    // Diagnostics and utilities
    // ==========================================================================

    /**
     * @brief Print MPI topology information
     */
    void print_topology() const;

    /**
     * @brief Check if this rank is on a domain boundary
     */
    bool on_left_boundary() const { return rank_x == 0; }
    bool on_right_boundary() const { return rank_x == npx - 1; }
    bool on_bottom_boundary() const { return rank_y == 0; }
    bool on_top_boundary() const { return rank_y == npy - 1; }

    /**
     * @brief Master rank (for I/O coordination)
     */
    bool is_master() const { return rank == 0; }

private:
    /**
     * @brief Pack ghost cells from device to send buffer
     */
    void pack_ghost_cells(const double* field, int nx, int ny, int nghost,
                         double* send_buffer, int direction);

    /**
     * @brief Unpack received ghost cells to device
     */
    void unpack_ghost_cells(double* field, int nx, int ny, int nghost,
                           const double* recv_buffer, int direction);
};

} // namespace jericho
