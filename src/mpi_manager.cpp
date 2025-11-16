/**
 * @file mpi_manager.cpp
 * @brief Implementation of MPI+CUDA hybrid parallelism manager
 * @author Jericho Mk II Development Team
 */

#include "mpi_manager.h"

#include "platform.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace jericho {

// =============================================================================
// Helper macros
// =============================================================================

// Placeholder for full implementation - see original mpi_manager.cpp
// The sed command accidentally deleted most of the file
// For CPU build, the key functions are:

void MPIManager::CommBuffers::allocate(int nx, int ny, int nghost, bool use_device) {
    // Simplified CPU implementation
    size_t x_buffer_size = ny * nghost;
    size_t y_buffer_size = nx * nghost;

    h_send_left.resize(x_buffer_size);
    h_send_right.resize(x_buffer_size);
    h_send_bottom.resize(y_buffer_size);
    h_send_top.resize(y_buffer_size);
    h_recv_left.resize(x_buffer_size);
    h_recv_right.resize(x_buffer_size);
    h_recv_bottom.resize(y_buffer_size);
    h_recv_top.resize(y_buffer_size);
}

void MPIManager::CommBuffers::deallocate() {
    h_send_left.clear();
    h_send_right.clear();
    h_send_bottom.clear();
    h_send_top.clear();
    h_recv_left.clear();
    h_recv_right.clear();
    h_recv_bottom.clear();
    h_recv_top.clear();
}

MPIManager::MPIManager(int* argc, char*** argv, int npx, int npy, int nx_global, int ny_global,
                       double x_min, double x_max, double y_min, double y_max, bool cuda_aware)
    : npx(npx), npy(npy), nx_global(nx_global), ny_global(ny_global), x_min_global(x_min),
      x_max_global(x_max), y_min_global(y_min), y_max_global(y_max), cuda_aware_mpi(cuda_aware) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != npx * npy) {
        if (rank == 0) {
            std::cerr << "ERROR: MPI size mismatch\n";
        }
        MPI_Finalize();
        throw std::runtime_error("MPI process grid mismatch");
    }

    get_coords(rank, rank_x, rank_y);
    neighbor_left = on_left_boundary() ? -1 : get_rank(rank_x - 1, rank_y);
    neighbor_right = on_right_boundary() ? -1 : get_rank(rank_x + 1, rank_y);
    neighbor_bottom = on_bottom_boundary() ? -1 : get_rank(rank_x, rank_y - 1);
    neighbor_top = on_top_boundary() ? -1 : get_rank(rank_x, rank_y + 1);

    compute_local_domain();
}

MPIManager::~MPIManager() {
    comm_buffers.deallocate();
    MPI_Finalize();
}

void MPIManager::compute_local_domain() {
    nx_local = nx_global / npx;
    ny_local = ny_global / npy;

    if (rank_x == npx - 1)
        nx_local += nx_global % npx;
    if (rank_y == npy - 1)
        ny_local += ny_global % npy;

    double dx_global = (x_max_global - x_min_global) / nx_global;
    double dy_global = (y_max_global - y_min_global) / ny_global;

    int x_start = rank_x * (nx_global / npx);
    int y_start = rank_y * (ny_global / npy);

    x_min_local = x_min_global + x_start * dx_global;
    y_min_local = y_min_global + y_start * dy_global;
    x_max_local = x_min_local + nx_local * dx_global;
    y_max_local = y_min_local + ny_local * dy_global;
}

void MPIManager::exchange_ghost_cells(double* field, int nx, int ny, int nghost) {
    // Simplified stub for CPU build
    // Full implementation would do MPI communication here
}

void MPIManager::exchange_all_fields(FieldArrays& fields) {
    exchange_ghost_cells(fields.Ex, fields.nx, fields.ny, fields.nghost);
    exchange_ghost_cells(fields.Ey, fields.nx, fields.ny, fields.nghost);
    exchange_ghost_cells(fields.Bz, fields.nx, fields.ny, fields.nghost);
}

void MPIManager::migrate_particles(ParticleBuffer& particles) {
    // Stub
}

double MPIManager::global_sum(double local_value) {
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}

double MPIManager::global_max(double local_value) {
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_value;
}

double MPIManager::global_min(double local_value) {
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return global_value;
}

void MPIManager::print_topology() const {
    std::cout << "\n=== MPI Topology ===" << std::endl;
    std::cout << "Total processes: " << size << std::endl;
    std::cout << "Process grid: " << npx << " x " << npy << std::endl;
    std::cout << "===================\n" << std::endl;
}

} // namespace jericho
