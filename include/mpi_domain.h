/**
 * MPI Setup and Domain Decomposition Framework
 * Provides foundational MPI infrastructure for jericho_mkII
 *
 * Features:
 * - 2D Cartesian domain decomposition
 * - Ghost cell communication
 * - Particle load balancing
 * - Global reductions for diagnostics
 */

#pragma once
#include <mpi.h>

#include <array>
#include <iostream>
#include <vector>

namespace mpi {

/**
 * MPI Context: Manages MPI communicators and rank information
 */
struct MPIContext {
    int rank = 0; // Global rank in MPI_COMM_WORLD
    int size = 1; // Total number of processes

    MPI_Comm comm_2d = MPI_COMM_NULL; // 2D Cartesian communicator

    std::array<int, 2> dims = {0, 0};    // Logical dimensions
    std::array<int, 2> coords = {0, 0};  // This rank's coordinates
    std::array<int, 2> periods = {0, 0}; // Boundary conditions (false = open)

    // Neighboring ranks
    int rank_top = MPI_PROC_NULL;
    int rank_bottom = MPI_PROC_NULL;
    int rank_left = MPI_PROC_NULL;
    int rank_right = MPI_PROC_NULL;
};

/**
 * MPI Domain: Describes local subdomain for this rank
 */
struct MPIDomain {
    // Local grid dimensions
    int nx_local = 0;
    int ny_local = 0;

    // Global grid dimensions
    int nx_global = 0;
    int ny_global = 0;

    // Domain bounds in physical space
    double x_min = 0.0, x_max = 0.0;
    double y_min = 0.0, y_max = 0.0;

    // Domain bounds in units of grid cells
    int ix_min = 0, ix_max = 0;
    int iy_min = 0, iy_max = 0;

    // Grid spacing
    double dx = 0.0;
    double dy = 0.0;

    // Ghost cell buffers for boundaries
    // Organized as [top, bottom, left, right]
    std::vector<double> ghost_top;
    std::vector<double> ghost_bottom;
    std::vector<double> ghost_left;
    std::vector<double> ghost_right;
};

/**
 * Initialize MPI and create 2D Cartesian communicator
 *
 * Args:
 *   argc, argv: Command line arguments
 *   ctx: MPI context (filled on output)
 *   preferred_dims: Preferred rank grid dimensions (can be {0,0} for auto)
 *
 * Returns: 0 on success
 */
inline int init_mpi(int argc, char** argv, MPIContext& ctx,
                    const std::array<int, 2>& preferred_dims = {0, 0}) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    // Determine 2D rank layout
    ctx.dims[0] = preferred_dims[0];
    ctx.dims[1] = preferred_dims[1];
    MPI_Dims_create(ctx.size, 2, ctx.dims.data());

    if (ctx.rank == 0) {
        std::cout << "MPI initialized with " << ctx.size << " processes\n"
                  << "Rank grid: " << ctx.dims[0] << " × " << ctx.dims[1] << std::endl;
    }

    // Create 2D Cartesian communicator
    // periods[i] = 0 means open boundaries (no wrapping)
    ctx.periods[0] = 0;
    ctx.periods[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, ctx.dims.data(), ctx.periods.data(), 1, &ctx.comm_2d);

    // Get coordinates of this rank
    MPI_Cart_coords(ctx.comm_2d, ctx.rank, 2, ctx.coords.data());

    // Find neighboring ranks
    MPI_Cart_shift(ctx.comm_2d, 0, 1, &ctx.rank_left, &ctx.rank_right);
    MPI_Cart_shift(ctx.comm_2d, 1, 1, &ctx.rank_bottom, &ctx.rank_top);

    if (ctx.rank == 0) {
        std::cout << "MPI Cartesian communicator created successfully\n"
                  << "Rank (" << ctx.coords[0] << "," << ctx.coords[1]
                  << "): neighbors L=" << ctx.rank_left << " R=" << ctx.rank_right
                  << " T=" << ctx.rank_top << " B=" << ctx.rank_bottom << std::endl;
    }

    return 0;
}

/**
 * Setup local domain for this rank
 *
 * Args:
 *   ctx: MPI context
 *   domain: Domain description (filled on output)
 *   nx_global, ny_global: Global grid dimensions
 *   dx, dy: Grid spacing
 *   x_min, y_min: Global domain lower bounds
 */
inline void setup_domain(const MPIContext& ctx, MPIDomain& domain, int nx_global, int ny_global,
                         double dx, double dy, double x_min, double y_min) {

    domain.nx_global = nx_global;
    domain.ny_global = ny_global;
    domain.dx = dx;
    domain.dy = dy;

    // Calculate local domain size
    domain.nx_local = nx_global / ctx.dims[0];
    domain.ny_local = ny_global / ctx.dims[1];

    // Calculate indices in grid
    domain.ix_min = ctx.coords[0] * domain.nx_local;
    domain.ix_max = domain.ix_min + domain.nx_local;
    domain.iy_min = ctx.coords[1] * domain.ny_local;
    domain.iy_max = domain.iy_min + domain.ny_local;

    // Calculate physical bounds
    domain.x_min = x_min + domain.ix_min * dx;
    domain.x_max = x_min + domain.ix_max * dx;
    domain.y_min = y_min + domain.iy_min * dy;
    domain.y_max = y_min + domain.iy_max * dy;

    // Allocate ghost cell buffers
    domain.ghost_top.assign(domain.nx_local, 0.0);
    domain.ghost_bottom.assign(domain.nx_local, 0.0);
    domain.ghost_left.assign(domain.ny_local, 0.0);
    domain.ghost_right.assign(domain.ny_local, 0.0);
}

/**
 * Check if particle is in this domain
 */
inline bool contains_particle(const MPIDomain& domain, double x, double y) {
    return (x >= domain.x_min && x < domain.x_max && y >= domain.y_min && y < domain.y_max);
}

/**
 * Exchange ghost cells for field data (e.g., charge density, potential)
 *
 * Non-blocking communication for overlapping with computation
 *
 * Args:
 *   field: Local field data (nx_local × ny_local)
 *   domain: Domain description
 *   ctx: MPI context
 *   requests: Array to store MPI_Request handles
 */
inline void exchange_ghosts_irecv(const std::vector<double>& field, const MPIDomain& domain,
                                  const MPIContext& ctx, std::vector<MPI_Request>& requests) {

    int nx = domain.nx_local;
    int ny = domain.ny_local;
    requests.clear();
    requests.resize(8); // 4 sends + 4 recvs

    int req_idx = 0;

    // Top boundary
    if (ctx.rank_top != MPI_PROC_NULL) {
        // Send top row to top neighbor
        std::vector<double> send_buf(nx);
        for (int i = 0; i < nx; i++) {
            send_buf[i] = field[0 * nx + i]; // Row 0
        }
        MPI_Isend(send_buf.data(), nx, MPI_DOUBLE, ctx.rank_top, 1, ctx.comm_2d,
                  &requests[req_idx++]);

        // Receive top ghost cells from top neighbor
        MPI_Irecv(domain.ghost_top.data(), nx, MPI_DOUBLE, ctx.rank_top, 2, ctx.comm_2d,
                  &requests[req_idx++]);
    }

    // Bottom boundary
    if (ctx.rank_bottom != MPI_PROC_NULL) {
        // Send bottom row to bottom neighbor
        std::vector<double> send_buf(nx);
        for (int i = 0; i < nx; i++) {
            send_buf[i] = field[(ny - 1) * nx + i]; // Last row
        }
        MPI_Isend(send_buf.data(), nx, MPI_DOUBLE, ctx.rank_bottom, 2, ctx.comm_2d,
                  &requests[req_idx++]);

        // Receive bottom ghost cells
        MPI_Irecv(domain.ghost_bottom.data(), nx, MPI_DOUBLE, ctx.rank_bottom, 1, ctx.comm_2d,
                  &requests[req_idx++]);
    }

    // Left boundary
    if (ctx.rank_left != MPI_PROC_NULL) {
        // Send left column
        std::vector<double> send_buf(ny);
        for (int j = 0; j < ny; j++) {
            send_buf[j] = field[j * nx + 0];
        }
        MPI_Isend(send_buf.data(), ny, MPI_DOUBLE, ctx.rank_left, 3, ctx.comm_2d,
                  &requests[req_idx++]);

        // Receive left ghost cells
        MPI_Irecv(domain.ghost_left.data(), ny, MPI_DOUBLE, ctx.rank_left, 4, ctx.comm_2d,
                  &requests[req_idx++]);
    }

    // Right boundary
    if (ctx.rank_right != MPI_PROC_NULL) {
        // Send right column
        std::vector<double> send_buf(ny);
        for (int j = 0; j < ny; j++) {
            send_buf[j] = field[j * nx + (nx - 1)];
        }
        MPI_Isend(send_buf.data(), ny, MPI_DOUBLE, ctx.rank_right, 4, ctx.comm_2d,
                  &requests[req_idx++]);

        // Receive right ghost cells
        MPI_Irecv(domain.ghost_right.data(), ny, MPI_DOUBLE, ctx.rank_right, 3, ctx.comm_2d,
                  &requests[req_idx++]);
    }

    // Note: Caller should use MPI_Waitall() when ghosts are needed
}

/**
 * Finalize MPI
 */
inline void finalize_mpi(MPIContext& ctx) {
    if (ctx.comm_2d != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx.comm_2d);
    }
    MPI_Finalize();
}

/**
 * Global reduction: Sum all values
 */
inline double global_sum(double local_value, const MPIContext& ctx) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, ctx.comm_2d);
    return global_value;
}

/**
 * Global reduction: Max value
 */
inline double global_max(double local_value, const MPIContext& ctx) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, ctx.comm_2d);
    return global_value;
}

} // namespace mpi
