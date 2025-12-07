/**
 * @file mpi_domain_state.h
 * @brief MPI domain decomposition state structure
 * @author Jericho Mk II Development Team
 * 
 * Defines the MPIDomainState structure used throughout the MPI framework
 */

#pragma once

#include <mpi.h>
#include <array>
#include <vector>

// ============================================================================
// MPI Non-Blocking Communication State
// ============================================================================

struct NonBlockingExchangeState {
    std::array<MPI_Request, 8> requests;  // 4 directions * 2 (send/recv) = 8
    std::vector<double> send_buffers[4];  // N, S, E, W
    std::vector<double> recv_buffers[4];  // N, S, E, W
    bool exchange_in_progress = false;
    
    void reset() {
        exchange_in_progress = false;
        for (int i = 0; i < 8; i++) {
            requests[i] = MPI_REQUEST_NULL;
        }
    }
};

// ============================================================================
// MPI Domain Decomposition State
// ============================================================================

/**
 * @struct MPIDomainState
 * @brief Holds MPI topology, boundaries, and non-blocking communication state
 * 
 * This structure encapsulates:
 * - MPI rank and Cartesian coordinates
 * - Neighbor ranks for 4-directional communication
 * - Local domain bounds and dimensions
 * - Global domain information
 * - Phase 10 non-blocking communication state (MPI_Isend/Irecv)
 * - Phase 10 batched diagnostics state
 */
struct MPIDomainState {
    // =========================================================================
    // MPI Topology
    // =========================================================================
    
    int rank;           ///< Global MPI rank
    int coords[2];      ///< (px, py) Cartesian coordinates
    int neighbors[4];   ///< N, S, E, W neighbor ranks (0=N, 1=S, 2=E, 3=W)
    
    // =========================================================================
    // Local Domain Bounds
    // =========================================================================
    
    double x_min_local, x_max_local;  ///< Local x extent
    double y_min_local, y_max_local;  ///< Local y extent
    int nx_local, ny_local;            ///< Local grid dimensions
    
    // =========================================================================
    // Global Domain Information
    // =========================================================================
    
    double x_min_global, x_max_global;  ///< Global x extent
    double y_min_global, y_max_global;  ///< Global y extent
    int nx_global, ny_global;            ///< Global grid dimensions
    
    // =========================================================================
    // MPI Communicators
    // =========================================================================
    
    MPI_Comm cart_comm;  ///< Cartesian communicator for 2D topology
    
    // =========================================================================
    // Phase 10: Non-Blocking Communication State
    // =========================================================================
    
    NonBlockingExchangeState exchange_state;  ///< Non-blocking MPI send/recv state
    
    // =========================================================================
    // Phase 10: Batched Diagnostics
    // =========================================================================
    
    double E_total_buffered = 0.0;      ///< Buffered energy for diagnostics
    int N_particles_buffered = 0;       ///< Buffered particle count
    int diagnostic_step_counter = 0;    ///< Counter for batching interval
    const int DIAGNOSTIC_INTERVAL = 10; ///< Flush diagnostics every N steps
};

