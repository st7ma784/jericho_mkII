/**
 * @file ghost_cell_exchange.h
 * @brief API for ghost cell exchange in MPI simulations
 * @author Jericho Mk II Development Team
 * 
 * Phase 11.1 Interface: Full ghost cell synchronization
 */

#pragma once

#include <vector>
#include "mpi_domain_state.h"

// Forward declarations
namespace jericho {
    class FieldArrays;
}

// MPI domain state (from main_mpi.cpp)

// ============================================================================
// Boundary Extraction API
// ============================================================================

/**
 * Extract North boundary (top interior row) for transmission
 * @param field Field array to extract from
 * @param buffer Output buffer (will be resized to nx - 2*nghost)
 * @param nx Grid size including ghosts
 * @param ny Grid size including ghosts
 * @param nghost Number of ghost layers
 */
void extract_north_boundary(const double* field, std::vector<double>& buffer,
                           int nx, int ny, int nghost);

/**
 * Extract South boundary (bottom interior row)
 */
void extract_south_boundary(const double* field, std::vector<double>& buffer,
                           int nx, int ny, int nghost);

/**
 * Extract East boundary (right interior column)
 */
void extract_east_boundary(const double* field, std::vector<double>& buffer,
                          int nx, int ny, int nghost);

/**
 * Extract West boundary (left interior column)
 */
void extract_west_boundary(const double* field, std::vector<double>& buffer,
                          int nx, int ny, int nghost);

// ============================================================================
// Ghost Cell Injection API
// ============================================================================

/**
 * Inject received North boundary into ghost cells
 * @param field Field array to inject into
 * @param buffer Input buffer (received from North neighbor)
 * @param nx Grid size including ghosts
 * @param ny Grid size including ghosts
 * @param nghost Number of ghost layers
 */
void inject_north_ghost(double* field, const std::vector<double>& buffer,
                       int nx, int ny, int nghost);

/**
 * Inject received South boundary into ghost cells
 */
void inject_south_ghost(double* field, const std::vector<double>& buffer,
                       int nx, int ny, int nghost);

/**
 * Inject received East boundary into ghost cells
 */
void inject_east_ghost(double* field, const std::vector<double>& buffer,
                      int nx, int ny, int nghost);

/**
 * Inject received West boundary into ghost cells
 */
void inject_west_ghost(double* field, const std::vector<double>& buffer,
                      int nx, int ny, int nghost);

// ============================================================================
// Full Synchronization API (Main Public Interface)
// ============================================================================

/**
 * Synchronize a single field across all MPI boundaries
 * 
 * Uses non-blocking MPI from Phase 10 framework:
 * - Extracts boundaries
 * - Posts MPI_Isend/Irecv
 * - Waits for completion
 * - Injects into ghost cells
 *
 * @param fields Field array container
 * @param mpi_state MPI domain state with neighbor info
 * @param field_ptr Pointer to specific field (Ex, Ey, Bz, rho, Jx, etc.)
 */
void synchronize_field_boundaries(jericho::FieldArrays& fields,
                                 struct MPIDomainState& mpi_state,
                                 double* field_ptr);

/**
 * Synchronize all electromagnetic fields (Ex, Ey, Bz)
 * 
 * Calls synchronize_field_boundaries for each field component.
 * This is the typical operation needed per timestep.
 *
 * @param fields Field array container  
 * @param mpi_state MPI domain state with neighbor info
 */
void synchronize_all_fields(jericho::FieldArrays& fields,
                           struct MPIDomainState& mpi_state);

// ============================================================================
// Diagnostics API
// ============================================================================

/**
 * Verify boundary data continuity across domain decomposition
 * 
 * @param fields Field arrays
 * @param mpi_state MPI domain state
 * @param field_ptr Field to verify
 * @return true if boundaries valid, false if NaN/Inf detected
 */
bool verify_boundary_continuity(const jericho::FieldArrays& fields,
                               struct MPIDomainState& mpi_state,
                               const double* field_ptr);

/**
 * Print boundary statistics for debugging
 * 
 * @param fields Field arrays
 * @param field_ptr Field to analyze
 * @param field_name Name for output
 */
void print_boundary_stats(const jericho::FieldArrays& fields,
                         const double* field_ptr,
                         const char* field_name);

