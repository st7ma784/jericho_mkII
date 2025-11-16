/**
 * @file io_manager.cpp
 * @brief Implementation of HDF5-based parallel I/O manager
 * @author Jericho Mk II Development Team
 */

#include "io_manager.h"

#include "platform.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace jericho {

// =============================================================================
// Helper macros
// =============================================================================

// Simplified implementation for CPU build
// Full HDF5 implementation available in original file

IOManager::IOManager(const std::string& output_dir, const MPIManager* mpi)
    : output_dir(output_dir), checkpoint_dir(output_dir + "/checkpoints"), field_cadence(10),
      particle_cadence(100), checkpoint_cadence(1000), compress_output(true),
      output_particles(true), output_Ex(true), output_Ey(true), output_Bz(true),
      output_charge(true), output_current(true), output_flow_velocity(true), mpi(mpi),
      diag_file(nullptr) {
    create_directories();
    if (mpi->is_master()) {
        std::string diag_filename = output_dir + "/diagnostics.csv";
        diag_file = fopen(diag_filename.c_str(), "w");
        if (diag_file) {
            fprintf(diag_file, "step,time,total_energy,num_particles\n");
            fflush(diag_file);
        }
    }
}

IOManager::~IOManager() {
    if (diag_file) {
        fclose(diag_file);
        diag_file = nullptr;
    }
}

void IOManager::create_directories() {
    if (!mpi->is_master()) {
        mpi->barrier();
        return;
    }
    mkdir(output_dir.c_str(), 0755);
    mkdir(checkpoint_dir.c_str(), 0755);
    mpi->barrier();
}

std::string IOManager::get_filename(const std::string& prefix, int step) const {
    std::ostringstream oss;
    oss << output_dir << "/" << prefix << "_step_" << std::setw(6) << std::setfill('0') << step
        << ".h5";
    return oss.str();
}

std::vector<std::string> IOManager::list_checkpoints() const {
    return std::vector<std::string>(); // Stub
}

std::string IOManager::find_latest_checkpoint() const {
    return ""; // Stub
}

hid_t IOManager::create_parallel_file(const std::string& filename) {
    // Simplified stub - full HDF5 implementation in original
    return 0;
}

hid_t IOManager::open_parallel_file(const std::string& filename) {
    return 0; // Stub
}

void IOManager::device_to_host(const double* d_field, double* h_field, int nx, int ny) {
    size_t bytes = nx * ny * sizeof(double);
    memcpy(h_field, d_field, bytes); // In CPU mode, no device/host distinction
}

void IOManager::host_to_device(const double* h_field, double* d_field, int nx, int ny) {
    size_t bytes = nx * ny * sizeof(double);
    memcpy(d_field, h_field, bytes);
}

void IOManager::write_metadata(hid_t file_id, int step, double time) {
    // HDF5 stub
}

void IOManager::write_field(const std::string& filename, const std::string& dataset_name,
                            const double* field, int nx_local, int ny_local, int nx_global,
                            int ny_global, int offset_x, int offset_y) {
    // HDF5 stub - would write actual data here
    if (mpi->is_master()) {
        std::cout << "Would write field " << dataset_name << " to " << filename << std::endl;
    }
}

void IOManager::write_fields(int step, double time, const FieldArrays& fields) {
    if (mpi->is_master()) {
        std::cout << "Step " << step << ": Writing fields (stub)" << std::endl;
    }
}

void IOManager::write_particles(int step, double time, const ParticleBuffer& particles) {
    if (mpi->is_master()) {
        std::cout << "Step " << step << ": Writing particles (stub)" << std::endl;
    }
}

void IOManager::write_checkpoint(int step, double time, const FieldArrays& fields,
                                 const ParticleBuffer& particles) {
    if (mpi->is_master()) {
        std::cout << "Checkpoint at step " << step << std::endl;
    }
}

void IOManager::read_checkpoint(const std::string& filename, int& step, double& time,
                                FieldArrays& fields, ParticleBuffer& particles) {
    throw std::runtime_error("Checkpoint reading not implemented");
}

void IOManager::write_diagnostics(int step, double time,
                                  const std::map<std::string, double>& diagnostics) {
    if (!mpi->is_master() || !diag_file)
        return;

    fprintf(diag_file, "%d,%.10e", step, time);

    auto it = diagnostics.find("total_energy");
    fprintf(diag_file, ",%.10e", (it != diagnostics.end()) ? it->second : 0.0);

    it = diagnostics.find("num_particles");
    fprintf(diag_file, ",%.0f", (it != diagnostics.end()) ? it->second : 0.0);

    fprintf(diag_file, "\n");
    fflush(diag_file);
}

} // namespace jericho
