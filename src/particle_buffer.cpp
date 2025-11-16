/**
 * @file particle_buffer.cpp
 * @brief Implementation of GPU-optimized SoA particle buffer
 * @author Jericho Mk II Development Team
 */

#include "particle_buffer.h"
#include "platform.h"
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cmath>
#include <random>

namespace jericho {

// =============================================================================
// Constructors / Destructors
// =============================================================================

ParticleBuffer::ParticleBuffer(size_t initial_capacity, int device_id)
    : capacity(0), count(0), device_id(device_id),
      x(nullptr), y(nullptr), vx(nullptr), vy(nullptr),
      weight(nullptr), type(nullptr), active(nullptr),
      free_slots(nullptr), free_count(0)
{
    stats = {0, 0, 0, 0};
    allocate(initial_capacity);
}

ParticleBuffer::~ParticleBuffer() {
    destroy();
}

ParticleBuffer::ParticleBuffer(ParticleBuffer&& other) noexcept
    : capacity(other.capacity), count(other.count), device_id(other.device_id),
      free_count(other.free_count), stats(other.stats)
{
    x = other.x; other.x = nullptr;
    y = other.y; other.y = nullptr;
    vx = other.vx; other.vx = nullptr;
    vy = other.vy; other.vy = nullptr;
    weight = other.weight; other.weight = nullptr;
    type = other.type; other.type = nullptr;
    active = other.active; other.active = nullptr;
    free_slots = other.free_slots; other.free_slots = nullptr;
}

ParticleBuffer& ParticleBuffer::operator=(ParticleBuffer&& other) noexcept {
    if (this != &other) {
        destroy();

        capacity = other.capacity;
        count = other.count;
        device_id = other.device_id;
        free_count = other.free_count;
        stats = other.stats;

        x = other.x; other.x = nullptr;
        y = other.y; other.y = nullptr;
        vx = other.vx; other.vx = nullptr;
        vy = other.vy; other.vy = nullptr;
        weight = other.weight; other.weight = nullptr;
        type = other.type; other.type = nullptr;
        active = other.active; other.active = nullptr;
        free_slots = other.free_slots; other.free_slots = nullptr;
    }
    return *this;
}

// =============================================================================
// Memory management
// =============================================================================

void ParticleBuffer::allocate(size_t new_capacity) {
    capacity = new_capacity;

    size_t double_bytes = capacity * sizeof(double);
    size_t bool_bytes = capacity * sizeof(bool);
    size_t uint8_bytes = capacity * sizeof(uint8_t);
    size_t size_t_bytes = capacity * sizeof(size_t);

    x = (double*)malloc(double_bytes);
    y = (double*)malloc(double_bytes);
    vx = (double*)malloc(double_bytes);
    vy = (double*)malloc(double_bytes);
    weight = (double*)malloc(double_bytes);
    type = (uint8_t*)malloc(uint8_bytes);
    active = (bool*)malloc(bool_bytes);
    free_slots = (size_t*)malloc(size_t_bytes);

    if (!x || !y || !vx || !vy || !weight || !type || !active || !free_slots) {
        throw std::runtime_error("Failed to allocate particle buffer memory");
    }

    // Initialize active flags to false
    memset(active, 0, bool_bytes);
    free_count = 0;
    count = 0;
}

void ParticleBuffer::destroy() {
    if (x) free(x);
    if (y) free(y);
    if (vx) free(vx);
    if (vy) free(vy);
    if (weight) free(weight);
    if (type) free(type);
    if (active) free(active);
    if (free_slots) free(free_slots);

    x = y = vx = vy = weight = nullptr;
    type = nullptr;
    active = nullptr;
    free_slots = nullptr;
    capacity = count = free_count = 0;
}

void ParticleBuffer::resize(size_t new_capacity) {
    if (new_capacity < count) {
        throw std::runtime_error("Cannot resize buffer smaller than current particle count");
    }

    // Allocate new arrays
    double* new_x = (double*)malloc(new_capacity * sizeof(double));
    double* new_y = (double*)malloc(new_capacity * sizeof(double));
    double* new_vx = (double*)malloc(new_capacity * sizeof(double));
    double* new_vy = (double*)malloc(new_capacity * sizeof(double));
    double* new_weight = (double*)malloc(new_capacity * sizeof(double));
    uint8_t* new_type = (uint8_t*)malloc(new_capacity * sizeof(uint8_t));
    bool* new_active = (bool*)malloc(new_capacity * sizeof(bool));
    size_t* new_free_slots = (size_t*)malloc(new_capacity * sizeof(size_t));

    // Copy old data
    memcpy(new_x, x, capacity * sizeof(double));
    memcpy(new_y, y, capacity * sizeof(double));
    memcpy(new_vx, vx, capacity * sizeof(double));
    memcpy(new_vy, vy, capacity * sizeof(double));
    memcpy(new_weight, weight, capacity * sizeof(double));
    memcpy(new_type, type, capacity * sizeof(uint8_t));
    memcpy(new_active, active, capacity * sizeof(bool));
    memcpy(new_free_slots, free_slots, free_count * sizeof(size_t));

    // Free old memory
    destroy();

    // Update pointers
    x = new_x;
    y = new_y;
    vx = new_vx;
    vy = new_vy;
    weight = new_weight;
    type = new_type;
    active = new_active;
    free_slots = new_free_slots;
    capacity = new_capacity;
}

void ParticleBuffer::compact() {
    // Stub for CPU build - would compact active particles in production
}

// =============================================================================
// Particle insertion / removal
// =============================================================================

size_t ParticleBuffer::add_particle(double px, double py, double pvx, double pvy,
                                   double pweight, uint8_t ptype) {
    size_t idx;

    // Use free slot if available, otherwise append
    if (free_count > 0) {
        idx = free_slots[--free_count];
    } else {
        if (count >= capacity) {
            resize(capacity * 3 / 2); // Grow by 1.5x
        }
        idx = count;
    }

    x[idx] = px;
    y[idx] = py;
    vx[idx] = pvx;
    vy[idx] = pvy;
    weight[idx] = pweight;
    type[idx] = ptype;
    active[idx] = true;

    count++;
    stats.particles_added++;
    if (count > stats.max_particles_seen) {
        stats.max_particles_seen = count;
    }

    return idx;
}

void ParticleBuffer::remove_particle(size_t idx) {
    if (idx >= capacity || !active[idx]) return;

    active[idx] = false;
    free_slots[free_count++] = idx;
    count--;
    stats.particles_removed++;
}

void ParticleBuffer::add_particles_batch(const double* positions, const double* velocities,
                                        const double* weights, const uint8_t* types,
                                        size_t n_particles) {
    // Ensure capacity
    if (count + n_particles > capacity) {
        resize(count + n_particles);
    }

    // Copy data
    for (size_t i = 0; i < n_particles; i++) {
        add_particle(positions[2*i], positions[2*i+1],
                    velocities[2*i], velocities[2*i+1],
                    weights[i], types[i]);
    }
}

// =============================================================================
// Data transfer
// =============================================================================

void ParticleBuffer::copy_to_device(const double* h_x, const double* h_y,
                                   const double* h_vx, const double* h_vy,
                                   const double* h_weight, const uint8_t* h_type,
                                   size_t n, size_t offset) {
    // In CPU mode, just copy
    memcpy(x + offset, h_x, n * sizeof(double));
    memcpy(y + offset, h_y, n * sizeof(double));
    memcpy(vx + offset, h_vx, n * sizeof(double));
    memcpy(vy + offset, h_vy, n * sizeof(double));
    memcpy(weight + offset, h_weight, n * sizeof(double));
    memcpy(type + offset, h_type, n * sizeof(uint8_t));

    // Mark as active
    for (size_t i = 0; i < n; i++) {
        active[offset + i] = true;
    }
}

void ParticleBuffer::copy_to_host(double* h_x, double* h_y, double* h_vx, double* h_vy,
                                 double* h_weight, uint8_t* h_type,
                                 size_t n, size_t offset) const {
    // In CPU mode, just copy
    memcpy(h_x, x + offset, n * sizeof(double));
    memcpy(h_y, y + offset, n * sizeof(double));
    memcpy(h_vx, vx + offset, n * sizeof(double));
    memcpy(h_vy, vy + offset, n * sizeof(double));
    memcpy(h_weight, weight + offset, n * sizeof(double));
    memcpy(h_type, type + offset, n * sizeof(uint8_t));
}

void ParticleBuffer::print_stats() const {
    std::cout << "Particle Buffer Statistics:\n"
              << "  Active particles: " << count << " / " << capacity << "\n"
              << "  Memory usage: " << get_memory_bytes() / 1e6 << " MB\n"
              << "  Particles added: " << stats.particles_added << "\n"
              << "  Particles removed: " << stats.particles_removed << "\n"
              << "  Particles reflected: " << stats.particles_reflected << "\n"
              << "  Peak count: " << stats.max_particles_seen << "\n";
}

// =============================================================================
// Helper functions
// =============================================================================

void initialize_uniform(ParticleBuffer& buffer,
                       double x_min, double x_max, double y_min, double y_max,
                       int particles_per_cell, int nx, int ny,
                       uint8_t ptype, double pweight,
                       double vx_mean, double vy_mean, double v_thermal,
                       unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;

    size_t total_particles = nx * ny * particles_per_cell;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            double cell_x_min = x_min + ix * dx;
            double cell_y_min = y_min + iy * dy;

            for (int ip = 0; ip < particles_per_cell; ip++) {
                double px = cell_x_min + uniform(rng) * dx;
                double py = cell_y_min + uniform(rng) * dy;
                double pvx = vx_mean + v_thermal * normal(rng);
                double pvy = vy_mean + v_thermal * normal(rng);

                buffer.add_particle(px, py, pvx, pvy, pweight, ptype);
            }
        }
    }
}

void initialize_maxwellian_velocities(ParticleBuffer& buffer,
                                     double v_thermal,
                                     double vx_drift, double vy_drift,
                                     unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (size_t i = 0; i < buffer.count; i++) {
        if (buffer.active[i]) {
            buffer.vx[i] = vx_drift + v_thermal * normal(rng);
            buffer.vy[i] = vy_drift + v_thermal * normal(rng);
        }
    }
}

} // namespace jericho
