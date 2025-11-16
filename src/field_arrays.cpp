/**
 * @file field_arrays.cpp
 * @brief Implementation of GPU-resident electromagnetic field arrays
 * @author Jericho Mk II Development Team
 */

#include "field_arrays.h"

#include "platform.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace jericho {

// =============================================================================
// Helper macros for CUDA error checking
// =============================================================================

// Simplified implementations for CPU build
FieldArrays::FieldArrays(int nx_local, int ny_local, int nghost, double dx, double dy, double x_min,
                         double y_min, int device_id)
    : nx_local(nx_local), ny_local(ny_local), nghost(nghost), dx(dx), dy(dy), x_min(x_min),
      y_min(y_min), device_id(device_id) {
    nx = nx_local + 2 * nghost;
    ny = ny_local + 2 * nghost;

    Ex = Ey = Bz = nullptr;
    charge_density = Jx = Jy = nullptr;
    Ux = Uy = nullptr;
    Lambda = Gamma_x = Gamma_y = nullptr;
    Ex_back = Ey_back = Bz_back = nullptr;
    tmp1 = tmp2 = tmp3 = nullptr;

    allocate();
}

FieldArrays::~FieldArrays() {
    destroy();
}

FieldArrays::FieldArrays(FieldArrays&& other) noexcept
    : nx(other.nx), ny(other.ny), nx_local(other.nx_local), ny_local(other.ny_local),
      nghost(other.nghost), dx(other.dx), dy(other.dy), x_min(other.x_min), y_min(other.y_min),
      device_id(other.device_id) {
    Ex = other.Ex;
    other.Ex = nullptr;
    Ey = other.Ey;
    other.Ey = nullptr;
    Bz = other.Bz;
    other.Bz = nullptr;
    charge_density = other.charge_density;
    other.charge_density = nullptr;
    Jx = other.Jx;
    other.Jx = nullptr;
    Jy = other.Jy;
    other.Jy = nullptr;
    Ux = other.Ux;
    other.Ux = nullptr;
    Uy = other.Uy;
    other.Uy = nullptr;
    Lambda = other.Lambda;
    other.Lambda = nullptr;
    Gamma_x = other.Gamma_x;
    other.Gamma_x = nullptr;
    Gamma_y = other.Gamma_y;
    other.Gamma_y = nullptr;
    Ex_back = other.Ex_back;
    other.Ex_back = nullptr;
    Ey_back = other.Ey_back;
    other.Ey_back = nullptr;
    Bz_back = other.Bz_back;
    other.Bz_back = nullptr;
    tmp1 = other.tmp1;
    other.tmp1 = nullptr;
    tmp2 = other.tmp2;
    other.tmp2 = nullptr;
    tmp3 = other.tmp3;
    other.tmp3 = nullptr;
}

FieldArrays& FieldArrays::operator=(FieldArrays&& other) noexcept {
    if (this != &other) {
        destroy();
        nx = other.nx;
        ny = other.ny;
        nx_local = other.nx_local;
        ny_local = other.ny_local;
        nghost = other.nghost;
        dx = other.dx;
        dy = other.dy;
        x_min = other.x_min;
        y_min = other.y_min;
        device_id = other.device_id;

        Ex = other.Ex;
        other.Ex = nullptr;
        Ey = other.Ey;
        other.Ey = nullptr;
        Bz = other.Bz;
        other.Bz = nullptr;
        charge_density = other.charge_density;
        other.charge_density = nullptr;
        Jx = other.Jx;
        other.Jx = nullptr;
        Jy = other.Jy;
        other.Jy = nullptr;
        Ux = other.Ux;
        other.Ux = nullptr;
        Uy = other.Uy;
        other.Uy = nullptr;
        Lambda = other.Lambda;
        other.Lambda = nullptr;
        Gamma_x = other.Gamma_x;
        other.Gamma_x = nullptr;
        Gamma_y = other.Gamma_y;
        other.Gamma_y = nullptr;
        Ex_back = other.Ex_back;
        other.Ex_back = nullptr;
        Ey_back = other.Ey_back;
        other.Ey_back = nullptr;
        Bz_back = other.Bz_back;
        other.Bz_back = nullptr;
        tmp1 = other.tmp1;
        other.tmp1 = nullptr;
        tmp2 = other.tmp2;
        other.tmp2 = nullptr;
        tmp3 = other.tmp3;
        other.tmp3 = nullptr;
    }
    return *this;
}

void FieldArrays::allocate() {
    size_t n_points = get_total_points();
    size_t bytes = n_points * sizeof(double);

    Ex = (double*)malloc(bytes);
    Ey = (double*)malloc(bytes);
    Bz = (double*)malloc(bytes);
    charge_density = (double*)malloc(bytes);
    Jx = (double*)malloc(bytes);
    Jy = (double*)malloc(bytes);
    Ux = (double*)malloc(bytes);
    Uy = (double*)malloc(bytes);
    Lambda = (double*)malloc(bytes);
    Gamma_x = (double*)malloc(bytes);
    Gamma_y = (double*)malloc(bytes);
    Ex_back = (double*)malloc(bytes);
    Ey_back = (double*)malloc(bytes);
    Bz_back = (double*)malloc(bytes);
    tmp1 = (double*)malloc(bytes);
    tmp2 = (double*)malloc(bytes);
    tmp3 = (double*)malloc(bytes);

    zero_fields();
}

void FieldArrays::destroy() {
    if (Ex)
        free(Ex);
    if (Ey)
        free(Ey);
    if (Bz)
        free(Bz);
    if (charge_density)
        free(charge_density);
    if (Jx)
        free(Jx);
    if (Jy)
        free(Jy);
    if (Ux)
        free(Ux);
    if (Uy)
        free(Uy);
    if (Lambda)
        free(Lambda);
    if (Gamma_x)
        free(Gamma_x);
    if (Gamma_y)
        free(Gamma_y);
    if (Ex_back)
        free(Ex_back);
    if (Ey_back)
        free(Ey_back);
    if (Bz_back)
        free(Bz_back);
    if (tmp1)
        free(tmp1);
    if (tmp2)
        free(tmp2);
    if (tmp3)
        free(tmp3);

    Ex = Ey = Bz = nullptr;
    charge_density = Jx = Jy = nullptr;
    Ux = Uy = nullptr;
    Lambda = Gamma_x = Gamma_y = nullptr;
    Ex_back = Ey_back = Bz_back = nullptr;
    tmp1 = tmp2 = tmp3 = nullptr;
}

void FieldArrays::zero_fields() {
    size_t bytes = get_total_points() * sizeof(double);
    memset(Ex, 0, bytes);
    memset(Ey, 0, bytes);
    memset(Bz, 0, bytes);
    zero_particle_quantities();
    memset(Lambda, 0, bytes);
    memset(Gamma_x, 0, bytes);
    memset(Gamma_y, 0, bytes);
    memset(Ex_back, 0, bytes);
    memset(Ey_back, 0, bytes);
    memset(Bz_back, 0, bytes);
    memset(tmp1, 0, bytes);
    memset(tmp2, 0, bytes);
    memset(tmp3, 0, bytes);
}

void FieldArrays::zero_particle_quantities() {
    size_t bytes = get_total_points() * sizeof(double);
    memset(charge_density, 0, bytes);
    memset(Jx, 0, bytes);
    memset(Jy, 0, bytes);
    memset(Ux, 0, bytes);
    memset(Uy, 0, bytes);
}

size_t FieldArrays::get_memory_bytes() const {
    size_t n_arrays = 18;
    return n_arrays * get_total_points() * sizeof(double);
}

void FieldArrays::copy_to_host(double* h_Ex, double* h_Ey, double* h_Bz) const {
    size_t bytes = get_total_points() * sizeof(double);
    memcpy(h_Ex, Ex, bytes);
    memcpy(h_Ey, Ey, bytes);
    memcpy(h_Bz, Bz, bytes);
}

void FieldArrays::copy_from_host(const double* h_Ex, const double* h_Ey, const double* h_Bz) {
    size_t bytes = get_total_points() * sizeof(double);
    memcpy(Ex, h_Ex, bytes);
    memcpy(Ey, h_Ey, bytes);
    memcpy(Bz, h_Bz, bytes);
}

void FieldArrays::initialize_harris_sheet(double B0, double L) {
    for (int iy = 0; iy < ny; iy++) {
        double y = y_min + iy * dy;
        double bz = B0 * tanh(y / L);
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            Bz[idx] = bz;
            Bz_back[idx] = bz;
        }
    }
}

void FieldArrays::initialize_uniform_field(double Bz0) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            Bz[idx] = Bz0;
            Bz_back[idx] = Bz0;
        }
    }
}

void FieldArrays::set_background_fields(double Ex0, double Ey0, double Bz0) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;
            Ex_back[idx] = Ex0;
            Ey_back[idx] = Ey0;
            Bz_back[idx] = Bz0;
        }
    }
}

} // namespace jericho
