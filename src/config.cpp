/**
 * @file config.cpp
 * @brief Implementation of TOML configuration parser
 * @author Jericho Mk II Development Team
 */

#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace jericho {

// =============================================================================
// Helper functions for TOML parsing
// =============================================================================

static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return str.substr(start, end - start + 1);
}

static bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

static std::string strip_quotes(const std::string& str) {
    std::string s = trim(str);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// =============================================================================
// Simple TOML parser
// =============================================================================

class SimpleTomlParser {
public:
    std::map<std::string, std::string> data;
    std::vector<std::map<std::string, std::string>> species_list;

    void parse_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }

        std::string line;
        std::string current_section;
        bool in_species = false;
        std::map<std::string, std::string> current_species;

        while (std::getline(file, line)) {
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            // Section headers
            if (line[0] == '[') {
                // Save previous species if we were in one
                if (in_species && !current_species.empty()) {
                    species_list.push_back(current_species);
                    current_species.clear();
                }

                size_t end = line.find(']');
                if (end == std::string::npos) {
                    throw std::runtime_error("Invalid section header: " + line);
                }

                std::string section = trim(line.substr(1, end - 1));

                // Handle array of tables [[species]]
                if (line[1] == '[') {
                    in_species = true;
                    current_section = "";
                } else {
                    in_species = false;
                    current_section = section;
                }
                continue;
            }

            // Key-value pairs
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));

            // Remove inline comments
            size_t comment_pos = value.find('#');
            if (comment_pos != std::string::npos) {
                value = trim(value.substr(0, comment_pos));
            }

            // Store in appropriate location
            if (in_species) {
                current_species[key] = value;
            } else {
                std::string full_key = current_section.empty() ? key :
                                      current_section + "." + key;
                data[full_key] = value;
            }
        }

        // Save last species if needed
        if (in_species && !current_species.empty()) {
            species_list.push_back(current_species);
        }
    }

    std::string get_string(const std::string& key, const std::string& default_val = "") {
        auto it = data.find(key);
        return (it != data.end()) ? strip_quotes(it->second) : default_val;
    }

    double get_double(const std::string& key, double default_val = 0.0) {
        auto it = data.find(key);
        return (it != data.end()) ? std::stod(it->second) : default_val;
    }

    int get_int(const std::string& key, int default_val = 0) {
        auto it = data.find(key);
        return (it != data.end()) ? std::stoi(it->second) : default_val;
    }

    bool get_bool(const std::string& key, bool default_val = false) {
        auto it = data.find(key);
        if (it == data.end()) return default_val;
        std::string val = trim(it->second);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        return (val == "true" || val == "1");
    }
};

// =============================================================================
// Config implementation
// =============================================================================

Config::Config() {
    // Simulation defaults
    dt = 1e-9;
    n_steps = 1000;
    use_cam = true;
    cam_alpha = 0.5;

    // Grid defaults
    nx_global = 256;
    ny_global = 256;
    x_min = 0.0;
    x_max = 1.0;
    y_min = 0.0;
    y_max = 1.0;
    nghost = 2;

    // MPI defaults
    npx = 1;
    npy = 1;
    cuda_aware_mpi = false;

    // Field defaults
    magnetic_field_type = "uniform";
    B0 = 0.0;
    L = 10.0;
    Ex0 = 0.0;
    Ey0 = 0.0;
    Bz0 = 0.0;

    // Output defaults
    output_dir = "./output";
    field_cadence = 10;
    particle_cadence = 100;
    checkpoint_cadence = 1000;
    compress_output = true;
    output_particles = true;
    output_Ex = true;
    output_Ey = true;
    output_Bz = true;
    output_charge = true;
    output_current = true;
    output_flow_velocity = false;

    // Boundary defaults
    bc_left.type = "periodic";
    bc_right.type = "periodic";
    bc_bottom.type = "periodic";
    bc_top.type = "periodic";
}

void Config::load(const std::string& filename) {
    SimpleTomlParser parser;
    parser.parse_file(filename);

    // =========================================================================
    // Parse [simulation] section
    // =========================================================================

    dt = parser.get_double("simulation.dt", dt);
    n_steps = parser.get_int("simulation.n_steps", n_steps);
    use_cam = parser.get_bool("simulation.use_cam", use_cam);
    cam_alpha = parser.get_double("simulation.cam_alpha", cam_alpha);

    // =========================================================================
    // Parse [grid] section
    // =========================================================================

    nx_global = parser.get_int("grid.nx", nx_global);
    ny_global = parser.get_int("grid.ny", ny_global);
    x_min = parser.get_double("grid.x_min", x_min);
    x_max = parser.get_double("grid.x_max", x_max);
    y_min = parser.get_double("grid.y_min", y_min);
    y_max = parser.get_double("grid.y_max", y_max);
    nghost = parser.get_int("grid.nghost", nghost);

    // =========================================================================
    // Parse [mpi] section
    // =========================================================================

    npx = parser.get_int("mpi.npx", npx);
    npy = parser.get_int("mpi.npy", npy);
    cuda_aware_mpi = parser.get_bool("mpi.cuda_aware", cuda_aware_mpi);

    // =========================================================================
    // Parse [[species]] array
    // =========================================================================

    species.clear();
    for (const auto& sp_data : parser.species_list) {
        SpeciesConfig sp;

        auto get_sp_string = [&](const std::string& key, const std::string& def = "") {
            auto it = sp_data.find(key);
            return (it != sp_data.end()) ? strip_quotes(it->second) : def;
        };

        auto get_sp_double = [&](const std::string& key, double def = 0.0) {
            auto it = sp_data.find(key);
            return (it != sp_data.end()) ? std::stod(it->second) : def;
        };

        auto get_sp_size_t = [&](const std::string& key, size_t def = 0) {
            auto it = sp_data.find(key);
            return (it != sp_data.end()) ? static_cast<size_t>(std::stoul(it->second)) : def;
        };

        sp.name = get_sp_string("name", "unnamed");
        sp.charge = get_sp_double("charge", 1.602e-19);
        sp.mass = get_sp_double("mass", 1.673e-27);
        sp.density = get_sp_double("density", 1e15);
        sp.temperature = get_sp_double("temperature", 100.0);
        sp.drift_vx = get_sp_double("drift_vx", 0.0);
        sp.drift_vy = get_sp_double("drift_vy", 0.0);
        sp.particles_per_cell = get_sp_size_t("particles_per_cell", 100);

        species.push_back(sp);
    }

    // =========================================================================
    // Parse [boundaries] section
    // =========================================================================

    auto parse_boundary = [&](const std::string& prefix, BoundaryConfig& bc) {
        bc.type = parser.get_string(prefix + ".type", bc.type);
        bc.inflow_density = parser.get_double(prefix + ".inflow_density", 1e15);
        bc.inflow_temperature = parser.get_double(prefix + ".inflow_temperature", 100.0);
        bc.inflow_vx = parser.get_double(prefix + ".inflow_vx", 0.0);
        bc.inflow_vy = parser.get_double(prefix + ".inflow_vy", 0.0);
    };

    bc_left.type = parser.get_string("boundaries.left", "periodic");
    bc_right.type = parser.get_string("boundaries.right", "periodic");
    bc_bottom.type = parser.get_string("boundaries.bottom", "periodic");
    bc_top.type = parser.get_string("boundaries.top", "periodic");

    parse_boundary("boundaries.left", bc_left);
    parse_boundary("boundaries.right", bc_right);
    parse_boundary("boundaries.bottom", bc_bottom);
    parse_boundary("boundaries.top", bc_top);

    // =========================================================================
    // Parse [fields] section
    // =========================================================================

    magnetic_field_type = parser.get_string("fields.magnetic_field_type",
                                           magnetic_field_type);
    B0 = parser.get_double("fields.B0", B0);
    L = parser.get_double("fields.L", L);
    Ex0 = parser.get_double("fields.Ex0", Ex0);
    Ey0 = parser.get_double("fields.Ey0", Ey0);
    Bz0 = parser.get_double("fields.Bz0", Bz0);

    // =========================================================================
    // Parse [output] section
    // =========================================================================

    output_dir = parser.get_string("output.output_dir", output_dir);
    field_cadence = parser.get_int("output.field_cadence", field_cadence);
    particle_cadence = parser.get_int("output.particle_cadence", particle_cadence);
    checkpoint_cadence = parser.get_int("output.checkpoint_cadence", checkpoint_cadence);
    compress_output = parser.get_bool("output.compress_output", compress_output);
    output_particles = parser.get_bool("output.output_particles", output_particles);
    output_Ex = parser.get_bool("output.output_Ex", output_Ex);
    output_Ey = parser.get_bool("output.output_Ey", output_Ey);
    output_Bz = parser.get_bool("output.output_Bz", output_Bz);
    output_charge = parser.get_bool("output.output_charge", output_charge);
    output_current = parser.get_bool("output.output_current", output_current);
    output_flow_velocity = parser.get_bool("output.output_flow_velocity",
                                          output_flow_velocity);

    // Validate configuration
    validate();
}

void Config::validate() const {
    // Check grid dimensions
    if (nx_global <= 0 || ny_global <= 0) {
        throw std::runtime_error("Grid dimensions must be positive");
    }

    // Check domain bounds
    if (x_max <= x_min || y_max <= y_min) {
        throw std::runtime_error("Invalid domain bounds");
    }

    // Check MPI decomposition
    if (npx <= 0 || npy <= 0) {
        throw std::runtime_error("MPI process counts must be positive");
    }

    // Check timestep
    if (dt <= 0.0) {
        throw std::runtime_error("Timestep must be positive");
    }

    // Check CAM parameter
    if (cam_alpha < 0.0 || cam_alpha > 1.0) {
        throw std::runtime_error("CAM alpha must be in [0, 1]");
    }

    // Check species
    if (species.empty()) {
        throw std::runtime_error("At least one particle species required");
    }

    for (const auto& sp : species) {
        if (sp.mass <= 0.0) {
            throw std::runtime_error("Species mass must be positive: " + sp.name);
        }
        if (sp.density <= 0.0) {
            throw std::runtime_error("Species density must be positive: " + sp.name);
        }
    }
}

void Config::print() const {
    std::cout << "\n=== Configuration ===" << std::endl;

    std::cout << "\n[Simulation]" << std::endl;
    std::cout << "  dt = " << dt << " s" << std::endl;
    std::cout << "  n_steps = " << n_steps << std::endl;
    std::cout << "  use_cam = " << (use_cam ? "true" : "false") << std::endl;
    std::cout << "  cam_alpha = " << cam_alpha << std::endl;

    std::cout << "\n[Grid]" << std::endl;
    std::cout << "  nx x ny = " << nx_global << " x " << ny_global << std::endl;
    std::cout << "  domain = [" << x_min << ", " << x_max << "] x ["
             << y_min << ", " << y_max << "]" << std::endl;
    std::cout << "  dx = " << get_dx() << " m" << std::endl;
    std::cout << "  dy = " << get_dy() << " m" << std::endl;

    std::cout << "\n[MPI]" << std::endl;
    std::cout << "  npx x npy = " << npx << " x " << npy
             << " (total: " << get_mpi_size() << ")" << std::endl;
    std::cout << "  CUDA-aware = " << (cuda_aware_mpi ? "yes" : "no") << std::endl;

    std::cout << "\n[Species] (" << species.size() << " total)" << std::endl;
    for (size_t i = 0; i < species.size(); i++) {
        const auto& sp = species[i];
        std::cout << "  [" << i << "] " << sp.name << std::endl;
        std::cout << "      q = " << sp.charge << " C" << std::endl;
        std::cout << "      m = " << sp.mass << " kg" << std::endl;
        std::cout << "      n = " << sp.density << " m^-3" << std::endl;
        std::cout << "      T = " << sp.temperature << " eV" << std::endl;
        std::cout << "      PPC = " << sp.particles_per_cell << std::endl;
    }

    std::cout << "\n[Boundaries]" << std::endl;
    std::cout << "  left   = " << bc_left.type << std::endl;
    std::cout << "  right  = " << bc_right.type << std::endl;
    std::cout << "  bottom = " << bc_bottom.type << std::endl;
    std::cout << "  top    = " << bc_top.type << std::endl;

    std::cout << "\n[Fields]" << std::endl;
    std::cout << "  type = " << magnetic_field_type << std::endl;
    std::cout << "  B0 = " << B0 << " T" << std::endl;
    if (magnetic_field_type == "harris_sheet") {
        std::cout << "  L = " << L << std::endl;
    }

    std::cout << "\n[Output]" << std::endl;
    std::cout << "  directory = " << output_dir << std::endl;
    std::cout << "  field_cadence = " << field_cadence << std::endl;
    std::cout << "  particle_cadence = " << particle_cadence << std::endl;
    std::cout << "  checkpoint_cadence = " << checkpoint_cadence << std::endl;

    std::cout << "\n=====================\n" << std::endl;
}

} // namespace jericho
