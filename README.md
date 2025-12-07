# Jericho Mk II - GPU-Accelerated Hybrid PIC-MHD

[![Documentation Status](https://github.com/st7ma784/jericho_mkII/workflows/docs/badge.svg)](https://st7ma784.github.io/jericho_mkII/)
[![Build Status](https://github.com/st7ma784/jericho_mkII/workflows/build/badge.svg)](https://github.com/st7ma784/jericho_mkII/actions)

**Next-generation hybrid Particle-in-Cell / Magnetohydrodynamic plasma simulation code**

Jericho Mk II is a complete rewrite of the JERICHO plasma simulation code, designed from the ground up for modern GPU+MPI HPC architectures. It combines kinetic ion treatment with fluid electron modeling for efficient simulation of magnetospheric plasma dynamics. We have implemented improved checks to ensure energy conservation and model other interesting factors and phenomenom.

## Key Features

### 🚀 Performance
- **CUDA-native implementation** - All compute kernels optimized for NVIDIA GPUs
- **Structure of Arrays (SoA) layout** - Coalesced memory access for 10-50x GPU speedup
- **MPI+CUDA hybrid parallelism** - Scale across multiple nodes and GPUs
- **Asynchronous computation** - Overlap compute and communication
- **Zero-copy transfers** - Pinned memory and CUDA-aware MPI

### 🔬 Physics
- **Hybrid PIC-MHD** - Kinetic ions + fluid electrons
- **Boris particle pusher** - Energy-conserving velocity integrator
- **Current Advance Method (CAM)** - Improved numerical stability
- **Ohm's Law with Hall term** - Full electromagnetic coupling
- **Multiple ion species** - Arbitrary number of species with different q/m ratios

### 🎯 Boundary Conditions
- **Periodic** - Wrap-around in x and/or y
- **Inflow** - Inject particles at boundaries
- **Outflow** - Remove particles leaving domain
- **Reflecting** - Elastic reflection
- **Mixed** - Different conditions per boundary

### 📊 Modern Architecture
- **Clean separation** - CPU host code, GPU device code clearly separated
- **Type-safe** - Modern C++17 with strong typing
- **Documented** - Sphinx documentation with auto-build
- **Tested** - Unit tests and integration tests
- **Reproducible** - CMake build system, Docker containers

## Quick Start

### Prerequisites
```bash
# CUDA Toolkit (tested with 11.0+)
# MPI (OpenMPI or MPICH with CUDA-aware support)
# CMake 3.18+
# C++17 compiler (gcc 9+, clang 10+)
```

### Build
```bash
git clone https://github.com/yourusername/jericho_mkII.git
cd jericho_mkII
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..  # Set to your GPU arch
make -j
```

### Run
```bash
# Single GPU
./jericho_mkII config.toml

# Multiple GPUs (1 GPU per MPI rank)
mpirun -np 4 ./jericho_mkII config.toml

# Multi-node (CUDA-aware MPI required)
mpirun -np 16 --hostfile hosts ./jericho_mkII config.toml
```

## 🎨 Real-Time Web Visualization

Jericho Mk II includes a **real-time web interface** for monitoring simulations as they run!

![Web Interface](docs/source/_static/web_interface_preview.png)

### Features

- 🌊 **Electromagnetic Fields** - Live heatmaps of Ex, Ey, Bz with vector overlays
- ⚛️ **Particle Distribution** - Real-time particle positions colored by type or velocity
- 📊 **Energy Diagnostics** - Time-series plots of energy conservation
- 🔄 **Phase Space** - Velocity distribution analysis (Vx, Vy)
- ⚡ **Current Density** - Electric current flow visualization (|J|, Jx, Jy, Jz)
- 🌊 **Plasma Flow** - Bulk velocity field with streamlines or vector arrows
- ⚛️ **Charge Density** - Net charge distribution with contour lines
- 🌡️ **Pressure & Temperature** - Thermal, magnetic pressure, and plasma β
- 🔲 **Boundary Fluxes** - Particle inflow/outflow at domain boundaries

### Quick Start

```bash
# Terminal 1: Run simulation
./jericho_mkII config.toml

# Terminal 2: Start web server
cd web
pip install -r requirements.txt
python server.py --output-dir ../output

# Open browser to http://localhost:8888
```

The web interface automatically streams new data as HDF5 files are written!

### Usage

```bash
# Basic usage
cd web
python server.py --output-dir ../output --port 8888

# Monitor specific simulation
python server.py --output-dir ../outputs/reconnection_run_01

# Custom host (for remote access)
python server.py --host 0.0.0.0 --port 8888
```

### Interactive Controls

**Electromagnetic Fields Panel:**
- Switch between Ex, Ey, Bz, |E|, |B|, current density, charge density
- Toggle vector field overlay
- Real-time colorbar scaling

**Particle Distribution Panel:**
- Color by type (ions/electrons) or velocity magnitude
- Enable motion trails for particle tracking
- Automatic downsampling for large particle counts

**Current Density Panel:**
- View |J| magnitude or individual components (Jx, Jy, Jz)
- Identifies current sheets and reconnection regions

**Plasma Flow Panel:**
- Bulk velocity field visualization
- Toggle between vector arrows and streamlines
- Switch between flow speed and vorticity (∇ × v)

**Charge Density Panel:**
- Net charge distribution (ρ = ions - electrons)
- Toggle contour lines at ρ = 0
- Diagnose charge separation

**Pressure Panel:**
- Thermal pressure (P = nkT)
- Magnetic pressure (B²/2μ₀)
- Total pressure and plasma β ratio

**Boundary Conditions Panel:**
- Real-time particle flux at boundaries
- Color-coded: Green (inflow), Red (outflow), Cyan (periodic)
- Particle counts crossing each boundary

### Performance

The web server automatically optimizes for browser display:
- Field grids downsampled to 512×512 maximum
- Particles limited to 5,000 displayed (randomly sampled from full dataset)
- Update rate: ~2 Hz (configurable)
- WebSocket streaming for low latency

### Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ Mobile (limited - large data transfers)

### Physics Interpretation

See [`web/VISUALIZATION_GUIDE.md`](web/VISUALIZATION_GUIDE.md) for detailed explanation of:
- How to read each visualization
- Physical interpretation of features
- Identifying magnetic reconnection signatures
- Understanding phase space distributions
- Energy conservation validation
- Current sheet diagnostics

### Demo Mode

Try the interface without running a simulation:

```bash
cd web
python -m http.server 8889
# Open http://localhost:8889/demo.html
```

The demo shows synthetic reconnection data with all visualization features.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MPI Domain Decomposition              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Rank 0     │  │   Rank 1     │  │   Rank N     │  │
│  │              │  │              │  │              │  │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │
│  │  │ GPU 0  │  │  │  │ GPU 1  │  │  │  │ GPU N  │  │  │
│  │  │────────│  │  │  │────────│  │  │  │────────│  │  │
│  │  │Particle│  │  │  │Particle│  │  │  │Particle│  │  │
│  │  │ Buffer │  │  │  │ Buffer │  │  │  │ Buffer │  │  │
│  │  │  (SoA) │  │  │  │  (SoA) │  │  │  │  (SoA) │  │  │
│  │  │────────│  │  │  │────────│  │  │  │────────│  │  │
│  │  │ Fields │  │  │  │ Fields │  │  │  │ Fields │  │  │
│  │  │  (2.5D) │  │  │  │  (2.5D) │  │  │  │  (2.5D) │  │  │
│  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                   CUDA-aware MPI                        │
└─────────────────────────────────────────────────────────┘
```

### Data Layout: Structure of Arrays (SoA)

**Traditional AoS (slow on GPU):**
```cpp
struct Particle { double x, y, vx, vy, weight; uint8_t type; };
std::vector<Particle> particles;  // Scattered memory access
```

**Jericho Mk II SoA (fast on GPU):**
```cpp
struct ParticleBuffer {
    double* x;       // Contiguous array on GPU
    double* y;       // Coalesced memory access
    double* vx;      // Perfect for SIMD/GPU
    double* vy;
    double* weight;
    uint8_t* type;
    bool* active;    // Dynamic particle management
    size_t capacity;
    size_t count;
};
```

**Performance Impact:**
- ✅ Coalesced memory access → 10-50x faster on GPU
- ✅ SIMD vectorization → 4-8x faster on CPU
- ✅ Better cache utilization
- ✅ Enables GPU kernel fusion

## Performance Comparison

| Configuration | Jericho (CPU) | Jericho Mk II (GPU) | Speedup |
|--------------|---------------|---------------------|---------|
| 100K particles, 128×128 grid | 14 min | ~30 sec | **28x** |
| 1M particles, 256×256 grid | 3.5 hours | ~8 min | **26x** |
| 10M particles, 512×512 grid | N/A (OOM) | ~1.5 hours | ∞ |

*Measured on V100 GPU vs dual Xeon CPU (24 cores)*

## Project Structure

```
jericho_mkII/
├── src/              # CPU host code
│   ├── main.cpp
│   ├── config.cpp
│   └── io_manager.cpp
├── cuda/             # GPU device code
│   ├── particles.cu  # Particle kernels
│   ├── fields.cu     # Field update kernels
│   └── boundaries.cu # Boundary condition kernels
├── include/          # Header files
│   ├── particle_buffer.h
│   ├── field_arrays.h
│   ├── mpi_manager.h
│   └── config.h
├── web/              # Real-time web visualization
│   ├── server.py     # WebSocket server
│   ├── index.html    # Main interface
│   ├── demo.html     # Standalone demo
│   ├── static/
│   │   └── visualization.js
│   ├── requirements.txt
│   ├── README.md
│   └── VISUALIZATION_GUIDE.md
├── docs/             # Sphinx documentation
│   ├── *.rst         # Documentation files
│   ├── api/          # API reference
│   ├── conf.py
│   └── Makefile
├── tests/            # Unit tests
├── examples/         # Example configs
│   ├── reconnection.toml
│   └── minimal_test.toml
├── inputs/           # Production configs
├── outputs/          # Simulation results
└── scripts/          # Utilities
```

## Documentation

Full documentation available at: https://st7ma784.github.io/jericho_mkII/

- **[Getting Started](docs/getting_started.rst)** - Installation and first run
- **[Configuration Guide](docs/configuration.rst)** - Complete TOML reference
- **[Running Simulations](docs/running_simulations.rst)** - Usage and examples
- **[Architecture](docs/architecture.rst)** - Physics and CS implementation
- **[CUDA Kernels](docs/cuda_kernels.rst)** - GPU optimization details
- **[MPI Parallelism](docs/mpi_parallelism.rst)** - Multi-GPU scaling
- **[Performance Tuning](docs/performance_tuning.rst)** - Optimization guide
- **[Output Formats](docs/output_formats.rst)** - HDF5 file structure and analysis
- **[Web Visualization](web/VISUALIZATION_GUIDE.md)** - Real-time monitoring guide
- **[API Reference](docs/api/)** - Code documentation
- **[Troubleshooting](docs/troubleshooting.rst)** - Common issues and solutions

## Citation

If you use Jericho Mk II in your research, please cite:

```bibtex
@software{jericho_mkII,
  author = {Wiggs, Josh and Arridge, Chris and Greenyer, George and Mander, Steve},
  title = {Jericho Mk II: GPU-Accelerated Hybrid PIC-MHD Code},
  year = {2025},
  url = {https://github.com/st7ma784/jericho_mkII}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

- Original Jericho code by J. Wiggs, C. Arridge, G. Greenyer
- Lancaster University Physics Department
- STFC DiRAC HPC Facility

## Contact

- Josh Wiggs - j.wiggs@lancaster.ac.uk
- Chris Arridge - c.arridge@lancaster.ac.uk
- GitHub Issues: https://github.com/st7ma784/jericho_mkII/issues
