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
│   └── io.cpp
├── cuda/             # GPU device code
│   ├── particles.cu  # Particle kernels
│   ├── fields.cu     # Field update kernels
│   ├── boundaries.cu # Boundary condition kernels
│   └── p2g.cu        # Particle-to-grid kernels
├── include/          # Header files
│   ├── particle_buffer.h
│   ├── field_arrays.h
│   ├── mpi_manager.h
│   └── config.h
├── docs/             # Sphinx documentation
│   ├── source/
│   └── conf.py
├── tests/            # Unit tests
├── examples/         # Example configs
└── scripts/          # Utilities
```

## Documentation

Full documentation available at: https://st7ma784.github.io/jericho_mkII/

- **[Getting Started](docs/source/getting_started.rst)** - Installation and first run
- **[User Guide](docs/source/user_guide.rst)** - Configuration and usage
- **[Developer Guide](docs/source/developer_guide.rst)** - Contributing and internals
- **[API Reference](docs/source/api.rst)** - Code documentation
- **[Performance Tuning](docs/source/performance.rst)** - Optimization tips

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
