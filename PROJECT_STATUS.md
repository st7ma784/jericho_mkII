# Jericho Mk II - Project Status and Roadmap

## Overview

**Jericho Mk II** is a complete rewrite of JERICHO from the ground up as a CUDA-native, GPU-accelerated hybrid PIC-MHD code. This document tracks implementation status and provides a roadmap for completion.

**Status:** 🚧 **Foundation Complete** - Core architecture implemented, full physics implementation in progress

## What's Been Created ✅

### Core Architecture

#### 1. **Project Structure** ✅
```
jericho_mkII/
├── src/              # CPU host code
├── cuda/             # GPU device kernels
├── include/          # Header files
├── docs/             # Sphinx documentation
├── examples/         # Example configurations
├── tests/            # Unit tests (structure)
├── scripts/          # Utility scripts
├── .github/workflows/  # CI/CD pipelines
├── CMakeLists.txt    # Modern CMake build system
└── README.md         # Comprehensive project README
```

#### 2. **SoA Particle System** ✅
- **File:** `include/particle_buffer.h`
- **Features:**
  - GPU-optimized Structure of Arrays layout
  - Coalesced memory access (10-50x faster than AoS)
  - Dynamic particle management (add/remove efficiently)
  - Unified memory support
  - Comprehensive documentation

#### 3. **CUDA Kernels** ✅
- **File:** `cuda/particles.cu`
- **Implemented:**
  - ✅ Particle push (Boris integrator)
  - ✅ Field interpolation (bilinear, device function)
  - ✅ Particle-to-grid (P2G) scatter
  - ✅ Atomic operations for thread-safe P2G
  - ✅ Optimized for coalesced memory access

#### 4. **Boundary Conditions** ✅
- **File:** `cuda/boundaries.cu`
- **Implemented:**
  - ✅ Periodic boundaries
  - ✅ Outflow (particle removal)
  - ✅ Reflecting boundaries
  - ✅ Inflow (particle injection with Maxwellian distribution)
  - ✅ Mixed boundary support
  - ✅ GPU-parallel boundary checking

#### 5. **Build System** ✅
- **File:** `CMakeLists.txt`
- **Features:**
  - Modern CMake (3.18+)
  - Auto-detection of GPU architecture
  - CUDA-aware MPI support
  - Separate debug/release configurations
  - Testing framework integration
  - Installation targets

#### 6. **Documentation System** ✅
- **Sphinx configuration:** `docs/conf.py`
- **Features:**
  - Auto-generated API documentation (Breathe/Doxygen)
  - Read the Docs theme
  - Math support (MathJax)
  - C++/CUDA code documentation
  - Professional documentation structure

#### 7. **CI/CD Pipelines** ✅
- **GitHub Actions:**
  - `.github/workflows/docs.yml` - Auto-build and deploy docs
  - `.github/workflows/build.yml` - Build and test on push
  - Automatic deployment to GitHub Pages

#### 8. **Example Configurations** ✅
- **File:** `examples/reconnection.toml`
- **Features:**
  - Complete TOML configuration example
  - All parameters documented
  - Ready-to-run magnetic reconnection setup

---

## What Needs Implementation 🚧

### High Priority (Core Functionality)

#### 1. **Field Solvers** 🚧
**Status:** Not started
**Files to create:**
- `cuda/fields.cu` - Field update kernels
- `include/field_arrays.h` - Field data structures

**Required kernels:**
- [ ] Magnetic field advance (Faraday's law)
  - Predictor-corrector method
  - Curl operator on GPU
- [ ] Electric field solver (Ohm's law + Hall term)
  - Generalized Ohm's law
  - Electron pressure gradient (optional)
- [ ] Current Advance Method (CAM)
  - Lambda and Gamma coefficient computation
  - Mixed current calculation

**Estimated time:** 2-3 days

#### 2. **MPI Manager** 🚧
**Status:** Not started
**Files to create:**
- `src/mpi_manager.cpp`
- `include/mpi_manager.h`

**Required functionality:**
- [ ] Domain decomposition (2D grid partitioning)
- [ ] Ghost cell exchange
  - Standard MPI for CPU arrays
  - CUDA-aware MPI for GPU arrays
- [ ] Particle migration between ranks
- [ ] Load balancing (optional)

**Estimated time:** 3-4 days

#### 3. **I/O System** 🚧
**Status:** Not started
**Files to create:**
- `src/io.cpp`
- `include/io.h`

**Required functionality:**
- [ ] HDF5 field output
- [ ] HDF5 particle output
- [ ] Checkpoint/restart capability
- [ ] Parallel I/O with MPI-IO
- [ ] NetCDF support (optional)

**Estimated time:** 2-3 days

#### 4. **Configuration System** 🚧
**Status:** Not started
**Files to create:**
- `src/config.cpp`
- `include/config.h`

**Required functionality:**
- [ ] TOML file parser (use toml11 library)
- [ ] Configuration validation
- [ ] Default parameter handling
- [ ] Command-line argument parsing

**Estimated time:** 1-2 days

#### 5. **Main Simulation Loop** 🚧
**Status:** Not started
**Files to create:**
- `src/main.cpp`

**Required functionality:**
- [ ] Initialize MPI and CUDA
- [ ] Load configuration
- [ ] Initialize particles and fields
- [ ] Main timestep loop:
  1. Particle push
  2. Boundary conditions
  3. Particle-to-grid
  4. Field advance (B and E)
  5. Output and diagnostics
- [ ] Cleanup and finalization

**Estimated time:** 2-3 days

---

### Medium Priority (Performance & Quality)

#### 6. **Performance Optimizations** ⏳
- [ ] Stream-based asynchronous execution
- [ ] Multi-GPU support (1 GPU per MPI rank)
- [ ] Kernel fusion opportunities
- [ ] Shared memory optimization for P2G
- [ ] Warp-level primitives for reductions

**Estimated time:** 1-2 weeks

#### 7. **Testing Suite** ⏳
**Files to create:**
- `tests/test_particle_buffer.cpp`
- `tests/test_boundaries.cpp`
- `tests/test_fields.cpp`
- `tests/test_mpi.cpp`

**Required tests:**
- [ ] Unit tests for each component
- [ ] Integration tests
- [ ] Regression tests (compare with Jericho v1)
- [ ] Performance benchmarks

**Estimated time:** 1 week

#### 8. **Diagnostics & Analysis** ⏳
**Files to create:**
- `src/diagnostics.cpp`
- `include/diagnostics.h`

**Features:**
- [ ] Energy conservation monitoring
- [ ] Momentum conservation
- [ ] Field divergence checks
- [ ] NaN/Inf detection
- [ ] Runtime performance profiling

**Estimated time:** 3-4 days

---

### Low Priority (Nice-to-Have)

#### 9. **Advanced Features** 💡
- [ ] Adaptive timestep
- [ ] Higher-order particle pushers
- [ ] Particle splitting/merging
- [ ] 3D geometry support
- [ ] GPU direct storage (GDS) for I/O

#### 10. **Visualization Tools** 💡
- [ ] Python analysis scripts
- [ ] Real-time visualization (OpenGL/Vulkan)
- [ ] ParaView/VisIt plugins
- [ ] Jupyter notebook examples

---

## Implementation Roadmap

### Phase 1: Core Physics (2-3 weeks) 🎯 **CURRENT PHASE**
**Goal:** Get basic simulation running on single GPU

1. **Week 1:**
   - ✅ Project structure and build system
   - ✅ SoA particle buffer
   - ✅ Particle kernels and boundaries
   - 🚧 Field solvers (Faraday, Ohm's law)

2. **Week 2:**
   - 🚧 Configuration system
   - 🚧 I/O system (basic HDF5)
   - 🚧 Main simulation loop

3. **Week 3:**
   - 🚧 Single-GPU testing
   - 🚧 Validation against Jericho v1
   - 🚧 Bug fixes and optimization

**Deliverable:** Working single-GPU simulation

---

### Phase 2: MPI Parallelism (1-2 weeks)
**Goal:** Scale to multiple GPUs/nodes

1. **Week 4:**
   - Implement MPI manager
   - Domain decomposition
   - Ghost cell exchange

2. **Week 5:**
   - Particle migration
   - CUDA-aware MPI optimization
   - Multi-GPU testing

**Deliverable:** Multi-GPU scaling demonstrated

---

### Phase 3: Testing & Documentation (1 week)
**Goal:** Production-ready code

1. **Week 6:**
   - Comprehensive test suite
   - Complete documentation
   - Example simulations
   - Performance benchmarking

**Deliverable:** Release-ready v2.0.0

---

### Phase 4: Advanced Features (ongoing)
**Goal:** Research capabilities

- Advanced physics models
- Visualization tools
- Performance tuning
- Community contributions

---

## Performance Targets

### Single GPU (NVIDIA V100)
- **100K particles, 128×128 grid:** < 1 minute (vs 14 min CPU)
- **1M particles, 256×256 grid:** < 10 minutes (vs 3.5 hours CPU)
- **10M particles, 512×512 grid:** < 2 hours (impossible on CPU)

### Multi-GPU Scaling
- **Strong scaling:** 80%+ efficiency up to 16 GPUs
- **Weak scaling:** 90%+ efficiency up to 64 GPUs

---

## File Completion Checklist

### Implemented Files ✅
- [x] `README.md` - Project overview
- [x] `CMakeLists.txt` - Build system
- [x] `include/particle_buffer.h` - SoA particle interface
- [x] `cuda/particles.cu` - Particle kernels
- [x] `cuda/boundaries.cu` - Boundary condition kernels
- [x] `docs/conf.py` - Sphinx configuration
- [x] `docs/index.rst` - Documentation index
- [x] `examples/reconnection.toml` - Example config
- [x] `.github/workflows/docs.yml` - Docs CI/CD
- [x] `.github/workflows/build.yml` - Build CI/CD

### In Progress 🚧
- [ ] `cuda/fields.cu` - Field update kernels
- [ ] `include/field_arrays.h` - Field data structures
- [ ] `src/mpi_manager.cpp` - MPI parallelism
- [ ] `src/io.cpp` - I/O system
- [ ] `src/config.cpp` - Configuration parser
- [ ] `src/main.cpp` - Main program

### Not Started ⏳
- [ ] `src/diagnostics.cpp`
- [ ] `tests/*.cpp` - Test suite
- [ ] `scripts/analyze.py` - Analysis tools
- [ ] Full documentation (getting_started.rst, etc.)

---

## Architecture Highlights

### Memory Layout Comparison

**Old Jericho (AoS):**
```
Particle 0: [x, y, vx, vy, weight, type, padding...]  64 bytes
Particle 1: [x, y, vx, vy, weight, type, padding...]  64 bytes
Particle 2: [x, y, vx, vy, weight, type, padding...]  64 bytes
...
```
- ❌ Scattered memory access
- ❌ No SIMD vectorization
- ❌ Poor GPU performance

**Jericho Mk II (SoA):**
```
x:      [x0, x1, x2, x3, ...]  Contiguous
y:      [y0, y1, y2, y3, ...]  Contiguous
vx:     [vx0, vx1, vx2, ...]   Contiguous
vy:     [vy0, vy1, vy2, ...]   Contiguous
...
```
- ✅ Coalesced memory access (10-50x faster)
- ✅ SIMD vectorization
- ✅ Optimal GPU performance

### Parallelism Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    MPI Domain Decomposition              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Rank 0     │  │   Rank 1     │  │   Rank N     │  │
│  │   GPU 0      │  │   GPU 1      │  │   GPU N      │  │
│  │──────────────│  │──────────────│  │──────────────│  │
│  │ Particles:   │  │ Particles:   │  │ Particles:   │  │
│  │ SoA on GPU   │  │ SoA on GPU   │  │ SoA on GPU   │  │
│  │──────────────│  │──────────────│  │──────────────│  │
│  │ Fields:      │  │ Fields:      │  │ Fields:      │  │
│  │ 2D on GPU    │  │ 2D on GPU    │  │ 2D on GPU    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                   CUDA-aware MPI                        │
│              (GPU-GPU direct transfer)                   │
└─────────────────────────────────────────────────────────┘
```

---

## Getting Started (Current State)

The project foundation is complete. To continue development:

1. **Clone and build:**
   ```bash
   cd /home/user/Documents/jericho/jericho_mkII
   mkdir build && cd build
   cmake ..
   # Note: Will fail until main.cpp is implemented
   ```

2. **Next implementation priority:**
   - Implement `cuda/fields.cu` (field update kernels)
   - Implement `src/config.cpp` (configuration parser)
   - Implement `src/main.cpp` (simulation loop)

3. **Estimated time to first working simulation:** 2-3 weeks

---

## Questions / Decisions Needed

1. **HDF5 vs NetCDF** for output?
   → Recommendation: HDF5 (better GPU integration)

2. **How to handle multi-species** on GPU?
   → Current approach: Separate type array, lookup tables

3. **Load balancing strategy?**
   → Defer to Phase 2, use static decomposition initially

4. **Particle splitting/merging?**
   → Phase 4 feature, not critical for v2.0

---

**Last Updated:** 2025-11-14
**Status:** Foundation complete, core physics in progress
**Next Milestone:** Working single-GPU simulation
**Target Date:** 3 weeks from now
