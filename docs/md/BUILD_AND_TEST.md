# Jericho Mk II - Build and Test Guide

## Quick Start (TL;DR)

```bash
cd /home/user/Documents/jericho/jericho_mkII
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release ..
make -j
./jericho_mkII ../examples/reconnection.toml
```

---

## Prerequisites

### System Requirements

- **OS:** Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **GPU:** NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** 10 GB for build + outputs

### Software Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| CMake | 3.18+ | Build system |
| GCC/G++ | 9.0+ | C++ compiler |
| CUDA Toolkit | 11.0+ | GPU programming |
| OpenMPI/MPICH | 3.0+ | Multi-GPU communication |
| HDF5 | 1.10+ | Parallel I/O (parallel build) |

---

## Installation Guide

### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y cmake build-essential git

# Install MPI (choose one)
sudo apt-get install -y libopenmpi-dev openmpi-bin  # OpenMPI
# OR
sudo apt-get install -y mpich libmpich-dev          # MPICH

# Install HDF5 with parallel support
sudo apt-get install -y libhdf5-openmpi-dev hdf5-tools

# CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads
# Or use package manager:
sudo apt-get install -y nvidia-cuda-toolkit
```

### CentOS/RHEL

```bash
# Enable EPEL repository
sudo yum install -y epel-release

# Install build tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git

# Install MPI
sudo yum install -y openmpi openmpi-devel
module load mpi/openmpi-x86_64

# Install HDF5
sudo yum install -y hdf5-openmpi hdf5-openmpi-devel

# CUDA Toolkit
# Download from NVIDIA website
```

### HPC Clusters (Module System)

```bash
# Load required modules
module purge
module load cmake/3.20
module load cuda/11.8
module load openmpi/4.1.1
module load hdf5/1.12.1  # Make sure it's parallel build

# Check loaded modules
module list
```

---

## Build Instructions

### 1. Configure Build

```bash
cd /home/user/Documents/jericho/jericho_mkII
mkdir build && cd build
```

#### Basic Configuration (Auto-detect GPU)

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

#### Specify GPU Architecture

Find your GPU compute capability:
- RTX 3090: 86
- RTX 2080 Ti: 75
- V100: 70
- A100: 80

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release ..
```

#### Enable CUDA-Aware MPI

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA_AWARE_MPI=ON ..
```

**Note:** Requires MPI built with CUDA support. Check with:
```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

#### Debug Build

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Debug \
      -DENABLE_PROFILING=ON ..
```

### 2. Compile

```bash
# Parallel build (use all cores)
make -j$(nproc)

# Or specify number of cores
make -j8

# Verbose output (for debugging)
make VERBOSE=1
```

### 3. Verify Build

```bash
# Check executable exists
ls -lh jericho_mkII

# Check dependencies
ldd jericho_mkII

# Check CUDA device
nvidia-smi
```

Expected output:
```
jericho_mkII: linked with libmpi, libhdf5, libcudart
nvidia-smi: shows available GPUs
```

---

## Running Simulations

### Single-GPU Test

```bash
# Run with example configuration
./jericho_mkII ../examples/reconnection.toml
```

**Expected output:**
```
=== Configuration ===
...
=== MPI Topology ===
Total processes: 1
...
=== Starting simulation ===
Step 10/10000 | t = 1e-8 s | particles = 131072
Step 20/10000 | t = 2e-8 s | particles = 131072
...
Written fields to ./output/fields_step_000010.h5
...
=== Simulation complete ===
```

### Multi-GPU Test (Single Node)

```bash
# 2 GPUs
mpirun -np 2 ./jericho_mkII ../examples/reconnection.toml

# 4 GPUs
mpirun -np 4 ./jericho_mkII ../examples/reconnection.toml
```

### Multi-Node HPC

Create a SLURM job script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=jericho_mkII
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load cuda/11.8
module load openmpi/4.1.1
module load hdf5/1.12.1

cd $SLURM_SUBMIT_DIR

mpirun -np 16 ./jericho_mkII config.toml
```

Submit:
```bash
sbatch job.slurm
```

---

## Verification Tests

### 1. Compilation Test

```bash
cd build
make clean
make -j$(nproc) 2>&1 | tee build.log
grep -i "error\|warning" build.log
```

✅ **Pass:** No errors, warnings acceptable

### 2. MPI Topology Test

```bash
mpirun -np 4 ./jericho_mkII ../examples/reconnection.toml | head -30
```

✅ **Pass:** Shows correct MPI grid (e.g., 2×2 for np=4)

### 3. GPU Memory Test

```bash
# Run while monitoring GPU
nvidia-smi dmon -s u &
./jericho_mkII ../examples/reconnection.toml
```

✅ **Pass:** GPU memory usage increases, then stabilizes

### 4. Output File Test

```bash
./jericho_mkII ../examples/reconnection.toml
ls -lh output/
h5dump -H output/fields_step_000000.h5
```

✅ **Pass:**
- Files created in `output/`
- HDF5 files have correct structure
- Diagnostics CSV readable

### 5. Field Conservation Test

```python
import h5py
import numpy as np

# Check field values are reasonable
f = h5py.File('output/fields_step_000010.h5', 'r')
Ex = f['Ex'][:]
print(f"Ex range: [{Ex.min()}, {Ex.max()}]")
assert not np.any(np.isnan(Ex)), "NaN detected!"
assert not np.any(np.isinf(Ex)), "Inf detected!"
```

✅ **Pass:** No NaN/Inf, values in expected range

### 6. Scaling Test

```bash
# Test different process counts
for np in 1 2 4 8; do
    echo "Testing with $np processes..."
    time mpirun -np $np ./jericho_mkII ../examples/reconnection.toml
done
```

✅ **Pass:** Runtime decreases with more processes (weak scaling)

---

## Performance Profiling

### NVIDIA Nsight Systems

```bash
# Profile GPU performance
nsys profile --stats=true \
    -o jericho_profile.qdrep \
    ./jericho_mkII ../examples/reconnection.toml

# View profile
nsys-ui jericho_profile.qdrep
```

**Look for:**
- Kernel execution time
- Memory transfer overhead
- GPU utilization (should be >50%)

### NVIDIA Nsight Compute

```bash
# Detailed kernel analysis
ncu --set full \
    -o jericho_kernel_profile \
    ./jericho_mkII ../examples/reconnection.toml

# View metrics
ncu-ui jericho_kernel_profile.ncu-rep
```

**Optimize for:**
- Memory bandwidth utilization (>70%)
- Occupancy (>50%)
- Coalesced memory access

### MPI Profiling

```bash
# Intel MPI Performance Snapshot
mpirun -np 4 -gtool "aps=./aps:4" ./jericho_mkII config.toml

# Or TAU profiling
export TAU_OPTIONS="-optVerbose -optCompInst"
mpirun -np 4 tau_exec ./jericho_mkII config.toml
```

---

## Debugging

### CUDA Errors

```bash
# Rebuild with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j

# Run with CUDA memcheck
cuda-memcheck ./jericho_mkII ../examples/reconnection.toml

# Use cuda-gdb
cuda-gdb --args ./jericho_mkII ../examples/reconnection.toml
```

### MPI Errors

```bash
# Enable MPI debugging
export OMPI_MCA_btl_base_verbose=100
mpirun -np 4 ./jericho_mkII config.toml

# Check MPI with simple test
mpirun -np 4 hostname
```

### HDF5 Errors

```bash
# Check HDF5 is parallel
h5pcc -showconfig | grep -i parallel

# Verify file integrity
h5dump -H output/fields_step_000000.h5

# Enable HDF5 error stack
export HDF5_DEBUG="all"
./jericho_mkII ../examples/reconnection.toml
```

### Segmentation Faults

```bash
# Run with GDB
gdb --args ./jericho_mkII ../examples/reconnection.toml
# (gdb) run
# (gdb) bt  # backtrace when it crashes

# Or with Valgrind (CPU code only)
valgrind --leak-check=full ./jericho_mkII config.toml
```

---

## Common Issues

### Issue: "CUDA error: no kernel image is available"

**Cause:** GPU architecture mismatch

**Fix:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..  # Use your GPU's CC
make clean && make -j
```

### Issue: "HDF5 is not parallel"

**Cause:** Non-parallel HDF5 build

**Fix:**
```bash
# Install parallel HDF5
sudo apt-get install libhdf5-openmpi-dev

# Or build from source with parallel support
./configure --enable-parallel --enable-shared
make && sudo make install
```

### Issue: MPI can't find CUDA devices

**Cause:** CUDA_VISIBLE_DEVICES not set

**Fix:**
```bash
# Manually set GPU mapping
mpirun -np 4 bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; ./jericho_mkII config.toml'
```

### Issue: Out of GPU memory

**Cause:** Too many particles or large grid

**Fix:**
1. Reduce `particles_per_cell` in config
2. Use smaller grid (nx, ny)
3. Use multiple GPUs with MPI

---

## Benchmarking

### Performance Metrics

```bash
# Create performance test config
cat > benchmark.toml << EOF
[simulation]
dt = 1e-9
n_steps = 1000

[grid]
nx = 512
ny = 512

[[species]]
name = "test"
particles_per_cell = 100
EOF

# Run benchmark
time ./jericho_mkII benchmark.toml
```

**Target Performance (V100 GPU):**
- 1000 steps, 512×512 grid, 100 PPC: ~30 seconds
- Should achieve >10M particle-steps/second

### Scaling Study

```bash
#!/bin/bash
for np in 1 2 4 8 16; do
    echo "=== $np processes ===" | tee -a scaling.log
    mpirun -np $np ./jericho_mkII benchmark.toml 2>&1 | \
        grep "Time per step" | tee -a scaling.log
done
```

Plot results to verify strong/weak scaling.

---

## Next Steps

After successful build and test:

1. ✅ Run validation suite (compare with original Jericho)
2. ✅ Optimize kernel configurations based on profiling
3. ✅ Run production science simulations
4. ✅ Publish performance results
5. ✅ Contribute improvements back to repository

---

## Support

- **Documentation:** See `docs/` directory
- **Examples:** See `examples/` directory
- **Issues:** GitHub Issues tracker
- **Performance:** Profile with Nsight, post results

---

**Last Updated:** 2025-11-15
**Build System:** CMake 3.18+
**Status:** ✅ Ready for testing
