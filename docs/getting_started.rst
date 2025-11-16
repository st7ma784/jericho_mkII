Getting Started
===============

Welcome to Jericho Mk II! This guide will help you install, build, and run your first plasma simulation.

Prerequisites
-------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

**Minimum:**

- NVIDIA GPU with compute capability 6.0+ (Pascal architecture or newer)
- 4 GB GPU memory
- 8 GB system RAM
- 4 CPU cores

**Recommended:**

- NVIDIA GPU with compute capability 7.0+ (Volta/Turing/Ampere)
- 8+ GB GPU memory
- 16+ GB system RAM
- 8+ CPU cores
- NVLink or high-bandwidth interconnect for multi-GPU

**For Multi-GPU/Multi-Node:**

- Multiple NVIDIA GPUs with GPUDirect support
- CUDA-aware MPI implementation (OpenMPI 4.0+ or MPICH 3.3+)
- InfiniBand or high-speed network interconnect

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

**Required:**

- **CUDA Toolkit** 11.0 or later (tested with 11.8, 12.0, 12.3)
  
  Download from: https://developer.nvidia.com/cuda-downloads

- **C++ Compiler** with C++17 support:
  
  - GCC 9.0 or later
  - Clang 10.0 or later
  - MSVC 2019 or later (Windows)

- **CMake** 3.18 or later

- **MPI Implementation** (for multi-GPU):
  
  - OpenMPI 4.0+ with CUDA support
  - MPICH 3.3+ with CUDA support

**Optional but Recommended:**

- **HDF5** 1.10+ for output file format
- **Python 3.7+** with matplotlib, numpy for visualization
- **Sphinx** for building documentation
- **Git** for version control

Installation
------------

Step 1: Install CUDA Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   # Download and install CUDA Toolkit
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda

   # Verify installation
   nvcc --version
   nvidia-smi

**Verify GPU Compute Capability:**

.. code-block:: bash

   # Check your GPU's compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader

Note the compute capability (e.g., 7.0, 8.0, 8.6) - you'll need this for CMake.

Step 2: Install MPI (for Multi-GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**OpenMPI with CUDA Support:**

.. code-block:: bash

   # Install dependencies
   sudo apt-get install build-essential

   # Download and build OpenMPI with CUDA support
   wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
   tar xzf openmpi-4.1.5.tar.gz
   cd openmpi-4.1.5
   
   ./configure --prefix=/usr/local --with-cuda=/usr/local/cuda
   make -j$(nproc)
   sudo make install

   # Verify CUDA-aware MPI
   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

Step 3: Clone and Build Jericho Mk II
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/st7ma784/jericho_mkII.git
   cd jericho_mkII

   # Create build directory
   mkdir build && cd build

   # Configure with CMake
   # Replace "80" with your GPU's compute capability (e.g., 70, 75, 80, 86, 89, 90)
   cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
         -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_MPI=ON \
         ..

   # Build (use -j for parallel compilation)
   make -j$(nproc)

   # Verify build
   ./jericho_mkII --version

**CMake Configuration Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Option
     - Description
     - Default
   * - ``CMAKE_CUDA_ARCHITECTURES``
     - GPU compute capability (60, 70, 75, 80, 86, 89, 90)
     - 70
   * - ``CMAKE_BUILD_TYPE``
     - Build type (Debug, Release, RelWithDebInfo)
     - Release
   * - ``ENABLE_MPI``
     - Enable multi-GPU support
     - ON
   * - ``ENABLE_HDF5``
     - Enable HDF5 output format
     - ON
   * - ``ENABLE_TESTING``
     - Build unit tests
     - OFF
   * - ``CUDA_SEPARABLE_COMPILATION``
     - Enable separate compilation
     - ON

**Common Build Issues:**

*Issue: "nvcc not found"*

.. code-block:: bash

   # Add CUDA to PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

*Issue: "compute capability mismatch"*

.. code-block:: bash

   # Clean and reconfigure with correct architecture
   rm -rf CMakeCache.txt CMakeFiles/
   cmake -DCMAKE_CUDA_ARCHITECTURES=YOUR_GPU_ARCH ..

Running Your First Simulation
------------------------------

Single GPU Example
~~~~~~~~~~~~~~~~~~

Let's run a simple magnetic reconnection simulation:

.. code-block:: bash

   # From the build directory
   cd /path/to/jericho_mkII/build

   # Run with example configuration
   ./jericho_mkII ../examples/reconnection.toml

You should see output like:

.. code-block:: text

   ╔════════════════════════════════════════════════════════════╗
   ║           Jericho Mk II - GPU PIC-MHD Simulator           ║
   ║                      Version 2.0.0                        ║
   ╚════════════════════════════════════════════════════════════╝
   
   [Config] Loading configuration: ../examples/reconnection.toml
   [System] Detected GPU: NVIDIA GeForce RTX 3080 (compute 8.6)
   [Domain] Grid: 256x256, Domain: [-10.0, 10.0] x [-10.0, 10.0]
   [Particles] 100 particles/cell, 2 species
   [Physics] Boris pusher: ON, CAM: ON, Hall term: ON
   
   Step     0/10000 | t = 0.000e+00 | particles = 6553600 | energy = 1.234e-12 J
   Step   100/10000 | t = 1.000e+00 | particles = 6553600 | energy = 1.234e-12 J
   Step   200/10000 | t = 2.000e+00 | particles = 6553600 | energy = 1.234e-12 J
   ...

**Output files will be created in:**

.. code-block:: bash

   ./output/
   ├── fields_000000.h5
   ├── fields_000100.h5
   ├── particles_000000.h5
   ├── diagnostics.csv
   └── checkpoints/
       └── checkpoint_000500.h5

Multi-GPU Example
~~~~~~~~~~~~~~~~~

Run on 4 GPUs with MPI:

.. code-block:: bash

   # 2x2 process grid (4 GPUs total)
   mpirun -np 4 ./jericho_mkII ../examples/reconnection.toml

   # Specify GPU assignments (if needed)
   mpirun -np 4 \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       ./jericho_mkII ../examples/reconnection.toml

**Multi-node example (8 GPUs across 2 nodes):**

.. code-block:: bash

   # Create hostfile
   cat > hostfile << EOF
   node1 slots=4
   node2 slots=4
   EOF

   # Run with MPI
   mpirun -np 8 --hostfile hostfile \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       ./jericho_mkII ../examples/reconnection.toml

Understanding the Configuration File
------------------------------------

The TOML configuration file controls all simulation parameters. Here's a minimal example:

.. code-block:: toml

   [simulation]
   name = "my_first_simulation"
   output_dir = "./output"
   dt = 0.01              # Timestep [ion gyroperiods]
   n_steps = 1000         # Number of timesteps
   output_cadence = 50    # Write output every 50 steps

   [domain]
   x_min = -5.0
   x_max = 5.0
   y_min = -5.0
   y_max = 5.0
   nx = 128              # Grid points in x
   ny = 128              # Grid points in y

   [particles]
   particles_per_cell = 50

   [[particles.species]]
   name = "H+"
   charge = 1.602e-19    # Coulombs
   mass = 1.673e-27      # kg (proton mass)
   temperature = 1.0e6   # Kelvin
   density = 1.0e6       # particles/m³

   [fields]
   B0 = 20.0e-9          # Initial magnetic field [Tesla]

   [boundaries]
   x_min = "periodic"
   x_max = "periodic"
   y_min = "periodic"
   y_max = "periodic"

See :doc:`configuration` for complete documentation of all options.

Visualizing Results
-------------------

Basic Visualization with Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np

   # Load field data
   with h5py.File('output/fields_001000.h5', 'r') as f:
       Bz = f['Bz'][:]
       Ex = f['Ex'][:]
       Ey = f['Ey'][:]

   # Plot magnetic field
   plt.figure(figsize=(12, 4))
   
   plt.subplot(131)
   plt.imshow(Bz.T, origin='lower', cmap='RdBu', aspect='auto')
   plt.colorbar(label='Bz [T]')
   plt.title('Magnetic Field')
   
   plt.subplot(132)
   plt.imshow(Ex.T, origin='lower', cmap='viridis', aspect='auto')
   plt.colorbar(label='Ex [V/m]')
   plt.title('Electric Field (x)')
   
   plt.subplot(133)
   # Vector plot of E field
   plt.quiver(Ex[::8, ::8], Ey[::8, ::8])
   plt.title('Electric Field Vectors')
   
   plt.tight_layout()
   plt.savefig('fields_visualization.png', dpi=150)
   plt.show()

**Load diagnostic data:**

.. code-block:: python

   import pandas as pd

   # Load diagnostics CSV
   diag = pd.read_csv('output/diagnostics.csv')

   # Plot energy conservation
   plt.figure()
   plt.plot(diag['time'], diag['total_energy'])
   plt.xlabel('Time [ion gyroperiods]')
   plt.ylabel('Total Energy [J]')
   plt.title('Energy Conservation')
   plt.grid(True)
   plt.savefig('energy_conservation.png', dpi=150)

Quick Reference: Common Commands
---------------------------------

**Build and run:**

.. code-block:: bash

   # Build (from build directory)
   cmake .. && make -j

   # Single GPU
   ./jericho_mkII config.toml

   # Multi-GPU
   mpirun -np 4 ./jericho_mkII config.toml

**Check GPU status:**

.. code-block:: bash

   # GPU info
   nvidia-smi

   # Monitor in real-time
   watch -n 1 nvidia-smi

**Test installation:**

.. code-block:: bash

   # Run minimal test
   ./jericho_mkII ../examples/minimal_test.toml

   # Run with verbose output
   ./jericho_mkII --verbose config.toml

**View output:**

.. code-block:: bash

   # List output files
   ls -lh output/

   # Check HDF5 file contents
   h5dump -H output/fields_000100.h5

Next Steps
----------

Now that you have Jericho Mk II running, you can:

1. **Explore Examples**: Try the configurations in ``examples/``

   - ``reconnection.toml`` - Magnetic reconnection
   - ``minimal_test.toml`` - Quick validation

2. **Configure Your Simulation**: See :doc:`configuration` for all options

3. **Optimize Performance**: Read :doc:`performance_tuning` for tips

4. **Understand the Physics**: See :doc:`architecture` for hybrid PIC-MHD details

5. **Develop Custom Features**: Check :doc:`contributing` to modify the code

Troubleshooting
---------------

GPU Not Detected
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check if NVIDIA driver is loaded
   nvidia-smi

   # If not found, install drivers
   sudo ubuntu-drivers autoinstall

MPI Errors
~~~~~~~~~~

.. code-block:: bash

   # Test MPI installation
   mpirun -np 2 hostname

   # Check if CUDA-aware
   ompi_info --parsable --all | grep cuda

Out of Memory
~~~~~~~~~~~~~

Reduce ``particles_per_cell`` or grid resolution in config:

.. code-block:: toml

   [particles]
   particles_per_cell = 25  # Reduce from 100

   [domain]
   nx = 128  # Reduce from 256
   ny = 128

Performance Issues
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run with profiling
   nvprof ./jericho_mkII config.toml

   # Or use Nsight Systems
   nsys profile ./jericho_mkII config.toml

Getting Help
------------

- **Documentation**: https://st7ma784.github.io/jericho_mkII/
- **GitHub Issues**: https://github.com/st7ma784/jericho_mkII/issues
- **Email**: j.wiggs@lancaster.ac.uk

**Before asking for help, please:**

1. Check the :doc:`troubleshooting` page
2. Run with ``--verbose`` flag for detailed output
3. Include your configuration file and error messages
4. Specify your GPU model and CUDA version
