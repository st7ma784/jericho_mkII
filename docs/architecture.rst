Architecture and Design
=======================

This document explains the computational architecture and design principles of Jericho Mk II, covering both the physics models and computer science implementation.

Overview
--------

Jericho Mk II is a **hybrid Particle-in-Cell / Magnetohydrodynamic (PIC-MHD)** code that simulates plasma dynamics by:

1. **Kinetic ions**: Treated as individual particles (PIC approach)
2. **Fluid electrons**: Treated as a massless, charge-neutralizing fluid (MHD approach)
3. **Self-consistent fields**: Electromagnetic fields updated from particle distributions

This hybrid approach balances accuracy (kinetic ions capture wave-particle interactions) with efficiency (fluid electrons avoid expensive high-frequency waves).

.. figure:: _static/hybrid_pic_mhd_schematic.png
   :width: 600px
   :align: center
   
   Hybrid PIC-MHD conceptual diagram

Hybrid PIC-MHD Physics
-----------------------

Governing Equations
~~~~~~~~~~~~~~~~~~~

**1. Particle equations of motion (ions):**

.. math::

   \frac{d\mathbf{x}_i}{dt} &= \mathbf{v}_i \\
   \frac{d\mathbf{v}_i}{dt} &= \frac{q_i}{m_i}(\mathbf{E} + \mathbf{v}_i \times \mathbf{B})

where :math:`\mathbf{x}_i`, :math:`\mathbf{v}_i` are particle position and velocity, :math:`q_i`, :math:`m_i` are charge and mass.

**2. Generalized Ohm's Law (electrons):**

.. math::

   \mathbf{E} = -\mathbf{v}_e \times \mathbf{B} + \frac{1}{en_e}\mathbf{J} \times \mathbf{B} + \eta\mathbf{J} - \frac{1}{en_e}\nabla p_e

Terms:

- :math:`-\mathbf{v}_e \times \mathbf{B}`: Convective electric field
- :math:`\frac{1}{en_e}\mathbf{J} \times \mathbf{B}`: **Hall term** (ion-electron decoupling)
- :math:`\eta\mathbf{J}`: Resistivity (dissipation)
- :math:`-\frac{1}{en_e}\nabla p_e`: Electron pressure gradient

**3. Faraday's Law:**

.. math::

   \frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}

**4. Charge continuity:**

.. math::

   \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \rho = \sum_s q_s n_s

**5. Current density:**

.. math::

   \mathbf{J} = \sum_s q_s n_s \mathbf{v}_s

where :math:`s` indexes particle species.

Why Hybrid?
~~~~~~~~~~~

**Advantages:**

1. **Computational efficiency**: Avoids electron time scales (:math:`\omega_{pe}` >> :math:`\omega_{pi}`)
2. **Physical insight**: Captures ion kinetics (key for reconnection, wave-particle interaction)
3. **Manageable particle count**: Only ions are particles; electrons are fluid

**When to use hybrid:**

- Ion-scale phenomena (:math:`\sim d_i = c/\omega_{pi}`)
- Magnetic reconnection
- Kelvin-Helmholtz instability
- Ion cyclotron waves
- Magnetospheric dynamics

**When NOT to use hybrid:**

- Electron-scale physics (use full PIC)
- Strongly collisional plasmas (use MHD)
- Relativistic effects (use relativistic PIC)

Numerical Methods
-----------------

Particle-in-Cell Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PIC cycle consists of:

.. code-block:: text

   1. Particle-to-Grid (P2G): Interpolate particle quantities → grid
   2. Field Solve: Update E, B from charge/current
   3. Grid-to-Particle (G2P): Interpolate fields → particles
   4. Particle Push: Advance particle positions/velocities

**Detailed steps:**

.. math::

   \begin{align}
   \rho(\mathbf{x}) &= \sum_i q_i S(\mathbf{x} - \mathbf{x}_i) & \text{(Charge deposition)} \\
   \mathbf{J}(\mathbf{x}) &= \sum_i q_i \mathbf{v}_i S(\mathbf{x} - \mathbf{x}_i) & \text{(Current deposition)} \\
   \mathbf{E}^{n+1} &= f(\rho^n, \mathbf{J}^n, \mathbf{B}^n) & \text{(Ohm's law)} \\
   \mathbf{B}^{n+1} &= \mathbf{B}^n - \Delta t \, \nabla \times \mathbf{E}^{n+1} & \text{(Faraday's law)} \\
   \mathbf{v}_i^{n+1} &= \mathbf{v}_i^n + \frac{q_i}{m_i}(\mathbf{E} + \mathbf{v}_i \times \mathbf{B})\Delta t & \text{(Boris push)} \\
   \mathbf{x}_i^{n+1} &= \mathbf{x}_i^n + \mathbf{v}_i^{n+1} \Delta t & \text{(Position update)}
   \end{align}

where :math:`S(\mathbf{x})` is the shape function (typically bilinear).

Boris Particle Pusher
~~~~~~~~~~~~~~~~~~~~~

The **Boris algorithm** is the gold standard for particle pushing in electromagnetic PIC codes. It's a leapfrog method that is:

- **Second-order accurate** in time
- **Velocity-conserving**: Preserves :math:`|\mathbf{v}|` when :math:`\mathbf{E} = 0`
- **Gyro-invariant**: Handles strong magnetic fields without instability

**Algorithm:**

.. code-block:: text

   1. Half-step electric acceleration: v⁻ = vⁿ + (q/m) E Δt/2
   2. Magnetic rotation: v⁺ = rotate(v⁻, B, Δt)
   3. Half-step electric acceleration: vⁿ⁺¹ = v⁺ + (q/m) E Δt/2
   4. Position update: xⁿ⁺¹ = xⁿ + vⁿ⁺¹ Δt

**Rotation step** (the "Boris trick"):

.. math::

   \mathbf{t} &= \frac{q}{m}\mathbf{B}\frac{\Delta t}{2} \\
   \mathbf{s} &= \frac{2\mathbf{t}}{1 + |\mathbf{t}|^2} \\
   \mathbf{v}' &= \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t} \\
   \mathbf{v}^+ &= \mathbf{v}^- + \mathbf{v}' \times \mathbf{s}

**Energy conservation:**

For :math:`\mathbf{E} = 0`, :math:`|\mathbf{v}^{n+1}| = |\mathbf{v}^n|` exactly (no numerical heating).

**Implementation** in ``boris_pusher.cpp``:

.. code-block:: cpp

   void boris_push(double& vx, double& vy,
                   double Ex, double Ey, double Bz,
                   double q_over_m, double dt) {
       const double half_dt = 0.5 * dt;
       const double qm_half_dt = q_over_m * half_dt;
       
       // Step 1: Half E acceleration
       double vx_minus = vx + qm_half_dt * Ex;
       double vy_minus = vy + qm_half_dt * Ey;
       
       // Step 2: B rotation (Boris rotation)
       double t = qm_half_dt * Bz;
       double t2 = t * t;
       double s = 2.0 * t / (1.0 + t2);
       
       double vx_prime = vx_minus + vy_minus * t;
       double vy_prime = vy_minus - vx_minus * t;
       
       double vx_plus = vx_minus + vy_prime * s;
       double vy_plus = vy_minus - vx_prime * s;
       
       // Step 3: Half E acceleration
       vx = vx_plus + qm_half_dt * Ex;
       vy = vy_plus + qm_half_dt * Ey;
   }

Current Advance Method (CAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Current Advance Method** improves numerical stability by updating current density in a way that respects charge conservation at the discrete level.

**Standard approach (unstable):**

.. math::

   \mathbf{J}^{n+1} = \sum_i q_i \mathbf{v}_i^{n+1} S(\mathbf{x} - \mathbf{x}_i^{n+1})

Problem: Can violate discrete charge conservation, leading to non-physical charge accumulation.

**CAM approach (stable):**

.. math::

   \mathbf{J}^{n+1} = \mathbf{J}^n + \frac{\rho^{n+1} - \rho^n}{\Delta t} + \text{corrections}

This ensures:

.. math::

   \frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{J} = 0

at the discrete level.

**Benefits:**

- Eliminates spurious charge accumulation
- Allows larger timesteps
- Critical for long-time simulations

**Implementation:** See ``ampere_solver.cpp`` for details.

GPU Architecture
----------------

Structure of Arrays (SoA)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Traditional Array of Structures (AoS):**

.. code-block:: cpp

   struct Particle {
       double x, y, vx, vy, weight;
       uint8_t type;
   };
   std::vector<Particle> particles;  // Bad for GPU!

**Memory layout:**

.. code-block:: text

   [x0,y0,vx0,vy0,w0,t0][x1,y1,vx1,vy1,w1,t1][x2,y2,vx2,vy2,w2,t2]...
   ^                     ^                     ^
   Thread 0              Thread 1              Thread 2

Problem: Adjacent threads access non-adjacent memory → **scattered reads** (slow).

**Structure of Arrays (SoA):**

.. code-block:: cpp

   struct ParticleBuffer {
       double* x;    // All x positions contiguous
       double* y;    // All y positions contiguous
       double* vx;
       double* vy;
       double* weight;
       uint8_t* type;
   };

**Memory layout:**

.. code-block:: text

   x:  [x0,x1,x2,x3,x4,...]
        ^  ^  ^  ^  ^
        T0 T1 T2 T3 T4  ← Coalesced access!
   
   y:  [y0,y1,y2,y3,y4,...]
   vx: [vx0,vx1,vx2,...]
   ...

**Performance impact:**

- **10-50x speedup** on GPU due to coalesced memory access
- **4-8x speedup** on CPU due to SIMD vectorization
- Better cache utilization

**Trade-off:** More complex indexing, but massively faster execution.

CUDA Kernel Design
~~~~~~~~~~~~~~~~~~

**Typical kernel structure:**

.. code-block:: cuda

   __global__ void advance_particles_kernel(
       double* x, double* y, double* vx, double* vy,
       const double* Ex, const double* Ey, const double* Bz,
       int n_particles, double dt, ...) {
       
       // Get particle index for this thread
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i >= n_particles) return;  // Guard against overrun
       
       // Load particle data (coalesced read)
       double px = x[i];
       double py = y[i];
       double pvx = vx[i];
       double pvy = vy[i];
       
       // Interpolate fields at particle position
       double Ex_particle = bilinear_interp(Ex, px, py, ...);
       double Ey_particle = bilinear_interp(Ey, px, py, ...);
       double Bz_particle = bilinear_interp(Bz, px, py, ...);
       
       // Boris push
       boris_push(pvx, pvy, Ex_particle, Ey_particle, Bz_particle, q_over_m, dt);
       
       // Update position
       px += pvx * dt;
       py += pvy * dt;
       
       // Store results (coalesced write)
       x[i] = px;
       y[i] = py;
       vx[i] = pvx;
       vy[i] = pvy;
   }

**Launch configuration:**

.. code-block:: cpp

   int threads_per_block = 256;
   int num_blocks = (n_particles + threads_per_block - 1) / threads_per_block;
   advance_particles_kernel<<<num_blocks, threads_per_block>>>(
       particles.x, particles.y, particles.vx, particles.vy,
       fields.Ex, fields.Ey, fields.Bz,
       particles.count, dt, ...);

**Key optimizations:**

1. **Coalesced memory access**: Adjacent threads access adjacent memory
2. **Minimize divergence**: Avoid ``if`` statements where possible
3. **Register reuse**: Keep data in registers (fast) rather than global memory (slow)
4. **Occupancy**: Aim for 50-100% occupancy (active warps / max warps)

MPI+CUDA Hybrid Parallelism
----------------------------

Domain Decomposition
~~~~~~~~~~~~~~~~~~~~

The simulation domain is divided into subdomains, each handled by one MPI rank (typically one GPU).

.. code-block:: text

   Global domain (256x256):
   
   ┌─────────┬─────────┐
   │ Rank 0  │ Rank 1  │  Each: 128x256
   │ GPU 0   │ GPU 1   │
   ├─────────┼─────────┤
   │ Rank 2  │ Rank 3  │
   │ GPU 2   │ GPU 3   │
   └─────────┴─────────┘

**Ghost cells:** Each subdomain includes extra layers (ghost cells) for boundary data from neighbors.

.. code-block:: text

   Rank 0 (with ghost cells):
   
   ┌─┬──────────┬─┐
   │G│          │G│  G = Ghost cells (from neighbors)
   ├─┼──────────┼─┤  Interior = Local computation
   │G│ Interior │G│
   ├─┼──────────┼─┤
   │G│          │G│
   └─┴──────────┴─┘

Ghost Cell Exchange
~~~~~~~~~~~~~~~~~~~

After each field update, boundary data is exchanged between neighbors:

.. code-block:: cpp

   void exchange_ghost_cells(FieldArrays& fields, MPIManager& mpi) {
       // Pack boundary data into send buffers
       pack_send_buffers(fields, mpi.comm_buffers);
       
       // Non-blocking sends/receives
       MPI_Isend(send_left, ..., neighbor_left, tag, comm, &req_send_left);
       MPI_Irecv(recv_left, ..., neighbor_left, tag, comm, &req_recv_left);
       // ... repeat for right, top, bottom
       
       // Wait for completion
       MPI_Waitall(8, requests, statuses);
       
       // Unpack received data into ghost cells
       unpack_recv_buffers(mpi.comm_buffers, fields);
   }

**CUDA-aware MPI** (if available):

.. code-block:: cpp

   // Send directly from GPU device memory!
   MPI_Isend(fields.Ex + offset,  // Device pointer
             count, MPI_DOUBLE, neighbor_left, tag, comm, &req);

This eliminates CPU staging and is **2-5x faster** for large messages.

Particle Migration
~~~~~~~~~~~~~~~~~~

Particles that leave a subdomain must be sent to the appropriate neighbor:

.. code-block:: cpp

   void migrate_particles(ParticleBuffer& particles, MPIManager& mpi) {
       // Identify particles that crossed boundaries
       std::vector<int> send_to_left, send_to_right, ...;
       
       for (size_t i = 0; i < particles.count; ++i) {
           if (particles.x[i] < mpi.x_min_local) {
               send_to_left.push_back(i);
           } else if (particles.x[i] >= mpi.x_max_local) {
               send_to_right.push_back(i);
           }
           // ... similar for y boundaries
       }
       
       // Pack and send
       pack_particles(particles, send_to_left, send_buffer_left);
       MPI_Isend(send_buffer_left, ..., neighbor_left, ...);
       
       // Receive and unpack
       MPI_Recv(recv_buffer_left, ..., neighbor_left, ...);
       unpack_particles(recv_buffer_left, particles);
       
       // Remove migrated particles
       remove_particles(particles, send_to_left);
   }

Memory Management
-----------------

GPU Memory Hierarchy
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 45

   * - Memory Type
     - Size
     - Latency
     - Usage
   * - **Registers**
     - 64 KB/SM
     - 1 cycle
     - Thread-local variables
   * - **Shared Memory**
     - 48-96 KB/SM
     - ~30 cycles
     - Block-local communication
   * - **L1 Cache**
     - 128 KB/SM
     - ~30 cycles
     - Automatic caching
   * - **L2 Cache**
     - 6-40 MB
     - ~200 cycles
     - Shared across SMs
   * - **Global Memory**
     - 8-80 GB
     - ~500 cycles
     - Main data arrays
   * - **Host Memory**
     - 64+ GB
     - ~100,000 cycles
     - Fallback storage

**Design implications:**

1. **Minimize global memory accesses**: Use registers/shared memory
2. **Coalesce accesses**: SoA layout ensures this
3. **Reuse data**: Keep frequently accessed data in faster memory
4. **Prefetch**: Use asynchronous transfers to hide latency

Memory Allocation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class ParticleBuffer {
   public:
       ParticleBuffer(size_t capacity, int device_id) {
           cudaSetDevice(device_id);
           
           // Allocate device memory
           cudaMalloc(&x, capacity * sizeof(double));
           cudaMalloc(&y, capacity * sizeof(double));
           cudaMalloc(&vx, capacity * sizeof(double));
           cudaMalloc(&vy, capacity * sizeof(double));
           cudaMalloc(&weight, capacity * sizeof(double));
           cudaMalloc(&type, capacity * sizeof(uint8_t));
           cudaMalloc(&active, capacity * sizeof(bool));
           
           // Initialize to zero
           cudaMemset(x, 0, capacity * sizeof(double));
           // ... repeat for other arrays
       }
       
       ~ParticleBuffer() {
           // Free device memory
           cudaFree(x);
           cudaFree(y);
           // ... etc
       }
   };

**Dynamic resizing:**

.. code-block:: cpp

   void ParticleBuffer::resize(size_t new_capacity) {
       if (new_capacity <= capacity) return;
       
       // Allocate new arrays
       double *new_x, *new_y, ...;
       cudaMalloc(&new_x, new_capacity * sizeof(double));
       cudaMalloc(&new_y, new_capacity * sizeof(double));
       // ...
       
       // Copy old data
       cudaMemcpy(new_x, x, count * sizeof(double), cudaMemcpyDeviceToDevice);
       cudaMemcpy(new_y, y, count * sizeof(double), cudaMemcpyDeviceToDevice);
       // ...
       
       // Free old arrays
       cudaFree(x);
       cudaFree(y);
       // ...
       
       // Update pointers
       x = new_x;
       y = new_y;
       // ...
       
       capacity = new_capacity;
   }

Code Organization
-----------------

Module Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

   jericho_mkII/
   ├── src/                 # CPU host code
   │   ├── main.cpp         # Entry point, simulation loop
   │   ├── config.cpp       # TOML parser
   │   ├── particle_buffer.cpp  # CPU memory management
   │   ├── field_arrays.cpp     # CPU field management
   │   ├── mpi_manager.cpp      # MPI domain decomposition
   │   └── io_manager.cpp       # HDF5 output
   │
   ├── cuda/                # GPU device code
   │   ├── particles.cu     # Particle push kernels
   │   ├── fields.cu        # Field update kernels
   │   ├── boundaries.cu    # Boundary condition kernels
   │   └── p2g.cu           # Particle-to-grid kernels
   │
   ├── include/             # Header files
   │   ├── particle_buffer.h
   │   ├── field_arrays.h
   │   ├── mpi_manager.h
   │   ├── boris_pusher.h
   │   └── platform.h       # CPU/GPU abstraction
   │
   └── tests/               # Unit tests
       ├── test_boris.cpp
       ├── test_p2g.cpp
       └── test_mpi.cpp

**Separation of concerns:**

- **src/**: High-level logic, orchestration, I/O
- **cuda/**: Low-level compute kernels
- **include/**: Interface definitions, constants

**Key design pattern: Platform abstraction**

``platform.h`` provides CPU/GPU compatibility:

.. code-block:: cpp

   // Compile for both CPU and GPU
   #ifdef __CUDACC__
   #define DEVICE_HOST __host__ __device__
   #define GLOBAL __global__
   #else
   #define DEVICE_HOST
   #define GLOBAL
   #endif
   
   // Same function works on CPU or GPU
   DEVICE_HOST inline double dot_product(double ax, double ay, double bx, double by) {
       return ax*bx + ay*by;
   }

Build System
~~~~~~~~~~~~

CMake handles cross-platform builds:

.. code-block:: cmake

   # CMakeLists.txt
   cmake_minimum_required(VERSION 3.18)
   project(jericho_mkII LANGUAGES CXX CUDA)
   
   # C++17 standard
   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CUDA_STANDARD 17)
   
   # CUDA architecture (user-specified)
   set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")
   
   # Find MPI
   find_package(MPI REQUIRED)
   find_package(HDF5 REQUIRED)
   
   # Executable
   add_executable(jericho_mkII
       src/main.cpp
       src/config.cpp
       src/particle_buffer.cpp
       src/field_arrays.cpp
       src/mpi_manager.cpp
       cuda/particles.cu
       cuda/fields.cu
       cuda/boundaries.cu
   )
   
   target_link_libraries(jericho_mkII
       MPI::MPI_CXX
       HDF5::HDF5
       cudart
   )

Testing Strategy
----------------

Unit Tests
~~~~~~~~~~

.. code-block:: cpp

   // tests/test_boris.cpp
   #include <gtest/gtest.h>
   #include "boris_pusher.h"
   
   TEST(BorisTest, EnergyConservation) {
       // Test that Boris pusher conserves energy for E=0
       double vx = 1.0e6, vy = 0.0;
       double Ex = 0.0, Ey = 0.0, Bz = 1.0e-9;
       double q_over_m = 9.578e7;  // Proton
       double dt = 0.01;
       
       double v0 = std::sqrt(vx*vx + vy*vy);
       
       // Push 1000 timesteps
       for (int i = 0; i < 1000; ++i) {
           boris_push(vx, vy, Ex, Ey, Bz, q_over_m, dt);
       }
       
       double v1 = std::sqrt(vx*vx + vy*vy);
       
       // Check energy conservation
       EXPECT_NEAR(v0, v1, 1e-10);
   }

Integration Tests
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run full simulation and check output
   ./jericho_mkII ../examples/minimal_test.toml
   
   # Verify output files exist
   test -f output/fields_000100.h5 || exit 1
   
   # Check energy conservation
   python3 scripts/check_energy_conservation.py output/diagnostics.csv

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Measure kernel performance
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   advance_particles_kernel<<<blocks, threads>>>(...);
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   
   printf("Particle push: %.3f ms (%.2f Gparticles/s)\n",
          milliseconds, n_particles / (milliseconds * 1e6));

Key Performance Metrics
-----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Target
     - Measurement
   * - Particles/sec
     - > 1 Gparticles/s
     - Kernel timing
   * - GPU occupancy
     - 50-100%
     - nvprof
   * - Memory bandwidth
     - > 70% peak
     - nvprof
   * - Energy conservation
     - |ΔE|/E < 1%
     - Diagnostics
   * - MPI efficiency
     - > 80%
     - Weak scaling
   * - CUDA-aware MPI speedup
     - 2-5x vs host-staged
     - Bandwidth test

See Also
--------

- :doc:`cuda_kernels` - Detailed CUDA implementation
- :doc:`mpi_parallelism` - Multi-GPU parallelization
- :doc:`performance_tuning` - Optimization techniques
- :doc:`api/particle_buffer` - API reference
