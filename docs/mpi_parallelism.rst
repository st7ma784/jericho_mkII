MPI Parallelism
===============

This document explains the MPI parallelization strategy in Jericho Mk II, covering domain decomposition, communication patterns, and scaling strategies for multi-GPU simulations.

Overview
--------

Jericho Mk II uses **MPI (Message Passing Interface)** for multi-GPU parallelism. The simulation domain is divided into subdomains, with each MPI rank (typically one GPU) responsible for one subdomain.

**Key features:**

- 2D Cartesian domain decomposition
- CUDA-aware MPI for direct GPU-GPU transfers
- Asynchronous communication with computation overlap
- Dynamic load balancing (experimental)
- Weak and strong scaling to 100s of GPUs

Why MPI?
--------

**Advantages:**

1. **Scalability**: Scales to multiple nodes/GPUs
2. **Portability**: Works on any HPC system
3. **Mature ecosystem**: Well-tested libraries and tools
4. **CUDA-aware MPI**: Direct GPU-GPU communication

**Alternatives considered:**

- **NCCL**: NVIDIA's collective communications library (GPU-only, less flexible)
- **UPC++**: Unified Parallel C++ (less mature)
- **Single-GPU**: Limited by GPU memory (~10-80 GB)

Domain Decomposition
--------------------

Cartesian Grid Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation domain is divided into a 2D grid of subdomains:

.. code-block:: text

   Global domain (512x512 grid):
   
   ┌──────────┬──────────┬──────────┬──────────┐
   │ Rank 0   │ Rank 1   │ Rank 2   │ Rank 3   │
   │ GPU 0    │ GPU 1    │ GPU 2    │ GPU 3    │
   │ 128×256  │ 128×256  │ 128×256  │ 128×256  │
   ├──────────┼──────────┼──────────┼──────────┤
   │ Rank 4   │ Rank 5   │ Rank 6   │ Rank 7   │
   │ GPU 0    │ GPU 1    │ GPU 2    │ GPU 3    │
   │ 128×256  │ 128×256  │ 128×256  │ 128×256  │
   └──────────┴──────────┴──────────┴──────────┘
   
   Configuration: npx=4, npy=2, total ranks=8

**Implementation:**

.. code-block:: cpp

   class MPIManager {
       int rank;         // This process's rank (0 to size-1)
       int size;         // Total number of MPI processes
       int npx, npy;     // Process grid dimensions
       int rank_x, rank_y;  // This rank's 2D coordinates
       
       // Compute rank's position in 2D grid
       void compute_topology() {
           rank_x = rank % npx;
           rank_y = rank / npx;
       }
       
       // Compute local domain bounds
       void compute_local_domain() {
           double Lx = x_max_global - x_min_global;
           double Ly = y_max_global - y_min_global;
           
           double dx_subdomain = Lx / npx;
           double dy_subdomain = Ly / npy;
           
           x_min_local = x_min_global + rank_x * dx_subdomain;
           x_max_local = x_min_local + dx_subdomain;
           y_min_local = y_min_global + rank_y * dy_subdomain;
           y_max_local = y_min_local + dy_subdomain;
           
           nx_local = nx_global / npx;
           ny_local = ny_global / npy;
       }
   };

Neighbor Identification
~~~~~~~~~~~~~~~~~~~~~~~~

Each rank identifies its 4 neighbors (left, right, bottom, top):

.. code-block:: cpp

   void compute_neighbors() {
       // Left neighbor
       if (rank_x > 0) {
           neighbor_left = rank - 1;
       } else {
           neighbor_left = -1;  // No left neighbor (boundary)
       }
       
       // Right neighbor
       if (rank_x < npx - 1) {
           neighbor_right = rank + 1;
       } else {
           neighbor_right = -1;
       }
       
       // Bottom neighbor
       if (rank_y > 0) {
           neighbor_bottom = rank - npx;
       } else {
           neighbor_bottom = -1;
       }
       
       // Top neighbor
       if (rank_y < npy - 1) {
           neighbor_top = rank + npx;
       } else {
           neighbor_top = -1;
       }
   }

Ghost Cells
~~~~~~~~~~~

Each subdomain includes **ghost cells** (boundary layers) that store data from neighbors:

.. code-block:: text

   Rank 5's subdomain with ghost cells:
   
   ┌─┬────────────────────────┬─┐
   │G│    From Rank 1         │G│  ← Top ghost (from rank 1)
   ├─┼────────────────────────┼─┤
   │G│                        │G│  ← Left/right ghosts
   │G│   Interior (owned by   │G│     (from ranks 4 and 6)
   │G│       rank 5)          │G│
   │G│                        │G│
   ├─┼────────────────────────┼─┤
   │G│    From Rank 9         │G│  ← Bottom ghost (from rank 9)
   └─┴────────────────────────┴─┘
   
   G = Ghost cells (nghost layers, typically 2)

**Purpose:**

- Provide boundary data for stencil operations (e.g., curl, divergence)
- Avoid if/else checks in inner loops (performance)
- Enable local computation without communication

Communication Patterns
----------------------

Ghost Cell Exchange
~~~~~~~~~~~~~~~~~~~

After each field update, ghost cells are exchanged between neighbors:

**Algorithm:**

.. code-block:: cpp

   void exchange_ghost_cells(FieldArrays& fields) {
       // 1. Pack boundary data into send buffers
       pack_boundary_data(fields);
       
       // 2. Post non-blocking sends and receives
       MPI_Request requests[8];  // 4 sends + 4 receives
       int req_count = 0;
       
       // Send/receive left
       if (neighbor_left >= 0) {
           MPI_Isend(send_buffer_left, size, MPI_DOUBLE,
                    neighbor_left, TAG_LEFT, MPI_COMM_WORLD,
                    &requests[req_count++]);
           MPI_Irecv(recv_buffer_left, size, MPI_DOUBLE,
                    neighbor_left, TAG_RIGHT, MPI_COMM_WORLD,
                    &requests[req_count++]);
       }
       
       // Send/receive right
       if (neighbor_right >= 0) {
           MPI_Isend(send_buffer_right, size, MPI_DOUBLE,
                    neighbor_right, TAG_RIGHT, MPI_COMM_WORLD,
                    &requests[req_count++]);
           MPI_Irecv(recv_buffer_right, size, MPI_DOUBLE,
                    neighbor_right, TAG_LEFT, MPI_COMM_WORLD,
                    &requests[req_count++]);
       }
       
       // ... similar for top and bottom
       
       // 3. Wait for all communication to complete
       MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
       
       // 4. Unpack received data into ghost cells
       unpack_boundary_data(fields);
   }

**Packing boundary data:**

.. code-block:: cpp

   void pack_boundary_data(FieldArrays& fields) {
       // Pack left boundary (2 layers of interior cells)
       for (int j = 0; j < ny_local; ++j) {
           for (int i = 0; i < nghost; ++i) {
               int idx = j * nx + (nghost + i);  // Interior near left edge
               send_buffer_left[j * nghost + i] = fields.Ex[idx];
           }
       }
       
       // ... similar for right, top, bottom
   }

CUDA-Aware MPI
~~~~~~~~~~~~~~

**Standard MPI** (slow):

.. code-block:: cpp

   // Copy from GPU to CPU
   cudaMemcpy(h_send_buffer, d_field, size, cudaMemcpyDeviceToHost);
   
   // Send from CPU
   MPI_Isend(h_send_buffer, size, MPI_DOUBLE, dest, tag, comm, &req);
   
   // Receive on CPU
   MPI_Recv(h_recv_buffer, size, MPI_DOUBLE, source, tag, comm, &status);
   
   // Copy from CPU to GPU
   cudaMemcpy(d_field, h_recv_buffer, size, cudaMemcpyHostToDevice);

**CUDA-aware MPI** (fast):

.. code-block:: cpp

   // Send directly from GPU device memory!
   MPI_Isend(d_send_buffer,  // Device pointer
            size, MPI_DOUBLE, dest, tag, comm, &req);
   
   // Receive directly to GPU device memory!
   MPI_Irecv(d_recv_buffer,  // Device pointer
            size, MPI_DOUBLE, source, tag, comm, &req);

**Performance gain**: 2-5x faster for large messages (> 1 MB)

**Check if available:**

.. code-block:: bash

   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

Should output ``true``.

Particle Migration
~~~~~~~~~~~~~~~~~~

Particles that leave a subdomain must be sent to the appropriate neighbor:

**Algorithm:**

.. code-block:: cpp

   void migrate_particles(ParticleBuffer& particles) {
       // 1. Identify particles to migrate
       std::vector<size_t> send_to_left, send_to_right,
                           send_to_bottom, send_to_top;
       
       for (size_t i = 0; i < particles.count; ++i) {
           if (!particles.active[i]) continue;
           
           double px = particles.x[i];
           double py = particles.y[i];
           
           // Check if particle crossed boundaries
           if (px < x_min_local && neighbor_left >= 0) {
               send_to_left.push_back(i);
           } else if (px >= x_max_local && neighbor_right >= 0) {
               send_to_right.push_back(i);
           } else if (py < y_min_local && neighbor_bottom >= 0) {
               send_to_bottom.push_back(i);
           } else if (py >= y_max_local && neighbor_top >= 0) {
               send_to_top.push_back(i);
           }
       }
       
       // 2. Pack particles to send
       pack_particles(particles, send_to_left, particle_send_buffer_left);
       // ... similar for other directions
       
       // 3. Exchange counts (how many to expect)
       int n_send_left = send_to_left.size();
       int n_recv_left;
       MPI_Sendrecv(&n_send_left, 1, MPI_INT, neighbor_left, TAG,
                    &n_recv_left, 1, MPI_INT, neighbor_left, TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       
       // 4. Send/receive particles
       MPI_Sendrecv(particle_send_buffer_left, n_send_left * particle_size, MPI_BYTE,
                    neighbor_left, TAG,
                    particle_recv_buffer_left, n_recv_left * particle_size, MPI_BYTE,
                    neighbor_left, TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       
       // 5. Unpack received particles
       unpack_particles(particle_recv_buffer_left, n_recv_left, particles);
       
       // 6. Remove migrated particles
       remove_particles(particles, send_to_left);
   }

**Particle packing:**

.. code-block:: cpp

   struct ParticleData {
       double x, y, vx, vy, weight;
       uint8_t type;
   };
   
   void pack_particles(const ParticleBuffer& particles,
                      const std::vector<size_t>& indices,
                      std::vector<ParticleData>& buffer) {
       buffer.resize(indices.size());
       for (size_t i = 0; i < indices.size(); ++i) {
           size_t idx = indices[i];
           buffer[i].x = particles.x[idx];
           buffer[i].y = particles.y[idx];
           buffer[i].vx = particles.vx[idx];
           buffer[i].vy = particles.vy[idx];
           buffer[i].weight = particles.weight[idx];
           buffer[i].type = particles.type[idx];
       }
   }

Collective Operations
~~~~~~~~~~~~~~~~~~~~~

**Reductions** (e.g., total energy, particle count):

.. code-block:: cpp

   // Compute local energy on each GPU
   double local_energy = compute_local_energy(particles, fields);
   
   // Sum across all ranks
   double global_energy;
   MPI_Allreduce(&local_energy, &global_energy, 1, MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD);
   
   // Now all ranks have global_energy

**Gather** (e.g., collect data for output):

.. code-block:: cpp

   // Rank 0 collects data from all ranks
   if (rank == 0) {
       std::vector<double> global_field(nx_global * ny_global);
   }
   
   MPI_Gather(local_field, nx_local * ny_local, MPI_DOUBLE,
              global_field.data(), nx_local * ny_local, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

Load Balancing
--------------

Static Load Balancing
~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Equal-sized subdomains

**Assumption**: Uniform particle distribution

**Works well when**:

- Particles uniformly distributed
- Minimal particle migration
- All ranks have similar GPU performance

Dynamic Load Balancing (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Adjust subdomain sizes based on work distribution

.. code-block:: cpp

   void rebalance_domain() {
       // 1. Measure work on each rank
       double local_work = measure_work();  // e.g., particle count × grid cells
       
       // 2. Gather work from all ranks
       std::vector<double> all_work(size);
       MPI_Allgather(&local_work, 1, MPI_DOUBLE,
                    all_work.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
       
       // 3. Compute optimal subdomain sizes
       double total_work = std::accumulate(all_work.begin(), all_work.end(), 0.0);
       double target_work = total_work / size;
       
       // 4. Adjust boundaries (complex, omitted)
       // ...
       
       // 5. Migrate particles and fields to new subdomains
       // ...
   }

**Challenges:**

- Expensive to rebalance (migrate particles, fields)
- Difficult to predict optimal decomposition
- May hurt more than help for uniform problems

**Status**: Experimental, disabled by default

Performance Optimization
------------------------

Overlap Communication and Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea**: Compute on interior cells while communicating ghost cells

.. code-block:: cpp

   void optimized_time_step() {
       // 1. Start non-blocking ghost cell exchange
       MPI_Request requests[8];
       start_ghost_cell_exchange(fields, requests);
       
       // 2. Compute on interior cells (no ghost data needed)
       compute_interior_fields(fields);  // Kernel launch (async)
       
       // 3. Wait for communication
       MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
       
       // 4. Compute on boundary cells (needs ghost data)
       compute_boundary_fields(fields);
   }

**Speedup**: Can hide communication latency (10-30% faster)

CUDA-Aware MPI with GPUDirect RDMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GPUDirect RDMA**: Direct GPU-to-GPU transfer over InfiniBand without CPU involvement

**Requirements:**

- InfiniBand network
- CUDA-aware MPI compiled with GPUDirect support
- Mellanox OFED drivers

**Configuration:**

.. code-block:: bash

   # Enable GPUDirect RDMA
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export OMPI_MCA_btl_openib_want_cuda_gdr=1
   
   mpirun -np 4 \
       -mca pml ucx \
       -x UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy \
       ./jericho_mkII config.toml

**Performance**: 5-10x faster than standard MPI for large messages

Minimize Ghost Cell Width
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Trade-off:**

- Smaller ghost width → less communication
- But requires higher-order schemes or more frequent exchanges

**Current**: 2 ghost cell layers (optimal for second-order schemes)

Scaling Studies
---------------

Weak Scaling
~~~~~~~~~~~~

**Definition**: Keep work per rank constant, increase number of ranks

**Ideal**: Time remains constant as we add more ranks

**Test:**

.. list-table::
   :header-rows: 1

   * - Ranks
     - Local Grid
     - Global Grid
     - Particles/rank
     - Time (s)
     - Efficiency
   * - 1
     - 256×256
     - 256×256
     - 6.5M
     - 120
     - 100%
   * - 4
     - 256×256
     - 512×512
     - 6.5M
     - 125
     - 96%
   * - 16
     - 256×256
     - 1024×1024
     - 6.5M
     - 135
     - 89%
   * - 64
     - 256×256
     - 2048×2048
     - 6.5M
     - 150
     - 80%

**Efficiency** = :math:`T_1 / T_N` where :math:`T_1` is time for 1 rank

Strong Scaling
~~~~~~~~~~~~~~

**Definition**: Keep total work constant, increase number of ranks

**Ideal**: Time decreases proportionally (2x ranks → 0.5x time)

**Test:**

.. list-table::
   :header-rows: 1

   * - Ranks
     - Local Grid
     - Time (s)
     - Speedup
     - Efficiency
   * - 1
     - 1024×1024
     - 960
     - 1.0x
     - 100%
   * - 4
     - 512×512
     - 250
     - 3.8x
     - 96%
   * - 16
     - 256×256
     - 68
     - 14.1x
     - 88%
   * - 64
     - 128×128
     - 20
     - 48x
     - 75%

**Efficiency** = Speedup / N

Scaling Limits
~~~~~~~~~~~~~~

**Factors limiting scaling:**

1. **Communication overhead**: Increases with more ranks
2. **Load imbalance**: Some ranks finish before others
3. **Collective operations**: Global reductions are expensive
4. **Ghost cell ratio**: Small subdomains → high ghost/interior ratio

**Rules of thumb:**

- Each rank should have ≥ 64×64 interior cells
- Aim for ≥ 1M particles per rank
- Communication should be < 10% of compute time

Debugging MPI Programs
-----------------------

Common Issues
~~~~~~~~~~~~~

**Deadlocks:**

.. code-block:: cpp

   // BAD: Can deadlock if message is large
   MPI_Send(..., neighbor_left, ...);
   MPI_Recv(..., neighbor_left, ...);
   
   // GOOD: Use non-blocking or Sendrecv
   MPI_Sendrecv(..., neighbor_left, ..., neighbor_left, ...);

**Race conditions:**

.. code-block:: cpp

   // BAD: Multiple ranks writing to same file
   if (rank == 0 || rank == 1) {
       FILE* f = fopen("output.txt", "w");  // Race!
       // ...
   }
   
   // GOOD: Only one rank writes
   if (rank == 0) {
       FILE* f = fopen("output.txt", "w");
       // ...
   }

Debugging Tools
~~~~~~~~~~~~~~~

**Print debug info:**

.. code-block:: cpp

   printf("[Rank %d] x_min=%.2f, x_max=%.2f, n_particles=%zu\n",
          rank, x_min_local, x_max_local, particles.count);

**Use MPI debugger (DDT/TotalView):**

.. code-block:: bash

   ddt mpirun -np 4 ./jericho_mkII config.toml

**Check for hangs:**

.. code-block:: bash

   # Run with timeout
   timeout 60 mpirun -np 4 ./jericho_mkII config.toml || echo "TIMEOUT!"

Testing MPI
~~~~~~~~~~~

**Unit test ghost cell exchange:**

.. code-block:: cpp

   TEST(MPITest, GhostCellExchange) {
       // Initialize with rank-specific pattern
       for (int i = 0; i < nx * ny; ++i) {
           fields.Ex[i] = rank * 100.0 + i;
       }
       
       // Exchange
       exchange_ghost_cells(fields);
       
       // Verify ghost cells contain neighbor data
       if (neighbor_left >= 0) {
           // Check left ghost cells
           for (int j = 0; j < ny; ++j) {
               int idx = j * nx;  // Leftmost cell (ghost)
               // Should contain data from right edge of left neighbor
               EXPECT_NEAR(fields.Ex[idx], neighbor_left * 100.0 + ..., 1e-10);
           }
       }
   }

Best Practices
--------------

1. **Use CUDA-aware MPI** if available (huge performance gain)
2. **Overlap communication and computation** when possible
3. **Minimize global collectives** (expensive at scale)
4. **Balance subdomain sizes** for uniform work distribution
5. **Test at multiple scales** (1, 4, 16, 64 ranks)
6. **Profile communication** (use nvprof, Nsight Systems)
7. **Handle edge cases** (periodic vs. open boundaries)

Configuration Example
---------------------

.. code-block:: toml

   [mpi]
   npx = 4        # 4 processes in x
   npy = 4        # 4 processes in y
   # Total: 16 ranks
   
   [cuda]
   device_id = -1               # Auto-assign GPUs
   use_cuda_aware_mpi = true    # Enable CUDA-aware MPI

Running:

.. code-block:: bash

   # Single node (4 GPUs)
   mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 ./jericho_mkII config.toml
   
   # Multi-node (16 GPUs across 4 nodes)
   mpirun -np 16 --hostfile hostfile \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       -mca pml ucx -x UCX_TLS=rc,sm,cuda_copy,cuda_ipc \
       ./jericho_mkII config.toml

See Also
--------

- :doc:`architecture` - Overall design
- :doc:`cuda_kernels` - GPU implementation
- :doc:`performance_tuning` - Optimization strategies
- :doc:`running_simulations` - Practical usage
