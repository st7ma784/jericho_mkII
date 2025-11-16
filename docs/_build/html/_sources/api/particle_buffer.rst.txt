API Reference: ParticleBuffer
==============================

Class: ``jericho::ParticleBuffer``
-----------------------------------

GPU-optimized Structure of Arrays (SoA) particle buffer for efficient particle storage and manipulation.

Overview
--------

The ``ParticleBuffer`` class implements a Structure of Arrays (SoA) memory layout for particle data, designed for optimal GPU performance through coalesced memory access patterns.

**Key features:**

- Coalesced memory access (10-50x GPU speedup vs AoS)
- Dynamic particle management (add/remove particles)
- CUDA device memory resident
- Efficient particle migration for MPI
- Support for multiple species via type field

Header File
-----------

.. code-block:: cpp

   #include "particle_buffer.h"

Class Declaration
-----------------

.. code-block:: cpp

   namespace jericho {
   
   class ParticleBuffer {
   public:
       // Data members (GPU device pointers)
       double* x;          // X positions [m]
       double* y;          // Y positions [m]
       double* vx;         // X velocities [m/s]
       double* vy;         // Y velocities [m/s]
       double* weight;     // Statistical weights
       uint8_t* type;      // Species type index
       bool* active;       // Active flag
       
       size_t capacity;    // Allocated capacity
       size_t count;       // Active particle count
       int device_id;      // CUDA device ID
       
       // Free slot management
       size_t* free_slots; // Stack of free indices
       size_t free_count;  // Number of free slots
       
       // Constructors/Destructors
       ParticleBuffer(size_t initial_capacity, int device_id = 0);
       ~ParticleBuffer();
       
       // Memory management
       void allocate(size_t capacity);
       void destroy();
       void resize(size_t new_capacity);
       void compact();
       
       // Particle operations
       size_t add_particle(double px, double py, double pvx, double pvy,
                          double pweight, uint8_t ptype);
       void remove_particle(size_t idx);
       void add_particles_batch(const double* positions, 
                               const double* velocities,
                               const double* weights, 
                               const uint8_t* types,
                               size_t n_particles);
       
       // Data transfer
       void copy_to_device(const double* h_x, const double* h_y,
                          const double* h_vx, const double* h_vy,
                          const double* h_weight, const uint8_t* h_type,
                          size_t n, size_t offset = 0);
       void copy_to_host(double* h_x, double* h_y, 
                        double* h_vx, double* h_vy,
                        double* h_weight, uint8_t* h_type,
                        size_t n, size_t offset = 0) const;
       
       // Accessors
       size_t get_count() const { return count; }
       size_t get_capacity() const { return capacity; }
       size_t get_memory_bytes() const;
       void print_stats() const;
   };
   
   } // namespace jericho

Constructors
------------

ParticleBuffer(size_t initial_capacity, int device_id = 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a particle buffer on specified GPU.

**Parameters:**

- ``initial_capacity`` (size_t): Initial number of particle slots to allocate
- ``device_id`` (int): CUDA device ID (default: 0, use -1 for auto-assignment)

**Example:**

.. code-block:: cpp

   // Create buffer for 1 million particles on GPU 0
   jericho::ParticleBuffer particles(1000000, 0);

**Notes:**

- Allocates GPU memory immediately
- Use RAII or explicitly call ``destroy()`` to free memory
- Capacity can be increased later with ``resize()``

~ParticleBuffer()
~~~~~~~~~~~~~~~~~

Destructor - automatically frees GPU memory.

.. code-block:: cpp

   // Automatic cleanup
   {
       jericho::ParticleBuffer particles(1000000);
       // Use particles...
   } // Destructor called here, GPU memory freed

Member Functions
----------------

Memory Management
~~~~~~~~~~~~~~~~~

allocate(size_t capacity)
^^^^^^^^^^^^^^^^^^^^^^^^^

Allocate GPU memory for particle arrays.

**Parameters:**

- ``capacity`` (size_t): Number of particle slots to allocate

**Example:**

.. code-block:: cpp

   ParticleBuffer particles(0);  // Empty buffer
   particles.allocate(1000000);  // Allocate for 1M particles

destroy()
^^^^^^^^^

Free all GPU memory.

.. code-block:: cpp

   particles.destroy();
   // All GPU arrays freed, capacity=0, count=0

**Note:** Called automatically by destructor.

resize(size_t new_capacity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Resize buffer (may trigger reallocation and copy).

**Parameters:**

- ``new_capacity`` (size_t): New capacity (must be ≥ count)

**Throws:** ``std::runtime_error`` if ``new_capacity < count``

**Example:**

.. code-block:: cpp

   // Initially 1M capacity
   ParticleBuffer particles(1000000);
   
   // Need more space
   particles.resize(2000000);  // Doubles capacity
   
   // Allocates new memory, copies existing particles, frees old memory

**Performance:** Expensive operation (involves GPU-GPU copy). Avoid frequent resizes by over-allocating initially.

compact()
^^^^^^^^^

Remove inactive particles by compacting buffer.

**Example:**

.. code-block:: cpp

   // After many particle removals
   std::cout << "Active: " << particles.count << "\n";
   std::cout << "Capacity: " << particles.capacity << "\n";
   std::cout << "Free slots: " << particles.free_count << "\n";
   
   // Compact to reclaim memory
   particles.compact();
   
   // Now capacity ≈ count, free_count = 0

**When to use:**

- After significant particle removal (e.g., outflow boundaries)
- When ``free_count`` is large (> 10% of capacity)
- Before checkpointing (to reduce file size)

**Performance:** :math:`O(n)` GPU kernel, synchronizes device.

Particle Operations
~~~~~~~~~~~~~~~~~~~

add_particle(double px, double py, double pvx, double pvy, double pweight, uint8_t ptype)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a single particle (inflow boundary).

**Parameters:**

- ``px, py`` (double): Position [m]
- ``pvx, pvy`` (double): Velocity [m/s]
- ``pweight`` (double): Statistical weight (macroparticle multiplicity)
- ``ptype`` (uint8_t): Species type index (0=electrons, 1+=ions)

**Returns:** ``size_t`` - Index where particle was inserted, or ``-1`` if buffer full

**Example:**

.. code-block:: cpp

   // Add H+ particle at origin with 1000 km/s velocity
   size_t idx = particles.add_particle(
       0.0, 0.0,           // Position (x, y)
       1.0e6, 0.0,         // Velocity (vx, vy) [m/s]
       1.0,                // Weight
       1                   // Type (1 = H+)
   );
   
   if (idx != static_cast<size_t>(-1)) {
       std::cout << "Particle added at index " << idx << "\n";
   }

**Notes:**

- If buffer is full, automatically resizes with 1.5x growth factor
- Relatively slow (host-device synchronization), prefer batch operations

remove_particle(size_t idx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove particle at index (outflow boundary).

**Parameters:**

- ``idx`` (size_t): Index of particle to remove

**Example:**

.. code-block:: cpp

   // Remove particle at index 12345
   particles.remove_particle(12345);

**Notes:**

- Marks particle as inactive (``active[idx] = false``)
- Adds index to free_slots stack for reuse
- Actual memory not freed until ``compact()`` is called
- Fast operation (no memory movement)

add_particles_batch(...)
^^^^^^^^^^^^^^^^^^^^^^^^

Batch add particles (efficient for inflow boundaries).

**Parameters:**

- ``positions`` (const double*): Host array ``[x0,y0,x1,y1,...]`` (size: 2*n)
- ``velocities`` (const double*): Host array ``[vx0,vy0,vx1,vy1,...]`` (size: 2*n)
- ``weights`` (const double*): Host array (size: n)
- ``types`` (const uint8_t*): Host array (size: n)
- ``n_particles`` (size_t): Number of particles to add

**Example:**

.. code-block:: cpp

   // Prepare data on host
   size_t n = 10000;
   std::vector<double> positions(2*n);
   std::vector<double> velocities(2*n);
   std::vector<double> weights(n, 1.0);
   std::vector<uint8_t> types(n, 1);  // All H+
   
   // Initialize positions and velocities
   for (size_t i = 0; i < n; ++i) {
       positions[2*i] = /* x */;
       positions[2*i+1] = /* y */;
       velocities[2*i] = /* vx */;
       velocities[2*i+1] = /* vy */;
   }
   
   // Batch add
   particles.add_particles_batch(
       positions.data(), velocities.data(),
       weights.data(), types.data(), n);

**Performance:** Much faster than calling ``add_particle()`` in a loop (batched GPU transfer).

Data Transfer
~~~~~~~~~~~~~

copy_to_device(...)
^^^^^^^^^^^^^^^^^^^

Copy particle data from host to device.

**Parameters:**

- ``h_x, h_y, h_vx, h_vy, h_weight`` (const double*): Host arrays
- ``h_type`` (const uint8_t*): Host array
- ``n`` (size_t): Number of particles to copy
- ``offset`` (size_t): Offset in device arrays (default: 0)

**Example:**

.. code-block:: cpp

   // Initialize particles on host
   std::vector<double> h_x(n), h_y(n), h_vx(n), h_vy(n), h_weight(n);
   std::vector<uint8_t> h_type(n);
   
   // ... fill arrays ...
   
   // Copy to GPU
   particles.copy_to_device(
       h_x.data(), h_y.data(), h_vx.data(), h_vy.data(),
       h_weight.data(), h_type.data(), n);

copy_to_host(...)
^^^^^^^^^^^^^^^^^

Copy particle data from device to host.

**Parameters:**

- ``h_x, h_y, h_vx, h_vy, h_weight`` (double*): Host arrays (pre-allocated)
- ``h_type`` (uint8_t*): Host array (pre-allocated)
- ``n`` (size_t): Number of particles to copy
- ``offset`` (size_t): Offset in device arrays (default: 0)

**Example:**

.. code-block:: cpp

   // Allocate host memory
   size_t n = particles.get_count();
   std::vector<double> h_x(n), h_y(n), h_vx(n), h_vy(n), h_weight(n);
   std::vector<uint8_t> h_type(n);
   
   // Copy from GPU
   particles.copy_to_host(
       h_x.data(), h_y.data(), h_vx.data(), h_vy.data(),
       h_weight.data(), h_type.data(), n);
   
   // Now h_x, h_y, etc. contain particle data

Accessors
~~~~~~~~~

get_count()
^^^^^^^^^^^

Get number of active particles.

**Returns:** ``size_t`` - Active particle count

.. code-block:: cpp

   size_t n = particles.get_count();
   std::cout << "Active particles: " << n << "\n";

get_capacity()
^^^^^^^^^^^^^^

Get buffer capacity.

**Returns:** ``size_t`` - Total allocated slots

.. code-block:: cpp

   size_t cap = particles.get_capacity();
   std::cout << "Buffer capacity: " << cap << "\n";
   std::cout << "Utilization: " << 100.0 * particles.get_count() / cap << "%\n";

get_memory_bytes()
^^^^^^^^^^^^^^^^^^

Get memory usage in bytes.

**Returns:** ``size_t`` - Total GPU memory used

.. code-block:: cpp

   size_t bytes = particles.get_memory_bytes();
   std::cout << "GPU memory: " << bytes / 1e6 << " MB\n";

**Formula:**

.. math::

   \text{bytes} = \text{capacity} \times (5 \times 8 + 1 + 1) = \text{capacity} \times 42

print_stats()
^^^^^^^^^^^^^

Print buffer statistics to stdout.

.. code-block:: cpp

   particles.print_stats();

**Output:**

.. code-block:: text

   ParticleBuffer Statistics:
     Device: 0
     Capacity: 1000000
     Active count: 856234
     Free slots: 143766
     Memory: 42.0 MB
     Utilization: 85.6%

Helper Functions
----------------

initialize_uniform(...)
~~~~~~~~~~~~~~~~~~~~~~~

Initialize particles uniformly in rectangular domain.

.. code-block:: cpp

   void initialize_uniform(
       ParticleBuffer& buffer,
       double x_min, double x_max, double y_min, double y_max,
       int particles_per_cell, int nx, int ny,
       uint8_t type, double weight,
       double vx_mean, double vy_mean, double v_thermal,
       unsigned int seed = 42);

**Example:**

.. code-block:: cpp

   ParticleBuffer particles(0);
   
   // Initialize H+ uniformly in domain
   initialize_uniform(
       particles,
       -10.0, 10.0, -10.0, 10.0,  // Domain bounds
       100,                        // 100 particles per cell
       256, 256,                   // 256x256 grid
       1,                          // Type = H+
       1.0,                        // Weight
       0.0, 0.0,                   // No drift
       1.0e5,                      // 100 km/s thermal velocity
       42                          // Random seed
   );
   
   std::cout << "Initialized " << particles.get_count() << " particles\n";

Usage Examples
--------------

Example 1: Basic Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "particle_buffer.h"
   #include <iostream>
   
   int main() {
       // Create buffer for 1M particles on GPU 0
       jericho::ParticleBuffer particles(1000000, 0);
       
       // Add some particles
       for (int i = 0; i < 100; ++i) {
           particles.add_particle(
               i * 0.1, 0.0,    // Position
               1.0e6, 0.0,      // Velocity
               1.0, 1           // Weight, type
           );
       }
       
       std::cout << "Added " << particles.get_count() << " particles\n";
       
       particles.print_stats();
       
       return 0;
   }

Example 2: Particle Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   void migrate_particles(ParticleBuffer& particles, 
                         double x_min, double x_max) {
       // Identify particles to remove (left domain)
       std::vector<size_t> to_remove;
       
       // Copy positions to host
       size_t n = particles.get_count();
       std::vector<double> h_x(n);
       particles.copy_to_host(h_x.data(), nullptr, nullptr, nullptr,
                             nullptr, nullptr, n);
       
       // Check which particles left domain
       for (size_t i = 0; i < n; ++i) {
           if (h_x[i] < x_min || h_x[i] >= x_max) {
               to_remove.push_back(i);
           }
       }
       
       // Remove them
       for (size_t idx : to_remove) {
           particles.remove_particle(idx);
       }
       
       std::cout << "Removed " << to_remove.size() << " particles\n";
       
       // Compact if many removed
       if (particles.free_count > 0.1 * particles.capacity) {
           particles.compact();
       }
   }

Performance Notes
-----------------

**Memory layout comparison:**

.. list-table::
   :header-rows: 1

   * - Layout
     - Memory pattern
     - GPU performance
   * - AoS (bad)
     - ``[x0,y0,vx0,vy0][x1,y1,vx1,vy1]...``
     - Scattered reads (slow)
   * - SoA (good)
     - ``[x0,x1,x2,...][y0,y1,y2,...]...``
     - Coalesced reads (fast)

**Performance gains:**

- 10-50x speedup on GPU vs AoS
- 4-8x speedup on CPU (SIMD vectorization)
- Better cache utilization

See Also
--------

- :doc:`../architecture` - Overall design
- :doc:`../cuda_kernels` - GPU kernels using ParticleBuffer
- ``include/particle_buffer.h`` - Full header file
- ``src/particle_buffer.cpp`` - Implementation
