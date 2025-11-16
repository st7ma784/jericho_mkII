Performance Tuning
==================

This guide provides practical tips and strategies for optimizing Jericho Mk II performance on GPU and multi-GPU systems.

Performance Overview
--------------------

Target Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Metric
     - Target
     - How to Measure
   * - Particle throughput
     - > 1 Gparticles/s per GPU
     - Runtime output
   * - GPU utilization
     - > 80%
     - ``nvidia-smi dmon``
   * - Memory bandwidth
     - > 70% of peak
     - ``nvprof --metrics dram_utilization``
   * - Energy conservation
     - |ΔE|/E < 1% per 1000 steps
     - Check ``diagnostics.csv``
   * - MPI efficiency (weak scaling)
     - > 80% at 64 ranks
     - Scaling studies
   * - Time per timestep
     - < 100 ms for 1M particles
     - Profile output

Typical Performance
~~~~~~~~~~~~~~~~~~~

**Single GPU (NVIDIA A100):**

.. list-table::
   :header-rows: 1

   * - Configuration
     - Particles
     - Grid
     - Time/step
     - Throughput
   * - Small
     - 100K
     - 128×128
     - 5 ms
     - 20 Gparticles/s
   * - Medium
     - 1M
     - 256×256
     - 25 ms
     - 40 Gparticles/s
   * - Large
     - 10M
     - 512×512
     - 180 ms
     - 56 Gparticles/s

**Multi-GPU scaling (4x A100):**

- 4M particles, 512×512 grid: ~60 ms/step (4x speedup vs 1 GPU)
- 16M particles, 1024×1024 grid: ~200 ms/step (3.6x speedup)

Configuration Tuning
--------------------

Timestep Selection
~~~~~~~~~~~~~~~~~~

**CFL condition:**

The timestep must satisfy:

.. math::

   \Delta t < \min\left(\frac{\Delta x}{v_{\max}}, \frac{1}{\omega_{pe}}, \frac{1}{\Omega_i}\right)

**In practice:**

.. code-block:: toml

   [simulation]
   dt = 0.01  # Start here (0.01 ion gyroperiods)

**If energy drifts:**

- Reduce dt by 2x, check if energy conservation improves
- If yes → timestep was too large
- If no → check physics settings (CAM, Hall term)

**If simulation is too slow:**

- Increase dt carefully (max 2x at a time)
- Monitor energy conservation
- Never exceed dt = 0.1 (unstable for most cases)

Particles Per Cell
~~~~~~~~~~~~~~~~~~

**Trade-off:**

- More particles → better statistics, slower
- Fewer particles → noisier, faster

.. code-block:: toml

   [particles]
   particles_per_cell = 100  # Baseline
   
   # For faster runs (acceptable for scoping studies):
   particles_per_cell = 25
   
   # For high-quality results:
   particles_per_cell = 200

**Effect on performance:**

.. list-table::
   :header-rows: 1

   * - PPC
     - Particle count
     - Time/step
     - Statistical noise
   * - 25
     - 1.6M
     - 10 ms
     - ±5%
   * - 50
     - 3.3M
     - 18 ms
     - ±3%
   * - 100
     - 6.6M
     - 35 ms
     - ±2%
   * - 200
     - 13.1M
     - 68 ms
     - ±1%

Grid Resolution
~~~~~~~~~~~~~~~

**Physics constraint:**

Grid spacing should resolve Debye length:

.. math::

   \Delta x \lesssim 0.5 \lambda_D = 0.5 \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}

**Computational constraint:**

.. code-block:: toml

   [domain]
   nx = 256  # Baseline
   ny = 256
   
   # For faster runs:
   nx = 128
   ny = 128
   
   # For high resolution:
   nx = 512
   ny = 512

**Effect:**

- 2x grid resolution → 4x cells → ~4x slower (if particle-limited)
- But: Doubling grid with same PPC → 4x more particles → ~8x slower

**Recommendation:** Start with 128×128, increase only if physics requires

Output Cadence
~~~~~~~~~~~~~~

**Writing output is slow!**

.. code-block:: toml

   [simulation]
   output_cadence = 100   # Write every 100 steps
   
   # For production (fewer files):
   output_cadence = 1000
   
   # For debugging (more frequent):
   output_cadence = 10

**Effect:**

- Writing fields: ~50-200 ms per output
- Writing particles: ~100-500 ms per output
- At output_cadence=10, I/O can dominate runtime!

**Best practice:**

- Use large cadence for production (500-1000)
- Use small cadence for debugging (10-50)
- Disable particle output if not needed:

  .. code-block:: toml

     [output]
     particles = false  # Saves a lot of time!

CUDA Optimization
-----------------

Threads Per Block
~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [cuda]
   threads_per_block = 256  # Baseline (good for most GPUs)

**Try different values:**

.. list-table::
   :header-rows: 1

   * - Threads/block
     - Occupancy
     - Performance
     - Best for
   * - 128
     - 50-75%
     - Moderate
     - Old GPUs (compute 6.0)
   * - 256
     - 75-100%
     - Good
     - Most GPUs (default)
   * - 512
     - 75-100%
     - Best
     - Ampere/Hopper (compute 8.0+)
   * - 1024
     - 100%
     - Variable
     - Register-limited kernels

**How to test:**

.. code-block:: bash

   # Benchmark different settings
   for threads in 128 256 512; do
       echo "Testing threads_per_block=$threads"
       # Edit config.toml: threads_per_block = $threads
       time ./jericho_mkII config.toml
   done

Memory Pool Size
~~~~~~~~~~~~~~~~

.. code-block:: toml

   [cuda]
   memory_pool_size = 0  # Auto (default)
   
   # For large simulations, pre-allocate:
   memory_pool_size = 8192  # 8 GB

**Benefits:**

- Avoids runtime allocation overhead
- Reduces memory fragmentation
- Faster particle insertion/removal

**Drawback:**

- Uses memory even if not needed
- May run out if too small

**Recommendation:** Use 0 (auto) unless you see many allocations in profile

GPU Selection
~~~~~~~~~~~~~

**For multi-GPU systems:**

.. code-block:: bash

   # Let code auto-assign
   ./jericho_mkII config.toml
   
   # Or manually specify
   CUDA_VISIBLE_DEVICES=1 ./jericho_mkII config.toml
   
   # Check which GPU is faster
   nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv

**Prefer:**

- Newer architecture (Ampere > Turing > Volta)
- Higher memory bandwidth
- More SMs (streaming multiprocessors)

MPI Optimization
----------------

Domain Decomposition
~~~~~~~~~~~~~~~~~~~~

**Rule 1: Match aspect ratios**

If domain is 1024×512 (2:1 aspect ratio):

.. code-block:: toml

   [mpi]
   # Good: 2:1 decomposition
   npx = 4
   npy = 2
   
   # Bad: 1:2 decomposition (mismatched)
   npx = 2
   npy = 4

**Rule 2: Keep subdomains large enough**

Each subdomain should have ≥ 64×64 interior cells:

.. code-block:: toml

   [domain]
   nx = 512
   ny = 512
   
   [mpi]
   # Good: 128×128 per rank
   npx = 4
   npy = 4
   
   # Bad: 64×64 per rank (too small, communication-heavy)
   npx = 8
   npy = 8

**Rule 3: Power-of-2 decompositions**

Use npx, npy as powers of 2 (1, 2, 4, 8, 16) for best MPI performance.

CUDA-Aware MPI
~~~~~~~~~~~~~~

**Enable if available:**

.. code-block:: toml

   [cuda]
   use_cuda_aware_mpi = true

**Verify:**

.. code-block:: bash

   # Check if MPI supports CUDA
   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

**Performance gain:** 2-5x faster communication

**If not available:**

- Upgrade MPI (OpenMPI 4.0+, MPICH 3.3+)
- Or build from source with ``--with-cuda`` flag

Overlap Communication
~~~~~~~~~~~~~~~~~~~~~

**Already implemented in code**, but you can tune:

- Increase local work to better hide communication
- Use larger subdomains (less boundary/interior ratio)

Profiling and Analysis
----------------------

Using nvidia-smi
~~~~~~~~~~~~~~~~

**Monitor in real-time:**

.. code-block:: bash

   # Terminal 1: Run simulation
   ./jericho_mkII config.toml
   
   # Terminal 2: Monitor GPU
   watch -n 0.5 nvidia-smi

**Look for:**

- GPU utilization: Should be > 80%
- Memory usage: Should not exceed GPU memory (causes swapping)
- Temperature: < 85°C (throttles if higher)
- Power usage: Near TDP means GPU is working hard

Using nvprof
~~~~~~~~~~~~

.. code-block:: bash

   # Profile kernel execution
   nvprof ./jericho_mkII config.toml 2>&1 | tee profile.txt

**Key metrics:**

.. code-block:: text

   Time(%)  Time      Calls  Avg        Min        Max   Name
   45.2%    2.105s    1000   2.105ms    2.098ms    2.112ms  advance_particles_kernel
   28.3%    1.317s    1000   1.317ms    1.310ms    1.324ms  deposit_charge_current_kernel
   15.1%    702.3ms   1000   702.3us    695.2us    710.1us  compute_electric_field_kernel
   8.7%     405.2ms   1000   405.2us    400.1us    410.5us  update_magnetic_field_kernel

**Analysis:**

- Particle push: 45% of time (expected, particle-heavy)
- Deposit: 28% (atomic operations, expected)
- Field updates: 24% (grid operations)

**If particle push is slow:**

- Check memory bandwidth: ``nvprof --metrics dram_utilization``
- Should be > 70% (memory-bound is expected)

**If deposit is slow:**

- Atomic contention (expected with many particles/cell)
- Try reducing particles_per_cell

Using Nsight Systems
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Detailed timeline
   nsys profile -o jericho_profile ./jericho_mkII config.toml
   
   # Open GUI
   nsys-ui jericho_profile.qdrep

**Look for:**

- GPU idle time (gaps between kernels)
- CPU-GPU synchronization overhead
- MPI communication time
- I/O bottlenecks

**Optimize:**

- Reduce CPU-GPU sync (use async operations)
- Overlap computation and communication
- Reduce I/O frequency

Memory Usage Optimization
--------------------------

Estimate Memory Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Particle memory:**

.. math::

   M_{\text{particles}} = 48 \times N_{\text{cells}} \times N_{\text{ppc}} \times N_{\text{species}} \text{ bytes}

**Field memory:**

.. math::

   M_{\text{fields}} = 8 \times N_{\text{fields}} \times N_{\text{cells}} \text{ bytes}

**Example:**

- 256×256 grid, 100 ppc, 2 species: 314 MB particles
- 256×256 grid, 10 fields: 52 MB fields
- Total: ~400 MB (fits easily on any GPU)

**For 1024×1024 grid, 200 ppc:**

- Particles: 5 GB
- Fields: 0.8 GB
- Total: ~6 GB (needs ≥ 8 GB GPU)

Reduce Memory Usage
~~~~~~~~~~~~~~~~~~~

**Option 1: Reduce particles per cell**

.. code-block:: toml

   [particles]
   particles_per_cell = 50  # Instead of 100

Saves 50% particle memory, but noisier statistics.

**Option 2: Reduce grid resolution**

.. code-block:: toml

   [domain]
   nx = 512  # Instead of 1024
   ny = 512

Saves 75% memory, but lower spatial resolution.

**Option 3: Use single precision (experimental)**

Not currently implemented, but could save 50% memory.

**Option 4: Remove unused species**

.. code-block:: toml

   [particles]
   # Remove O+ if not needed
   [[particles.species]]
   name = "H+"
   # ...

Common Performance Issues
--------------------------

Slow Particle Push
~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Low GPU utilization (< 50%)
- Low memory bandwidth (< 50%)

**Causes:**

1. **Too few particles**: GPU underutilized
   
   **Fix:** Increase particles_per_cell or grid resolution

2. **Divergent branches**: if/else in kernel
   
   **Fix:** Minimize conditionals (already optimized in code)

3. **Uncoalesced memory access**: Wrong data layout
   
   **Fix:** Use SoA (already implemented)

Slow Particle-to-Grid
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Long time in deposit_charge_current_kernel
- Many atomic operations

**Causes:**

1. **Atomic contention**: Many particles in same cell
   
   **Fix:** 
   - Reduce particles_per_cell
   - Use shared memory optimization (already implemented)

2. **Irregular particle distribution**: Load imbalance
   
   **Fix:** Use MPI domain decomposition

Poor MPI Scaling
~~~~~~~~~~~~~~~~~

**Symptoms:**

- 4 GPUs not 4x faster than 1 GPU
- High MPI communication time

**Causes:**

1. **Subdomains too small**: High communication/computation ratio
   
   **Fix:** Use larger subdomains (≥ 128×128 per rank)

2. **No CUDA-aware MPI**: Slow CPU staging
   
   **Fix:** Enable CUDA-aware MPI

3. **Load imbalance**: Some ranks finish before others
   
   **Fix:** Check particle distribution, use load balancing (experimental)

High Memory Usage
~~~~~~~~~~~~~~~~~

**Symptoms:**

- Out of memory error
- GPU memory usage > available

**Causes:**

1. **Too many particles**
   
   **Fix:** Reduce particles_per_cell or grid size

2. **Memory leak** (bug)
   
   **Fix:** Report issue on GitHub

3. **Multiple simulations on same GPU**
   
   **Fix:** Use CUDA_VISIBLE_DEVICES to assign different GPUs

Best Practices Summary
----------------------

**For production runs:**

1. Start with baseline config (256×256, 100 ppc, dt=0.01)
2. Run short test (100 steps) and check energy conservation
3. Profile to identify bottlenecks
4. Tune one parameter at a time
5. Document changes and performance impact
6. Use large output_cadence (500-1000)
7. Enable checkpointing (checkpoint_cadence=500)

**For development/debugging:**

1. Use small config (128×128, 25 ppc)
2. Small output_cadence (10-50)
3. Enable verbose output
4. Run short tests (10-100 steps)

**For scaling studies:**

1. Fix work per rank (weak scaling) or total work (strong scaling)
2. Measure time to solution
3. Plot scaling efficiency
4. Identify scaling bottlenecks (communication, load imbalance)

Performance Checklist
----------------------

Before running production simulations, verify:

.. code-block:: text

   ☐ Energy conservation < 1% per 1000 steps
   ☐ GPU utilization > 80% (check nvidia-smi)
   ☐ Memory usage < 90% of GPU memory
   ☐ Timestep satisfies CFL condition
   ☐ Adequate particles per cell (≥ 50 for statistics)
   ☐ Grid resolution adequate (≥ 2 cells per Debye length)
   ☐ Output cadence reasonable (not writing every step)
   ☐ CUDA-aware MPI enabled (if using multi-GPU)
   ☐ Subdomain sizes adequate (≥ 64×64 per rank)
   ☐ Profile shows no obvious bottlenecks

See Also
--------

- :doc:`architecture` - System design
- :doc:`cuda_kernels` - GPU implementation details
- :doc:`mpi_parallelism` - Multi-GPU scaling
- :doc:`troubleshooting` - Common issues
