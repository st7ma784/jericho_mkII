Troubleshooting
===============

This guide helps diagnose and resolve common issues with Jericho Mk II.

Build Issues
------------

"nvcc not found"
~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   CMake Error: Could not find CUDA compiler

**Cause:** CUDA toolkit not installed or not in PATH

**Solution:**

.. code-block:: bash

   # Check if nvcc exists
   which nvcc
   
   # If not found, add CUDA to PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   
   # Add to ~/.bashrc to make permanent
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

"compute capability mismatch"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   nvcc fatal: Unsupported gpu architecture 'compute_XX'

**Cause:** Wrong GPU architecture specified in CMake

**Solution:**

.. code-block:: bash

   # Check your GPU's compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   # Output: 8.6 (for example)
   
   # Clean and reconfigure
   cd build
   rm -rf CMakeCache.txt CMakeFiles/
   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # Use 86 for compute 8.6
   make -j

**Common compute capabilities:**

- 6.0: Pascal (P100)
- 7.0: Volta (V100)
- 7.5: Turing (RTX 2080)
- 8.0: Ampere (A100)
- 8.6: Ampere (RTX 3080, 3090)
- 9.0: Hopper (H100)

"MPI not found"
~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   Could NOT find MPI (missing: MPI_C_FOUND MPI_CXX_FOUND)

**Solution:**

.. code-block:: bash

   # Install MPI
   sudo apt-get install libopenmpi-dev openmpi-bin
   
   # Or for MPICH
   sudo apt-get install libmpich-dev mpich
   
   # Verify
   which mpicc
   which mpicxx

"HDF5 not found"
~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   Could NOT find HDF5

**Solution:**

.. code-block:: bash

   # Install HDF5
   sudo apt-get install libhdf5-dev
   
   # Or specify HDF5 location
   cmake -DHDF5_ROOT=/path/to/hdf5 ..

Runtime Issues
--------------

"CUDA out of memory"
~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   CUDA error: out of memory
   cudaMalloc failed

**Cause:** Simulation requires more GPU memory than available

**Solutions:**

1. **Reduce particle count:**

   .. code-block:: toml

      [particles]
      particles_per_cell = 50  # Reduce from 100

2. **Reduce grid size:**

   .. code-block:: toml

      [domain]
      nx = 128  # Reduce from 256
      ny = 128

3. **Check available memory:**

   .. code-block:: bash

      nvidia-smi
      # Look at "Memory-Usage" column

4. **Close other GPU programs:**

   .. code-block:: bash

      # List GPU processes
      nvidia-smi
      
      # Kill process using GPU
      kill -9 <PID>

5. **Use different GPU:**

   .. code-block:: bash

      # Check which GPUs have free memory
      nvidia-smi
      
      # Use specific GPU
      CUDA_VISIBLE_DEVICES=1 ./jericho_mkII config.toml

"No CUDA-capable device"
~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   cudaGetDeviceCount returned 0
   no CUDA-capable device is detected

**Solutions:**

1. **Check if GPU exists:**

   .. code-block:: bash

      lspci | grep -i nvidia
      # Should show NVIDIA GPU

2. **Check driver:**

   .. code-block:: bash

      nvidia-smi
      # Should show GPU info
      
      # If fails, install driver
      sudo ubuntu-drivers autoinstall
      sudo reboot

3. **Check CUDA installation:**

   .. code-block:: bash

      nvcc --version
      # Should show CUDA version

"MPI_Init failed"
~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   MPI_Init failed

**Causes and solutions:**

1. **Wrong number of processes:**

   .. code-block:: toml

      [mpi]
      npx = 2
      npy = 2  # Total = 4
   
   .. code-block:: bash

      # Must use 4 processes
      mpirun -np 4 ./jericho_mkII config.toml  # Correct
      mpirun -np 2 ./jericho_mkII config.toml  # Wrong!

2. **Permission denied:**

   .. code-block:: bash

      # Check executable permissions
      ls -l jericho_mkII
      chmod +x jericho_mkII

3. **MPI not initialized:**

   .. code-block:: bash

      # Test MPI installation
      mpirun -np 2 hostname

Physics Issues
--------------

Energy Not Conserved
~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   WARNING: Energy conservation error: 5.2%

**Causes and solutions:**

1. **Timestep too large:**

   .. code-block:: toml

      [simulation]
      dt = 0.005  # Reduce from 0.01
   
   **Test:** Halve dt and check if error reduces

2. **CAM disabled:**

   .. code-block:: toml

      [physics]
      cam_method = true  # Must be true!

3. **Grid too coarse:**

   .. code-block:: toml

      [domain]
      nx = 256  # Increase from 128
      ny = 256

4. **Check diagnostics:**

   .. code-block:: python

      import pandas as pd
      diag = pd.read_csv('output/diagnostics.csv')
      
      E0 = diag['total_energy'].iloc[0]
      rel_error = abs((diag['total_energy'] - E0) / E0).max()
      print(f"Max energy error: {rel_error*100:.2f}%")

No Magnetic Reconnection
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Harris sheet remains stable, no X-point forms

**Solutions:**

1. **Add perturbation:**

   .. code-block:: toml

      [fields]
      init_type = "harris_sheet"
      B0 = 20.0e-9
      L_sheet = 1.0
      
      # Add small perturbation to trigger instability
      perturbation_amplitude = 0.01  # 1% perturbation

2. **Enable Hall term:**

   .. code-block:: toml

      [physics]
      ohms_law_hall = true  # Essential for reconnection!

3. **Check resistivity:**

   Magnetic reconnection requires some resistivity. Check config or add:

   .. code-block:: toml

      [physics]
      resistivity = 1.0e-7  # Small but non-zero

4. **Run longer:**

   Reconnection can take time to develop. Run for > 1000 steps.

5. **Check field strength:**

   Too weak or too strong fields can suppress reconnection.

Particles Escaping Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Particle count decreases rapidly

**Solutions:**

1. **Check boundary conditions:**

   .. code-block:: toml

      [boundaries]
      x_min = "periodic"  # Or "reflecting" to keep particles
      x_max = "periodic"
      y_min = "periodic"
      y_max = "periodic"

2. **Check timestep (CFL violation):**

   .. code-block:: toml

      [simulation]
      dt = 0.005  # Reduce if particles move > 1 cell per step

3. **Check initial velocities:**

   Velocities too high can cause particles to escape. Check:

   .. code-block:: toml

      [[particles.species]]
      temperature = 1.0e6  # Not 1.0e10!
      drift_vx = 0.0       # Not 1.0e8!

NaN Values Appearing
~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   WARNING: NaN detected in field Ex at step 1234

**Causes and solutions:**

1. **Divide by zero in density:**

   Occurs when density → 0. Enable floor:

   .. code-block:: toml

      [physics]
      density_floor = 1.0e3  # Minimum density [m^-3]

2. **Unstable timestep:**

   .. code-block:: toml

      [simulation]
      dt = 0.001  # Drastically reduce

3. **Corrupted initial conditions:**

   Check input files for NaN/Inf values

4. **Bug:** Report to developers with:

   - Configuration file
   - Last working step
   - Hardware info (GPU model, CUDA version)

Performance Issues
------------------

Simulation Too Slow
~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- < 0.1 Gparticles/s throughput
- Hours for 1000 timesteps

**Diagnostics:**

.. code-block:: bash

   # Check GPU utilization
   nvidia-smi dmon -s u
   # Should show > 80% GPU utilization

**Solutions:**

1. **Reduce output frequency:**

   .. code-block:: toml

      [simulation]
      output_cadence = 1000  # Instead of 10

2. **Disable particle output:**

   .. code-block:: toml

      [output]
      particles = false

3. **Increase grid size or particle count:**

   Small problems don't fully utilize GPU:

   .. code-block:: toml

      [domain]
      nx = 512  # Up from 128
      ny = 512

4. **Check for CPU bottleneck:**

   .. code-block:: bash

      # Profile
      nvprof ./jericho_mkII config.toml
      # Look for long CPU times between kernels

5. **Use CUDA-aware MPI:**

   .. code-block:: toml

      [cuda]
      use_cuda_aware_mpi = true

Poor MPI Scaling
~~~~~~~~~~~~~~~~

**Symptom:** 4 GPUs not 4x faster than 1 GPU

**Diagnostics:**

.. code-block:: bash

   # Time 1 GPU
   time ./jericho_mkII config.toml
   
   # Time 4 GPUs
   time mpirun -np 4 ./jericho_mkII config.toml
   # Should be ~4x faster

**Solutions:**

1. **Subdomains too small:**

   Each rank needs ≥ 64×64 cells:

   .. code-block:: toml

      [domain]
      nx = 512  # At least 64 per rank
      ny = 512
      
      [mpi]
      npx = 4
      npy = 2
      # Each rank gets 128×256 ✓

2. **Check MPI communication time:**

   .. code-block:: bash

      # Use Nsight Systems
      nsys profile -o profile mpirun -np 4 ./jericho_mkII config.toml
      nsys-ui profile.qdrep
      # Look for MPI overhead

3. **Enable CUDA-aware MPI:**

   .. code-block:: bash

      # Check if available
      ompi_info --parsable | grep cuda_support
      
      # Enable
      [cuda]
      use_cuda_aware_mpi = true

GPU Thermal Throttling
~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Simulation starts fast, then slows down

**Diagnosis:**

.. code-block:: bash

   # Monitor temperature
   watch -n 1 nvidia-smi

**Solutions:**

1. **Improve cooling:**

   - Clean dust from fans
   - Improve case airflow
   - Lower ambient temperature

2. **Reduce power limit (if overheating):**

   .. code-block:: bash

      # Reduce to 80% power
      sudo nvidia-smi -pl 200  # For 250W GPU

3. **Check thermal paste** (for older GPUs)

Output Issues
-------------

"Permission denied" Writing Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   Error: Cannot create output directory

**Solutions:**

.. code-block:: bash

   # Check permissions
   ls -ld output/
   
   # Create directory with correct permissions
   mkdir -p output
   chmod 755 output
   
   # Or specify writable location
   [simulation]
   output_dir = "/tmp/jericho_output"

Output Files Corrupted
~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Cannot open HDF5 files

**Diagnosis:**

.. code-block:: bash

   # Check file integrity
   h5dump -H output/fields_001000.h5
   # Should show structure, not error

**Solutions:**

1. **Check disk space:**

   .. code-block:: bash

      df -h
      # Ensure sufficient space

2. **Check for crashes during write:**

   Enable more frequent checkpointing:

   .. code-block:: toml

      [simulation]
      checkpoint_cadence = 100  # More frequent

3. **Use checkpoint to recover:**

   .. code-block:: bash

      # Find last good checkpoint
      ls -lrt checkpoints/
      
      # Restart from it
      ./jericho_mkII --restart checkpoints/checkpoint_000900.h5 config.toml

Diagnostic Collection
---------------------

Before Reporting Issues
~~~~~~~~~~~~~~~~~~~~~~~

Collect this information:

1. **System info:**

   .. code-block:: bash

      # GPU info
      nvidia-smi
      nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
      
      # CUDA version
      nvcc --version
      
      # OS info
      uname -a
      lsb_release -a

2. **Build info:**

   .. code-block:: bash

      # CMake configuration
      cat build/CMakeCache.txt | grep -E "(CUDA|MPI|HDF5)"

3. **Configuration file:**

   Include your ``config.toml``

4. **Error messages:**

   Save complete error output:

   .. code-block:: bash

      ./jericho_mkII config.toml 2>&1 | tee error.log

5. **Last working state:**

   - Last successful timestep
   - Particle count evolution
   - Energy conservation up to failure

Getting Help
------------

Self-Help Resources
~~~~~~~~~~~~~~~~~~~

1. **Documentation:** https://st7ma784.github.io/jericho_mkII/
2. **Example configs:** ``examples/`` directory
3. **Test cases:** ``tests/`` directory

Reporting Bugs
~~~~~~~~~~~~~~

Create GitHub issue with:

1. **Title:** Brief description
2. **Environment:**

   - GPU model and compute capability
   - CUDA version
   - OS and version
   - MPI implementation

3. **To Reproduce:**

   - Configuration file
   - Command line used
   - Steps to reproduce

4. **Expected behavior:** What should happen

5. **Actual behavior:** What actually happens

6. **Logs:**

   - Error messages
   - Last 50 lines of output
   - ``diagnostics.csv`` if available

7. **Additional context:**

   - Did it work before?
   - Changed anything recently?

Example bug report:

.. code-block:: markdown

   **Title:** CUDA out of memory with 256x256 grid on RTX 3080
   
   **Environment:**
   - GPU: NVIDIA GeForce RTX 3080 (10 GB, compute 8.6)
   - CUDA: 11.8
   - OS: Ubuntu 22.04
   - MPI: OpenMPI 4.1.2
   
   **To Reproduce:**
   1. Use attached config.toml (256x256 grid, 100 ppc, 2 species)
   2. Run: `./jericho_mkII config.toml`
   3. Crashes at step 0 with "cudaMalloc failed"
   
   **Expected:** Should run (only ~400 MB needed)
   
   **Actual:** Crashes immediately with OOM error
   
   **Logs:**
   ```
   [attach error.log]
   ```

Community Support
~~~~~~~~~~~~~~~~~

- **GitHub Issues:** https://github.com/st7ma784/jericho_mkII/issues
- **Email:** j.wiggs@lancaster.ac.uk
- **Documentation:** Read the docs first!

Quick Checks
------------

Before asking for help, try these:

.. code-block:: bash

   # 1. Does the example config work?
   ./jericho_mkII examples/minimal_test.toml
   
   # 2. Is your config valid?
   ./jericho_mkII --dry-run config.toml
   
   # 3. Is GPU working?
   nvidia-smi
   
   # 4. Is MPI working?
   mpirun -np 2 hostname
   
   # 5. Is build correct?
   ldd jericho_mkII  # Check linked libraries
   
   # 6. Clean and rebuild?
   rm -rf build && mkdir build && cd build && cmake .. && make -j

Common Mistakes
---------------

1. **Forgetting to set npx × npy = total MPI ranks**
2. **Timestep too large (energy not conserved)**
3. **Output directory doesn't exist**
4. **Wrong GPU architecture in CMake**
5. **CUDA-aware MPI enabled but not available**
6. **Running on wrong GPU** (use CUDA_VISIBLE_DEVICES)
7. **Periodic boundaries when open needed** (or vice versa)
8. **Forgetting to enable Hall term** (for reconnection)
9. **Too few particles per cell** (noisy statistics)
10. **Writing output too frequently** (I/O bottleneck)

See Also
--------

- :doc:`getting_started` - Installation guide
- :doc:`running_simulations` - Usage guide
- :doc:`performance_tuning` - Optimization tips
- GitHub Issues - Community Q&A
