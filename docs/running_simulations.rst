Running Simulations
===================

This guide covers practical aspects of running plasma simulations with Jericho Mk II, from basic usage to advanced techniques.

Basic Usage
-----------

Single GPU Simulation
~~~~~~~~~~~~~~~~~~~~~

The simplest way to run a simulation:

.. code-block:: bash

   # From build directory
   ./jericho_mkII path/to/config.toml

Example output:

.. code-block:: text

   ╔════════════════════════════════════════════════════════════╗
   ║           Jericho Mk II - GPU PIC-MHD Simulator           ║
   ║                      Version 2.0.0                        ║
   ╚════════════════════════════════════════════════════════════╝
   
   [Config] Loading: examples/reconnection.toml
   [System] GPU: NVIDIA RTX 3080 (compute 8.6, 10 GB)
   [Domain] Grid: 256x256, Box: [-10, 10] x [-10, 10] d_i
   [Particles] Species: H+ (1.0e6 m⁻³), O+ (1.0e5 m⁻³)
   [Particles] Total: 6,553,600 macroparticles
   [Fields] Initial: Harris sheet, B0 = 20 nT
   [Boundaries] All periodic
   [Physics] Boris pusher: ON, CAM: ON, Hall: ON
   
   Starting simulation...
   
   Step     0 | t=  0.00 | KE= 4.532e-11 J | EM= 1.234e-12 J | dE/E= 0.00%
   Step   100 | t=  1.00 | KE= 4.534e-11 J | EM= 1.232e-12 J | dE/E= 0.02%
   Step   200 | t=  2.00 | KE= 4.536e-11 J | EM= 1.230e-12 J | dE/E= 0.04%
   ...
   Step 10000 | t=100.00 | KE= 4.578e-11 J | EM= 1.189e-12 J | dE/E= 0.45%
   
   Simulation complete!
   Total time: 14m 23s (11.6 particles/μs)
   Output: ./output/reconnection

Command Line Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./jericho_mkII [OPTIONS] config.toml

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--version``
     - Print version and exit
   * - ``--help``
     - Print help message
   * - ``--verbose``
     - Enable verbose output
   * - ``--dry-run``
     - Parse config without running
   * - ``--device <id>``
     - Override CUDA device ID
   * - ``--output-dir <path>``
     - Override output directory
   * - ``--restart <checkpoint>``
     - Restart from checkpoint

**Examples:**

.. code-block:: bash

   # Verbose output
   ./jericho_mkII --verbose config.toml
   
   # Use specific GPU
   ./jericho_mkII --device 1 config.toml
   
   # Restart from checkpoint
   ./jericho_mkII --restart checkpoints/checkpoint_005000.h5 config.toml

Multi-GPU Simulations
---------------------

MPI Basics
~~~~~~~~~~

For multi-GPU simulations, use MPI:

.. code-block:: bash

   # 4 GPUs (2x2 decomposition)
   mpirun -np 4 ./jericho_mkII config.toml

**Prerequisites:**

1. Configure MPI decomposition in config file:

   .. code-block:: toml

      [mpi]
      npx = 2  # 2 processes in x
      npy = 2  # 2 processes in y

2. Ensure ``npx * npy = total MPI ranks``

3. For multi-node: Use CUDA-aware MPI

GPU Assignment
~~~~~~~~~~~~~~

**Automatic assignment:**

By default, MPI rank ``i`` uses GPU ``i % num_gpus``:

.. code-block:: bash

   # 4 ranks, 4 GPUs → each rank gets one GPU
   mpirun -np 4 ./jericho_mkII config.toml

**Manual assignment:**

.. code-block:: bash

   # Specify visible GPUs
   mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 ./jericho_mkII config.toml
   
   # Or per-rank (for multi-node)
   mpirun -np 8 \
       -host node1 -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       -host node2 -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       ./jericho_mkII config.toml

**In config file:**

.. code-block:: toml

   [cuda]
   device_id = -1  # Auto-assign (recommended)
   # device_id = 0  # Force specific GPU (single-node only)

Multi-Node Simulations
~~~~~~~~~~~~~~~~~~~~~~

**Create hostfile:**

.. code-block:: bash

   cat > hostfile << EOF
   node1 slots=4
   node2 slots=4
   node3 slots=4
   node4 slots=4
   EOF

**Run with hostfile:**

.. code-block:: bash

   mpirun -np 16 --hostfile hostfile \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       -mca pml ob1 -mca btl openib,self,sm \
       ./jericho_mkII config.toml

**For InfiniBand:**

.. code-block:: bash

   # Enable GPU Direct RDMA
   mpirun -np 16 --hostfile hostfile \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
       -mca pml ucx \
       -x UCX_TLS=rc,sm,cuda_copy,cuda_ipc \
       ./jericho_mkII config.toml

**Check CUDA-aware MPI:**

.. code-block:: bash

   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

Should output ``true``.

Output Files
------------

File Structure
~~~~~~~~~~~~~~

After running, output directory contains:

.. code-block:: text

   output/
   ├── diagnostics.csv           # Time series data
   ├── fields_000000.h5          # Initial fields
   ├── fields_000100.h5          # Fields at t=1.0
   ├── fields_000200.h5          # Fields at t=2.0
   ├── ...
   ├── particles_000000.h5       # Initial particles
   ├── particles_000100.h5
   ├── ...
   └── checkpoints/
       ├── checkpoint_000500.h5  # Restart checkpoint
       ├── checkpoint_001000.h5
       └── ...

Field Output Files
~~~~~~~~~~~~~~~~~~

HDF5 files contain 2D field arrays:

.. code-block:: python

   import h5py
   
   with h5py.File('output/fields_001000.h5', 'r') as f:
       print("Available datasets:", list(f.keys()))
       # ['Ex', 'Ey', 'Bz', 'density', 'current_x', 'current_y', 'time', 'step']
       
       # Load magnetic field
       Bz = f['Bz'][:]           # Shape: (ny, nx)
       time = f['time'][()]      # Scalar: current time
       step = f['step'][()]      # Scalar: timestep number
       
       # Metadata
       print("Grid shape:", Bz.shape)
       print("Time:", time, "ion gyroperiods")

**Available fields:**

- ``Ex, Ey``: Electric field [V/m]
- ``Bz``: Magnetic field (2.5D) [T]
- ``density``: Number density [m⁻³]
- ``current_x, current_y``: Current density [A/m²]
- ``temperature``: Temperature [K] (if enabled)
- ``pressure``: Pressure [Pa] (if enabled)

Particle Output Files
~~~~~~~~~~~~~~~~~~~~~

HDF5 files contain particle arrays:

.. code-block:: python

   with h5py.File('output/particles_001000.h5', 'r') as f:
       print("Particle groups:", list(f.keys()))
       # ['H+', 'O+', 'time', 'step']
       
       # Load H+ particles
       h_plus = f['H+']
       x = h_plus['x'][:]        # Shape: (n_particles,)
       y = h_plus['y'][:]
       vx = h_plus['vx'][:]
       vy = h_plus['vy'][:]
       weight = h_plus['weight'][:]
       
       print(f"H+ particles: {len(x)}")
       print(f"Mean velocity: vx={vx.mean():.2e}, vy={vy.mean():.2e} m/s")

Diagnostics CSV
~~~~~~~~~~~~~~~

Time series of integrated quantities:

.. code-block:: python

   import pandas as pd
   
   diag = pd.read_csv('output/diagnostics.csv')
   print(diag.columns)
   # ['step', 'time', 'kinetic_energy', 'em_energy', 'total_energy',
   #  'energy_change', 'momentum_x', 'momentum_y', 'n_particles']
   
   # Plot energy conservation
   import matplotlib.pyplot as plt
   plt.plot(diag['time'], diag['total_energy'])
   plt.xlabel('Time [ion gyroperiods]')
   plt.ylabel('Total Energy [J]')
   plt.title('Energy Conservation')
   plt.show()

Checkpoints
~~~~~~~~~~~

Checkpoint files contain full simulation state for restart:

.. code-block:: python

   with h5py.File('checkpoints/checkpoint_001000.h5', 'r') as f:
       # Contains everything needed to restart
       print("Groups:", list(f.keys()))
       # ['fields', 'particles', 'mpi_state', 'config', 'metadata']
       
       # Restart simulation from this point
       time = f['metadata/time'][()]
       step = f['metadata/step'][()]

**Restarting:**

.. code-block:: bash

   ./jericho_mkII --restart checkpoints/checkpoint_001000.h5 config.toml

Visualization
-------------

Quick Plots with Python
~~~~~~~~~~~~~~~~~~~~~~~~

**Plot magnetic field:**

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np
   
   def plot_field(filename, field='Bz'):
       with h5py.File(filename, 'r') as f:
           data = f[field][:]
           time = f['time'][()]
       
       plt.figure(figsize=(10, 8))
       plt.imshow(data.T, origin='lower', cmap='RdBu', aspect='auto')
       plt.colorbar(label=f'{field} [SI units]')
       plt.title(f'{field} at t = {time:.2f}')
       plt.xlabel('x [grid points]')
       plt.ylabel('y [grid points]')
       plt.tight_layout()
       plt.savefig(f'{field}_t{int(time):04d}.png', dpi=150)
       plt.show()
   
   plot_field('output/fields_001000.h5', 'Bz')

**Animation:**

.. code-block:: python

   import matplotlib.animation as animation
   import glob
   
   def animate_fields(field='Bz'):
       files = sorted(glob.glob('output/fields_*.h5'))
       
       fig, ax = plt.subplots(figsize=(10, 8))
       
       def update(frame):
           with h5py.File(files[frame], 'r') as f:
               data = f[field][:]
               time = f['time'][()]
           
           ax.clear()
           im = ax.imshow(data.T, origin='lower', cmap='RdBu', aspect='auto')
           ax.set_title(f'{field} at t = {time:.2f}')
           return [im]
       
       anim = animation.FuncAnimation(fig, update, frames=len(files),
                                     interval=100, blit=True)
       anim.save(f'{field}_animation.mp4', writer='ffmpeg', fps=10)
   
   animate_fields('Bz')

**Particle phase space:**

.. code-block:: python

   def plot_phase_space(filename, species='H+'):
       with h5py.File(filename, 'r') as f:
           x = f[f'{species}/x'][:]
           vx = f[f'{species}/vx'][:]
           time = f['time'][()]
       
       plt.figure(figsize=(10, 6))
       plt.hexbin(x, vx, gridsize=100, cmap='viridis', mincnt=1)
       plt.colorbar(label='Particle count')
       plt.xlabel('x [m]')
       plt.ylabel('vx [m/s]')
       plt.title(f'{species} Phase Space at t = {time:.2f}')
       plt.tight_layout()
       plt.savefig(f'phase_space_{species}_t{int(time):04d}.png', dpi=150)
       plt.show()
   
   plot_phase_space('output/particles_001000.h5', 'H+')

**Energy conservation:**

.. code-block:: python

   def plot_energy_conservation():
       diag = pd.read_csv('output/diagnostics.csv')
       
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
       
       # Total energy
       ax1.plot(diag['time'], diag['total_energy'])
       ax1.set_ylabel('Total Energy [J]')
       ax1.set_title('Energy Conservation')
       ax1.grid(True)
       
       # Relative energy change
       E0 = diag['total_energy'].iloc[0]
       rel_change = (diag['total_energy'] - E0) / E0 * 100
       ax2.plot(diag['time'], rel_change)
       ax2.set_xlabel('Time [ion gyroperiods]')
       ax2.set_ylabel('ΔE/E₀ [%]')
       ax2.axhline(1.0, color='r', linestyle='--', label='±1% threshold')
       ax2.axhline(-1.0, color='r', linestyle='--')
       ax2.legend()
       ax2.grid(True)
       
       plt.tight_layout()
       plt.savefig('energy_conservation.png', dpi=150)
       plt.show()
   
   plot_energy_conservation()

ParaView Visualization
~~~~~~~~~~~~~~~~~~~~~~

For interactive 3D visualization, use ParaView:

.. code-block:: bash

   # Convert HDF5 to XDMF (ParaView-compatible)
   python3 scripts/convert_to_xdmf.py output/

Then open ``output/fields.xdmf`` in ParaView.

**ParaView workflow:**

1. File → Open → ``fields.xdmf``
2. Apply
3. Choose field to visualize (Bz, Ex, etc.)
4. Add filters: Contour, Slice, Warp, etc.
5. Animate with time controls

Monitoring and Diagnostics
---------------------------

Real-Time Monitoring
~~~~~~~~~~~~~~~~~~~~

Monitor simulation progress in real-time:

.. code-block:: bash

   # Terminal 1: Run simulation
   ./jericho_mkII config.toml
   
   # Terminal 2: Monitor GPU
   watch -n 1 nvidia-smi
   
   # Terminal 3: Monitor output
   tail -f output/diagnostics.csv | column -t -s,

Check Energy Conservation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def check_energy_conservation(csv_file, tolerance=0.01):
       diag = pd.read_csv(csv_file)
       E0 = diag['total_energy'].iloc[0]
       rel_error = abs((diag['total_energy'] - E0) / E0)
       
       max_error = rel_error.max()
       mean_error = rel_error.mean()
       
       print(f"Energy conservation:")
       print(f"  Max relative error: {max_error*100:.3f}%")
       print(f"  Mean relative error: {mean_error*100:.3f}%")
       
       if max_error < tolerance:
           print(f"  ✓ PASS (< {tolerance*100}%)")
           return True
       else:
           print(f"  ✗ FAIL (> {tolerance*100}%)")
           print(f"  → Consider reducing timestep dt")
           return False
   
   check_energy_conservation('output/diagnostics.csv')

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile with nvprof
   nvprof ./jericho_mkII config.toml
   
   # Or with Nsight Systems
   nsys profile -o jericho_profile ./jericho_mkII config.toml
   
   # View profile
   nsys-ui jericho_profile.qdrep

Common Simulation Scenarios
----------------------------

Magnetic Reconnection
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [simulation]
   name = "reconnection"
   dt = 0.01
   n_steps = 20000
   
   [domain]
   x_min = -10.0
   x_max =  10.0
   y_min = -10.0
   y_max =  10.0
   nx = 256
   ny = 256
   
   [fields]
   init_type = "harris_sheet"
   B0 = 20.0e-9
   L_sheet = 1.0
   
   [boundaries]
   x_min = "periodic"
   x_max = "periodic"
   y_min = "periodic"
   y_max = "periodic"

**What to look for:**

- X-point formation in magnetic field
- Hall quadrupole in out-of-plane magnetic field
- Ion acceleration in separatrix region
- Energy conversion: EM → kinetic

Kelvin-Helmholtz Instability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [simulation]
   name = "kelvin_helmholtz"
   dt = 0.005
   n_steps = 30000
   
   [domain]
   x_min = 0.0
   x_max = 50.0
   y_min = 0.0
   y_max = 25.0
   nx = 512
   ny = 256
   
   [particles]
   particles_per_cell = 100
   
   [[particles.species]]
   name = "H+"
   # ... (as before)
   # Add shear flow in y > 12.5:
   drift_vx = 5.0e5  # m/s (for y > 12.5)
   
   [boundaries]
   x_min = "periodic"
   x_max = "periodic"
   y_min = "reflecting"
   y_max = "reflecting"

**What to look for:**

- Vortex formation at shear layer
- Vortex merging
- Density mixing
- Energy dissipation

Solar Wind Interaction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [simulation]
   name = "solar_wind"
   dt = 0.01
   n_steps = 50000
   
   [domain]
   x_min = 0.0
   x_max = 100.0
   y_min = -25.0
   y_max =  25.0
   nx = 512
   ny = 256
   
   [boundaries]
   x_min = "inflow"
   x_max = "outflow"
   y_min = "periodic"
   y_max = "periodic"
   
   [boundaries.inflow]
   enabled = true
   rate = 10000
   velocity_mean = [4.0e5, 0.0]  # Solar wind velocity
   velocity_thermal = 1.0e5
   species = ["H+"]

**What to look for:**

- Bow shock formation
- Magnetopause boundary
- Particle entry into magnetosphere

Troubleshooting
---------------

Simulation Crashes
~~~~~~~~~~~~~~~~~~

**Error: "CUDA out of memory"**

Solution: Reduce particle count or grid size:

.. code-block:: toml

   [particles]
   particles_per_cell = 50  # Reduce from 100
   
   [domain]
   nx = 128  # Reduce from 256
   ny = 128

**Error: "Energy conservation violated"**

Solution: Reduce timestep:

.. code-block:: toml

   [simulation]
   dt = 0.005  # Reduce from 0.01

**Error: "MPI communication timeout"**

Solution: Increase timeout or check network:

.. code-block:: bash

   mpirun --mca btl_tcp_if_include eth0 ...

Unexpected Results
~~~~~~~~~~~~~~~~~~

**No reconnection happening:**

- Check initial field strength is sufficient
- Verify Harris sheet thickness is correct
- Add small perturbation to trigger instability

**Particle loss:**

- Check boundary conditions
- Verify timestep satisfies CFL condition
- Enable verbose logging to see where particles exit

**Slow performance:**

- Check GPU utilization: ``nvidia-smi``
- Reduce output cadence
- Profile with ``nsys profile``

Best Practices
--------------

1. **Start small**: Test with small grid/few particles before production runs
2. **Check energy**: Always monitor energy conservation
3. **Save checkpoints**: Enable checkpointing for long runs
4. **Validate**: Compare against known benchmarks
5. **Document**: Keep notes on simulation parameters and results

See Also
--------

- :doc:`configuration` - Complete config reference
- :doc:`performance_tuning` - Optimization tips
- :doc:`troubleshooting` - Common issues
- :doc:`output_formats` - Detailed output documentation
