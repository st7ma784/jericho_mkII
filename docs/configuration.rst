Configuration Reference
=======================

This document provides a complete reference for all configuration options in Jericho Mk II. Configuration files use TOML format for human-readable, hierarchical settings.

Configuration File Structure
----------------------------

A complete configuration file has the following sections:

.. code-block:: toml

   [simulation]     # Time integration and output settings
   [domain]         # Spatial grid and domain decomposition
   [mpi]            # Multi-GPU parallelization (optional)
   [cuda]           # GPU-specific settings
   [particles]      # Particle species and initialization
   [fields]         # Electromagnetic field initialization
   [boundaries]     # Boundary conditions
   [physics]        # Physics model options
   [output]         # Output format and cadence
   [diagnostics]    # Runtime diagnostics

Simulation Section
------------------

Controls time integration and basic simulation parameters.

.. code-block:: toml

   [simulation]
   name = "my_simulation"              # Simulation name (used in output filenames)
   output_dir = "./output"             # Output directory path
   checkpoint_dir = "./checkpoints"    # Checkpoint directory
   
   dt = 0.01                          # Timestep [ion gyroperiods]
   n_steps = 10000                    # Total number of timesteps
   output_cadence = 100               # Write output every N steps
   checkpoint_cadence = 500           # Write checkpoint every N steps
   
   restart_from = ""                  # Path to restart checkpoint (optional)
   start_time = 0.0                   # Initial simulation time [ion gyroperiods]

**Parameter Details:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 50 15

   * - Parameter
     - Type
     - Description
     - Default
   * - ``name``
     - string
     - Simulation identifier (alphanumeric, underscore, hyphen)
     - "simulation"
   * - ``output_dir``
     - path
     - Directory for output files (created if doesn't exist)
     - "./output"
   * - ``checkpoint_dir``
     - path
     - Directory for checkpoint files
     - "./checkpoints"
   * - ``dt``
     - float
     - Timestep in units of ion gyroperiod: :math:`\Delta t \cdot \Omega_i` where :math:`\Omega_i = qB/m_i`
     - 0.01
   * - ``n_steps``
     - int
     - Total timesteps to simulate
     - 1000
   * - ``output_cadence``
     - int
     - Write output every N steps (0 = no output)
     - 100
   * - ``checkpoint_cadence``
     - int
     - Write checkpoint every N steps (0 = no checkpoints)
     - 500
   * - ``restart_from``
     - path
     - HDF5 checkpoint file to restart from (empty = fresh start)
     - ""
   * - ``start_time``
     - float
     - Initial time (used when restarting)
     - 0.0

**Choosing timestep:**

The timestep must satisfy the CFL condition for stability:

.. math::

   \Delta t < \min\left(\frac{\Delta x}{v_{\text{max}}}, \frac{1}{\omega_{pe}}, \frac{1}{\Omega_i}\right)

where:

- :math:`\Delta x` = grid spacing
- :math:`v_{\text{max}}` = maximum particle velocity
- :math:`\omega_{pe} = \sqrt{n_e e^2 / \epsilon_0 m_e}` = electron plasma frequency
- :math:`\Omega_i = eB/m_i` = ion gyrofrequency

**Rule of thumb:** Start with ``dt = 0.01`` (0.01 ion gyroperiods) and check energy conservation. Decrease if energy drifts > 1% per 1000 steps.

Domain Section
--------------

Defines the spatial grid and physical domain size.

.. code-block:: toml

   [domain]
   # Physical domain bounds [meters]
   x_min = -10.0    # Minimum x coordinate
   x_max =  10.0    # Maximum x coordinate
   y_min = -10.0    # Minimum y coordinate
   y_max =  10.0    # Maximum y coordinate
   
   # Grid resolution
   nx = 256         # Grid points in x (must be even)
   ny = 256         # Grid points in y (must be even)
   
   # Optional: specify units
   length_unit = 1.0  # Multiply all lengths by this [m]

**Parameter Details:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Constraints
   * - ``x_min, x_max``
     - float
     - Domain bounds in x [m] or ion inertial lengths
     - ``x_max > x_min``
   * - ``y_min, y_max``
     - float
     - Domain bounds in y [m] or ion inertial lengths
     - ``y_max > y_min``
   * - ``nx, ny``
     - int
     - Number of grid cells
     - Must be even, ≥16, recommend powers of 2
   * - ``length_unit``
     - float
     - Unit conversion factor
     - > 0

**Grid spacing and resolution:**

Grid spacing is computed as:

.. math::

   \Delta x = \frac{x_{\max} - x_{\min}}{n_x}, \quad \Delta y = \frac{y_{\max} - y_{\min}}{n_y}

**Resolution requirements:**

For accurate physics, the grid spacing should satisfy:

.. math::

   \Delta x < \lambda_D = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}

where :math:`\lambda_D` is the Debye length. Typically use :math:`\Delta x \sim 0.5\lambda_D`.

**Common domain sizes:**

- **Magnetic reconnection:** :math:`\pm 10 \, d_i` where :math:`d_i = c/\omega_{pi}` (ion inertial length)
- **Magnetotail:** :math:`100 \times 50 \, d_i`
- **Kelvin-Helmholtz:** :math:`50 \times 25 \, d_i`

MPI Section
-----------

Multi-GPU domain decomposition (required for multi-GPU runs).

.. code-block:: toml

   [mpi]
   npx = 2    # Number of processes in x direction
   npy = 2    # Number of processes in y direction
   
   # Optional: load balancing
   load_balance = true          # Enable dynamic load balancing
   rebalance_cadence = 100      # Rebalance every N steps

**Parameter Details:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Constraints
   * - ``npx``
     - int
     - Processes in x
     - ``npx * npy`` = total MPI ranks
   * - ``npy``
     - int
     - Processes in y
     - Must divide ``nx``, ``ny`` evenly
   * - ``load_balance``
     - bool
     - Enable dynamic load balancing
     - false (experimental)
   * - ``rebalance_cadence``
     - int
     - Rebalance frequency
     - > 0

**Choosing decomposition:**

Total MPI ranks = :math:`n_{px} \times n_{py}` must equal your ``mpirun -np`` value.

**Example:** For 8 GPUs, you can use:

- 4×2 decomposition: ``npx=4, npy=2``
- 2×4 decomposition: ``npx=2, npy=4``
- 8×1 decomposition: ``npx=8, npy=1`` (not recommended)

**Best practices:**

1. Keep aspect ratio close to domain aspect ratio
2. Ensure ``nx/npx`` and ``ny/npy`` are ≥ 32 (each rank needs enough work)
3. Match physical decomposition (e.g., split along flow direction)

CUDA Section
------------

GPU-specific performance settings.

.. code-block:: toml

   [cuda]
   device_id = 0                      # CUDA device to use (-1 = auto-assign)
   threads_per_block = 256            # CUDA threads per block
   use_cuda_aware_mpi = true          # Enable CUDA-aware MPI
   use_unified_memory = false         # Use CUDA unified memory (experimental)
   memory_pool_size = 0               # Pre-allocate memory pool [MB] (0 = auto)

**Parameter Details:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Recommended
   * - ``device_id``
     - int
     - CUDA device ID (use ``nvidia-smi`` to see IDs)
     - -1 (auto)
   * - ``threads_per_block``
     - int
     - Threads per CUDA block (must be multiple of 32)
     - 256
   * - ``use_cuda_aware_mpi``
     - bool
     - Direct GPU-GPU MPI transfers
     - true
   * - ``use_unified_memory``
     - bool
     - CUDA unified memory (slower but simpler)
     - false
   * - ``memory_pool_size``
     - int
     - Pre-allocate memory [MB]
     - 0 (auto)

**Performance tuning:**

- **threads_per_block**: 256 is optimal for most GPUs. Try 512 for compute capability 8.0+
- **cuda_aware_mpi**: Set to ``true`` if your MPI supports it (huge performance gain)
- **unified_memory**: Only for prototyping; slower than explicit transfers

Particles Section
-----------------

Defines particle species and initialization.

.. code-block:: toml

   [particles]
   particles_per_cell = 100           # Particles per grid cell per species
   
   # Species definitions
   [[particles.species]]
   name = "H+"                        # Human-readable name
   charge = 1.602e-19                 # Particle charge [C]
   mass = 1.673e-27                   # Particle mass [kg]
   temperature = 1.0e6                # Temperature [K]
   density = 1.0e6                    # Number density [m^-3]
   drift_vx = 0.0                     # Drift velocity x [m/s]
   drift_vy = 0.0                     # Drift velocity y [m/s]
   
   [[particles.species]]
   name = "O+"
   charge = 1.602e-19
   mass = 2.656e-26                   # Oxygen-16
   temperature = 1.0e6
   density = 1.0e5                    # 10:1 H+:O+ ratio

**Parameter Details:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Typical Values
   * - ``particles_per_cell``
     - int
     - Macroparticles per cell per species
     - 25-200
   * - ``name``
     - string
     - Species identifier
     - "H+", "O+", "electrons"
   * - ``charge``
     - float
     - Particle charge [C]
     - :math:`\pm 1.602 \times 10^{-19}` C
   * - ``mass``
     - float
     - Particle mass [kg]
     - See table below
   * - ``temperature``
     - float
     - Initial temperature [K]
     - :math:`10^5 - 10^7` K
   * - ``density``
     - float
     - Number density [m⁻³]
     - :math:`10^5 - 10^8` m⁻³
   * - ``drift_vx, drift_vy``
     - float
     - Bulk flow velocity [m/s]
     - :math:`0 - 10^6` m/s

**Common ion species:**

.. list-table::
   :header-rows: 1

   * - Species
     - Mass [kg]
     - Charge [C]
     - Notes
   * - Electron
     - :math:`9.109 \times 10^{-31}`
     - :math:`-1.602 \times 10^{-19}`
     - Treated as fluid
   * - H⁺ (proton)
     - :math:`1.673 \times 10^{-27}`
     - :math:`+1.602 \times 10^{-19}`
     - Most common
   * - He⁺
     - :math:`6.645 \times 10^{-27}`
     - :math:`+1.602 \times 10^{-19}`
     - Alpha particles
   * - O⁺
     - :math:`2.656 \times 10^{-26}`
     - :math:`+1.602 \times 10^{-19}`
     - Heavy ions
   * - O²⁺
     - :math:`2.656 \times 10^{-26}`
     - :math:`+3.204 \times 10^{-19}`
     - Doubly charged

**Particle statistics:**

The number of macroparticles per species is:

.. math::

   N_{\text{particles}} = n_x \times n_y \times N_{\text{ppc}}

where :math:`N_{\text{ppc}}` is particles_per_cell.

**Memory usage:**

Each particle requires 48 bytes (6 doubles + 2 bytes overhead). Total memory:

.. math::

   \text{Memory} \approx 48 \times n_x \times n_y \times N_{\text{ppc}} \times N_{\text{species}} \text{ bytes}

Example: 256×256 grid, 100 ppc, 2 species = 314 MB

Fields Section
--------------

Initial electromagnetic field configuration.

.. code-block:: toml

   [fields]
   # Initial magnetic field
   B0 = 20.0e-9                       # Asymptotic field strength [T]
   L_sheet = 1.0                      # Harris sheet thickness [ion inertial lengths]
   
   # Background electric field
   E0_x = 0.0                         # Constant Ex [V/m]
   E0_y = 0.0                         # Constant Ey [V/m]
   
   # Field initialization type
   init_type = "harris_sheet"         # Options: harris_sheet, uniform, custom

**Initialization types:**

**1. Harris Sheet (magnetic reconnection):**

.. code-block:: toml

   init_type = "harris_sheet"
   B0 = 20.0e-9        # Asymptotic field [T]
   L_sheet = 1.0       # Sheet thickness [d_i]

Produces magnetic field profile:

.. math::

   B_z(y) = B_0 \tanh\left(\frac{y}{L}\right)

**2. Uniform field:**

.. code-block:: toml

   init_type = "uniform"
   Bz_uniform = 20.0e-9    # Constant Bz [T]

**3. Custom (from file):**

.. code-block:: toml

   init_type = "custom"
   field_file = "initial_fields.h5"

Boundaries Section
------------------

Boundary conditions for particles and fields.

.. code-block:: toml

   [boundaries]
   # Particle boundary conditions
   x_min = "periodic"      # Left boundary: periodic, outflow, inflow, reflecting
   x_max = "periodic"      # Right boundary
   y_min = "periodic"      # Bottom boundary
   y_max = "periodic"      # Top boundary
   
   # Inflow boundary parameters (if using inflow)
   [boundaries.inflow]
   enabled = false
   rate = 1000                        # Particles injected per timestep per boundary
   velocity_mean = [0.0, 0.0]        # Mean velocity [m/s]
   velocity_thermal = 1.0e5          # Thermal velocity [m/s]
   species = ["H+"]                   # Species to inject

**Boundary types:**

.. list-table::
   :header-rows: 1

   * - Type
     - Particles
     - Fields
     - Use Case
   * - ``periodic``
     - Wrap to opposite side
     - Periodic BCs
     - Closed systems, turbulence
   * - ``outflow``
     - Remove when exiting
     - Zero gradient
     - Open boundaries
   * - ``inflow``
     - Inject new particles
     - Fixed value
     - Driven systems, solar wind
   * - ``reflecting``
     - Elastic reflection
     - Zero gradient
     - Solid walls, mirrors

Physics Section
---------------

Controls physics models and numerical methods.

.. code-block:: toml

   [physics]
   boris_pusher = true                # Use Boris particle pusher
   cam_method = true                  # Current Advance Method
   ohms_law_hall = true               # Include Hall term in Ohm's law
   electron_pressure = false          # Electron pressure gradient (experimental)
   
   # Collisions (optional)
   collisions_enabled = false
   collision_frequency = 0.0          # Collision rate [Hz]

**Physics options:**

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Recommended
   * - ``boris_pusher``
     - Energy-conserving particle pusher
     - true (always)
   * - ``cam_method``
     - Current Advance Method for stability
     - true
   * - ``ohms_law_hall``
     - Hall term in generalized Ohm's law
     - true (for reconnection)
   * - ``electron_pressure``
     - Include :math:`\nabla p_e` term
     - false (slow, experimental)

**Ohm's Law:**

With Hall term enabled:

.. math::

   \mathbf{E} = -\mathbf{v}_e \times \mathbf{B} + \frac{1}{en_e}\mathbf{J} \times \mathbf{B} + \eta \mathbf{J}

Without Hall term:

.. math::

   \mathbf{E} = -\mathbf{v}_e \times \mathbf{B} + \eta \mathbf{J}

Output Section
--------------

Configure output files and formats.

.. code-block:: toml

   [output]
   # Field output
   fields = ["Ex", "Ey", "Bz", "density", "current_x", "current_y"]
   field_format = "hdf5"              # Options: hdf5, netcdf, binary
   
   # Particle output
   particles = true                   # Write particle data
   particle_cadence = 100             # Output particles every N steps
   particle_format = "hdf5"
   
   # Checkpoint output
   checkpoint_cadence = 500           # Checkpoint every N steps
   keep_all_checkpoints = false       # Keep all or just latest 2

**Available field outputs:**

- ``Ex, Ey`` - Electric field components [V/m]
- ``Bz`` - Magnetic field (2.5D, only z-component) [T]
- ``density`` - Number density [m⁻³]
- ``current_x, current_y`` - Current density [A/m²]
- ``temperature`` - Temperature [K]
- ``pressure`` - Pressure [Pa]

Diagnostics Section
-------------------

Runtime monitoring and diagnostics.

.. code-block:: toml

   [diagnostics]
   compute_energy = true              # Track total energy
   compute_momentum = true            # Track total momentum
   check_nans = true                  # Check for NaN values
   verbose = true                     # Print detailed output
   
   # Energy conservation threshold
   energy_tolerance = 0.01            # Warn if |ΔE|/E > 1%
   
   # Performance profiling
   profile = false                    # Enable detailed profiling
   profile_cadence = 10               # Profile every N steps

Complete Example
----------------

Here's a complete configuration file for magnetic reconnection:

.. code-block:: toml

   # ============================================================================
   # Magnetic Reconnection Simulation
   # ============================================================================
   
   [simulation]
   name = "reconnection_test"
   output_dir = "./output/reconnection"
   checkpoint_dir = "./checkpoints/reconnection"
   
   dt = 0.01
   n_steps = 10000
   output_cadence = 100
   checkpoint_cadence = 500
   
   [domain]
   x_min = -10.0
   x_max =  10.0
   y_min = -10.0
   y_max =  10.0
   nx = 256
   ny = 256
   
   [mpi]
   npx = 2
   npy = 2
   
   [cuda]
   device_id = -1
   threads_per_block = 256
   use_cuda_aware_mpi = true
   
   [particles]
   particles_per_cell = 100
   
   [[particles.species]]
   name = "H+"
   charge = 1.602e-19
   mass = 1.673e-27
   temperature = 1.0e6
   density = 1.0e6
   drift_vx = 0.0
   drift_vy = 0.0
   
   [[particles.species]]
   name = "O+"
   charge = 1.602e-19
   mass = 2.656e-26
   temperature = 1.0e6
   density = 1.0e5
   
   [fields]
   init_type = "harris_sheet"
   B0 = 20.0e-9
   L_sheet = 1.0
   E0_x = 0.0
   E0_y = 0.0
   
   [boundaries]
   x_min = "periodic"
   x_max = "periodic"
   y_min = "periodic"
   y_max = "periodic"
   
   [physics]
   boris_pusher = true
   cam_method = true
   ohms_law_hall = true
   electron_pressure = false
   
   [output]
   fields = ["Ex", "Ey", "Bz", "density", "current_x", "current_y"]
   field_format = "hdf5"
   particles = true
   particle_cadence = 100
   particle_format = "hdf5"
   
   [diagnostics]
   compute_energy = true
   compute_momentum = true
   check_nans = true
   verbose = true
   energy_tolerance = 0.01

Validation and Testing
-----------------------

After creating a configuration file, validate it:

.. code-block:: bash

   # Dry run (parse config without running)
   ./jericho_mkII --dry-run config.toml
   
   # Test with few steps
   # Edit config: n_steps = 10
   ./jericho_mkII config.toml
   
   # Check output
   ls -lh output/
   h5dump -H output/fields_000010.h5

See Also
--------

- :doc:`getting_started` - Installation and first run
- :doc:`running_simulations` - Advanced simulation techniques
- :doc:`performance_tuning` - Optimize your configuration
