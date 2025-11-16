Output Formats
==============

This document describes the output file formats produced by Jericho Mk II and how to work with them.

Overview
--------

Jericho Mk II produces three types of output:

1. **Field files** - Electromagnetic fields on grid (HDF5)
2. **Particle files** - Particle positions and velocities (HDF5)
3. **Diagnostics** - Time series of integrated quantities (CSV)
4. **Checkpoints** - Full simulation state for restart (HDF5)

All spatial data uses **SI units** unless otherwise noted.

Field Output Files
------------------

File Naming Convention
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   fields_NNNNNN.h5
   
   where NNNNNN is the timestep number (zero-padded to 6 digits)
   
   Examples:
   fields_000000.h5  - Initial conditions (t=0)
   fields_000100.h5  - After 100 timesteps
   fields_001000.h5  - After 1000 timesteps

HDF5 Structure
~~~~~~~~~~~~~~

.. code-block:: text

   fields_001000.h5
   ├── Ex              [Dataset: (ny, nx), float64]
   ├── Ey              [Dataset: (ny, nx), float64]
   ├── Bz              [Dataset: (ny, nx), float64]
   ├── density         [Dataset: (ny, nx), float64]
   ├── current_x       [Dataset: (ny, nx), float64]
   ├── current_y       [Dataset: (ny, nx), float64]
   ├── time            [Scalar: float64]
   ├── step            [Scalar: int32]
   └── metadata        [Group]
       ├── nx          [Scalar: int32]
       ├── ny          [Scalar: int32]
       ├── x_min       [Scalar: float64]
       ├── x_max       [Scalar: float64]
       ├── y_min       [Scalar: float64]
       ├── y_max       [Scalar: float64]
       └── units       [Group]
           ├── Ex      [String: "V/m"]
           ├── Ey      [String: "V/m"]
           ├── Bz      [String: "T"]
           ├── density [String: "m^-3"]
           └── current [String: "A/m^2"]

Field Datasets
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 50

   * - Dataset
     - Shape
     - Units
     - Description
   * - ``Ex``
     - (ny, nx)
     - V/m
     - Electric field, x-component
   * - ``Ey``
     - (ny, nx)
     - V/m
     - Electric field, y-component
   * - ``Bz``
     - (ny, nx)
     - T (Tesla)
     - Magnetic field, z-component (2.5D assumption)
   * - ``density``
     - (ny, nx)
     - m⁻³
     - Total number density (all species)
   * - ``current_x``
     - (ny, nx)
     - A/m²
     - Current density, x-component
   * - ``current_y``
     - (ny, nx)
     - A/m²
     - Current density, y-component
   * - ``temperature``
     - (ny, nx)
     - K
     - Ion temperature (if enabled)
   * - ``pressure``
     - (ny, nx)
     - Pa
     - Ion pressure (if enabled)

**Note:** Array indexing is ``[y_index, x_index]`` (row-major order).

Reading Field Files in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import numpy as np
   import matplotlib.pyplot as plt
   
   def read_fields(filename):
       """Read field data from HDF5 file."""
       with h5py.File(filename, 'r') as f:
           # Read field arrays
           Ex = f['Ex'][:]
           Ey = f['Ey'][:]
           Bz = f['Bz'][:]
           density = f['density'][:]
           
           # Read metadata
           time = f['time'][()]
           step = f['step'][()]
           nx = f['metadata/nx'][()]
           ny = f['metadata/ny'][()]
           x_min = f['metadata/x_min'][()]
           x_max = f['metadata/x_max'][()]
           y_min = f['metadata/y_min'][()]
           y_max = f['metadata/y_max'][()]
           
       # Create coordinate arrays
       x = np.linspace(x_min, x_max, nx)
       y = np.linspace(y_min, y_max, ny)
       
       return {
           'Ex': Ex, 'Ey': Ey, 'Bz': Bz, 'density': density,
           'x': x, 'y': y, 'time': time, 'step': step
       }
   
   # Usage
   fields = read_fields('output/fields_001000.h5')
   print(f"Time: {fields['time']:.2f} ion gyroperiods")
   print(f"Grid shape: {fields['Bz'].shape}")
   
   # Plot magnetic field
   plt.figure(figsize=(10, 8))
   plt.imshow(fields['Bz'].T, origin='lower', 
              extent=[fields['x'][0], fields['x'][-1], 
                      fields['y'][0], fields['y'][-1]],
              cmap='RdBu', aspect='auto')
   plt.colorbar(label='Bz [T]')
   plt.xlabel('x [m]')
   plt.ylabel('y [m]')
   plt.title(f"Magnetic Field at t={fields['time']:.2f}")
   plt.show()

Particle Output Files
---------------------

File Naming Convention
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   particles_NNNNNN.h5
   
   Examples:
   particles_000000.h5  - Initial particles
   particles_000100.h5  - After 100 timesteps

HDF5 Structure
~~~~~~~~~~~~~~

.. code-block:: text

   particles_001000.h5
   ├── H+              [Group]
   │   ├── x           [Dataset: (n_particles,), float64]
   │   ├── y           [Dataset: (n_particles,), float64]
   │   ├── vx          [Dataset: (n_particles,), float64]
   │   ├── vy          [Dataset: (n_particles,), float64]
   │   ├── weight      [Dataset: (n_particles,), float64]
   │   └── metadata    [Group]
   │       ├── charge  [Scalar: float64]
   │       ├── mass    [Scalar: float64]
   │       └── count   [Scalar: int64]
   ├── O+              [Group]
   │   └── ... (same structure)
   ├── time            [Scalar: float64]
   └── step            [Scalar: int32]

Particle Datasets
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Dataset
     - Shape
     - Units
     - Description
   * - ``x``
     - (n,)
     - m
     - Particle x positions
   * - ``y``
     - (n,)
     - m
     - Particle y positions
   * - ``vx``
     - (n,)
     - m/s
     - Particle x velocities
   * - ``vy``
     - (n,)
     - m/s
     - Particle y velocities
   * - ``weight``
     - (n,)
     - dimensionless
     - Statistical weight (macroparticle multiplicity)

Reading Particle Files in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def read_particles(filename, species='H+'):
       """Read particle data from HDF5 file."""
       with h5py.File(filename, 'r') as f:
           # Check if species exists
           if species not in f:
               print(f"Available species: {list(f.keys())}")
               raise ValueError(f"Species '{species}' not found")
           
           # Read particle arrays
           x = f[f'{species}/x'][:]
           y = f[f'{species}/y'][:]
           vx = f[f'{species}/vx'][:]
           vy = f[f'{species}/vy'][:]
           weight = f[f'{species}/weight'][:]
           
           # Read metadata
           charge = f[f'{species}/metadata/charge'][()]
           mass = f[f'{species}/metadata/mass'][()]
           time = f['time'][()]
           step = f['step'][()]
           
       return {
           'x': x, 'y': y, 'vx': vx, 'vy': vy, 'weight': weight,
           'charge': charge, 'mass': mass, 'time': time, 'step': step
       }
   
   # Usage
   particles = read_particles('output/particles_001000.h5', 'H+')
   print(f"Number of H+ particles: {len(particles['x'])}")
   print(f"Mean velocity: vx={particles['vx'].mean():.2e} m/s")
   
   # Phase space plot
   plt.figure(figsize=(10, 6))
   plt.hexbin(particles['x'], particles['vx'], 
              gridsize=100, cmap='viridis', mincnt=1)
   plt.colorbar(label='Particle count')
   plt.xlabel('x [m]')
   plt.ylabel('vx [m/s]')
   plt.title(f"H+ Phase Space at t={particles['time']:.2f}")
   plt.show()

Diagnostics CSV File
--------------------

File Structure
~~~~~~~~~~~~~~

.. code-block:: text

   diagnostics.csv

Contains time series data with comma-separated values:

.. code-block:: text

   step,time,kinetic_energy,em_energy,total_energy,energy_change,momentum_x,momentum_y,n_particles
   0,0.000000e+00,4.532e-11,1.234e-12,4.655e-11,0.000e+00,1.234e-20,-5.678e-21,6553600
   100,1.000000e+00,4.534e-11,1.232e-12,4.657e-11,2.000e-15,1.235e-20,-5.679e-21,6553600
   200,2.000000e+00,4.536e-11,1.230e-12,4.659e-11,4.000e-15,1.236e-20,-5.680e-21,6553605
   ...

Columns
~~~~~~~

.. list-table::
   :header-rows: 1

   * - Column
     - Units
     - Description
   * - ``step``
     - integer
     - Timestep number
   * - ``time``
     - ion gyroperiods
     - Simulation time
   * - ``kinetic_energy``
     - J (Joules)
     - Total kinetic energy of all particles
   * - ``em_energy``
     - J
     - Electromagnetic field energy
   * - ``total_energy``
     - J
     - Sum of kinetic and EM energy
   * - ``energy_change``
     - J
     - Change from initial energy
   * - ``momentum_x``
     - kg·m/s
     - Total momentum, x-component
   * - ``momentum_y``
     - kg·m/s
     - Total momentum, y-component
   * - ``n_particles``
     - integer
     - Total active particle count

Reading Diagnostics in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   def read_diagnostics(filename='output/diagnostics.csv'):
       """Read diagnostics CSV file."""
       df = pd.read_csv(filename)
       return df
   
   # Usage
   diag = read_diagnostics()
   
   # Plot energy conservation
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
   
   # Absolute energy
   ax1.plot(diag['time'], diag['kinetic_energy'], label='Kinetic')
   ax1.plot(diag['time'], diag['em_energy'], label='EM')
   ax1.plot(diag['time'], diag['total_energy'], label='Total', linewidth=2)
   ax1.set_ylabel('Energy [J]')
   ax1.set_title('Energy Evolution')
   ax1.legend()
   ax1.grid(True)
   
   # Relative energy change
   E0 = diag['total_energy'].iloc[0]
   rel_change = (diag['total_energy'] - E0) / E0 * 100
   ax2.plot(diag['time'], rel_change)
   ax2.axhline(1, color='r', linestyle='--', label='±1%')
   ax2.axhline(-1, color='r', linestyle='--')
   ax2.set_xlabel('Time [ion gyroperiods]')
   ax2.set_ylabel('ΔE/E₀ [%]')
   ax2.set_title('Relative Energy Change')
   ax2.legend()
   ax2.grid(True)
   
   plt.tight_layout()
   plt.savefig('energy_conservation.png', dpi=150)
   plt.show()
   
   # Check energy conservation
   max_error = abs(rel_change).max()
   print(f"Maximum energy error: {max_error:.3f}%")
   if max_error < 1.0:
       print("✓ Energy conservation PASSED (<1%)")
   else:
       print("✗ Energy conservation FAILED (>1%)")
       print("  → Consider reducing timestep dt")

Checkpoint Files
----------------

File Structure
~~~~~~~~~~~~~~

.. code-block:: text

   checkpoint_NNNNNN.h5

Contains complete simulation state for restart:

.. code-block:: text

   checkpoint_001000.h5
   ├── fields              [Group]
   │   ├── Ex              [Dataset: (ny, nx), float64]
   │   ├── Ey              [Dataset: (ny, nx), float64]
   │   ├── Bz              [Dataset: (ny, nx), float64]
   │   └── ... (all fields)
   ├── particles           [Group]
   │   ├── H+              [Group]
   │   │   ├── x           [Dataset]
   │   │   └── ... (all particle data)
   │   └── O+              [Group]
   ├── mpi_state           [Group] (multi-GPU only)
   │   ├── rank
   │   ├── size
   │   └── decomposition
   ├── config              [Group]
   │   └── ... (copy of configuration)
   └── metadata            [Group]
       ├── time
       ├── step
       └── version

Restarting from Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Restart from checkpoint
   ./jericho_mkII --restart checkpoints/checkpoint_001000.h5 config.toml

**Notes:**

- Original config file still required
- Simulation continues from checkpoint time/step
- Output directory must be writable
- MPI decomposition must match checkpoint

Data Analysis Examples
----------------------

Compute Derived Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Magnetic field magnitude:**

.. code-block:: python

   def compute_field_magnitude(filename):
       fields = read_fields(filename)
       # For 2.5D: B = Bz (only z-component)
       B_mag = np.abs(fields['Bz'])
       return B_mag

**Electric field magnitude:**

.. code-block:: python

   def compute_E_magnitude(filename):
       fields = read_fields(filename)
       E_mag = np.sqrt(fields['Ex']**2 + fields['Ey']**2)
       return E_mag

**Current density magnitude:**

.. code-block:: python

   def compute_current_magnitude(filename):
       fields = read_fields(filename)
       J_mag = np.sqrt(fields['current_x']**2 + fields['current_y']**2)
       return J_mag

**Particle kinetic energy distribution:**

.. code-block:: python

   def compute_energy_distribution(filename, species='H+', bins=50):
       particles = read_particles(filename, species)
       
       # Kinetic energy per particle
       v2 = particles['vx']**2 + particles['vy']**2
       KE = 0.5 * particles['mass'] * v2
       
       # Weight by macroparticle multiplicity
       hist, edges = np.histogram(KE, bins=bins, weights=particles['weight'])
       
       return hist, edges

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

**Fourier analysis of energy:**

.. code-block:: python

   def analyze_energy_spectrum(diag):
       from scipy.fft import fft, fftfreq
       
       # FFT of energy signal
       energy = diag['total_energy'].values
       fft_energy = fft(energy)
       freqs = fftfreq(len(energy), d=diag['time'].iloc[1] - diag['time'].iloc[0])
       
       # Power spectrum
       power = np.abs(fft_energy)**2
       
       # Plot
       plt.figure(figsize=(10, 6))
       plt.loglog(freqs[1:len(freqs)//2], power[1:len(power)//2])
       plt.xlabel('Frequency [1/ion gyroperiod]')
       plt.ylabel('Power')
       plt.title('Energy Spectrum')
       plt.grid(True)
       plt.show()

**Particle count evolution:**

.. code-block:: python

   def plot_particle_evolution(diag):
       plt.figure(figsize=(10, 6))
       plt.plot(diag['time'], diag['n_particles'])
       plt.xlabel('Time [ion gyroperiods]')
       plt.ylabel('Number of particles')
       plt.title('Particle Count Evolution')
       plt.grid(True)
       plt.show()
       
       # Check for particle loss/gain
       n0 = diag['n_particles'].iloc[0]
       n_final = diag['n_particles'].iloc[-1]
       change = (n_final - n0) / n0 * 100
       print(f"Particle count change: {change:+.2f}%")

Spatial Analysis
~~~~~~~~~~~~~~~~

**Line plots along axes:**

.. code-block:: python

   def plot_field_lineout(filename, field='Bz', axis='x', position=0.0):
       fields = read_fields(filename)
       
       if axis == 'x':
           # Find nearest y index
           iy = np.argmin(np.abs(fields['y'] - position))
           values = fields[field][iy, :]
           coords = fields['x']
           xlabel = 'x [m]'
       else:  # axis == 'y'
           ix = np.argmin(np.abs(fields['x'] - position))
           values = fields[field][:, ix]
           coords = fields['y']
           xlabel = 'y [m]'
       
       plt.figure(figsize=(10, 6))
       plt.plot(coords, values)
       plt.xlabel(xlabel)
       plt.ylabel(f'{field}')
       plt.title(f'{field} lineout along {axis} at {axis}={position}')
       plt.grid(True)
       plt.show()

**2D histograms:**

.. code-block:: python

   def plot_2d_histogram(filename, species='H+'):
       particles = read_particles(filename, species)
       
       plt.figure(figsize=(12, 5))
       
       # Real space
       plt.subplot(121)
       plt.hexbin(particles['x'], particles['y'], 
                  gridsize=50, cmap='viridis', mincnt=1)
       plt.colorbar(label='Count')
       plt.xlabel('x [m]')
       plt.ylabel('y [m]')
       plt.title('Real Space')
       
       # Velocity space
       plt.subplot(122)
       plt.hexbin(particles['vx'], particles['vy'],
                  gridsize=50, cmap='viridis', mincnt=1)
       plt.colorbar(label='Count')
       plt.xlabel('vx [m/s]')
       plt.ylabel('vy [m/s]')
       plt.title('Velocity Space')
       
       plt.tight_layout()
       plt.show()

Exporting to Other Formats
---------------------------

Convert to VTK (ParaView)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyevtk.hl import gridToVTK
   
   def export_to_vtk(input_h5, output_vtk):
       fields = read_fields(input_h5)
       
       # Create coordinate arrays (cell-centered)
       x = fields['x']
       y = fields['y']
       z = np.array([0.0])  # 2D data
       
       # Export (note: VTK uses point data, so we need to reshape)
       gridToVTK(output_vtk,
                x, y, z,
                pointData={
                    'Ex': fields['Ex'],
                    'Ey': fields['Ey'],
                    'Bz': fields['Bz'],
                    'density': fields['density']
                })
   
   # Usage
   export_to_vtk('output/fields_001000.h5', 'output/fields_001000')
   # Creates: fields_001000.vtr (open in ParaView)

Convert to NetCDF
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from netCDF4 import Dataset
   
   def export_to_netcdf(input_h5, output_nc):
       fields = read_fields(input_h5)
       
       # Create NetCDF file
       nc = Dataset(output_nc, 'w', format='NETCDF4')
       
       # Define dimensions
       nc.createDimension('x', len(fields['x']))
       nc.createDimension('y', len(fields['y']))
       
       # Create coordinate variables
       x_var = nc.createVariable('x', 'f8', ('x',))
       y_var = nc.createVariable('y', 'f8', ('y',))
       x_var[:] = fields['x']
       y_var[:] = fields['y']
       
       # Create field variables
       for field_name in ['Ex', 'Ey', 'Bz', 'density']:
           var = nc.createVariable(field_name, 'f8', ('y', 'x'))
           var[:] = fields[field_name]
       
       nc.close()

See Also
--------

- :doc:`running_simulations` - How to generate output
- :doc:`configuration` - Configure output options
- Example Python scripts in ``scripts/`` directory
