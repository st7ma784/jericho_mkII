API Reference: Boundary Conditions
====================================

Documentation for boundary condition implementations.

.. note::
   This is a placeholder. Full documentation is available in the source files.

Boundary Types
--------------

Jericho Mk II supports four boundary condition types:

Periodic Boundaries
~~~~~~~~~~~~~~~~~~~

Particles wrap to opposite side:

.. code-block:: cpp

   // CUDA kernel
   __global__ void apply_periodic_boundaries_kernel(
       double* x, double* y,
       double x_min, double x_max, double y_min, double y_max,
       size_t n_particles);

**Use case:** Closed systems, turbulence, periodic structures

Outflow Boundaries
~~~~~~~~~~~~~~~~~~

Particles are removed when leaving domain:

.. code-block:: cpp

   __global__ void mark_outflow_particles_kernel(
       const double* x, const double* y,
       bool* active,
       double x_min, double x_max, double y_min, double y_max,
       size_t n_particles);

**Use case:** Open boundaries, solar wind

Inflow Boundaries
~~~~~~~~~~~~~~~~~

New particles injected at boundaries:

.. code-block:: cpp

   void inject_particles(
       ParticleBuffer& particles,
       double x_boundary, double y_min, double y_max,
       double velocity_mean, double velocity_thermal,
       int rate, uint8_t type);

**Use case:** Driven systems, solar wind interaction

Reflecting Boundaries
~~~~~~~~~~~~~~~~~~~~~

Particles elastically reflect at boundaries:

.. code-block:: cpp

   __global__ void apply_reflecting_boundaries_kernel(
       double* x, double* y, double* vx, double* vy,
       double x_min, double x_max, double y_min, double y_max,
       size_t n_particles);

**Use case:** Solid walls, magnetic mirrors

Configuration
-------------

Set in config file:

.. code-block:: toml

   [boundaries]
   x_min = "periodic"
   x_max = "outflow"
   y_min = "reflecting"
   y_max = "inflow"
   
   [boundaries.inflow]
   rate = 1000
   velocity_mean = [4.0e5, 0.0]
   velocity_thermal = 1.0e5

See Also
--------

- :doc:`../configuration` - Boundary configuration options
- :doc:`../running_simulations` - Usage examples
- ``cuda/boundaries.cu`` - CUDA implementation
