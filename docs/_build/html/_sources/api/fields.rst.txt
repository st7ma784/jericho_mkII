API Reference: Field Arrays
============================

Documentation for the ``FieldArrays`` class - GPU-resident electromagnetic field arrays.

.. note::
   This is a placeholder. Full API documentation is available in ``include/field_arrays.h``.

Quick Reference
---------------

.. code-block:: cpp

   #include "field_arrays.h"
   
   // Create field arrays
   jericho::FieldArrays fields(
       nx_local, ny_local, nghost,
       dx, dy, x_min, y_min, device_id);
   
   // Zero all fields
   fields.zero_fields();
   
   // Initialize Harris sheet
   fields.initialize_harris_sheet(B0, L_sheet);
   
   // Copy to/from host
   fields.copy_to_host(h_Ex, h_Ey, h_Bz);
   fields.copy_from_host(h_Ex, h_Ey, h_Bz);

Key Features
------------

- 2.5D electromagnetic fields (Ex, Ey, Bz)
- GPU device memory resident
- Ghost cell support for MPI boundaries
- Particle-derived quantities (charge, current)
- Background field support

Data Members
------------

.. code-block:: cpp

   class FieldArrays {
   public:
       // Grid dimensions
       int nx, ny;              // Including ghost cells
       int nx_local, ny_local;  // Excluding ghost cells
       int nghost;              // Ghost cell layers
       
       double dx, dy;           // Grid spacing
       double x_min, y_min;     // Domain origin
       
       // Electromagnetic fields (device pointers)
       double* Ex;          // Electric field x [V/m]
       double* Ey;          // Electric field y [V/m]
       double* Bz;          // Magnetic field z [T]
       
       // Plasma quantities
       double* charge_density;  // [C/m³]
       double* Jx;              // Current density x [A/m²]
       double* Jy;              // Current density y [A/m²]
       
       // ... (see header for full list)
   };

See Also
--------

- :doc:`particle_buffer` - Particle data structures
- :doc:`../cuda_kernels` - Field solver kernels
- ``include/field_arrays.h`` - Full header file
