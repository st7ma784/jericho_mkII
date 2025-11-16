API Reference: MPI Manager
==========================

Documentation for the ``MPIManager`` class - MPI+CUDA hybrid parallelism.

.. note::
   This is a placeholder. Full API documentation is available in ``include/mpi_manager.h``.

Quick Reference
---------------

.. code-block:: cpp

   #include "mpi_manager.h"
   
   // Initialize MPI
   jericho::MPIManager mpi(argc, argv, npx, npy,
                           nx_global, ny_global,
                           x_min, x_max, y_min, y_max,
                           cuda_aware);
   
   // Exchange ghost cells
   mpi.exchange_ghost_cells(fields);
   
   // Migrate particles
   mpi.migrate_particles(particles);
   
   // Collective reduction
   double global_energy = mpi.reduce_sum(local_energy);

Key Features
------------

- 2D Cartesian domain decomposition
- Ghost cell exchange for fields
- Particle migration between ranks
- CUDA-aware MPI support
- Asynchronous communication

Data Members
------------

.. code-block:: cpp

   class MPIManager {
   public:
       int rank;        // This process's rank
       int size;        // Total MPI processes
       int npx, npy;    // Process grid dimensions
       
       // Neighbors
       int neighbor_left, neighbor_right;
       int neighbor_bottom, neighbor_top;
       
       // Local domain
       double x_min_local, x_max_local;
       double y_min_local, y_max_local;
       int nx_local, ny_local;
       
       bool cuda_aware_mpi;
   };

Key Methods
-----------

Domain Decomposition
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Compute rank's position in 2D grid
   void compute_topology();
   
   // Compute local domain bounds
   void compute_local_domain();
   
   // Identify neighbor ranks
   void compute_neighbors();

Communication
~~~~~~~~~~~~~

.. code-block:: cpp

   // Exchange ghost cells
   void exchange_ghost_cells(FieldArrays& fields);
   
   // Migrate particles
   void migrate_particles(ParticleBuffer& particles);
   
   // Collective operations
   double reduce_sum(double local_value);
   double reduce_max(double local_value);

Configuration
-------------

.. code-block:: toml

   [mpi]
   npx = 4  # Processes in x
   npy = 2  # Processes in y

Running:

.. code-block:: bash

   mpirun -np 8 ./jericho_mkII config.toml

See Also
--------

- :doc:`../mpi_parallelism` - MPI parallelization details
- :doc:`../performance_tuning` - Multi-GPU optimization
- ``include/mpi_manager.h`` - Full header file
- ``src/mpi_manager.cpp`` - Implementation
