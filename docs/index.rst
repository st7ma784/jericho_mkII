Welcome to Jericho Mk II Documentation
========================================

**Jericho Mk II** is a next-generation GPU-accelerated hybrid PIC-MHD code for plasma simulation.

.. image:: https://img.shields.io/badge/version-2.0.0-blue
   :alt: Version 2.0.0

.. image:: https://github.com/yourusername/jericho_mkII/workflows/docs/badge.svg
   :alt: Documentation Status

.. image:: https://github.com/yourusername/jericho_mkII/workflows/build/badge.svg
   :alt: Build Status

Key Features
------------

🚀 **Performance**
  - CUDA-native implementation for 10-50x GPU speedup
  - Structure of Arrays (SoA) layout for coalesced memory access
  - MPI+CUDA hybrid parallelism for multi-GPU scaling
  - Asynchronous compute and communication overlap

🔬 **Physics**
  - Hybrid PIC-MHD: Kinetic ions + fluid electrons
  - Boris particle pusher with energy conservation
  - Current Advance Method (CAM) for numerical stability
  - Ohm's Law with Hall term
  - Multiple ion species support

🎯 **Boundary Conditions**
  - Periodic, inflow, outflow, and reflecting boundaries
  - Dynamic particle management
  - Mixed boundary configurations

📊 **Modern Architecture**
  - Clean C++17/CUDA codebase
  - CMake build system
  - Comprehensive testing
  - Auto-generated documentation

Quick Start
-----------

**Installation:**

.. code-block:: bash

   git clone https://github.com/yourusername/jericho_mkII.git
   cd jericho_mkII
   mkdir build && cd build
   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
   make -j

**Run a simulation:**

.. code-block:: bash

   # Single GPU
   ./jericho_mkII ../examples/reconnection.toml

   # Multi-GPU
   mpirun -np 4 ./jericho_mkII ../examples/reconnection.toml

Performance
-----------

Compared to the original Jericho (CPU-only):

.. list-table::
   :header-rows: 1

   * - Configuration
     - Jericho (CPU)
     - Jericho Mk II (GPU)
     - Speedup
   * - 100K particles, 128×128 grid
     - 14 min
     - ~30 sec
     - **28x**
   * - 1M particles, 256×256 grid
     - 3.5 hours
     - ~8 min
     - **26x**
   * - 10M particles, 512×512 grid
     - N/A (OOM)
     - ~1.5 hours
     - ∞

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   configuration
   running_simulations
   output_formats

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   cuda_kernels
   mpi_parallelism
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/particle_buffer
   api/fields
   api/boundaries
   api/mpi_manager

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   performance_tuning
   troubleshooting
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
