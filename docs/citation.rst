Citation and Contributing
=========================

How to Cite
-----------

If you use Jericho Mk II in your research, please cite:

BibTeX Entry
~~~~~~~~~~~~

.. code-block:: bibtex

   @software{jericho_mkII,
     author = {Wiggs, Joshua and Arridge, Christopher S. and 
               Greenyer, George and Mander, Steve},
     title = {Jericho Mk II: GPU-Accelerated Hybrid PIC-MHD Code for Plasma Simulation},
     year = {2025},
     url = {https://github.com/st7ma784/jericho_mkII},
     version = {2.0.0},
     doi = {10.5281/zenodo.XXXXXXX}  % Will be assigned upon publication
   }

Published Papers
~~~~~~~~~~~~~~~~

If there are published papers using Jericho Mk II, cite the relevant paper(s):

.. code-block:: bibtex

   @article{wiggs2025jericho,
     author = {Wiggs, J. and Arridge, C. S. and Others},
     title = {Jericho Mk II: A Next-Generation Hybrid PIC-MHD Code},
     journal = {Journal Name},
     year = {2025},
     volume = {XX},
     pages = {XXX--XXX},
     doi = {XX.XXXX/XXXXXX}
   }

Acknowledgments
~~~~~~~~~~~~~~~

In your acknowledgments section, please include:

.. code-block:: text

   This research made use of Jericho Mk II, a GPU-accelerated hybrid PIC-MHD 
   plasma simulation code developed at Lancaster University 
   (https://github.com/st7ma784/jericho_mkII).

Contributing
------------

We welcome contributions! Here's how to get involved.

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~~

- **Bug reports:** Found an issue? Open a GitHub issue
- **Bug fixes:** Submit a pull request
- **New features:** Propose via GitHub issue first
- **Documentation:** Improve or add documentation
- **Examples:** Share interesting simulation setups
- **Performance improvements:** Optimize kernels or algorithms

Getting Started
~~~~~~~~~~~~~~~

1. **Fork the repository:**

   Visit https://github.com/st7ma784/jericho_mkII and click "Fork"

2. **Clone your fork:**

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/jericho_mkII.git
      cd jericho_mkII

3. **Create a branch:**

   .. code-block:: bash

      git checkout -b feature/my-new-feature

4. **Make changes:**

   Edit code, add tests, update documentation

5. **Test your changes:**

   .. code-block:: bash

      # Build and test
      mkdir build && cd build
      cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
      make -j
      
      # Run tests
      ctest
      
      # Run example
      ./jericho_mkII ../examples/minimal_test.toml

6. **Commit changes:**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of feature"

7. **Push to your fork:**

   .. code-block:: bash

      git push origin feature/my-new-feature

8. **Open pull request:**

   Go to GitHub and click "New Pull Request"

Code Style
~~~~~~~~~~

**C++ Code:**

- Follow Google C++ Style Guide
- Use meaningful variable names
- Add comments for complex algorithms
- Keep functions < 50 lines when possible

.. code-block:: cpp

   // Good
   double compute_kinetic_energy(const ParticleBuffer& particles, 
                                  double mass) {
       double total_energy = 0.0;
       for (size_t i = 0; i < particles.count; ++i) {
           if (!particles.active[i]) continue;
           double v2 = particles.vx[i] * particles.vx[i] + 
                      particles.vy[i] * particles.vy[i];
           total_energy += 0.5 * mass * particles.weight[i] * v2;
       }
       return total_energy;
   }
   
   // Bad
   double f(const ParticleBuffer& p, double m) {
       double e = 0.0;
       for (size_t i = 0; i < p.count; ++i) {
           if (!p.active[i]) continue;
           double v2 = p.vx[i]*p.vx[i]+p.vy[i]*p.vy[i];
           e += 0.5*m*p.weight[i]*v2;
       }
       return e;
   }

**CUDA Kernels:**

- Optimize for coalesced memory access
- Minimize divergence
- Document thread/block requirements
- Add performance notes

.. code-block:: cuda

   /**
    * @brief Advance particle velocities using Boris pusher
    * 
    * @param[in,out] vx, vy Particle velocities (modified in-place)
    * @param[in] Ex, Ey, Bz Field values at particle positions
    * @param[in] n_particles Number of particles
    * @param[in] dt Timestep
    * 
    * @note Launch with 256 threads per block for optimal occupancy
    * @note Memory bandwidth bound: ~100 GB/s on A100
    */
   __global__ void boris_push_kernel(...) {
       // ...
   }

**Python Scripts:**

- Follow PEP 8
- Add docstrings
- Include type hints (Python 3.7+)

.. code-block:: python

   def read_fields(filename: str) -> dict:
       """
       Read field data from HDF5 file.
       
       Args:
           filename: Path to HDF5 file
           
       Returns:
           Dictionary with field arrays and metadata
       """
       # ...

Documentation
~~~~~~~~~~~~~

- Update docs when adding features
- Use reStructuredText (.rst) format
- Include examples
- Add to appropriate section in ``docs/index.rst``

Testing
~~~~~~~

**Unit tests:**

.. code-block:: cpp

   // tests/test_boris.cpp
   #include <gtest/gtest.h>
   #include "boris_pusher.h"
   
   TEST(BorisTest, EnergyConservation) {
       // Setup
       double vx = 1.0e6, vy = 0.0;
       double Ex = 0.0, Ey = 0.0, Bz = 1.0e-9;
       double q_over_m = 9.578e7;
       double dt = 0.01;
       
       double v0 = std::sqrt(vx*vx + vy*vy);
       
       // Push 1000 steps
       for (int i = 0; i < 1000; ++i) {
           boris_push(vx, vy, Ex, Ey, Bz, q_over_m, dt);
       }
       
       double v1 = std::sqrt(vx*vx + vy*vy);
       
       // Verify energy conservation
       EXPECT_NEAR(v0, v1, 1e-10);
   }

**Integration tests:**

.. code-block:: bash

   # tests/integration/test_reconnection.sh
   #!/bin/bash
   
   # Run reconnection simulation
   ./jericho_mkII tests/integration/reconnection.toml
   
   # Check output exists
   test -f output/fields_000100.h5 || exit 1
   
   # Check energy conservation
   python3 scripts/check_energy.py output/diagnostics.csv || exit 1
   
   echo "Integration test PASSED"

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

1. **One feature per PR:** Keep PRs focused
2. **Write tests:** Add unit tests for new code
3. **Update docs:** Document new features
4. **Pass CI:** Ensure all tests pass
5. **Clear description:** Explain what and why

**Good PR description:**

.. code-block:: markdown

   ## Description
   Adds support for multiple ion species with different charge/mass ratios.
   
   ## Motivation
   Current code only supports single ion species. Many plasma scenarios 
   (e.g., magnetosphere) have multiple ion populations (H+, O+, He+).
   
   ## Changes
   - Modified ParticleBuffer to support multiple species
   - Added species-specific q/m ratios in config
   - Updated Boris pusher to handle different masses
   - Added test case with H+ and O+ ions
   
   ## Testing
   - Unit tests pass (test_multi_species)
   - Integration test with H+/O+ reconnection
   - Energy conservation verified
   
   ## Documentation
   - Updated configuration.rst with species examples
   - Added tutorial for multi-species simulations

Reporting Bugs
~~~~~~~~~~~~~~

Use GitHub Issues with this template:

.. code-block:: markdown

   **Bug Description:**
   Clear description of the bug
   
   **To Reproduce:**
   1. Step 1
   2. Step 2
   3. See error
   
   **Expected Behavior:**
   What should happen
   
   **Actual Behavior:**
   What actually happens
   
   **Environment:**
   - GPU: NVIDIA RTX 3080
   - CUDA: 11.8
   - OS: Ubuntu 22.04
   - MPI: OpenMPI 4.1.2
   
   **Config File:**
   ```toml
   [paste config here]
   ```
   
   **Error Log:**
   ```
   [paste error messages]
   ```

Requesting Features
~~~~~~~~~~~~~~~~~~~

Use GitHub Issues with this template:

.. code-block:: markdown

   **Feature Request:**
   Brief title
   
   **Problem:**
   What problem does this solve?
   
   **Proposed Solution:**
   How would you implement this?
   
   **Alternatives:**
   Other approaches considered?
   
   **Additional Context:**
   Any examples, references, or use cases?

Code of Conduct
---------------

**Be Respectful:**

- Treat all contributors with respect
- Welcome diverse perspectives
- Accept constructive criticism gracefully

**Be Helpful:**

- Help newcomers get started
- Provide constructive feedback
- Share knowledge and experience

**Be Professional:**

- Keep discussions on-topic
- No harassment or discrimination
- Resolve disagreements constructively

License
-------

Jericho Mk II is released under the **MIT License**:

.. code-block:: text

   MIT License
   
   Copyright (c) 2025 Joshua Wiggs, Christopher S. Arridge, 
                      George Greenyer, Steve Mander
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

**What this means:**

- ✅ Use for any purpose (commercial, academic, personal)
- ✅ Modify and distribute
- ✅ Private use
- ⚠️ Must include license and copyright notice
- ❌ No warranty provided

Credits
-------

Core Developers
~~~~~~~~~~~~~~~

- **Joshua Wiggs** - Lead developer, GPU implementation
- **Christopher S. Arridge** - Physics algorithms, validation
- **George Greenyer** - Original Jericho code
- **Steve Mander** - Original Jericho code

Acknowledgments
~~~~~~~~~~~~~~~

- Lancaster University Physics Department
- STFC DiRAC HPC Facility
- NVIDIA GPU Grant Program
- Original Jericho development team

Third-Party Libraries
~~~~~~~~~~~~~~~~~~~~~

- **CUDA Toolkit** - NVIDIA Corporation
- **MPI** - Open MPI or MPICH
- **HDF5** - The HDF Group
- **CMake** - Kitware, Inc.

Contact
-------

- **Project Lead:** Josh Wiggs (j.wiggs@lancaster.ac.uk)
- **Principal Investigator:** Chris Arridge (c.arridge@lancaster.ac.uk)
- **GitHub:** https://github.com/st7ma784/jericho_mkII
- **Documentation:** https://st7ma784.github.io/jericho_mkII/

Funding
-------

This work was supported by:

- Lancaster University
- Science and Technology Facilities Council (STFC)
- [Add other funding sources]

Publications Using Jericho Mk II
---------------------------------

If you've published work using Jericho Mk II, let us know! We'll add it here.

1. [Your paper here]

See Also
--------

- :doc:`getting_started` - Installation and usage
- :doc:`architecture` - System design
- GitHub repository - Source code and issues
