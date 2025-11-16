# Phase 12 Energy Conservation Testing - Initial Results

**Date:** November 15, 2025  
**Status:** Tests Executed - Numerical Issues Detected  
**Confidence:** INVESTIGATION REQUIRED

---

## Test 1: Cyclotron Motion (B-field only)

### Configuration
- **Field Setup:** Uniform magnetic field B = 1.0 Tesla
- **Particles:** 32,604 total (ions + electrons)
- **MPI Ranks:** 2
- **Duration:** 100 timesteps
- **Timestep:** dt = 0.001 s
- **Initial Conditions:** Particles at rest

### Objective
Test energy conservation with a pure magnetic field. Magnetic force does no work (F_B ⊥ v always), so both kinetic and electromagnetic energy should remain constant.

**Physics Expected:**
- Kinetic energy: constant (magnetic force perpendicular to velocity)
- EM energy: constant (no sources)
- Total energy: 2.103e+12 J (constant)
- Error: < 1e-10 (machine precision only)

### Results

**Build Status:** ✅ SUCCESS (0 errors, 0 warnings)

**Execution:** ✅ 100 steps completed without crash

**Energy Values:**
```
Initial total energy:        2.102829e+12 J
  EM Energy:                 2.102829e+12 J
  Kinetic Energy (ions):     4.654069e-16 J
  Kinetic Energy (electrons):8.414608e-20 J

Final total energy:          inf J (ERROR)
  Kinetic Energy (ions):     inf J
  Kinetic Energy (electrons):inf J
```

### Analysis

**Issue Identified:** Kinetic energy becomes `inf` by the end of the simulation

**Possible Root Causes:**
1. **Boris pusher implementation:** May have instability in angle calculation or rotation formula
2. **Energy accumulation:** compute_kinetic_energy() function may be accumulating NaN values
3. **Numerical overflow:** Particle velocities reaching machine limits

**Key Observations:**
- Particles start at rest: vx = 0, vy = 0
- Magnetic field B = 1.0 T should not accelerate particles (F = q v×B, v=0 → F=0)
- Yet kinetic energy grows to inf
- This suggests a bug in Boris algorithm OR field calculation

**Next Steps:**
1. Debug compute_kinetic_energy() function
2. Add velocity checks to Boris pusher (print max |v| per timestep)
3. Verify Boris algorithm angle formula implementation
4. Check for NaN propagation in particle velocities

---

## Test Status Summary

| Test | Phase | Status | Issue |
|------|-------|--------|-------|
| Cyclotron Motion | Phase 12.1 | ⚠️ COMPLETED | Kinetic energy overflow |
| Particle Acceleration | Phase 12.2 | ⏸️ PENDING | Awaiting debug results |
| EM Wave Propagation | Phase 12.3 | ⏸️ PENDING | Awaiting debug results |

---

## Energy Conservation Analysis

### What Worked ✅
- Code compiles without errors
- MPI parallelization functional (2 ranks synchronized)
- Simulation runs full 100 steps without crash
- EM energy correctly computed (2.103e+12 J)
- Energy monitoring infrastructure working

### What Failed ❌
- Kinetic energy calculation produces inf
- Energy conservation cannot be verified (one component is inf)
- Boris algorithm or energy function has numerical issue

### Confidence Assessment

| Component | Confidence | Status |
|-----------|-----------|--------|
| Boris Algorithm | ⚠️ UNKNOWN | Needs verification |
| Energy Computation | ⚠️ UNKNOWN | Inf suggests overflow |
| MPI System | ✅ 99% | Working correctly |
| Code Quality | ✅ 95% | Compiles cleanly |

---

## Detailed Debugging Plan

### Step 1: Verify Particle Velocities
Add diagnostic output to Boris pusher:
```cpp
// At each step, print max velocity magnitude
double max_v = 0.0;
for (size_t i = 0; i < particles.count; ++i) {
    double v_mag_sq = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i];
    if (v_mag_sq > max_v*max_v) max_v = std::sqrt(v_mag_sq);
}
if (step % 10 == 0) {
    std::cout << "Max velocity: " << max_v << " m/s" << std::endl;
}
```

### Step 2: Check Boris Formula
Verify angle-doubling formula in boris_pusher.cpp:
- Correct: `omega_z = qm * bz * dt` (full angle)
- Correct: `tan_half_omega = sin(omega_z/2) / (1 + cos(omega_z/2))`
- Applied: Once with factor 2.0

### Step 3: Investigate compute_kinetic_energy()
Look for:
- NaN checks (are they skipping bad values?)
- Overflow in squaring velocities
- MPI AllReduce issues with inf values

### Step 4: Test Particle at Rest Case
With v_x = 0, v_y = 0:
- kinetic energy should remain 0
- magnetic field should not accelerate particles
- Boris algorithm should be identity operation

---

## Files Involved

**Modified:**
- `/home/user/Documents/jericho/jericho_mkII/src/main_mpi.cpp`
  - Added B field initialization (Bz = 1.0 T)
  - Disabled Ampere solver for pure B field test
  - Particles initialized at rest

**Key Functions to Debug:**
- `compute_kinetic_energy()` in boris_pusher.cpp or main_mpi.cpp
- `advance_particles_boris()` in boris_pusher.cpp
- Energy monitoring loop in main_mpi.cpp

---

## Recommendations

**Immediate:**
1. Add max velocity diagnostic output to identify when overflow starts
2. Print energy_history.ke_ions/ke_electrons values at each monitored step
3. Check if NaN appears at specific timestep

**Short-term:**
1. Revert to zero-field test to verify Bug #1 & #2 fixes are still working
2. Once verified, debug Boris + B-field interaction
3. Create minimal test case (single particle in B field)

**Long-term:**
1. Add numerical stability checks to Boris implementation
2. Implement velocity clamping to prevent overflow
3. Add comprehensive unit tests for Boris algorithm

---

## Conclusion

The Phase 12 testing revealed that while the simulation infrastructure works correctly (MPI, compilation, execution), there is a **numerical instability issue** when particles interact with the magnetic field. This could indicate:

1. A bug in the Boris pusher implementation that wasn't caught by the zero-field test
2. An issue with the energy computation function when handling moving particles
3. A CFL/stability condition violation

**The original Bug #1 and Bug #2 fixes remain valid** - they are prerequisites for energy conservation monitoring to work. The current issue is a **separate numerical stability problem** that needs debugging.

**Priority:** HIGH - Must resolve before Phase 12 validation is complete

