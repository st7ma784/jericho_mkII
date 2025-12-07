# Understanding Jericho Mk II Visualizations

## Physics Interpretation Guide

This guide explains what each visualization means and how to interpret the plasma physics.

---

## 🌊 Electromagnetic Field Panel

### What It Shows
A 2D cross-section of the magnetic field (Bz component) perpendicular to the simulation plane.

### How to Read It
- **Colors:**
  - 🔴 **Red**: Strong positive Bz (field pointing out of screen)
  - 🔵 **Blue**: Strong negative Bz (field pointing into screen)
  - 🟢 **Green/Yellow**: Weak field or null point (where field = 0)

### Physical Interpretation

**Magnetic Reconnection Signature:**
1. **X-Point Structure**: The green/yellow region at center is where opposing magnetic fields meet
2. **Field Reversal**: Notice how colors flip across the X-point (red → blue)
3. **Current Sheet**: The narrow green band is where intense electric currents flow
4. **Separatrices**: The boundary lines dividing different field topologies

**What's Happening Physically:**
- Magnetic field lines are "frozen into" the plasma
- When oppositely-directed fields are pushed together, they can break and reconnect
- This converts stored magnetic energy → kinetic energy (heating/acceleration)
- Analogous to a magnetic "short circuit"

**Real-World Examples:**
- ☀️ Solar flares: Releases 10³² ergs in minutes
- 🌍 Earth's magnetosphere: Causes auroras during substorms
- ⚡ Tokamak disruptions: Major concern for fusion reactors

### Advanced Features
- **Toggle Vectors**: Shows electric field direction (E = -v × B)
- **Component Selection**: View Ex, Ey separately to see field structure
- **|B| magnitude**: Shows total field strength regardless of direction

---

## ⚛️ Particle Distribution Panel

### What It Shows
Position (x, y) of ion particles in the simulation domain.

### How to Read It
- **Each dot** = One ion (typically protons in space plasmas)
- **Position** = Where in 2D space that particle is located
- **Color** = Velocity magnitude (blue=slow, red=fast) or particle type

### Physical Interpretation

**Key Features to Look For:**

1. **Density Variations**
   - Bright clusters = high density regions
   - Dark areas = plasma voids or depletions
   - Reconnection creates density asymmetries

2. **Outflow Jets**
   - Particles accelerated perpendicular to X-line
   - Forms fast plasma jets (100-1000 km/s in space)
   - Visible as particle streams moving top/bottom

3. **Mixing**
   - Reconnection mixes plasmas from different regions
   - Can see different particle populations interpenetrating

**What's Happening:**
- Near X-point: Particles gain perpendicular velocity
- Along separatrices: Particles channeled into jets
- In background: Thermal plasma slowly diffusing

**Color Modes:**
- **By Type**: Ions (cyan) vs Electrons (red) - shows charge separation
- **By Velocity**: Shows where particles are being accelerated
- **By Vx/Vy**: Shows preferential acceleration direction

### Why It Matters
- Particle heating is a major unsolved problem in astrophysics
- Reconnection is thought to be the primary heating mechanism
- This view shows WHERE particles get heated

---

## 📊 Energy Conservation Panel

### What It Shows
Time evolution of different energy components in the simulation.

### How to Read It
- **X-axis**: Time (or timestep)
- **Y-axis**: Energy (arbitrary units)
- **Three curves**: Total, Kinetic, Field energy

### Physical Interpretation

**The Energy Budget:**

```
E_total = E_kinetic + E_field + E_thermal
```

**Expected Behavior:**

1. **Total Energy (Cyan)** 
   - Should be FLAT (conserved)
   - Small oscillations < 1% are acceptable
   - Large drift indicates numerical errors

2. **Kinetic Energy (Green)**
   - Energy of particle motion: ½m Σv²
   - Increases during reconnection
   - Shows plasma heating/acceleration

3. **Field Energy (Red)**
   - Energy stored in EM fields: (E² + B²)/2μ₀
   - Decreases during reconnection
   - Magnetic field "annihilation"

**The Energy Transfer:**
```
Magnetic Energy → Kinetic Energy
     ↓                  ↓
  (Field Energy)   (Particle Motion)
```

### What Good vs Bad Looks Like

✅ **Good Simulation:**
- Total energy constant within 0.1%
- Kinetic and field energies anticorrelated (inverse oscillations)
- Smooth curves without sudden jumps

❌ **Bad Simulation:**
- Total energy drifts > 1% 
- Kinetic energy grows unbounded
- Discontinuities or NaN values

### Why This Matters
Energy conservation is a fundamental physics law. If violated, the simulation is unphysical and results cannot be trusted.

---

## 🔄 Phase Space Panel

### What It Shows
The **velocity distribution** of particles - NOT their positions.

### How to Read It (CRITICAL!)
- **X-axis**: Velocity in x-direction (Vx)
- **Y-axis**: Velocity in y-direction (Vy)
- **Brightness**: Number of particles with that velocity
- **Distance from center**: Speed |v| = √(Vx² + Vy²)
- **Angle**: Direction of motion

### Physical Interpretation

This is the hardest plot to understand but the most useful for plasma physics!

**Key Concepts:**

1. **Thermal Equilibrium** (Maxwellian)
   - Gaussian blob centered at (0, 0)
   - Circular symmetry
   - Width ~ thermal velocity √(kT/m)

2. **Beam Distribution**
   - Elongated structure offset from origin
   - Shows organized bulk flow
   - Created by acceleration

3. **Non-Thermal Features**
   - Tails extending to high velocity
   - Asymmetric shapes
   - Multiple peaks (multiple populations)

**During Reconnection:**
- Initially: Centered Gaussian (cold plasma)
- During: Develops tails/beams (heating)
- After: Broader distribution (hotter plasma)
- Asymmetry: Shows preferential acceleration direction

### Real-World Connection

**Spacecraft Measurements:**
- Magnetospheric Multiscale (MMS) mission measures particle velocity distributions
- Phase space plots identify reconnection regions
- Non-Maxwellian distributions = energy input

**Plasma Instabilities:**
- Certain velocity distributions are unstable
- Can trigger waves (Weibel, two-stream instabilities)
- Phase space diagnostics predict instability growth

### Reading Example

Imagine phase space as a "velocity map":
```
      Vy
       ↑
       |  • ← Particle moving up and right (Vx>0, Vy>0)
   •   |     ← Particle moving left and up
   ----+---- Vx
       |
     • ↓ Particle moving right and down
```

Bright spots = many particles with similar velocity
Sparse regions = few particles moving that way

---

## Advanced Interpretation

### Reconnection Rate
Can be estimated from outflow velocity:
```
Reconnection Rate ∝ V_outflow / V_Alfvén
```
Fast reconnection: ~0.1-0.3 (seen in simulations and space)

### Magnetic Topology
The X-point is a **null point** where B = 0. The four regions separated by separatrices have different magnetic connectivity.

### Energy Conversion Efficiency
Measure how much magnetic energy → particle energy:
```
η = ΔE_kinetic / ΔE_magnetic ≈ 10-50%
```

### Particle Acceleration
Reconnection accelerates particles to superthermal energies. Look for:
- Power-law tails in velocity distribution
- Nonthermal X-rays/energetic particles in observations

---

## Quick Reference

| Panel | Shows | Key Feature | Diagnosis |
|-------|-------|-------------|-----------|
| 🌊 Field | B(x,y) | X-point | Reconnection location |
| ⚛️ Particles | Position | Jets | Acceleration efficiency |
| 📊 Energy | E(t) | Conservation | Simulation validity |
| 🔄 Phase | f(Vx,Vy) | Distribution | Heating mechanism |

---

## Common Questions

**Q: Why does the magnetic field have an X-shape?**
A: This is the fundamental geometry of 2D reconnection. Field lines from opposite sides meet at the null point, forming the characteristic X.

**Q: Why do particles form jets instead of spreading uniformly?**
A: The reconnection electric field (E = -v × B) accelerates particles perpendicular to the X-line, creating focused jets rather than isotropic expansion.

**Q: What should the energy ratio be?**
A: Depends on plasma β (thermal/magnetic pressure). For β ~ 1, expect roughly equal kinetic and magnetic energy.

**Q: How do I know if reconnection is happening?**
A: Look for: (1) X-point in fields, (2) bidirectional jets, (3) magnetic → kinetic energy transfer, (4) non-Maxwellian distributions

**Q: What's the difference between real space and phase space?**
A: Real space shows WHERE particles are (position x,y). Phase space shows HOW FAST they're moving (velocity Vx,Vy). Both are needed for complete description.

---

## Further Reading

1. **Yamada et al. (2010)** "Magnetic reconnection" - Reviews of Modern Physics
2. **Biskamp (2000)** "Magnetic Reconnection in Plasmas" - Textbook
3. **Birn & Priest (2007)** "Reconnection of Magnetic Fields" - Monograph
4. **Cassak & Shay (2007)** "Scaling of asymmetric reconnection" - Physics of Plasmas
5. **MMS Mission Data** - Real spacecraft observations of reconnection

---

*This guide is part of the Jericho Mk II documentation.*
*For simulation usage, see `/docs/getting_started.rst`*
