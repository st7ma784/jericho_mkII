# Jericho Mk II Web Visualization

Real-time web-based visualization interface for the Jericho Mk II plasma simulation.

## Features

- 🌊 **Live Electromagnetic Fields** - Real-time heatmaps of Ex, Ey, Bz with optional vector overlay
- ⚛️ **Particle Visualization** - Interactive particle distribution colored by type or velocity
- 📊 **Energy Diagnostics** - Live plotting of energy conservation and system metrics
- 🔄 **Phase Space** - Velocity distribution in (Vx, Vy) space
- 🚀 **WebSocket Streaming** - Low-latency updates as simulation runs
- 🎨 **GPU-Accelerated** - Canvas 2D rendering for smooth performance

## Installation

```bash
cd web
pip install -r requirements.txt
```

## Usage

### Basic Usage

Start the visualization server:

```bash
python server.py --output-dir ../output --port 8888
```

Then open your browser to: `http://localhost:8888`

### With Running Simulation

1. Start your simulation with output to a directory:
   ```bash
   ./jericho_mkII config.toml
   ```

2. In another terminal, start the web server:
   ```bash
   cd web
   python server.py --output-dir ../output
   ```

3. Open `http://localhost:8888` in your browser

The visualization will automatically update as new data files are written!

### Multi-GPU Runs

The web interface works with MPI runs too:

```bash
# Terminal 1: Run simulation
mpirun -np 4 ./jericho_mkII_mpi config.toml

# Terminal 2: Visualize
cd web
python server.py --output-dir ../outputs/my_simulation
```

## Configuration

Command-line options:

```
--output-dir, -o   Directory to monitor for simulation output [default: ./output]
--port, -p         Server port [default: 8888]
--host             Server host [default: 127.0.0.1]
```

## Controls

### Electromagnetic Fields Panel
- **Field Selection**: Choose Ex, Ey, Bz, |E|, or |B|
- **Toggle Vectors**: Overlay electric field vectors on heatmap

### Particle Distribution Panel
- **Color Mode**: Color particles by type (ions/electrons) or velocity
- **Toggle Trails**: Enable motion trails for particle tracking

### Phase Space Panel
- **Color Mode**: Density or velocity-weighted phase space

## Performance

The server automatically downsamples data for web display:
- Field grids limited to 512×512
- Particles limited to 5,000 displayed (randomly sampled)
- Update rate: ~2 Hz (configurable in code)

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support  
- Safari: ✅ Full support
- Mobile: ⚠️ Limited (large data transfers)

## Troubleshooting

**"No data appearing"**
- Check that simulation is writing to the correct output directory
- Verify HDF5 files exist: `ls -l output/*.h5`
- Check server logs for errors

**"Connection failed"**
- Ensure server is running: `python server.py`
- Check firewall settings
- Try 127.0.0.1 instead of localhost

**"Slow performance"**
- Reduce simulation output frequency
- Increase downsampling in server.py
- Close other browser tabs

## Architecture

```
┌─────────────┐         WebSocket         ┌──────────────┐
│   Browser   │ ←──────────────────────→ │    Server    │
│  (Canvas2D) │    JSON + Binary Data    │   (Python)   │
└─────────────┘                           └──────────────┘
                                                 ↓
                                          Monitor Output
                                                 ↓
                                          ┌──────────────┐
                                          │  HDF5 Files  │
                                          │   (Fields,   │
                                          │  Particles)  │
                                          └──────────────┘
```

## Development

To modify the visualization:

1. Edit `static/visualization.js` for rendering logic
2. Edit `index.html` for UI layout
3. Edit `server.py` for data processing
4. Refresh browser (no server restart needed for client changes)

## Citation

If you use this visualization tool in publications, please cite:

```bibtex
@software{jericho_mkII_viz,
  title = {Jericho Mk II Real-time Web Visualization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/st7ma784/jericho_mkII}
}
```
